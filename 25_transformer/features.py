"""Feature extraction for both stages, and the one-stop conditioning-hint API.

Two roles:

* stage-1 features / stage-2 `phone_ids` -- Qwen3-ASR + forced alignment, RMVPE
  pitch, the HorsePhonemizer/G2P front end (`align`, `phonemize_items`,
  `extract_features_ac`, ...);
* conditioning hints (`hint_intensity`, `hint_register`, `hint_semantic`,
  `hint_words`, `extract_hints`) -- thin wrappers over the offline `hints/`
  package, so an inference UI has a single place to ask for every hint.  Those
  imports are lazy: `hints/` adds parselmouth (Praat) and loads the DSP /
  sentence-embedding models, which stage 1/2 must not pay for unless a hint is
  actually requested.  (scikit-learn and pandas are already pulled in by
  `g2p_en` via `horsephonemizer`, so they are not the reason.)
"""

import math
import unicodedata
import numpy as np
import torch
from svc_helper.pitch.rmvpe import RMVPEModel
from horsephonemizer import HorsePhonemizer
from transformers import AutoTokenizer 
from modeling.vits import spectrogram, utils
import librosa

# ---------------------------------------------------------------------------
# Conditioning hints (offline package `hints/`; see hints/README.md)
# ---------------------------------------------------------------------------
#: Code frames per second for the shipped configs:
#: ``data.sampling_rate / data.hop_length / down_factor`` = 48000/480/4.
#: `Generator` hard-codes ``down_factor=4`` (it is not a config key), and
#: `FeatureExtractor.code_frame_rate` recomputes this from the loaded config.
DEFAULT_CODE_FPS = 25.0

#: 48 kHz, i.e. the stage-1 waveform rate (`data.sampling_rate`).
HINT_WAVE_SR = 48000


def _hint_helpers():
    """Lazily import ``hints.build`` (DSP model, sentence encoder, `.cond` I/O).

    Lazy on purpose: `hints/` adds **parselmouth** (Praat) to the dependency set
    and loads the DSP regressor / bge encoder, none of which stage 1/2 needs.
    (scikit-learn and pandas come in anyway via ``g2p_en``.)  The module caches
    the DSP model in a process global, so repeated calls are cheap.
    """
    import hints.build
    return hints.build



def _f0_span(f0, start_time, end_time, hop=160, sr=16000):
    """Slice a frame-level f0 contour to the frames centred in a word span.

    Args:
        f0: 1-D array of f0 in Hz (0 = unvoiced), one value per frame.
        start_time: Word start in seconds.
        end_time: Word end in seconds.
        hop: Frame hop in samples.
        sr: Sample rate the contour was computed at.

    Returns:
        A 1-D ``float64`` copy of ``f0`` over the span (may be empty).
    """
    n_frames = f0.shape[0]
    start_frame = int(math.floor(start_time * sr / hop))
    end_frame = int(math.ceil(end_time * sr / hop))
    start_frame = max(0, min(start_frame, n_frames))
    end_frame = max(0, min(end_frame, n_frames))
    if end_frame < start_frame:
        end_frame = start_frame
    return f0[start_frame:end_frame].astype(np.float64, copy=True)


# Defaults for the optional ``data.audio_norm`` block in configs/base.yaml.
# ``mode: none`` reproduces the historical behaviour (raw audio), which is what
# every stage-1/stage-2 tensor and checkpoint produced before this key existed
# was built with.
AUDIO_NORM_DEFAULTS = {
    'mode': 'none',
    'target_db': -1.0,
    'max_gain_db': 30.0,
    'frame_ms': 20.0,
}

# Levels below this are treated as digital silence: normalising them would only
# amplify the noise floor (or divide by ~0).
SILENCE_LEVEL = 1e-8


def normalize_waveform(wav, mode='none', target_db=-1.0, max_gain_db=30.0,
                       frame_ms=20.0, sampling_rate=48000):
    """Scale one clip to a fixed level. Returns ``(wav, gain_db)``.

    This is applied by `FeatureExtractor.extract_features_ac`, i.e. to the
    audio that produces the stage-1 ``.wave``/``.spec`` tensors and -- through
    ``2_preprocess.py``, which uses the same extractor -- the stage-2 code
    tensors, so both stages see the same operating point. The pitch contour is
    gain-invariant, so only ``wave`` and ``spec`` change.

    Args:
        wav: 1-D mono float waveform in [-1, 1] (48 kHz for stage 1).
        mode: ``'none'`` (identity), ``'peak'`` (loudest sample) or ``'rms'``
            (loudest `frame_ms` window). Mirrors ``data.audio_norm.mode``.
        target_db: Level the reference is scaled *to*, in dBFS. Keep this below
            0: the stage-1 decoder ends in ``tanh``, so a target at full scale
            is not reachable.
        max_gain_db: Never boost by more than this, so a clip whose peak is a
            single click (or a noise blip) is not amplified by 40 dB. The cap
            only limits boosting, never attenuation.
        frame_ms: Window length used by ``mode='rms'``.
        sampling_rate: Sample rate of `wav`; only used by ``mode='rms'``.

    Returns:
        ``(wav, gain_db)`` where `wav` is the scaled waveform (same dtype,
        clipped to [-1, 1]) and `gain_db` is the gain applied. ``mode='none'``,
        near-silent input and a gain within 1e-4 of unity (float rounding) all
        return the input unchanged with ``gain_db = 0.0``. The function is
        idempotent: applying it to an already-normalised clip yields
        ``gain_db = 0.0``.

    Raises:
        ValueError: `mode` is not one of ``none``/``peak``/``rms``.
    """
    wav = np.asarray(wav)
    if mode in (None, 'none') or wav.size == 0:
        return wav, 0.0

    if mode == 'peak':
        level = float(np.max(np.abs(wav)))
    elif mode == 'rms':
        frame = max(1, int(round(float(frame_ms) * 1e-3 * sampling_rate)))
        usable = (wav.size // frame) * frame
        if usable >= frame:
            frames = np.asarray(wav[:usable], dtype=np.float64)
            level = float(np.sqrt(
                np.mean(frames.reshape(-1, frame) ** 2, axis=1)).max())
        else:
            level = float(np.sqrt(np.mean(
                np.asarray(wav, dtype=np.float64) ** 2)))
    else:
        raise ValueError("data.audio_norm.mode must be 'none', 'peak' or "
                         f"'rms', got {mode!r}")

    if not np.isfinite(level) or level < SILENCE_LEVEL:
        return wav, 0.0

    target = 10.0 ** (float(target_db) / 20.0)
    gain = min(target / level, 10.0 ** (float(max_gain_db) / 20.0))
    # A gain this close to unity is float rounding (e.g. a clip that is already
    # at `target_db`): return the input untouched so the function is *exactly*
    # idempotent and preprocessing never rewrites audio for ~1e-8 dB.
    if abs(gain - 1.0) < 1e-4:
        return wav, 0.0
    # Clip defensively: target_db >= 0 (or float rounding at full scale) could
    # otherwise leave samples outside the [-1, 1] range the decoder's tanh
    # output lives in.
    out = np.clip(wav * gain, -1.0, 1.0)
    return out.astype(wav.dtype, copy=False), float(20.0 * math.log10(gain))


def _aligner_keeps(ch):
    """Whether the Qwen forced aligner keeps `ch` inside an aligned token.

    Mirrors ``Qwen3ForceAlignProcessor.is_kept_char`` in ``qwen_asr``: letters
    and digits (by Unicode category) plus the ASCII apostrophe. Everything
    else -- ``, . ? ! - ; :`` and Unicode quotes -- is stripped from the token
    it belongs to, which is why `attach_punctuation` exists.
    """
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith('L') or cat.startswith('N')


def attach_punctuation(text, items):
    """Put the punctuation the forced aligner strips back onto `items`.

    ``Qwen3-ForcedAligner`` tokenises its input with
    ``Qwen3ForceAlignProcessor``, whose ``clean_token`` keeps only letters,
    digits and ``'`` in each token. For ``"I don't know, Spike."`` it therefore
    returns the items ``["I", "don't", "know", "Spike"]`` -- the ``,`` and ``.``
    are gone before ``phonemize_items`` ever sees them. That matters, because
    g2p_en emits punctuation as its own ARPAbet token, the g2pen tokenizer has
    ids for ``, . ? ! -``, and the published stage-2 corpora carry them in
    essentially every clip: dropping them removes a pause cue the transformer
    was trained on.

    This stores the punctuation that follows each word in an extra ``punct``
    field, which `phonemize_items` appends to that word's phone string -- the
    same shape the whole-transcript path produces (``SPAY1K.`` rather than
    ``SPAY1K``).

    Matching is positional: `text` is split on whitespace, each segment reduced
    with the aligner's own rule gives one word per item (CJK excepted, where
    the aligner emits one item per character, so item texts are joined until
    they cover the word). Punctuation on a segment with no word at all (a lone
    ``--`` / ``...``) is given to the preceding word.

    Args:
        text: The string that was aligned -- the caller's transcript, or the
            ASR text that produced `items`. Both keep their punctuation.
        items: Mutable item dicts, each with a ``text`` key.

    Returns:
        True when the punctuation was attached (also when there was none).
        False when `text` and `items` disagree on the word sequence; `items` is
        then left untouched, because losing the punctuation is bad but putting
        it on the wrong word would be worse.
    """
    if not text or not items:
        return False
    pairs = []  # (word, punctuation) in reading order
    pending = ''  # punctuation seen before the first word
    for seg in str(text).split():
        word = ''.join(ch for ch in seg if _aligner_keeps(ch))
        punct = ''.join(ch for ch in seg if not _aligner_keeps(ch))
        if not word:
            if pairs:
                pairs[-1] = (pairs[-1][0], pairs[-1][1] + punct)
            else:
                pending += punct
            continue
        pairs.append((word, pending + punct))
        pending = ''
    if not pairs:
        return False

    attached = []  # (item index, punctuation)
    idx = 0
    for word, punct in pairs:
        target = word.lower()
        consumed = ''
        while idx < len(items) and len(consumed) < len(target):
            consumed += str(items[idx].get('text', '')).lower()
            idx += 1
        if consumed != target:
            return False
        attached.append((idx - 1, punct))
    if idx != len(items):
        return False
    for i, punct in attached:
        if punct:
            items[i]['punct'] = items[i].get('punct', '') + punct
    return True


class FeatureExtractor:
    def __init__(self, hp, device="cuda:0",
        load_qwen=True):
        self.dtype = torch.bfloat16
        self.device = device
        self.hp = hp
        # Gain applied by the last `extract_features_ac` / `normalize_audio`
        # call, in dB (0.0 when `data.audio_norm.mode` is `none`). The gain is
        # not written to disk, so this is the only record of it.
        self.last_gain_db = 0.0

        # ``None`` means "not loaded": only ``align()`` needs the ASR /
        # forced-aligner weights, so code that only wants spec/f0 or phones
        # can skip them (see ``load_qwen``).
        self.qwen_model = None
        if load_qwen:
            from qwen_asr import Qwen3ASRModel
            self.qwen_model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B",
                dtype=self.dtype,
                device_map=self.device,
                # attn_implementation="flash_attention_2", 
                max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
                max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=self.dtype,
                    device_map=self.device
                    # attn_implementation="flash_attention_2",
                ),
            )
        # RMVPE weights are only needed for pitch tracking, i.e. never for
        # ``phonemize_items`` / ``phonemize_text``. Build them on first use so a
        # phones-only pass does not pay for them.
        self._rmvpe_model = None
        self.hphzr = HorsePhonemizer()

        # https://huggingface.co/therealvul/tokenizer_g2pen_v3/blob/main/tokenizer.json
        tokenizer_g2p = 'therealvul/g2pen_tokenizer_clean'
        g2p_special = {
            'unk_token': "[UNK]",
            'pad_token': "[PAD]",
            'cls_token': "[CLS]",
            'eos_token': "[SEP]",
            'mask_token':"[MASK]",
        }
        self.space_id = 5
        self.tokenizer_g2p = AutoTokenizer.from_pretrained(
            tokenizer_g2p, **g2p_special)

    @property
    def rmvpe_model(self):
        """Lazily-built RMVPE pitch model (see ``extract_pitch``)."""
        if self._rmvpe_model is None:
            self._rmvpe_model = RMVPEModel(device=self.device, is_half=True)
        return self._rmvpe_model

    def audio_norm_settings(self):
        """The ``data.audio_norm`` block as a plain dict with defaults filled in.

        A config without the key (or without a ``data`` block) resolves to
        ``mode: 'none'``, i.e. the historical raw-audio behaviour.
        """
        settings = dict(AUDIO_NORM_DEFAULTS)
        try:
            data_cfg = self.hp.get('data', None)
        except (AttributeError, TypeError):
            data_cfg = None
        cfg = data_cfg.get('audio_norm', None) if data_cfg is not None else None
        if cfg is not None:
            for key in settings:
                if key in cfg:
                    settings[key] = cfg[key]
        return settings

    def normalize_audio(self, data, sampling_rate=None):
        """Apply the configured ``data.audio_norm`` to one clip.

        Args:
            data: 1-D mono waveform at `sampling_rate` (48 kHz for stage 1).
            sampling_rate: Rate of `data`; defaults to
                ``hp.data.sampling_rate``. Only used by ``mode='rms'``.

        Returns:
            ``(wav, gain_db)``; the gain is also stored on
            ``self.last_gain_db`` for logging.
        """
        if sampling_rate is None:
            # `.get`, not `getattr`: OmegaConf raises ConfigAttributeError for a
            # missing key, which `getattr`'s default does not swallow.
            try:
                data_cfg = self.hp.get('data', None)
            except (AttributeError, TypeError):
                data_cfg = None
            sampling_rate = int(data_cfg.get('sampling_rate', 48000)
                                if data_cfg is not None else 48000)
        wav, gain_db = normalize_waveform(
            data, sampling_rate=sampling_rate, **self.audio_norm_settings())
        self.last_gain_db = gain_db
        return wav, gain_db

    def extract_spec(self, data_16k):
        data_16k = torch.from_numpy(data_16k).unsqueeze(0)
        n_fft = self.hp.data.filter_length
        sampling_rate = self.hp.data.sampling_rate
        hop_size = self.hp.data.sampling_rate // 100
        win_size = self.hp.data.win_length
        spec = spectrogram.spectrogram_torch(
            data_16k, n_fft, sampling_rate, hop_size, win_size, center=False)
        spec = torch.squeeze(spec, 0) # [769, -1]
        return spec.transpose(0, 1) # [-1, 769]

    def extract_pitch(self, data_16k):
        f0_extracted, _ = self.rmvpe_model.extract_pitch2(torch.from_numpy(data_16k))
        return f0_extracted

    def align(self, data_16k, language="English", transcription=None):
        """Word-level timing for one clip (ASR, or forced alignment of `transcription`).

        This is the *only* stage that needs the Qwen3-ASR / forced-aligner
        weights, so it is the expensive one.  It yields plain dicts, which
        `phonemize_items` consumes without touching the audio — that is what
        lets a phonemizer change be re-run without re-aligning the corpus (see
        ``2_preprocess.py --only phone_ids``).

        Args:
            data_16k: 1-D numpy array of mono audio at 16 kHz.
            language: Language hint forwarded to qwen_transcribe, or — when
                `transcription` is given — forwarded directly to the forced
                aligner as its `language` argument.
            transcription: Optional user-provided transcript string. When
                given (non-empty), the full ASR pipeline is skipped and
                `self.qwen_model.forced_aligner.align()` is called directly
                with (audio, transcription, language) to produce timestamps.
                Empty/whitespace-only strings return [] without calling the
                aligner.

        Returns:
            List of dicts, one per ForcedAlignItem (in order), each with
                ``text`` (str): word / token text from the forced aligner.
                ``start_time`` (float): word start in seconds.
                ``end_time`` (float): word end in seconds.
                ``punct`` (str, optional): punctuation that followed this word
                    in the aligned text. The aligner strips it from its tokens
                    (see `attach_punctuation`), so this is the only way it
                    survives; `phonemize_items` appends it to the phones.
        """
        if self.qwen_model is None:
            raise RuntimeError(
                "align() needs the Qwen3-ASR / forced-aligner weights; "
                "construct FeatureExtractor(..., load_qwen=True).")
        # (text the items were aligned against, aligner items) per utterance.
        # The text -- not the items -- is what carries the punctuation: both
        # the caller's transcript and the ASR output keep it, while the
        # aligner's tokenizer strips it from the tokens it returns.
        groups = []
        if transcription is not None and str(transcription).strip() != "":
            if self.qwen_model.forced_aligner is None:
                raise ValueError(
                    "transcription=... requires `forced_aligner` to be "
                    "provided at initialization (got None).")
            aligned = self.qwen_model.forced_aligner.align(
                audio=(np.asarray(data_16k), 16000),
                text=str(transcription),
                language=language,
            )
            # align() returns a list with one ForcedAlignResult per sample;
            # single-audio input yields exactly one result.
            for res in aligned:
                inner = getattr(res, "items", res)
                groups.append((str(transcription), list(inner)))
        elif transcription is not None:
            # Explicitly empty transcription: nothing to align.
            groups = []
        else:
            results = self.qwen_model.transcribe(
                audio=(np.asarray(data_16k), 16000),
                language=language,
                return_time_stamps=True,
            )
            # Flatten aligned items across utterances (single-audio input
            # normally yields exactly one ASRTranscription). The utterance text
            # is kept next to its items so each can be punctuated on its own.
            for utt in results:
                ts = getattr(utt, "time_stamps", None)
                if ts is None:
                    continue
                inner = getattr(ts, "items", ts)
                groups.append((getattr(utt, "text", "") or "", list(inner)))

        # Everything downstream only needs these fields (`punct` is filled in
        # below when the aligned text is known).
        items = []
        for text, inner in groups:
            dicts = [{
                "text": getattr(it, "text", ""),
                "start_time": float(getattr(it, "start_time", 0.0)),
                "end_time": float(getattr(it, "end_time", 0.0)),
            } for it in inner]
            attach_punctuation(text, dicts)
            items.extend(dicts)
        return items

    def phonemize_items(self, items):
        """Attach ``phones`` / ``phone_ids`` to aligned items. No audio needed.

        Pure CPU work (g2p_en + the g2pen tokenizer), so re-running it after a
        phonemizer / dictionary change costs seconds per thousand clips.

        The items are tagged *together* (`HorsePhonemizer.tag_words`) before
        phonemization, so g2p_en can pick the right pronunciation for
        heteronyms ("read" / "refuse" / "present" ...); see AGENTS.md
        "Heteronyms (homographs)".

        Each item's ``punct`` field (set by `attach_punctuation`, i.e. `align`)
        is appended to that item's phone string, so the codes carry the
        punctuation exactly like the whole-transcript path does (``SPAY1K.``
        rather than ``SPAY1K``). Characters the g2pen tokenizer has no id for
        (``;``, ``:``, curly quotes) are dropped instead of becoming ``[UNK]``.

        Args:
            items: Dicts as returned by `align`; ``text`` and the optional
                ``punct`` are used.

        Returns:
            New list of dicts: a copy of each input item plus ``phones`` (str),
            ``phone_ids`` (list[int]) and ``punct`` (the kept punctuation).
        """
        texts = [str(it.get("text", "")) for it in items]
        tags = self.hphzr.tag_words(texts)
        out = []
        for item, text, tag in zip(items, texts, tags):
            punct = self.keep_punctuation(item.get("punct", ""))
            phones = self.hphzr.phonemize(text, tag=tag) + punct
            entry = dict(item)
            entry["text"] = text
            entry["punct"] = punct
            entry["phones"] = phones
            entry["phone_ids"] = self.tokenizer_g2p(phones)['input_ids']
            out.append(entry)
        return out

    def keep_punctuation(self, punct):
        """The part of `punct` the g2pen tokenizer can encode.

        g2p_en passes punctuation through untouched (``SPAY1K.``) and the
        tokenizer has ids for ``, . ? ! -``; anything else (``;``, ``:`` or a
        Unicode quote) would tokenize to ``[UNK]``, so it is dropped here --
        ``semicolons stay out of phone_ids`` is one of the invariants
        `test_phonemize.py` checks.
        """
        if not punct:
            return ''
        unk = self.tokenizer_g2p.unk_token_id
        return ''.join(ch for ch in str(punct)
                       if self.tokenizer_g2p.convert_tokens_to_ids(ch) != unk)

    def phonemize_text(self, text):
        """Phone ids for a whole transcript, without audio or alignment.

        Uses `HorsePhonemizer.phonemize_sentence`, i.e. the words are tagged as
        one sentence (heteronyms resolved from context). This is the cheapest
        way to (re)build ``phone_ids``: no ASR, no aligner, no audio, no GPU.

        Unlike `align` + `phonemize_items` it does not verify that the
        transcript matches the audio, so it keeps clips the aligner would have
        left empty (e.g. an "hmm"-only transcript) and loses word boundaries —
        nothing downstream uses those.

        Args:
            text: Transcript string (the filelist's third field).

        Returns:
            List[int] phone ids (empty when the text has no phonemes).
        """
        phones = self.hphzr.phonemize_sentence(text)
        return self.tokenizer_g2p(phones)['input_ids']

    def extract(self, data_16k, language="English", transcription=None):
        """Align + phonemize + per-word f0 (the original all-in-one path).

        Kept for callers that want word timings *and* pitch contours. The
        stages are available separately via `align`, `phonemize_items`,
        `phonemize_text` and `extract_features_ac`, which is what
        ``2_preprocess.py --only ...`` uses to rebuild one tensor at a time.

        Args:
            data_16k: 1-D numpy array of mono audio at 16 kHz.
            language: See `align`.
            transcription: See `align`.

        Returns:
            ``(items, f0)`` where ``items`` is the list of dicts from
            `phonemize_items`, each additionally holding:

                pitch_contour (np.ndarray): 1-D float array of f0 in Hz for
                    frames whose centers fall in [start_time, end_time).
                    Unvoiced frames are 0. May be empty if the word is
                    shorter than one frame or out of range.
                nonzero_pitch_mean (float): mean of pitch_contour over frames
                    with f0 > 0 (i.e. voiced mean). 0.0 if fully unvoiced /
                    empty.

            and ``f0`` is the whole-clip frame-level f0 contour.
        """
        items = self.align(data_16k, language=language,
                           transcription=transcription)
        f0 = np.asarray(self.extract_pitch(data_16k)).reshape(-1)

        # RMVPE uses hop_length samples @ 16 kHz with center=True STFT, so
        # frame i is centered at t = i * hop / sr (10 ms steps by default).
        mel_extractor = getattr(
            getattr(self.rmvpe_model, "model", None),
            "mel_extractor", None)
        hop = getattr(mel_extractor, "hop_length", 160)
        sr = getattr(mel_extractor, "sampling_rate", 16000)

        out = []
        for item in self.phonemize_items(items):
            contour = _f0_span(f0, item["start_time"], item["end_time"],
                               hop=hop, sr=sr)
            voiced = contour[contour > 0]
            item["pitch_contour"] = contour
            item["nonzero_pitch_mean"] = (
                float(voiced.mean()) if voiced.size > 0 else 0.0)
            out.append(item)
        return out, f0

    def extract_features_ac(self, data_48k):
        # Level-normalise once, up front: `wave` is the stage-1 GAN target and
        # `spec` is what the (frozen) codec sees in stage 2, so both must be on
        # the same scale. The pitch contour is gain-invariant. See
        # `data.audio_norm` in configs/base.yaml (default: `none`).
        data_48k, _ = self.normalize_audio(data_48k)
        data_16k = librosa.resample(data_48k, orig_sr=48000, target_sr=16000)
        ret = {
            'wave': torch.from_numpy(data_48k),
            'spec': self.extract_spec(data_48k),
            'f0': torch.from_numpy(self.extract_pitch(data_16k))
        }
        d = min(ret['spec'].shape[0], ret['f0'].shape[0])
        ret['spec'] = ret['spec'][:d,:]
        ret['f0'] = ret['f0'][:d]
        return ret


    # ------------------------------------------------------------------ #
    # Conditioning hints (one-stop; shared with hints/build.py)
    # ------------------------------------------------------------------ #
    def code_frame_rate(self) -> float:
        """Code frames per second, from the loaded config (25 Hz by default).

        ``data.sampling_rate / data.hop_length / down_factor``; ``down_factor``
        is a ``Generator`` constructor default (4), not a config key, so it is
        hard-coded here with :data:`DEFAULT_CODE_FPS` as the fallback.
        """
        try:
            return float(self.hp.data.sampling_rate) / float(
                self.hp.data.hop_length) / 4.0
        except Exception:
            return DEFAULT_CODE_FPS

    def hint_intensity(self, data_48k) -> float:
        """Perceptual intensity on the 1-9 scale, from a 48 kHz waveform.

        Label-free: a DSP feature set + a HistGradientBoosting regressor ported
        from ``20_intensity_annotation`` (``hints/models/``). Loudness is
        removed (the features peak-normalise), so this is timbre/phonation
        effort, not volume. Unreliable on clips that are mostly digital silence
        -- see ``hints/README.md``.
        """
        wave = np.asarray(data_48k, dtype=np.float32).reshape(-1)
        return _hint_helpers().intensity_from_wave_array(wave, sr=HINT_WAVE_SR)

    def hint_register(self, f0) -> float:
        """Utterance register: voiced-median f0 in semitones re: 100 Hz.

        NaN when the clip has no voiced frames.
        """
        return _hint_helpers().register_from_f0_array(f0)

    def hint_semantic(self, text, model_name=None, device=None):
        """Sentence embedding for `text`: 384-d ``float16``, L2-normalised.

        ``bge-small-en-v1.5`` CLS pooling, the same recipe as
        ``hints/build.py``, so a vector computed here and one stored offline are
        interchangeable. Returns ``None`` for empty text.
        """
        if not text or not str(text).strip():
            return None
        hb = _hint_helpers()
        out = hb.encode_semantics([str(text)], model_name or hb.DEFAULT_SEMANTIC_MODEL,
                                  device)
        return None if out is None else out[0]

    @staticmethod
    def hint_level(value, edges):
        """Bucket a continuous hint value into a level id (0 = absent).

        `edges` is the fitted quantile list from ``hints.json`` ->
        ``levels.<name>.edges`` (``hints.levels.load_level_edges``). This is the
        same mapping the dataset applies at load time.
        """
        from hints.levels import bucket_levels
        return bucket_levels(value, edges)

    @staticmethod
    def load_hint_levels(hints_json):
        """Read every hint's ``{n_levels, edges}`` from a ``hints.json``."""
        from hints.levels import load_levels
        return load_levels(hints_json)

    @staticmethod
    def write_hints(path, record):
        """Atomically save an `extract_hints` record as a ``.cond`` file."""
        _hint_helpers().write_cond(path, record)


    def hint_words(self, items, total_seconds=None, code_fps=None):
        """Word-level timing + relative-pitch hints from aligned items.

        `items` are the dicts from `extract` (or `align` + `phonemize_items`), so
        each must carry ``phone_ids`` and ``start_time`` and, for the pitch hint,
        ``nonzero_pitch_mean`` (which `extract` computes; `align` alone does not).

        Returns the word-level half of a ``.cond`` record:

            phone_word_idx    int16(n_phone)   word index of each phone
            word_start_f      int32(n_word)    word start, in code frames
            word_slot_f       int32(n_word)    start-to-start slot, code frames
            pitch_word_st     float16(n_word)  word voiced mean in semitones
                                               relative to the utterance; NaN if
                                               the word has no voiced frames
            pitch_word_voiced bool(n_word)

        The slots are **start-to-start** (the last one runs to the end of the
        clip), so they partition ``[0, total_seconds * code_fps)`` and per-word
        timing is a complete length specification -- changing one word's pacing
        shifts the clip by that much, with no separate utterance length input.

        Args:
            items: aligned+phonemized word dicts, in time order.
            total_seconds: clip duration (default: the last word's ``end_time``).
            code_fps: code frames per second (default: `code_frame_rate`).

        Raises:
            ValueError: word start times are not non-decreasing.
        """
        items = list(items)
        if not items:
            return {}
        fps = float(code_fps or self.code_frame_rate())
        n_word = len(items)
        starts = np.asarray([float(it["start_time"]) for it in items],
                            dtype=np.float64)
        if np.any(np.diff(starts) < 0):
            raise ValueError("word start times must be non-decreasing")
        if total_seconds is None:
            total_seconds = float(max(float(it["end_time"]) for it in items))

        word_start_f = np.rint(starts * fps).astype(np.int32)
        next_start = np.concatenate([starts[1:], [float(total_seconds)]])
        word_slot_f = np.maximum(np.rint((next_start - starts) * fps), 1.0)
        word_slot_f = word_slot_f.astype(np.int32)

        means = np.asarray([float(it.get("nonzero_pitch_mean") or 0.0)
                            for it in items], dtype=np.float64)
        voiced = means > 0
        pitch_st = np.full(n_word, np.nan, dtype=np.float64)
        if voiced.any():
            ref = float(np.median(means[voiced]))
            if ref > 0:
                pitch_st[voiced] = 12.0 * np.log2(means[voiced] / ref)

        n_phone = int(sum(np.asarray(it.get("phone_ids", [])).size
                          for it in items))
        phone_word_idx = np.zeros(n_phone, dtype=np.int16)
        offset = 0
        for i, it in enumerate(items):
            k = int(np.asarray(it.get("phone_ids", [])).size)
            phone_word_idx[offset:offset + k] = i
            offset += k

        return {
            "phone_word_idx": torch.from_numpy(phone_word_idx),
            "word_start_f": torch.from_numpy(word_start_f),
            "word_slot_f": torch.from_numpy(word_slot_f),
            "pitch_word_st": torch.from_numpy(pitch_st).to(torch.float16),
            "pitch_word_voiced": torch.from_numpy(voiced),
        }

    def extract_hints(self, data_48k, transcription=None, *, intensity=True,
                      register=True, semantic=True, words=False,
                      language="English", code_fps=None):
        """Every conditioning hint for one clip, in the ``.cond`` layout.

        Mirrors ``hints/build.py`` key-for-key (same helpers, same tensor
        dtypes), so a ``.cond`` built offline and one built from live audio are
        interchangeable -- which is what makes this the one-stop entry point for
        an inference UI. `write_hints` saves the result.

        Args:
            data_48k: 1-D mono float waveform at 48 kHz.
            transcription: text for the sentence embedding and, with
                ``words=True``, for the forced alignment.
            intensity: 1-9 DSP intensity (no text, no aligner, ~20 ms CPU).
            register: utterance register in semitones re: 100 Hz. Needs f0, so
                it builds the RMVPE model (the stage-1 pitch extractor).
            semantic: 384-d sentence embedding; needs `transcription`.
            words: also align and emit the per-word timing/pitch fields; needs
                the Qwen / forced-aligner weights (``load_qwen=True``).
            language: forwarded to `align`.
            code_fps: code-frame rate for the word slots (default 25 Hz).

        Returns:
            dict with ``version`` plus whichever of ``intensity``,
            ``register_st``, ``semantic``, ``n_phone``, ``phone_word_idx``,
            ``word_start_f``, ``word_slot_f``, ``pitch_word_st``,
            ``pitch_word_voiced`` were computed. Non-finite values (a fully
            unvoiced register) are omitted rather than stored as NaN, matching
            ``hints/build.py``.
        """
        wave = np.asarray(data_48k, dtype=np.float32).reshape(-1)
        out = {"version": 1}
        if intensity:
            out["intensity"] = torch.tensor([self.hint_intensity(wave)],
                                            dtype=torch.float32)

        data_16k = librosa.resample(wave, orig_sr=HINT_WAVE_SR, target_sr=16000)
        f0 = None
        items = None
        if words:
            if self.qwen_model is None:
                raise RuntimeError(
                    "extract_hints(words=True) needs the Qwen3-ASR / "
                    "forced-aligner weights; construct "
                    "FeatureExtractor(..., load_qwen=True)")
            items, f0 = self.extract(data_16k, language=language,
                                     transcription=transcription)
        elif register:
            f0 = np.asarray(self.extract_pitch(data_16k)).reshape(-1)

        if register and f0 is not None:
            value = self.hint_register(f0)
            if np.isfinite(value):
                out["register_st"] = torch.tensor([value], dtype=torch.float32)
        if semantic:
            vec = self.hint_semantic(transcription)
            if vec is not None:
                out["semantic"] = vec
        if words and items:
            out.update(self.hint_words(
                items, total_seconds=len(wave) / float(HINT_WAVE_SR),
                code_fps=code_fps))
            out["n_phone"] = int(sum(len(it["phone_ids"]) for it in items))
        return out


if __name__ == '__main__':
    from commons import elapsed_timer
    from omegaconf import OmegaConf
    data_16k, _ = librosa.load("test.wav", sr=16000)
    with elapsed_timer() as elapsed:
        fe = FeatureExtractor(hp=OmegaConf.load("configs/base.yaml"))
        print("Loaded extractor %.2fs" % elapsed())
        out, f0 = fe.extract(data_16k)
        print(out)
        print("Finished combined extraction %.2fs" % elapsed())

        data_48k, _ = librosa.load("test.wav", sr=48000)
        feats = fe.extract_features_ac(data_48k)
        for k,v in feats.items():
            print(k, v.shape)