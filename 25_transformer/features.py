import math
import unicodedata
import numpy as np
import torch
from svc_helper.pitch.rmvpe import RMVPEModel
from horsephonemizer import HorsePhonemizer
from transformers import AutoTokenizer 
from modeling.vits import spectrogram, utils
import librosa

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