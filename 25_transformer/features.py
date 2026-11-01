import math
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


class FeatureExtractor:
    def __init__(self, hp, device="cuda:0",
        load_qwen=True):
        self.dtype = torch.bfloat16
        self.device = device
        self.hp = hp

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
        """
        if self.qwen_model is None:
            raise RuntimeError(
                "align() needs the Qwen3-ASR / forced-aligner weights; "
                "construct FeatureExtractor(..., load_qwen=True).")
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
            items = []
            for res in aligned:
                inner = getattr(res, "items", res)
                items.extend(list(inner))
        else:
            if transcription is not None:
                # Explicitly empty transcription: nothing to align.
                items = []
            else:
                results = self.qwen_model.transcribe(
                    audio=(np.asarray(data_16k), 16000),
                    language=language,
                    return_time_stamps=True,
                )
                # Flatten aligned items across utterances (single-audio input
                # normally yields exactly one ASRTranscription).
                items = []
                for utt in results:
                    ts = getattr(utt, "time_stamps", None)
                    if ts is None:
                        continue
                    inner = getattr(ts, "items", ts)
                    items.extend(list(inner))
        # Flatten aligned items across utterances (single-audio input normally
        # yields exactly one ASRTranscription) into plain, serialisable dicts.
        # Everything downstream only needs these three fields.
        return [{
            "text": getattr(it, "text", ""),
            "start_time": float(getattr(it, "start_time", 0.0)),
            "end_time": float(getattr(it, "end_time", 0.0)),
        } for it in items]

    def phonemize_items(self, items):
        """Attach ``phones`` / ``phone_ids`` to aligned items. No audio needed.

        Pure CPU work (g2p_en + the g2pen tokenizer), so re-running it after a
        phonemizer / dictionary change costs seconds per thousand clips.

        The items are tagged *together* (`HorsePhonemizer.tag_words`) before
        phonemization, so g2p_en can pick the right pronunciation for
        heteronyms ("read" / "refuse" / "present" ...); see AGENTS.md
        "Heteronyms (homographs)".

        Args:
            items: Dicts as returned by `align`; only ``text`` is used.

        Returns:
            New list of dicts: a copy of each input item plus ``phones`` (str)
            and ``phone_ids`` (list[int]).
        """
        texts = [str(it.get("text", "")) for it in items]
        tags = self.hphzr.tag_words(texts)
        out = []
        for item, text, tag in zip(items, texts, tags):
            phones = self.hphzr.phonemize(text, tag=tag)
            entry = dict(item)
            entry["text"] = text
            entry["phones"] = phones
            entry["phone_ids"] = self.tokenizer_g2p(phones)['input_ids']
            out.append(entry)
        return out

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