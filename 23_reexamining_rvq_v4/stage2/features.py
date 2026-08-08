import numpy as np
import gc
from svc_helper.speaker.models import SVC5SpeakerEncoder
from svc_helper.pitch.rmvpe import RMVPEModel
from svc5hubert import hubert_model
from utils import print_memory_usage
from vits_extend.stft import TacotronSTFT
from transformers import WhisperFeatureExtractor, WhisperModel
import librosa
from modeling.vits import utils, spectrogram
import torch
import io
import soundfile as sf
from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# Augmentation primitives
# --------------------------------------------------------------------------- #

def _rand_uniform(rng, lo, hi):
    return lo + rng.random() * (hi - lo)


def augment_insert_silence(
    data: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    min_silence_s: float = 0.15,
    max_silence_s: float = 0.6,
    max_insertions: int = 2,
) -> np.ndarray:
    """
    Insert 1..max_insertions random-length silent gaps at random positions.
    This is deliberately meant to be disruptive to Whisper's alignment/
    attention (long silent runs can cause hallucinated or dropped tokens),
    which is the point of training against it.
    """
    n_insertions = rng.integers(1, max_insertions + 1)
    out = data
    for _ in range(n_insertions):
        silence_len = int(_rand_uniform(rng, min_silence_s, max_silence_s) * sr)
        if silence_len <= 0 or out.shape[0] == 0:
            continue
        pos = int(rng.integers(0, out.shape[0]))
        silence = np.zeros(silence_len, dtype=out.dtype)
        out = np.concatenate([out[:pos], silence, out[pos:]])
    return out


def augment_gain(
    data: np.ndarray,
    rng: np.random.Generator,
    min_db: float = -6.0,
    max_db: float = 6.0,
) -> np.ndarray:
    """Apply a random constant gain, in dB, to the whole clip."""
    gain_db = _rand_uniform(rng, min_db, max_db)
    gain = 10.0 ** (gain_db / 20.0)
    out = data * gain
    # avoid clipping introduced by positive gain
    peak = np.abs(out).max() if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out


def augment_waveform(
    data_16k: np.ndarray,
    data_48k: np.ndarray,
    seed: int,
    insert_silence: bool = True,
    gain: bool = True,
):
    """
    Apply the same augmentation "decision" to both the 16k and 48k copies
    of a waveform, using one seeded RNG so the two stay consistent with
    each other (same silence position/length in relative terms, same gain).

    NOTE: silence insertion changes sample count. We insert independently
    at the correct sample rate for each stream but drive both from the same
    rng stream + the same *relative* position (0..1) and *same* duration in
    seconds, so the two views of the signal stay time-aligned enough for
    feature extraction downstream (which treats 16k/48k as separate inputs
    anyway, so exact sample parity isn't required).
    """
    rng = np.random.default_rng(seed)

    out_16k, out_48k = data_16k, data_48k

    if insert_silence:
        # Draw shared silence params once, apply at each sample rate.
        n_insertions = rng.integers(1, 3)
        for _ in range(n_insertions):
            silence_s = _rand_uniform(rng, 0.15, 0.6)
            rel_pos = rng.random()

            len_16k = int(silence_s * 16000)
            len_48k = int(silence_s * 48000)
            pos_16k = int(rel_pos * out_16k.shape[0])
            pos_48k = int(rel_pos * out_48k.shape[0])

            if len_16k > 0 and out_16k.shape[0] > 0:
                out_16k = np.concatenate([
                    out_16k[:pos_16k],
                    np.zeros(len_16k, dtype=out_16k.dtype),
                    out_16k[pos_16k:],
                ])
            if len_48k > 0 and out_48k.shape[0] > 0:
                out_48k = np.concatenate([
                    out_48k[:pos_48k],
                    np.zeros(len_48k, dtype=out_48k.dtype),
                    out_48k[pos_48k:],
                ])

    if gain:
        gain_db = _rand_uniform(rng, -6.0, 6.0)
        gain_lin = 10.0 ** (gain_db / 20.0)
        out_16k = out_16k * gain_lin
        out_48k = out_48k * gain_lin
        peak_16k = np.abs(out_16k).max() if out_16k.size else 0.0
        peak_48k = np.abs(out_48k).max() if out_48k.size else 0.0
        peak = max(peak_16k, peak_48k, 1.0)
        if peak > 1.0:
            out_16k = out_16k / peak
            out_48k = out_48k / peak

    return out_16k, out_48k


class MyFeatures:
    def __init__(self, 
        device='cuda',
        config='configs/base_linux.yaml',
        whisper = 'openai/whisper-large-v2',
        feats_to_extract : set[str] = {'whisper', 'spk', 'f0', 'spec', 'wave'},
        do_normalize=True):
        config = OmegaConf.load(config)
        self.feats_to_extract = feats_to_extract
        self.spk_expected_sample_rate=16000
        self.expected_sample_rate=48000

        if 'whisper' in feats_to_extract:
            whisper_model = whisper
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
            self.device = 'cuda'
            self.model = WhisperModel.from_pretrained(whisper_model).to(self.device).half()
            del self.model.decoder

            # Force freeing memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model.eval()
        if 'hubert' in feats_to_extract:
            self.load_hubert()
        if 'spk' in feats_to_extract:
            self.svc5_spk_model = SVC5SpeakerEncoder(device=device)
        if 'f0' in feats_to_extract:
            self.rmvpe_model = RMVPEModel(device=device, is_half=True)
        if 'spec' in feats_to_extract:
            self.config = config
            self.device = device
        self.do_normalize = do_normalize


    def load_hubert(self):
        print("loading hubert...")
        model = hubert_model.hubert_soft("pretrain/hubert-soft-0d54a1f4.pt")
        model.eval()
        model.half()
        model.to(self.device)
        self.hubert_model = model

    def extract_hubert(self, wav_16k):
        feats = torch.from_numpy(wav_16k).to(self.device)
        feats = feats[None, None, :].half()
        with torch.no_grad():
            vec = self.hubert_model.units(feats).squeeze().data.cpu()
        return vec

    def extract_whisper_features(self, audio_16k):
        inputs = self.feature_extractor(
            audio_16k, sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device).half()
        with torch.no_grad():
            encoder_outputs = self.model.encoder(input_features)
            encoder_features = encoder_outputs.last_hidden_state
            feature_len = inputs.attention_mask.sum(-1).item() / 2 # divide by 2 to get feature length
            feature_len = int(feature_len)
            encoder_features = encoder_features[:, :feature_len, :]
        return encoder_features

    def extract_whisper_features_chunked(self, audio_16k, chunk_len : int = 25):
        CHUNK_SAMPLE = chunk_len * 16000 # samples

        chunks = [audio_16k[i : i + CHUNK_SAMPLE] for i in range(0, len(audio_16k), CHUNK_SAMPLE)]
        chunks_feats = [self.extract_whisper_features(chunk) for chunk in chunks]
        out = torch.cat(chunks_feats, dim=1)
        return out

    def extract_speaker_features(self, file : str):
        return self.svc5_spk_model.extract_feature(file)

    def load_waveforms(self, file: str):
        """Load a file at 16k and 48k, return both arrays. Split out of
        extract_features so callers (e.g. augmentation) can load once and
        run extraction multiple times on modified copies."""
        data_16k, _ = librosa.load(file, sr=16000)
        if type(file) is io.BytesIO:
            file.seek(0)
        data_48k, _ = librosa.load(file, sr=48000)
        return data_16k, data_48k

    def extract_features(self, file: str, no_spk: bool = False, whisper_chunk_len: int = 25):
        data_16k, data_48k = self.load_waveforms(file)
        return self.extract_features_data(data_16k, data_48k, no_spk, whisper_chunk_len)

    def extract_features_augmented(
        self,
        file: str,
        seed: int,
        no_spk: bool = False,
        whisper_chunk_len: int = 25,
        insert_silence: bool = True,
        gain: bool = True,
    ):
        """Load a file, apply waveform-level augmentation (random silence
        insertion + random gain), then run normal feature extraction on
        the augmented waveform. `seed` should differ per augmentation pass
        so repeated calls for the same file produce different augmentations
        but remain reproducible."""

        data_16k, data_48k = self.load_waveforms(file)
        data_16k, data_48k = augment_waveform(
            data_16k, data_48k, seed=seed,
            insert_silence=insert_silence, gain=gain,
        )
        return self.extract_features_data(data_16k, data_48k, no_spk, whisper_chunk_len)

    def extract_features_data(self, data_16k, data_48k, no_spk = False, whisper_chunk_len : int = 25):
        if self.do_normalize:
            data_16k = data_16k / (np.abs(data_16k).max() + 1e-4) * 0.99
            data_48k = data_48k / (np.abs(data_48k).max() + 1e-4) * 0.99
        feat = {}
        if 'whisper' in self.feats_to_extract:
            feat['whisper'] = self.extract_whisper_features_chunked(data_16k, whisper_chunk_len).squeeze(0)
        if 'hubert' in self.feats_to_extract:
            feat['hubert'] = self.extract_hubert(data_16k)
        if 'spk' in self.feats_to_extract and not no_spk:
            spk_file = io.BytesIO()
            sf.write(spk_file, data_16k, samplerate=self.spk_expected_sample_rate,
                        format='WAV', subtype='PCM_16')
            spk_file.seek(0)
            feat['spk'] = self.extract_speaker_features(spk_file) # spk
        if 'f0' in self.feats_to_extract:
            f0_extracted, extras = self.rmvpe_model.extract_pitch2(torch.from_numpy(data_16k),
                return_confidence=True, return_subharmonic_confidence=True, 
                return_inharmonic_confidence=True)
            feat['f0'] = torch.from_numpy(f0_extracted)  # f0
            feat['f0_confidence'] = torch.from_numpy(extras['confidence'])
            feat['f0_subharmonic'] = torch.from_numpy(extras['subharmonic_confidence'])
            feat['f0_inharmonic'] = torch.from_numpy(extras['inharmonic_confidence'])
        if 'spec' in self.feats_to_extract:
            hps = self.config.data
            audio = data_48k
            audio = torch.from_numpy(audio).unsqueeze(0)
            n_fft = hps.filter_length
            sampling_rate = hps.sampling_rate
            hop_size = hps.hop_length
            win_size = hps.win_length
            feat['spec'] = spectrogram.spectrogram_torch(
                audio, n_fft, sampling_rate, hop_size, win_size, center=False).squeeze(0).transpose(0, 1)
        if 'wave' in self.feats_to_extract:
            data_spec = data_48k
            feat['wave'] = torch.from_numpy(data_spec)
        return feat

    def orig_spectrogram(self, file : str):
        # linear spectrogram using the original so-vits-svc 5.0 params
        audio, _ = librosa.load(file, sr=32000)
        if self.do_normalize:
            audio = librosa.util.normalize(audio)
        audio = torch.from_numpy(audio).unsqueeze(0)
        n_fft = 1024
        sampling_rate = 32000
        hop_size = 320
        win_size = 1024

        return spectrogram.spectrogram_torch(
            audio, n_fft, sampling_rate, hop_size, win_size, center=False).squeeze(0).transpose(0, 1)

if __name__ == '__main__':
    import os
    # From so-vits-svc 5.0: inference expects
    # whisper [1, 356, 1280]
    # hubert [1, 356, 256]
    # pitch [1, 356]
    # spk [1, 256]
    # spec [1, 356, 769]

    extractor = MyFeatures(do_normalize=False)
    feats = extractor.extract_features('test/test.wav')
    print_memory_usage()

    for key, value in feats.items():
        print(f"{key} shape: {value.shape} {type(value)}")
        # It looks like: whisper and hubert are half the expected lengths.
        # pitch is the expected length.
        # So we will have to double whisper and hubert in the dataloader, or before inference

        torch.save(value, os.path.join("test", f"test.{key}"))
