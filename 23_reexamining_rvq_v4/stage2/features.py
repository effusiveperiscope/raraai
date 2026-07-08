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

    def extract_features(self, file : str, no_spk : bool = False, whisper_chunk_len : int = 25):
        data_16k, _ = librosa.load(file, sr=16000)
        data_48k, _ = librosa.load(file, sr=48000)
        if data_16k.sum() == 0:
            raise ValueError(f'File {file} is empty')
        return self.extract_features_data(data_16k, data_48k, no_spk, whisper_chunk_len)

    def extract_features_data(self, data_16k, data_48k, no_spk = False, whisper_chunk_len : int = 25):
        if self.do_normalize:
            data_16k = data_16k / (np.abs(data_16k).max()) * 0.99
            data_48k = data_48k / (np.abs(data_48k).max()) * 0.99
        feat = {}
        if 'whisper' in self.feats_to_extract:
            feat['whisper'] = self.extract_whisper_features_chunked(data_16k, whisper_chunk_len).squeeze(0)
        if 'hubert' in self.feats_to_extract:
            feat['hubert'] = self.extract_hubert(data_16k)
        if 'spk' in self.feats_to_extract and not no_spk:
            file = io.BytesIO()
            sf.write(file, data_16k, samplerate=self.spk_expected_sample_rate,
                        format='WAV', subtype='PCM_16')
            file.seek(0)
            feat['spk'] = self.extract_speaker_features(file) # spk
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
