from svc_helper.speaker.models import SVC5SpeakerEncoder
from svc_helper.pitch.rmvpe import RMVPEModel
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
        feats_to_extract : set[str] = {'whisper', 'spk', 'f0', 'spec', 'wave'},
        do_normalize=True):
        config = OmegaConf.load(config)
        self.feats_to_extract = feats_to_extract
        self.expected_sample_rate=16000

        if 'whisper' in feats_to_extract:
            whisper_model = 'openai/whisper-base'
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
            self.device = 'cuda'
            self.model = WhisperModel.from_pretrained(whisper_model).to(self.device).half()
            del self.model.decoder
            self.model.eval()
        if 'spk' in feats_to_extract:
            self.svc5_spk_model = SVC5SpeakerEncoder(device=device)
        if 'f0' in feats_to_extract:
            self.rmvpe_model = RMVPEModel(device=device, is_half=True)
        if 'spec' in feats_to_extract:
            self.config = config
            self.device = device
        self.do_normalize = do_normalize

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

    def extract_speaker_features(self, file : str):
        return self.svc5_spk_model.extract_feature(file)

    def extract_features(self, file : str):
        data, _ = librosa.load(file, sr=self.expected_sample_rate)
        if data.sum() == 0:
            raise ValueError(f'File {file} is empty')
        if self.do_normalize:
            data = librosa.util.normalize(data)
        feat = {}
        if 'whisper' in self.feats_to_extract:
            feat['whisper'] = self.extract_whisper_features(data).squeeze(0)
            if type(data) == io.BytesIO:
                data.seek(0)
        if 'spk' in self.feats_to_extract:
            feat['spk'] = self.extract_speaker_features(file) # spk
            if type(data) == io.BytesIO:
                data.seek(0)
        if 'f0' in self.feats_to_extract:
            f0_extracted, extras = self.rmvpe_model.extract_pitch2(torch.from_numpy(data),
                return_confidence=True, return_subharmonic_confidence=True, 
                return_inharmonic_confidence=True)
            feat['f0'] = torch.from_numpy(f0_extracted)  # f0
            feat['f0_confidence'] = torch.from_numpy(extras['confidence'])
            feat['f0_subharmonic'] = torch.from_numpy(extras['subharmonic_confidence'])
            feat['f0_inharmonic'] = torch.from_numpy(extras['inharmonic_confidence'])
            if type(data) == io.BytesIO:
                data.seek(0)
        if 'spec' in self.feats_to_extract:

            hps = self.config.data
            audio, _ = librosa.load(file, sr=hps.sampling_rate)
            if self.do_normalize:
                audio = librosa.util.normalize(audio)
            audio = torch.from_numpy(audio).unsqueeze(0)
            n_fft = hps.filter_length
            sampling_rate = hps.sampling_rate
            hop_size = hps.hop_length
            win_size = hps.win_length

            feat['spec'] = spectrogram.spectrogram_torch(
                audio, n_fft, sampling_rate, hop_size, win_size, center=False).squeeze(0).transpose(0, 1)
            if type(data) == io.BytesIO:
                data.seek(0)
        if 'wave' in self.feats_to_extract:
            data_spec, _ = librosa.load(file, sr=self.config.data.sampling_rate)
            if self.do_normalize:
                data_spec = librosa.util.normalize(data_spec)
            feat['wave'] = torch.from_numpy(data_spec)
            if type(data) == io.BytesIO:
                data.seek(0)
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