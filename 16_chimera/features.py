from svc_helper.sfeatures.models import SVC5HubertModel, SVC5WhisperModel
from svc_helper.speaker.models import SVC5SpeakerEncoder
from svc_helper.pitch.rmvpe import RMVPEModel
from utils import print_memory_usage
from vits_extend.stft import TacotronSTFT
import librosa
from modeling.vits import utils, spectrogram
import torch
from omegaconf import OmegaConf

class MyFeatures:
    def __init__(self, 
        device='cuda',
        config='config/svc5_base.yaml',
        feats_to_extract : set[str] = {'whisper', 'hubert', 'spk', 'f0', 'spec', 'wave'},
        do_normalize=True):
        config = OmegaConf.load(config)
        self.feats_to_extract = feats_to_extract
        self.expected_sample_rate=16000
        if 'whisper' in feats_to_extract:
            self.svc5_whisper_model = SVC5WhisperModel(device=device, is_half=True)
        if 'hubert' in feats_to_extract:
            self.svc5_hubert_model = SVC5HubertModel(device=device, is_half=True)
        if 'spk' in feats_to_extract:
            self.svc5_spk_model = SVC5SpeakerEncoder(device=device)
        if 'f0' in feats_to_extract:
            self.rmvpe_model = RMVPEModel(device=device, is_half=True)
        if 'spec' in feats_to_extract:
            self.config = config
            self.device = device
        self.do_normalize = do_normalize

    def extract_features(self, file : str):
        data, _ = librosa.load(file, sr=self.expected_sample_rate)
        if self.do_normalize:
            data = librosa.util.normalize(data)
        feat = {}
        if 'whisper' in self.feats_to_extract:
            feat['whisper'] = self.svc5_whisper_model.extract_features(torch.from_numpy(data)) # ppg
        if 'hubert' in self.feats_to_extract:
            feat['hubert'] = self.svc5_hubert_model.extract_features(torch.from_numpy(data)) # vec
        if 'spk' in self.feats_to_extract:
            feat['spk'] = self.svc5_spk_model.extract_feature(file) # spk
        if 'f0' in self.feats_to_extract:
            feat['f0'] = torch.from_numpy(self.rmvpe_model.extract_pitch(torch.from_numpy(data))) # f0
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
        if 'wave' in self.feats_to_extract:
            data_spec, _ = librosa.load(file, sr=self.config.data.sampling_rate)
            if self.do_normalize:
                data_spec = librosa.util.normalize(data_spec)
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