import numpy as np
from svc_helper.sfeatures.models import RVCHubertModel, SVC5WhisperModel
from svc_helper.pitch.rmvpe import RMVPEModel
from einops import rearrange
import torch
import torch.nn.functional as F

from transformers import WhisperFeatureExtractor, WhisperModel
import numpy
import torch

class MyFeatures:
    def __init__(self):
        self.device = 'cuda'
        self.window = 160 # from RVC
        self.rvc_model = RVCHubertModel(device=self.device, is_half=True)

        model_name = "openai/whisper-small.en"
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name)
        model.eval()
        self.feature_extractor = feature_extractor
        self.device = "cuda"
        self.whisper_model = model.to(self.device)
        del self.whisper_model.decoder
        self.whisper_model = model

        self.rmvpe_model = RMVPEModel(device=self.device)

    def get_whisper_features(self, audio: numpy.ndarray):
        inputs = self.feature_extractor(audio, 
            sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True)
        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            hidden_states = self.whisper_model.encoder(input_features).last_hidden_state
            feature_len = inputs.attention_mask.sum(-1).item() / 2 # divide by 2 to get feature length
            feature_len = int(feature_len)
            hidden_states = hidden_states[:, :feature_len, :]
        return hidden_states

    def get_features(self, data_16k : np.ndarray):
        # [1, T, D]
        rvc_feat = self.rvc_model.extract_features(torch.from_numpy(data_16k))

        whisp_feat = self.get_whisper_features(data_16k)
        
        # [T2]
        pitch = self.rmvpe_model.extract_pitch(data_16k)
        
        # interpolate rvc up 2x
        rvc_feat = rearrange(rvc_feat, "b t d -> b d t")
        rvc_feat = F.interpolate(rvc_feat, scale_factor=2)
        rvc_feat = rearrange(rvc_feat, "b d t -> b t d")

        # interpolate whisp_feat up 2x
        whisp_feat = rearrange(whisp_feat, "b t d -> b d t")
        whisp_feat = F.interpolate(whisp_feat, scale_factor=2)
        whisp_feat = rearrange(whisp_feat, "b d t -> b t d")

        # coarse f0 for encoder input
        f0bak = pitch.copy()
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_mel = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        f0_coarse = torch.from_numpy(f0_coarse).unsqueeze(0)

        # truncation
        p_len = data_16k.shape[0] // self.window
        if rvc_feat.shape[1] < p_len:
            p_len = rvc_feat.shape[1]
            whisp_feat = whisp_feat[:, :p_len, :]
            f0_coarse = f0_coarse[:, :p_len]
            f0bak = f0bak[:p_len]

        return {
            "rvc_feat": rvc_feat,
            "whisp_feat": whisp_feat,
            "pitch": f0_coarse,
            "pitch_fine": torch.from_numpy(f0bak).unsqueeze(0)
        }

if __name__ == '__main__':
    import librosa
    data, rate = librosa.load('tests/test_speech2.flac', sr=RVCHubertModel.expected_sample_rate)
    features = MyFeatures().get_features(data)
    print(features["rvc_feat"].shape)
    print(features["whisp_feat"].shape)
    print(features["pitch"].shape)
    print(features["pitch_fine"].shape)