import os
import numpy as np
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from modeling.vevo_repcodec import VevoRepCodec
from svc_helper.sfeatures.models import RVCHubertModel, SVC5WhisperModel
from svc_helper.pitch.rmvpe import RMVPEModel
import torchaudio
from mel_processing import spectrogram_torch
from einops import rearrange
import torch
import torch.nn.functional as F
import librosa

from transformers import WhisperFeatureExtractor, WhisperModel
import numpy
import torch


class MyFeatures:
    def __init__(self, 
        extract_hubert=True,
        extract_whisper=True,
        extract_vevo=False,
        config_path = 'data_config.yaml',):
        config = OmegaConf.load(config_path)
        self.device = 'cuda'
        self.window = 160 # from RVC
        if extract_hubert:
            self.rvc_model = RVCHubertModel(device=self.device, is_half=True)

        if extract_whisper:
            model_name = "openai/whisper-base.en"
            feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
            model = WhisperModel.from_pretrained(model_name)
            model.eval()
            self.feature_extractor = feature_extractor
            self.device = "cuda"
            self.whisper_model = model.to(self.device)
            del self.whisper_model.decoder
            self.whisper_model = model

        self.rmvpe_model = RMVPEModel(device=self.device)
        self.extract_hubert = extract_hubert
        self.extract_whisper = extract_whisper
        self.extract_vevo = extract_vevo

        if extract_vevo:
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir="./vevo_ckpts/Vevo",
                allow_patterns=["tokenizer/vq32/*"],
            )
            self.content_tokenizer_ckpt_path = os.path.join(
                local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
            )

            stat = np.load(config.features.get('hubert_stats_path', 'hubert_large_stat.npz'))
            self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
            self.hubert_feat_norm_std = torch.tensor(stat["std"])

            self.vqvae = self.load_content_tokenizer(
                OmegaConf.to_container(config.vevo))
            print("===== Content tokenizer loaded =====")

            self.large_hubert_model = self.build_hubert_model()
            print("===== Hubert Large model loaded =====")

    def load_content_tokenizer(self, cfg):
        vqvae = VevoRepCodec(**cfg)
        vqvae.eval()

        ckpt = torch.load(self.content_tokenizer_ckpt_path, map_location="cpu")
        vqvae.load_state_dict(ckpt["model"]["repcodec"])
        del vqvae.decoder # not needed

        vqvae.to(self.device)
        return vqvae

    def build_hubert_model(self):
        bundle = torchaudio.pipelines.HUBERT_LARGE
        hubert = bundle.get_model()
        hubert.eval()
        hubert.to(self.device)
        return hubert

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

    def f0_to_coarse(pitch: np.ndarray) -> torch.Tensor:
        """Converts f0 to coarse representation."""
        if type(pitch) is torch.Tensor:
            pitch = pitch.detach().cpu().numpy()
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
        return torch.from_numpy(f0_coarse).unsqueeze(0)

    def extract_hubert_feature(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            feats: [B, T, D]
            feat_lengths: [B]
        """
        if wav_lens is None:
            wav_lens = torch.tensor([wavs.shape[1]] * wavs.shape[0]).to(wavs).int()

        wavs = wavs.to(self.device)
        wav_lens = wav_lens.to(self.device)
        feats, feat_lengths = self.large_hubert_model.extract_features(
            wavs, lengths=wav_lens, num_layers=output_layer
        )
        feats = feats[-1]
        return feats, feat_lengths

    def extract_hubert_codes(self, wavs, wav_lens=None, output_layer=18):
        feats, feat_lengths = self.extract_hubert_feature(wavs, wav_lens, output_layer)

        # token_type == "hubert_codec"
        x = self.vqvae.encoder(feats.transpose(1, 2))
        z = self.vqvae.projector(x)
        quantized, idx = self.vqvae.quantizer.codebook.forward_index(z.transpose(2, 1))
        codecs = idx[0]  # (B, T)

        T = codecs.shape[1]
        arange_tensor = torch.arange(T).expand(codecs.shape[0], T).to(codecs)
        codec_masks = (arange_tensor < feat_lengths.unsqueeze(-1)).int()

        return quantized, codecs, codec_masks

    def get_features(self, data_16k : np.ndarray, data_48k : np.ndarray=None):
        data_16k = librosa.util.normalize(data_16k) * 0.95
        if data_48k is not None:
            data_48k = librosa.util.normalize(data_48k) * 0.95

        # [1, T, D]
        if self.extract_hubert:
            rvc_feat = self.rvc_model.extract_features(torch.from_numpy(data_16k))

        if self.extract_whisper:
            whisp_feat = self.get_whisper_features(data_16k)

        if data_48k is not None:
            spec = spectrogram_torch(
                torch.from_numpy(data_48k).squeeze().unsqueeze(0),
                n_fft=2048,
                sampling_rate=48000,
                hop_size=480,
                win_size=2048,
                center=False
            ) # [1, 1025, T]
            spec = rearrange(spec, "1 c t -> 1 t c")
        
        # [T2]
        pitch = self.rmvpe_model.extract_pitch(data_16k)

        if self.extract_vevo:
            quantized, codecs, _ = self.extract_hubert_codes(
                torch.from_numpy(data_16k).unsqueeze(0))

            quantized = rearrange(quantized, "b t d -> b d t")
            quantized = F.interpolate(quantized, scale_factor=2)
            quantized = rearrange(quantized, "b d t -> b t d")
        
        if self.extract_hubert:
            # interpolate rvc up 2x
            rvc_feat = rearrange(rvc_feat, "b t d -> b d t")
            rvc_feat = F.interpolate(rvc_feat, scale_factor=2)
            rvc_feat = rearrange(rvc_feat, "b d t -> b t d")

        # interpolate whisp_feat up 2x
        if self.extract_whisper:
            whisp_feat = rearrange(whisp_feat, "b t d -> b d t")
            whisp_feat = F.interpolate(whisp_feat, scale_factor=2)
            whisp_feat = rearrange(whisp_feat, "b d t -> b t d")

        # coarse f0 for encoder input
        f0bak = pitch.copy()
        f0_coarse = MyFeatures.f0_to_coarse(pitch)

        # truncation
        p_len = data_16k.shape[0] // self.window
        if self.extract_hubert:
            p_len = min(p_len, rvc_feat.shape[1])
        if self.extract_whisper:
            p_len = min(p_len, whisp_feat.shape[1])
        if self.extract_vevo:
            p_len = min(p_len, quantized.shape[1])

        f0_coarse = f0_coarse[:, :p_len]
        f0bak = f0bak[:p_len]
        if self.extract_hubert:
            rvc_feat = rvc_feat[:, :p_len, :]
        if self.extract_whisper:
            whisp_feat = whisp_feat[:, :p_len, :]
        if data_48k is not None:
            spec = spec[:, :p_len, :]
        if self.extract_vevo:
            quantized = quantized[:, :p_len, :]

        ret = {
            "pitch": f0_coarse,
            "pitch_fine": torch.from_numpy(f0bak).unsqueeze(0)
        }

        if self.extract_whisper:
            ret["whisp_feat"] = whisp_feat

        if self.extract_hubert:
            ret["rvc_feat"] = rvc_feat
        
        if self.extract_vevo:
            ret["vevo_quantized"] = quantized

        if data_48k is not None:
            ret["spec"] = spec
            ret["wave"] = torch.from_numpy(data_48k) # [T3]
        return ret

if __name__ == '__main__':
    import librosa
    data_16k, rate = librosa.load('tests/test_speech2.flac', sr=16000)
    data_48k, _ = librosa.load('tests/test_speech2.flac', sr=48000)
    features = MyFeatures(extract_hubert=False, extract_vevo=True).get_features(data_16k, data_48k)