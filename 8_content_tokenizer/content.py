from huggingface_hub import snapshot_download
from repcodec import RepCodec
from vevo_repcodec import VevoRepCodec
import os
import yaml
import safetensors
import safetensors.torch
import torch
import torchaudio
import librosa
import numpy as np

class ContentStyleTokenizer:
    # (note this uses hubert large instead of base like RVC)
    def __init__(self, 
        content_tokenizer_cfg_path="./content_style_tokenizer.yaml",):

        local_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            cache_dir="./ckpts/Vevo",
            allow_patterns=["tokenizer/vq8192/*"],
        )
        self.content_tokenizer_cfg_path = content_tokenizer_cfg_path
        self.content_tokenizer_ckpt_path = os.path.join(
            local_dir, "tokenizer/vq8192/model.safetensors"
        )
        stat = np.load("hubert_large_stat.npz")
        self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
        self.hubert_feat_norm_std = torch.tensor(stat["std"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vqvae = self.load_content_tokenizer()
        print("===== Content style tokenizer loaded =====")

        self.hubert_model = self.build_hubert_model()
        print("===== Hubert Large model loaded =====")
    pass

    def load_wav(self, wav_path):
        speech = librosa.load(wav_path, sr=24000)[0]
        speech_tensor = torch.tensor(speech).unsqueeze(0).to(self.device)
        speech16k = torchaudio.functional.resample(speech_tensor, 24000, 16000)
        return speech, speech_tensor, speech16k

    def build_hubert_model(self):
        bundle = torchaudio.pipelines.HUBERT_LARGE
        hubert = bundle.get_model()
        hubert.eval()
        hubert.to(self.device)
        return hubert

    def load_content_tokenizer(self):
        with open(self.content_tokenizer_cfg_path, "r") as f:
            content_tokenizer_cfg = yaml.safe_load(f)
        vqvae = RepCodec(**content_tokenizer_cfg)
        vqvae.eval()

        #ckpt = torch.load(self.content_tokenizer_ckpt_path, map_location="cpu")
        #vqvae.load_state_dict(ckpt["model"]["repcodec"])
        safetensors.torch.load_model(vqvae, self.content_tokenizer_ckpt_path)

        del vqvae.decoder # not needed

        vqvae.to(self.device)
        return vqvae

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

        feats, feat_lengths = self.hubert_model.extract_features(
            wavs, lengths=wav_lens, num_layers=output_layer
        )
        feats = feats[-1]
        return feats, feat_lengths

    def extract_hubert_codes(self, wavs, wav_lens=None, output_layer=18):
        feats, feat_lengths = self.extract_hubert_feature(wavs, wav_lens, output_layer)

        # token_type == "hubert_codec"
        feats = (
            feats - self.hubert_feat_norm_mean.to(feats)
        ) / self.hubert_feat_norm_std.to(feats)
        codecs, quantized = self.vqvae.quantize(feats)  # (B, T)

        T = feats.shape[1]
        arange_tensor = torch.arange(T).expand(codecs.shape[0], T).to(codecs)
        codec_masks = (arange_tensor < feat_lengths.unsqueeze(-1)).int()

        return quantized, codecs, codec_masks

class ContentTokenizer:
    # (note this uses hubert large instead of base like RVC)
    def __init__(self, 
        content_tokenizer_cfg_path="./hubert_large_l18_c32.yaml",):

        local_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            cache_dir="./ckpts/Vevo",
            allow_patterns=["tokenizer/vq32/*"],
        )
        self.content_tokenizer_cfg_path = content_tokenizer_cfg_path
        self.content_tokenizer_ckpt_path = os.path.join(
            local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
        )
        stat = np.load("hubert_large_stat.npz")
        self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
        self.hubert_feat_norm_std = torch.tensor(stat["std"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vqvae = self.load_content_tokenizer()
        print("===== Content tokenizer loaded =====")

        self.hubert_model = self.build_hubert_model()
        print("===== Hubert Large model loaded =====")
    pass

    def load_wav(self, wav_path):
        speech = librosa.load(wav_path, sr=24000)[0]
        speech_tensor = torch.tensor(speech).unsqueeze(0).to(self.device)
        speech16k = torchaudio.functional.resample(speech_tensor, 24000, 16000)
        return speech, speech_tensor, speech16k

    def build_hubert_model(self):
        bundle = torchaudio.pipelines.HUBERT_LARGE
        hubert = bundle.get_model()
        hubert.eval()
        hubert.to(self.device)
        return hubert

    def load_content_tokenizer(self):
        with open(self.content_tokenizer_cfg_path, "r") as f:
            content_tokenizer_cfg = yaml.safe_load(f)
        vqvae = VevoRepCodec(**content_tokenizer_cfg)
        vqvae.eval()

        ckpt = torch.load(self.content_tokenizer_ckpt_path, map_location="cpu")
        vqvae.load_state_dict(ckpt["model"]["repcodec"])
        del vqvae.decoder # not needed

        vqvae.to(self.device)
        return vqvae

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

        feats, feat_lengths = self.hubert_model.extract_features(
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

if __name__ == "__main__":
    content_style_tokenizer = ContentStyleTokenizer()
    speech24k, speech_tensor, speech16k = content_style_tokenizer.load_wav("test.wav")
    quantized, codecs, codec_masks = content_style_tokenizer.extract_hubert_codes(speech16k)
    print("=== Content Style Tokenizer ===")
    print(f"Quantized shape: {quantized.shape}")    
    print(f"Codecs shape: {codecs.shape}")

    content_tokenizer = ContentTokenizer()
    quantized, codecs, codec_masks = content_tokenizer.extract_hubert_codes(speech16k)
    print("=== Content Tokenizer ===")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Codecs shape: {codecs.shape}")