import os
from huggingface_hub import snapshot_download
import librosa
import numpy as np
import torchaudio
import whisper
import torch
import torchvision
import random
import dac
from svc_helper.pitch.rmvpe import RMVPEModel
from einops import rearrange
from modeling.rep_coco_model import CocoContentStyle, CocoContent, build_coco_model
from modeling.vevo_repcodec import VevoRepCodec
from commons import vevo_load_checkpoint
from omegaconf import OmegaConf
import torch.nn.functional as F
from audiotools import AudioSignal


class MyFeatures:
    def __init__(self, config: OmegaConf, device):
        if "whisper" in config.features.want:
            self.whisper_model = whisper.load_model("medium", device=device)
            self.whisper_model.eval()
            print("===== Whisper loaded =====")

            # "normed_whisper"
            if config.features.use_normed_whisper:
                whisper_stats = torch.load(
                    config.features.whisper_stats_path,
                    map_location=device)
                self.whisper_mean = whisper_stats["mean"]  # (1024,)
                self.whisper_std = whisper_stats["std"]  # (1024,)

        if "content_style_tokens" in config.features.want:
            raise NotImplementedError
            local_dir = snapshot_download(
                repo_id="amphion/Vevo1.5",
                repo_type="model",
                cache_dir="./ckpts/Vevo1.5",
                allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
            )
            contentstyle_tokenizer_ckpt_path = os.path.join(
                local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
            )
            coco_config = config.coco
            self.coco_model = vevo_load_checkpoint(
                build_coco_model,
                coco_config, contentstyle_tokenizer_ckpt_path, device
            )
            print("===== Content style tokenizer loaded =====")

        if "pitch" in config.features.want or "content_interp_pitch" in config.features.want:
            self.rmvpe_model = RMVPEModel(device=device)
            print("===== RMVPE loaded =====")

        if "acoustic" in config.features.want or "acoustic_codes" in config.features.want:
            model_path = dac.utils.download(model_type="44khz")
            self.dac_model = dac.DAC.load(model_path)
            # The decoder IS needed for usage during compress()
            # To calculate the output size, DAC counts
            # all of its convolution layers including the decoder

            self.dac_model.eval()
            self.dac_model.to(device)
            print("===== DAC loaded =====")

        self.device = device
        self.config = config

        if "content_tokens" in config.features.want:
            local_dir = snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir="./ckpts/Vevo",
                allow_patterns=["tokenizer/vq32/*"],
            )
            self.content_tokenizer_ckpt_path = os.path.join(
                local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
            )

            stat = np.load(config.features.hubert_stats_path)
            self.hubert_feat_norm_mean = torch.tensor(stat["mean"])
            self.hubert_feat_norm_std = torch.tensor(stat["std"])

            self.vqvae = self.load_content_tokenizer(
                OmegaConf.to_container(config.vevo))
            print("===== Content tokenizer loaded =====")

            self.hubert_model = self.build_hubert_model()
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

    def spec_augment(self, mel, height):
        """
        Args:
            mel: tensor (..., n_mels, frames)
            height: int 68-92 for default 80 mels
        """
        tgt = torchvision.transforms.functional.resize(mel, (height, mel.shape[-1]))
        if height >= mel.shape[-2]:
            return tgt[:, : mel.shape[-2], :]
        else:
            silence = tgt[:, -1:, :].repeat(1, mel.shape[-2] - height, 1)
            silence += torch.randn_like(silence) / 10
            return torch.cat((tgt, silence), 1)

    def extract_whisper_features(self, wavs, frame_lens, spec_perturb=False):
        """
        Args:
            wavs: (B, T) at 16khz. Note that the max duration should be 30s
            frame_lens: (B,)
        Returns:
            features: (B, T, D)
        """
        # wavs: (batch, max_len)
        wavs = wavs.to(self.device)
        frame_lens = frame_lens.to(self.device)

        wavs = whisper.pad_or_trim(wavs)
        # batch_mel: (batch, 80, 3000)
        batch_mel = whisper.log_mel_spectrogram(wavs, device=self.device)

        if spec_perturb:
            height = random.randint(68, 92)
            batch_mel = self.spec_augment(batch_mel, height)

        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.whisper_model.embed_audio(batch_mel)

        max_len = int(frame_lens.max().item())
        mask = torch.arange(features.size(1), device=features.device).expand(
            len(frame_lens), -1
        ) < frame_lens.unsqueeze(1)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))

        if features.shape[1] >= max_len:
            features = features[:, :max_len, :]
        else:
            padding_frames = max_len - features.shape[1]
            last_frame = features[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            features = torch.cat([features, padding], dim=1)

        if self.config.features.use_normed_whisper:
            features = (features - self.whisper_mean) / self.whisper_std

        return features 

    def extract_pitch(self, wavs):
        """
        Returns (B, T)
        """
        pitches = []
        pitch_lens = []
        wavs = wavs.cpu() # These take CPU
        for wav in wavs:
            pitch = self.rmvpe_model.extract_pitch(wav)
            pitches.append(torch.from_numpy(pitch))
            pitch_lens.append(len(pitch))
        return torch.stack(pitches), torch.tensor(pitch_lens)

    def extract_acoustic(self, wav_path, in_sample_rate=44100):
        # new method
        dacf = self.dac_model.compress(wav_path, normalize_db=-14)
        codes = dacf.codes
        codes = rearrange(codes, 'b d t -> b t d')
        return codes

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

    def extract_features(self, wav_file_path, normalize=True):
        wav_16k, _ = librosa.load(wav_file_path, sr=16000)
        if normalize:
            wav_16k = librosa.util.normalize(wav_16k)

        ret = {}
        want_features : list[str]
        want_features = self.config.features.want

        if ("pitch" in want_features or 
            "pitch_lens" in want_features 
            or "content_interp_pitch" in want_features):
            pitch, pitch_lens = self.extract_pitch(
                torch.from_numpy(wav_16k).unsqueeze(0))
            pitch = pitch.float()
            ret["pitch"] = pitch
            ret["pitch_lens"] = pitch_lens

        if "whisper" in want_features:
            features = self.extract_whisper_features(
                torch.from_numpy(wav_16k).unsqueeze(0), pitch_lens)
            ret["whisper"] = features

        if ("content_quantized" in want_features 
            or "content_tokens" in want_features):
            quantized, codecs, codec_masks = self.extract_hubert_codes(
                torch.from_numpy(wav_16k).unsqueeze(0))
            ret["content_quantized"] = quantized
            ret["content_tokens"] = codecs

        if ("acoustic_codes" in want_features):
            codes = self.extract_acoustic(wav_file_path)
            ret["acoustic_codes"] = codes

        if "content_interp_pitch" in want_features:
            ret["content_interp_pitch"] = F.interpolate(
                pitch.unsqueeze(1),
                size=quantized.shape[1],
                mode="linear",).squeeze(1)

        ret = { k: v.cpu() for k,v in ret.items() }
        ret = { k: ret[k] for k in want_features }


        return ret


if __name__ == "__main__":
    import torch
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/common.yaml")
    config.features.want = ["pitch", "content_interp_pitch", "whisper", "content_tokens", "acoustic_codes"]
    myfeatures = MyFeatures(config, "cuda")

    wavs = torch.randn(2, 32000)

    quantized, codecs, codec_masks = myfeatures.extract_hubert_codes(wavs)
    print("Quantized shape:", quantized.shape) # [2, 99, 1024]
    print("Codecs shape:", codecs.shape) # [2, 99]

    pitch, pitch_lens = myfeatures.extract_pitch(wavs)
    print("Pitch shape:", pitch.shape) # [2, 201]

    features = myfeatures.extract_whisper_features(wavs, pitch_lens)
    print("Whisper shape:", features.shape) # [2, 201, 1024]

    import librosa
    wav_16k, _ = librosa.load("test.wav", sr=16000)
    print("Wav shape:", wav_16k.shape)
    print("Wav length (s): ", wav_16k.shape[0] / 16000)

    codes = myfeatures.extract_acoustic("test.wav")
    print("Acoustic codes shape:", codes.shape) # [1, 360, 9]

    features = myfeatures.extract_features("test.wav")
    print("Content tokens shape:", features["content_tokens"].shape) # [1, 161]
    print("Content interpolation pitch shape:", features["content_interp_pitch"].shape) # [1, 161]