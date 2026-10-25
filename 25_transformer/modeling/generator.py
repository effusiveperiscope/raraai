from einops import rearrange
import math
import torch
from modeling.vevo_repcodec import VevoRepCodec
from modeling.rvc.my_nsf import MyGeneratorNSF
from modeling.vits.models import PosteriorEncoder
from modeling.vits.commons import rand_slice_segments_with_pitch
import torch.nn as nn
import torch.nn.functional as F

# Normalization scale for pitch: log1p(f0) / PITCH_LOG_SCALE maps
# ~[0, 1000 Hz] to ~[0, 1] for stable codec training.
PITCH_LOG_SCALE = math.log1p(1000.0)


def _get_cfg(cfg, key, default):
    try:
        if cfg is None:
            return default
        if key in cfg:
            v = cfg[key]
            return v if v is not None else default
    except Exception:
        pass
    try:
        return getattr(cfg, key, default)
    except Exception:
        return default

class Generator(nn.Module):
    def __init__(self,
        hp,
        down_factor: int = 4):
        super().__init__()
        self.enc_q = PosteriorEncoder(
            hp.data.filter_length // 2 + 1,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.codec = VevoRepCodec(
            input_channels=hp.vits.inter_channels,
            output_channels=hp.vits.inter_channels,
            encode_channels=hp.vits.inter_channels,
            decode_channels=hp.vits.inter_channels,
            code_dim=hp.codec.code_dim,
            codebook_size=hp.codec.codebook_size,
            codebook_num=hp.codec.codebook_num,
            gin_channels=hp.vits.gin_channels
        )
        self.dec = MyGeneratorNSF(hp=hp)
        self.spk_emb = nn.Embedding(hp.vits.max_spk_count, hp.vits.gin_channels)
        self.hp = hp
        # Temporal resampling around the codec so that
        # code rate == decoder frame rate / down_factor (default 1/4).
        # Convs operate on (B, C, T).
        self.down_factor = down_factor
        self.downsample = nn.Conv1d(
            hp.vits.inter_channels,
            hp.vits.inter_channels,
            kernel_size=down_factor,
            stride=down_factor,
        )
        self.upsample = nn.ConvTranspose1d(
            hp.vits.inter_channels,
            hp.vits.inter_channels,
            kernel_size=down_factor,
            stride=down_factor,
        )

        # --- Low-parameter pitch codec ---
        # Same downsampling/upsampling factor as the spectrogram path so
        # that pitch codes are at the same rate as the spec codes.
        # Operates on (B, 1, T) f0 contours. Config overridable via
        # hp.pitch_codec: {hidden_channels, code_dim, codebook_size,
        # codebook_num}.
        pitch_codec_cfg = _get_cfg(hp, "pitch_codec", None)
        p_hidden = int(_get_cfg(pitch_codec_cfg, "hidden_channels", 32))
        p_code_dim = int(_get_cfg(pitch_codec_cfg, "code_dim", 32))
        p_codebook_size = int(_get_cfg(pitch_codec_cfg, "codebook_size", 32))
        p_codebook_num = int(_get_cfg(pitch_codec_cfg, "codebook_num", 2))
        self.pitch_downsample = nn.Conv1d(
            1,
            p_hidden,
            kernel_size=down_factor,
            stride=down_factor,
        )
        self.pitch_codec = VevoRepCodec(
            input_channels=p_hidden,
            output_channels=p_hidden,
            encode_channels=p_hidden,
            decode_channels=p_hidden,
            code_dim=p_code_dim,
            codebook_size=p_codebook_size,
            codebook_num=p_codebook_num,
            gin_channels=hp.vits.gin_channels,
        )
        self.pitch_upsample = nn.ConvTranspose1d(
            p_hidden,
            p_hidden,
            kernel_size=down_factor,
            stride=down_factor,
        )
        self.pitch_out_proj = nn.Conv1d(p_hidden, 1, kernel_size=1)
        # Voicing (V/UV) head. It predicts, per decoder frame, whether the
        # pitch code represents voiced speech. Trained with BCE against
        # (f0 > 0), so the pitch codes carry their own unvoiced decision and
        # synthesis no longer needs an external voicing mask.
        self.pitch_uv_proj = nn.Conv1d(p_hidden, 1, kernel_size=1)
        # Start biased towards "voiced" with near-zero weights: a checkpoint
        # that predates this head (so these weights are randomly initialized
        # by load_state_dict_mismatch) then reproduces the legacy all-voiced
        # behaviour instead of randomly dropping frames to silence.
        nn.init.normal_(self.pitch_uv_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pitch_uv_proj.bias, 4.0)
        # Decision threshold on sigmoid(uv_logit) used at inference.
        self.pitch_uv_threshold = float(_get_cfg(
            _get_cfg(hp, "train", None), "pitch_uv_threshold", 0.5))

    def freeze_non_pitch(self):
        """Freeze every parameter except the pitch path components.

        Keeps ``pitch_downsample``, ``pitch_codec``, ``pitch_upsample``,
        ``pitch_out_proj`` and ``pitch_uv_proj`` trainable and freezes
        everything else (posterior encoder, spectrogram codec, resampling
        layers, NSF decoder and the speaker table). Used by stage-1
        ``--pitch_only`` fine-tuning so only the pitch path is updated.
        """
        pitch_prefixes = (
            "pitch_downsample.",
            "pitch_codec.",
            "pitch_upsample.",
            "pitch_out_proj.",
            "pitch_uv_proj.",
        )
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(pitch_prefixes))

    def _pitch_codec_forward(self, f0, t_pad, g):
        """Run the low-parameter pitch codec.

        Args:
            f0: (B, T) f0 contour in Hz, 0 == unvoiced.
            t_pad: same right-pad used for the spectrogram path so that
                pitch codes land at exactly the same rate as spec codes.
            g: speaker embedding.
        Returns:
            f0_hat: (B, T_full) reconstructed log-normalized pitch at the
                decoder frame rate (trimmed of padding).
            pq_ds: (B, T_ds, H) quantized pitch codes at the code rate.
            px_ds: (B, T_ds, H) pre-quantization pitch features.
            p_vqloss, p_perplexity: VQ stats from the pitch codec.
            f0_target: (B, T_full) log-normalized pitch target.
            p_uv_logits: (B, T_full) voiced/unvoiced logits (higher ==
                voiced), decoded from the same quantized pitch codes.
        """
        f0_in = f0.unsqueeze(1)  # (B, 1, T)
        f0_target = torch.log1p(torch.clamp(f0_in, min=0.0)) / PITCH_LOG_SCALE
        t_full = f0_target.shape[-1]
        if t_pad > 0:
            f0_target = F.pad(f0_target, (0, t_pad))
        # Same down_factor as the spec path -> same code rate.
        px = self.pitch_downsample(f0_target)  # (B, H, T_ds)
        px_ds = rearrange(px, "b c t -> b t c")
        pq, _, _, _, p_vqloss, p_perplexity = self.pitch_codec(
            px_ds, do_uq=False, g=g)
        pq_ds = pq
        pq_ct = rearrange(pq, "b t c -> b c t")
        pq_up = self.pitch_upsample(pq_ct)
        if t_pad > 0:
            pq_up = pq_up[..., : pq_up.shape[-1] - t_pad]
        f0_hat = self.pitch_out_proj(pq_up).squeeze(1)  # (B, T_full)
        p_uv_logits = self.pitch_uv_proj(pq_up).squeeze(1)  # (B, T_full)
        return (f0_hat, pq_ds, px_ds, p_vqloss, p_perplexity,
                f0_target[..., :t_full].squeeze(1), p_uv_logits)

    def forward(self, spec, spec_l, f0, sid):
        g = self.spk_emb(sid)
        # enc_q returns (B, C, T) with T = decoder frame rate.
        x, x_mask = self.enc_q(spec, spec_l)

        # Downsample encoder frames by down_factor so codec runs at
        # 1/down_factor of the decoder frame rate (default 1/4).
        # Pad time so it is divisible by down_factor (pad on the right).
        t_pad = (-x.shape[-1]) % self.down_factor
        if t_pad > 0:
            x = F.pad(x, (0, t_pad))
        x_ds = self.downsample(x)  # (B, C, T/down_factor)
        # VevoRepCodec.forward expects (B, T, C).
        x_ds = rearrange(x_ds, "b c t -> b t c")

        yq, _, _, _, vqloss, perplexity = self.codec(
            x_ds, do_uq=False, g=g)
        yq_ds = yq

        # Upsample codec output back to decoder frame rate.
        # Codec returns yq as (B, T', C).
        yq = rearrange(yq, "b t c -> b c t")
        yq = self.upsample(yq)
        if t_pad > 0:
            yq = yq[..., : yq.shape[-1] - t_pad]
        # Align f0 length with the (unpadded) encoder frame length so the
        # pitch branch sees the same T as the spec path.
        t_frames = x.shape[-1] - t_pad
        if f0.shape[-1] < t_frames:
            f0_aligned = F.pad(f0, (0, t_frames - f0.shape[-1]))
        else:
            f0_aligned = f0[..., :t_frames]
        (f0_hat, pq_ds, px_ds, p_vqloss, p_perplexity,
            f0_target, p_uv_logits) = self._pitch_codec_forward(
                f0_aligned, t_pad, g)
        y_slice, f0_slice, ids_slice = rand_slice_segments_with_pitch(
            yq, f0, spec_l, self.hp.data.segment_size // self.hp.data.hop_length)
        audio = self.dec(g, y_slice, f0_slice)
        return (audio, ids_slice, yq_ds, x_ds, vqloss, perplexity,
                f0_hat, f0_target, pq_ds, px_ds, p_vqloss, p_perplexity,
                p_uv_logits)

    def infer(self, spec, spec_l, f0, sid, use_pitch_recon=False,
              predict_voicing=False):
        g = self.spk_emb(sid)
        # enc_q returns (B, C, T) with T = decoder frame rate.
        x, x_mask = self.enc_q(spec, spec_l)

        # Downsample encoder frames by down_factor so codec runs at
        # 1/down_factor of the decoder frame rate (default 1/4).
        # Pad time so it is divisible by down_factor (pad on the right).
        t_pad = (-x.shape[-1]) % self.down_factor
        if t_pad > 0:
            x = F.pad(x, (0, t_pad))
        x_ds = self.downsample(x)  # (B, C, T/down_factor)
        # VevoRepCodec.forward expects (B, T, C).
        x_ds = rearrange(x_ds, "b c t -> b t c")

        yq, _, _, _, vqloss, perplexity = self.codec(
            x_ds, do_uq=False, g=g)

        # Upsample codec output back to decoder frame rate.
        # Codec returns yq as (B, T', C).
        yq = rearrange(yq, "b t c -> b c t")
        yq = self.upsample(yq)
        if t_pad > 0:
            yq = yq[..., : yq.shape[-1] - t_pad]
        if use_pitch_recon:
            t_frames = x.shape[-1] - t_pad
            if f0.shape[-1] < t_frames:
                f0_aligned = F.pad(f0, (0, t_frames - f0.shape[-1]))
            else:
                f0_aligned = f0[..., :t_frames]
            (f0_hat_norm, _, _, _, _, _,
                uv_logits) = self._pitch_codec_forward(f0_aligned, t_pad, g)
            # Denormalize log1p pitch back to Hz, keep unvoiced at 0.
            # `predict_voicing` uses the codec's own V/UV head; the default
            # keeps the legacy behaviour of trusting the input f0 mask.
            if predict_voicing:
                voiced = (torch.sigmoid(uv_logits)
                          >= self.pitch_uv_threshold).float()
            else:
                voiced = (f0_aligned > 0).float()
            f0 = (torch.expm1(torch.clamp(
                f0_hat_norm, min=0.0) * PITCH_LOG_SCALE) * voiced)
            if f0.shape[-1] < yq.shape[-1]:
                f0 = F.pad(f0, (0, yq.shape[-1] - f0.shape[-1]))
            else:
                f0 = f0[..., : yq.shape[-1]]
        audio = self.dec(g, yq, f0)
        return audio

    @staticmethod
    def _normalize_code_indices(indices, codebook_num, name):
        """Accept int codes as (Q, B, T) or (B, T, Q); return (Q, B, T)."""
        idx = indices if torch.is_tensor(indices) else torch.tensor(indices)
        if idx.dim() != 3:
            raise ValueError(
                f"{name} must be 3D (Q, B, T) or (B, T, Q), "
                f"got shape {tuple(idx.shape)}")
        if idx.shape[0] == codebook_num:
            return idx.long()
        if idx.shape[-1] == codebook_num:
            return idx.permute(2, 0, 1).long()
        raise ValueError(
            f"{name} must have codebook dim {codebook_num} first or last, "
            f"got shape {tuple(idx.shape)}")

    @staticmethod
    def _residual_lookup(quantizer, indices_qbt):
        """Sum per-layer codebook embeddings for RVQ indices (Q, B, T)."""
        layers = quantizer.codebook.layers
        if indices_qbt.shape[0] != len(layers):
            raise ValueError(
                f"Expected {len(layers)} codebooks, got {indices_qbt.shape[0]}")
        out = None
        for q, layer in enumerate(layers):
            v = F.embedding(indices_qbt[q], layer.codebook)  # (B, T, D)
            out = v if out is None else out + v
        return out.transpose(1, 2)  # (B, D, T)

    def encode_codes(self, spec, spec_l, f0, sid):
        """Encode spec + f0 into integer codes at the shared code rate.

        Returns (spec_codes, pitch_codes) as (Q, B, T_ds) int tensors.
        """
        g = self.spk_emb(sid)
        x, _ = self.enc_q(spec, spec_l)
        t_pad = (-x.shape[-1]) % self.down_factor
        if t_pad > 0:
            x = F.pad(x, (0, t_pad))
        x_ds = self.downsample(x)  # (B, C, T_ds)
        z = self.codec.projector(x_ds)
        _, spec_codes = self.codec.quantizer.inference(z)

        t_frames = x.shape[-1] - t_pad
        if f0.shape[-1] < t_frames:
            f0_aligned = F.pad(f0, (0, t_frames - f0.shape[-1]))
        else:
            f0_aligned = f0[..., :t_frames]
        f0_target = (torch.log1p(torch.clamp(
            f0_aligned, min=0.0)) / PITCH_LOG_SCALE).unsqueeze(1)
        if t_pad > 0:
            f0_target = F.pad(f0_target, (0, t_pad))
        px = self.pitch_downsample(f0_target)
        pz = self.pitch_codec.projector(px)
        _, pitch_codes = self.pitch_codec.quantizer.inference(pz)
        return spec_codes, pitch_codes

    def infer_from_codes(self, spec_codes, pitch_codes, sid,
                         out_len=None, voiced=None, uv_threshold=None):
        """Synthesize audio directly from integer spec + pitch codes.

        Args:
            spec_codes: (Q_spec, B, T) or (B, T, Q_spec) int codes.
            pitch_codes: (Q_pitch, B, T) or (B, T, Q_pitch) int codes at
                the same rate/length as spec_codes.
            sid: (B,) speaker ids.
            out_len: optional frame length at the decoder rate to trim to.
            voiced: optional voicing mask, (B, T_frames) or (B, T_codes).
                If None, the pitch codec's voicing head decides which frames
                are voiced; unvoiced frames render as 0 Hz. Pass a mask to
                override the prediction.
            uv_threshold: optional override for the voicing decision
                threshold applied to ``sigmoid(uv_logits)``.
        Returns:
            audio: (B, 1, T_audio) waveform.
        """
        g = self.spk_emb(sid)
        spec_idx = self._normalize_code_indices(
            spec_codes, len(self.codec.quantizer.codebook.layers),
            "spec_codes").to(g.device)
        pitch_idx = self._normalize_code_indices(
            pitch_codes, len(self.pitch_codec.quantizer.codebook.layers),
            "pitch_codes").to(g.device)
        if spec_idx.shape[1] != pitch_idx.shape[1]:
            raise ValueError(
                "spec/pitch batch mismatch: "
                f"{spec_idx.shape[1]} vs {pitch_idx.shape[1]}")
        if spec_idx.shape[2] != pitch_idx.shape[2]:
            raise ValueError(
                "spec/pitch code lengths must match (same rate): "
                f"{spec_idx.shape[2]} vs {pitch_idx.shape[2]}")
        if sid.shape[0] != spec_idx.shape[1]:
            raise ValueError(
                f"sid batch {sid.shape[0]} != codes batch {spec_idx.shape[1]}")

        # Spec path: indices -> quantized latent -> codec decoder -> upsample.
        zq = self._residual_lookup(self.codec.quantizer, spec_idx)
        y = self.codec.decoder(zq, self.codec.g_proj(g))  # (B, C, T)
        yq = self.upsample(y)  # (B, C, T * down_factor)
        t_full = yq.shape[-1]
        if out_len is not None:
            yq = yq[..., :out_len]
            t_full = yq.shape[-1]

        # Pitch path: indices -> quantized latent -> codec decoder ->
        # upsample -> 1-channel log-norm pitch -> Hz.
        pzq = self._residual_lookup(self.pitch_codec.quantizer, pitch_idx)
        py = self.pitch_codec.decoder(pzq, self.pitch_codec.g_proj(g))
        py_up = self.pitch_upsample(py)[..., :t_full]
        f0_norm = self.pitch_out_proj(py_up).squeeze(1)  # (B, T_full)
        uv_logits = self.pitch_uv_proj(py_up).squeeze(1)  # (B, T_full)

        if voiced is None:
            # No external mask: decode voicing from the pitch codes.
            thr = (self.pitch_uv_threshold if uv_threshold is None
                   else float(uv_threshold))
            v = (torch.sigmoid(uv_logits) >= thr).float()
        else:
            vt = (voiced if torch.is_tensor(voiced)
                  else torch.tensor(voiced, device=f0_norm.device))
            vt = vt.float().to(f0_norm.device)
            if vt.dim() != 2 or vt.shape[0] != f0_norm.shape[0]:
                raise ValueError(
                    "voiced must be (B, T_frames) or (B, T_codes), got "
                    f"{tuple(vt.shape)}")
            if vt.shape[1] == spec_idx.shape[2]:
                # Code-rate mask: expand by the downsampling factor.
                vt = vt.repeat_interleave(self.down_factor, dim=1)
            v = vt[..., :t_full]
            if v.shape[1] < t_full:
                v = F.pad(v, (0, t_full - v.shape[1]), value=1.0)
        f0 = (torch.expm1(torch.clamp(
            f0_norm, min=0.0) * PITCH_LOG_SCALE) * v)

        audio = self.dec(g, yq, f0)
        return audio

    def infer_pitch(self, f0, sid, predict_voicing=False):
        """Encode/decode a raw f0 contour through the pitch codec only.

        Returns (f0_hat_hz, pq_ds, px_ds) where f0_hat_hz is at the input
        frame rate and codes are at 1/down_factor of it. With
        ``predict_voicing=True`` the codec's own V/UV head (rather than the
        input f0 mask) decides which frames are rendered.
        """
        self.eval()
        with torch.no_grad():
            g = self.spk_emb(sid)
            t_pad = (-f0.shape[-1]) % self.down_factor
            (f0_hat_norm, pq_ds, px_ds, _, _, _,
                uv_logits) = self._pitch_codec_forward(f0, t_pad, g)
            if predict_voicing:
                voiced = (torch.sigmoid(uv_logits)
                          >= self.pitch_uv_threshold).float()
            else:
                voiced = (f0 > 0).float()
            f0_hat = (torch.expm1(torch.clamp(
                f0_hat_norm, min=0.0) * PITCH_LOG_SCALE) * voiced)
        return f0_hat, pq_ds, px_ds


if __name__ == "__main__":
    from omegaconf import OmegaConf
    spec = torch.randn(1, 769, 356) # Apparently the model expects the channel to be first for spec
    sid = torch.randint(0, 5, (1,))
    spec_l = torch.tensor([356])
    f0 = torch.randn(1, 356)

    hp = OmegaConf.load("configs/base.yaml")
    decoder = Generator(hp)

    print(decoder.infer(spec, spec_l, f0, sid).shape)
