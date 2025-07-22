import torch.nn as nn
import torch
from omegaconf import OmegaConf
from svc_helper.svc.rvc.lib.infer_pack import attentions, commons, modules
from einops import rearrange
import math
from modeling.grl import grad_reverse

from modeling.commons import DepthwiseSeparableConv1d, SinusoidalPositionalEncoding

class PitchConditioner(nn.Module):
    def __init__(self, inter_channels):
        super().__init__()
        self.pitch_uv_emb = nn.Parameter(torch.randn(inter_channels))
        self.pitch_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, inter_channels)
        )

    def forward(self, pitch):
        mel_pitch = 1127 * torch.log1p(pitch / 700)
        mel_pitch = mel_pitch.unsqueeze(-1)

        voiced_mask = (pitch > 0).to(pitch.dtype).unsqueeze(-1)

        pitch_feat = self.pitch_proj(mel_pitch) * voiced_mask
        pitch_feat += (1 - voiced_mask) * self.pitch_uv_emb

        return pitch_feat

class V10Encoder(nn.Module):
    def __init__(
        self,
        config: OmegaConf,
        f0=True,
    ):
        super(V10Encoder, self).__init__()
        self.out_channels = config.out_channels
        self.hidden_channels = config.hidden_channels
        self.filter_channels = config.filter_channels
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.kernel_size = config.kernel_size
        self.p_dropout = float(config.p_dropout)
        self.emb_phone = nn.Linear(512, config.hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = PitchConditioner(config.hidden_channels)
        self.sipe = SinusoidalPositionalEncoding(config.model.inter_channels)
        self.encoder = attentions.Encoder(
            config.hidden_channels,
            config.filter_channels,
            config.n_heads,
            config.n_layers,
            config.kernel_size,
            float(config.p_dropout),
        )
        self.proj = nn.Conv1d(config.hidden_channels, config.out_channels * 2, 1)
        self.speaker_encoder = SpeakerEncoder(config, n_layers=6) # so-vits-svc 5.0 disentanglement objective

    def forward(self, phone: torch.Tensor, pitchf: torch.Tensor, lengths: torch.Tensor,
        lam_grl = 1.0):
        if pitchf is None:
            x_phone = self.emb_phone(phone)
            x_phone = x_phone / (x_phone.std(dim=-1, keepdim=True) + 1e-6)
            x = x_phone
        else:
            x_phone = self.emb_phone(phone)
            x_pitch = self.emb_pitch(pitchf)
            x_phone = x_phone / (x_phone.std(dim=-1, keepdim=True) + 1e-6)
            x_pitch = x_pitch / (x_pitch.std(dim=-1, keepdim=True) + 1e-6)

            x = x_phone + x_pitch
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)

        x = self.sipe(x)

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask) 
        pre_proj_x = x
        spk_feat_pred = self.speaker_encoder(grad_reverse(x, lam_grl))

        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask, spk_feat_pred, pre_proj_x

class SpeakerEncoder(nn.Module):
    def __init__(self, config, n_layers=2):
        super(SpeakerEncoder, self).__init__()
        embed_dim = config.model.content_channels
        spk_dim = config.model.spk_emb_channels

        self.encoder = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(
                embed_dim, embed_dim, kernel_size=5, padding=2, spectral_norm=True) for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(n_layers)]
        )
        self.final_proj = DepthwiseSeparableConv1d(
            embed_dim, spk_dim, kernel_size=3, padding=1
        )

    def forward(self, x, mask):
        x = rearrange(x, "b t c -> b c t")
        for i,layer in enumerate(self.encoder):
            x = layer(x)
            x = x * rearrange(mask, "b t -> b 1 t")

            x = rearrange(x, "b c t -> b t c")
            x = self.norms[i](x)
            x = rearrange(x, "b t c -> b c t")

            x = F.silu(x)
        outputs = self.final_proj(x)
        outputs = torch.mean(outputs, dim=-1)
        return outputs