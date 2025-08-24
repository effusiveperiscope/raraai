from omegaconf import OmegaConf
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from .commons import PitchConditioner

class SiLUResBlock(nn.Module):
    def __init__(self, convs : list):
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.GroupNorm(1, c.out_channels) for c in self.convs])

    def forward(self, x, channels_last=True):
        if channels_last:
            x = rearrange(x, "b t c -> b c t")

        for i,conv in enumerate(self.convs):
            xs = x
            x = F.silu(x)
            x = conv(x)
            x = self.norms[i](x)
            x = x + xs

        if channels_last:
            x = rearrange(x, "b c t -> b t c")
        return x

class F0PredictorSmall(nn.Module): # No special conditioning
    """
    Simple F0 predictor. Training objective should be log f0 + 1
    """
    def __init__(self, 
        speech_dim: int,
        pitch_quant_dim: int,
        hidden_dim: int):
        super().__init__()

        self.speech_proj = nn.Sequential( # Bottleneck to avoid leaking too much info
            nn.Linear(speech_dim, 32),
            nn.Linear(32, hidden_dim),
            )
        self.mean_proj = nn.Linear(1, hidden_dim)
        self.pitch_cond = PitchConditioner(hidden_dim)
        self.convs = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            ])
        self.final_proj = nn.Linear(hidden_dim, 1)

    def forward(self, quant_pitch, target_f0_mean, speech_mask):
        speech_mask = speech_mask.unsqueeze(-1)
        v_mask = (quant_pitch != 0).unsqueeze(-1).float()
        x = self.pitch_cond(quant_pitch,
            convert_mel=False, use_dtype=target_f0_mean.dtype) * speech_mask * v_mask
        x = x + self.mean_proj(target_f0_mean.unsqueeze(-1)).unsqueeze(1) * speech_mask
        x = self.convs(x) * speech_mask * v_mask
        x = F.layer_norm(x, x.shape[1:])
        x = F.silu(x)
        x = self.final_proj(x) * speech_mask  * v_mask
        return x # we predict log pitch plus 1

from torch.nn.utils.parametrizations import spectral_norm
class F0Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(1, 64, 7, padding=3)),
            spectral_norm(nn.Conv1d(64, 128, 3, padding=1)),
            spectral_norm(nn.Conv1d(128, 256, 3, padding=1)),
            spectral_norm(nn.Conv1d(256, 256, 3, padding=1)),
            spectral_norm(nn.Conv1d(256, 128, 3, padding=1)),
            spectral_norm(nn.Conv1d(128, 64, 3, padding=1)),
            spectral_norm(nn.Conv1d(64, 1, 3, padding=1)),
        ])
        self.res_layers = nn.ModuleList([
            nn.Conv1d(1, 64, 1),
            nn.Conv1d(64, 128, 1),
            nn.Conv1d(128, 256, 1),
            nn.Conv1d(256, 256, 1),
            nn.Conv1d(256, 128, 1),
            nn.Conv1d(128, 64, 1),
            nn.Identity()
        ])
        self.norms = nn.ModuleList()
        for c in self.convs:
            if c.out_channels > 1:
                self.norms.append(nn.GroupNorm(1, c.out_channels))
            else:
                self.norms.append(nn.Identity())  # no norm for last conv (degenerate case)

        self.final_proj = nn.Linear(1, 1)

    def forward(self, pit):
        pit = rearrange(pit, "b t c -> b c t")
        vuv = (pit != 0).float()
        for i, x in enumerate(zip(self.convs, self.res_layers, self.norms)):
            conv, res, norms = x
            xs = pit
            pit = conv(pit) * vuv
            if i < len(self.convs) - 1:
                pit = res(xs) + pit
            pit = F.silu(pit)
            pit = norms(pit)
        pit = rearrange(pit, "b c t -> b t c")
        pit = self.final_proj(pit)
        return pit


class F0PredictorLarge(nn.Module):
    def __init__(self, 
        speech_dim: int,
        pitch_quant_dim: int,
        spk_emb_dim: int,
        hidden_dim: int):
        super().__init__()

        self.speech_proj = nn.Sequential( # Bottleneck to avoid leaking too much info
            nn.Linear(speech_dim, 32),
            nn.Linear(32, hidden_dim),
            )
        self.mean_proj = nn.Linear(1, hidden_dim)
        self.pitch_emb = nn.Embedding(pitch_quant_dim + 1, hidden_dim) # +1 for unvoiced
        self.speech_cond = nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)
        self.spk_cond = nn.Conv1d(spk_emb_dim, 2 * hidden_dim, 1)

        self.speech_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)])
        self.inter_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)])
        self.final_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, 1, 1, padding=0)])

    def forward(self, 
        quant_pitch, 
        target_f0_mean, 
        speech, 
        speech_mask,
        spk_emb):

        x = self.pitch_emb(quant_pitch) * speech_mask
        x = x + self.mean_proj(target_f0_mean) * speech_mask

        speech = self.speech_proj(speech) * speech_mask

        speech = self.speech_conv(rearrange(speech, "b t c -> b c t")) 
        speech = speech * rearrange(speech_mask, "b t -> b 1 t")

        x = rearrange(x, "b t c -> b c t")

        m, v = self.speech_cond(speech)
        x = (x - m) * torch.exp(-v) * rearrange(speech_mask, "b t -> b 1 t")

        x = self.inter_conv(x)

        m, v = self.spk_cond(spk_emb.unsqueeze(1))
        x = (x - m) * torch.exp(-v) * rearrange(speech_mask, "b t -> b 1 t")

        x = self.final_conv(x)
        x = rearrange(x, "b c t -> b t c")
        x = F.relu(x)

        return x