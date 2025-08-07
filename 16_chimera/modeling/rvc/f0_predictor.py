from omegaconf import OmegaConf
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange

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

class F0Predictor(nn.Module):
    def __init__(self, 
        speech_dim: int,
        pitch_quant_dim: int,
        spk_emb_dim: int,
        hidden_dim: int,
        dropout: float = 0.1):
        super().__init__()

        self.speech_proj = nn.Sequential( # Bottleneck to avoid leaking too much info
            nn.Linear(speech_dim, 32),
            nn.Linear(32, hidden_dim),
            )
        self.mean_proj = nn.Linear(1, hidden_dim)
        self.pitch_emb = nn.Embedding(pitch_quant_dim + 1, hidden_dim) # +1 for unvoiced
        self.speech_cond = nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)

        self.speech_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)])
        self.inter_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)])
        self.final_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=3),
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
        x = self.final_conv(x)
        x = rearrange(x, "b c t -> b t c")

        return x