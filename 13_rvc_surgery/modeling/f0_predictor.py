from omegaconf import OmegaConf
from torch import nn
from modeling.spk_cond import FiLMGenerator
from modeling.commons import SinusoidalPositionalEncoding
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

class F0DeltaPredictor(nn.Module):
    def __init__(self, 
        speech_dim: int,
        pitch_dim: int,
        spk_emb_dim: int,
        hidden_dim: int,
        dropout: float = 0.1):
        super().__init__()

        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.pitch_emb = nn.Embedding(pitch_dim, hidden_dim)
        self.speech_cond = FiLMGenerator(hidden_dim, hidden_dim)
        self.spk_cond = FiLMGenerator(spk_emb_dim, hidden_dim)

        self.speech_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)])
        self.inter_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)])
        self.final_conv = SiLUResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, 1, 1, padding=0)])

    def forward(self, quant_pitch, speech, speech_mask,
        spk_emb):

        x = self.pitch_emb(quant_pitch) * speech_mask

        speech = self.speech_proj(speech) * speech_mask
        speech = self.speech_conv(speech) * speech_mask

        gamma, beta = self.spk_cond(spk_emb.unsqueeze(1))
        x = gamma * x + beta

        X = self.inter_conv(x)

        gamma, beta = self.speech_cond(speech)
        x = gamma * x + beta

        return x