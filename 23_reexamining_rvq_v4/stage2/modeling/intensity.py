import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ChannelMiddleLayerNorm(nn.Module):
    def __init__(self, hidden_channels=96): 
        super().__init__()
        self.ln = nn.LayerNorm(hidden_channels, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c t -> b t c')
        x = self.ln(x)
        x = rearrange(x, 'b t c -> b c t')
        return x

class Block(nn.Module):
    def __init__(self, hidden_channels=96, dropout=0.3): 
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3,
                groups=hidden_channels), # depthwise convolution
            ChannelMiddleLayerNorm(hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels*4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels*4, hidden_channels, kernel_size=1),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)

class IntensityModel(nn.Module):
    def __init__(self,
            in_channels, hidden_dim=128,
            output_channels=1, dropout=0.3, bottleneck_dim=8):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.stages = nn.ModuleList([
            Block(hidden_dim, dropout),
            Block(hidden_dim, dropout),
            Block(hidden_dim, dropout),
        ])
        self.feature_bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.intensity_proj = nn.Linear(bottleneck_dim, output_channels)
        self.attention_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, x_mask, return_feat=False):
        x = self.in_proj(x) * (x_mask.unsqueeze(-1))
        x = rearrange(x, 'b t c -> b c t')
        for i, stage in enumerate(self.stages):
            if i == 0:
                x = (stage(x)) * (x_mask.unsqueeze(1))
            else:
                x = (stage(x) + x) * (x_mask.unsqueeze(1))
        x = rearrange(x, 'b c t -> b t c')

        feat = self.feature_bottleneck(x)
        intensity = self.intensity_proj(feat)

        attn = self.attention_proj(x)
        attn = attn.masked_fill((x_mask.unsqueeze(-1)) == 0, float('-inf'))
        attn_weights = F.softmax(attn, dim=1)
        if not return_feat:
            return intensity, attn_weights
        else:
            return intensity, attn_weights, feat