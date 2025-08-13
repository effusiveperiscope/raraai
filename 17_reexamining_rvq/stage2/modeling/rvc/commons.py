import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class PitchConditioner(nn.Module):
    """Conditioning module that treats 0 as a special embedding, intended for
    use with f0 contours (i.e. where 0 = unvoiced)"""
    def __init__(self, inter_channels):
        super().__init__()
        self.pitch_uv_emb = nn.Parameter(torch.randn(inter_channels))
        self.pitch_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, inter_channels),
            nn.SiLU(),
            nn.Linear(inter_channels, inter_channels),
        )

    def forward(self, pitch, convert_mel=True):
        if pitch.dtype == torch.long: # Handle quantized condition
            pitch = pitch.float()

        if convert_mel:
            mel_pitch = 1127 * torch.log1p(pitch / 700)
            mel_pitch = mel_pitch.unsqueeze(-1)
        else:
            mel_pitch = pitch.unsqueeze(-1)

        voiced_mask = (pitch > 0).to(pitch.dtype).unsqueeze(-1)

        pitch_feat = self.pitch_proj(mel_pitch) * voiced_mask
        pitch_feat += (1 - voiced_mask) * self.pitch_uv_emb

        pitch_feat = F.layer_norm(pitch_feat, pitch_feat.shape[-1:])

        return pitch_feat

class FiLMGenerator(nn.Module):
    """ Generates FiLM scale and shift parameters """
    def __init__(self, condition_dim, target_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, target_dim * 2),
        )

        nn.init.constant_(self.mlp[-1].bias[0:target_dim], 1.0)  # gamma bias
        nn.init.constant_(self.mlp[-1].bias[target_dim:], 0.0)   # beta bias

    def forward(self, condition):
        """
        Args:
            condition: (batch_size, seq_len, condition_dim)
        Returns:
            gamma: (batch_size, seq_len, target_dim) - Scale
            beta: (batch_size, seq_len, target_dim) - Shift
        """
        params = self.mlp(condition)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            DepthwiseSeparableConv1d(
                input_dim, input_dim, 
                kernel_size=3, padding=1,
                spectral_norm=True),
            nn.SiLU(),
            DepthwiseSeparableConv1d(
                input_dim, input_dim // 2, 
                kernel_size=3, padding=1,
                spectral_norm=True),
            nn.SiLU(),
            DepthwiseSeparableConv1d(
                input_dim // 2, 1, 
                kernel_size=3, padding=1,
                spectral_norm=True),
            nn.SiLU(),
        )

    def forward(self, x, x_mask):
        x = rearrange(x, "b t c -> b c t")
        attn_scores = self.attn(x).squeeze(-1)  # (B, T)
        x = rearrange(x, "b c t -> b t c")

        # Check if any sequence has all tokens masked
        valid_tokens = x_mask.sum(dim=-1)  # (B,)
        if (valid_tokens == 0).any():
            print("Warning: Found sequences with no valid tokens!")

        attn_scores = attn_scores.float().masked_fill(x_mask == 0, -1e5)
        attn_weights = F.softmax(attn_scores, dim=-1).to(x.dtype)  # (B, T)
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, C)
        return pooled

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=10000):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            dim: Embedding dimension size (d_model)
            max_len: Maximum sequence length to pre-compute positions for
        """
        super().__init__()
        
        # Create empty tensor to store positional encodings
        # Shape: [max_len, dim]
        pe = torch.zeros(max_len, dim)
        
        # Create position indices tensor
        # Shape: [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Create division terms for angle calculations
        # Shape: [dim/2]
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        
        # Calculate sin component for even indices
        # pe[:, 0::2] shape: [max_len, dim/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Calculate cos component for odd indices
        # pe[:, 1::2] shape: [max_len, dim/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        # Add batch dimension
        # Final shape: [1, max_len, dim]
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # self.pe[:, :x.size(1)] slices the PE to match input sequence length
        # Returns: [batch_size, seq_len, dim]
        return x + self.pe[:, :x.size(1)]

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if spectral_norm:
            self.depthwise = nn.utils.spectral_norm(nn.Conv1d(in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, padding_mode='reflect'))
            self.pointwise = nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        else:
            self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, padding_mode='reflect')
            self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x