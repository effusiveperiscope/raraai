from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modeling.commons import DepthwiseSeparableConv1d, AttentionPooling
import math

class SpeakerClassifierCNN(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.in_proj = nn.Linear(config.model.content_size, config.model.disc_channels)
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=7, padding=3, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=5, padding=2, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=3, padding=1, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=3, padding=1, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=3, padding=1, spectral_norm=True),
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, config.model.disc_channels) for _ in range(len(self.convs))
        ])

        self.pool = AttentionPooling(config.model.disc_channels)
        self.out_proj = nn.Linear(config.model.disc_channels, config.model.spk_embed_dim)

    def forward(self, x, x_mask):
        x = self.in_proj(x)
        x = rearrange(x, "b t c -> b c t")
        for i, conv in enumerate(self.convs):
            xs = x

            x = F.silu(x)
            x = self.norms[i](x)
            x = conv(x) * x_mask.unsqueeze(1)

            x = x + xs

        x = rearrange(x, "b c t -> b t c")
        x = self.pool(x, x_mask)
        x = self.out_proj(x)
        return x

class SpeakerClassifier(nn.Module):
    def __init__(self, 
        inter_channels,
        num_speakers,
        n_layers=4):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=inter_channels,
                nhead=8,
                dim_feedforward=768,
                activation=F.silu
            ),
            num_layers=n_layers
        )
        self.out_proj = nn.Linear(inter_channels, num_speakers)
        self.sipe = SinusoidalPositionalEncoding(inter_channels)
        self.n_layers = n_layers

    def forward(self, x, x_mask):
        """
        x: Tensor of shape [batch_size, seq_len, inter_channels]
        x_mask: Tensor of shape [batch_size, seq_len], True for valid positions
        """

        # Transformer expects [seq_len, batch_size, features]
        x = rearrange(x, "b t c -> t b c")
        x = self.sipe(x)

        x = self.encoder(x, src_key_padding_mask=~x_mask)  # [seq_len, batch_size, inter_channels]

        # Mean pooling across sequence dimension (temporal)
        x_mask = rearrange(x_mask, "b t -> t b")
        x = ((x * x_mask.unsqueeze(-1)).sum(dim=0) /
             x_mask.sum(dim=0).clamp(min=1.0).unsqueeze(-1)) # avoid zero division

        x = self.out_proj(x)  # [batch_size, spk_embed_dim]
        return x

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