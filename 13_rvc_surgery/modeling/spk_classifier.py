import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

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
        x: Tensor of shape [batch_size, inter_channels, seq_len]
        """

        # Transformer expects [seq_len, batch_size, features]
        x = rearrange(x, "b c t -> t b c")
        x = self.sipe(x)

        x = self.encoder(x, src_key_padding_mask=x_mask)  # [seq_len, batch_size, inter_channels]

        # Mean pooling across sequence dimension (temporal)
        x = x.mean(dim=0)  # [batch_size, inter_channels]

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