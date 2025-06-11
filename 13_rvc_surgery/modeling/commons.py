import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, x_mask):
        attn_scores = self.attn(x).squeeze(-1)  # (B, T)
        attn_scores = attn_scores.masked_fill(x_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T)
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