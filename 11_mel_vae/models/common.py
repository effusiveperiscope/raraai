import torch
import torch.nn as nn
import math
import torch.nn.utils.spectral_norm as sn


# --- FiLM Layer ---
class FiLMGenerator(nn.Module):
    """ Generates FiLM scale and shift parameters """
    def __init__(self, condition_dim, target_dim):
        super().__init__()
        # Project condition_dim to 2*target_dim (for scale and shift)
        self.projection = nn.Linear(condition_dim, target_dim * 2)

        nn.init.constant_(self.projection.bias[0:target_dim], 1.0)  # gamma bias
        nn.init.constant_(self.projection.bias[target_dim:], 0.0)   # beta bias

    def forward(self, condition):
        """
        Args:
            condition: (batch_size, seq_len, condition_dim)
        Returns:
            gamma: (batch_size, seq_len, target_dim) - Scale
            beta: (batch_size, seq_len, target_dim) - Shift
        """
        params = self.projection(condition)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableConv1dTransposed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.depthwise = nn.ConvTranspose1d(in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution Layer with optional Spectral Normalization.

    Applies depthwise convolution followed by pointwise convolution.
    Spectral normalization can be applied to both internal convolution layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, use_spectral_norm=True):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm

        # Depthwise convolution: applies spatial filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels, # Output channels = Input channels
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels, # Key for depthwise conv
            bias=bias
        )

        # Pointwise convolution: 1x1 conv to mix channels and change dimension
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        # Apply spectral normalization if requested
        if self.use_spectral_norm:
            self.depthwise = sn(self.depthwise)
            self.pointwise = sn(self.pointwise)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
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