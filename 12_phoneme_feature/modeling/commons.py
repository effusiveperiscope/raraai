import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding implementation."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the rotation matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device, q.dtype)
        
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]
        
        # Apply RoPE
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RoPEEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(out)


class ConvolutionModule(nn.Module):
    """Convolution module from Conformer."""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = self.layer_norm(x)
        
        # Transpose for conv1d: (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        
        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise conv + BatchNorm + Swish
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        
        # Final pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: (batch, seq_len, d_model)
        return x.transpose(1, 2)


class FeedForwardModule(nn.Module):
    """Feed-forward module from Conformer."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.self_attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ff2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First FF module (half residual)
        x = x + 0.5 * self.dropout(self.ff1(x))
        
        # Multi-head self-attention
        x = x + self.dropout(self.self_attn(self.norm_attn(x), attn_mask))
        
        # Convolution module
        x = x + self.dropout(self.conv(x))
        
        # Second FF module (half residual)
        x = x + 0.5 * self.dropout(self.ff2(x))
        
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer encoder with RoPE, similar to PyTorch TransformerEncoder API.
    
    Args:
        encoder_layer: A single ConformerBlock instance
        num_layers: Number of encoder layers
        norm: Optional layer normalization
    """
    
    def __init__(
        self, 
        encoder_layer: ConformerBlock, 
        num_layers: int, 
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            type(encoder_layer)(
                encoder_layer.self_attn.d_model,
                encoder_layer.self_attn.num_heads,
                encoder_layer.ff1.linear2.out_features,
                encoder_layer.conv.depthwise_conv.kernel_size[0],
                encoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self, 
        src: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask (not typically used in Conformer)
            src_key_padding_mask: Key padding mask for variable length sequences
        
        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model)
        """
        output = src
        
        # Convert key padding mask to attention mask format if provided
        attn_mask = None
        if src_key_padding_mask is not None:
            # src_key_padding_mask: (batch_size, seq_len) where True means padding
            # Convert to attention mask format for multi-head attention
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = ~attn_mask  # Invert: False means padding, True means valid
        
        for layer in self.layers:
            output = layer(output, attn_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


# Helper function to create a Conformer encoder (similar to PyTorch's API)
def create_conformer_encoder(
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    conv_kernel_size: int = 31,
    dropout: float = 0.1,
    norm_first: bool = False
) -> ConformerEncoder:
    """
    Create a Conformer encoder with the specified parameters.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of encoder layers
        d_ff: Feed-forward dimension
        conv_kernel_size: Convolution kernel size
        dropout: Dropout rate
        norm_first: Whether to apply layer norm first (not used in standard Conformer)
    
    Returns:
        ConformerEncoder instance
    """
    encoder_layer = ConformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        conv_kernel_size=conv_kernel_size,
        dropout=dropout
    )
    
    norm = nn.LayerNorm(d_model)
    
    return ConformerEncoder(encoder_layer, num_layers, norm)

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