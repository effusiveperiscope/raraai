import torch
import torch.nn as nn

class FeedForwardModule(nn.Module):
    """Feed Forward Network with residual scaling as used in Conformer."""
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        self.residual_scale = 0.5  # Scaling factor for FFN residual connection

    def forward(self, x):
        return x + self.residual_scale * self.layer(x)

class MultiHeadSelfAttention(nn.Module):
    """Standard Transformer Multi-Head Self-Attention with LayerNorm."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x = self.norm(x)
        attn_output, _ = self.attn(x, x, x)
        return x + attn_output  # Residual connection

class ConvolutionModule(nn.Module):
    """Depthwise Separable Convolution module for local feature extraction."""
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim * 2, kernel_size=1),  # Pointwise conv
            nn.GLU(dim=1),  # Gated Linear Unit
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2),  # Depthwise conv
            nn.BatchNorm1d(dim),
            nn.SiLU(),  # Swish activation
            nn.Conv1d(dim, dim, kernel_size=1),  # Another pointwise conv
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Reshape for Conv1d: (batch, seq_len, dim) → (batch, dim, seq_len)
        x = x.transpose(1, 2)
        x = self.layer(x)
        x = x.transpose(1, 2)  # Restore shape
        return x

class ConformerBlock(nn.Module):
    """A single Conformer Block combining FFN, MHSA, and CNN modules."""
    def __init__(self, dim, num_heads=8, ffn_expansion=4, conv_kernel=5, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.norm(x)  # Final normalization