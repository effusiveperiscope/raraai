import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create cached cos and sin values for rotary embeddings
        # We use dim/2 because each dimension pair shares the same angle
        self.freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.freq = self.freq.unsqueeze(0)  # [1, dim/2]
        
        # Initialize cache for position indices
        self.register_buffer("positions", torch.arange(0, max_seq_len).float().unsqueeze(1))  # [max_seq_len, 1]
        
        # Compute cos and sin values: [max_seq_len, dim/2]
        cos_cached = torch.cos(self.positions * self.freq)
        sin_cached = torch.sin(self.positions * self.freq)
        
        # Interleave to match pair structure: [max_seq_len, dim]
        cos_cached = torch.repeat_interleave(cos_cached, 2, dim=1)
        sin_cached = torch.repeat_interleave(sin_cached, 2, dim=1)
        
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
        
    def forward(self, x, past_len=0, seq_dim=1):
        """
        Apply rotary embeddings to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim] or [seq_len, batch_size, dim]
            seq_dim: Dimension corresponding to sequence length (default: 1)
            
        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_len = x.shape[seq_dim]
        
        # Get cos and sin values for this sequence length
        cos = self.cos_cached[past_len: past_len + seq_len]  # [seq_len, dim]
        sin = self.sin_cached[past_len: past_len + seq_len]  # [seq_len, dim]
        
        # Reshape depending on input shape
        if seq_dim == 0:
            # [seq_len, batch_size, dim]
            cos = cos.unsqueeze(1) 
            sin = sin.unsqueeze(1)
        else:
            # [batch_size, seq_len, dim]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Apply rotary embeddings
        # For each dimension pair (x_i, x_{i+1}), apply rotation:
        # [x_i, x_{i+1}] -> [x_i*cos - x_{i+1}*sin, x_i*sin + x_{i+1}*cos]
        
        # First, reshape x to separate even and odd dimensions
        x_shape = x.shape
        x_reshaped = x.reshape(*x_shape[:-1], -1, 2)
        
        # Get even and odd dimensions
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        
        # Reshape cos and sin to match
        cos_view = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
        sin_view = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]
        
        # Apply rotation
        x_rotated_even = x_even * cos_view - x_odd * sin_view
        x_rotated_odd = x_even * sin_view + x_odd * cos_view
        
        # Stack even and odd dimensions
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        
        # Reshape back to original shape
        x_rotated = x_rotated.reshape(*x_shape)
        
        return x_rotated