from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class FeedForward(nn.Module):
    """
    Feed Forward module for Conformer block.
    """
    def __init__(self, d_model, expansion_factor=4, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout_p),
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention module using PyTorch's MultiheadAttention.
    """
    def __init__(self, d_model, num_heads=8, dropout_p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True
        )
    
    def forward(self, x):
        x_norm = self.norm(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        return attn_output

class ConvModule(nn.Module):
    """
    Convolutional module for Conformer block.
    """
    def __init__(self, d_model, kernel_size=31, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        inner_dim = d_model * expansion_factor
        padding = (kernel_size - 1) // 2
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            inner_dim // 2, 
            inner_dim // 2, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=inner_dim // 2
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim // 2)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(inner_dim // 2, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        # x: [B, T, d_model]
        x = self.layer_norm(x)  # Apply LayerNorm in the original dimension [B, T, d_model]
        
        # Convert to channel-first for convolutions
        x = x.transpose(1, 2)  # [B, d_model, T]
        
        x = self.pointwise_conv1(x)  # [B, inner_dim, T]
        x = self.glu(x)  # [B, inner_dim//2, T]
        x = self.depthwise_conv(x)  # [B, inner_dim//2, T]
        x = self.batch_norm(x)  # [B, inner_dim//2, T]
        x = self.activation(x)  # [B, inner_dim//2, T]
        x = self.pointwise_conv2(x)  # [B, d_model, T]
        x = self.dropout(x)  # [B, d_model, T]
        
        # Convert back to sequence-first
        x = x.transpose(1, 2)  # [B, T, d_model]
        
        return x


class ConformerBlock(nn.Module):
    """
    Conformer Block combining Feed Forward modules, Multi-Head Self-Attention and Convolution module.
    """
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        ff_expansion_factor=4,
        conv_expansion_factor=2,
        ff_dropout_p=0.1,
        conv_dropout_p=0.1,
        mha_dropout_p=0.1,
        conv_kernel_size=31
    ):
        super().__init__()
        
        self.ff_module1 = FeedForward(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout_p=ff_dropout_p
        )
        
        self.mha_module = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_p=mha_dropout_p
        )
        
        self.conv_module = ConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p
        )
        
        self.ff_module2 = FeedForward(
            d_model=d_model,
            expansion_factor=ff_expansion_factor,
            dropout_p=ff_dropout_p
        )
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of the transformer components."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model]
        
        Returns:
            torch.Tensor: Output tensor of shape [B, T, d_model]
        """
        # First Feed Forward module (1/2 scaling)
        x = x + 0.5 * self.ff_module1(x)
        
        # Multi-Head Self-Attention module
        x = x + self.mha_module(x)
        
        # Convolution module
        x = x + self.conv_module(x)
        
        # Second Feed Forward module (1/2 scaling)
        x = x + 0.5 * self.ff_module2(x)
        
        # Final Layer Normalization
        x = self.final_layer_norm(x)
        
        return x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1e6:
        return f"{total_params} params"  # Parameters
    elif total_params < 1e9:
        return f"{total_params / 1e6:.2f} M"  # Millions
    else:
        return f"{total_params / 1e9:.2f} B"  # Billions

class SpeechFeatureSummarizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            self.block(hidden_dim, dropout, kernel_size=11),
            self.block(hidden_dim, dropout, kernel_size=7),
            self.block(hidden_dim, dropout, kernel_size=5),
            self.block(hidden_dim, dropout, kernel_size=3),
        ])
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(4)
        ])
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activ = nn.ReLU()

    def block(self, hidden_dim: int, dropout: float = 0.1, kernel_size: int = 3,):
        return nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, feature_dim)
        """
        x = self.in_proj(x)

        x = x.transpose(1, 2)  # (batch_size, feature_dim, seq_len)
        for i, conv in enumerate(self.convs):
            x = x + conv(x)

            x = x.transpose(1, 2)
            x = self.fcs[i](x)
            x = self.norm(x)
            x = self.activ(x)
            x = x.transpose(1, 2)

        x = x.transpose(1, 2)

        x = torch.mean(x, dim=1)  # Average pooling over time
        x = self.out_proj(x)
        return x  # Output shape: (batch_size, output_dim)

class TokenConvertModel(nn.Module):
    def __init__(self, 
        config: OmegaConf):
        super(TokenConvertModel, self).__init__()
        self.in_proj = nn.Linear(config.in_dim, config.hidden_dim)
        self.summarizer = SpeechFeatureSummarizer(config.output_dim, config.summary_dim, config.hidden_dim)
        self.conformers = nn.ModuleList([
            ConformerBlock(
                d_model=config.hidden_dim,
                num_heads=8,
                ff_expansion_factor=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                ff_dropout_p=config.dropout_p,
                conv_dropout_p=config.dropout_p,
                mha_dropout_p=config.dropout_p
            ) for _ in range(config.n_layers)
        ])
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)
        self.spk_emb = nn.Embedding(config.n_spk, config.hidden_dim)

    def summarize(self, x):
        return self.summarizer(x)

    def forward(self, tok, tok_mask, sid=None, summary=None):
        x = self.in_proj(tok)
        tok_mask = tok_mask.unsqueeze(2)
        x = x * tok_mask

        if summary is not None:
            x = x + summary.unsqueeze(1)

        if sid is not None:
            x = x + self.spk_emb(sid).unsqueeze(1)

        for conformer in self.conformers:
            x = conformer(x)
            x = x * tok_mask

        x = self.fc(x)
        x = x * tok_mask
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 4
    seq_length = 50
    
    x = torch.rand(batch_size, seq_length, 1024)
    
    model = TokenConvertModel()
    
    output = model(x) 
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print(f"Number of parameters: {count_parameters(model)}")