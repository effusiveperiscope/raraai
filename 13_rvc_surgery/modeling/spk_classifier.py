import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        self.n_layers = n_layers

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, inter_channels, seq_len]
        """

        # Transformer expects [seq_len, batch_size, features]
        x = rearrange(x, "b c t -> t b c")

        x = self.encoder(x)  # [seq_len, batch_size, inter_channels]

        # Mean pooling across sequence dimension (temporal)
        x = x.mean(dim=0)  # [batch_size, inter_channels]

        x = self.out_proj(x)  # [batch_size, spk_embed_dim]
        return x