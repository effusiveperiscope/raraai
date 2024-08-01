import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from modules import PositionalEncoding

# Basic transformer-based model
class SpeechClassifier1(nn.Module):
    def __init__(self, hidden_dim, n_speakers):
        super().__init__()

        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim, dropout=0.4)
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=128, dropout=0.4
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers, num_layers=5
        )
        self.decoder = nn.Linear(hidden_dim, n_speakers)
        self.lrelu = nn.LeakyReLU()

    def forward(self,
        speech_features, # [N x B x C]
        ):
        # Perturbation (from so-vits-svc 5.0)
        # Average std of a speech feature is 0.33
        speech_features = speech_features + torch.randn_like(speech_features) * 0.03
        x = self.positional_encoding(speech_features)
        x = self.transformer_encoder(x)
        logits = self.lrelu(self.decoder(x))
        return logits.mean(0) # [B x C]

# Extremely dumb model for testing
class SpeechClassifier0(nn.Module):
    def __init__(self, embedding_dim, n_speakers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, n_speakers),
            nn.LeakyReLU(),
        )

    def forward(self,
        speech_features, # [N x B x C]
        ):
        logits = self.encoder(speech_features)
        # Obviously we don't expect this to work too well
        # Since each logits item is essentially independent
        return logits.mean(0) # [B x C]