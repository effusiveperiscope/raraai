import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from modules import PositionalEncoding

# Transformer encoder model with embedding
class SpeechClassifier1(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, n_speakers):
        super().__init__()

        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim, dropout=0.4)
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=256, dropout=0.4
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers, num_layers=4
        )
        self.n_speakers = n_speakers
        self.decoder = nn.Linear(hidden_dim, embedding_dim)
        self.batchnorm = nn.BatchNorm1d(embedding_dim)
        self.lrelu = nn.LeakyReLU()
        self.embedding = nn.Embedding(
            num_embeddings=n_speakers,
            embedding_dim=embedding_dim)

    def forward(self,
        speech_features, # [N x B x C]
        gt_label # [B x L]
        ):
        x = self.positional_encoding(speech_features)
        x = self.transformer_encoder(x) # [N x B x C]
        x = self.decoder(x) # [N x B x embedding_dim]

        x = rearrange(x, 'n b e -> b e n')
        x = self.batchnorm(x)
        x = rearrange(x, 'b e n -> n b e')

        pred_emb = self.lrelu(x).mean(0) # [B x C]
        tgt_emb = self.embedding(gt_label)

        return pred_emb, tgt_emb

    def infer(self, speech_features):
        x = self.positional_encoding(speech_features)
        x = self.transformer_encoder(x)
        pred_emb = self.decoder(x)
        return pred_emb

    """Returns argmax of nMSE (most confident prediction index) and nMSE"""
    def pseudo_logits(self, pred_batch): # [batch x emb_dim]
        with torch.no_grad():
            logits = []

            for pred_emb in pred_batch: # [1 x emb_dim]
                # embedding.weight = [n_speaker x emb_dim]
                neg_mse = (-torch.mean(
                    (self.embedding.weight - pred_emb) ** 2, dim=1)) # [n_speaker]
                logits.append(neg_mse)
            logits = torch.stack(tuple(logits)) # [batch x n_speaker]

            return logits.argmax(dim=1), logits

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