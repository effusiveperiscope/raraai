from omegaconf import OmegaConf
from torch import nn
from modeling.spk_cond import FiLMGenerator
from modeling.commons import SinusoidalPositionalEncoding
import torch.nn.functional as F

class F0DeltaPredictor(nn.Module):
    def __init__(self, 
        speech_dim: int,
        pitch_dim: int,
        spk_emb_dim: int,
        hidden_dim: int,
        n_layers_spk: int = 2,
        n_layers_speech: int = 2,
        dropout: float = 0.1):
        super().__init__()

        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.pitch_emb = nn.Embedding(pitch_dim, hidden_dim)
        self.speech_cond = FiLMGenerator(hidden_dim, hidden_dim)
        self.spk_cond = FiLMGenerator(spk_emb_dim, hidden_dim)
        self.sipe = SinusoidalPositionalEncoding(hidden_dim)

        self.spk_encode = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=768,
                activation=F.silu,
                dropout=dropout
            ),
            num_layers=n_layers_spk
        )
        self.speech_encode = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=768,
                activation=F.silu,
                dropout=dropout
            ),
            num_layers=n_layers_speech
        )

    def forward(self, quant_pitch, speech, speech_mask,
        spk_emb):

        x = self.pitch_emb(quant_pitch) * speech_mask
        x = self.sipe(quant_pitch)

        speech = self.speech_proj(speech) * speech_mask

        gamma, beta = self.spk_cond(spk_emb.unsqueeze(1))
        x = gamma * x + beta

        x = self.spk_encode(x, src_key_padding_mask=~speech_mask)

        gamma, beta = self.speech_cond(speech)
        x = gamma * x + beta

        x = self.speech_encode(x, src_key_padding_mask=~speech_mask)
        return x