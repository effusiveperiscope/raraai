from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from modeling.grl import grad_reverse
from modeling.commons import SinusoidalPositionalEncoding
from omegaconf import OmegaConf
from einops import rearrange
import torch.nn.functional as F

import pdb
import sys
from traceback import format_exception
sys.excepthook = lambda exc_type, exc_value, exc_traceback: print(format_exception(exc_type, exc_value, exc_traceback)) or pdb.post_mortem(exc_traceback)

class _VAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(config.model.inter_channels,
            max_len=2000)
        self.phone_proj = nn.Linear(
            config.model.phonemes_count + 1,
            config.model.inter_channels
        )
        self.vae_proj = nn.Linear(
            config.model.vae_dim, 
            config.model.inter_channels
        )
        self.decoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.model.inter_channels, 
                nhead=config.model.decoder.n_head, 
                activation=F.silu,
                batch_first=True),
            num_layers=config.model.decoder.num_layers
        )
        self.out_proj = nn.Linear(
            config.model.inter_channels, 
            config.model.whisper_channels
        )
        self.spk_emb = nn.Embedding(
            config.model.n_spk, 
            config.model.inter_channels
        )

    def forward(self, phone_logits, z, x_mask, spk_id):
        phone_logits = self.phone_proj(phone_logits)
        x = self.vae_proj(z)
        x = phone_logits + x
        x = self.spk_emb(spk_id).unsqueeze(1) + x
        x = self.pos_enc(x)
        x = self.decoder(x, src_key_padding_mask=x_mask)
        x = self.out_proj(x)
        return x

class PASIFVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.whisper_proj = nn.Linear(
            config.model.whisper_channels, 
            config.model.inter_channels)
        self.base_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.model.inter_channels, 
                nhead=config.model.base_encoder.n_head, 
                activation=F.silu,
                batch_first=True),
            num_layers=config.model.base_encoder.num_layers
        )

        self.pos_enc = SinusoidalPositionalEncoding(config.model.inter_channels,
            max_len=2000)

        self.phoneme_hint_emb = nn.Embedding(
            config.model.phonemes_count + 1, 
            config.model.inter_channels
        )
        self.phoneme_hint_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.model.inter_channels, 
                nhead=config.model.phoneme_hint_encoder.n_head, 
                activation=F.silu,
                batch_first=True),
            num_layers=config.model.phoneme_hint_encoder.num_layers
        )
        
        # Phoneme predictor tower
        self.phone_tower = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.model.inter_channels, 
                nhead=config.model.phone_tower.n_head, 
                activation=F.silu,
                batch_first=True),
            num_layers=config.model.phone_tower.num_layers
        )
        self.phone_proj = nn.Linear(
            config.model.inter_channels, 
            config.model.phonemes_count + 1
            # g2p from soundchoice includes space which we treat as blank token
        )

        # Residual VAE encoder tower
        self.vae_tower = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.model.inter_channels, 
                nhead=config.model.vae_tower.n_head, 
                activation=F.silu,
                batch_first=True),
            num_layers=config.model.vae_tower.num_layers
        )
        self.vae_proj = nn.Linear(
            config.model.inter_channels, 
            config.model.vae_dim * 2
        )

        self.decoder = _VAEDecoder(config)

        self.speaker_classif = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.model.vae_dim, 
                nhead=config.model.speaker_classifier.n_head, 
                activation=F.silu,
                batch_first=True),
                num_layers=config.model.speaker_classifier.num_layers
        )
        self.speaker_classif_proj = nn.Linear(
            config.model.vae_dim,
            config.model.n_spk
        )
        self.config = config

    def forward(self, x, x_mask, spk_id,
            phoneme_hint=None, phoneme_hint_mask=None):
        """
        Args:
            x: [B, T, C] float (whisper features)
            x_mask: [B, T] bool
            spk_id: [B] long
            phoneme_hint: [B, T2] long (phoneme id)
            phoneme_hint_mask: [B, T2] bool
        """
        x = self.whisper_proj(x)
        x = self.pos_enc(x)
        x = self.base_encoder(x, src_key_padding_mask=x_mask)

        if phoneme_hint is not None:
            phoneme_hint = self.phoneme_hint_emb(phoneme_hint)
            phoneme_hint = self.pos_enc(phoneme_hint)
            phoneme_hint = self.phoneme_hint_encoder(
                phoneme_hint, src_key_padding_mask=phoneme_hint_mask
            )
            # Pad or truncate phoneme_hint to match the sequence length of x
            if phoneme_hint.size(1) < x.size(1):
                padding_size = x.size(1) - phoneme_hint.size(1)
                phoneme_hint = F.pad(phoneme_hint, (0, 0, 0, padding_size))
                phoneme_hint_mask = F.pad(phoneme_hint_mask, (0, 0, 0, padding_size), value=False)
            else:
                phoneme_hint = phoneme_hint[:, :x.size(1)]
                phoneme_hint_mask = phoneme_hint_mask[:, :x.size(1)]
            x = x + phoneme_hint

        phone_x = self.phone_tower(x, src_key_padding_mask=x_mask)
        phone_logits = self.phone_proj(phone_x)

        vae_x = self.vae_tower(x, src_key_padding_mask=x_mask)
        vae_x = self.vae_proj(vae_x)
        m_p, log_var_p = vae_x.chunk(2, dim=-1)

        # sample
        z = m_p + torch.exp(log_var_p / 2) * torch.randn_like(m_p)

        # decode
        y = self.decoder(phone_logits, z, x_mask, spk_id)

        # classify
        speaker_logits = self.speaker_classif(grad_reverse(m_p), src_key_padding_mask=x_mask)
        speaker_logits = self.speaker_classif_proj(speaker_logits).mean(dim=1)

        return phone_logits, m_p, log_var_p, y, speaker_logits

if __name__ == "__main__":
    config = OmegaConf.load("configs/config.yaml")
    model = PASIFVAE(config)
    x = torch.randn(2, 1000, config.model.whisper_channels)
    x_mask = torch.randint(0, 2, (2, 1000)).bool()
    spk_id = torch.randint(0, config.model.n_spk, (2,))
    phoneme_hint = torch.randint(0, config.model.phonemes_count, (2, 100))
    phoneme_hint_mask = torch.randint(0, 2, (2, 100)).bool()

    phone_logits, m_p, log_var_p, y, speaker_logits = model(x, x_mask, spk_id, phoneme_hint, phoneme_hint_mask)
    print(phone_logits.shape, m_p.shape, log_var_p.shape, y.shape, speaker_logits.shape)

    phone_logits, m_p, log_var_p, y, speaker_logits = model(x, x_mask, spk_id, None, None)

    # ctc loss
    ctc_loss = nn.CTCLoss(blank=config.model.blank_id)
    ctc_loss = ctc_loss(rearrange(
        phone_logits, "B T C -> T B C"), 
        phoneme_hint, 
        x_mask.sum(-1), 
        phoneme_hint_mask.sum(-1))

    # reconstruction loss
    recon_loss = F.l1_loss(y, x)

    # kl div
    kl_loss = (-0.5 * torch.sum(1 + log_var_p - m_p.pow(2) - log_var_p.exp())) / x.shape[0]

    # speaker classification loss
    speaker_loss = F.cross_entropy(speaker_logits, spk_id)

    print(ctc_loss, recon_loss, kl_loss, speaker_loss)