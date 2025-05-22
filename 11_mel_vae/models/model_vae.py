import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock
from einops import rearrange
from common import huber
from models.common import (
    FiLMGenerator)
import logging

logger = logging.getLogger(__name__)

# Downsampling conformer encoder
class MelVAEEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.model.input_size, config.model.hidden_size)
        self.conformers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=config.model.hidden_size,
                    dim_head=config.model.dim_head,
                    heads=config.model.num_heads,
                    ff_mult=config.model.ff_mult,
                    conv_expansion_factor=config.model.conv_expansion_factor,
                    conv_kernel_size=config.model.conv_kernel_size,
                    attn_dropout=config.model.attn_dropout,
                    ff_dropout=config.model.ff_dropout,
                    conv_dropout=config.model.conv_dropout
                ) for i in range(len(config.model.encoder.conv_kernel_sizes))
            ]
        )
        self.downsampling_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.model.hidden_size,
                    out_channels=config.model.hidden_size,
                    kernel_size=config.model.encoder.conv_kernel_sizes[i],
                    stride=config.model.encoder.conv_strides[i],
                    padding=config.model.encoder.conv_paddings[i]
                )
                for i in range(len(config.model.encoder.conv_kernel_sizes))]
        )
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(config.model.hidden_size)
        self.out_proj = nn.Linear(config.model.hidden_size, config.model.latent_dim*2)
        self.config = config

    def forward(self, x, x_mask):
        """
        Args:
            x: (batch_size, seq_len, num_els) - ideally padded to multiple of 16 or downsample rate
            x_mask: (batch_size, seq_len) boolean
        
        Returns:
            z_mean: (batch_size, seq_len2, latent_dim)
            z_log_var: (batch_size, seq_len2, latent_dim)
            z: (batch_size, seq_len2, latent_dim)
            z_mask: (batch_size, seq_len2)

        seq_len2 is the new sequence length after downsampling
        """
        x = self.in_proj(x)

        current_seq_len = x.shape[1]
        current_mask = x_mask

        for i in range(len(self.conformers)):
            x = self.conformers[i](x, current_mask) # Takes [batch_size, seq_len, channels]
            x = x * current_mask.unsqueeze(2)
            x = rearrange(x, "b s c -> b c s")
            x = self.downsampling_convs[i](x) # takes [batch_size, channels, seq_len]
            x = self.activation(x) # SiLU activation after convolution
            x = rearrange(x, "b c s -> b s c")

            new_seq_len = x.shape[1]
            if current_seq_len != new_seq_len:
                current_mask = F.interpolate(
                    current_mask.unsqueeze(1).float(),
                    size=new_seq_len,
                    mode='nearest'
                ).squeeze(1).bool()
                current_seq_len = new_seq_len

        x = self.layer_norm(x)
        x = x * current_mask.unsqueeze(2)
        x = self.out_proj(x)

        z_mean, z_log_var = torch.split(x, x.shape[2]//2, dim=2)

        # reparameterization
        std = torch.exp(0.5 * z_log_var)
        z = z_mean + std * torch.randn_like(std)

        return z_mean, z_log_var, z, current_mask

class MelVAEDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.model.latent_dim, config.model.hidden_size)
        self.conformers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=config.model.hidden_size,
                    dim_head=config.model.dim_head,
                    heads=config.model.num_heads,
                    ff_mult=config.model.ff_mult,
                    conv_expansion_factor=config.model.conv_expansion_factor,
                    conv_kernel_size=config.model.conv_kernel_size,
                    attn_dropout=config.model.attn_dropout,
                    ff_dropout=config.model.ff_dropout,
                    conv_dropout=config.model.conv_dropout
                ) for i in range(len(config.model.decoder.conv_kernel_sizes))
            ]
        )
        self.upsampling_convs = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels=config.model.hidden_size,
                    out_channels=config.model.hidden_size,
                    kernel_size=config.model.decoder.conv_kernel_sizes[i],
                    stride=config.model.decoder.conv_strides[i],
                    padding=config.model.decoder.conv_paddings[i],
                    output_padding=config.model.decoder.conv_output_padding[i]
                )
                for i in range(len(config.model.decoder.conv_kernel_sizes))]
        )
        if config.model.decoder.pitch_conditioning:
            self.pitch_embed = nn.Linear(1, config.model.hidden_size)
            self.film_generators = nn.ModuleList(
                [
                    FiLMGenerator(condition_dim=config.model.hidden_size,
                                  target_dim=config.model.hidden_size)
                    for i in range(len(config.model.decoder.conv_kernel_sizes))
                ]
            )
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(config.model.hidden_size)
        self.out_proj = nn.Linear(config.model.hidden_size, config.model.input_size)
        self.config = config
    
    def forward(self, z, z_mask=None, pitch=None):
        """
        Args:
            z: (batch_size, seq_len, latent_dim) - Latent representation
            z_mask: (batch_size, seq_len) boolean - Optional mask for latent
            pitch: (batch_size, seq_len) - Optional pitch condition
        
        Returns:
            x_recon: (batch_size, seq_len2, output_size) - Reconstructed mel spectrogram
        """
        x = self.in_proj(z)
        if self.config.model.decoder.pitch_conditioning:
            pitch = pitch.unsqueeze(-1)
            # Embed pitch: (B, T_final) -> (B, T_final, 1) -> (B, T_final, H)
            pitch_embedded = self.pitch_embed(pitch)

        current_seq_len = x.shape[1]
        current_mask = z_mask

        # Process through conformers and upsampling convs
        for i in range(len(self.conformers)):
            if self.config.model.decoder.pitch_conditioning:
                pitch_embedded_for_interp = rearrange(pitch_embedded, "b s c -> b c s")
                pitch_cond_interp = F.interpolate(
                    pitch_embedded_for_interp,
                    size=current_seq_len,
                    mode='linear',
                    align_corners=False
                )
                pitch_cond = rearrange(pitch_cond_interp, "b c s -> b s c")
                gamma, beta = self.film_generators[i](pitch_cond)
                x = gamma * x + beta
            x = x * current_mask.unsqueeze(2)

            x = self.conformers[i](x, current_mask)
            x = x * current_mask.unsqueeze(2)

            x = rearrange(x, "b s c -> b c s")
            x = self.upsampling_convs[i](x)
            x = self.activation(x)
            x = rearrange(x, "b c s -> b s c")

            new_seq_len = x.shape[1]
            if current_seq_len != new_seq_len:
                current_mask = F.interpolate(
                    current_mask.unsqueeze(1).float(),
                    size=new_seq_len,
                    mode='nearest'
                ).squeeze(1).bool()
                current_seq_len = new_seq_len
        
        x = self.layer_norm(x)
        x_recon = self.out_proj(x)
        
        return x_recon

class MelVAE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MelVAEEncoder(config)
        self.decoder = MelVAEDecoder(config)
        self.config = config

    def forward(self, x, x_mask, pitch=None):
        """
        Args:
            x: (B, T_in, C_in) - Input mel spectrogram
            x_mask: (B, T_in) - Input mask
            pitch: (B, T_in) - Pitch curve matching input mel length
                                (Decoder will use this length internally)
        """
        seq_len = x.shape[1]
        if seq_len % self.config.model.sampling_ratio != 0:
            logger.warning(f"Warning: Sequence length {seq_len} is not a multiple of sampling ratio {self.config.model.sampling_ratio}.")

        z_mean, z_log_var, z, z_mask = self.encoder(x, x_mask)
        x_recon = self.decoder(z, z_mask, pitch=pitch)
        x_recon = x_recon[:, :x.shape[1], :]
        return x_recon, z_mean, z_log_var, z_mask

    def loss(self, x, x_mask, x_recon, z_mean, z_log_var, z_mask, kl_beta = 1e-3):
        # Reconstruction loss (Huber, masked)
        recon_loss = huber(
            x=x * x_mask.unsqueeze(2),
            y=x_recon * x_mask.unsqueeze(2))
        recon_loss = recon_loss.sum() / x_mask.sum()
        
        # KL Divergence (masked)
        kl_per_elem = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        kl_loss = (kl_per_elem * z_mask.unsqueeze(2)).sum() / z_mask.sum()

        loss = recon_loss + kl_beta * kl_loss
        return loss, recon_loss, kl_loss

if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config.yaml")
    encoder = MelVAEEncoder(config)
    print(f"Encoder has {sum([p.numel() for p in encoder.parameters()])} parameters")

    x = torch.randn(2, 600, 128)
    x_mask = torch.ones(2, 600).bool()
    z, _, _, z_mask = encoder(x, x_mask)
    print(z.shape)

    pitch = torch.randn(2, 600)
    decoder = MelVAEDecoder(config)
    print(f"Decoder has {sum([p.numel() for p in decoder.parameters()])} parameters")

    x_recon = decoder(z, z_mask, pitch=pitch)
    x_recon = x_recon[:, :x.shape[1], :]
    print(x_recon.shape)

    model = MelVAE(config)
    print(f"Model has {sum([p.numel() for p in model.parameters()])} parameters")

    x_recon, z_mean, z_log_var, z_mask = model(x, x_mask, pitch=pitch)
    loss, recon_loss, kl_loss = model.loss(
        x, x_mask, x_recon, z_mean, z_log_var, z_mask)
    print(loss, recon_loss, kl_loss)