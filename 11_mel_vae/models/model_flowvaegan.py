import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import huber, create_half_mask
from augs import time_mask_synced, freq_mask_synced, time_jitter_synced
from conformer import ConformerBlock
import logging
import normflows as nf
from models.common import (
    FiLMGenerator, DepthwiseSeparableConv1d, DepthwiseSeparableConv1dTransposed,
    SinusoidalPositionalEncoding)
from einops import rearrange
from models.disc import (DepthwisePatchGANDiscriminator, 
    calculate_feature_map_mask, calculate_masked_feature_matching_loss, collapse_mask_to_lengths)

logger = logging.getLogger(__name__)

import sys
import pdb
def myexcepthook(type, value, tb):
    print(type, value, tb)
    pdb.post_mortem(tb)
sys.excepthook = myexcepthook

# Inspired by StyleGAN
class LearnableNoiseLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.condition = FiLMGenerator(channels, channels)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, channels) - feature map used for conditioning noise
        Returns:
            x: (batch_size, seq_len, channels) - feature map with added conditioned noise
        """
        gamma, beta = self.condition(x)
        noise = torch.randn_like(x)
        noise = gamma * noise + beta
        return x + noise

class MelFlowVAEGANEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.model.hidden_size, max_len=config.train.max_len * 2)
        self.in_proj = nn.Linear(config.model.input_size, config.model.hidden_size)
        self.in_conformer = ConformerBlock(
            dim=config.model.hidden_size,
            dim_head=config.model.dim_head,
            heads=config.model.num_heads,
            ff_mult=config.model.ff_mult,
            conv_expansion_factor=config.model.conv_expansion_factor,
            conv_kernel_size=config.model.conv_kernel_size,
            attn_dropout=config.model.attn_dropout,
            ff_dropout=config.model.ff_dropout,
            conv_dropout=config.model.conv_dropout
        )
        self.resample_conformers = nn.ModuleList(
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
                DepthwiseSeparableConv1d(
                    in_channels=config.model.hidden_size,
                    out_channels=config.model.hidden_size,
                    kernel_size=config.model.encoder.conv_kernel_sizes[i],
                    stride=config.model.encoder.conv_strides[i],
                    padding=config.model.encoder.conv_paddings[i]
                )
                for i in range(len(config.model.encoder.conv_kernel_sizes))]
        )
        self.skip_convs = nn.ModuleList(
            [nn.Conv1d(
                in_channels=config.model.hidden_size,
                out_channels=config.model.hidden_size,
                kernel_size=config.model.encoder.downsample_ratio[i],
                stride=config.model.encoder.downsample_ratio[i]
            ) for i in range(len(config.model.encoder.downsample_ratio))]
        )
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(config.model.hidden_size)
        self.out_proj = nn.Linear(config.model.hidden_size, config.model.latent_dim*2)
        self.config = config

    def forward(self, x, x_mask):
        """
        Args:
            x: (batch_size, seq_len, num_mels) - ideally padded to multiple of 16 or downsample rate.
                This is the log mel spectrogram normalized using global statistics.
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

        x = x + self.positional_encoding(x)
        x = self.in_conformer(x, current_mask)
        x = x * current_mask.unsqueeze(2)
        x = self.layer_norm(x)

        for i in range(len(self.resample_conformers)):
            resid_x = self.skip_convs[i](rearrange(x, "b s c -> b c s"))

            x = self.resample_conformers[i](x, current_mask) # Takes [batch_size, seq_len, channels]
            x = x * current_mask.unsqueeze(2)
            x = rearrange(x, "b s c -> b c s")
            x = self.downsampling_convs[i](x) # takes [batch_size, channels, seq_len]

            x = x + resid_x # skip connection

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

        # tanh clamping
        z_mean = torch.tanh(z_mean / self.config.model.z_tanh_max)
        z_log_var = torch.tanh(z_log_var / self.config.model.z_logvar_tanh_max)

        # reparameterization
        std = torch.exp(0.5 * z_log_var)
        z = z_mean + std * torch.randn_like(std)

        return z_mean, z_log_var, z, current_mask

class MelFlowVAEGANDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.model.latent_dim, config.model.hidden_size)
        self.out_conformer = ConformerBlock(
            dim=config.model.hidden_size,
            dim_head=config.model.dim_head,
            heads=config.model.num_heads,
            ff_mult=config.model.ff_mult,
            conv_expansion_factor=config.model.conv_expansion_factor,
            conv_kernel_size=config.model.conv_kernel_size,
            attn_dropout=config.model.attn_dropout,
            ff_dropout=config.model.ff_dropout,
            conv_dropout=config.model.conv_dropout
        )
        self.resample_conformers = nn.ModuleList(
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
                DepthwiseSeparableConv1dTransposed(
                    in_channels=config.model.hidden_size,
                    out_channels=config.model.hidden_size,
                    kernel_size=config.model.decoder.conv_kernel_sizes[i],
                    stride=config.model.decoder.conv_strides[i],
                    padding=config.model.decoder.conv_paddings[i],
                    output_padding=config.model.decoder.conv_output_padding[i]
                )
                for i in range(len(config.model.decoder.conv_kernel_sizes))]
        )
        self.noise_layers = nn.ModuleList(
            [
                LearnableNoiseLayer(config.model.hidden_size)
                for i in range(sum(config.model.decoder.noise_layers))
            ]
        )
        if config.model.decoder.pitch_conditioning:
            self.pitch_embed = nn.Linear(1, config.model.hidden_size)
            self.positional_encoding = SinusoidalPositionalEncoding(
                config.model.hidden_size, max_len=config.train.max_len * 2)
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
            pitch_mask = pitch != 0
            # Use log pitch - closer to normal distribution and perceptually aligned
            pitch = torch.log(pitch + 1e-5)
            pitch = pitch.unsqueeze(-1)
            # Embed pitch: (B, T_final) -> (B, T_final, 1) -> (B, T_final, H)
            pitch_embedded = self.pitch_embed(pitch)
            # We use the pitch mask to mask out the pitch embeddings rather than
            # masking out the log pitch - because otherwise it would conflict with
            # f0 = 1 (log f0 = 0)
            pitch_embedded = pitch_embedded * pitch_mask.unsqueeze(-1)
            pitch_embedded = pitch_embedded + self.positional_encoding(pitch_embedded)

            # Noise augmentation
            if self.training:
                noise = torch.randn_like(pitch_embedded) * self.config.train.pitch_noise_std
                pitch_embedded = pitch_embedded + noise

        current_seq_len = x.shape[1]
        current_mask = z_mask

        # Process through conformers and upsampling convs
        for i in range(len(self.resample_conformers)): # Upsample first
            x = rearrange(x, "b s c -> b c s")
            x = self.upsampling_convs[i](x) # Or the Upsample+Conv block
            x = self.activation(x)
            x = rearrange(x, "b c s -> b s c")

            # Interpolate mask to the *new* sequence length *before* Conformer
            new_seq_len = x.shape[1]
            if current_seq_len != new_seq_len:
                current_mask = F.interpolate(
                    current_mask.unsqueeze(1).float(),
                    size=new_seq_len,
                    mode='nearest'
                ).squeeze(1).bool()
                current_seq_len = new_seq_len

            # Apply FiLM conditioning (if used) *after* upsampling
            if self.config.model.decoder.pitch_conditioning:
                # Interpolate pitch to the new sequence length
                pitch_embedded_for_interp = rearrange(pitch_embedded, "b s c -> b c s")
                pitch_cond_interp = F.interpolate(
                    pitch_embedded_for_interp,
                    size=current_seq_len, # Use the *new* sequence length
                    mode='linear',
                    align_corners=False
                )
                pitch_cond = rearrange(pitch_cond_interp, "b c s -> b s c")
                pitch_cond = pitch_cond * current_mask.unsqueeze(2) # shouldn't be necessary but just in case
                gamma, beta = self.film_generators[i](pitch_cond)
                x = gamma * x + beta
            x = x * current_mask.unsqueeze(2)

            # Apply learnable noise for finer details
            if self.config.model.decoder.noise_layers[i] == True:
                x = self.noise_layers[i](x)
            x = x * current_mask.unsqueeze(2) 

            # Apply Conformer block
            x = self.resample_conformers[i](x, current_mask)
            x = x * current_mask.unsqueeze(2) 
        
        x = self.layer_norm(x)

        x = self.out_conformer(x, current_mask)
        x = x * current_mask.unsqueeze(2)
        x = self.layer_norm(x)

        x_recon = self.out_proj(x)
        
        return x_recon

class MelFlowVAEGAN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MelFlowVAEGANEncoder(config)
        self.decoder = MelFlowVAEGANDecoder(config)
        self.discriminator = DepthwisePatchGANDiscriminator(config)

        self.adversarial_loss = nn.MSELoss(reduction='none') # Least squares GAN

        # Flows
        self.base = nf.distributions.base.DiagGaussian(config.model.latent_dim)
        flows = [
            nf.flows.CoupledRationalQuadraticSpline(
                num_input_channels=config.model.latent_dim,
                num_blocks=config.model.flows.num_blocks // 2,
                num_hidden_channels=config.model.flows.hidden_size
            ),
            nf.flows.InvertibleAffine(config.model.latent_dim),
            nf.flows.CoupledRationalQuadraticSpline(
                num_input_channels=config.model.latent_dim,
                num_blocks=config.model.flows.num_blocks // 2,
                num_hidden_channels=config.model.flows.hidden_size
            ),
        ]
        self.flows = nf.NormalizingFlow(self.base, flows)

        self.config = config


    def forward(self, x, x_mask, pitch=None, disable_flow=False):
        """
        Args:
            x: (B, T_in, C_in) - Input log mel spectrogram normalized by global statistics
            x_mask: (B, T_in) - Input mask
            pitch: (B, T_in) or None - Pitch curve matching input mel length
                (Decoder will use this length internally if pitch conditioning is enabled)
        """
        seq_len = x.shape[1]
        if seq_len % self.config.model.sampling_ratio != 0:
            logger.warning(f"Warning: Sequence length {seq_len} is not a multiple of sampling ratio {self.config.model.sampling_ratio}.")

        z_mean, z_log_var, z, z_mask = self.encoder(x, x_mask)
        
        if disable_flow:
            z_transformed = z.clone()
            log_det = torch.zeros(z.shape[0], z.shape[1]).to(z.device)
        else: 
            z = rearrange(z, 'b n c -> (b n) c')
            z_transformed, log_det = self.flows.forward_and_log_det(z) 
            z = rearrange(z, '(b n) c -> b n c', b=x.shape[0])
            z_transformed = rearrange(z_transformed, '(b n) c -> b n c', b=x.shape[0])
            log_det = rearrange(log_det, '(b n) -> b n', b=x.shape[0])

        x_recon = self.decoder(z_transformed, z_mask, pitch=pitch)
        x_recon = x_recon[:, :x.shape[1], :]
        return x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask


    def sample(self, z_len, batch_size, pitch=None, disable_flow=False):
        z = torch.randn((batch_size, z_len, self.config.model.latent_dim)).to(self.config.device)

        z = rearrange(z, 'b n c -> (b n) c')
        z_transformed, log_det = self.flows.forward_and_log_det(z)
        z_transformed = rearrange(z_transformed, '(b n) c -> b n c', b=batch_size)

        x_recon = self.decoder(z_transformed, 
            torch.ones(z.shape[0], z.shape[1]).bool().to(self.config.device),
            pitch=pitch)
        return x_recon


    def log_gaussian_prob(self, x, mu, log_var, mask): # log pdf
        # Ensure log_var has the same device as x
        if not torch.is_tensor(log_var):
            log_var = torch.tensor(log_var)
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu)
        log_var = log_var.to(x.device)
        mu = mu.to(x.device)

        log_scale = 0.5 * log_var
        scale = torch.exp(log_scale)
        term1 = -0.5 * math.log(2 * math.pi)
        term2 = -log_scale
        term3 = -0.5 * ((x - mu) / scale)**2
        # sum on channel
        return torch.sum((term1 + term2 + term3) * mask.unsqueeze(2), dim=2) 

    # Assumes p_z ~ N(0, I)
    def kl_loss(self, z, z_mean, z_log_var, z_mask, z_transformed, log_det):
        log_q_z = self.log_gaussian_prob(z, z_mean, z_log_var, z_mask)
        log_p_z = self.log_gaussian_prob(z_transformed, 0, 0, z_mask)
        kl_div = ((log_q_z - log_p_z - log_det) * z_mask).sum() / z_mask.sum()
        return kl_div

    def disc_aug_losses(self, real_x, fake_x, x_mask):
        if self.config.train.disc_noise_std > 0.0:
            real_noise = torch.randn_like(real_x) * self.config.train.disc_noise_std
            fake_noise = torch.randn_like(fake_x) * self.config.train.disc_noise_std
            real_noise = real_noise * x_mask.unsqueeze(2).float()
            fake_noise = fake_noise * x_mask.unsqueeze(2).float()

            real_x = real_x + real_noise
            fake_x = fake_x + fake_noise

        if random.random() < self.config.train.time_mask_prob:
            out = time_mask_synced(
                [real_x, fake_x],
                [x_mask, x_mask])
            real_x = out[0]
            fake_x = out[1]

        if random.random() < self.config.train.freq_mask_prob:
            out = freq_mask_synced(
                [real_x, fake_x])
            real_x = out[0] * x_mask.unsqueeze(2).float()
            fake_x = out[1] * x_mask.unsqueeze(2).float()

        if random.random() < self.config.train.jitter_prob:
            out = time_jitter_synced(
                [real_x, fake_x],
                [x_mask, x_mask])
            real_x = out[0]
            fake_x = out[1]

        return self.disc_losses(real_x, fake_x, x_mask)

    def disc_losses(self, real_x, fake_x, x_mask):
        real_features_list = self.discriminator(real_x)
        fake_features_list = self.discriminator(fake_x)

        fm_loss = calculate_masked_feature_matching_loss(
            discriminator=self.discriminator,
            real_features_list=real_features_list,
            fake_features_list=fake_features_list,
            real_input_lengths=collapse_mask_to_lengths(x_mask),
            max_input_length=real_x.shape[1],
            input_feature_dim=self.config.model.input_size,
            device=real_x.device,
            loss_type='l1'
        )

        final_real_features = real_features_list[-1]
        final_fake_features = fake_features_list[-1]
        target_real = torch.ones_like(final_real_features) - self.config.train.alpha # label smoothing
        target_fake = torch.zeros_like(final_fake_features)
        target_gen = torch.ones_like(final_fake_features) - self.config.train.alpha

        final_layer_index = len(self.discriminator.conv_params) - 1
        mask_real_final = calculate_feature_map_mask(
            conv_params_list=self.discriminator.conv_params,
            layer_index=final_layer_index,
            input_lengths=collapse_mask_to_lengths(x_mask),
            max_input_length=real_x.shape[1],
            input_feature_dim=self.config.model.input_size,
            device=real_x.device
        )
        mask_fake_final = mask_real_final.clone()
        criterion = self.adversarial_loss

        loss_real = criterion(final_real_features, target_real) * mask_real_final.float()
        loss_fake = criterion(final_fake_features, target_fake) * mask_fake_final.float()
        loss_gen = criterion(final_fake_features, target_gen) * mask_fake_final.float()

        real_loss = loss_real.sum() / (mask_real_final.sum() + 1e-8)
        fake_loss = loss_fake.sum() / (mask_fake_final.sum() + 1e-8)
        gen_loss = loss_gen.sum() / (mask_fake_final.sum() + 1e-8)

        return real_loss, fake_loss, fm_loss, gen_loss

    def recon_loss(self, x, x_recon, x_mask):
        x_mask_unsqueeze = x_mask.unsqueeze(2)
        recon_loss = huber(
            x=x * x_mask_unsqueeze,
            y=x_recon * x_mask_unsqueeze)
        recon_loss = recon_loss.sum() / x_mask.sum()
        return recon_loss


if __name__ == '__main__':
    from omegaconf import OmegaConf 
    config = OmegaConf.load("configs/vaegan.yaml")

    net = MelFlowVAEGAN(config)
    preweights_path = "mel_flowvae_log_f0.ckpt"
    net.load_state_dict(torch.load(preweights_path, weights_only=False)['state_dict'], strict=False)
    print("Loaded preweights")

    from functools import partial

    def nan_detector_hook_with_name(module, input, output, module_name):
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            print(f"NaN detected in the output of module: {module_name} ({module.__class__.__name__})")
            import pdb; pdb.set_trace()

    def trace_nans(model):
        hook_handles = []
        for name, module in model.named_modules():
            # Register a forward hook for every module
            hook = partial(nan_detector_hook_with_name, module_name=name)
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)
        return hook_handles

    hooks = trace_nans(net)

    x = torch.randn(2, 608, config.model.input_size)
    x_mask = torch.ones(2, 608).bool()
    pitch = (torch.randn(2, 608) * (880 - 55) + 55).clamp(min=55, max=880) # (negative values will result in NANs)
    x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask = net(x, x_mask, pitch=pitch)
    print(z.shape)

    disc_loss = net.disc_losses(x, x_recon, x_mask)
    print(disc_loss)

    recon_loss = net.recon_loss(x, x_recon, x_mask)
    print(recon_loss)

    kl_loss = net.kl_loss(z, z_mean, z_log_var, z_mask, z_transformed, log_det)
    print(kl_loss)