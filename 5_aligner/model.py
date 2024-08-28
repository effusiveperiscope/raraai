# LAF Language Aware Features
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from commons import sequence_mask, generate_path, convert_pad_shape
from mambapy.mamba import Mamba, MambaConfig
import monotonic_align
import math

class MambaBlock(nn.Module):
    def __init__(self,
        n_embed,
        out_channels,
        dropout=0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 2*n_embed),
            nn.ReLU(),
            nn.Linear(2*n_embed, n_embed),
            nn.Dropout(dropout)
        )
        mamba_config = MambaConfig(
            d_model = n_embed,
            d_state=16,
            d_conv=4,
            expand_factor=2,
            n_layers=2
        )
        self.mamba = Mamba(mamba_config)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.final_proj = nn.Linear(n_embed, out_channels)

    def forward(self, x):
        x = x + self.mamba(self.ln1(x))
        x = x + self.ffn(self.ln2(x)) # (B,N,C)
        return self.final_proj(x)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels = 512,
        out_channels = 512,
        num_groups = 8
    ):
        super().__init__()
        self.ff1 = nn.Linear(in_channels, out_channels)
        self.silu = nn.SiLU()
        self.conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                kernel_size=5, padding=2)
        self.groupnorm = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.ff1(x)
        x = self.silu(x)
        x = rearrange(x, 'b n c -> b c n')
        x = x + self.conv(x)
        x = self.silu(x)
        x = self.groupnorm(x)
        x = rearrange(x, 'b c n -> b n c')
        return x

# Speech features are encoded into latent features
# Model generates an alignment from ASR decoder features to latent features
# With assistance from the original speech features via a bottleneck
# Then latents are sampled and run through decoder to produce final features
class SimpleAlignmentModel(nn.Module):
    def __init__(
        self,
        decoder_input_channels = 512,
        speech_input_channels = 768,
        hidden_channels = 512,
        bottleneck_channels = 64,
        num_heads = 8
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.aligner_stats = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels*2),
        )
        self.aligner_norm = nn.GroupNorm(8, hidden_channels*2)
        
        # In the future these can be "smarter" models
        self.latent_encoder = EncoderBlock(
            speech_input_channels, hidden_channels)
        self.latent_decoder = MambaBlock(
            hidden_channels, speech_input_channels)

        self.in_decoder_proj = nn.Linear(decoder_input_channels, hidden_channels)
        self.in_speech_proj = nn.Linear(speech_input_channels, hidden_channels)
        
        self.bottleneck_attn_decoder = EncoderBlock(
            hidden_channels, bottleneck_channels)
        self.bottleneck_attn_speech = EncoderBlock(
            hidden_channels, bottleneck_channels)
        self.q_proj = EncoderBlock(bottleneck_channels, bottleneck_channels)
        self.k_proj = EncoderBlock(bottleneck_channels, bottleneck_channels)
        self.v_proj = EncoderBlock(bottleneck_channels, bottleneck_channels)
        self.asr_attention = nn.MultiheadAttention(
            bottleneck_channels, num_heads=8,
            batch_first=True)
        self.attn_proj = nn.Linear(bottleneck_channels, hidden_channels)

    # x is the decoder features, z is the target speech features
    # Use MHA to "add" speech feature information to the aligner stats

    # TODO: add speaker embeddings

    def forward(self, x, x_lens, z, z_lens, spk=None):
        x = self.in_decoder_proj(x)
        latent_feats = self.latent_encoder(z)
        z = self.in_speech_proj(z)

        bottle_x = self.bottleneck_attn_decoder(x)
        bottle_z = self.bottleneck_attn_speech(z)
        attn_output, _ = self.asr_attention(
            query=self.q_proj(bottle_x), # Query is from decoder features
            key=self.k_proj(bottle_z), # K/V are from speech features
            value=self.v_proj(bottle_z)
        )
        x = x + self.attn_proj(attn_output)
        x_stats = self.aligner_stats(x)
        x_stats = rearrange(x_stats, 'b n c -> b c n')
        x_stats = self.aligner_norm(x_stats)
        x_stats = rearrange(x_stats, 'b c n -> b n c')

        # stats (b, n, c*2)
        encoder_means = x_stats[:,:,:self.hidden_channels]
        encoder_logs = x_stats[:,:,self.hidden_channels:]

        # Alignment
        x_m = rearrange(encoder_means, 'b n c -> b c n')
        x_logs = rearrange(encoder_logs, 'b n c -> b c n')
        z = rearrange(z, 'b n c -> b c n')

        # From GlowTTS/VITS:
        # log gaussian pdf
        x_s_sq_r = torch.exp(-2 * x_logs)
        # [b, t, 1]
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(
            -1) 
        # [b, t, d] x [b, d, t'] = [b, t, t']
        logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) 
        # [b, t, d] x [b, d, t'] = [b, t, t']
        logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) 
        # [b, t, 1]
        logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) 
        # [b, t, t']
        logp = logp1 + logp2 + logp3 + logp4 

        x_mask = torch.unsqueeze(sequence_mask(x_lens), 1)
        z_mask = torch.unsqueeze(sequence_mask(z_lens), 1)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        attn = monotonic_align.maximum_path(
            logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # [b, t', t], [b, t, d] -> [b, d, t']
        align_m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(
                1, 2) 
        # [b, t', t], [b, t, d] -> [b, d, t']
        align_logs = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(
                1, 2)

        true_decoded = self.latent_decoder(latent_feats)
        sampled_latent = (
            align_m + torch.exp(align_logs) * torch.randn_like(align_m) * z_mask
            ).transpose(1,2)
        pred_decoded = self.latent_decoder(sampled_latent)

        return align_m, align_logs, latent_feats, true_decoded, pred_decoded
