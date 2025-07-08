import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from modeling.grl import grad_reverse
from modeling.commons import AttentionPooling, DepthwiseSeparableConv1d, SinusoidalPositionalEncoding
from modeling.spk_cond import FiLMGenerator
from torch.nn.utils import spectral_norm
from commons import check_logits
from einops import rearrange

class PitchConditioner(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.pitch_uv_emb = nn.Parameter(torch.randn(config.model.inter_channels))
        self.pitch_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, config.model.inter_channels)
        )

    def forward(self, pitch):
        mel_pitch = 1127 * torch.log1p(pitch / 700)
        mel_pitch = mel_pitch.unsqueeze(-1)

        voiced_mask = (pitch > 0).float().unsqueeze(-1)

        pitch_feat = self.pitch_proj(mel_pitch) * voiced_mask
        pitch_feat += (1 - voiced_mask) * self.pitch_uv_emb

        return pitch_feat

class ContentEncoder(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=5, padding=2),
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=5, padding=2),
        ] + [
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=3, padding=1) for _ in range(config.model.content_n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.model.inter_channels) for _ in self.convs
        ])
        self.out_proj = nn.Linear(config.model.inter_channels, config.model.content_channels)

    def forward(self, x, mask):
        x = rearrange(x, "b t c -> b c t")

        for i, layer in enumerate(self.convs):
            r_x = x
            x = F.silu(x)

            x = rearrange(x, "b c t -> b t c")
            x = self.norms[i](x)
            x = rearrange(x, "b t c -> b c t")

            x = x * rearrange(mask, "b t -> b 1 t")
            x += r_x
        x = rearrange(x, "b c t -> b t c")
        return self.out_proj(x)

class ColoringTower(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.in_proj = nn.Linear(config.model.content_channels, config.model.inter_channels)
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=5, padding=2),
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=5, padding=2),
        ] + [
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=3, padding=1) for _ in range(config.model.coloring_n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.model.inter_channels) for _ in self.convs
        ])
        self.conditions = nn.ModuleList([
            FiLMGenerator(config.model.gin_channels, config.model.inter_channels) for _ in self.convs
        ])
        self.out_proj = nn.Linear(config.model.inter_channels, config.model.inter_channels)
        self.config = config
        self.emb_g = nn.Embedding(config.model.spk_embed_dim, config.model.gin_channels)

    def forward(self, x, mask, spk_id):
        # g is embedded speaker [batch_size, inter_channels]
        x = self.in_proj(x)
        x = rearrange(x, "b t c -> b c t")

        g = self.emb_g(spk_id)

        content_feature = None
        for i,layer in enumerate(self.convs):
            r_x = x

            if i == self.config.model.content_guide_layer:
                content_feature = rearrange(x, "b c t -> b t c")

            x = F.silu(x)

            gamma, beta = self.conditions[i](g.unsqueeze(1))
            x = rearrange(x, "b c t -> b t c")
            x = gamma * x + beta
            x = self.norms[i](x)
            x = rearrange(x, "b t c -> b c t")

            x = x * rearrange(mask, "b t -> b 1 t")

            x += r_x
        x = rearrange(x, "b c t -> b t c")
        return self.out_proj(x), content_feature

class SpeakerEncoder(nn.Module):
    def __init__(self, config, n_layers=2):
        super(SpeakerEncoder, self).__init__()
        embed_dim = config.model.content_channels
        spk_dim = config.model.spk_emb_channels

        self.encoder = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(
                embed_dim, embed_dim, kernel_size=5, padding=2, spectral_norm=True) for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(n_layers)]
        )
        self.final_proj = DepthwiseSeparableConv1d(
            embed_dim, spk_dim, kernel_size=3, padding=1
        )

    def forward(self, x, mask):
        x = rearrange(x, "b t c -> b c t")
        for i,layer in enumerate(self.encoder):
            x = layer(x)
            x = x * rearrange(mask, "b t -> b 1 t")

            x = rearrange(x, "b c t -> b t c")
            x = self.norms[i](x)
            x = rearrange(x, "b t c -> b c t")

            x = F.silu(x)
        outputs = self.final_proj(x)
        outputs = torch.mean(outputs, dim=-1)
        return outputs

class SpeakerConditionalDiscriminator(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.emb_g = nn.Embedding(config.model.spk_embed_dim, config.model.gin_channels)
        self.in_proj = nn.Linear(config.model.inter_channels, config.model.disc_channels)
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=3, padding=1, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=3, padding=1, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=1, padding=0, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=1, padding=0, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=1, padding=0, spectral_norm=True),
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.model.disc_channels) for _ in range(len(self.convs))
        ])
        self.conditions = nn.ModuleList([
            FiLMGenerator(config.model.gin_channels, conv.in_channels) for conv in self.convs
        ])
        self.pool = AttentionPooling(config.model.disc_channels)
        self.out_proj = nn.Linear(config.model.disc_channels, 1)
        self.debug_flag = False

    def forward(self, x, x_mask, spk):
        g = self.emb_g(spk).unsqueeze(1)

        x = self.in_proj(x)
        x = rearrange(x, "b t c -> b c t")

        for i, conv in enumerate(self.convs):
            xs = x

            x = F.silu(x)

            gamma, beta = self.conditions[i](g)
            x = rearrange(x, "b c t -> b t c")
            x = gamma * x + beta
            x = self.norms[i](x)
            x = rearrange(x, "b t c -> b c t")

            x = conv(x) * x_mask.unsqueeze(1)

            x = x + xs

        x = rearrange(x, "b c t -> b t c")
        x = x * x_mask.unsqueeze(-1)

        x = self.pool(x, x_mask)
        x = self.out_proj(x)
        return x

class V09Encoder(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.model.get('hubert_dim', 768), config.model.inter_channels)
        self.sipe = SinusoidalPositionalEncoding(config.model.inter_channels)
        self.base_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model.inter_channels,
                nhead=8,
                dim_feedforward=768,
                activation=F.silu,
                batch_first=True
            ),
            num_layers=config.model.base_encoder_n_layers
        )
        self.pitch_cond = PitchConditioner(config)
        self.content_encoder = ContentEncoder(config)
        self.coloring_tower = ColoringTower(config)
        self.speaker_encoder = SpeakerEncoder(config, n_layers=2) # so-vits-svc 5.0 disentanglement objective
        self.speaker_discriminator = SpeakerConditionalDiscriminator(config)

        self.final_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model.inter_channels, config.model.inter_channels * 2))

    def content_encode(self, h, h_mask, pitch):
        x = self.in_proj(h) + self.pitch_cond(pitch)
        x = self.sipe(x)
        x = x.to(h.dtype)
        x = self.base_encoder(x, src_key_padding_mask=~h_mask)
        x = self.content_encoder(x, h_mask)
        return x

    def forward(self, h, h_mask, pitch, spk, noise_scale=1.0):
        c = self.content_encode(h, h_mask, pitch)
        col, cf = self.coloring_tower(c, h_mask, spk)
        p = self.final_proj(col)
        m_p, logs_p = p.chunk(2, dim=-1)
        z = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale
        return z, m_p, logs_p

    def disc_logits(self, h, h_mask, pitch, spk):
        c = self.content_encode(h, h_mask, pitch)
        col, cf = self.coloring_tower(c, h_mask, spk)
        disc_logits = self.speaker_discriminator(col, h_mask, spk)
        return disc_logits

    def train_step(self,
        h_A, h_A_mask, # h is hubert features
        h_B, h_B_mask,
        pitch_A, pitch_B, spk_A, 
        spk_emb_B,
        lambda_grl, label_alpha=0.1):

        c_A = self.content_encode(h_A, h_A_mask, pitch_A)
        c_B = self.content_encode(h_B, h_B_mask, pitch_B)

        # Content is speaker agnostic.
        # so-vits-svc 5.0 disentanglement objective
        # We do this on B because that gets the full diversity of speakers
        spk_emb_pred_B = self.speaker_encoder(
            grad_reverse(c_B, lambda_grl * self.config.train.mul_grl_content), h_A_mask)
        loss_content_inv = F.cosine_embedding_loss(
            spk_emb_pred_B, spk_emb_B, torch.ones(spk_emb_B.shape[0]).to(h_A.device)
        )

        # Coloring is speaker-correct.
        # Discriminator wants to classify BA as fake, A as real.
        # Upstream network wants to trick discriminator.
        col_A, _ = self.coloring_tower(c_A, h_A_mask, spk_A)
        col_BA, _ = self.coloring_tower(c_B, h_B_mask, spk_A)
        disc_logits_BA = self.speaker_discriminator(
            # less GRL here because it is deeper in the network 
            # and harder to train
            grad_reverse(col_BA, lambda_grl), h_B_mask, spk_A)
        disc_logits_A = self.speaker_discriminator(
            col_A.detach(), h_A_mask, spk_A)
        bce = nn.BCEWithLogitsLoss()
        spk_fake_loss = bce(
            disc_logits_BA, torch.zeros_like(disc_logits_BA, device=h_A.device))
        spk_real_loss = bce(
            disc_logits_A, torch.full_like(disc_logits_A, 1.0 - label_alpha, device=h_A.device))

        # Get stats for downstream KL div.
        p_A = self.final_proj(col_A)
        m_p_A, logs_p_A = p_A.chunk(2, dim=-1)
        z_A = m_p_A + torch.exp(logs_p_A) * torch.randn_like(m_p_A)

        return loss_content_inv, spk_fake_loss, spk_real_loss, m_p_A, logs_p_A, z_A