import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from modeling.grl import grad_reverse
from modeling.spk_classifier import SpeakerClassifierCNN
from modeling.commons import AttentionPooling, DepthwiseSeparableConv1d, SinusoidalPositionalEncoding
from modeling.spk_cond import FiLMGenerator
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
                kernel_size=3, padding=1),
        ] + [
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=1, padding=0) for _ in range(config.model.content_n_layers)
        ])
        self.out_proj = nn.Linear(config.model.inter_channels, config.model.content_size)

    def forward(self, x, mask):
        x = rearrange(x, "b t c -> b c t")
        for layer in self.convs:
            r_x = x
            x = F.silu(x)
            x = layer(x)
            x = x * rearrange(mask, "b t -> b 1 t")
            x += r_x
        x = rearrange(x, "b c t -> b t c")
        return self.out_proj(x)

class ColoringTower(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.in_proj = nn.Linear(config.model.content_size, config.model.inter_channels)
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=5, padding=2),
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=3, padding=1),
        ] + [
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=config.model.inter_channels, 
                kernel_size=1, padding=0) for _ in range(config.model.coloring_n_layers)
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

        if spk_id is not None:
            g = self.emb_g(spk_id)
        else:
            g = None

        content_feature = None
        for i,layer in enumerate(self.convs):
            r_x = x

            if i == self.config.model.content_guide_layer:
                content_feature = rearrange(x, "b c t -> b t c")

            x = F.silu(x)
            x = layer(x)

            gamma, beta = self.conditions[i](g.unsqueeze(1))
            x = rearrange(x, "b c t -> b t c")
            x = gamma * x + beta
            x = rearrange(x, "b t c -> b c t")

            x = x * rearrange(mask, "b t -> b 1 t")

            x += r_x
        x = rearrange(x, "b c t -> b t c")
        return self.out_proj(x), content_feature


class SpeakerConditionalDiscriminator(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.emb_g = nn.Embedding(config.model.spk_embed_dim, config.model.disc_channels)
        self.in_proj = nn.Linear(config.model.inter_channels, config.model.disc_channels)
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=7, padding=3, spectral_norm=True),
            DepthwiseSeparableConv1d(
                in_channels=config.model.disc_channels, 
                out_channels=config.model.disc_channels, 
                kernel_size=5, padding=2, spectral_norm=True),
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
                kernel_size=3, padding=1, spectral_norm=True),
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, config.model.disc_channels) for _ in range(len(self.convs))
        ])
        self.conditions = nn.ModuleList([
            FiLMGenerator(config.model.gin_channels, conv.in_channels) for conv in self.convs
        ])
        self.pool = AttentionPooling(config.model.disc_channels)
        self.out_proj = nn.Linear(config.model.disc_channels, 1)

    def forward(self, x, x_mask, spk):
        # Disable transformer encoder for now
        # x = self.sipe(x)
        # x_enc = self.encoder(x, src_key_padding_mask=~mask)
        g = self.emb_g(spk).unsqueeze(1)

        x = self.in_proj(x)
        x = rearrange(x, "b t c -> b c t")

        for i, conv in enumerate(self.convs):
            xs = x

            x = F.silu(x)

            gamma, beta = self.conditions[i](g)
            x = rearrange(x, "b c t -> b t c")
            x = gamma * x + beta
            x = rearrange(x, "b t c -> b c t")

            x = self.norms[i](x)
            x = conv(x) * x_mask.unsqueeze(1)

            x = x + xs

        x = rearrange(x, "b c t -> b t c")
        x = x * x_mask.unsqueeze(-1)

        x = self.pool(x, x_mask)
        x = self.out_proj(x)
        return x


class V05Encoder(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
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
            num_layers=3
        )
        self.pitch_cond = PitchConditioner(config)
        self.content_encoder = ContentEncoder(config)
        self.coloring_tower = ColoringTower(config)
        self.speaker_classifier = SpeakerClassifierCNN(config)
        self.speaker_discriminator = SpeakerConditionalDiscriminator(config)

        self.final_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model.inter_channels, config.model.inter_channels * 2))

    def content_only(self, x, mask, pitch):
        x = self.in_proj(x) + self.pitch_cond(pitch)
        x = self.sipe(x)
        x = self.base_encoder(x, src_key_padding_mask=~mask)
        x = self.content_encoder(x, mask)
        return x

    def train_step(self,
        h_A, h_A_mask, # h is hubert features
        h_B, h_B_mask,
        pitch_A, pitch_B,
        spk_A, spk_B, lambda_grl, label_alpha=0.1):

        h_A = self.in_proj(h_A) + self.pitch_cond(pitch_A)
        h_A = self.sipe(h_A)

        h_B = self.in_proj(h_B) + self.pitch_cond(pitch_B)
        h_B = self.sipe(h_B)

        # RVC losses are downstream; not included in here.

        u_A = self.base_encoder(h_A, src_key_padding_mask=~h_A_mask)
        c_A = self.content_encoder(u_A, h_A_mask)

        # Content is locally speaker agnostic.
        # This is a little more sensitive than the conditioned discriminator
        spk_logits = self.speaker_classifier(grad_reverse(c_A, lambda_grl * 0.5), h_A_mask)
        #check_logits(spk_logits)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=label_alpha)
        spk_loss = ce_loss(spk_logits, spk_A)

        # Alignment of u_A space with color space.
        col_A, cf_A = self.coloring_tower(c_A, h_A_mask, spk_A)

        # Coloring is speaker-correct.
        # Discriminator wants to classify BA as fake, A as real
        # Upstream network wants to trick discriminator
        u_B = self.base_encoder(h_B, src_key_padding_mask=~h_B_mask)
        c_B = self.content_encoder(u_B, h_B_mask)
        col_BA, _ = self.coloring_tower(c_B, h_B_mask, spk_A)

        disc_logits_BA = self.speaker_discriminator(
            grad_reverse(col_BA, lambda_grl), h_B_mask, spk_A)
        disc_logits_A = self.speaker_discriminator(
            col_A.detach(), h_A_mask, spk_A)
        bce = nn.BCEWithLogitsLoss()
        fake_loss = bce(
            disc_logits_BA, torch.zeros_like(disc_logits_BA, device=h_A.device))
        real_loss = bce(
            disc_logits_A, torch.full_like(disc_logits_A, 1.0 - label_alpha, device=h_A.device))
        #check_logits(disc_logits_BA)
        #check_logits(disc_logits_A)

        # Also get stats for downstream KL div
        col_B, _ = self.coloring_tower(c_B, h_B_mask, spk_B)
        p_A = self.final_proj(col_A)
        p_B = self.final_proj(col_B)
        m_p_A, logs_p_A = p_A.chunk(2, dim=-1)
        m_p_B, logs_p_B = p_B.chunk(2, dim=-1)

        # Log clamping for KL stability
        logs_p_A = torch.clamp(logs_p_A, min=-20.0, max=20.0)
        logs_p_B = torch.clamp(logs_p_B, min=-20.0, max=20.0)

        z_A = m_p_A + torch.exp(logs_p_A) * torch.randn_like(m_p_A)
        z_B = m_p_B + torch.exp(logs_p_B) * torch.randn_like(m_p_B)

        return spk_loss, fake_loss, real_loss, \
            m_p_A, logs_p_A, m_p_B, logs_p_B, z_A, z_B

    def forward(self, h, h_mask, pitch, spk_id, noise_scale=1.0):
        h = self.in_proj(h) + self.pitch_cond(pitch)
        h = self.sipe(h)
        u = self.base_encoder(h, src_key_padding_mask=~h_mask)

        c = self.content_encoder(u, h_mask)
        col, _ = self.coloring_tower(c, h_mask, spk_id)

        p = self.final_proj(col)

        m_p, logs_p = p.chunk(2, dim=-1)
        logs_p = torch.clamp(logs_p, min=-20.0, max=20.0)

        z = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale
        return z, m_p, logs_p, u, c, col

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from commons import count_parameters
    config = OmegaConf.load("configs/v05.yaml")
    model = V05Encoder(config)

    print(count_parameters(model))

    h_A = torch.randn((2, 100, 768))
    h_B = torch.randn((2, 100, 768))
    h_mask_A = torch.ones((2, 100), dtype=torch.bool)
    h_mask_B = torch.ones((2, 100), dtype=torch.bool)
    spk_A = torch.randn((2, config.model.gin_channels))
    spk_B = torch.randn((2, config.model.gin_channels))

    model.train_step(h_A, h_mask_A, h_B, h_mask_B, spk_A, spk_B, 0.1)