import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from modeling.grl import grad_reverse
from modeling.commons import DepthwiseSeparableConv1d, SinusoidalPositionalEncoding
from modeling.spk_cond import FiLMGenerator
from einops import rearrange

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

    def forward(self, x, mask, spk):
        # g is embedded speaker [batch_size, inter_channels]
        x = self.in_proj(x)
        x = rearrange(x, "b t c -> b c t")
        for i,layer in enumerate(self.convs):
            r_x = x
            x = F.silu(x)
            x = layer(x)

            gamma, beta = self.conditions[i](spk.unsqueeze(1))
            x = rearrange(x, "b c t -> b t c")
            x = gamma * x + beta
            x = rearrange(x, "b t c -> b c t")

            x = x * rearrange(mask, "b t -> b 1 t")

            x += r_x
        x = rearrange(x, "b c t -> b t c")
        return self.out_proj(x)


class SpeakerConditionalDiscriminator(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels=config.model.inter_channels, out_channels=64, 
                kernel_size=5, padding=2),
            DepthwiseSeparableConv1d(
                in_channels=64, out_channels=128, 
                kernel_size=3, padding=1),
            DepthwiseSeparableConv1d(
                in_channels=128, out_channels=256, 
                kernel_size=1, padding=0),
            DepthwiseSeparableConv1d(
                in_channels=256, out_channels=512, 
                kernel_size=1, padding=0),
            DepthwiseSeparableConv1d(
                in_channels=512, out_channels=1024, 
                kernel_size=1, padding=0),
        ])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model.inter_channels,
                nhead=8,
                dim_feedforward=768,
                activation=F.silu,
                batch_first=True
            ),
            num_layers=4
        )
        self.conditions = nn.ModuleList([
            FiLMGenerator(config.model.gin_channels, conv.in_channels) for conv in self.convs
        ])
        self.encoder_proj = nn.Linear(config.model.inter_channels, 1024)
        self.out_proj = nn.Linear(1024, 1)

    def forward(self, x, mask, spk):
        x_enc = self.encoder(x, src_key_padding_mask=~mask)

        x = rearrange(x, "b t c -> b c t")

        for i, layer in enumerate(self.convs):
            x = F.silu(x)

            gamma, beta = self.conditions[i](spk.unsqueeze(1))
            x = rearrange(x, "b c t -> b t c")
            x = gamma * x + beta
            x = rearrange(x, "b t c -> b c t")

            x = layer(x)

        x = rearrange(x, "b c t -> b t c")
        x = x * mask.unsqueeze(-1)

        x = x + self.encoder_proj(x_enc)
        x = self.out_proj(x)
        return x


class V05Encoder(nn.Module):
    def __init__(self, config : OmegaConf):
        super().__init__()
        self.in_proj = nn.Linear(768, config.model.inter_channels)
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
        self.content_encoder = ContentEncoder(config)
        self.coloring_tower = ColoringTower(config)
        self.speaker_discriminator = SpeakerConditionalDiscriminator(config)

        self.final_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model.inter_channels, config.model.inter_channels * 2))

    def train_step(self,
        h_A, h_A_mask, # h is hubert features
        h_B, h_B_mask,
        spk_A, spk_B, 
        lambda_grl, label_alpha=0.1):

        h_A = self.in_proj(h_A) 
        h_A = self.sipe(h_A)

        h_B = self.in_proj(h_B) 
        h_B = self.sipe(h_B)

        # RVC losses are downstream; not included in here.

        # Coloring is content preserving. 
        # c_A = cont(col(c_A|s_B))
        u_A = self.base_encoder(h_A, src_key_padding_mask=~h_A_mask)
        c_A = self.content_encoder(u_A, h_A_mask)
        col_AB = self.coloring_tower(c_A, h_A_mask, spk_B)
        c_AB = self.content_encoder(col_AB, h_A_mask)

        c_loss = F.l1_loss(c_A, c_AB)

        # Alignment of u_A space with color space.
        col_A = self.coloring_tower(c_A, h_A_mask, spk_A)
        align_loss = F.l1_loss(u_A.detach(), col_A)

        # Coloring is speaker-correct.
        # Discriminator wants to classify BA as fake, A as real
        # Upstream network wants to trick discriminator
        u_B = self.base_encoder(h_B, src_key_padding_mask=~h_B_mask)
        c_B = self.content_encoder(u_B, h_B_mask)
        col_BA = self.coloring_tower(c_B, h_B_mask, spk_A)

        disc_logits_BA = self.speaker_discriminator(
            grad_reverse(col_BA, lambda_grl), h_B_mask, spk_A)
        disc_logits_A = self.speaker_discriminator(
            col_A, h_A_mask, spk_A)
        bce = nn.BCEWithLogitsLoss()
        fake_loss = bce(
            disc_logits_BA, torch.zeros_like(disc_logits_BA, device=h_A.device))
        real_loss = bce(
            disc_logits_A, torch.full_like(disc_logits_A, 1.0 - label_alpha, device=h_A.device))

        # Also get stats for downstream KL div
        col_B = self.coloring_tower(c_B, h_B_mask, spk_B)
        p_A = self.final_proj(col_A)
        p_B = self.final_proj(col_B)
        m_p_A, logs_p_A = p_A.chunk(2, dim=-1)
        m_p_B, logs_p_B = p_B.chunk(2, dim=-1)

        z_A = m_p_A + torch.exp(logs_p_A) * torch.randn_like(m_p_A)
        z_B = m_p_B + torch.exp(logs_p_B) * torch.randn_like(m_p_B)

        return c_loss, align_loss, fake_loss, real_loss, \
            m_p_A, logs_p_A, m_p_B, logs_p_B, z_A, z_B

    def forward(self, h, h_mask, spk_emb, noise_scale=1.0):
        u = self.base_encoder(h, src_key_padding_mask=~h_mask)

        c = self.content_encoder(u, h_mask)
        col = self.coloring_tower(c, h_mask, spk_emb)

        p = self.final_proj(col)

        m_p, logs_p = p.chunk(2, dim=-1)

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