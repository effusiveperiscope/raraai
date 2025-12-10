from omegaconf import OmegaConf
from torch import nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange

from .commons import PitchConditioner

class ResBlock(nn.Module):
    def __init__(self, convs : list):
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.GroupNorm(1, c.out_channels) if isinstance(c, nn.Conv1d) else nn.Identity() 
            for c in self.convs])

    def forward(self, x, channels_last=True):
        if channels_last:
            x = rearrange(x, "b t c -> b c t")

        for i,conv in enumerate(self.convs):
            xs = x
            x = F.silu(x)
            x = conv(x)
            x = self.norms[i](x)
            x = x + xs

        if channels_last:
            x = rearrange(x, "b c t -> b t c")
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class F0Predictor2(nn.Module): # No special conditioning
    """
    Simple F0 predictor. Training objective should be log f0 + 1
    """
    def __init__(self, 
        speech_dim: int,
        pitch_quant_dim: int,
        hidden_dim: int):
        super().__init__()

        # self.pos_encoder = PositionalEncoding(hidden_dim)
        self.mean_proj = nn.Linear(1, hidden_dim)
        self.pitch_cond = PitchConditioner(hidden_dim)
        self.convs = ResBlock([
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            ])
        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, 1)

    def forward(self, quant_pitch, target_f0_mean, speech_mask, z=None):
        speech_mask = speech_mask.unsqueeze(-1)
        v_mask = (quant_pitch != 0).unsqueeze(-1).float()
        x = self.pitch_cond(quant_pitch,
            convert_mel=False, use_dtype=target_f0_mean.dtype) * speech_mask * v_mask
        x = x + self.mean_proj(target_f0_mean.unsqueeze(-1)).unsqueeze(1) * speech_mask

        if z is None:
            z = torch.randn_like(x)
        x = x + self.noise_proj(z)

        x = self.convs(x) * speech_mask * v_mask
        x = F.layer_norm(x, x.shape[1:])
        x = F.leaky_relu(x)
        x = self.final_proj(x) * speech_mask  * v_mask
        return x # we predict log pitch plus 1

from torch.nn.utils.parametrizations import spectral_norm
class F0Discriminator2(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(1, 128, 15, padding=7)),
            spectral_norm(nn.Conv1d(128, 256, 7, padding=3)),
            spectral_norm(nn.Conv1d(256, 128, 3, padding=1)),
            spectral_norm(nn.Conv1d(128, 1, 3, padding=1)),
        ])

        self.norms = nn.ModuleList()
        self.projections = nn.ModuleList()

        in_chs = [1, 128, 256, 128]
        out_chs = [128, 256, 128, 1]

        for cin, cout in zip(in_chs, out_chs):
            # group norm except last conv
            if cout > 1:
                self.norms.append(nn.GroupNorm(1, cout))
            else:
                self.norms.append(nn.Identity())

            # projection if channels mismatch
            if cin != cout:
                self.projections.append(nn.Conv1d(cin, cout, kernel_size=1))
            else:
                self.projections.append(nn.Identity())

    def forward(self, pit):
        pit = rearrange(pit, "b t c -> b c t")

        for conv, norm, proj in zip(self.convs, self.norms, self.projections):
            residual = proj(pit)
            out = conv(pit)
            out = F.leaky_relu(out)
            # out = norm(out)
            pit = out + residual  # residual add

        pit = rearrange(pit, "b c t -> b t c")
        return pit

class F0Predictor2(nn.Module):
    def __init__(self, hidden_dim=192, speech_dim=512):
        super().__init__()
        self.mean_proj = nn.Linear(1, hidden_dim)
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.pitch_cond = PitchConditioner(hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 4, 1024, batch_first=True, layer_norm_eps=1e-4),
            num_layers=3
        )
        self.refiner = ResBlock(
            [
                spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)),
                spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3)),
                spectral_norm(nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3)),
            ]
        )
        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, 1)
        self.pos_enc = PositionalEncoding(hidden_dim)

    def forward(self, 
        quant_pitch, target_f0_mean, speech, speech_mask, z=None):

        speech_mask = speech_mask.unsqueeze(-1)
        speech = self.speech_proj(speech) * speech_mask
        pitch = self.pitch_cond(quant_pitch,
            convert_mel=False, use_dtype=target_f0_mean.dtype) * speech_mask

        x = speech + pitch
        x = x + self.mean_proj(target_f0_mean.unsqueeze(-1)).unsqueeze(1) * speech_mask

        if z is None:
            z = torch.randn_like(x)
        x = x + self.noise_proj(z)
        x = x + self.pos_enc(x) 
        x = F.layer_norm(x, x.shape[2:])
        x = self.encoder(x, src_key_padding_mask=~(speech_mask.squeeze(-1))) * speech_mask
        x = self.refiner(x) * speech_mask
        x = F.silu(x)

        v_mask = (quant_pitch != 0).unsqueeze(-1).float()
        x = self.final_proj(x) * speech_mask * v_mask

        return x

class ConvLinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv(x)
        x = rearrange(x, 'b c t  -> b t c')
        x = self.proj(x)
        return x

class F0Discriminator2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.in_proj = nn.Linear(1, hidden_size)
        self.convs = nn.ModuleList([
            ConvLinear(hidden_size, hidden_size, 3, 1),
            ConvLinear(hidden_size, hidden_size, 7, 3),
            ConvLinear(hidden_size, hidden_size, 15, 7),
        ])
        self.pos_enc = PositionalEncoding(hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, 4, 1024, activation="gelu", batch_first=True
            ),
            3,
        )
        self.final_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, x, x_mask):
        x_mask = x_mask.unsqueeze(-1)
        x = self.in_proj(x)
        for conv in self.convs:
            x = x + conv(x)
            x = F.silu(x) * x_mask
            x = F.layer_norm(x, x.shape[2:])
        x = x + self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=(~x_mask).squeeze(-1))
        x = self.final_proj(x)
        return x

class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        # Use a similar conv stack as your original discriminator,
        # but maybe slightly smaller/simpler.
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(self.period, 128, 15, padding=7)),
            spectral_norm(nn.Conv1d(128, 256, 7, padding=3)),
            spectral_norm(nn.Conv1d(256, 128, 7, padding=3)),
            spectral_norm(nn.Conv1d(128, 1, 7, padding=3)),
        ])

    def forward(self, x):
        b, t, c = x.shape
        # Pad to be divisible by period
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, 0, 0, pad_len), "reflect")
            t = t + pad_len
        
        # Reshape to view the signal with the specified period
        x = x.view(b, t // self.period, self.period, c)
        x = x.permute(0, 3, 2, 1).contiguous() # B, C, Period, T'
        x = x.view(b, c * self.period, t // self.period)
        
        # Now pass through the conv stack
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
        return x

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
        )

    def forward(self, x):
        outputs = []
        for d in self.discriminators:
            outputs.append(d(x))
        return outputs

    def gen_loss(self, disc_fake_outputs, target_label=1.0):
        loss = 0
        for d in disc_fake_outputs:
            loss += torch.mean((target_label - d) ** 2)
        return loss

    def disc_loss(self, disc_fake_outputs, disc_real_outputs, target_label=1.0):
        fake_loss = 0
        for d in disc_fake_outputs:
            fake_loss += torch.mean(d ** 2)
        for d in disc_real_outputs:
            fake_loss += torch.mean((target_label - d) ** 2)
        return fake_loss

if __name__ == '__main__':
    model = MultiPeriodDiscriminator()
    model.eval()
    x = torch.randn(1, 256, 1)
    y = model(x)
    for feat in y:
        print(feat.shape)