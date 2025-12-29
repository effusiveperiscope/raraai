
import torch

from torch import nn
from torch.nn import functional as F
from . import attentions
from . import commons
from . import modules
from .utils import f0_to_coarse
from ..rvc.my_nsf import MyGeneratorNSF, SpeakerAdapter
from ..rvc.commons import FiLMGenerator, PitchConditioner2
from einops import rearrange


class TextEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 max_spk_count,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)

        self.pit = PitchConditioner2(hidden_channels)

        self.spk_emb = nn.Embedding(max_spk_count, hidden_channels)
        self.spk_adapter = FiLMGenerator(
            condition_dim=hidden_channels,
            target_dim=hidden_channels
        )

        self.intensity_proj = nn.Linear(1, hidden_channels)

        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, f0, pitch_extras=None, sid=None,
        intensity=None,
        noise_scale=1.0):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        ).to(x.device)
        x = self.pre(x) * x_mask

        if pitch_extras is None:
            pitch_extras = {}

        pit_cond = self.pit(f0, **pitch_extras, use_dtype=x.dtype).transpose(1, 2)
        x = x + pit_cond

        spk = self.spk_emb(sid)
        gamma, beta = self.spk_adapter(spk.unsqueeze(1))
        x = rearrange(x, "b c t -> b t c")
        x = x + x * gamma + beta

        x = x + self.intensity_proj(intensity)
        x = rearrange(x, "b t c -> b c t")

        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale) * x_mask
        # from commons import plot_spectrogram
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()
        return z, m, logs, x_mask, x


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        p_dropout=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                    p_dropout=p_dropout,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        ).to(x.device)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.emb_g = nn.Linear(hp.vits.spk_dim, hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            hp.codec.whisper_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            hp.vits.max_spk_count,
            2,
            6,
            3,
            0.1,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim,
            p_dropout=0.2
        )
        self.dec = MyGeneratorNSF(hp=hp)

    def forward(self, ppg_q, pit, spec, spk, ppg_l, spec_l, sid, intensity, pitch_extras=None):
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg_q, ppg_l, f0=f0_to_coarse(pit), pitch_extras=pitch_extras, sid=sid, intensity=intensity)
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g)

        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_q, pit, spec_l, self.segment_size)
        if pitch_extras is not None and len(pitch_extras):
            for key, value in pitch_extras.items():
                pitch_extras[key] = commons.slice_pitch_segments(
                    value, ids_slice, self.segment_size)

        audio = self.dec(spk, z_slice, pit_slice, pitch_extras=pitch_extras)

        # SNAC to flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=spk)
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk, reverse=True)
        return audio, ids_slice, spec_mask, \
            (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r)

    def posterior_test(self, spec, spec_l, pit, spk, pitch_extras=None):
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g)

        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_q, pit, spec_l, self.segment_size)
        if pitch_extras is not None and len(pitch_extras):
            for key, value in pitch_extras.items():
                pitch_extras[key] = commons.slice_pitch_segments(
                    value, ids_slice, self.segment_size)

        audio = self.dec(spk, z_slice, pit_slice, pitch_extras=pitch_extras)
        return audio

    def infer(self, ppg_q, pit, spk, ppg_l, sid, intensity, noise_scale=0.3, pitch_extras=None):
        ppg_q = ppg_q + torch.randn_like(ppg_q) * 0.0001  # Perturbation
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg_q, ppg_l, f0=f0_to_coarse(pit), pitch_extras=pitch_extras, sid=sid, intensity=intensity,
                noise_scale=noise_scale)

        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec(spk, z * ppg_mask, f0=pit, pitch_extras=pitch_extras)
        return o

import pdb
import sys
import traceback
from PyQt5.QtCore import pyqtRemoveInputHook
def custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook that prints the exception information
    and then drops into a pdb debugger session.
    """
    pyqtRemoveInputHook()
    # First, print the exception information as Python normally would.
    # We use traceback.print_exception to ensure consistent formatting.
    print("An unhandled exception occurred:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nDropping into debugger...")

    # Then, drop into the pdb debugger.
    # The post_mortem function starts the debugger at the point of the exception.
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = custom_excepthook


if __name__ == "__main__":
    from omegaconf import OmegaConf
    ppg = torch.randn(1, 356, 1280)
    vec = torch.randn(1, 356, 256)
    pit = torch.randn(1, 356)
    spec = torch.randn(1, 769, 356) # Apparently the model expects the channel to be first for spec
    spk = torch.randn(1, 256)
    ppg_l = torch.tensor([356])
    spec_l = torch.tensor([356])
    quant_pitch = torch.round(torch.randn(1, 356)).clamp(-4, 4).abs().long()
    target_f0_mean = torch.randn(1)

    hp = OmegaConf.load("config/svc5_base.yaml")

    model = SynthesizerTrn(
        spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp
    )
    audio, ids_slice, spec_mask, (
        z_f, z_r, z_p, m_p, logs_p, z_q, 
        m_q, logs_q, logdet_f, logdet_r), f0_pred = \
        model(ppg, vec, pit, spec, spk, ppg_l, spec_l, quant_pitch, target_f0_mean)
    print(f0_pred.shape)