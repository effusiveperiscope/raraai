import logging
from typing import Optional
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch

logger = logging.getLogger(__name__)

from svc_helper.svc.rvc.lib.infer_pack.models import PosteriorEncoder, ResidualCouplingBlock, GeneratorNSF
from svc_helper.svc.rvc.lib.infer_pack import commons
from modeling.v05.encoder import V05Encoder
from modeling.my_nsf import MyGeneratorNSF
from commons import count_parameters

import pdb
from PyQt5.QtCore import pyqtRemoveInputHook
import sys
import traceback
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


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

class SynthesizerV05(nn.Module):
    def __init__(
        self,
        config,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        **kwargs
    ):
        super(SynthesizerV05, self).__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.config = config
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim

        self.enc_p = V05Encoder(self.config)

        self.dec = MyGeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=kwargs["is_half"],
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        logger.debug(
            "gin_channels: "
            + str(gin_channels)
            + ", self.spk_embed_dim: "
            + str(self.spk_embed_dim)
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(
        self, 
        phone_A, phone_lengths_A, 
        phone_B, phone_lengths_B,
        pitchf_A, 
        spks_A, spks_B,
        y_A, y_lengths_A,
        lambda_grl, label_alpha
    ):  
        g_A = self.emb_g(spks_A).unsqueeze(-1)  # [b, 256, 1]
        g_B = self.emb_g(spks_B).unsqueeze(-1)  # [b, 256, 1]

        spk_loss, fake_loss, real_loss, \
            m_p_A, logs_p_A, m_p_B, logvar_p_B, z_A, z_B = self.enc_p.train_step(
                h_A = phone_A, h_A_mask = commons.sequence_mask(phone_lengths_A, phone_A.size(1)),
                h_B = phone_B, h_B_mask = commons.sequence_mask(phone_lengths_B, phone_B.size(1)),
                spk_A = g_A.squeeze(-1), 
                spk_B = g_B.squeeze(-1),
                ids_A = spks_A,
                lambda_grl=lambda_grl, label_alpha=label_alpha
            )
        logs_p_A = rearrange(logs_p_A, 'b t c -> b c t')
        m_p_A = rearrange(m_p_A, 'b t c -> b c t')

        z, m_q_A, logs_q_A, z_mask = self.enc_q(rearrange(y_A, 'b t c -> b c t'), y_lengths_A, g=g_A)

        z_p = self.flow(z, z_mask, g=g_A)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths_A, self.segment_size
        )
        pitchf_A = commons.slice_segments2(pitchf_A, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf_A, g=g_A)

        return o, z_mask, ids_slice, \
            (z, z_p, m_p_A, logs_p_A, m_q_A, logs_q_A), \
            (spk_loss, fake_loss, real_loss)

    @torch.jit.ignore
    def step_finetune(
        self,
        phone,
        phone_lengths,
        pitchf,
        spks,
        y,
        y_lengths,
    ): 
        g = self.emb_g(spks).unsqueeze(-1)
        _, m_p, logs_p, u, c, col = self.enc_p(
            h=phone, h_mask=commons.sequence_mask(phone_lengths, phone.size(1)),
            spk_emb=g.squeeze(-1)
        )

        z, m_q, logs_q, z_mask = self.enc_q(
            rearrange(y, 'b t c -> b c t'), y_lengths, g=g
        )
        z_p = self.flow(z, z_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)

        return o, z_mask, ids_slice, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        noise_scale: float = 0.66666,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        _, m_p, logs_p, u, c, col = self.enc_p(
            h=phone, 
            h_mask = commons.sequence_mask(phone_lengths, phone.size(1)),
            spk_emb=g.squeeze(-1), noise_scale=noise_scale)
        logs_p = rearrange(logs_p, 'b t c -> b c t')
        m_p = rearrange(m_p, 'b t c -> b c t')

        z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale

        x_mask = commons.sequence_mask(phone_lengths, phone.size(1)).unsqueeze(1)
        z_p = z_p * x_mask

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/v07.yaml')
    model = SynthesizerV05(config, **config.model, is_half=True)
    model.eval()

    phone = torch.randn((2, 100, config.model.hubert_dim))
    phone_lengths = torch.tensor([100, 100])
    pitch = torch.randn((2, 100)).to(torch.int32)
    nsff0 = torch.randn((2, 100))
    sid = torch.tensor([0, 1])
    model.infer(phone, phone_lengths, pitch, nsff0, sid)
    print(count_parameters(model))