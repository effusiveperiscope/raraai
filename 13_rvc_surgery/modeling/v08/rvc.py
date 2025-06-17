import math
import itertools
import logging
from typing import Optional
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
import numpy as np
import torch

logger = logging.getLogger(__name__)

from svc_helper.svc.rvc.lib.infer_pack.models import TextEncoder768, PosteriorEncoder, ResidualCouplingBlock
from svc_helper.svc.rvc.lib.infer_pack import attentions, commons, modules
from svc_helper.svc.rvc.lib.infer_pack.commons import get_padding, init_weights
from modeling.spk_cond import FiLMGenerator
from einops import rearrange
from modeling.v08.encoder import V08Encoder
from modeling.my_nsf import MyGeneratorNSF
from modeling.f0_predictor import F0Predictor


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class V08Synthesizer(nn.Module):
    def __init__(
        self,
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
        use_pitch_predictor=False,
        pitch_quant_dim=8, # Number of discrete pitch levels
        **kwargs
    ):
        super(V08Synthesizer, self).__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
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
        self.enc_p = V08Encoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            gin_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
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
            spk_embed_dim=spk_embed_dim,
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

        if use_pitch_predictor:
            self.pitch_predictor = F0Predictor(
                speech_dim=inter_channels,
                pitch_quant_dim=pitch_quant_dim,
                spk_emb_dim=gin_channels
            )

    def last_n_enc_parameters(self, last_n):
        params = []
        layers_count = len(self.enc_p.encoder.attn_layers)
        for i in range(last_n):
            truei = layers_count - 1 - i
            params.append(self.enc_p.encoder.attn_layers[truei])
            params.append(self.enc_p.encoder.ffn_layers[truei])
        return itertools.chain.from_iterable(params)

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

    def prior_only(
        self, phone, phone_lengths, pitchf):
        _, _, x_mask, spk_emb_pred, pre_proj_x = self.enc_p(phone, pitchf, phone_lengths)
        return x_mask, spk_emb_pred, pre_proj_x

    @torch.jit.ignore
    def forward(
        self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds,
        lam_grl = 1.0, pitchq=None
    ):  # 这里ds是id，[bs,1]
        # print(1,pitch.shape)#[bs,t]
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        m_p, logs_p, x_mask, spk_emb_pred, pre_proj_x = self.enc_p(phone, pitchf, phone_lengths, 
            lam_grl=lam_grl)

        if hasattr(self, 'pitch_predictor'):
            assert pitchq is not None, "pitchq must be provided for pitch predictor"
            f0_pred = self.pitch_predictor(
                quant_pitch=pitchq,
                target_f0_mean=pitchf.mean(),
                speech=pre_proj_x.detach(),
                speech_mask=x_mask,
                spk_emb=g)
        else:
            f0_pred = None

        z, m_q, logs_q, y_mask = self.enc_q(rearrange(y, 'b t c -> b c t'), y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        # print(-1,pitchf.shape,ids_slice,self.segment_size,self.hop_length,self.segment_size//self.hop_length)
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        # print(-2,pitchf.shape,z_slice.shape)
        o = self.dec(x=z_slice, f0=pitchf, spk_id=ds)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), spk_emb_pred, f0_pred

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        rate: Optional[torch.Tensor] = None,
        noise_scale: float = 0.66666,
        prior_pitch: torch.Tensor = None
    ):
        if prior_pitch is None:
            prior_pitch = nsff0
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask, spk_emb_pred, pre_proj_x = self.enc_p(phone, prior_pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            nsff0 = nsff0[:, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, spk_id=sid)
        return o, x_mask, (z, z_p, m_p, logs_p)
