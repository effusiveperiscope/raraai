from einops import rearrange
from torch import nn
import torch
from modeling.my_nsf import MyGeneratorNSF
from svc_helper.svc.rvc.lib.infer_pack import commons
from modeling.v10.encoder import V10Encoder
from modeling.v05.rvc import PosteriorEncoder, ResidualCouplingBlock

class V10Synthesizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_p = V10Encoder(config)
        self.enc_q = PosteriorEncoder(
            in_channels=config.model.spec_channels,
            out_channels=config.model.inter_channels,
            hidden_channels=config.model.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=config.model.gin_channels,
            spk_embed_dim=config.model.spk_embed_dim)
        self.dec = MyGeneratorNSF(
            initial_channel=config.model.inter_channels,
            resblock=config.model.resblock,
            resblock_kernel_sizes=config.model.resblock_kernel_sizes,
            resblock_dilation_sizes=config.model.resblock_dilation_sizes,
            upsample_rates=config.model.upsample_rates,
            upsample_initial_channel=config.model.upsample_initial_channel,
            upsample_kernel_sizes=config.model.upsample_kernel_sizes,
            gin_channels=config.model.gin_channels,
            spk_embed_dim=config.model.spk_embed_dim,
            sr=config.model.sr,
            is_half=True)
        self.flow = ResidualCouplingBlock(
            channels=config.model.inter_channels,
            hidden_channels=config.model.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=3,
            gin_channels=config.model.gin_channels,
            spk_embed_dim=config.model.spk_embed_dim)
        self.segment_size = config.model.segment_size

    @torch.jit.ignore
    def forward(
        self, phone, phone_lengths, pitchf, y, y_lengths, ds,
        lam_grl = 1.0, pitchq=None
    ):  
        m_p, logs_p, x_mask, spk_emb_pred, pre_proj_x = self.enc_p(phone, pitchf, phone_lengths, 
            lam_grl=lam_grl)

        z, m_q, logs_q, y_mask = self.enc_q(rearrange(y, 'b t c -> b c t'), y_lengths, spk_id=ds)
        z_p = self.flow(z, y_mask, spk_id=ds)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(x=z_slice, f0=pitchf, spk_id=ds)
        return o, y_mask, ids_slice, (z, z_p, m_p, logs_p, m_q, logs_q), spk_emb_pred 

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        noise_scale: float = 0.66666,
    ):
        z_p, m_p, logs_p = self.enc_p(
            phone=phone, pitchf=nsff0, lengths=phone_lengths)
        z_p = rearrange(z_p, 'b t c -> b c t')
        m_p = rearrange(m_p, 'b t c -> b c t')
        logs_p = rearrange(logs_p, 'b t c -> b c t')

        x_mask = commons.sequence_mask(
            phone_lengths, phone.size(1)).unsqueeze(1)
        z_p = z_p * x_mask

        z = self.flow(z_p, x_mask, spk_id=sid, reverse=True)
        o = self.dec(z * x_mask, nsff0, spk_id=sid)
        return o, x_mask, (z, z_p, m_p, logs_p)