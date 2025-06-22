from einops import rearrange
from torch import nn
import torch
from modeling.my_nsf import MyGeneratorNSF
from svc_helper.svc.rvc.lib.infer_pack import commons
from modeling.v05.rvc import PosteriorEncoder, ResidualCouplingBlock
from modeling.v09.encoder import V09Encoder

class V09Synthesizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_p = V09Encoder(config)
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
            sr=config.model.sr)
        self.flow = ResidualCouplingBlock(
            channels=config.model.inter_channels,
            hidden_channels=config.model.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=3,
            gin_channels=config.model.gin_channels,
            spk_embed_dim=config.model.spk_embed_dim)
        self.segment_size = config.model.segment_size

    def forward(
        self,
        phone_A, phone_A_mask,
        phone_B, phone_B_mask,
        pitchf_A, pitchf_B,
        y_A, y_lengths_A,
        spk_A, spk_emb_A, lam_grl=1.0, pitchq=None
    ):
        loss_content_inv, spk_fake_loss, spk_real_loss, m_p_A, logs_p_A, z_A = self.enc_p.train_step(
            h_A=phone_A, h_A_mask=phone_A_mask,
            h_B=phone_B, h_B_mask=phone_B_mask,
            pitch_A=pitchf_A, pitch_B=pitchf_B,
            spk_A=spk_A, spk_emb_A=spk_emb_A,
            lambda_grl=lam_grl
        )
        logs_p_A = rearrange(logs_p_A, 'b t c -> b c t')
        m_p_A = rearrange(m_p_A, 'b t c -> b c t')

        z, m_q_A, logs_q_A, z_mask = self.enc_q(rearrange(y_A, 'b t c -> b c t'), y_lengths_A, spk_id=spk_A)

        z_p = self.flow(z, z_mask, spk_id=spk_A)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths_A, self.segment_size
        )
        pitchf_A = commons.slice_segments2(pitchf_A, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf_A, spk_id=spk_A)

        return o, z_mask, ids_slice, (z, z_p, m_p_A, logs_p_A, m_q_A, logs_q_A), \
            (loss_content_inv, spk_fake_loss, spk_real_loss)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/v09.yaml')
    synth = V09Synthesizer(config)
    phone_A = torch.randn((2, 100, config.model.hubert_dim))
    phone_B = torch.randn((2, 100, config.model.hubert_dim))
    phone_A_mask = torch.ones((2, 100), dtype=torch.bool)
    phone_B_mask = torch.ones((2, 100), dtype=torch.bool)
    pitchf_A = torch.randn((2, 100)) * 100
    pitchf_B = torch.randn((2, 100)) * 100
    y_A = torch.randn((2, 100, config.model.spec_channels))
    y_lengths_A = torch.tensor([100, 100])
    spk_A = (torch.randn((2)).abs() * 2).round().long()
    spk_emb_A = torch.randn((2, config.model.gin_channels))

    o, z_mask, ids_slice, (z, z_p, m_p_A, logs_p_A, m_q_A, logs_q_A), (loss_content_inv, spk_fake_loss, spk_real_loss) = synth(
        phone_A=phone_A, phone_A_mask=phone_A_mask,
        phone_B=phone_B, phone_B_mask=phone_B_mask,
        pitchf_A=pitchf_A, pitchf_B=pitchf_B,
        y_A=y_A, y_lengths_A=y_lengths_A,
        spk_A=spk_A, spk_emb_A=spk_emb_A
    )
