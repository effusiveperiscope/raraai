from einops import rearrange
import torch
from modeling.vevo_repcodec import VevoRepCodec
from modeling.rvc.my_nsf import MyGeneratorNSF
from modeling.vits.models import PosteriorEncoder
from modeling.vits.commons import rand_slice_segments_with_pitch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self,
        hp,
        down_factor: int = 4):
        super().__init__()
        self.enc_q = PosteriorEncoder(
            hp.data.filter_length // 2 + 1,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.codec = VevoRepCodec(
            input_channels=hp.vits.inter_channels,
            output_channels=hp.vits.inter_channels,
            encode_channels=hp.vits.inter_channels,
            decode_channels=hp.vits.inter_channels,
            code_dim=hp.codec.code_dim,
            codebook_size=hp.codec.codebook_size,
        )
        self.dec = MyGeneratorNSF(hp=hp)
        self.spk_emb = nn.Embedding(hp.vits.max_spk_count, hp.vits.gin_channels)
        self.hp = hp
        # Temporal resampling around the codec so that
        # code rate == decoder frame rate / down_factor (default 1/4).
        # Convs operate on (B, C, T).
        self.down_factor = down_factor
        self.downsample = nn.Conv1d(
            hp.vits.inter_channels,
            hp.vits.inter_channels,
            kernel_size=down_factor,
            stride=down_factor,
        )
        self.upsample = nn.ConvTranspose1d(
            hp.vits.inter_channels,
            hp.vits.inter_channels,
            kernel_size=down_factor,
            stride=down_factor,
        )

    def forward(self, spec, spec_l, f0, sid):
        g = self.spk_emb(sid)
        # enc_q returns (B, C, T) with T = decoder frame rate.
        x, x_mask = self.enc_q(spec, spec_l)

        # Downsample encoder frames by down_factor so codec runs at
        # 1/down_factor of the decoder frame rate (default 1/4).
        # Pad time so it is divisible by down_factor (pad on the right).
        t_pad = (-x.shape[-1]) % self.down_factor
        if t_pad > 0:
            x = F.pad(x, (0, t_pad))
        x_ds = self.downsample(x)  # (B, C, T/down_factor)
        # VevoRepCodec.forward expects (B, T, C).
        x_ds = rearrange(x_ds, "b c t -> b t c")

        yq, _, _, _, vqloss, perplexity = self.codec(x_ds, do_uq=False)

        # Upsample codec output back to decoder frame rate.
        # Codec returns yq as (B, T', C).
        yq = rearrange(yq, "b t c -> b c t")
        yq = self.upsample(yq)
        if t_pad > 0:
            yq = yq[..., : yq.shape[-1] - t_pad]
        y_slice, f0_slice, ids_slice = rand_slice_segments_with_pitch(
            yq, f0, spec_l, self.hp.data.segment_size)
        audio = self.dec(g, y_slice, f0_slice)
        return audio

if __name__ == "__main__":
    from omegaconf import OmegaConf
    spec = torch.randn(1, 769, 356) # Apparently the model expects the channel to be first for spec
    sid = torch.randint(0, 5, (1,))
    spec_l = torch.tensor([356])
    f0 = torch.randn(1, 356)

    hp = OmegaConf.load("configs/base.yaml")
    decoder = Decoder(hp)

    print(decoder(spec, spec_l, f0, sid).shape)
