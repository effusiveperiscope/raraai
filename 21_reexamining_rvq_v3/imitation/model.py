# Transformer Encoder
# Intake quantized latent from VeVo, quantized pitch, intensity curve? or just mean intensity
# Find duplicate timesteps

import torch
from torch import nn

class VoiceImitator(nn.Module):
    def __init__(self,
        codec_dim=768):
        super.__init__()

    def forward(self,
        emb_q, emb_id, f0_norm, 
        f0_mean, f0_std, spk_emb):

        # cull duplicates

        # combine embedding + f0 norm

        # stats to film

        # decoder

        # output stats + f0_pred

        pass