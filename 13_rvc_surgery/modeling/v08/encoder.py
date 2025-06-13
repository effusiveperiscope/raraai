import torch.nn as nn
import torch
from omegaconf import OmegaConf
from svc_helper.svc.rvc.lib.infer_pack import attentions, commons, modules
from einops import rearrange
import math

class PitchConditioner(nn.Module):
    def __init__(self, inter_channels):
        super().__init__()
        self.pitch_uv_emb = nn.Parameter(torch.randn(inter_channels))
        self.pitch_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, inter_channels)
        )

    def forward(self, pitch):
        mel_pitch = 1127 * torch.log1p(pitch / 700)
        mel_pitch = mel_pitch.unsqueeze(-1)

        voiced_mask = (pitch > 0).float().unsqueeze(-1)

        pitch_feat = self.pitch_proj(mel_pitch) * voiced_mask
        pitch_feat += (1 - voiced_mask) * self.pitch_uv_emb

        return pitch_feat

class V08Encoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        gin_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        f0=True,
    ):
        super(V08Encoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(512, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = PitchConditioner(hidden_channels)
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.spk_classifier = SpeakerClassifier(hidden_channels, 256)

    def forward(self, phone: torch.Tensor, pitchf: torch.Tensor, lengths: torch.Tensor):
        if pitchf is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitchf)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask) # Should run GRL here
        pre_proj_x = x
        spk_feat_pred = self.spk_classifier(x)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask, spk_feat_pred, pre_proj_x


import torch
import torch.nn as nn

from torch.autograd import Function
from torch.nn.utils import weight_norm

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    ''' Gradient Reversal Layer
            Y. Ganin, V. Lempitsky,
            "Unsupervised Domain Adaptation by Backpropagation",
            in ICML, 2015.
        Forward pass is the identity function
        In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    '''

    def __init__(self, lambda_reversal=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class SpeakerClassifier(nn.Module):
    def __init__(self, embed_dim, spk_dim):
        super(SpeakerClassifier, self).__init__()
        self.classifier = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            weight_norm(nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(embed_dim, spk_dim, kernel_size=5, padding=2))
        )

    def forward(self, x):
        ''' Forward function of Speaker Classifier:
            x = (B, embed_dim, len)
        '''
        # pass through classifier
        outputs = self.classifier(x)  # (B, nb_speakers)
        outputs = torch.mean(outputs, dim=-1)
        return outputs
