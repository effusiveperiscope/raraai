import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _pad(kernel_size):
    return (kernel_size - 1) // 2


class PitchScaleDiscriminator(nn.Module):
    """A single-scale 1D convolutional discriminator over a pitch contour.

    The pitch contour is a (B, 1, T) signal of log-normalized f0 values with
    unvoiced frames set to 0 (see modeling/generator.PITCH_LOG_SCALE). This is
    the pitch analogue of the audio codec's ScaleDiscriminator, but much
    smaller since a smooth f0 curve carries far less information than audio.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, channels, 15, 1, padding=_pad(15))),
            weight_norm(nn.Conv1d(channels, channels * 2, 15, 2,
                                  groups=channels // 4, padding=_pad(15))),
            weight_norm(nn.Conv1d(channels * 2, channels * 4, 15, 2,
                                  groups=channels // 2, padding=_pad(15))),
            weight_norm(nn.Conv1d(channels * 4, channels * 8, 15, 2,
                                  groups=channels, padding=_pad(15))),
            weight_norm(nn.Conv1d(channels * 8, channels * 8, 15, 2,
                                  groups=channels * 2, padding=_pad(15))),
            weight_norm(nn.Conv1d(channels * 8, channels * 8, 5, 1,
                                  padding=_pad(5))),
        ])
        self.conv_post = weight_norm(
            nn.Conv1d(channels * 8, 1, 3, 1, padding=_pad(3)))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return [(fmap, x)]


class PitchDiscriminator(nn.Module):
    """Multi-scale discriminator for pitch contours.

    Mirrors the audio ``Discriminator`` interface: ``forward(x)`` returns a
    list of ``(feature_maps, score)`` tuples (one per scale), where ``score``
    is the flattened raw logit used for least-squares GAN matching.

    Args:
        channels: base channel count of each scale (default 32).
        n_scales: number of input resolution scales (default 3).
    """

    def __init__(self, channels: int = 32, n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales
        self.scales = nn.ModuleList(
            [PitchScaleDiscriminator(channels) for _ in range(n_scales)])

    def forward(self, x):
        # Accept (B, 1, T) or (B, T) pitch contours.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        outs = []
        for i, scale in enumerate(self.scales):
            if i == 0:
                xi = x
            else:
                k = 2 ** i
                # Skip scales whose pooling would collapse the sequence.
                if x.shape[-1] < k:
                    continue
                xi = F.avg_pool1d(x, kernel_size=k, stride=k)
            outs.extend(scale(xi))
        return outs


if __name__ == '__main__':
    model = PitchDiscriminator()
    x = torch.randn(3, 1, 800)
    print(x.shape)

    output = model(x)
    for features, score in output:
        for feat in features:
            print(feat.shape)
        print(score.shape)

    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print("params:", pytorch_total_params)