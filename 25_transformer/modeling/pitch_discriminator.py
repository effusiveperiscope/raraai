"""Patch discriminators for f0 contours.

Stage 1 trains the pitch codec with ``pitch_dec_loss``, a masked MSE on the
log-normalized f0 contour (``modeling/generator.PITCH_LOG_SCALE``). Two failure
modes survive an L2 reconstruction loss:

* the code grid runs at ``frame_rate / down_factor`` (25 Hz with the default
  ``down_factor=4``), so the decoded contour is assembled from 40 ms tiles. If
  the upsample that returns to the frame rate has ``kernel_size == stride``
  (the historical setting), one code only ever affects its own four frames:
  neighbouring tiles are decoded by unrelated linear maps and are not asked to
  join up, which is what makes the contour look jagged;
* L2 regression moves towards the conditional mean, so what survives is the
  average trajectory rather than the shape of the movement.

Both are **slope** statistics, not level statistics. A natural f0 contour has
small, smoothly varying first differences; a tile-wise decoded one has steps at
the tile edges and runs of near-zero slope in between. The critics here are
therefore patch discriminators fed with the *first difference* of the contour,
and they are deliberately small: a few hundred milliseconds of context is
enough to recognise the tile pattern, while a wider receptive field turns the
critic into a global statistics matcher that merely re-does what the
reconstruction loss already optimises.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Number of input channels assembled in ``PitchPatchDiscriminator.forward``:
# log-f0, first difference of log-f0, voicing mask.
_IN_CHANNELS = 3


class PitchPatchDiscriminator(nn.Module):
    """Single patch critic over short windows of a pitch contour.

    Input channels (built in :meth:`forward`):

    * ``log_f0`` -- log-normalized contour, 0 on unvoiced frames;
    * ``slope``  -- first difference of ``log_f0``, zeroed across
      voiced/unvoiced boundaries (a V/UV edge is not a pitch discontinuity)
      and scaled up by ``delta_scale``;
    * ``voiced`` -- the voicing mask itself, so the critic can tell a V/UV edge
      from a pitch jump instead of smearing them together.

    With the default ``strides=(1, 2, 1, 2)`` the stack has a receptive field of
    33 frames (~330 ms at the 10 ms pitch frame rate), an output stride of 4
    frames and ~52 k parameters. That is small on purpose: the point is to see
    the local slope pattern, not to summarise the whole utterance.

    Args:
        channels: base channel count.
        kernel_size: conv kernel size for the stride stack (must be odd).
        dilation: dilation applied to the whole stride stack. Larger dilations
            see longer time scales without changing the output stride.
        delta_scale: gain applied to the slope channel. Log-normalized f0 lives
            in ~[0, 1] while per-frame slopes are ~1e-3, so the slope is
            multiplied by something on the order of ``1 / typical_peak_slope``
            (default 50 maps a fast ~1e-2 hop per frame to ~0.5) to keep both
            channels in a comparable range.
        strides: per-layer strides; their product is the output stride.
    """

    def __init__(self,
                 channels: int = 32,
                 kernel_size: int = 5,
                 dilation: int = 1,
                 delta_scale: float = 50.0,
                 strides: tuple[int, ...] = (1, 2, 1, 2)):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.dilation = int(dilation)
        self.delta_scale = float(delta_scale)
        self.output_stride = 1
        for s in strides:
            self.output_stride *= int(s)

        # (in) -> C -> 2C -> 2C -> 2C. Full channel mixing: at this size the
        # grouped convs of a HiFi-GAN MSD would only bridge ~4 channels per
        # layer, far too narrow to recognise a step pattern.
        dims = [_IN_CHANNELS, channels, channels * 2, channels * 2,
                channels * 2]
        self.convs = nn.ModuleList()
        for i, s in enumerate(strides):
            self.convs.append(weight_norm(nn.Conv1d(
                dims[i], dims[i + 1], kernel_size, stride=int(s),
                dilation=self.dilation,
                padding=self.dilation * (kernel_size - 1) // 2)))
        self.conv_post = weight_norm(nn.Conv1d(channels * 2, 1, 3, padding=1))

    def _slope(self, x: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
        """First difference of the contour, valid only inside voiced runs.

        ``x`` holds 0 on unvoiced frames, so a plain difference would turn
        every V/UV boundary into an enormous artificial step. Multiplying by
        the product of the two adjacent mask values keeps the slope at 0
        wherever either endpoint is unvoiced.
        """
        d = torch.zeros_like(x)
        d[:, 1:] = (x[:, 1:] - x[:, :-1]) * voiced[:, 1:] * voiced[:, :-1]
        return d * self.delta_scale

    @staticmethod
    def _mask_on_grid(voiced: torch.Tensor, length: int) -> torch.Tensor:
        """Downsample a (B, T) voicing mask to the critic's output grid."""
        if voiced.shape[-1] == length:
            return voiced
        m = F.interpolate(voiced.unsqueeze(1), size=length, mode='area')
        return (m.squeeze(1) >= 0.5).float()

    def forward(self, x: torch.Tensor,
                voiced: torch.Tensor | None = None):
        """Score a pitch contour.

        Args:
            x: ``(B, 1, T)`` or ``(B, T)`` log-normalized f0, 0 == unvoiced.
            voiced: optional ``(B, T)`` / ``(B, 1, T)`` voicing mask. When
                given, only logits whose output window is voiced are returned,
                so the caller's ``score.pow(2).mean()`` is an honest voiced-only
                mean instead of one diluted by silence.

        Returns:
            ``[(feature_maps, score)]`` -- a one-element list, mirroring the
            audio ``Discriminator`` interface. ``score`` holds raw logits
            flattened over batch and time (voiced positions only when
            ``voiced`` is supplied). It is 1-D rather than 2-D so that a mean
            over it works for batches of variable-length voiced regions.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(
                "expected (B, T) or (B, 1, T) pitch contour, got "
                f"{tuple(x.shape)}")

        # Run in fp32 regardless of the trainer's mixed precision: bf16's ulp
        # at a log-f0 value of ~0.77 is ~2e-3 (about 24 cents), which is far
        # coarser than the per-frame movement this critic exists to judge.
        x = x.float()
        if voiced is None:
            voiced = (x > 0).float()
        else:
            voiced = voiced.float()
            if voiced.dim() == 3:
                voiced = voiced.squeeze(1)
        if voiced.shape != x.shape:
            raise ValueError(
                "voiced mask must match the contour shape, got "
                f"{tuple(voiced.shape)} vs {tuple(x.shape)}")

        with torch.autocast(device_type=x.device.type, enabled=False):
            feat = torch.stack([x, self._slope(x, voiced), voiced], dim=1)
            fmap = []
            for conv in self.convs:
                feat = F.leaky_relu(conv(feat), 0.1)
                fmap.append(feat)
            logits = self.conv_post(feat).squeeze(1)  # (B, L)
            fmap.append(logits.unsqueeze(1))

            score = logits.flatten()
            sel = self._mask_on_grid(voiced, logits.shape[-1]).reshape(-1).bool()
            if sel.any():
                score = logits.reshape(-1)[sel]
        return [(fmap, score)]


class PitchDiscriminator(nn.Module):
    """Multi-branch patch discriminator for pitch contours.

    Branch ``i`` runs a :class:`PitchPatchDiscriminator` with dilation
    ``2 ** i``, so the branches look at roughly 0.33 s / 0.57 s / 1.05 s of
    contour instead of three nearly identical near-global views (the previous
    ``avg_pool1d`` pyramid had a 3.2 s receptive field, so scales 1 and 2 both
    covered the entire 8 s training segment).

    Mirrors the audio ``Discriminator`` interface: ``forward(x)`` returns a list
    of ``(feature_maps, score)`` tuples, one per branch, where ``score`` is the
    flattened raw logit used for least-squares GAN matching.

    Args:
        channels: base channel count per branch (default 32).
        n_scales: number of branches; branch ``i`` uses dilation ``2 ** i``.
        kernel_size: conv kernel size of the stride stack.
        delta_scale: gain on the slope channel (see
            :class:`PitchPatchDiscriminator`).
    """

    def __init__(self,
                 channels: int = 32,
                 n_scales: int = 3,
                 kernel_size: int = 5,
                 delta_scale: float = 50.0):
        super().__init__()
        if n_scales < 1:
            raise ValueError(f"n_scales must be >= 1, got {n_scales}")
        self.n_scales = int(n_scales)
        self.scales = nn.ModuleList([
            PitchPatchDiscriminator(
                channels=channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                delta_scale=delta_scale)
            for i in range(self.n_scales)])

    def forward(self, x: torch.Tensor,
                voiced: torch.Tensor | None = None):
        """Score ``x`` with every branch; see the branch ``forward`` docstring."""
        outs = []
        for branch in self.scales:
            outs.extend(branch(x, voiced=voiced))
        return outs


if __name__ == '__main__':
    model = PitchDiscriminator()
    # 8 s of contour at the 10 ms pitch frame rate, with a voiced/unvoiced
    # pattern plus a smooth reference and its 25 Hz code-grid rasterisation.
    T = 800
    voiced = torch.ones(3, T)
    voiced[:, 200:260] = 0.0
    voiced = F.max_pool1d(voiced.unsqueeze(1), 5, 1, 2).squeeze(1)
    real = (0.6 + 0.05 * torch.sin(torch.arange(T) * 0.3)).unsqueeze(0)
    real = real.expand(3, T) * voiced
    raster = real.reshape(3, T // 4, 4).mean(-1).repeat_interleave(4, dim=1)
    raster = raster * voiced

    for name, contour in (('smooth', real), ('raster', raster)):
        out = model(contour, voiced=voiced)
        print(name, 'branches:', len(out),
              '| scores:', [tuple(score.shape) for _, score in out])

    # The slope channel must not fire across V/UV boundaries.
    slope = model.scales[0]._slope(real, voiced)
    edges = (voiced.diff(dim=1) != 0)
    print('max |slope| at V/UV edges:', float(slope[:, 1:][edges].abs().max()))
    interior = (voiced[:, 1:] * voiced[:, :-1]).bool()
    print('max |slope| inside voiced runs:',
          float(slope[:, 1:][interior].abs().max()))

    # Without a mask the score keeps every position (`x > 0` is only derived
    # when no mask is given, so use a strictly positive contour here).
    print('unmasked score:', tuple(model(real + 1.0)[0][1].shape))

    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print("params:", pytorch_total_params)

