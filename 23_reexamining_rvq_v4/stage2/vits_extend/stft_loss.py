# -*- coding: utf-8 -*-
# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules with V/UV-aware weighting."""

import torch
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def make_frame_weight(voiced_mask_mel, T_stft, unvoiced_weight, device):
    """Resample a voiced mask from mel-frame resolution to STFT-frame resolution
    and convert to a per-frame scalar weight tensor.

    Args:
        voiced_mask_mel (Tensor | None): (B, T_mel) binary mask, 1=voiced, 0=unvoiced.
            If None, returns uniform weights of 1.0.
        T_stft (int): Number of STFT frames to resample to.
        unvoiced_weight (float): Weight assigned to unvoiced frames (voiced frames get 1.0).
        device: torch device.

    Returns:
        Tensor: (B, T_stft, 1) weight tensor, ready to broadcast over freq bins.
    """
    if voiced_mask_mel is None:
        return None

    # (B, T_mel) -> (B, 1, T_mel) for interpolate
    mask = voiced_mask_mel.float().unsqueeze(1)
    # nearest-neighbour resample to STFT frame count
    mask = F.interpolate(mask, size=T_stft, mode='nearest')  # (B, 1, T_stft)
    mask = mask.squeeze(1)                                    # (B, T_stft)

    # 1.0 for voiced, unvoiced_weight for unvoiced
    weight = mask + (1.0 - mask) * unvoiced_weight            # (B, T_stft)
    return weight.unsqueeze(2).to(device)                     # (B, T_stft, 1)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss, optionally weighted per frame."""

    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag, frame_weight=None):
        """
        Args:
            x_mag (Tensor): (B, T, F) predicted magnitude spectrogram.
            y_mag (Tensor): (B, T, F) target magnitude spectrogram.
            frame_weight (Tensor | None): (B, T, 1) per-frame weights.

        Returns:
            Tensor: Scalar loss.
        """
        diff = y_mag - x_mag  # (B, T, F)

        if frame_weight is None:
            # Original behaviour: plain Frobenius norm
            return torch.norm(diff, p="fro") / torch.norm(y_mag, p="fro")

        # Weighted Frobenius: scale each frame's contribution before norming.
        # sqrt(w) on the residual is equivalent to w on the squared residual,
        # which is the correct generalisation of the Frobenius norm with weights.
        sqrt_w = frame_weight.sqrt()          # (B, T, 1) — broadcasts over F
        num = torch.norm(diff * sqrt_w, p="fro")
        den = torch.norm(y_mag * sqrt_w, p="fro")
        return num / (den + 1e-8)


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss, optionally weighted per frame."""

    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag, frame_weight=None):
        """
        Args:
            x_mag (Tensor): (B, T, F) predicted magnitude spectrogram.
            y_mag (Tensor): (B, T, F) target magnitude spectrogram.
            frame_weight (Tensor | None): (B, T, 1) per-frame weights.

        Returns:
            Tensor: Scalar loss.
        """
        diff = torch.abs(torch.log(y_mag) - torch.log(x_mag))  # (B, T, F)

        if frame_weight is None:
            return diff.mean()

        # Weighted mean: sum(w * loss) / sum(w * F)  — keeps units consistent
        num = (diff * frame_weight).sum()
        den = frame_weight.expand_as(diff).sum() + 1e-8
        return num / den


class STFTLoss(torch.nn.Module):
    """STFT loss module with optional V/UV frame weighting."""

    def __init__(self, device, fft_size=1024, shift_size=120, win_length=600,
                 window="hann_window", unvoiced_weight=0.1):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.unvoiced_weight = unvoiced_weight
        self.device = device
        self.window = getattr(torch, window)(win_length).to(device)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y, voiced_mask_mel=None):
        """
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
            voiced_mask_mel (Tensor | None): (B, T_mel) binary V/UV mask at mel-hop
                resolution (1=voiced, 0=unvoiced). Pass None to disable weighting
                and reproduce original behaviour exactly.

        Returns:
            Tensor: Spectral convergence loss.
            Tensor: Log STFT magnitude loss.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)

        # Build per-frame weight at this STFT's temporal resolution
        T_stft = x_mag.shape[1]
        frame_weight = make_frame_weight(
            voiced_mask_mel, T_stft, self.unvoiced_weight, self.device
        )

        sc_loss  = self.spectral_convergence_loss(x_mag, y_mag, frame_weight)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag, frame_weight)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss with optional V/UV frame weighting."""

    def __init__(self, device, resolutions, window="hann_window", unvoiced_weight=0.1):
        """
        Args:
            resolutions (list): List of (fft_size, hop_size, win_length) tuples.
            window (str): Window function type.
            unvoiced_weight (float): Loss weight for unvoiced frames (0–1).
                1.0 = uniform (original behaviour). 0.1 = unvoiced frames
                contribute 10% as much as voiced frames.
        """
        super().__init__()
        self.stft_losses = torch.nn.ModuleList([
            STFTLoss(device, fs, ss, wl, window, unvoiced_weight)
            for fs, ss, wl in resolutions
        ])

    def forward(self, x, y, voiced_mask_mel=None):
        """
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
            voiced_mask_mel (Tensor | None): (B, T_mel) V/UV mask at mel-hop resolution.
                Derived from F0: (f0 > 0).float(). Pass None for original behaviour.

        Returns:
            Tensor: Mean spectral convergence loss across resolutions.
            Tensor: Mean log STFT magnitude loss across resolutions.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y, voiced_mask_mel)
            sc_loss  += sc_l
            mag_loss += mag_l

        sc_loss  /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss