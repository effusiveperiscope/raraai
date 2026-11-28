"""Code tensor -> 48 kHz audio (shared by eval and TensorBoard audio logging)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stage2_wer import code_frame_samples


@torch.no_grad()
def codes_to_audio(net_g, codes: torch.Tensor, sids: torch.Tensor,
                   return_f0: bool = False):
    """``(B, 10, T)`` codes (8 spec + 2 pitch) -> list of waveforms."""
    spec = codes[:, :8].transpose(0, 1).contiguous()
    pitch = codes[:, 8:].transpose(0, 1).contiguous()
    audio, f0 = net_g.infer_from_codes(spec, pitch, sids, return_pitch=True)
    hop = code_frame_samples(net_g)
    out = [audio[i, 0, : codes.shape[-1] * hop].float().cpu().numpy()
           for i in range(codes.shape[0])]
    if return_f0:
        return out, f0.float().cpu().numpy()
    return out


def original_wave(phone_path: str | Path) -> np.ndarray | None:
    """The stage-1 ``.wave`` that corresponds to a stage-2 ``.phone_ids`` path."""
    s = str(phone_path)
    if "_stage2/" in s:
        s = s.replace("_stage2/", "_stage1/")
    p = Path(s[: -len(".phone_ids")] + ".wave")
    if not p.exists():
        return None
    try:
        w = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    return torch.as_tensor(w).float().reshape(-1).numpy()
