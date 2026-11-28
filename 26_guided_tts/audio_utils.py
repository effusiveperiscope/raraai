"""Code tensor -> 48 kHz audio (shared by eval and TensorBoard audio logging)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import shared  # noqa: F401  (puts ../25_transformer on sys.path)
from stage2_wer import code_frame_samples
from modeling.generator import PITCH_LOG_SCALE


@torch.no_grad()
def codes_to_audio(net_g, codes: torch.Tensor, sids: torch.Tensor,
                   lengths=None, return_f0: bool = False):
    """``(B, 10, T)`` codes -> waveforms, **one item at a time**.

    The collator pads ``codes`` to the batch's longest item.  Decoding the
    padded tensor would feed the codec a tail of zero-code frames (an audible
    drone) and return audio longer than the item's plan.  ``lengths[i]`` is the
    item's real code-frame count; each item is decoded with exactly that many
    frames, so the output length is ``lengths[i] * hop`` and nothing after the
    plan is synthesised.
    """
    hop = code_frame_samples(net_g)
    wavs, f0s = [], []
    for i in range(codes.shape[0]):
        T = int(lengths[i]) if lengths is not None else int(codes.shape[-1])
        c = codes[i:i + 1, :, :T]
        spec = c[:, :8].transpose(0, 1).contiguous()
        pitch = c[:, 8:].transpose(0, 1).contiguous()
        audio, f0 = net_g.infer_from_codes(spec, pitch, sids[i:i + 1],
                                           return_pitch=True)
        wavs.append(audio[0, 0, :T * hop].float().cpu().numpy())
        f0s.append(f0[0, :4 * T].float().cpu().numpy())
    if return_f0:
        return wavs, f0s
    return wavs


def _f0_from_pitch_codes(net_g, pitch_codes: torch.Tensor, g: torch.Tensor,
                         T: int) -> torch.Tensor:
    """2 pitch books -> (B, 4T) f0 in Hz, the same path infer_from_codes uses."""
    pq = net_g._decode_from_indices(net_g.pitch_codec, pitch_codes, g, 0)
    pq_up = net_g.pitch_upsample(pq)[..., : 4 * T]
    f0_norm = net_g.pitch_out_proj(pq_up).squeeze(1)
    uv = net_g.pitch_uv_proj(pq_up).squeeze(1)
    v = (torch.sigmoid(uv) >= net_g.pitch_uv_threshold).float()
    return torch.expm1(torch.clamp(f0_norm, min=0.0) * PITCH_LOG_SCALE) * v


@torch.no_grad()
def zq_to_audio(net_g, zq: torch.Tensor, pitch_codes: torch.Tensor,
                sids: torch.Tensor, lengths=None, return_f0: bool = False):
    """Continuous spec latent ``(B,1024,T)`` + pitch codes -> audio.

    This is the continuous alternative to :func:`codes_to_audio`: the spec
    latent goes straight into the frozen codec decoder (which is defined on
    ``zq = sum of codebook embeddings``), with no quantisation step.  Decoding
    is per item at its own ``lengths[i]`` for the same padding reason.
    """
    hop = code_frame_samples(net_g)
    wavs, f0s = [], []
    for i in range(zq.shape[0]):
        T = int(lengths[i]) if lengths is not None else int(zq.shape[-1])
        g = net_g.spk_emb(sids[i:i + 1])
        # zq_pred is (B, T, 1024); the codec decoder wants (B, 1024, T)
        z = zq[i:i + 1, :T, :].transpose(1, 2).float().contiguous()
        yq = net_g.upsample(net_g.codec.decode_latent(z, g))
        pc = pitch_codes[i:i + 1, :, :T].transpose(0, 1).contiguous()
        f0 = _f0_from_pitch_codes(net_g, pc, g, T)
        audio = net_g._decode_wave(g, yq, f0, yq.shape[-1] * net_g.dec.upp, 0)
        wavs.append(audio[0, 0, : T * hop].float().cpu().numpy())
        f0s.append(f0[0].float().cpu().numpy())
    if return_f0:
        return wavs, f0s
    return wavs


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
