"""CPU guard for ``renderer.py``: shapes, length exactness, backward, both
pitch-head modes.  Run ``python test_renderer.py``.
"""

from __future__ import annotations

import torch

from renderer import GuidedRenderer, renderer_loss


def _batch(B=2, T=12, N=6):
    torch.manual_seed(0)
    P = 4 * T
    phone_ids = torch.randint(1, 32, (B, N))
    phone_mask = torch.ones(B, N, dtype=torch.bool)
    if B > 1:
        phone_mask[1, -1] = False
        phone_ids[1, -1] = 0
    # each code frame gets four subframes; use -1 on the tail of item 1
    phone_index = torch.full((B, P), -1, dtype=torch.long)
    for b in range(B):
        n = N if b == 0 else N - 1
        idx = torch.arange(n).repeat_interleave(P // n)[:P]
        phone_index[b, :idx.numel()] = idx
    feats = torch.randn(B, T, 6)
    sid = (torch.arange(B) % 8).long()
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    if B > 1:
        frame_mask[1, 8:] = False
    spec = torch.randint(0, 64, (B, 8, T))
    pitch = torch.randint(0, 128, (B, 2, T))
    frame_phone = torch.randint(1, 32, (B, T))
    frame_phone[~frame_mask] = -1
    f0 = torch.zeros(B, P)
    f0_valid = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        n = int(frame_mask[b].sum())
        f0[b, :4 * n] = 120.0 + 20 * torch.rand(4 * n)
        f0_valid[b, :4 * n] = True
    return {
        "phone_ids": phone_ids, "phone_mask": phone_mask,
        "phone_index": phone_index, "frame_feats": feats,
        "sid": sid, "frame_mask": frame_mask,
        "spec_codes": spec, "pitch_codes": pitch,
        "f0": f0, "f0_valid": f0_valid,
        "frame_phone": frame_phone,
    }


KEYS = ("phone_ids", "phone_mask", "phone_index", "frame_feats", "sid",
        "frame_mask")


def _inputs(batch):
    return {k: batch[k] for k in KEYS}


def test_shapes_and_backward():
    batch = _batch()
    model = GuidedRenderer(vocab=32, num_speakers=8, width=64,
                           phone_layers=2, frame_layers=3, dropout=0.0,
                           phone_attn_layers=1, frame_attn_layers=2, nhead=4,
                           phone_aux=True)
    out = model(**_inputs(batch), pitch_targets=batch["pitch_codes"])
    B, T = batch["frame_mask"].shape
    assert out["spec_logits"].shape == (B, T, 8, 64), out["spec_logits"].shape
    assert len(out["pitch_logits"]) == 2
    assert out["pitch_logits"][0].shape == (B, T, 128)
    assert out["pitch_logits"][1].shape == (B, T, 128)
    assert out["f0_pred"].shape == (B, T)
    assert out["phone_logits"].shape == (B, T, 32)

    losses = renderer_loss(out, batch, phone_weight=0.5)
    assert torch.isfinite(losses["loss"]), losses
    assert "phone_ce" in losses and torch.isfinite(losses["phone_ce"])
    losses["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
    return B, T


def test_zq_head():
    batch = _batch()
    model = GuidedRenderer(vocab=32, num_speakers=8, width=64,
                           phone_layers=1, frame_layers=2, dropout=0.0,
                           phone_attn_layers=1, frame_attn_layers=1, nhead=4,
                           zq_aux=True, zq_dim=256)
    out = model(**_inputs(batch), pitch_targets=batch["pitch_codes"])
    B, T = batch["frame_mask"].shape
    assert out["zq_pred"].shape == (B, T, 256)
    zqt = torch.randn(B, T, 256)
    losses = renderer_loss(out, batch, zq_target=zqt, zq_weight=1.0)
    assert "zq_l1" in losses and torch.isfinite(losses["zq_l1"])
    losses["loss"].backward()


def test_inference_and_parallel_ablation():
    batch = _batch()
    model = GuidedRenderer(vocab=32, num_speakers=8, width=64,
                           phone_layers=1, frame_layers=2, dropout=0.0,
                           phone_attn_layers=1, frame_attn_layers=1, nhead=4)
    with torch.no_grad():
        out = model(**_inputs(batch), pitch_targets=None, temperature=0.0)
    assert out["pitch_logits"][0].shape == (2, 12, 128)
    assert out["pitch_codes"].shape == (2, 2, 12)

    abl = GuidedRenderer(vocab=32, num_speakers=8, width=64,
                         phone_layers=1, frame_layers=2, dropout=0.0,
                         depth_ar_pitch=False)
    with torch.no_grad():
        out2 = abl(**_inputs(batch), pitch_targets=None, temperature=1.0, top_k=8)
    assert len(out2["pitch_logits"]) == 2
    assert out2["pitch_logits"][0].shape == (2, 12, 128)


def test_length_is_exact():
    """The output covers exactly T frames; padding is masked, never appended."""
    batch = _batch(B=1, T=17, N=5)
    model = GuidedRenderer(vocab=32, num_speakers=8, width=32,
                           phone_layers=1, frame_layers=1, dropout=0.0,
                           phone_attn_layers=1, frame_attn_layers=1, nhead=4)
    with torch.no_grad():
        out = model(**_inputs(batch), pitch_targets=None)
    assert out["spec_logits"].shape[1] == 17
    assert out["f0_pred"].shape[1] == 17


if __name__ == "__main__":
    test_shapes_and_backward()
    print("ok  test_shapes_and_backward")
    test_zq_head()
    print("ok  test_zq_head")
    test_inference_and_parallel_ablation()
    print("ok  test_inference_and_parallel_ablation")
    test_length_is_exact()
    print("ok  test_length_is_exact")
    m = GuidedRenderer(vocab=1024, num_speakers=8192, width=512,
                       phone_layers=4, frame_layers=6,
                       phone_attn_layers=1, frame_attn_layers=2)
    n = sum(p.numel() for p in m.parameters())
    print(f"default 512/4+6 + attn 1/2 renderer: {n / 1e6:.1f} M params")
    m = GuidedRenderer(vocab=1024, num_speakers=8192, width=768,
                       phone_layers=8, frame_layers=8)
    n = sum(p.numel() for p in m.parameters())
    print(f"       768/8+8 renderer: {n / 1e6:.1f} M params")
    print("renderer tests passed")
