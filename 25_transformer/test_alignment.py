#!/usr/bin/env python3
"""CPU smoke test for alignment supervision (``transformer.align_supervision``).

Guards the four things that make guided attention work:

1. ``align_ce`` starts at ``log(n_phones)`` -- the attention map is uniform, so
   the loss is the target's cross-entropy against a uniform distribution;
2. the total loss is exactly ``code_ce + align_weight * align_ce`` (the term is
   additive, not folded into some other average);
3. a few optimiser steps on a *fixed* batch actually move the map --
   ``align_entropy`` falls well below 1.0.  This is the one that matters: it
   proves the gradient reaches the cross-attention projections and that the
   uniform-attention regime is escapable, which is what the earlier prototypes
   (a duration side channel, a word-progress head) failed to show;
4. no schedule -> no capture and no extra loss, and ``generate`` never enables
   capture (it must stay on the fused path).

Run: ``python test_alignment.py``
"""

from __future__ import annotations

import logging
import math

logging.basicConfig(level=logging.WARNING)

import torch
from omegaconf import OmegaConf

from modeling.transformer import CodeTransformer

OK = []
FAIL = []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make_model(align=True, layers=(0,)):
    cfg = OmegaConf.create({
        "codec": {"codebook_num": 2, "codebook_size": 32},
        "pitch_codec": {"codebook_num": 1, "codebook_size": 32},
        "transformer": {
            "hidden_size": 32, "n_heads": 2, "encoder_layers": 1,
            "decoder_layers": 2, "dim_feedforward": 64, "dropout": 0.0,
            "max_positions": 256, "phone_vocab_size": 64, "n_speakers": 16,
            "eos_token": True, "rope": True,
            "align_supervision": bool(align),
            "align_layers": list(layers),
            "align_sigma": 1.0,
            "align_weight": 0.5 if align else 0.0,
        },
    })
    torch.manual_seed(0)
    return CodeTransformer(cfg).eval()


def make_batch(B=2, N=8, T=10):
    torch.manual_seed(1)
    n_ph = torch.tensor([N, N - 2])
    phones = torch.full((B, N), 0, dtype=torch.long)
    for b in range(B):
        phones[b, :n_ph[b]] = torch.randint(1, 40, (int(n_ph[b]),))
    progress = torch.zeros(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        rate = n_ph[b] / T
        t = torch.arange(T, dtype=torch.float32)
        progress[b] = ((t * rate).floor() + 0.5) / n_ph[b]
        mask[b] = True
    return {
        "phone_ids": phones,
        "phone_lengths": n_ph,
        "sid": torch.tensor([1, 2]),
        "spec_codes": torch.randint(0, 32, (B, 2, T)),
        "pitch_codes": torch.randint(0, 32, (B, 1, T)),
        "code_lengths": torch.tensor([T, T - 2]),
        "complete": torch.tensor([True, False]),
        "dur_prog": progress,
        "dur_prog_mask": mask,
    }


def test_init_and_additivity():
    print("loss at init / additivity")
    net = make_model()
    batch = make_batch()
    with torch.no_grad():
        plain = net.compute_batch_loss({k: v for k, v in batch.items()
                                        if not k.startswith("dur_")}, None, True)
        out = net.compute_batch_loss(batch, None, True)
    n_ph = float(batch["phone_lengths"].float().mean())
    check("align_ce is reported", "align_ce" in out)
    check("align_ce starts near log(n_phones)",
          abs(float(out["align_ce"]) - math.log(n_ph)) < 0.6,
          f"{float(out['align_ce']):.3f} vs log {math.log(n_ph):.3f}")
    check("align_entropy starts near 1 (uniform)",
          float(out["align_entropy"]) > 0.85, f"{float(out['align_entropy']):.3f}")
    want = float(plain["loss"]) + net.align_weight * float(out["align_ce"])
    check("loss == code_ce + w * align_ce",
          abs(float(out["loss"]) - want) < 1e-4,
          f"{float(out['loss']):.4f} vs {want:.4f}")
    check("code_ce is the pure code term",
          abs(float(out["code_ce"]) - float(plain["loss"])) < 1e-5)

    no_sched = net.compute_batch_loss({k: v for k, v in batch.items()
                                       if not k.startswith("dur_")}, None, True)
    check("no schedule -> no align term", "align_ce" not in no_sched)
    check("no schedule -> loss unchanged",
          abs(float(no_sched["loss"]) - float(plain["loss"])) < 1e-5)
    check("no schedule -> nothing captured",
          net.decoder.decoder.layers[0].multihead_attn.last_attn is None)


def test_can_learn_alignment():
    print("a few steps actually move the attention")
    net = make_model()
    net.train()
    batch = make_batch()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=3e-3)
    with torch.no_grad():
        start = float(net.compute_batch_loss(batch, None, False)["align_entropy"])
    for _ in range(80):
        out = net.compute_batch_loss(batch, None, False)
        opt.zero_grad()
        # only the alignment term: this is a test of the plumbing, not of the
        # code model
        (net.align_weight * out["align_ce"]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        end = float(net.compute_batch_loss(batch, None, False)["align_entropy"])
    check("entropy falls from uniform", end < 0.75, f"{start:.3f} -> {end:.3f}")
    check("it moved a lot", start - end > 0.2, f"delta {start - end:+.3f}")


def test_generate_does_not_capture():
    print("generate stays on the fast path")
    net = make_model()
    layer = net.decoder.decoder.layers[0].multihead_attn
    net.generate(torch.randint(1, 40, (1, 6)), torch.tensor([6]),
                 torch.tensor([1]), max_frames=4, temperature=0.9)
    check("no capture during generate", layer.last_attn is None)
    check("capture flag left off", layer.capture is False)


def test_bad_config():
    print("config guards")
    try:
        make_model(align=True, layers=(0, 99))
        check("out-of-range layer raises", False)
    except ValueError:
        check("out-of-range layer raises", True)


if __name__ == "__main__":
    test_init_and_additivity()
    test_can_learn_alignment()
    test_generate_does_not_capture()
    test_bad_config()
    print(f"\n{len(OK)} passed, {len(FAIL)} failed")
    if FAIL:
        raise SystemExit(1)
    print("all alignment checks passed")
