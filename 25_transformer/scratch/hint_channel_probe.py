#!/usr/bin/env python3
"""Per-channel hint liveness + counterfactual level sweep on a checkpoint.

Answers two questions that ``val_loss`` / ``val_hint_gap`` cannot:

1. **Is each channel load-bearing?**  Re-score the *same* val batches with one
   channel at a time forced to its trained "not provided" state (level 0 /
   ``hint_sem_valid=False``).  ``val_hint_gap`` drops all five at once, so one
   live channel hides four dead ones.  Also reports ``all`` (the null state
   generation runs in) and ``omit`` (no hint vectors at all -- the pre-Phase-0
   inference path), so their difference is that bug's cost in nats.

2. **Does a channel behave like a knob?**  Sweep one channel's level over its
   whole table for the same batch and report the CE profile.  A flat profile
   means the table is decorative *for the loss* -- which is expected for a
   style channel (the real knob test is the audio-level sweep, scored with the
   DSP intensity model / RMVPE).

Nothing is written; the checkpoint is only read.

Usage::

    python scratch/hint_channel_probe.py --checkpoint checkpoints_stage2/<exp>/last.ckpt
    python scratch/hint_channel_probe.py --checkpoint ... --batches 4 \
        --sweep intensity,register --device cpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# fairseq (pulled in by modeling.generator via the vendored NSF decoder) sets
# the root logger to DEBUG at import time -- see AGENTS.md 6.18.
logging.basicConfig(level=logging.WARNING)

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_submodule_prefix  # noqa: E402
from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402


def load_checkpoint(net, path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    prefix = "net." if any(k.startswith("net.") for k in sd) else ""
    load_submodule_prefix(net, prefix, sd, quiet=True)
    return ck.get("epoch"), ck.get("global_step")


def hint_kwargs(config):
    tc = config.transformer
    return dict(
        hints=bool(tc.get("hints", False)),
        hints_json=tc.get("hints_json", None),
        hint_dropout=dict(tc.get("hint_dropout", {}) or {}),
        hint_null_all=0.0,
        hint_pitch_edges=tc.get("hint_pitch_edges", None),
        n_slot_buckets=int(tc.get("n_slot_buckets", 8) or 8),
    )


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def ce(net, batch, **override):
    """Teacher-forced CE of ``batch`` (with optional ``hints=`` overrides)."""
    return float(net.compute_batch_loss(batch, epoch=None, diagnostics=False,
                                        **override)["loss"])


@torch.no_grad()
def channel_deltas(net, batches):
    """Mean per-channel CE deltas over ``batches``."""
    acc = {}
    for batch in batches:
        for key, value in net.hint_channel_deltas(batch).items():
            acc[key] = acc.get(key, 0.0) + float(value)
    n = max(len(batches), 1)
    return {k: v / n for k, v in acc.items()}


@torch.no_grad()
def level_sweep(net, batch, channel):
    """CE as one channel's level goes over its whole table (0 = absent)."""
    key, n_levels = {
        "intensity": ("hint_int", net.n_intensity_levels),
        "register": ("hint_reg", net.n_register_levels),
        "pitch": ("hint_pitch", net.n_pitch_levels),
        "slot": ("hint_slot", net.n_slot_buckets),
    }[channel]
    if batch.get(key) is None:
        return []
    out = []
    for level in range(n_levels + 1):
        shifted = dict(batch)
        shifted[key] = torch.full_like(batch[key], level)
        out.append((level, ce(net, shifted)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_filelist", default=None,
                    help="default: config.train.val_filelist")
    ap.add_argument("--batches", type=int, default=2,
                    help="validation batches to probe (default 2)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sweep", default="intensity,register",
                    help="comma separated channels to level-sweep (or 'none')")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))

    net = CodeTransformer(config).to(device)
    epoch, step = load_checkpoint(net, args.checkpoint)
    net.eval()
    print(f"{args.checkpoint}  (epoch {epoch}, step {step})")
    print(net.describe() + "\n")
    if not net.hints:
        print("This checkpoint has no hint tables -- nothing to probe.")
        return 0

    ds = CodeDataset(args.val_filelist or config.train.val_filelist,
                     is_train=False,
                     max_frames=config.train.get("max_frames", None),
                     max_phone_length=config.train.get("max_phone_length", None),
                     speaker_map=config.train.get("speaker_map", None),
                     **hint_kwargs(config))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=CodeCollator(config=config), num_workers=0)
    batches = []
    for batch in loader:
        batch = to_device(batch, device)
        if batch.get("hint_int") is None:
            print("The val items carry no .cond hints -- run hints/build.py "
                  "over this split first.")
            return 1
        batches.append(batch)
        if len(batches) >= args.batches:
            break

    deltas = channel_deltas(net, batches)
    base = deltas.pop("base", 0.0)
    print(f"teacher-forced CE with all hints: {base:.4f} nats")
    print(f"{'channel':<10} {'CE if dropped':>14} {'delta':>9}")
    for name in [*net.HINT_CHANNEL_KEYS, "all", "omit"]:
        if name not in deltas:
            continue
        note = {"all": "   <- trained null state (generation default)",
                "omit": "  <- no hint vectors (pre-Phase-0 path)"}.get(name, "")
        print(f"{name:<10} {base + deltas[name]:>14.4f} {deltas[name]:>+9.4f}"
              f"{note}")
    if "all" in deltas and "omit" in deltas:
        print(f"\nall - omit = {deltas['all'] - deltas['omit']:+.4f} nats "
              f"(the Phase-0 generation bug's cost)")

    sweep = [s.strip() for s in args.sweep.split(",")
             if s.strip() and s.strip() != "none"]
    for channel in sweep:
        rows = level_sweep(net, batches[0], channel)
        if not rows:
            continue
        best = min(rows, key=lambda r: r[1])
        print(f"\n{channel} level sweep (teacher-forced CE, batch 1 of "
              f"{len(batches)}):")
        print("  " + "  ".join(f"L{lv}:{c - base:+.4f}" for lv, c in rows))
        print(f"  argmin L{best[0]} ({best[1] - base:+.4f} nats vs the real "
              f"levels)")
        print("  (a flat profile means the table is decorative *for the "
              "loss*; a style\n   channel's real test is the audio sweep, not "
              "CE)")

    print("\nCE is not the KPI for a style channel: for the knob itself, sweep "
          "the level\nat generation time and score the decoded audio (DSP "
          "intensity model / RMVPE),\nand check WER + repetition at the "
          "extremes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

