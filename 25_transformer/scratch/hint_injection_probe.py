#!/usr/bin/env python3
"""Where does a clip-level hint die: the memory, the logits, or the learning?

A hint can fail for three different reasons, and the CE/knob numbers cannot tell
them apart on their own:

1. **The injection site never delivers it.**  ``PhonemeEncoder._temporal_norm``
   standardises each dimension over time, so a per-dimension constant added
   *before* it cancels (HINTS_V1_PLAN.md §3).  ``hint_film_memory`` exists to
   inject *after* the norm; if it does not enlarge the memory shift, the site is
   the problem.
2. **It reaches the memory but the decoder ignores it.**  A large memory shift
   with unchanged logits/CE means the decoder is not using the channel.
3. **It reaches the logits but only as a softmax-invariant offset.**  Softmax
   is blind to adding a constant across the vocabulary, so a shift that is
   mostly "the same number on every class" changes nothing.  ``nonuniform``
   below is the fraction of the logit shift that *can* change the distribution
   (0 = pure constant offset, 1 = fully class-dependent).

This script reports, per checkpoint and per level of one channel (the item's
other channels stay at their real values, so level 0 is "this channel absent"):

``mem``
    ``||m(level) - m(0)|| / ||m(0)||`` on the post-norm encoder memory -- the
    tensor the decoder cross-attends to.
``logit``
    The same relative L2 shift on the decoder output, averaged over codebooks.
``nonunif``
    Fraction of that logit shift which survives the per-position softmax
    invariance (see 3 above).
``CE``
    Teacher-forced loss at that level (the standard weighted loss), so the
    "did it change the prediction" question has a number too.

Read it as a chain: if ``mem`` is ~unchanged the site is wrong; if ``mem`` moves
but ``nonunif`` is small the shift is being cancelled by the heads; if both move
and the audio knob (``scratch/hint_knob_report.py``) does not, the model has the
signal and has not learned to use it -- which is an objective/data problem, not
a wiring one.

Read-only, a handful of forwards per checkpoint.

Usage::

    # the four Phase 2 wiring arms, intensity, levels 0..8
    python scratch/hint_injection_probe.py \
        --checkpoints checkpoints_stage2/hints_v1_p2_baseline_w1/last.ckpt \
                      checkpoints_stage2/hints_v1_p2_film_w1/last.ckpt \
                      checkpoints_stage2/hints_v1_p2_proj_w1/last.ckpt \
                      checkpoints_stage2/hints_v1_p2_film_proj_w1/last.ckpt

    # the finished v0.9, register instead
    python scratch/hint_injection_probe.py \
        --checkpoint checkpoints_stage2/transformer_v0.9/last.ckpt \
        --channel register --levels 0,1,3,5
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Dict, List, Optional

# fairseq (pulled in by modeling.generator) sets the root logger to DEBUG at
# import time; install our handler first (AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_submodule_prefix  # noqa: E402
from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402

#: Only the clip-level channels are interesting here: the plan's §3 algebra is
#: about a value that is constant over time, and those are exactly the channels
#: whose CE liveness is ~0.
CHANNEL_KEYS = {"intensity": "hint_int", "register": "hint_reg"}


def load_checkpoint(net, path):
    """Load the weights, and **fail loudly** if the model is the wrong layout.

    ``load_submodule_prefix`` is quiet, so a checkpoint trained with
    ``hint_film_memory`` / ``hint_decoder_proj`` loaded into a model built
    without them loses exactly those modules with no error -- the probe would
    then measure the baseline wiring for every arm and report identical numbers
    (which is how this guard came to exist: the Phase 2 gate scripts used the
    wrong config and did exactly that, see HINTS_V1_PLAN.md §5.6).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    prefix = "net." if any(k.startswith("net.") for k in sd) else ""
    own = set(net.state_dict())
    dropped = sorted(k for k in sd
                     if k.startswith(prefix) and k[len(prefix):] not in own)
    layout = [k for k in dropped
              if any(t in k for t in ("film_gamma", "film_beta", "hint_out"))]
    if layout:
        raise SystemExit(
            f"{path} was trained with a post-norm hint module this model does "
            f"not have ({', '.join(layout[:4])}...). Pass that arm's own config "
            f"with --configs, e.g. scratch/hints_v1_runs/<run>/configs/<exp>.yaml; "
            f"otherwise the module is silently dropped and every arm measures "
            f"the baseline wiring.")
    if dropped:
        print(f"    note: {len(dropped)} checkpoint tensor(s) not in this model "
              f"(e.g. {dropped[:3]})")
    load_submodule_prefix(net, prefix, sd, quiet=True)
    return ck.get("epoch"), ck.get("global_step")


def hint_kwargs(config):
    tc = config.transformer
    return dict(
        hints=bool(tc.get("hints", False)),
        hints_json=tc.get("hints_json", None),
        hint_dropout={},                  # never drop on a probe batch
        hint_null_all=0.0,
        hint_pitch_edges=tc.get("hint_pitch_edges", None),
        n_slot_buckets=int(tc.get("n_slot_buckets", 8) or 8),
    )


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def level_count(net, channel: str) -> int:
    if channel == "intensity":
        return int(net.n_intensity_levels)
    if channel == "register":
        return int(net.n_register_levels)
    raise ValueError(f"unknown channel {channel!r}")


def with_level(batch: dict, key: str, level: int) -> dict:
    out = dict(batch)
    out[key] = torch.full_like(batch[key], int(level))
    return out


def rel_shift(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-6))


def nonuniform_fraction(new: List[torch.Tensor],
                        old: List[torch.Tensor]) -> float:
    """Fraction of the logit shift that softmax can see.

    A constant added across the vocabulary is invisible to softmax, so the
    "useful" part of ``delta`` is ``delta - mean_v(delta)``.  Returns
    ``||delta_perp|| / ||delta||`` aggregated over codebooks (energy-weighted).
    """
    num = den = 0.0
    for a, b in zip(new, old):
        d = (a - b).float()
        d_perp = d - d.mean(dim=-1, keepdim=True)
        num += float(d_perp.pow(2).sum())
        den += float(d.pow(2).sum())
    return math.sqrt(num / den) if den > 0 else float("nan")


@torch.no_grad()
def decoder_logits(net, batch, memory, mask, hints):
    """Teacher-forced AR logits (no EOS frame -- identical for every arm)."""
    codes = torch.cat([batch["spec_codes"], batch["pitch_codes"]], dim=1).long()
    delayed = net.build_delay_pattern(codes, batch["code_lengths"])
    L = delayed.shape[2]
    start = net._pad_column(codes.shape[0], codes.device)
    inputs = torch.cat([start, delayed], dim=2)[..., :L]
    logits = net.decoder(inputs, memory, mask, sid=batch["sid"],
                         hint_vec=net._dec_hint(hints))
    return [lg.float() for lg in logits]


def module_norms(net) -> Dict[str, float]:
    """Norm of the two optional post-norm modules (0.0 / missing when off)."""
    out = {}
    enc = net.encoder
    for name in ("film_gamma", "film_beta"):
        mod = getattr(enc, name, None)
        if mod is not None:
            out[name] = float(mod.weight.float().norm())
    dec = net.decoder
    mod = getattr(dec, "hint_out", None)
    if mod is not None:
        out["dec_hint_out"] = float(mod.weight.float().norm())
    return out


@torch.no_grad()
def probe_checkpoint(net, loader, channel: str, levels: List[int], device,
                     max_batches: int):
    """Per-level ``(mem, logit, nonunif, ce)`` averaged over the probe batches."""
    key = CHANNEL_KEYS[channel]
    agg: Dict[int, Dict[str, float]] = {lv: {} for lv in levels}
    counts = {lv: 0 for lv in levels}
    for b_i, batch in enumerate(loader):
        if b_i >= max_batches:
            break
        batch = to_device(batch, device)
        if batch.get(key) is None:
            raise SystemExit("the probe items carry no hint channels -- run "
                             "hints/build.py over this split first.")
        # Baseline tensors at level 0: the reference every level is compared to.
        ref = with_level(batch, key, 0)
        mem0, mask0 = net.encode(batch["phone_ids"], batch["phone_lengths"],
                                 batch["sid"], hints=ref)
        lg0 = decoder_logits(net, batch, mem0, mask0, ref)
        for lv in levels:
            h = with_level(batch, key, lv)
            mem, mask = net.encode(batch["phone_ids"], batch["phone_lengths"],
                                   batch["sid"], hints=h)
            lg = decoder_logits(net, batch, mem, mask, h)
            ce = float(net.compute_batch_loss(batch, epoch=None,
                                              diagnostics=False,
                                              hints=h)["loss"])
            row = agg[lv]
            row["mem"] = row.get("mem", 0.0) + rel_shift(mem, mem0)
            row["logit"] = row.get("logit", 0.0) + rel_shift(
                torch.cat([x.reshape(-1) for x in lg]),
                torch.cat([x.reshape(-1) for x in lg0]))
            row["nonunif"] = row.get("nonunif", 0.0) + nonuniform_fraction(lg, lg0)
            row["ce"] = row.get("ce", 0.0) + ce
            counts[lv] += 1
    for lv in levels:
        n = max(counts[lv], 1)
        agg[lv] = {k: v / n for k, v in agg[lv].items()}
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="one config per --checkpoints entry, for arms whose "
                         "*architecture* differs (hint_film_memory / "
                         "hint_decoder_proj). The training arms generate these "
                         "under scratch/hints_v1_runs/<run>/configs/. Default: "
                         "use --config for all of them.")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--channel", default="intensity",
                    help="intensity or register (the clip-level channels)")
    ap.add_argument("--levels", default="0,2,4,6,8",
                    help="levels to measure; the first should be 0 (absent)")
    ap.add_argument("--val_filelist", default=None,
                    help="default: config.train.val_filelist")
    ap.add_argument("--batches", type=int, default=2,
                    help="probe batches (teacher-forced, so a few is plenty)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.channel not in CHANNEL_KEYS:
        raise SystemExit(f"--channel must be one of {sorted(CHANNEL_KEYS)}: "
                         f"the §3 cancellation argument is about a value that "
                         f"is constant over time.")

    configs = args.configs or [args.config] * len(args.checkpoints)
    if len(configs) != len(args.checkpoints):
        raise SystemExit(f"--configs has {len(configs)} entries for "
                         f"{len(args.checkpoints)} checkpoints; they must match")

    # The dataset depends on the *corpus*, not the arm (same hints_json, same
    # val split), so it is built once from the first config and reused.
    ds = CodeDataset(
        args.val_filelist or config.train.val_filelist, is_train=False,
        max_frames=config.train.get("max_frames", None),
        max_phone_length=config.train.get("max_phone_length", None),
        speaker_map=config.train.get("speaker_map", None), **hint_kwargs(config))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=CodeCollator(config=config), num_workers=0)
    print(f"{args.channel} levels {args.levels} (0 = absent), "
          f"{args.batches} batch(es) of {args.batch_size}  device={device}\n")

    for path, cfg_path in zip(args.checkpoints, configs):
        # One model per checkpoint: the arms differ in *architecture*.
        net = CodeTransformer(OmegaConf.load(cfg_path)).to(device)
        net.eval()
        if not net.hints:
            raise SystemExit(f"{cfg_path}: transformer.hints is false")
        max_level = level_count(net, args.channel)
        levels = [int(v) for v in args.levels.split(",") if v.strip()]
        levels = [lv for lv in levels if 0 <= lv <= max_level]
        if not levels or levels[0] != 0:
            raise SystemExit(f"levels must start at 0 (the reference arm); got "
                             f"{levels} (max {max_level})")
        epoch, step = load_checkpoint(net, path)
        agg = probe_checkpoint(net, loader, args.channel, levels, device,
                               args.batches)
        tag = (f"{os.path.basename(os.path.dirname(path))}/"
               f"{os.path.basename(path)}")
        mods = net.hint_film_memory or net.hint_decoder_proj
        print(f"=== {tag}  (epoch {epoch}, step {step})")
        print(f"    film_memory={bool(net.hint_film_memory)} "
              f"decoder_proj={bool(net.hint_decoder_proj)}"
              + "".join(f"  |{k}|={v:.2f}" for k, v in module_norms(net).items()))
        print(f"    {'lvl':>4} {'mem':>8} {'logit':>8} {'nonunif':>8} {'CE':>9}")
        for lv in levels:
            row = agg[lv]
            print(f"    {lv:>4} {row['mem']:>8.3f} {row['logit']:>8.3f} "
                  f"{row['nonunif']:>8.3f} {row['ce']:>9.4f}")
        top = levels[-1]
        print(f"    -> L{top}: memory moves {100 * agg[top]['mem']:.1f}%, "
              f"logits {100 * agg[top]['logit']:.1f}% "
              f"(of which {100 * agg[top]['nonunif']:.0f}% is class-dependent), "
              f"CE {agg[top]['ce'] - agg[0]['ce']:+.4f} nats\n")
        if mods:
            print("    (compare with the audio knob: if the numbers above move "
                  "but scratch/hint_knob_report.py does not, the model has the\n"
                  "     signal and has not learned to act on it -- objective/data,"
                  " not wiring)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
