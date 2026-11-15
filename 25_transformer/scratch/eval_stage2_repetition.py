#!/usr/bin/env python3
"""Repetition / degeneration probe for stage-2 free-running samples.

WER alone does not tell you *why* a checkpoint is bad.  The v0.8 failure was a
**repetition attractor** (HINTS_PLAN.md §9): the model said a phrase, re-said
it, and never advanced, so the rest of the sentence was missing -- which WER
can only report as deletions.  This script measures that failure directly on
the sampled code stream, and on the item's ground-truth codes as a control.

Per sampled item, on each codebook of the stage-2 stream (spec books then pitch
books, ``0``-indexed; codebook 0 is the coarse spectral envelope that carries
prosodic continuity), over the generated span:

``repeat``
    Fraction of adjacent frame pairs ``(t, t+1)`` whose code is identical.
    This is the headline number: a healthy model repeats a coarse code
    sometimes (speech does), a degenerate one repeats it constantly.
``uniq``
    Distinct codes / frames.  Greedy decoding collapsing to a constant frame
    reads ~0.03 here.
``maxrun``
    Longest run of one identical code / frames.  ``repeat`` can be high for
    benign reasons; a single run covering most of the clip cannot.
``lag2``
    Fraction of frames equal to the frame *two* steps back (``t == t-2``, which
    is what a "A B A B..." alternation looks like).  Reported because v0.8's
    transcripts showed phrase-level loops rather than frame-level holds.

Nothing is written; checkpoints are only read.  Same generation protocol as
``eval_stage2_wer.py`` (temperature/top-k from the config, the item's own
length as the cap, EOS by default), so the numbers can be compared to its WER
on the same items.

Usage::

    # one checkpoint, 8 val items, ground-truth control included
    python scratch/eval_stage2_repetition.py \
        --checkpoints checkpoints_stage2/transformer_v0.9/last.ckpt \
                    checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt

    # real .cond hints instead of the trained null state (the generation default)
    python scratch/eval_stage2_repetition.py --hints --num_items 10 ...

    # Phase 3: does guidance make the loop worse?
    python scratch/eval_stage2_repetition.py --hints --hint_guidance 2.0 ...

Notes:
  * ``--hints`` supplies each item's ``.cond`` channels; without it generation
    runs in the model's trained null state (level 0 / ``sem_null``), the honest
    no-oracle state and the generation default -- see ``HINTS_V1_PLAN.md`` §1.
  * ``--gt`` (default on) scores the item's own codes with the same metric.  It
    is the floor: real speech has a non-zero ``repeat`` on book 0, so a
    generated stream at the ground-truth value is *not* degenerate.  Compare a
    checkpoint to that line, not to 0.
  * The historical ad-hoc numbers in ``HINTS_V1_PLAN.md`` §5.3 were produced by
    an undocumented definition; ``repeat`` on codebook 0 is the closest match
    (a model with a repetition attractor has the highest value) but the exact
    values will not line up.  Treat this script's table as the reproducible
    baseline from now on.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

# fairseq (pulled in by modeling.generator) sets the root logger to DEBUG at
# import time; install our handler first so that call becomes a no-op
# (AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_submodule_prefix  # noqa: E402
from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402

METRICS = ("repeat", "uniq", "maxrun", "lag2")


def load_checkpoint(net, path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    prefix = "net." if any(k.startswith("net.") for k in sd) else ""
    load_submodule_prefix(net, prefix, sd, quiet=True)
    return ck.get("epoch"), ck.get("global_step")


def code_stats(codes: torch.Tensor) -> Dict[str, float]:
    """Repetition statistics of one 1-D code stream.

    ``codes`` is a ``(T,)`` integer tensor (one codebook, one utterance).
    Returns NaNs for a stream shorter than 3 frames, where the definitions
    degenerate.
    """
    c = codes.detach().reshape(-1).to("cpu")
    t = int(c.numel())
    if t < 3:
        return {k: float("nan") for k in METRICS}
    eq1 = (c[1:] == c[:-1])
    eq2 = (c[2:] == c[:-2])
    # Longest run of one value, in frames.
    change = torch.nonzero(c[1:] != c[:-1], as_tuple=False).reshape(-1)
    bounds = torch.cat([torch.tensor([-1]), change,
                        torch.tensor([t - 1])])
    maxrun = int((bounds[1:] - bounds[:-1]).max())
    return {
        "repeat": float(eq1.float().mean()),
        "uniq": float(c.unique().numel()) / t,
        "maxrun": maxrun / t,
        "lag2": float(eq2.float().mean()),
    }


def mean_stats(stats: Sequence[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for key in METRICS:
        vals = [s[key] for s in stats if s[key] == s[key]]       # drop NaN
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


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


def build_transformer(config, device):
    net = CodeTransformer(config).to(device)
    net.eval()
    return net


@torch.no_grad()
def sample_codes(net, dataset: CodeDataset, indices: Sequence[int], args,
                 device, collator=None):
    """Sample one item at a time.

    Yields ``(gen_spec, gen_pitch, gt_spec, gt_pitch, cover)``.  The generated
    streams are trimmed to the model's own frame count; the ground-truth ones
    are **not** -- they always cover the item's full length, so the control arm
    is identical for every checkpoint (trimming it to a model's shorter output
    made the control move with the model under test).
    """
    for i in indices:
        item = dataset[i]
        phone = item["phone_ids"].unsqueeze(0).to(device)
        plen = torch.tensor([phone.shape[1]], dtype=torch.long, device=device)
        sid = torch.tensor([int(item["sid"])], dtype=torch.long, device=device)
        gt = torch.cat([item["spec_codes"], item["pitch_codes"]], dim=0).long()
        item_frames = min(int(item["spec_codes"].shape[1]), args.max_frames)
        hint_arg = None
        if collator is not None:
            batch = collator([item])
            hint_arg = {k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()}
        gen, glen = net.generate(
            phone, plen, sid, max_frames=item_frames,
            temperature=args.temperature, top_k=args.top_k,
            stop_on_eos=(False if args.ignore_eos else None),
            hints=hint_arg,
            hint_guidance=args.hint_guidance,
            hint_guidance_books=args.hint_guidance_books,
            return_lengths=True)
        # Drop the delay-pattern tail of the shorter books so both sides are
        # frame-aligned; ``glen`` is the model's own frame count.
        span = max(min(int(glen[0]), int(gen.shape[-1])), 0)
        n_spec = int(net.spec_codebook_num)
        yield (gen[0, :n_spec, :span],
               gen[0, n_spec:, :span],
               gt[:n_spec, :item_frames],
               gt[n_spec:, :item_frames],
               span / max(item_frames, 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--val_filelist", default=None,
                    help="default: config.train.val_filelist")
    ap.add_argument("--num_items", type=int, default=8,
                    help="val items to sample (evenly spaced over the split)")
    ap.add_argument("--max_frames", type=int, default=1000,
                    help="upper bound on generated frames (the item's own "
                         "length is used when shorter)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: train.sample_temperature; 0 = greedy")
    ap.add_argument("--top_k", type=int, default=None,
                    help="default: train.sample_top_k; 0 disables")
    ap.add_argument("--ignore_eos", action="store_true",
                    help="decode exactly the item's own length (removes the "
                         "model's duration errors, as in eval_stage2_wer)")
    ap.add_argument("--hints", action="store_true",
                    help="supply each item's .cond hints (default: the model's "
                         "trained null state)")
    ap.add_argument("--hint_guidance", "--hint-guidance", type=float,
                    default=None, dest="hint_guidance",
                    help="CFG weight over the hint channels (needs --hints)")
    ap.add_argument("--hint_guidance_books", "--hint-guidance-books",
                    default=None, dest="hint_guidance_books",
                    help="comma separated per-codebook weights "
                         "(spec0..spec7,pitch8,pitch9)")
    ap.add_argument("--no_gt", action="store_true",
                    help="skip the ground-truth control arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.hint_guidance_books:
        args.hint_guidance_books = [float(w) for w in
                                    str(args.hint_guidance_books).split(",")
                                    if w.strip()]

    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.temperature is None:
        args.temperature = float(config.train.get("sample_temperature", 0.9))
    if args.top_k is None:
        k = config.train.get("sample_top_k", None)
        args.top_k = int(k) if k else None
    elif args.top_k == 0:
        args.top_k = None
    if args.hint_guidance is not None and not args.hints:
        print("NOTE: --hint_guidance with the null-state arm is a no-op "
              "(cond == uncond); add --hints.")

    val_filelist = args.val_filelist or config.train.val_filelist
    dataset = CodeDataset(
        val_filelist, is_train=False, max_frames=None, max_phone_length=None,
        speaker_map=config.train.get("speaker_map", None), **hint_kwargs(config))
    n = len(dataset)
    if args.num_items <= 0 or args.num_items >= n:
        indices = list(range(n))
    elif args.num_items == 1:
        indices = [n // 2]
    else:
        indices = [round(j * (n - 1) / (args.num_items - 1))
                   for j in range(args.num_items)]

    collator = None
    if args.hints:
        if not bool(config.transformer.get("hints", False)):
            raise SystemExit("--hints needs transformer.hints: true")
        collator = CodeCollator(config=config)

    # The val items' hint levels, for the note about the null-state arm.
    print(f"val: {val_filelist} ({n} items) -> {len(indices)} items  "
          f"device={device}  temp={args.temperature} top_k={args.top_k}  "
          f"hints={'cond' if args.hints else 'null'}"
          f"{'' if args.hint_guidance is None else f'  guidance={args.hint_guidance}'}\n")
    header = (f"{'':<26}" + "".join(f"{k:>9}" for k in METRICS))
    print(header)

    net = build_transformer(config, device)
    for path in args.checkpoints:
        epoch, step = load_checkpoint(net, path)
        torch.manual_seed(args.seed)
        gen_spec, gen_pitch, gt_spec, gt_pitch = [], [], [], []
        covers = []
        n_used = 0
        for gs, gp, ts, tp, cover in sample_codes(net, dataset, indices, args,
                                                  device, collator=collator):
            if gs.shape[1] == 0:
                continue
            n_used += 1
            covers.append(cover)
            gen_spec.append([code_stats(gs[q]) for q in range(gs.shape[0])])
            gen_pitch.append([code_stats(gp[q]) for q in range(gp.shape[0])])
            if not args.no_gt:
                gt_spec.append([code_stats(ts[q]) for q in range(ts.shape[0])])
                gt_pitch.append([code_stats(tp[q]) for q in range(tp.shape[0])])
        tag = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
        print(f"=== {tag}  (epoch {epoch}, step {step}, {n_used} items, "
              f"mean length {mean(covers):.3f} of the reference)")
        m = mean_stats([s for item in gen_spec for s in item])
        print(f"  {'spec all books':<24}" + "".join(f"{m[k]:>9.3f}"
                                                    for k in METRICS))
        m = mean_stats([s for item in gen_pitch for s in item])
        print(f"  {'pitch all books':<24}" + "".join(f"{m[k]:>9.3f}"
                                                     for k in METRICS))
        if gen_spec:
            book0 = [item[0] for item in gen_spec]
            m = mean_stats(book0)
            print(f"  {'spec book0':<24}"
                  + "".join(f"{m[k]:>9.3f}" for k in METRICS))
            # per-book table for the spec stream (book 0 is the headline)
            if gen_spec[0] and len(gen_spec[0]) > 1:
                per = [mean_stats([item[q] for item in gen_spec])
                       for q in range(len(gen_spec[0]))]
                print("  spec per book  repeat: "
                      + " ".join(f"{v['repeat']:.3f}" for v in per))
            if gen_pitch[0] and len(gen_pitch[0]) > 1:
                per = [mean_stats([item[q] for item in gen_pitch])
                       for q in range(len(gen_pitch[0]))]
                print("  pitch per book repeat: "
                      + " ".join(f"{v['repeat']:.3f}" for v in per))
        if gt_spec:
            m = mean_stats([item[0] for item in gt_spec])
            print(f"  {'ground truth book0':<24}"
                  + "".join(f"{m[k]:>9.3f}" for k in METRICS))
            m = mean_stats([s for item in gt_pitch for s in item])
            print(f"  {'ground truth pitch':<24}"
                  + "".join(f"{m[k]:>9.3f}" for k in METRICS))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
