"""Ad-hoc step-time benchmark for stage-2 training (no Lightning, no Trainer).

Measures one real optimisation step on real preprocessed code tensors and
reports the *cumulative* cost of each part, because the parts have very
different scaling behaviour and a single "ms/step" hides that:

    fwd+bwd   -> scales with batch_size x max_frames
    +clip     -> scales with parameter count
    +adamw    -> same
    +muon     -> 5 x 3 matmuls x N params: FIXED per step, batch independent

Usage (from the repo root, in the raraai25 env):

    python scratch/bench_stage2_step.py                    # shipped config
    python scratch/bench_stage2_step.py --no_bucket        # pre-v2.3 order
    python scratch/bench_stage2_step.py --batch_size 24 --pool_batches 8
    python scratch/bench_stage2_step.py --ns_steps 5 --muon_all   # pre-v0.6
    python scratch/bench_stage2_step.py --compile
    python scratch/bench_stage2_step.py --diagnostics_per_step  # no gating

``--bucket`` (the default) measures the length-bucketed batch order that
``train.bucket_batches`` uses, ``--no_bucket`` the historical uniform shuffle.
Same items, different padded width, so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from importlib import import_module

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf  # noqa: E402

from dataset_codes import (CodeCollator, CodeDataset,
                           LengthBucketBatchSampler)  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--filelist", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ns_steps", type=int, default=None)
    ap.add_argument("--muon_all", action="store_true",
                    help="route embeddings/heads back to Muon (pre-v0.6 layout)")
    ap.add_argument("--diagnostics_per_step", action="store_true",
                    help="compute per-codebook acc/entropy on every step")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--bucket", dest="bucket", action="store_true",
                    default=True,
                    help="length-bucketed batches (the training default; "
                         "this changes the batch width, so it is what you want "
                         "to measure)")
    ap.add_argument("--no_bucket", dest="bucket", action="store_false",
                    help="uniform shuffle, the historical/pre-v2.3 order")
    ap.add_argument("--pool_batches", type=int, default=8,
                    help="pools per bucket (train.bucket_pool_batches)")
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.max_frames:
        config.train.max_frames = args.max_frames
    if args.ns_steps is None:
        args.ns_steps = int(config.train.get("muon_ns_steps", 5) or 5)
    if args.muon_all:
        config.train.muon_include_embeddings = True
        config.train.muon_include_heads = True

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = int(config.train.batch_size)
    frames = int(config.train.max_frames)

    ds = CodeDataset(args.filelist or config.train.train_filelist, is_train=True,
                     max_frames=frames,
                     max_phone_length=config.train.get("max_phone_length"),
                     speaker_map=config.train.get("speaker_map"))
    # Batch order matters here: the collator pads to the batch's longest item,
    # so the bucketed order (the training default, see
    # dataset_codes.LengthBucketBatchSampler) runs on ~half the width of the
    # uniform shuffle.  Report which one produced the numbers.
    #
    # The probe batches are spread evenly across a whole epoch instead of being
    # the first few produced: with bucketing, the first batches of an epoch come
    # from the shortest pool (a band is emitted short -> long), so the first N
    # would understate the width -- optimistically, which is exactly the wrong
    # direction for a benchmark.
    coll = CodeCollator(config=config)
    if args.bucket:
        plan = list(LengthBucketBatchSampler(
            ds.compute_lengths(), bs, pool_batches=args.pool_batches,
            drop_last=True, seed=1234))
    else:
        order = list(range(len(ds)))
        random.Random(1234).shuffle(order)
        plan = [order[i:i + bs] for i in range(0, len(order) - bs + 1, bs)]
    n_probe = args.iters + args.warmup
    picks = [plan[int(i * (len(plan) - 1) / max(n_probe - 1, 1))]
             for i in range(n_probe)]
    batches, widths = [], []
    for idx in picks:
        b = coll([ds[j] for j in idx])
        widths.append(int(b["code_lengths"].max()))
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in b.items()})

    net = CodeTransformer(config).to(device)
    module = import_module("2_training").TrainModule(net=net, config=config)
    muon_params, adamw_params = module._split_params()
    wd = float(config.train.get("weight_decay", 0.0) or 0.0)
    opt_adamw = torch.optim.AdamW(
        adamw_params, lr=config.train.lr, betas=tuple(config.train.betas),
        eps=config.train.eps, weight_decay=wd)
    opt_muon = torch.optim.Muon(
        muon_params,
        lr=config.train.lr * float(config.train.get("muon_lr_mult", 200.0)),
        weight_decay=wd, ns_steps=args.ns_steps)

    print(f"batch_size={bs}  max_frames={frames}  ns_steps={args.ns_steps}  "
          f"muon_all={args.muon_all}  compile={args.compile}  "
          f"clip={args.clip}")
    print(f"batching={'bucketed(pool=%d)' % args.pool_batches if args.bucket else 'uniform shuffle'}"
          f"  mean batch width={statistics.mean(widths):.1f} frames")
    print(f"Muon: {len(muon_params)} tensors "
          f"({sum(p.numel() for p in muon_params)/1e6:.1f} M) | "
          f"AdamW: {len(adamw_params)} tensors")
    if args.compile:
        module._maybe_compile_decoder()

    every = max(int(config.train.get("log_interval", 50) or 50), 1)
    state = {"i": 0}

    def step():
        batch = batches[state["i"] % len(batches)]
        diagnostics = (args.diagnostics_per_step or state["i"] % every == 0)
        state["i"] += 1
        net.train()
        out = net.compute_batch_loss(batch, epoch=1, diagnostics=diagnostics)
        opt_adamw.zero_grad(set_to_none=True)
        opt_muon.zero_grad(set_to_none=True)
        out["loss"].backward()
        yield "fwd+bwd"
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.grad is not None], args.clip)
        yield "+clip"
        opt_adamw.step()
        yield "+adamw"
        opt_muon.step()
        yield "+muon"

    for _ in range(args.warmup):
        for _ in step():
            pass
    torch.cuda.synchronize()

    cum: dict[str, list[float]] = {}
    for _ in range(args.iters):
        t0 = time.perf_counter()
        for tag in step():
            torch.cuda.synchronize()
            cum.setdefault(tag, []).append((time.perf_counter() - t0) * 1e3)

    prev = 0.0
    print(f"\n{'cumulative':<12}{'median ms':>11}{'delta ms':>11}")
    for tag in ("fwd+bwd", "+clip", "+adamw", "+muon"):
        med = statistics.median(cum[tag])
        print(f"{tag:<12}{med:>11.1f}{(med - prev):>11.1f}")
        prev = med
    total = statistics.median(cum["+muon"])
    print(f"\nstep total   {total:.1f} ms  -> "
          f"{3600.0 / (total / 1000.0) * bs:.0f} items/hour")


if __name__ == "__main__":
    main()
