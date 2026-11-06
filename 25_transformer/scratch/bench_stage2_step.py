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
    python scratch/bench_stage2_step.py --ns_steps 5 --muon_all   # pre-v0.6
    python scratch/bench_stage2_step.py --compile
    python scratch/bench_stage2_step.py --batch_size 32
    python scratch/bench_stage2_step.py --diagnostics_per_step  # no gating
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from importlib import import_module

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf  # noqa: E402

from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
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
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                    collate_fn=CodeCollator(config=config), num_workers=2,
                    persistent_workers=True)
    batches = []
    for b in dl:
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in b.items()})
        if len(batches) >= args.iters + 2:
            break

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
          f"{3600.0 / total * bs:.0f} items/hour")


if __name__ == "__main__":
    main()
