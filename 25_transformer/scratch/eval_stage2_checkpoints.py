"""Compare several stage-2 checkpoints on the *same* validation batches.

Stage-2 ``val_loss`` is the only number that decides ``best-checkpoint.ckpt``,
but it is teacher-forced: it says nothing about whether the free-running
autoregressive sample still tracks the target codes (which is what decides
whether the synthesised audio is intelligible).  This script reports both, so a
claim like "the last 100 k steps made it better" can be checked against what the
model actually generates.

Per checkpoint it prints:

  * teacher-forced ``val_loss`` (+ spec/pitch split and per-codebook accuracy),
  * the FIM probe's base arm (middle CE / teacher-forced accuracy) and its
    ``text`` / ``suffix`` / ``region`` / ``length`` control-arm deltas -- the
    same numbers ``2_training.py:_fim_probe`` logs per epoch,
  * free-running (sampled) code match for the spec and pitch codebooks on a few
    val items, i.e. the analogue of ``sample/{spec,pitch}_code_match_frac_*``.

Read-only: writes no checkpoint and creates no TensorBoard run.  A handful of
forwards per checkpoint, so it is a seconds-scale check on GPU.

Usage::

    python scratch/eval_stage2_checkpoints.py \
        --checkpoints checkpoints_stage2/<exp>/best-checkpoint.ckpt \
                      checkpoints_stage2/<exp>/last.ckpt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

# fairseq (pulled in by modeling.generator or the dataset modules) calls
# logging.basicConfig(DEBUG) at import time; install our handler first so that
# call becomes a no-op (AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_submodule_prefix  # noqa: E402
from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402

SCALARS = ("loss", "spec_loss", "pitch_loss", "acc", "spec_acc", "pitch_acc")


def build_model(config, device):
    net = CodeTransformer(config).to(device)
    net.eval()
    return net


def load_checkpoint(net, path):
    """Load a Lightning checkpoint (``net.*`` prefixed) or a raw state_dict."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    prefix = "net." if any(k.startswith("net.") for k in sd) else ""
    load_submodule_prefix(net, prefix, sd, quiet=True)
    return ck.get("epoch"), ck.get("global_step")


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def teacher_forced(net, loader, batches, device) -> Dict[str, object]:
    """Mean teacher-forced loss/accuracy over the first ``batches`` batches."""
    sums = {k: 0.0 for k in SCALARS}
    code_accs = None
    n = 0
    for batch in loader:
        if n >= batches:
            break
        out = net.compute_batch_loss(to_device(batch, device), epoch=None,
                                     diagnostics=True)
        n += 1
        for k in SCALARS:
            sums[k] += float(out[k])
        a = out["code_accs"].detach().float().cpu()
        code_accs = a if code_accs is None else code_accs + a
    n = max(n, 1)
    res = {k: v / n for k, v in sums.items()}
    if code_accs is not None:
        res["code_accs"] = (code_accs / n).tolist()
    return res


@torch.no_grad()
def fim_probe(net, loader, batches, device) -> Dict[str, float]:
    """``2_training.py:_fim_probe``'s numbers, averaged over val batches."""
    if not net.fim:
        return {}
    tot = {"middle_loss": 0.0, "middle_acc": 0.0,
           "text": 0.0, "suffix": 0.0, "region": 0.0, "length": 0.0}
    n = 0
    for batch in loader:
        if n >= batches:
            break
        batch = to_device(batch, device)
        codes = torch.cat([batch["spec_codes"], batch["pitch_codes"]],
                          dim=1).long()
        lengths = batch["code_lengths"]
        T = int(lengths.min())
        p = max(1, T // 3)
        m = max(1, T // 3)
        if p + m >= T:               # keep at least one suffix frame
            m = max(1, T - p - 1)
        if T < 4 or m < 1:
            continue
        B = codes.shape[0]
        fim = torch.ones(B, dtype=torch.bool, device=device)
        pfx = torch.full((B,), p, dtype=torch.long, device=device)
        mid = torch.full((B,), m, dtype=torch.long, device=device)
        rot = torch.roll(torch.arange(B, device=device), 1)

        mem, mmask = net.encode(batch["phone_ids"], batch["phone_lengths"],
                                batch["sid"])
        mem_rot, mmask_rot = net.encode(batch["phone_ids"][rot],
                                        batch["phone_lengths"][rot],
                                        batch["sid"])

        def run(memory, mask, codes_use=None, **override):
            return net.code_loss(
                memory, mask, codes if codes_use is None else codes_use,
                code_lengths=lengths, sid=batch["sid"],
                complete=batch["complete"], diagnostics=True,
                fim_mask=fim, fim_prefix_len=pfx, fim_mid_len=mid,
                **override)

        base = run(mem, mmask)
        if "fim_middle_loss" not in base:
            continue
        ce = float(base["fim_middle_loss"])
        swapped = codes.clone()
        swapped[:, :, p + m:] = codes[rot][:, :, p + m:]
        tot["middle_loss"] += ce
        tot["middle_acc"] += float(base["fim_middle_acc"])
        tot["text"] += float(run(mem_rot, mmask_rot)["fim_middle_loss"]) - ce
        tot["suffix"] += float(run(mem, mmask, codes_use=swapped)
                               ["fim_middle_loss"]) - ce
        tot["region"] += float(run(mem, mmask, ablate_region=True)
                               ["fim_middle_loss"]) - ce
        if net.decoder.len_emb is not None:
            tot["length"] += float(run(mem, mmask, ablate_length=True)
                                   ["fim_middle_loss"]) - ce
        n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def free_running(net, dataset: CodeDataset, items: int, device,
                 temperature: float = 0.9, top_k: int = 20) -> Dict[str, float]:
    """Sample codes for the first ``items`` val items and match them.

    Mirrors ``2_training.py``'s sampling pass (temperature/top-k, generation
    length = the item's own frame count) but skips the audio decode, which is
    the slow part and needs a stage-1 checkpoint.
    """
    n_spec = net.spec_codebook_num
    match_spec = match_pitch = 0.0
    n = 0
    for i in range(min(items, len(dataset))):
        item = dataset[i]
        phone = item["phone_ids"].unsqueeze(0).to(device)
        plen = torch.tensor([phone.shape[1]], device=device)
        sid = torch.tensor([int(item["sid"])], device=device)
        gt = torch.cat([item["spec_codes"], item["pitch_codes"]],
                       dim=0).long().to(device)
        gen, glen = net.generate(phone, plen, sid, max_frames=gt.shape[1],
                                 temperature=temperature, top_k=top_k,
                                 return_lengths=True)
        span = min(int(glen[0]), gt.shape[1], gen.shape[-1])
        if span <= 0:
            continue
        g, t = gen[0, :, :span], gt[:, :span]
        match_spec += float((g[:n_spec] == t[:n_spec]).float().mean())
        match_pitch += float((g[n_spec:] == t[n_spec:]).float().mean())
        n += 1
    return {"spec_match": match_spec / max(n, 1),
            "pitch_match": match_pitch / max(n, 1), "items": n}


def main():
    ap = argparse.ArgumentParser(
        description="Compare stage-2 checkpoints on identical val batches.")
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--val_filelist", default=None,
                    help="default: config.train.val_filelist")
    ap.add_argument("--batches", type=int, default=8,
                    help="val batches for the teacher-forced / FIM metrics")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sample_items", type=int, default=4,
                    help="val items to free-run for the code-match numbers")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    val_filelist = args.val_filelist or config.train.val_filelist
    dataset = CodeDataset(
        val_filelist, is_train=False,
        max_frames=config.train.get("max_frames", None),
        max_phone_length=config.train.get("max_phone_length", None),
        speaker_map=config.train.get("speaker_map", None),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=CodeCollator(config=config), num_workers=0)
    print(f"val: {val_filelist} ({len(dataset)} items)  device={device}  "
          f"{args.batches} batches of {args.batch_size}, "
          f"{args.sample_items} sampled items\n")

    net = build_model(config, device)
    results = []
    for path in args.checkpoints:
        epoch, step = load_checkpoint(net, path)
        tf = teacher_forced(net, loader, args.batches, device)
        fim = fim_probe(net, loader, args.batches, device)
        fr = free_running(net, dataset, args.sample_items, device)
        results.append((tf, fim, fr))
        print(f"=== {os.path.basename(os.path.dirname(path))}/"
              f"{os.path.basename(path)}  (epoch {epoch}, step {step})")
        print(f"    teacher-forced val_loss {tf['loss']:.4f} "
              f"(spec {tf['spec_loss']:.4f} / pitch {tf['pitch_loss']:.4f})  "
              f"acc {tf['acc']:.4f} (spec {tf['spec_acc']:.4f} / "
              f"pitch {tf['pitch_acc']:.4f})")
        if "code_accs" in tf:
            print("    per-book acc: " +
                  " ".join(f"{v:.3f}" for v in tf["code_accs"]))
        if fim:
            print(f"    FIM middle CE {fim['middle_loss']:.4f} "
                  f"acc {fim['middle_acc']:.4f} | deltas (nats) "
                  f"text {fim['text']:+.3f} suffix {fim['suffix']:+.3f} "
                  f"region {fim['region']:+.3f} length {fim['length']:+.3f}")
        print(f"    free-running match spec {fr['spec_match']:.4f} / "
              f"pitch {fr['pitch_match']:.4f} ({fr['items']} items)\n")

    if len(results) > 1:
        tf0, fim0, fr0 = results[0]
        tf1, fim1, fr1 = results[-1]
        print("delta (last minus first):")
        print(f"    val_loss {tf1['loss'] - tf0['loss']:+.4f}   "
              f"acc {tf1['acc'] - tf0['acc']:+.4f}")
        if fim0 and fim1:
            print(f"    FIM middle CE {fim1['middle_loss'] - fim0['middle_loss']:+.4f}"
                  f"   acc {fim1['middle_acc'] - fim0['middle_acc']:+.4f}")
        print(f"    free-running spec {fr1['spec_match'] - fr0['spec_match']:+.4f}   "
              f"pitch {fr1['pitch_match'] - fr0['pitch_match']:+.4f}")


if __name__ == "__main__":
    main()

