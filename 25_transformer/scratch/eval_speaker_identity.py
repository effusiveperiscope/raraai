"""How much does a stage-2 checkpoint actually use the speaker id?

``val_loss`` and WER are largely blind to voice identity: a model that ignores
``sid`` can score just as well, it just renders every line in one averaged voice.
This script measures the conditioning directly, teacher-forced on the *same*
validation batches, by re-scoring each batch with the speaker ids changed:

  true      the real ``sid`` (the baseline)
  rolled    ``sid`` rotated across the batch (every row gets another item's id)
  random    a uniformly random id in range per row

The reported deltas are ``CE(changed) - CE(true)`` in nats.  A model that uses
the speaker has a positive delta (the right id predicts the codes better); a
delta near zero means the speaker channel is decorative.  Also reported per
training-clip bucket, because the rank-64/zero-init layout can leave rare ids at
the neutral vector while high-data ids move.

For context it prints speaker-table geometry (mean vector norm and the effective
rank of the per-id vectors), which is where a collapsed layout shows up first.

Read-only: no checkpoint is written, no TensorBoard run is created.

Usage::

    python scratch/eval_speaker_identity.py \
        --checkpoints checkpoints_stage2/transformer_hints_v2.4/last.ckpt \
                      checkpoints_stage2/transformer_hints_v2.5/last.ckpt
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

logging.basicConfig(level=logging.WARNING)

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_submodule_prefix  # noqa: E402
from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer, uses_rope  # noqa: E402

SPK_KEYS = ("n_speakers",)


def speaker_bucket(n: int) -> str:
    return ("0" if n <= 0 else "1" if n == 1 else "2-5" if n <= 5
            else "6-20" if n <= 20 else "21-100" if n <= 100 else "100+")


def clip_counts(path: str) -> Counter:
    counts = Counter()
    if not path or not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "longform" in line:
                continue
            try:
                counts[int(line.rsplit("|", 1)[1])] += 1
            except ValueError:
                pass
    return counts


def peek_layout(path: str) -> Dict[str, object]:
    """Read the speaker/rope layout flags out of a checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    mark = sd.get("net.spk_layout_marker")
    if mark is None:
        layout = {"spk_rank": 0, "spk_norm": "none", "spk_init": "default",
                  "spk_kl_weight": 0.0, "spk_film_memory": False}
    else:
        vals = [float(v) for v in mark]
        layout = {"spk_rank": int(vals[0]),
                  "spk_norm": "l2" if vals[1] > 0 else "none",
                  "spk_init": "zero" if vals[2] > 0 else "default",
                  "spk_kl_weight": float(vals[3]),
                  "spk_film_memory": bool(vals[4])}
    layout["rope"] = uses_rope(sd)
    return layout


def apply_layout(config, layout):
    cfg = copy.deepcopy(config)
    for k in ("spk_rank", "spk_norm", "spk_init", "spk_kl_weight",
              "spk_film_memory"):
        cfg.transformer[k] = layout[k]
    cfg.transformer.rope = bool(layout["rope"])
    return cfg


def load_net(config, path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    state = None
    if "net.spk_lowrank.code.weight" in sd:
        state = ("lowrank", sd["net.encoder.spk_lowrank.code.weight"].shape[1])
    net = CodeTransformer(config).to(device)
    load_submodule_prefix(net, "net.", sd, quiet=True)
    net.eval()
    return net, ck, state


def speaker_geometry(sd, counts) -> Dict[str, float]:
    out = {}
    ids = [i for i in range(4096) if counts.get(i, 0) > 0]
    if not ids:
        return out
    for site in ("encoder", "decoder"):
        key = f"net.{site}.spk_emb.weight"
        if key in sd:
            V = sd[key].float()
        else:
            V = (sd[f"net.{site}.spk_lowrank.code.weight"].float()
                 @ sd[f"net.{site}.spk_lowrank.proj.weight"].float().T)
        Vi = V[ids]
        X = Vi - Vi.mean(0)
        sv = torch.linalg.svdvals(X)
        p = (sv ** 2) / (sv ** 2).sum()
        eff = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
        out[f"{site}_norm"] = float(Vi.norm(dim=-1).mean())
        out[f"{site}_eff_rank"] = eff
        for b in ("1", "2-5", "6-20", "21-100", "100+"):
            sel = [i for i in ids if speaker_bucket(counts.get(i, 0)) == b]
            if sel:
                out[f"{site}_norm_{b}"] = float(V[sel].norm(dim=-1).mean())
    return out


@torch.no_grad()
def score(net, loader, batches: int, device, mode: str,
          n_speakers: int, seed: int = 0) -> float:
    """Mean teacher-forced CE over ``batches`` val batches, ``mode`` speaker arm."""
    total = 0.0
    n = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for batch in loader:
        if n >= batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        if mode == "rolled":
            batch["sid"] = torch.roll(batch["sid"], 1)
        elif mode == "random":
            batch["sid"] = torch.randint(0, n_speakers, batch["sid"].shape,
                                         generator=gen).to(device)
        out = net.compute_batch_loss(batch, epoch=None)
        total += float(out["loss"])
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--train_filelist", default=None,
                    help="clip counts for the bucket table (default: config)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available()
                                         else "cpu"))
    counts = clip_counts(args.train_filelist
                         or config.train.get("train_filelist", None))

    layout0 = peek_layout(args.checkpoints[0])
    cfg0 = apply_layout(config, layout0)
    tc = cfg0.transformer
    hint_kwargs = dict(
        hints=bool(tc.get("hints", False)),
        hints_json=tc.get("hints_json", None),
        hint_dropout={},
        hint_null_all=0.0,
        hint_pitch_edges=tc.get("hint_pitch_edges", None),
        n_slot_buckets=int(tc.get("n_slot_buckets", 8) or 8),
    )
    dataset = CodeDataset(
        config.train.val_filelist, is_train=False,
        max_frames=config.train.get("max_frames", None),
        max_phone_length=config.train.get("max_phone_length", None),
        speaker_map=config.train.get("speaker_map", None),
        alignment_blocklist=config.train.get("alignment_blocklist", None),
        transcript_blocklist=config.train.get("transcript_blocklist", None),
        **hint_kwargs,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=CodeCollator(config=cfg0), num_workers=0)
    print(f"val: {config.train.val_filelist} ({len(dataset)} items), "
          f"{args.batches} batches of {args.batch_size}, device={device}\n")

    for path in args.checkpoints:
        layout = peek_layout(path)
        cfg = apply_layout(config, layout)
        net, ck, state = load_net(cfg, path, device)
        print("=" * 78)
        print(path)
        print(f"  step {ck.get('global_step')}  layout {layout}  "
              f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")
        # Batch-size note: a rolled batch mixes speaker ids, so the true arm is
        # the identity baseline and the deltas are the conditioning signal.
        n_spk = int(cfg.transformer.n_speakers)
        base = score(net, loader, args.batches, device, "true", n_spk)
        rolled = score(net, loader, args.batches, device, "rolled", n_spk)
        rand = score(net, loader, args.batches, device, "random", n_spk)

        # geometry needs the raw state dict of this checkpoint
        ck_sd = ck.get("state_dict", ck)
        geo = speaker_geometry(ck_sd, counts)
        print(f"  CE true   {base:.4f}")
        print(f"  CE rolled {rolled:.4f}  delta {rolled - base:+.4f}")
        print(f"  CE random {rand:.4f}  delta {rand - base:+.4f}")
        print("  speaker geometry: " + " ".join(
            f"{k}={v:.3g}" for k, v in geo.items()))


if __name__ == "__main__":
    main()
