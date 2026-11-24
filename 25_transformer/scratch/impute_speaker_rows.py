"""Impute under-trained speaker-embedding rows in a stage-1 or stage-2 checkpoint.

Why: both stages use a per-speaker ``nn.Embedding`` lookup, so there is no
parameter sharing between speakers at all -- "transfer" only happens through the
shared transformer/vocoder weights.  With this corpus that is the dominant
low-data effect: 1 559 / 2 704 stage-2 training speakers have <= 5 clips (median
4), and the embedding-row norms track the clip count almost linearly (stage 2,
encoder: 0.03 / 0.83 / 1.08 / 1.52 / 1.87 / 2.30 for 0 / 1 / 2-5 / 6-20 /
21-100 / 100+ clips).  For stage 1 the rows of un-/rarely-seen speakers are
still at the random initialisation norm (~6.0 vs ~5.55 for 100+ clips), and that
row *is* the voice the frozen Generator renders.

This script replaces every row with fewer than ``--min_clips`` clips with:
  ``mean``  the mean of the reliable rows (the population voice) -- recommended
  ``zero``  the zero vector (neutral additive offset)

It is a stop-gap, not a speaker encoder: listen to the output.  For a real fix,
condition on a representation shared across speakers (a speaker encoder, or the
stage-1 voice embedding) instead of an independent row per id.

Usage::

    # stage 1: patch the voice table (writes a new checkpoint; original intact)
    python scratch/impute_speaker_rows.py --stage stage1 \
        --checkpoint checkpoints_stage1/base_v0.7/last.ckpt \
        --filelist data/genshin_full_stage1/train.txt \
        --min_clips 6 --mode mean \
        --out checkpoints_stage1/base_v0.7/last_spkmean.ckpt

    # stage 2: patch the transformer's encoder+decoder speaker tables
    python scratch/impute_speaker_rows.py --stage stage2 \
        --checkpoint checkpoints_stage2/transformer_hints_v2.3/last.ckpt \
        --filelist data/genshin_plus_extras/train_stage2.txt \
        --min_clips 6 --mode mean --out /tmp/opencode/spkmean.ckpt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2_wer import (  # noqa: E402
    _impute_speaker_rows, speaker_clip_counts,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True,
                    help="output checkpoint (use a new path; the input is "
                         "never modified)")
    ap.add_argument("--filelist", required=True,
                    help="the stage's training filelist (path|...|sid) used for "
                         "the clip counts")
    ap.add_argument("--min_clips", type=int, default=6,
                    help="rows with fewer clips than this are replaced")
    ap.add_argument("--mode", choices=("mean", "zero"), default="mean")
    args = ap.parse_args()

    if os.path.abspath(args.checkpoint) == os.path.abspath(args.out):
        raise SystemExit("--out must differ from --checkpoint")
    counts = speaker_clip_counts(args.filelist)
    reliable = [s for s, n in counts.items() if n >= args.min_clips]
    print(f"{len(counts)} speakers in {args.filelist}, "
          f"{len(reliable)} with >= {args.min_clips} clips")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    if args.stage == "stage1":
        keys = [k for k in sd if k.endswith("spk_emb.weight")
                and ("net_g." in k or "." not in k)]
    else:
        keys = [k for k in sd if k.endswith("spk_emb.weight")]
    if not keys:
        raise SystemExit(f"no spk_emb.weight tensor found in {args.checkpoint}")
    for key in keys:
        w = sd[key]
        before = w[reliable].norm(dim=1).mean().item() if reliable else 0.0
        n = _impute_speaker_rows(w, counts, min_clips=args.min_clips,
                                 mode=args.mode)
        print(f"{key}: {w.shape} -> replaced {n} row(s) with "
              f"{args.mode!r}; reliable-row norm mean {before:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(ck, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
