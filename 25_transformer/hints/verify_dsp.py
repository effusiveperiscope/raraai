#!/usr/bin/env python3
"""Acceptance test for the ported DSP intensity pipeline (M0).

Compares, for clips that appear both in a stage-1 tensor directory and in an
upstream ``dsp_features*.csv``:

* the **16 tier-1 feature values** recomputed here from ``<base>.wave`` (48 kHz)
  against the upstream row (computed from the matching 16 kHz ``.pt``), and
* the **predicted intensity** from those two feature vectors, plus both against
  the human ``label``.

It exists because the port changes the audio path in two ways --
48 kHz -> 16 kHz resampling that the upstream pipeline did not do, and
``data.audio_norm`` peak normalisation -- and wants a number rather than an
argument for whether that moved the hint.

Usage::

    python hints/verify_dsp.py \
        --csv /mnt/data/Code/raraai/20_intensity_annotation/dsp_features_augmented.csv \
        --wave-dir data/genshin_full_stage1 --n 20

Exit code is 0 when the predictions agree within :data:`MAX_MEAN_DELTA` /
:data:`MAX_ANY_DELTA`; 1 otherwise (so it can gate a re-fit or a code change).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import hints.build as B

MAX_MEAN_DELTA = 0.10   # mean |prediction difference| allowed, 1-9 scale
MAX_ANY_DELTA = 0.40    # worst single clip (model MAE is ~0.59)


def tier1_features(wave_path: str, tier: int = 1):
    """Recompute tier-1 features from a stage-1 ``.wave``.

    Returns ``(features, silent_frac)``.  ``silent_frac`` is the fraction of
    exactly-zero samples at 16 kHz: the DSP feature set puts ``alpha = -120 dB``
    on a digitally silent frame, so a clip that is mostly digital silence has an
    unstable frame *mean* (see ``hints/README.md``).  It is reported rather than
    silently tolerated.
    """
    import librosa
    from dsp_features import extract

    wave = np.asarray(torch.load(wave_path, weights_only=True),
                      dtype=np.float32).reshape(-1)
    w16 = librosa.resample(wave, orig_sr=B.WAVE_SR, target_sr=B.INTENSITY_SR,
                           res_type="soxr_hq")
    return extract(w16, tier=tier, all_features=True), float((w16 == 0).mean())


def predict(features: dict) -> float:
    import pandas as pd
    return float(B._MODEL.predict(pd.DataFrame([features]))[0])


def find_wave(wave_dirs, stem: str) -> Optional[str]:
    """Locate ``<stem><ext>.wave``; upstream ``.pt`` names drop the extension."""
    for ext in ("", ".wav", ".flac", ".mp3"):
        p = B.find_sidecar(wave_dirs, stem + ext, B.WAVE_SUFFIX)
        if p is not None:
            return p
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="upstream dsp_features*.csv")
    p.add_argument("--wave-dir", nargs="+", required=True,
                   help="stage-1 tensor dirs holding <base>.wave")
    p.add_argument("--model", default=B.DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=20, help="clips to check (0 = all)")
    p.add_argument("--corpus", default="genshin",
                   help="only rows whose `corpus` column matches this")
    p.add_argument("--silent-eps", type=float, default=0.0,
                   help="fraction of exact-zero samples at/below which a clip "
                        "counts as 'clean' (the gate is applied to those only; "
                        "see hints/README.md)")
    args = p.parse_args(argv)

    from dsp_features import FEATURE_TIERS
    keys = FEATURE_TIERS[1]

    B.init_worker(args.model)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    picked: List[tuple] = []
    n_no_wave = 0
    for row in rows:
        if args.corpus and row.get("corpus") != args.corpus:
            continue
        stem = os.path.splitext(os.path.basename(row["path"]))[0]   # X.pt -> X
        wave = find_wave(args.wave_dir, stem)
        if wave is None:
            n_no_wave += 1
            continue
        feats, silent_frac = tier1_features(wave)
        picked.append((dict(row), feats, silent_frac))
        if args.n and len(picked) >= args.n:
            break
    if not picked:
        print(f"no usable overlapping clips in {args.csv}", file=sys.stderr)
        return 1

    records = []          # (clean, row, ref_pred, our_pred)
    feat_deltas = {"clean": {k: [] for k in keys}, "silent": {k: [] for k in keys}}
    for row, feats, silent_frac in picked:
        ref = {k: float(row[k]) for k in row
               if not k.startswith("_")
               and k not in ("path", "split", "speaker", "corpus", "notes")
               and row[k] not in ("", None)}
        ours = dict(ref)
        ours.update({k: float(feats[k]) for k in feats})
        r, o = predict(ref), predict(ours)
        clean = silent_frac <= args.silent_eps
        group = "clean" if clean else "silent"
        records.append((clean, row, r, o))
        for k in keys:
            feat_deltas[group][k].append(float(feats[k]) - float(row[k]))

    def summarise(clean: bool) -> Optional[np.ndarray]:
        rows_g = [x for x in records if x[0] is clean]
        if not rows_g:
            return None
        d = np.asarray([x[3] - x[2] for x in rows_g])
        mae_ref = np.abs(np.asarray([x[2] for x in rows_g])
                         - np.asarray([float(x[1]["label"]) for x in rows_g])).mean()
        mae_our = np.abs(np.asarray([x[3] for x in rows_g])
                         - np.asarray([float(x[1]["label"]) for x in rows_g])).mean()
        tag = "clean " if clean else "silent"
        print(f"/{tag} n={len(rows_g):3d}  mean|d|={np.abs(d).mean():.3f}  "
              f"max|d|={np.abs(d).max():.3f}  "
              f"ref-vs-label MAE={mae_ref:.3f}  ours-vs-label MAE={mae_our:.3f}")
        return d

    print(f"clips={len(picked)}  corpus={args.corpus}  tier=1  "
          f"({n_no_wave} without .wave)")
    print("prediction agreement (ours - upstream prediction):")
    d_clean = summarise(True)
    summarise(False)   # informational: digital silence changes the frame means
    print("features (ours - upstream) on clean clips, relative to cross-clip std:")
    for k in keys:
        v = np.asarray(feat_deltas["clean"][k])
        if v.size == 0:
            continue
        sd = np.asarray([float(x[1][k]) for x in records if x[0]]).std() or 1.0
        print(f"  {k:16s} mean|d|={np.abs(v).mean():9.4f}  "
              f"max|d|={np.abs(v).max():9.4f}  |d|/std={np.abs(v).mean()/sd:6.3f}")
    if d_clean is None:
        print("FAIL (no clean clips to judge)", file=sys.stderr)
        return 1
    ok = (np.abs(d_clean).mean() <= MAX_MEAN_DELTA
          and np.abs(d_clean).max() <= MAX_ANY_DELTA)
    print("PASS" if ok else "FAIL",
          f"(clean clips only; thresholds: mean<={MAX_MEAN_DELTA}, "
          f"max<={MAX_ANY_DELTA})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
