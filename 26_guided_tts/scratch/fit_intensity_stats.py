"""Fit the per-speaker intensity reference used by the plan's style slider.

``cond.intensity`` is a 1-9 DSP score whose *level* is partly a speaker trait
(measured: ~1/3 of its variance is between speakers, and it correlates 0.62
with ``register_st``).  A user-facing slider must therefore be **speaker
relative**: "more intense than this character usually is", not "a 6 on an
absolute scale".

This writes ``intensity_stats.json``:

    {
      "version": 1,
      "global": {"mean": 4.57, "std": 0.68, "n": ...},
      "shrink_k": 5.0,
      "speakers": {"<stage-2 sid>": {"mean": ..., "std": ..., "n": ...}}
    }

The loader shrinks each speaker mean towards the global mean with
``k = shrink_k`` pseudo-clips, then reports ``intensity_rel`` as a z-score
using the **global** std (per-speaker std is too noisy for short characters).
Fit on the training split only.

Usage (from ``26_guided_tts``)::

    python scratch/fit_intensity_stats.py \
        --filelists data/genshin_plus_extras/train_stage2.txt \
        --out intensity_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "intensity_stats.json"


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, ROOT.parent / "25_transformer"):
        if (base / p).exists():
            return base / p
    return ROOT / p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelists", nargs="+", default=[
        "data/genshin_plus_extras/train_stage2.txt"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--shrink-k", type=float, default=5.0,
                    help="pseudo-clips pulling a speaker mean to the global mean")
    args = ap.parse_args()

    seen: set[str] = set()
    by_sid: dict[int, list[float]] = defaultdict(list)
    total: list[float] = []

    for fl in args.filelists:
        path = _resolve(fl)
        if not path.exists():
            print(f"skip missing filelist {path}")
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4 or parts[0] in seen:
                continue
            seen.add(parts[0])
            base = parts[1][: -len(".spec_codes")] if parts[1].endswith(".spec_codes") else parts[1]
            cond_p = Path(base + ".cond")
            if not cond_p.exists():
                continue
            try:
                cond = torch.load(cond_p, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001
                continue
            if "intensity" not in cond:
                continue
            v = float(torch.as_tensor(cond["intensity"]).reshape(-1)[0])
            if not math.isfinite(v):
                continue
            sid = int(parts[3])
            by_sid[sid].append(v)
            total.append(v)

    if not total:
        print("no intensity values found")
        return 1

    import numpy as np
    a = np.asarray(total, dtype=np.float64)
    g_mean, g_std = float(a.mean()), float(a.std())
    speakers = {}
    for sid, vals in sorted(by_sid.items()):
        v = np.asarray(vals, dtype=np.float64)
        speakers[str(sid)] = {"mean": round(float(v.mean()), 6),
                              "std": round(float(v.std()), 6),
                              "n": int(v.size)}

    out = {
        "version": 1,
        "source": [str(_resolve(f)) for f in args.filelists],
        "global": {"mean": round(g_mean, 6), "std": round(g_std, 6),
                   "n": int(a.size)},
        "shrink_k": float(args.shrink_k),
        "speakers": speakers,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1) + "\n")

    ns = np.array([s["n"] for s in speakers.values()])
    print(f"clips {a.size}  speakers {len(speakers)}  "
          f"global mean {g_mean:.3f} std {g_std:.3f}")
    for lo in (1, 5, 20):
        print(f"  speakers with >= {lo:2d} clips: {int((ns >= lo).sum())}")
    print(f"wrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
