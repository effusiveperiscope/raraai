#!/usr/bin/env python3
"""Can the *code space* express each hint at all?  (speaker-grouped CV)

For every hint channel there are two possible reasons it does nothing in the
model:

* the wiring does not deliver it (fixable: injection site / guidance), or
* the codes simply do not carry the information, so no wiring could.

This probe separates them by predicting each clip-level hint from the
**ground-truth codes** alone (per-codebook code histograms + duration), with
5-fold cross-validation **grouped by speaker** -- so a model cannot win by
memorising who is speaking -- and compares against the best possible
speaker-only predictor (a per-speaker mean, leave-one-out inside each training
fold).  ``R2(codes)`` well above ``R2(speaker)`` means the covariate is in the
codes and the channel is worth wiring; ``R2(codes) ~= R2(speaker)`` means the
channel is a second, coarser ``sid`` (which is exactly what the absolute
register turned out to be -- HINTS_PLAN.md 8.5).

CPU only, a few seconds for a few thousand clips.  Nothing is written.

Usage::

    python scratch/hint_codespace_probe.py --n 4000
    python scratch/hint_codespace_probe.py --filelist data/genshin_plus_extras/train_stage2.txt \
        --n 8000 --hints-json data/genshin_plus_extras/hints_rel.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_codes import cond_path_for  # noqa: E402
from hints.levels import load_levels, register_rel_st  # noqa: E402


def read_rows(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "longform" in line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            rows.append(parts)
            if len(rows) >= limit:
                break
    return rows


def code_features(spec, pitch, sizes):
    """Per-book code histograms (normalised) + log duration."""
    feats = []
    for codes, size in ((spec, sizes[0]), (pitch, sizes[1])):
        if codes is None or codes.numel() == 0:
            feats.append(np.zeros(codes.shape[0] * size if codes is not None
                                  else 0, dtype=np.float32))
            continue
        codes = codes.long()
        t = max(int(codes.shape[1]), 1)
        for q in range(codes.shape[0]):
            hist = torch.bincount(codes[q].clamp(min=0), minlength=size)
            feats.append((hist.float() / t).numpy())
    feats.append(np.array([np.log1p(int(spec.shape[1]))], dtype=np.float32))
    return np.concatenate(feats)


def speaker_mean_oof(groups, y, train_idx, test_idx):
    """Best speaker-only predictor: leave-one-out per-speaker mean."""
    out = np.empty(len(test_idx), dtype=np.float64)
    g_train, y_train = groups[train_idx], y[train_idx]
    order = np.argsort(g_train, kind="stable")
    g_sorted, y_sorted = g_train[order], y_train[order]
    uniq, starts = np.unique(g_sorted, return_index=True)
    sums = np.add.reduceat(y_sorted, starts)
    counts = np.diff(np.append(starts, len(g_sorted)))
    table = dict(zip(uniq.tolist(), zip(sums.tolist(), counts.tolist())))
    glob = float(y_train.mean())
    for i, g in enumerate(groups[test_idx]):
        total, n = table.get(int(g), (0.0, 0))
        out[i] = (total - 0.0) / n if n else glob
    return out


def r2(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="data/genshin_plus_extras/train_stage2.txt")
    ap.add_argument("--n", type=int, default=4000, help="clips to load")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="ridge alpha (features are dense histograms)")
    ap.add_argument("--hints-json", default=None,
                    help="default: next to the filelist (hints_rel.json or hints.json)")
    args = ap.parse_args(argv)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    rows = read_rows(args.filelist, args.n)
    if not rows:
        print(f"no usable lines in {args.filelist}")
        return 1

    hints_json = args.hints_json
    if not hints_json:
        base = os.path.dirname(os.path.abspath(args.filelist))
        for name in ("hints_rel.json", "hints.json"):
            cand = os.path.join(base, name)
            if os.path.exists(cand):
                hints_json = cand
                break
    if not hints_json or not os.path.exists(hints_json):
        print("no hints.json found next to the filelist -- pass --hints-json")
        return 1
    levels = load_levels(hints_json)
    speaker_means = (levels.get("register") or {}).get("speaker_means")
    print(f"{args.filelist}: {len(rows)} clip(s); levels from {hints_json}")
    print(f"register: {'relative to the speaker' if speaker_means else 'absolute'}\n")

    first = torch.load(rows[0][1], map_location="cpu", weights_only=True)
    sizes = (int(first.max()) + 1, 1)
    feats, targets, groups = [], {"intensity": [], "register": []}, []
    for parts in rows:
        phone_path, spec_path, pitch_path = parts[0], parts[1], parts[2]
        sid = int(parts[3])
        cond_path = cond_path_for(phone_path)

        if not os.path.exists(cond_path):
            continue
        cond = torch.load(cond_path, map_location="cpu", weights_only=True)
        if "intensity" not in cond or "register_st" not in cond:
            continue
        spec = torch.load(spec_path, map_location="cpu", weights_only=True)
        pitch = torch.load(pitch_path, map_location="cpu", weights_only=True)
        psize = int(pitch.max()) + 1 if pitch.numel() else 1
        sizes = (max(sizes[0], int(spec.max()) + 1), max(sizes[1], psize))
        reg = float(cond["register_st"].reshape(-1)[0])
        if speaker_means is not None:
            reg = register_rel_st(reg, sid, speaker_means)
        feats.append(code_features(spec, pitch, sizes))
        targets["intensity"].append(float(cond["intensity"].reshape(-1)[0]))
        targets["register"].append(reg)
        groups.append(sid)

    X = np.stack(feats)
    groups = np.asarray(groups)
    print(f"loaded {len(X)} clip(s), {X.shape[1]} features "
          f"(vocab {sizes[0]} spec / {sizes[1]} pitch), "
          f"{len(set(groups.tolist()))} speaker(s)\n")

    print(f"{'target':<10} {'R2 codes':>9} {'R2 speaker':>11} {'n':>7}")
    for name, values in targets.items():
        y = np.asarray(values, dtype=np.float64)
        oof = np.empty_like(y)
        oof_spk = np.empty_like(y)
        cv = GroupKFold(n_splits=min(args.folds, len(set(groups.tolist()))))
        for train, test in cv.split(X, y, groups):
            model = Ridge(alpha=args.alpha)
            model.fit(X[train], y[train])
            oof[test] = model.predict(X[test])
            oof_spk[test] = speaker_mean_oof(groups, y, train, test)
        print(f"{name:<10} {r2(y, oof):>9.3f} {r2(y, oof_spk):>11.3f} "
              f"{len(y):>7}")
    print("\nR2(codes) >> R2(speaker): the covariate is in the code space, so a "
          "dead channel is\na wiring problem.  R2(codes) ~ R2(speaker): the "
          "channel is mostly a second,\ncoarser `sid`, and a warm start cannot "
          "learn it from the speaker embedding it\nalready has -- that was the "
          "absolute register (HINTS_PLAN 8.5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
