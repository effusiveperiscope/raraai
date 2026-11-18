#!/usr/bin/env python3
"""Is the semantic hint redundant, and is its content even in the codes?

Two questions decide whether the 384-d bge CLS channel is worth keeping:

1. **Redundancy with the text channel.**  The model already sees the phoneme
   sequence.  If the sentence vector is largely a function of that sequence,
   the semantic channel only re-encodes information the encoder has -- which
   predicts exactly what was measured: ~0 CE gain and no behavioural change.
   Ridge from a phone unigram+bigram histogram to the 384 dims, cross-validated
   **grouped by speaker**, answers this quantitatively.

2. **Presence in the code space.**  If the ground-truth codes do not carry
   sentence-level variation beyond the speaker, no wiring could use it.  Ridge
   from code histograms to the vector, again grouped by speaker, compared with
   the best speaker-only predictor (a per-speaker mean).

Reads ``.cond`` + ``phone_ids`` + ``spec_codes``/``pitch_codes``; CPU only.

    python scratch/hint_semantic_value.py --n 3000
    python scratch/hint_semantic_value.py --n 6000 --alpha 30
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_codes import cond_path_for  # noqa: E402
from scratch.hint_codespace_probe import (  # noqa: E402
    code_features, r2, speaker_mean_oof)

PHONE_VOCAB = 1024
HASH_BITS = 11                      # 2048 hashed bigram bins


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


def phone_features(ids):
    """Unigram histogram + hashed bigram histogram, L2-normalised halves."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    uni = np.bincount(np.clip(ids, 0, PHONE_VOCAB - 1), minlength=PHONE_VOCAB
                      ).astype(np.float32)
    if ids.size > 1:
        key = (ids[:-1] * PHONE_VOCAB + ids[1:]) % (1 << HASH_BITS)
        bi = np.bincount(key, minlength=1 << HASH_BITS).astype(np.float32)
    else:
        bi = np.zeros(1 << HASH_BITS, dtype=np.float32)
    for v in (uni, bi):
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
    return np.concatenate([uni, bi])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="data/genshin_plus_extras/train_stage2.txt")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--group_by_speaker", action="store_true", default=True)
    args = ap.parse_args(argv)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    rows = read_rows(args.filelist, args.n)
    print(f"{len(rows)} clip(s) from {args.filelist}")

    feats_text, feats_code, y, groups = [], [], [], []
    skipped = 0
    for parts in rows:
        phone_path, spec_path, pitch_path, sid = parts[:4]
        cond = cond_path_for(phone_path)
        try:
            c = torch.load(cond, map_location="cpu", weights_only=True, mmap=True)
            if "semantic" not in c:
                skipped += 1
                continue
            ids = torch.load(phone_path, map_location="cpu",
                             weights_only=True).reshape(-1).numpy()
            spec = torch.load(spec_path, map_location="cpu", weights_only=True)
            pitch = torch.load(pitch_path, map_location="cpu", weights_only=True)
        except Exception:
            skipped += 1
            continue
        feats_text.append(phone_features(ids))
        feats_code.append(code_features(spec, pitch, (64, 128)))
        y.append(np.asarray(c["semantic"], dtype=np.float32).reshape(-1))
        groups.append(int(sid))
    if not y:
        print("no clips with a semantic vector -- nothing to measure")
        return 1
    Xt = np.stack(feats_text)
    Xc = np.stack(feats_code)
    Y = np.stack(y)
    G = np.asarray(groups)
    print(f"{len(Y)} usable clip(s), {skipped} skipped; semantic dim {Y.shape[1]}")

    # PCA summary (unsupervised, so it does not leak the regression target).
    Yc = Y - Y.mean(0)
    u, s, _ = np.linalg.svd(Yc, full_matrices=False)
    var = (s ** 2) / float((s ** 2).sum())
    k = 4
    print("  semantic variance in PC1..4: "
          + "  ".join(f"{100 * v:.0f}%" for v in var[:k]))

    gkf = GroupKFold(n_splits=min(args.folds, len(np.unique(G))))
    splits = list(gkf.split(Xt, Y, G))

    def cv_r2(X, name):
        pred = np.zeros_like(Y)
        for tr, te in splits:
            m = Ridge(alpha=args.alpha)
            m.fit(X[tr], Y[tr])
            pred[te] = m.predict(X[te])
        per_dim = np.array([r2(Y[:, d], pred[:, d]) for d in range(Y.shape[1])])
        # R2 on the leading PCs, from the same predictions
        P = np.zeros((len(Y), k))
        T = np.zeros((len(Y), k))
        for tr, te in splits:
            _, _, vt = np.linalg.svd((Y[tr] - Y[tr].mean(0)), full_matrices=False)
            T[te] = (Y[te] - Y[tr].mean(0)) @ vt[:k].T
            P[te] = (pred[te] - Y[tr].mean(0)) @ vt[:k].T
        pc = [r2(T[:, i], P[:, i]) for i in range(k)]
        print(f"  R2(semantic <- {name:<14s}) median {np.nanmedian(per_dim):+.3f} "
              f"| mean {np.nanmean(per_dim):+.3f} "
              f"| PC1..{k}: " + "  ".join(f"{v:+.2f}" for v in pc))
        return float(np.nanmedian(per_dim))

    # best speaker-only predictor, per dimension
    sp = np.zeros_like(Y)
    for tr, te in splits:
        for d in range(Y.shape[1]):
            sp[te, d] = speaker_mean_oof(G, Y[:, d], tr, te)
    sp_med = float(np.nanmedian([r2(Y[:, d], sp[:, d])
                                 for d in range(Y.shape[1])]))
    print(f"  R2(semantic <- {'speaker mean':<14s}) median {sp_med:+.3f}")

    t = cv_r2(Xt, "phones")
    c = cv_r2(Xc, "codes")

    print()
    if t > 0.5:
        print("VERDICT: the sentence vector is mostly a function of the phoneme")
        print(f"  sequence (R2 {t:.2f}) -- the text channel already carries it,")
        print("  so a separate semantic embedding is redundant by construction.")
    else:
        print(f"VERDICT: the vector is NOT explained by the phonemes (R2 {t:.2f});")
        print("  it carries something extra that a text encoder cannot see.")
    if c > sp_med + 0.05:
        print(f"  It IS present in the codes (R2 {c:.2f} vs speaker-only "
              f"{sp_med:.2f}), so")
        print("  a dead channel would be a wiring/dropout problem.")
    else:
        print(f"  It is NOT in the codes beyond the speaker (R2 {c:.2f} vs "
              f"{sp_med:.2f}),")
        print("  so no injection site could make the channel useful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
