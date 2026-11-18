#!/usr/bin/env python3
"""Rank word-level semantic candidates for the hint stack (probe only).

The sentence-level bge CLS channel measured dead (HINTS_PLAN.md §11): ~45 % of
its leading variance is predictable from a bag of phone bigrams, and only 1.5 %
is linearly recoverable from the codes (intensity: 50 %, register: 16 %).  The
*word* granularity was never tested, and it is a different question: POS, BERT
token states and bge token states vary **within** an utterance, so they can
describe *which* word is lengthened/stressed rather than what the sentence is
about.

The criterion is not "is the vector meaningful" but "does the word's state
predict the word's prosody **beyond the pronunciation**", because the model
already sees the phones.  Targets come from the existing `.cond` files, so no
new corpus pass is needed:

* ``log duration``  -- ``word_slot_f`` (start-to-start, code frames),
* ``pitch``         -- ``pitch_word_st`` (semitones re: the utterance median),
* ``f0 range``      -- ``max - min`` over the word's valid anchors (st).

Candidates (per word, then repeated over the word's phones when wired):

============  ================================================================
``pos``       Penn POS tag from ``nltk.pos_tag`` (one-hot, the cheap baseline)
``bert``      ``bert-base-uncased`` last-hidden-state, mean over the wordpieces
``bge_tok``   ``BAAI/bge-small-en-v1.5`` **token** states (the CLS's own model)
``wordhash``  hashed spelling (**lexical-identity control**): if this matches an
              encoder, the encoder's gain is just the word, not its context
``cls``       the sentence CLS already in ``.cond`` -- the **null control**
============  ================================================================

Measured (3 000 clips, 30 098 words, 5 speaker-grouped folds; mean increment in
out-of-fold R2 over ``phones + speaker``)::

    bert      +0.0179   (dur +0.017  pitch +0.029  range +0.008)
    pos       +0.0095   (dur +0.013  pitch +0.008  range +0.008)
    bge_tok   +0.0067   (dur +0.008  pitch +0.006  range +0.006)
    wordhash  +0.0018   (dur +0.004  pitch +0.000  range +0.001)
    cls       +0.0013   (dur +0.002  pitch -0.001  range +0.003)

so: word-level context does help (unlike the sentence CLS), BERT-base is the
best of the three, POS gets half of it for free, and the gain is *not* lexical
identity -- the spelling hash is 46 % predictable from the phones yet adds
nothing.  Pitch is where the relative gain is largest (base R2 0.055 -> 0.084).

Feature sets are cumulative and cross-validated **grouped by speaker** (a
speaker's own habits cannot leak into another speaker's fold):

    phones            bag of phone ids + log length
    + speaker         one-hot speaker id
    + candidate       one of the four above

Read the **increment** over ``phones + speaker``.  Also reports how much of each
candidate is predictable from ``phones + speaker`` (redundancy -- a candidate
that is fully explained by the pronunciation cannot add anything).

CPU/GPU, read-only:

    python scratch/hint_word_semantics_probe.py --n 1500
    python scratch/hint_word_semantics_probe.py --n 3000 --folds 5 --alpha 3
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_codes import cond_path_for  # noqa: E402
from features import transcript_words  # noqa: E402
from horsephonemizer import HorsePhonemizer  # noqa: E402

SOURCE_FILELISTS = [
    "filelists_2/genshin_full_eng_text_linux.txt",
    "filelists_2/mlp_linux.txt",
    "filelists_2/vctk_linux.txt",
    "filelists_2/expresso_read_linux.txt",
]
PHONE_BINS = 128          # top ids by frequency; everything else -> one bin
MIN_TAG_COUNT = 30
WORD_HASH_BINS = 512      # lexical-identity control: crc32(word) hashed
MAX_WORDS = 200           # tokenizer window (a few clips are longer; skipped)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
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


def read_transcripts(paths):
    out = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3:
                    continue
                text = parts[2].strip()
                if text:
                    out[os.path.basename(parts[0])] = text
    return out


def load_clips(rows, texts):
    """Per clip: words, per-word phone ids, sid, and the three word targets."""
    clips, skipped = [], collections.Counter()
    for parts in rows:
        phone_path, spec_path, pitch_path, sid = parts[:4]
        base = os.path.basename(phone_path)[: -len(".phone_ids")]
        text = texts.get(base)
        if not text:
            skipped["no transcript"] += 1
            continue
        words = [w for w, _ in transcript_words(text)]
        if not words or len(words) > MAX_WORDS:
            skipped["empty/long"] += 1
            continue
        try:
            ids = torch.load(phone_path, map_location="cpu",
                             weights_only=True).reshape(-1).numpy()
            cond = torch.load(cond_path_for(phone_path), map_location="cpu",
                              weights_only=True, mmap=True)
        except Exception:
            skipped["unreadable"] += 1
            continue
        if "phone_word_idx" not in cond or "n_phone" not in cond:
            skipped["no word fields"] += 1
            continue
        idx = cond["phone_word_idx"].long().numpy()
        n_word = int(cond["word_start_f"].numel())
        if idx.size != ids.size or n_word != len(words) or idx.max(initial=-1) >= n_word:
            skipped["word/phone mismatch"] += 1
            continue
        # phones per word
        per_word = [[] for _ in range(n_word)]
        for pid, wid in zip(ids.tolist(), idx.tolist()):
            per_word[wid].append(int(pid))
        if any(len(p) == 0 for p in per_word):
            skipped["word without phones"] += 1
            continue

        slot = cond["word_slot_f"].float().numpy()
        pst = cond["pitch_word_st"].float().numpy()
        voiced = cond["pitch_word_voiced"].bool().numpy()
        if "pitch_anchors_st" in cond:
            a_st = cond["pitch_anchors_st"].float().numpy()
            a_n = cond["pitch_anchor_n"].long().numpy()
            rng = np.zeros(n_word, dtype=np.float32)
            for w in range(n_word):
                k = int(a_n[w])
                if k >= 2:
                    rng[w] = float(a_st[w, :k].max() - a_st[w, :k].min())
        else:
            rng = np.zeros(n_word, dtype=np.float32)
        cls = (cond["semantic"].float().numpy()
               if "semantic" in cond else None)

        clips.append({
            "words": words, "phones": per_word, "sid": int(sid),
            "dur": np.log1p(np.maximum(slot, 1.0)),
            "pitch": np.where(voiced, pst, np.nan),
            "range": np.where(voiced, rng, np.nan),
            "cls": cls,
        })
    return clips, skipped


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_words(model, tok, word_lists, device, batch=16):
    """Per-word mean of the subword states (``(n_words, D)`` per clip)."""
    out = []
    for i in range(0, len(word_lists), batch):
        chunk = word_lists[i:i + batch]
        enc = tok(chunk, is_split_into_words=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(device)
        hidden = model(**enc).last_hidden_state.float().cpu().numpy()
        for b, words in enumerate(chunk):
            ids = enc.word_ids(batch_index=b)
            vecs = np.zeros((len(words), hidden.shape[-1]), dtype=np.float32)
            buckets = collections.defaultdict(list)
            for j, wid in enumerate(ids):
                if wid is not None:
                    buckets[wid].append(j)
            for w, js in buckets.items():
                vecs[w] = hidden[b, js].mean(0)
            out.append(vecs)
    return out


# --------------------------------------------------------------------------- #
# sparse/categorical feature helpers
# --------------------------------------------------------------------------- #
def one_hot(values, vocab):
    """``values`` -> dense one-hot, ``(n, len(vocab))`` float32."""
    pos = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for i, v in enumerate(values):
        j = pos.get(v)
        if j is not None:
            out[i, j] = 1.0
    return out


def phone_hist(phone_lists):
    """Bag of phone ids (one bin per frequent id, plus an ``other`` bin)."""
    counts = collections.Counter(p for lst in phone_lists for p in lst)
    vocab = [p for p, _ in counts.most_common(PHONE_BINS)]
    pos = {p: i for i, p in enumerate(vocab)}
    out = np.zeros((len(phone_lists), len(vocab) + 1), dtype=np.float32)
    for i, lst in enumerate(phone_lists):
        for p in lst:
            out[i, pos.get(p, len(vocab))] += 1.0
        n = max(len(lst), 1)
        out[i, len(vocab)] /= n                       # normalise the other bin
    extra = np.log1p(np.array([len(l) for l in phone_lists],
                              dtype=np.float32)).reshape(-1, 1)
    return np.concatenate([out, extra], axis=1)


def word_hash(words, bins=WORD_HASH_BINS):
    """Hashed word form: the lexical-identity control.

    If a 64-bin hash of the *spelling* buys as much prosody as a contextual
    encoder, the encoder's gain is word identity (predictable from the phones
    anyway), not context -- and a small lookup table would do.
    """
    import zlib

    out = np.zeros((len(words), bins), dtype=np.float32)
    for i, w in enumerate(words):
        out[i, zlib.crc32(str(w).lower().encode()) % bins] = 1.0
    return out


def r2(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="data/genshin_plus_extras/train_stage2.txt")
    ap.add_argument("--n", type=int, default=1500, help="clips to load")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip_bert", action="store_true")
    ap.add_argument("--pca", type=int, default=32,
                    help="leading PCs used for a dense candidate (a raw 768-d "
                         "state is mostly noise at this sample size); 0 = all")
    ap.add_argument("--pca_list", default=None,
                    help="comma-separated widths (or 'raw') for a saturation "
                         "sweep of the dense candidates, e.g. '8,16,32,64,128,256,raw'")
    ap.add_argument("--alphas", default="3,30",
                    help="Ridge alphas tried per width in the sweep")
    args = ap.parse_args(argv)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device or
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    rows = read_rows(args.filelist, args.n)
    texts = read_transcripts(SOURCE_FILELISTS)
    clips, skipped = load_clips(rows, texts)
    print(f"{len(clips)} usable clip(s) of {len(rows)} requested"
          + (f"  (skipped: {dict(skipped)})" if skipped else ""))

    word_lists = [c["words"] for c in clips]
    # --- POS ---------------------------------------------------------------
    tags_per_clip = [HorsePhonemizer.tag_words(w) for w in word_lists]
    all_tags = collections.Counter(t for tags in tags_per_clip for t in tags)
    tag_vocab = [t for t, n in all_tags.most_common() if n >= MIN_TAG_COUNT]
    print(f"POS: {len(tag_vocab)} tag(s) with >= {MIN_TAG_COUNT} occurrences "
          f"of {len(all_tags)}")

    # --- contextual encoders ----------------------------------------------
    def run_encoder(name, words):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device).eval()
        out = encode_words(model, tok, word_lists, device)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return out

    bert_per_clip = None
    if not args.skip_bert:
        print("encoding bert-base-uncased ...")
        bert_per_clip = run_encoder("bert-base-uncased", word_lists)
    print("encoding BAAI/bge-small-en-v1.5 token states ...")
    bge_per_clip = run_encoder("BAAI/bge-small-en-v1.5", word_lists)

    # --- flatten to word rows ---------------------------------------------
    feats = collections.defaultdict(list)
    targets = collections.defaultdict(list)
    groups = []
    for i, c in enumerate(clips):
        n = len(c["words"])
        for w in range(n):
            feats["phones"].append(c["phones"][w])
            feats["sid"].append(c["sid"])
            feats["pos"].append(tags_per_clip[i][w])
            feats["word"].append(c["words"][w])
            if bert_per_clip is not None:
                feats["bert"].append(bert_per_clip[i][w])
            feats["bge_tok"].append(bge_per_clip[i][w])
            if c["cls"] is not None:
                feats["cls"].append(c["cls"])
            targets["dur"].append(c["dur"][w])
            targets["pitch"].append(c["pitch"][w])
            targets["range"].append(c["range"][w])
            groups.append(c["sid"])
    G = np.asarray(groups)
    n_rows = len(G)
    print(f"{n_rows} word row(s), {len(np.unique(G))} speaker(s)")

    X_phone = phone_hist(feats["phones"])
    sids = sorted(set(feats["sid"]))
    X_sid = one_hot(feats["sid"], sids)
    X_base = np.concatenate([X_phone, X_sid], axis=1)
    cands = {
        "pos": one_hot(feats["pos"], tag_vocab),
        "wordhash": word_hash(feats["word"]),
        "bge_tok": np.stack(feats["bge_tok"]),
    }
    if bert_per_clip is not None:
        cands["bert"] = np.stack(feats["bert"])
    if len(feats["cls"]) == n_rows:
        cands["cls"] = np.stack(feats["cls"])
    else:
        print("  (.cond has no sentence CLS for every row -- control arm skipped)")
    cands_full = {k: v.copy() for k, v in cands.items()}
    if args.pca:
        from sklearn.decomposition import PCA
        for name in list(cands):
            Xc = cands[name]
            if Xc.shape[1] > args.pca:
                cands[name] = PCA(n_components=args.pca,
                                  random_state=0).fit_transform(Xc)
    print("feature widths: base "
          + f"{X_base.shape[1]} "
          + "  ".join(f"{k} {v.shape[1]}" for k, v in cands.items()))

    def oof_r2(X, y, groups, alpha=None):
        pred = np.empty(len(y), dtype=np.float64)
        gkf = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            m = Ridge(alpha=(args.alpha if alpha is None else alpha),
                      solver="lsqr")
            m.fit(Xtr, y[tr])
            pred[te] = m.predict(Xte)
        return r2(y, pred)

    # --- redundancy: how much of each candidate is already in phones+sid? --
    print("\nredundancy (R2 of the candidate's leading PC from phones+speaker):")
    for name, Xc in cands.items():
        if Xc.shape[1] > 64:                # dense: use the leading component
            from sklearn.decomposition import PCA
            y = PCA(n_components=1).fit_transform(Xc).reshape(-1)
        else:
            y = Xc.mean(1)
        print(f"  {name:<8s} R2 {oof_r2(X_base, y, G):+.3f}")

    # --- the decision table ----------------------------------------------
    print("\nword-prosody prediction (out-of-fold R2, grouped by speaker)")
    print("  base = phones + speaker; increments are over base")
    head = f"  {'target':<8s} {'base':>7s}" + "".join(
        f"{'+' + k:>10s}" for k in cands)
    print(head)
    verdict = collections.defaultdict(list)
    for tname, vals in targets.items():
        y = np.asarray(vals, dtype=np.float64)
        ok = np.isfinite(y)
        if ok.sum() < 100:
            continue
        base = oof_r2(X_base[ok], y[ok], G[ok])
        row = f"  {tname:<8s} {base:>7.3f}"
        for name, Xc in cands.items():
            inc = oof_r2(np.concatenate([X_base[ok], Xc[ok]], axis=1),
                         y[ok], G[ok]) - base
            verdict[name].append(inc)
            row += f"{inc:>+10.3f}"
        print(row + f"   (n={int(ok.sum())})")

    print("\nmean increment over the three targets (the ranking):")
    for name, incs in sorted(verdict.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {name:<8s} {np.mean(incs):+.4f}   "
              f"(per target: " + " ".join(f"{v:+.3f}" for v in incs) + ")")
    if args.pca_list:
        from sklearn.decomposition import PCA
        print("\nPCA/dim sweep (mean increment over the 3 targets, dense only;"
              "\npos is fixed-width and shown for reference).  'raw' = no"
              " projection.")
        widths = [w.strip() for w in str(args.pca_list).split(",") if w.strip()]
        alphas = [float(a) for a in str(args.alphas).split(",") if a.strip()]
        masks = {t: np.isfinite(np.asarray(v, dtype=np.float64))
                 for t, v in targets.items()}
        dense = [k for k, v in cands_full.items()
                 if v.shape[1] > 64 and k in ("bert", "bge_tok")]
        fixed = [k for k in cands_full if k not in dense]
        base_cache = {}
        for a in alphas:
            for t, vals in targets.items():
                ok = masks[t]
                if ok.sum() < 100:
                    continue
                y = np.asarray(vals, dtype=np.float64)[ok]
                base_cache[(a, t)] = oof_r2(X_base[ok], y, G[ok], a)

        def incs_for(Xc, a):
            out = []
            for t, vals in targets.items():
                ok = masks[t]
                if ok.sum() < 100:
                    continue
                y = np.asarray(vals, dtype=np.float64)[ok]
                out.append(oof_r2(np.concatenate([X_base[ok], Xc[ok]], 1),
                                  y, G[ok], a) - base_cache[(a, t)])
            return out

        print("  width  " + "".join(f"{k:>9s}" for k in dense)
              + " | bert per target: dur / pitch / range")
        best_width = {}
        for w in widths:
            row = f"  {w:>5s}  "
            bert_incs = None
            for name in dense:
                Xc = cands_full[name]
                if w != "raw" and Xc.shape[1] > int(w):
                    Xc = PCA(n_components=int(w), random_state=0
                             ).fit_transform(Xc)
                best = None
                for a in alphas:
                    incs = incs_for(Xc, a)
                    m = float(np.mean(incs))
                    if best is None or m > best[1]:
                        best = (a, m, incs)
                row += f"{best[1]:>+9.4f}"
                if name == "bert":
                    bert_incs = best[2]
                best_width[(w, name)] = best
            if bert_incs is not None:
                row += " | " + "  ".join(f"{v:+.3f}" for v in bert_incs)
            print(row)
        for name in fixed:
            Xc = cands_full[name]
            best = max((((a,) + (float(np.mean(incs_for(Xc, a))),
                                 incs_for(Xc, a)))
                        for a in alphas), key=lambda z: z[1])
            print(f"  {'fixed':>5s}  {name}={best[1]:+.4f} "
                  f"({Xc.shape[1]} dims, alpha {best[0]:g})")

    print("\nread: the sentence CLS is the control and must sit at ~0.  A")
    print("candidate is only worth wiring if its increment is clearly positive")
    print("on duration AND pitch (the two things the codes carry).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
