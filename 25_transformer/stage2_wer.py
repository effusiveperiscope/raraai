"""Shared end-to-end WER machinery for stage 2.

Both ``scratch/eval_stage2_wer.py`` (the offline eval matrix) and
``2_training.py`` (the per-epoch probe) import this module, so there is exactly
one implementation of "codes -> stage-1 audio -> ASR -> WER".  Keeping it here
rather than in the scratch script is what makes the training-time number
comparable to the offline one: the decoding modes (oracle length vs EOS stop),
the text normalisation and the WER definition are shared.

Pipeline per val item (one row of the stage-2 filelist)::

    phone_ids --> CodeTransformer.generate --> (spec, pitch) codes
              --> frozen stage-1 Generator.infer_from_codes --> 48 kHz wav
              --> Qwen3-ASR (16 kHz) --> text --> WER vs the reference text

Two arms are supported:

``generated``
    codes sampled by the checkpoint under test (plain autoregressive).
``reference``
    the item's *ground-truth* codes through the same stage-1 Generator.  This
    does not depend on the transformer, so it is the floor: the WER contributed
    by the vocoder + ASR, i.e. the best a perfect code predictor could score.

Reference text comes from the *source* stage-2 filelists (``path|sid|text``),
matched to the preprocessed filelist by basename -- the preprocessed tensors are
named ``<basename>.phone_ids`` etc., so no audio path survives into
``data/*/val_stage2.txt``.

Nothing here imports ``2_training`` (the dependency points the other way), and
the heavy ASR import is lazy inside :class:`Asr`, so importing this module costs
nothing at train start.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Preprocessed tensors are named by the source basename plus one of these.
TENSOR_SUFFIXES = (".phone_ids", ".spec_codes", ".pitch_codes")

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")

#: User-facing channel names -> ``CodeTransformer.HINT_CHANNEL_KEYS``.  The
#: mapping itself is not duplicated here: the model owns which batch keys carry
#: a channel, so a new channel cannot drift out of sync with the ablation.
#: ``semantic`` is absent (the sentence-CLS channel was removed in V2.2);
#: ``word_sem`` is the per-word state (V2.1).
HINT_CHANNEL_ALIASES = {"intensity": "int", "register": "reg",
                        "pitch": "pitch", "slot": "slot",
                        "wordsem": "word_sem", "word_sem": "word_sem"}


def hint_channel_keys(name: str):
    """Batch keys of a channel (``KeyError`` for an unknown name)."""
    from modeling.transformer import CodeTransformer

    return CodeTransformer.HINT_CHANNEL_KEYS[HINT_CHANNEL_ALIASES[name]]


# --------------------------------------------------------------------------- #
# Text / WER
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Lowercase, unify apostrophes, keep words only (WER tokenisation)."""
    text = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(_WORD_RE.findall(text))


def align_words(ref: Sequence[str], hyp: Sequence[str]
                ) -> List[Tuple[str, int, int]]:
    """Levenshtein alignment between two word lists, in time order.

    Returns ``(op, ref_index, hyp_index)`` tuples with ``op`` one of
    ``"match"``, ``"sub"``, ``"del"``, ``"ins"``; ``ref_index`` (resp.
    ``hyp_index``) is ``-1`` for an insertion (resp. a deletion).  The DP and
    the tie-breaking (substitution > deletion > insertion) are the ones
    :func:`edit_counts` has always used -- that function is now defined in
    terms of this one, so the headline counts and the position diagnostics
    cannot disagree.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost,   # match / substitute
                           dp[i - 1][j] + 1,          # delete from ref
                           dp[i][j - 1] + 1)          # insert from hyp
    ops: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("match" if ref[i - 1] == hyp[j - 1] else "sub",
                        i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            ops.append(("ins", -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Minimum (substitutions, insertions, deletions) turning ``ref`` into ``hyp``.

    Returned counts give ``WER = (S + D + I) / len(ref)``, i.e. the standard
    definition with ``ref`` as the denominator.  See :func:`align_words` for
    the DP and the tie-breaking.
    """
    sub = ins = dele = 0
    for op, _, _ in align_words(ref, hyp):
        if op == "sub":
            sub += 1
        elif op == "ins":
            ins += 1
        elif op == "del":
            dele += 1
    return sub, ins, dele


def item_errors(ref: str, hyp: str) -> Dict[str, float]:
    """Per-utterance error counts -- the unit of the paired bootstrap."""
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    a, b, c = edit_counts(r, h)
    return {"sub": a, "ins": b, "del": c, "errors": a + b + c,
            "words": len(r), "hyp_words": len(h)}


def wer_per_item(pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
    """``item_errors`` for every ``(reference, hypothesis)`` pair."""
    return [item_errors(ref, hyp) for ref, hyp in pairs]


def wer_summary(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """Corpus-level WER over ``(reference, hypothesis)`` text pairs."""
    s = i = d = words = 0
    for e in wer_per_item(pairs):
        s, i, d = s + e["sub"], i + e["ins"], d + e["del"]
        words += e["words"]
    errors = s + i + d
    return {"wer": errors / words if words else float("nan"),
            "sub": s, "ins": i, "del": d, "errors": errors, "words": words,
            "n": len(pairs)}


def _rate(count: float, total: float) -> float:
    return count / total if total else float("nan")


def wer_diagnostics(pairs: Sequence[Tuple[str, str]],
                    n_bins: int = 10) -> Dict[str, object]:
    """Where the deletions are, and how long the hypotheses are.

    :func:`wer_summary` says *how many* words are missing; this says *where*.
    A deletion concentrated in the last 20 % of the utterance is the model
    stopping early (EOS / duration), while deletions spread through the middle
    are the "skips a word" failure -- and the two need completely different
    fixes.  ``hyp_words`` length is reported the same way, because a short
    hypothesis produces deletions everywhere by construction.

    Bin ``b`` holds the reference words whose normalised position falls in
    ``[b/n_bins, (b+1)/n_bins)``, so every bin holds about the same number of
    words and ``del_rate_by_bin`` is comparable across bins: a flat profile
    means "no positional bias", a rising one means the tail is being lost.
    """
    counts = [0] * n_bins
    words_in = [0] * n_bins
    words = del_total = 0
    ratios: List[float] = []
    for ref, hyp in pairs:
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        words += len(r)
        if r:
            ratios.append(len(h) / len(r))
        for b in range(len(r)):
            words_in[min(int(n_bins * b / max(len(r), 1)), n_bins - 1)] += 1
        for op, ri, _ in align_words(r, h):
            if op == "del":
                del_total += 1
                counts[min(int(n_bins * ri / max(len(r), 1)), n_bins - 1)] += 1
    n_edge = max(1, n_bins // 5)                       # first / last 20 %
    head, head_w = sum(counts[:n_edge]), sum(words_in[:n_edge])
    tail, tail_w = sum(counts[-n_edge:]), sum(words_in[-n_edge:])
    mid, mid_w = del_total - head - tail, words - head_w - tail_w
    return {
        "words": words, "del": del_total,
        "del_rate": _rate(del_total, words),
        "del_rate_head": _rate(head, head_w),
        "del_rate_mid": _rate(mid, mid_w),
        "del_rate_tail": _rate(tail, tail_w),
        "del_counts_by_bin": counts,
        "del_rate_by_bin": [_rate(c, w) for c, w in zip(counts, words_in)],
        "len_ratio_median": float(np.median(ratios)) if ratios
        else float("nan"),
        "short_frac": (sum(1 for x in ratios if x < 0.8) / len(ratios)
                       if ratios else float("nan")),
        "long_frac": (sum(1 for x in ratios if x > 1.2) / len(ratios)
                      if ratios else float("nan")),
        "n": len(pairs),
    }


def format_diagnostics(diag: Dict[str, object], indent: str = "    ") -> str:
    """One-line-per-fact rendering of :func:`wer_diagnostics`."""
    deciles = " ".join(f"{100.0 * x:5.1f}" for x in diag["del_rate_by_bin"])
    return "\n".join([
        f"{indent}deletions {diag['del']}/{diag['words']} words "
        f"({100.0 * diag['del_rate']:.1f}%)   positions: "
        f"head {100.0 * diag['del_rate_head']:.1f}% / "
        f"mid {100.0 * diag['del_rate_mid']:.1f}% / "
        f"tail {100.0 * diag['del_rate_tail']:.1f}%",
        f"{indent}delete rate by ref decile: {deciles}",
        f"{indent}hyp/ref words: median {diag['len_ratio_median']:.3f}   "
        f"short (<0.8) {100.0 * diag['short_frac']:.1f}%   "
        f"long (>1.2) {100.0 * diag['long_frac']:.1f}%",
    ])


def wer_by_length(pairs: Sequence[Tuple[str, str]], n_bins: int = 4
                  ) -> List[Dict[str, float]]:
    """WER / deletion rate by utterance length (reference word count).

    Splits the items into ``n_bins`` equal-count groups ordered by how many
    words the reference has, and reports the corpus WER and deletion rate of
    each.  A rising curve means the failure is length-dependent -- the model
    loses the thread / wraps up as the sentence grows -- which is a completely
    different diagnosis from a fixed per-word error rate.
    """
    rows = []
    for ref, hyp in pairs:
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        a, b, c = edit_counts(r, h)
        rows.append({"words": len(r), "errors": a + b + c, "del": c,
                     "pair": (ref, hyp)})
    if not rows:
        return []
    rows.sort(key=lambda x: x["words"])
    out: List[Dict[str, float]] = []
    n_bins = max(1, min(n_bins, len(rows)))
    for b in range(n_bins):
        lo = round(b * len(rows) / n_bins)
        hi = round((b + 1) * len(rows) / n_bins)
        chunk = rows[lo:hi]
        if not chunk:
            continue
        words = sum(x["words"] for x in chunk)
        errors = sum(x["errors"] for x in chunk)
        dels = sum(x["del"] for x in chunk)
        out.append({
            "items": len(chunk),
            "words": words,
            "ref_words_mean": words / len(chunk),
            "wer": _rate(errors, words),
            "del_rate": _rate(dels, words),
        })
    return out


def format_by_length(rows: Sequence[Dict[str, float]],
                     indent: str = "    ") -> str:
    """Render :func:`wer_by_length` as one line per length quartile."""
    return "\n".join(
        f"{indent}  {r['ref_words_mean']:5.1f} ref words  "
        f"WER {100.0 * r['wer']:5.1f}%   delete {100.0 * r['del_rate']:4.1f}%"
        f"   [{r['items']} utts]" for r in rows)


def paired_wer_delta(pairs_a: Sequence[Tuple[str, str]],
                     pairs_b: Sequence[Tuple[str, str]], *,
                     n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Paired bootstrap of ``WER(a) - WER(b)`` over the same utterances.

    Both arms must be scored on the *same* utterances in the same order -- true
    for every checkpoint in ``eval_stage2_wer.py``, which reuses the item list
    and the sampling seed.  Resampling *utterances* (not errors) keeps the
    pairing, which is what makes a 1-2 pp difference resolvable at 100-200
    items; comparing two independent corpus WERs at 50 items cannot.

    Returns ``delta`` (a - b; negative means ``a`` is better), its
    ``[lo, hi]`` percentile interval and ``p_better`` (bootstrap fraction with
    ``a`` better).
    """
    a = wer_per_item(pairs_a)
    b = wer_per_item(pairs_b)
    n = min(len(a), len(b))
    if n == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_better": float("nan"), "n_boot": 0}
    ea = np.array([x["errors"] for x in a[:n]], dtype=np.float64)
    wa = np.array([x["words"] for x in a[:n]], dtype=np.float64)
    eb = np.array([x["errors"] for x in b[:n]], dtype=np.float64)
    wb = np.array([x["words"] for x in b[:n]], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    delta = (ea[idx].sum(1) / np.maximum(wa[idx].sum(1), 1)
             - eb[idx].sum(1) / np.maximum(wb[idx].sum(1), 1))
    point = ea.sum() / max(wa.sum(), 1) - eb.sum() / max(wb.sum(), 1)
    lo, hi = np.percentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Ties count half, so two identical arms report 0.5 rather than 0.0.
    p_better = float((delta < 0).mean() + 0.5 * (delta == 0).mean())
    return {"delta": float(point), "lo": float(lo), "hi": float(hi),
            "p_better": p_better, "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Filelists / items
# --------------------------------------------------------------------------- #
def apply_wer_reference_audit(refs: Dict[str, str],
                              audit_path: Optional[str],
                              mode: str = "drop") -> Dict[str, str]:
    """Apply ``audit_transcripts.py``'s ``--out-references`` to a ref dict.

    ``mode="drop"`` (default) removes every clip whose reference the audit could
    not trust (truncated/mismatched/ASR-short) and keeps the original text for
    the rest; this is the honest benchmark -- the ASR text would be circular
    (generated audio scored against an ASR transcript) and changes the word
    denominator.  ``mode="override"`` instead substitutes the ASR text for a
    truncated/mismatched line and drops only the uncertain ones.
    """
    if not audit_path or not os.path.exists(audit_path):
        return refs
    import json

    if mode not in ("drop", "override"):
        raise ValueError(f"reference audit mode must be drop|override, got "
                         f"{mode!r}")
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    untrusted = {"prefix_truncated", "suffix_truncated", "mismatch",
                 "asr_short", "asr_empty", "empty_reference", "audio_missing"}
    n_text = n_drop = 0
    for basename, rec in audit.items():
        if not isinstance(rec, dict):
            continue
        cls = rec.get("class", "")
        if mode == "drop":
            if cls in untrusted and refs.pop(basename, None) is not None:
                n_drop += 1
            continue
        if rec.get("drop") or cls in ("asr_empty", "empty_reference",
                                      "audio_missing"):
            if refs.pop(basename, None) is not None:
                n_drop += 1
        elif rec.get("text"):
            if basename in refs and refs[basename] != rec["text"]:
                n_text += 1
            refs[basename] = rec["text"]
    print(f"WER reference audit ({mode}): {n_text} corrected, {n_drop} "
          f"dropped ({audit_path})")
    return refs


def item_basename(path: str) -> str:
    """``data/x/foo.wav.phone_ids`` -> ``foo.wav`` (the source basename)."""
    base = os.path.basename(path)
    for suf in TENSOR_SUFFIXES:
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def load_references(paths: Sequence[str]) -> Dict[str, str]:
    """basename -> transcript from source filelists (``path|sid|text``)."""
    refs: Dict[str, str] = {}
    clashes = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "longform" in line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                try:
                    int(parts[1])
                except ValueError:
                    continue
                key = os.path.basename(parts[0].replace("\\", "/"))
                text = parts[2].strip()
                if not text:
                    continue
                if key in refs and refs[key] != text:
                    clashes += 1
                    continue
                refs.setdefault(key, text)
    if clashes:
        print(f"NOTE: {clashes} basename collision(s) with differing text in the "
              f"source filelists; the first one wins.")
    return refs


def resolve_source_filelists(explicit: Optional[Sequence[str]] = None
                             ) -> List[str]:
    """``--source_filelists`` or every ``filelists_2/*.txt``.

    A config may write a single path as a scalar string rather than a one-item
    list; ``list("filelists_2/x.txt")`` would silently turn it into a list of
    characters, so unwrap it here.
    """
    if isinstance(explicit, str):
        explicit = [explicit]
    return list(explicit or sorted(glob.glob("filelists_2/*.txt")))


def pick_indices(avail: List[int], n: int) -> List[int]:
    """``n`` evenly spaced entries of ``avail`` (deterministic, order-preserving)."""
    if n <= 0 or n >= len(avail):
        return list(avail)
    if n == 1:
        return [avail[len(avail) // 2]]
    return [avail[round(j * (len(avail) - 1) / (n - 1))] for j in range(n)]


def chunked(seq: Sequence, size: int):
    for i in range(0, len(seq), max(size, 1)):
        yield seq[i:i + max(size, 1)]


def build_wer_dataset(config, *, hints: bool = True, blocklists: bool = True,
                      duration: bool = True):
    """A whole-utterance ``CodeDataset`` for WER (no ``max_frames`` clamp).

    The training/validation datasets crop to ``train.max_frames``; WER must score
    the utterance the reference text describes.  Hint channels are built when
    the model has them (``hints=False`` reproduces the old, hint-less numbers),
    exactly like ``scratch/eval_stage2_wer.py`` does.

    ``duration`` emits the oracle phone-progress schedule
    (``transformer.duration_cond``) for the whole utterance.  Pass ``False`` to
    score the same checkpoint in the trained "no schedule" state, which is the
    control arm for the duration-channel prototype.
    """
    from dataset_codes import CodeDataset

    tc = config.transformer
    hint_kwargs = dict(
        hints=bool(tc.get("hints", False)) if hints else False,
        hints_json=tc.get("hints_json", None),
        hint_pitch_edges=tc.get("hint_pitch_edges", None),
        n_slot_buckets=int(tc.get("n_slot_buckets", 8) or 8),
    )
    return CodeDataset(
        config.train.val_filelist,
        is_train=False,
        max_frames=None,
        max_phone_length=None,
        speaker_map=config.train.get("speaker_map", None),
        # The schedule is used by two independent features: as a decoder input
        # (``duration_cond``) and as the alignment target
        # (``align_supervision``).  Emit it when either is on.
        duration=(bool(tc.get("duration_cond", False))
                  or bool(tc.get("align_supervision", False))) and duration,
        duration_dropout=0.0,
        alignment_blocklist=(config.train.get("alignment_blocklist", None)
                             if blocklists else None),
        stage_direction_blocklist=(
            config.train.get("stage_direction_blocklist", None)
            if blocklists else None),
        **hint_kwargs,
    )


def select_items(dataset, refs: Dict[str, str], num_items: int,
                 sid_counts: Optional[Dict[int, int]] = None,
                 buckets: Optional[Sequence[str]] = None,
                 avail: Optional[Sequence[int]] = None,
                 require_ref: bool = True
                 ) -> Tuple[List[int], List[dict], List[str]]:
    """Pick ``num_items`` val rows that have a reference transcript.

    ``avail`` optionally fixes the candidate pool *before* a reference audit
    dropped/replaced entries.  With ``require_ref=False`` the evenly spaced pick
    is made over the whole pre-audit pool and dropped entries are then removed
    from it, so the audit arm scores a *subset of the raw arm's utterances*
    (paired) instead of shifting every index.
    """
    if avail is None:
        avail = [i for i in range(len(dataset))
                 if item_basename(dataset.items[i]["spec_codes"]) in refs]
    elif require_ref:
        avail = [i for i in avail
                 if item_basename(dataset.items[i]["spec_codes"]) in refs]
    if buckets:
        allowed = set(buckets)
        counts = sid_counts or {}
        avail = [i for i in avail
                 if speaker_bucket(counts.get(int(dataset.items[i]["sid"]), 0))
                 in allowed]
    if not avail:
        raise RuntimeError(
            "No val item has a reference transcript.  Pass --source_filelists "
            "with the filelist(s) the val set came from (they need "
            "'path|sid|text' lines).")
    picked = pick_indices(avail, num_items)
    if not require_ref:
        picked = [i for i in picked
                  if item_basename(dataset.items[i]["spec_codes"]) in refs]
    items = [dataset[i] for i in picked]
    texts = [refs[item_basename(dataset.items[i]["spec_codes"])]
             for i in picked]
    return picked, items, texts


def speaker_clip_counts(filelist: str) -> Dict[int, int]:
    """Stage-2 speaker id -> number of training clips (``path|...|sid``)."""
    counts: Dict[int, int] = {}
    if not filelist or not os.path.exists(filelist):
        return counts
    with open(filelist, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "longform" in line:
                continue
            parts = line.split("|")
            try:
                sid = int(parts[-1])
            except (IndexError, ValueError):
                continue
            counts[sid] = counts.get(sid, 0) + 1
    return counts


def speaker_bucket(n: int) -> str:
    """Training-clip bucket label for a speaker (0 = id not in the counts)."""
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 20:
        return "6-20"
    if n <= 100:
        return "21-100"
    return "100+"


def grouped_wer(pairs, groups) -> Dict[str, Dict[str, float]]:
    """``wer_summary`` per group label (groups must align with ``pairs``)."""
    out: Dict[str, Dict[str, float]] = {}
    by: Dict[str, List[Tuple[str, str]]] = {}
    for pair, g in zip(pairs, groups):
        if g is not None:
            by.setdefault(g, []).append(pair)
    for g, ps in by.items():
        out[g] = wer_summary(ps)
    return out


# --------------------------------------------------------------------------- #
# Audio / ASR
# --------------------------------------------------------------------------- #
def code_frame_samples(net_g) -> int:
    """Audio samples per *code* frame: ``hop_length * down_factor`` (40 ms here).

    Code frames are ``data.hop_length`` mel frames wide but the codec decimates
    the decoder rate by ``Generator.down_factor`` (4), so a code frame is
    4 x 10 ms and ``infer_from_codes`` returns ``T * 1920`` samples at 48 kHz.
    Using the bare ``hop_length`` here truncates the audio to 25 %.
    """
    hop = int(net_g.hp.data.hop_length)
    return hop * int(getattr(net_g, "down_factor", 1))


class Asr:
    """Thin wrapper around ``qwen_asr.Qwen3ASRModel`` (text only)."""

    def __init__(self, model_name, device, language,
                 max_inference_batch_size: int = 8):
        from qwen_asr import Qwen3ASRModel

        self.language = language
        self.model = Qwen3ASRModel.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=str(device),
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=256)

    def transcribe(self, wavs16k: Sequence[np.ndarray]) -> List[str]:
        audio = [(np.ascontiguousarray(w, dtype=np.float32), 16000)
                 for w in wavs16k]
        outs = self.model.transcribe(audio=audio, language=self.language)
        return [getattr(o, "text", "") or "" for o in outs]


def to_16k(wavs, src_rate):
    """48 kHz float32 numpy -> 16 kHz (torchaudio polyphase resampler)."""
    import torchaudio

    out = []
    for w in wavs:
        t = torch.from_numpy(np.clip(np.nan_to_num(w), -1.0, 1.0))
        if src_rate != 16000:
            t = torchaudio.functional.resample(t, src_rate, 16000)
        out.append(t.cpu().numpy().astype(np.float32))
    return out


# --------------------------------------------------------------------------- #
# Code generation / decoding
# --------------------------------------------------------------------------- #
def resolve_generator_path(path: str):
    """Accept a full path, a name inside ``checkpoints_stage1/``, or a run/name."""
    if not path:
        return None
    cands = [path, os.path.join("checkpoints_stage1", path)]
    if not os.path.isabs(path) and os.sep not in path:
        cands += sorted(glob.glob(os.path.join("checkpoints_stage1", "*",
                                               path)))
    for cand in cands:
        if os.path.exists(cand):
            return cand
    return None


def _impute_speaker_rows(weight: torch.Tensor, counts: Dict[int, int], *,
                         min_clips: int = 6, mode: str = "mean") -> int:
    """Replace rows of ``weight`` for speakers with < ``min_clips`` clips."""
    if mode not in ("mean", "zero"):
        raise ValueError(f"speaker fallback mode must be mean|zero, got {mode!r}")
    n = weight.shape[0]
    keep = torch.tensor([counts.get(i, 0) >= min_clips for i in range(n)],
                        dtype=torch.bool, device=weight.device)
    if not bool(keep.any()):
        return 0
    fill = (torch.zeros_like(weight[:1]) if mode == "zero"
            else weight[keep].mean(dim=0, keepdim=True))
    idx = (~keep).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        weight[idx] = fill.to(weight.dtype)
    return int(idx.numel())


def stage1_speaker_fallback(net_g, counts: Dict[int, int], *,
                            min_clips: int = 6, mode: str = "mean") -> int:
    """Replace under-trained stage-1 speaker embeddings (in place).

    The frozen ``Generator.spk_emb`` is a per-id lookup table trained on the
    stage-1 corpus.  A speaker with a handful of clips (or none at all) leaves
    its row close to the random initialisation, so the vocoder renders a
    meaningless voice -- measured here as ASR hallucinations on near-silent /
    babble audio, i.e. a WER floor above 100 %.  No stage-2 change can repair
    that, because the transformer's codes are decoded with *that* row.

    ``mode="mean"`` replaces every row whose stage-1 clip count is below
    ``min_clips`` with the mean of the well-trained rows (the population voice);
    ``"zero"`` uses the zero vector (the historical additive offset's neutral
    point).  This is a diagnostic / stop-gap for evaluation: it changes the
    voice, so compare buckets and listen, it is not a substitute for training a
    speaker encoder or a real few-shot embedding.

    Returns the number of rows replaced.
    """
    return _impute_speaker_rows(net_g.spk_emb.weight, counts,
                                min_clips=min_clips, mode=mode)


def transformer_speaker_fallback(net, counts: Dict[int, int], *,
                                 min_clips: int = 6, mode: str = "mean") -> int:
    """Same imputation for the stage-2 encoder/decoder ``spk_emb`` tables.

    A low-data speaker's stage-2 row is also under-trained (measured: row norms
    scale monotonically with clip count), so the transformer gets little or no
    voice conditioning for exactly the speakers that need it most.  Imputing the
    population row gives them a meaningful starting point; ``"zero"`` is the
    trained "absent" state the hint tables use.  Evaluation-time control only --
    a real fix is to initialise/regularise these rows during training.
    """
    n = 0
    for module in (getattr(net, "encoder", None), getattr(net, "decoder", None)):
        emb = getattr(module, "spk_emb", None) if module is not None else None
        if emb is not None:
            n += _impute_speaker_rows(emb.weight, counts,
                                      min_clips=min_clips, mode=mode)
    return n


def peek_layout(path) -> Dict[str, object]:
    """Layout flags stored in a checkpoint (no weights loaded).

    ``rope`` changes the attention function, so the eval scripts use this to
    rebuild/patch the model per checkpoint.
    """
    from modeling.transformer import uses_rope

    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    return {"rope": uses_rope(sd)}


def load_sid_map(path: Optional[str]) -> Dict[int, int]:
    """Load the ``sid_map`` of a ``speaker_map.py`` JSON (stage-2 -> stage-1)."""
    if not path:
        return {}
    import json

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    raw = obj.get("sid_map", obj) if isinstance(obj, dict) else {}
    return {int(k): int(v) for k, v in raw.items()}


def map_sids(sids, sid_map: Optional[Dict[int, int]]):
    """Apply a stage-2 -> stage-1 ``sid_map`` (identity for unmapped ids)."""
    if not sid_map:
        return list(sids)
    return [int(sid_map.get(int(s), int(s))) for s in sids]


@torch.no_grad()
def decode_codes(net_g, codes: torch.Tensor, sids: torch.Tensor) -> List[np.ndarray]:
    """``(B, K, T)`` codes -> a list of 48 kHz float32 numpy waveforms.

    ``K`` is split into the spec codebooks followed by the pitch ones (the
    stage-2 layout).  Every row is trimmed to its own frame count.
    """
    n_spec = int(net_g.hp.codec.codebook_num)
    hop = code_frame_samples(net_g)
    spec = codes[:, :n_spec].transpose(0, 1).contiguous()       # (Q, B, T)
    pitch = codes[:, n_spec:].transpose(0, 1).contiguous()
    audio = net_g.infer_from_codes(spec, pitch, sids)           # (B, 1, N)
    out = []
    for i in range(codes.shape[0]):
        n = int(codes.shape[-1]) * hop
        out.append(audio[i, 0, :n].float().cpu().numpy())
    return out


@torch.no_grad()
def synthesize_generated(net, net_g, items, pad_id, device, *,
                         batch_size: int = 4, max_frames: int = 1000,
                         temperature: float = 0.9,
                         top_k: Optional[int] = None,
                         stop_on_eos: bool = True,
                         hint_arg_fn=None,
                         hint_guidance: Optional[float] = None,
                         hint_guidance_books=None,
                         sid_map: Optional[Dict[int, int]] = None,
                         return_lengths: bool = False,
                         duration: bool = False):
    """Sample codes per item (oracle length cap) and decode them to audio.

    ``hint_arg_fn(chunk)`` (a ``CodeCollator`` closure, typically) turns the
    chunk into the hint dict the model expects; without it generation runs in
    the model's **trained null state** (level 0 everywhere), the state
    ``hint_dropout`` trains and the one ``resolve_hints(None)`` maps to.

    Returns the waveforms, or ``(wavs, lengths)`` when ``return_lengths`` is
    set -- ``lengths[i]`` is the number of *code frames* the model produced for
    item ``i`` (its own EOS decision unless ``stop_on_eos=False``).  Comparing
    that with the item's ground-truth frame count is the direct "is it
    under-generating?" measurement.
    """
    wavs = []
    out_lengths: List[int] = []
    hop = code_frame_samples(net_g)
    for chunk in chunked(items, batch_size):
        B = len(chunk)
        Qs = int(chunk[0]["spec_codes"].shape[0])
        Qp = int(chunk[0]["pitch_codes"].shape[0])
        P = max(int(it["phone_ids"].numel()) for it in chunk)
        phone = torch.full((B, P), pad_id, dtype=torch.long, device=device)
        plen = torch.zeros(B, dtype=torch.long, device=device)
        sid = torch.zeros(B, dtype=torch.long, device=device)
        lens = []
        for i, it in enumerate(chunk):
            p = it["phone_ids"]
            phone[i, : p.numel()] = p.to(device)
            plen[i] = p.numel()
            sid[i] = int(it["sid"])
            lens.append(min(int(it["spec_codes"].shape[1]), max_frames))
        prog = prog_mask = None
        if duration:
            # Oracle phone-progress schedule, one row per item, padded to the
            # longest schedule in the chunk (column tags are masked per frame).
            widths = [min(int(it["dur_prog"].numel()), max_frames)
                      for it in chunk]
            Tp = max(widths)
            prog = torch.zeros(B, Tp, dtype=torch.float32)
            prog_mask = torch.zeros(B, Tp, dtype=torch.bool)
            for i, (it, w) in enumerate(zip(chunk, widths)):
                if w > 0:
                    prog[i, :w] = it["dur_prog"][:w].float()
                    prog_mask[i, :w] = it["dur_prog_mask"][:w].bool()
            prog, prog_mask = prog.to(device), prog_mask.to(device)
        gen, glen = net.generate(
            phone, plen, sid, max_frames=max(lens),
            temperature=temperature, top_k=top_k,
            stop_on_eos=(None if stop_on_eos else False),
            hints=(hint_arg_fn(chunk) if hint_arg_fn is not None else None),
            hint_guidance=hint_guidance,
            hint_guidance_books=hint_guidance_books,
            progress=prog, progress_mask=prog_mask,
            return_lengths=True)
        spec = gen[:, :Qs].transpose(0, 1).contiguous()
        pitch = gen[:, Qs:Qs + Qp].transpose(0, 1).contiguous()
        # The transformer is conditioned on its own (stage-2) ids, as trained.
        # The frozen Generator's voice is a stage-1 id, which is a *different*
        # space: pass the mapped ids to `infer_from_codes` only.
        gen_sid = torch.tensor(
            map_sids([int(it["sid"]) for it in chunk], sid_map),
            dtype=torch.long, device=device)
        audio = net_g.infer_from_codes(spec, pitch, gen_sid)     # (B, 1, N)
        for i in range(B):
            n = min(int(glen[i]), lens[i]) * hop
            out_lengths.append(min(int(glen[i]), lens[i]))
            wavs.append(audio[i, 0, :n].float().cpu().numpy())
    if return_lengths:
        return wavs, out_lengths
    return wavs


@torch.no_grad()
def synthesize_reference(net_g, items, device,
                         sid_map: Optional[Dict[int, int]] = None
                         ) -> List[np.ndarray]:
    """Decode the item's ground-truth codes (the WER floor / control arm)."""
    wavs = []
    hop = code_frame_samples(net_g)
    for it in items:
        codes = torch.cat([it["spec_codes"], it["pitch_codes"]], dim=0)
        sid = map_sids([int(it["sid"])], sid_map)
        wavs.extend(decode_codes(
            net_g, codes.unsqueeze(0).to(device),
            torch.tensor(sid, dtype=torch.long, device=device)))
    return wavs
