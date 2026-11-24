"""Audit Genshin transcripts against ASR, for stage-2 supervision/eval cleaning.

The v2 phones were built with ``2_preprocess.py --only phones --phones_from
transcript`` (``scratch/v2_pipeline/v2_rebuild.sh``), so the filelist text is
*the* stage-2 text conditioning.  Many lines are fragments (16.3 % end in
``...``), while the audio often contains the whole sentence: the model is then
trained to say only the prefix of what it hears, and a WER reference is wrong
by construction.

This script ASRs the **original source audio** (not decoded codes), compares it
with the filelist transcript, and classifies each clip:

  match             low WER -- keep, even if the line ends in ``...``
  prefix_truncated  the transcript is a word-prefix of the ASR text (audio has
                    extra words) -- the tell is real
  mismatch          substantial word disagreement, not a prefix/suffix
  asr_short         ASR heard much less than the transcript (ambiguous)
  review            neither a clear match nor a clear mismatch
  audio_missing / empty_reference / asr_empty

It is resumable: every finished clip is appended to ``--cache`` and a re-run
only processes cache misses.  Outputs are a drop list (``--out-drops``) and a
per-basename WER-reference override file (``--out-references``).

Usage::

    # calibrate on the flagged subset + 1k random non-flagged clips
    python scratch/audit_transcripts.py --control 1000

    # audit only the 10 known suspects (by basename substring)
    python scratch/audit_transcripts.py --only Asakura_English__321605 ...

    # full flagged subset, resume if interrupted
    python scratch/audit_transcripts.py --control 2000

    # everything (hours)
    python scratch/audit_transcripts.py --scope all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2_wer import edit_counts, normalize_text, speaker_bucket  # noqa: E402

FLAG_ELLIPSIS = re.compile(r"(\.\.\.|…)\s*$")
FLAG_DASH = re.compile(r"[—–-]\s*$")

# Short non-lexical words whose exact spelling is not meaningful for ASR
# agreement ("aw" vs "oh", "hmm" vs "um", "ha haha" vs "ha ha ha").
_INTERJECTIONS = {
    "ah", "ahem", "aw", "awa", "aye", "eh", "ha", "haha", "heh", "hey",
    "hm", "hmm", "hmmm", "ho", "hoh", "huh", "mm", "mhm", "oh", "ooh", "oof",
    "ow", "oww", "owww", "phew", "psh", "shh", "uh", "um", "whoa", "woah",
    "wow", "yeah", "yep", "yes", "yup", "nah", "nope",
}


def _is_interjection(words: List[str]) -> bool:
    if not words or len(words) > 3:
        return False
    return all(re.sub(r"(.)\1{1,}", r"\1", w) in _INTERJECTIONS for w in words)


def char_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(
        None, a.replace(" ", ""), b.replace(" ", "")).ratio()


def word_lcs_ratio(ref: List[str], hyp: List[str]) -> float:
    """Longest common *word* subsequence / len(ref) (containment of the ref)."""
    if not ref:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for r in ref:
        cur = [0] * (len(hyp) + 1)
        for j, h in enumerate(hyp, 1):
            cur[j] = prev[j - 1] + 1 if r == h else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(ref)


def strip_outer_parens(text: str) -> str:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def clean_ref(text: str) -> str:
    """Normalise the filelist side (stage directions removed, parens unwrapped)."""
    try:
        from features import strip_stage_directions
        text = strip_stage_directions(text)
    except Exception:
        pass
    return normalize_text(strip_outer_parens(text))


def flags_of(text: str) -> List[str]:
    out = []
    t = text.strip()
    if FLAG_ELLIPSIS.search(t):
        out.append("ellipsis")
    if FLAG_DASH.search(t):
        out.append("dash")
    if t.startswith("("):
        out.append("open_paren")
    return out


def read_rows(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "longform" in line:
                    continue
                parts = line.rsplit("|", 2)
                if len(parts) != 3:
                    continue
                try:
                    sid = int(parts[1])
                except ValueError:
                    continue
                rows.append({"path": parts[0].replace("\\", "/"),
                             "sid": sid, "text": parts[2].strip(),
                             "basename": os.path.basename(
                                 parts[0].replace("\\", "/"))})
    return rows


def classify(ref: str, hyp: str, match_thr: float = 0.25,
             mismatch_thr: float = 0.5) -> Tuple[str, dict]:
    rw = ref.split()
    hw = hyp.split()
    stats = {"n_ref": len(rw), "n_hyp": len(hw), "wer": float("nan"),
             "sub": 0, "ins": 0, "del": 0, "prefix": False, "suffix": False,
             "char_sim": 0.0}
    if not rw:
        return "empty_reference", stats
    if not hw:
        return "asr_empty", stats
    s, i, d = edit_counts(rw, hw)
    wer = (s + i + d) / len(rw)
    cs = char_similarity(ref, hyp)
    lcs = word_lcs_ratio(rw, hw)
    stats.update(wer=wer, sub=s, ins=i, char_sim=cs, lcs=lcs,
                 prefix=hw[:len(rw)] == rw,
                 suffix=hw[-len(rw):] == rw)
    stats["del"] = d

    if stats["prefix"] and len(hw) > len(rw):
        return "prefix_truncated", stats
    if stats["suffix"] and len(hw) > len(rw):
        return "suffix_truncated", stats
    if wer <= match_thr:
        return "match", stats

    # Short lines: word WER is meaningless ("aw" vs "oh", "alright" vs
    # "all right", "hmm" vs "um").  Keep them unless the ASR is clearly a
    # longer, different utterance (a truncated transcript such as
    # "Lady Furina..." -> the full line).
    if len(rw) <= 3:
        if len(hw) <= len(rw) + 2:
            if _is_interjection(rw) and _is_interjection(hw):
                return "match_interjection", stats
            if cs >= 0.45 or _is_interjection(rw):
                return "match_short", stats
            return "review_short", stats
        if cs >= 0.55:
            return "match_short", stats
        # clearly longer content: a truncated/partial transcript
        return ("suffix_truncated" if stats["suffix"] else "prefix_truncated",
                stats)

    # Longer lines: a much longer ASR that still contains most of the ref words
    # is the truncated transcript (even with a name/number substitution, e.g.
    # "the deployment of gardes" -> "... deployment of guards at all times").
    # Only a low-coverage, high-WER pair is a confident different transcript.
    hyp_longer = len(hw) > len(rw) + max(3, int(0.5 * len(rw)))
    if len(hw) < 0.6 * len(rw):
        return "asr_short", stats
    if hyp_longer and lcs >= 0.5:
        return "prefix_truncated", stats
    if wer >= mismatch_thr and cs < 0.5 and lcs < 0.5:
        return "mismatch", stats
    if cs >= 0.6 or lcs >= 0.6:
        return "match_paraphrase", stats
    return "review", stats


def audio_16k(path: str) -> Optional[np.ndarray]:
    import soundfile as sf
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        return None
    if sr != 16000:
        import torch
        import torchaudio
        t = torch.from_numpy(np.clip(wav, -1.0, 1.0))
        t = torchaudio.functional.resample(t, sr, 16000)
        wav = t.numpy().astype(np.float32)
    return wav


def load_cache(path: str) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not path or not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[rec["basename"]] = rec
    return cache


def append_cache(path: str, recs: List[dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--filelist", nargs="+",
                    default=["filelists_2/genshin_full_eng_text_linux.txt"])
    ap.add_argument("--scope", choices=("flagged", "all"), default="flagged")
    ap.add_argument("--control", type=int, default=1000,
                    help="random non-flagged clips to audit as a control")
    ap.add_argument("--include-short", action="store_true",
                    help="also flag lines with <=3 words (ASR is unreliable)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="basename substrings to audit (calibration)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of clips in this run (0 = all)")
    ap.add_argument("--cache",
                    default="data/genshin_plus_extras/transcript_audit.jsonl")
    ap.add_argument("--out-drops",
                    default="data/genshin_plus_extras/transcript_drops.txt")
    ap.add_argument("--out-references",
                    default="data/genshin_plus_extras/wer_references_clean.json")
    ap.add_argument("--out-summary",
                    default="data/genshin_plus_extras/transcript_audit_summary.json")
    ap.add_argument("--stage2-filelist",
                    default="data/genshin_plus_extras/train_stage2.txt",
                    help="used only for the clip-count buckets in the summary")
    ap.add_argument("--drop-classes",
                    default="mismatch,prefix_truncated,suffix_truncated,"
                            "empty_reference,asr_empty,audio_missing",
                    help="classes written to --out-drops (train cleaning)")
    ap.add_argument("--wer-drop-classes", default="asr_short",
                    help="val classes whose references are too uncertain to "
                         "score; prefix/mismatch get the ASR text as reference")
    ap.add_argument("--match-threshold", type=float, default=0.25)
    ap.add_argument("--mismatch-threshold", type=float, default=0.5)
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--language", default="English")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()
    if args.control < 0:
        raise SystemExit("--control must be >= 0")
    drop_classes = {c.strip() for c in args.drop_classes.split(",") if c.strip()}
    wer_drop_classes = {c.strip() for c in args.wer_drop_classes.split(",")
                        if c.strip()}

    rows = read_rows(args.filelist)
    print(f"{len(rows)} transcript lines from {len(args.filelist)} filelist(s)")
    for row in rows:
        row["flags"] = flags_of(row["text"])
        row["ref"] = clean_ref(row["text"])

    if args.only:
        needles = [n.lower() for n in args.only]
        todo = [r for r in rows
                if any(n in r["basename"].lower() for n in needles)]
    else:
        flagged = [r for r in rows if r["flags"]]
        if args.include_short:
            flagged = [r for r in rows if r["flags"] or len(r["ref"].split()) <= 3]
        if args.scope == "all":
            todo = list(rows)
        else:
            rng = random.Random(args.seed)
            unflagged = [r for r in rows if not r["flags"]]
            control = rng.sample(unflagged, min(args.control, len(unflagged)))
            todo = flagged + control
    # deterministic order, unique by basename (duplicates across filelists)
    seen = set()
    uniq: List[dict] = []
    for r in todo:
        if r["basename"] in seen:
            continue
        seen.add(r["basename"])
        uniq.append(r)
    todo = uniq
    if args.limit:
        todo = todo[: args.limit]

    cache = load_cache(args.cache)
    missing = [r for r in todo if r["basename"] not in cache]
    print(f"scope={args.scope}: {len(todo)} selected, {len(cache)} cached, "
          f"{len(missing)} to run")

    if missing:
        from stage2_wer import Asr
        device = args.device or ("cuda:0" if __import__("torch").cuda.is_available()
                                 else "cpu")
        print(f"loading ASR {args.asr_model} on {device}")
        asr = Asr(args.asr_model, device, args.language,
                  max_inference_batch_size=args.batch_size)
        t0 = time.time()
        done = 0
        for start in range(0, len(missing), args.batch_size):
            chunk = missing[start:start + args.batch_size]
            wavs, kept = [], []
            for row in chunk:
                w = audio_16k(row["path"])
                if w is None:
                    cache[row["basename"]] = {
                        **{k: row[k] for k in ("basename", "path", "sid",
                                               "text", "flags")},
                        "ref": row["ref"], "hyp": "", "class": "audio_missing",
                        "wer": None, "n_ref": len(row["ref"].split()),
                        "n_hyp": 0, "sub": 0, "ins": 0, "del": 0,
                        "prefix": False, "suffix": False,
                    }
                    continue
                wavs.append(w)
                kept.append(row)
            hyps = asr.transcribe(wavs) if wavs else []
            recs = []
            for row, hyp_raw in zip(kept, hyps):
                hyp = normalize_text(hyp_raw)
                cls, stats = classify(row["ref"], hyp,
                                      args.match_threshold,
                                      args.mismatch_threshold)
                rec = {**{k: row[k] for k in ("basename", "path", "sid",
                                              "text", "flags")},
                       "ref": row["ref"], "hyp": hyp, "hyp_raw": hyp_raw,
                       "class": cls, **stats}
                recs.append(rec)
                cache[rec["basename"]] = rec
            append_cache(args.cache, recs)
            done += len(recs)
            if args.log_every and (start // args.batch_size) % args.log_every == 0:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                print(f"  {done}/{len(missing)}  {rate:.1f} clips/s  "
                      f"eta {(len(missing) - done) / max(rate, 1e-6) / 60:.1f} min")

    # ---- outputs ---------------------------------------------------------
    selected = {r["basename"]: r for r in todo}
    recs = [cache[b] for b in selected if b in cache]
    # Re-derive the class from the cached ref/hyp every time, so threshold and
    # classifier changes apply without re-running ASR.  ``audio_missing`` and
    # the empty cases have no hypothesis and keep their class.
    for r in recs:
        if r.get("class") in ("audio_missing",) or "hyp" not in r:
            continue
        cls, stats = classify(r["ref"], r.get("hyp", ""),
                              args.match_threshold, args.mismatch_threshold)
        r["class"] = cls
        r.update(stats)
    classes = Counter(r["class"] for r in recs)
    by_flag = defaultdict(Counter)
    for r in recs:
        for f in (r.get("flags") or ["none"]):
            by_flag[f][r["class"]] += 1

    # stage-2 bucket summary
    counts = Counter()
    if args.stage2_filelist and os.path.exists(args.stage2_filelist):
        with open(args.stage2_filelist, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "longform" in line:
                    continue
                try:
                    counts[int(line.rsplit("|", 1)[1])] += 1
                except ValueError:
                    pass
    by_bucket = defaultdict(Counter)
    for r in recs:
        by_bucket[speaker_bucket(counts.get(r["sid"], 0))][r["class"]] += 1

    drops = sorted({r["basename"] for r in recs if r["class"] in drop_classes})
    os.makedirs(os.path.dirname(os.path.abspath(args.out_drops)) or ".",
                exist_ok=True)
    with open(args.out_drops, "w", encoding="utf-8") as f:
        for b in drops:
            f.write(b + "\n")
    print(f"wrote {len(drops)} drop basename(s) -> {args.out_drops}")

    # WER reference overrides: a truncated/mismatched line uses the ASR text
    # (the spoken content); an uncertain short ASR is dropped from WER.
    refs = {}
    for r in recs:
        cls = r["class"]
        if cls in wer_drop_classes or cls in ("audio_missing", "asr_empty",
                                              "empty_reference"):
            refs[r["basename"]] = {"drop": True, "class": cls,
                                   "source": "audit"}
        elif cls in ("prefix_truncated", "suffix_truncated", "mismatch"):
            if r.get("hyp"):
                refs[r["basename"]] = {"drop": False, "class": cls,
                                       "text": r["hyp"], "source": "asr"}
            else:
                refs[r["basename"]] = {"drop": True, "class": cls,
                                       "source": "audit"}
        else:
            refs[r["basename"]] = {"drop": False, "class": cls,
                                   "text": r["text"], "source": "filelist"}
    with open(args.out_references, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=0)
    print(f"wrote {len(refs)} WER reference override(s) -> "
          f"{args.out_references}")

    summary = {
        "filelists": args.filelist,
        "scope": args.scope,
        "control": args.control,
        "include_short": args.include_short,
        "n_selected": len(todo),
        "n_audited": len(recs),
        "classes": dict(classes),
        "by_flag": {k: dict(v) for k, v in by_flag.items()},
        "by_stage2_bucket": {k: dict(v) for k, v in by_bucket.items()},
        "dropped": len(drops),
        "drop_classes": sorted(drop_classes),
        "wer_drop_classes": sorted(wer_drop_classes),
        "thresholds": {"match": args.match_threshold,
                       "mismatch": args.mismatch_threshold},
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ("n_selected", "n_audited", "classes", "dropped")},
                     indent=2))
    print(f"summary -> {args.out_summary}")


if __name__ == "__main__":
    main()
