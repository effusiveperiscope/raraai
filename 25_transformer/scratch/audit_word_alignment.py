#!/usr/bin/env python3
"""Audit per-word forced-alignment durations in the stage-2 ``.cond`` files.

A word's ``word_slot_f`` is start-to-start (word ``i`` ends where word ``i+1``
starts), so ``word_slot_f[i] / phones_in_word[i]`` is its average code frames
per phone.  On the Genshin corpus the median is ~3-4 frames/phone (25 Hz code
rate).  A word below ~1 frame/phone is 40 ms per phone, i.e. physically
impossible for speech: either the clip cut the word off or the aligner
collapsed it (``Paimon_English__163589`` gives "Traveler!" 4 frames).  Those
clips teach the model to swallow the word, which is the "skips the first word /
Traveler!" symptom.

**Pass ``--source-filelist`` to count real phones.**  Without it the denominator
is ``phone_word_idx``'s BPE-token count, which is *not* the phone count: the
g2pen tokenizer merges phones (``FAY1N`` = three phones) and attaches word
boundary markers, so the proxy flags ~3.7 % of clips where the true-phone rule
flags ~0.25 %.  The script prints which denominator was used.

This script reports the distribution, and can write either a **blocklist** of
``phone_ids`` paths (for ``train.alignment_blocklist``) or a filtered filelist.

Usage::

    # report only (real phones)
    python scratch/audit_word_alignment.py \
        --filelist data/genshin_plus_extras/train_stage2.txt \
        --source-filelist filelists_2/genshin_full_eng_text.txt

    # write a blocklist consumed by 2_training.py (train.alignment_blocklist)
    python scratch/audit_word_alignment.py \
        --filelist data/genshin_plus_extras/train_stage2.txt \
        --source-filelist filelists_2/genshin_full_eng_text.txt \
        --first-word-only --min-word-phones 3 \
        --blocklist data/genshin_plus_extras/degenerate_words.txt

    # ...or a filtered filelist
    python scratch/audit_word_alignment.py \
        --filelist data/genshin_plus_extras/train_stage2.txt \
        --source-filelist filelists_2/genshin_full_eng_text.txt \
        --out-filelist data/genshin_plus_extras/train_stage2_clean.txt

Read-only apart from the two optional outputs.  Only ``.cond`` files (and the
optional transcripts, on CPU) are loaded; no GPU.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset_codes import cond_path_for  # noqa: E402


def load_word_stats(path: str):
    """``(slot_f, phones_per_word)`` or ``None`` when the file cannot say."""
    try:
        cond = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if "phone_word_idx" not in cond or "word_slot_f" not in cond:
        return None
    pwi = cond["phone_word_idx"].long().reshape(-1).numpy()
    slot = cond["word_slot_f"].float().reshape(-1).numpy()
    starts = (cond["word_start_f"].long().reshape(-1).numpy()
              if "word_start_f" in cond else None)
    if pwi.size == 0 or slot.size == 0:
        return None
    n_word = int(max(int(pwi.max()) + 1, slot.size))
    counts = np.bincount(pwi.clip(min=0), minlength=n_word).astype(np.float64)
    if counts.size < slot.size:
        counts = np.pad(counts, (0, slot.size - counts.size))
    if starts is None or starts.size != counts.size:
        starts = None
    return starts, slot[: counts.size], counts[: slot.size]


def parse_filelist(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "longform" in line:
                continue
            out.append(line)
    return out


def load_transcripts(paths: List[str]) -> Dict[str, str]:
    """``basename -> transcript`` from ``path|sid|text`` source filelists."""
    out: Dict[str, str] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3:
                    continue
                key = os.path.basename(parts[0].replace("\\", "/"))
                if key and key not in out and parts[2].strip():
                    out[key] = parts[2]
    return out


#: ARPAbet phones in ``HorsePhonemizer``'s concatenated output (``TUW1`` ->
#: ``T``, ``UW1``; stress digit optional).
_PHONE_RE = re.compile(r"[A-Z]+[0-2]?")


def make_phone_counter():
    """Cached ``word -> phone count`` using the corpus's phonemizer."""
    from horsephonemizer import HorsePhonemizer

    hp = HorsePhonemizer()
    cache: Dict[str, int] = {}

    def count(word: str) -> int:
        key = word.lower()
        n = cache.get(key)
        if n is None:
            n = max(len(_PHONE_RE.findall(hp.phonemize(word))), 1)
            cache[key] = n
        return n

    return count


def transcript_words_strict(text: str):
    """``[(word, punct)]``; imported lazily so the no-transcript path is light."""
    from features import transcript_words
    return transcript_words(text)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report (and optionally filter) degenerate per-word "
                    "alignments in stage-2 .cond files.")
    ap.add_argument("--filelist", nargs="+", required=True,
                    help="stage-2 filelist(s), path|spec|pitch|sid per line")
    ap.add_argument("--min-frames-per-phone", type=float, default=1.0,
                    help="flag a word below this many code frames per phone "
                         "(default 1.0 = 40 ms/phone at 25 Hz)")
    ap.add_argument("--first-word-only", action="store_true",
                    help="only consider word 0 (the common failure at a cut)")
    ap.add_argument("--min-word-phones", type=int, default=1,
                    help="only consider words with at least this many phones "
                         "(real phones with --source-filelist, BPE tokens "
                         "otherwise).  3 filters out one-/two-phone function "
                         "words, whose short duration is not necessarily an "
                         "error")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N clips (0 = all)")
    ap.add_argument("--source-filelist", nargs="+", default=None,
                    help="path|sid|text filelists to map a clip to its "
                         "transcript and count real phones.  Without it the "
                         "denominator is BPE tokens (a noisy proxy: the "
                         "tokenizer merges phones and adds boundary markers).")
    ap.add_argument("--blocklist", default=None,
                    help="write the phone_ids paths of flagged clips here "
                         "(one per line; pass to train.alignment_blocklist)")
    ap.add_argument("--out-filelist", default=None,
                    help="write the flagged clips *removed* to a new filelist "
                         "(single --filelist only)")
    ap.add_argument("--json", action="store_true", help="machine-readable stats")
    args = ap.parse_args(argv)

    if args.out_filelist and len(args.filelist) > 1:
        ap.error("--out-filelist accepts exactly one --filelist")
    if args.min_frames_per_phone <= 0:
        ap.error("--min-frames-per-phone must be > 0")

    # Per-word aggregates: word index -> ratio list (bucketed by "first word"
    # vs the rest) so the report says *where* the problem is.
    ratios_first: List[float] = []
    ratios_rest: List[float] = []
    flagged_per_clip: List[int] = []
    flagged_first = 0
    n_clips = 0
    n_no_cond = 0
    n_true_phones = 0
    n_token_proxy = 0
    dup_start_words = 0
    flagged_dup = 0
    flagged_paths: List[str] = []
    examples: List[dict] = []
    all_lines: Dict[str, List[str]] = {p: parse_filelist(p) for p in args.filelist}
    texts = load_transcripts(args.source_filelist) if args.source_filelist else {}
    phone_count = make_phone_counter() if texts else None

    total = 0
    for lines in all_lines.values():
        total += len(lines)

    seen = 0
    for lines in all_lines.values():
        for line in lines:
            if args.limit and seen >= args.limit:
                break
            seen += 1
            parts = line.split("|")
            if len(parts) < 4:
                continue
            phone_ids_path = parts[0]
            cond = cond_path_for(phone_ids_path)
            if not os.path.exists(cond):
                n_no_cond += 1
                continue
            stats = load_word_stats(cond)
            if stats is None:
                n_no_cond += 1
                continue
            starts, slot, token_counts = stats
            # Prefer real phones from the clip's transcript.  ``phone_word_idx``
            # is a BPE-token index, and the tokenizer merges phones
            # (``FAY1N`` = F AY1 N) and attaches boundary markers, so the token
            # ratio is a noisy proxy, not frames/phone.
            counts = token_counts
            denom = "token"
            clip_words = None
            if texts:
                key = os.path.basename(phone_ids_path)
                if key.endswith(".phone_ids"):
                    key = key[: -len(".phone_ids")]
                text = texts.get(key)
                if text:
                    tw = transcript_words_strict(text)
                    if len(tw) == len(starts):
                        counts = np.array([phone_count(w) for w, _ in tw],
                                          dtype=np.float64)
                        denom = "phone"
                        clip_words = tw
            if texts:
                if denom == "phone":
                    n_true_phones += 1
                else:
                    n_token_proxy += 1
            valid = counts > 0
            if not valid.any():
                continue
            ratio = slot / np.maximum(counts, 1.0)
            n_clips += 1
            if starts is not None and starts.size > 1:
                dup = np.zeros(starts.size, dtype=bool)
                dup[:-1] = starts[1:] == starts[:-1]
                dup_start_words += int(dup.sum())
            else:
                dup = np.zeros(ratio.size, dtype=bool)
            first = float(ratio[0]) if valid[0] else float("nan")
            if np.isfinite(first):
                ratios_first.append(first)
            rest = ratio[valid]
            if rest.size > 1:
                ratios_rest.extend(rest[1:].tolist())
            bad = np.zeros(ratio.size, dtype=bool)
            bad[valid] = ratio[valid] < args.min_frames_per_phone
            bad &= counts >= max(1, int(args.min_word_phones))
            if args.first_word_only:
                bad[1:] = False
            n_bad = int(bad.sum())
            if n_bad:
                flagged_paths.append(phone_ids_path)
                flagged_per_clip.append(n_bad)
                if bad[0]:
                    flagged_first += 1
                flagged_dup += int(np.sum(bad & dup))
                if len(examples) < 20:
                    idxs = np.flatnonzero(bad)[:6]
                    examples.append({
                        "clip": os.path.basename(phone_ids_path),
                        "bad_words": [int(i) for i in idxs],
                        "word": ([clip_words[i][0] for i in idxs]
                                 if clip_words is not None else None),
                        "ratio": [round(float(ratio[i]), 2) for i in idxs],
                        "frames": [int(slot[i]) for i in idxs],
                        "phones": [int(counts[i]) for i in idxs],
                        "dup_start": [bool(dup[i]) for i in idxs],
                    })
        if args.limit and seen >= args.limit:
            break

    def q(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")

    flagged_n = len(flagged_paths)
    stats = {
        "clips_scanned": n_clips,
        "clips_without_usable_cond": n_no_cond,
        "clips_with_real_phones": n_true_phones,
        "clips_with_token_proxy": n_token_proxy,
        "denominator": "phone" if not texts or n_true_phones >= n_token_proxy
                       else "token",
        "min_frames_per_phone": args.min_frames_per_phone,
        "min_word_phones": int(args.min_word_phones),
        "first_word_only": bool(args.first_word_only),
        "duplicate_start_words": dup_start_words,
        "flagged_duplicate_start_words": flagged_dup,
        "first_word_ratio": {
            "median": q(ratios_first, 50), "p5": q(ratios_first, 5),
            "p1": q(ratios_first, 1), "min": min(ratios_first) if ratios_first else None,
        },
        "other_words_ratio": {
            "median": q(ratios_rest, 50), "p5": q(ratios_rest, 5),
            "p1": q(ratios_rest, 1),
        },
        "flagged_clips": flagged_n,
        "flagged_fraction": (flagged_n / n_clips) if n_clips else 0.0,
        "flagged_first_word": flagged_first,
        "flagged_words_total": int(sum(flagged_per_clip)),
    }

    if args.json:
        import json
        stats["examples"] = examples
        print(json.dumps(stats, indent=2))
    else:
        print(f"scanned {n_clips} clip(s) with usable word timings"
              f" ({n_no_cond} without a usable .cond)")
        if texts:
            print(f"denominator: real phones for {n_true_phones} clip(s), "
                  f"BPE-token proxy for {n_token_proxy} (transcript missing or "
                  f"word count mismatch)")
        else:
            print("denominator: BPE tokens (no --source-filelist); this is a "
                  "noisy proxy -- pass transcripts for real phones")
        print()
        unit = "phone" if (texts and n_true_phones > n_token_proxy) else "token"
        if ratios_first:
            print(f"frames/{unit}, FIRST word: "
                  f"median {q(ratios_first, 50):.2f}  "
                  f"p5 {q(ratios_first, 5):.2f}  "
                  f"min {min(ratios_first):.2f}")
        if ratios_rest:
            print(f"frames/{unit}, other words: "
                  f"median {q(ratios_rest, 50):.2f}  "
                  f"p5 {q(ratios_rest, 5):.2f}")
        print(f"\nbelow {args.min_frames_per_phone:g} frames/{unit} with "
              f">= {args.min_word_phones} {unit}(s): "
              f"{flagged_n} clip(s) ({100 * stats['flagged_fraction']:.1f} %), "
              f"{stats['flagged_words_total']} word(s); "
              f"{flagged_first} have the FIRST word flagged")
        print(f"duplicate starts (two adjacent words, same start frame): "
              f"{dup_start_words} word(s), of which {flagged_dup} are flagged")
        print("-> those clips teach the model to say a word in ~0.04-0.15 s")
        if dup_start_words and not args.first_word_only:
            print("   (duplicate starts are an aligner artefact, not necessarily "
                  "a missing word; --first-word-only targets the audible case)")
        if examples:
            print("\nexamples:")
            for e in examples[:10]:
                print(f"  {e['clip']}: words {e['bad_words']} {e['word']} "
                      f"({e['frames']} frames / {e['phones']} phones = "
                      f"{e['ratio']} f/ph, dup_start={e['dup_start']})")

    if args.blocklist:
        os.makedirs(os.path.dirname(os.path.abspath(args.blocklist)) or ".",
                    exist_ok=True)
        with open(args.blocklist, "w", encoding="utf-8") as f:
            for p in sorted(set(flagged_paths)):
                f.write(p + "\n")
        print(f"\nwrote {len(set(flagged_paths))} path(s) to {args.blocklist}; "
              f"set train.alignment_blocklist to it")

    if args.out_filelist:
        bad = set(flagged_paths)
        kept = 0
        os.makedirs(os.path.dirname(os.path.abspath(args.out_filelist)) or ".",
                    exist_ok=True)
        with open(args.out_filelist, "w", encoding="utf-8") as f:
            for line in all_lines[args.filelist[0]]:
                parts = line.split("|")
                if parts and parts[0] in bad:
                    continue
                f.write(line + "\n")
                kept += 1
        print(f"\nwrote {kept} kept line(s) to {args.out_filelist} "
              f"({len(bad)} dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
