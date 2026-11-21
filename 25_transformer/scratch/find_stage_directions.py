#!/usr/bin/env python3
"""Find (and optionally exclude) clips whose transcript has non-speech markup.

The Genshin transcripts write nonverbal sounds as stage directions:
``*sigh*``, ``*panting*``, ``*clears throat*``, ... (~1.2 % of clips).  The
current corpus was phonemized *before* `features.strip_stage_directions` existed,
so those spans became real words in ``phone_ids`` (``*sigh*`` -> ``SAY1``,
``*ahem*`` -> ``AH0 HHIY1 M``) and the model was trained to pronounce a
sigh/cough.  The text pipeline now strips them, but existing tensors keep the
old phonemes until a rebuild:

    # cheap: rebuild phone_ids from the transcripts (no audio/aligner)
    python 2_preprocess.py ... --only phones --phones_from transcript
    # ...but the .cond word fields must then be realigned:
    python hints/build.py ... --words

Until then, this script writes a blocklist for
``train.stage_direction_blocklist`` so those clips are dropped for the run.

Only **stage directions** are flagged.  ``*word*`` used for emphasis (all of
Expresso, plus ``*are*``/``*some*`` in Genshin) is kept: only the asterisks are
removed by `strip_stage_directions`, and the word is genuinely spoken.

Usage::

    python scratch/find_stage_directions.py \
        --filelist data/genshin_plus_extras/train_stage2.txt \
                   data/genshin_plus_extras/val_stage2.txt \
        --source-filelist filelists_2/genshin_full_eng_text.txt \
            filelists_2/mlp_linux.txt filelists_2/vctk_linux.txt \
            filelists_2/expresso_read_linux.txt \
        --out-blocklist data/genshin_plus_extras/stage_direction_clips.txt

Read-only apart from the optional outputs.  CPU only.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from features import (STAGE_DIRECTION_ANNOTATIONS,  # noqa: E402
                      strip_stage_directions)

_STAR_RE = re.compile(r"\*([^*\n]{1,60})\*")
_PAREN_RE = re.compile(r"[\(\[]([^\)\]\n]{1,60})[\)\]]")


def _key(text: str) -> str:
    return text.strip().lower().strip(" .,!?:;'\"")


def removed_annotations(text: str) -> List[str]:
    """The stage-direction spans in ``text`` (emphasis is not returned)."""
    out: List[str] = []
    for m in _STAR_RE.finditer(text or ""):
        key = _key(m.group(1))
        if key in STAGE_DIRECTION_ANNOTATIONS:
            out.append(key)
    for m in _PAREN_RE.finditer(text or ""):
        key = _key(m.group(1))
        if key in STAGE_DIRECTION_ANNOTATIONS:
            out.append(key)
    return out


def load_transcripts(paths: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3 or not parts[2].strip():
                    continue
                key = os.path.basename(parts[0].replace("\\", "/"))
                out.setdefault(key, parts[2])
    return out


def parse_filelist(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "longform" not in line:
                out.append(line)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Find clips with *sigh*-style non-speech annotations.")
    ap.add_argument("--filelist", nargs="+", required=True,
                    help="stage-2 code filelist(s)")
    ap.add_argument("--source-filelist", nargs="+", required=True,
                    help="path|sid|text filelists to map a clip to its text")
    ap.add_argument("--out-blocklist", default=None,
                    help="write flagged phone_ids paths here (for "
                         "train.stage_direction_blocklist)")
    ap.add_argument("--out-filelist", default=None,
                    help="write a filtered code filelist (single --filelist)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.out_filelist and len(args.filelist) > 1:
        ap.error("--out-filelist accepts exactly one --filelist")

    texts = load_transcripts(args.source_filelist)
    lines_by_file = {p: parse_filelist(p) for p in args.filelist}
    n_total = n_flagged = n_only_annotation = n_first = n_unmatched = 0
    annotations: collections.Counter = collections.Counter()
    flagged_paths: List[str] = []
    examples: List[dict] = []
    seen = 0
    for lines in lines_by_file.values():
        for line in lines:
            if args.limit and seen >= args.limit:
                break
            seen += 1
            parts = line.split("|")
            if len(parts) < 4:
                continue
            n_total += 1
            phone_ids_path = parts[0]
            key = os.path.basename(phone_ids_path)
            if key.endswith(".phone_ids"):
                key = key[: -len(".phone_ids")]
            text = texts.get(key)
            if text is None:
                n_unmatched += 1
                continue
            anns = removed_annotations(text)
            if not anns:
                continue
            n_flagged += 1
            flagged_paths.append(phone_ids_path)
            for a in anns:
                annotations[a] += 1
            # Everything gone?  The stage direction was the whole clip.
            if not strip_stage_directions(text).strip():
                n_only_annotation += 1
            if re.match(r"\s*[*\(\[]", text):
                n_first += 1
            if len(examples) < 15:
                examples.append({"clip": key, "annotations": anns,
                                 "text": text[:110]})
        if args.limit and seen >= args.limit:
            break

    stats = {
        "clips": n_total,
        "flagged": n_flagged,
        "flagged_fraction": (n_flagged / n_total) if n_total else 0.0,
        "annotation_only": n_only_annotation,
        "annotation_first": n_first,
        "unmatched": n_unmatched,
        "top_annotations": annotations.most_common(20),
    }
    if args.json:
        stats["examples"] = examples
        print(json.dumps(stats, indent=2))
    else:
        print(f"scanned {n_total} clip(s); {n_unmatched} had no source "
              f"transcript")
        print(f"stage-direction clips: {n_flagged} "
              f"({100 * stats['flagged_fraction']:.2f} %), "
              f"{n_first} with the annotation first, "
              f"{n_only_annotation} with no other speech")
        print("top annotations:", annotations.most_common(12))
        for e in examples[:8]:
            print(f"  {e['clip']}: {e['annotations']} | {e['text']}")

    if args.out_blocklist:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_blocklist)) or ".",
                    exist_ok=True)
        with open(args.out_blocklist, "w", encoding="utf-8") as f:
            for p in sorted(set(flagged_paths)):
                f.write(p + "\n")
        print(f"\nwrote {len(set(flagged_paths))} path(s) to "
              f"{args.out_blocklist}; set train.stage_direction_blocklist")
    if args.out_filelist:
        bad = set(flagged_paths)
        kept = 0
        out_path = args.out_filelist
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines_by_file[args.filelist[0]]:
                if line.split("|")[0] in bad:
                    continue
                f.write(line + "\n")
                kept += 1
        print(f"\nwrote {kept} kept line(s) to {out_path} "
              f"({len(bad)} dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
