#!/usr/bin/env python3
"""Audit which corpus tokens fall through to g2p_en's neural fallback.

``g2p_en`` answers OOV words with a character-level seq2seq model
(``G2p.predict``).  Its output is phonotactically plausible, so a bad
pronunciation does not look bad -- which is how hyphenated compounds shipped as
garbage through two corpus builds (``well-known`` -> ``WEH1L K AO1 N ER0``,
``self-esteem`` -> ``S EH1L F S IY1 T S IY0``; see
``horsephonemizer._compound_pronunciation``).

This script replicates the phonemizer's lookup order *without* running the
fallback, so it lists exactly the tokens that hit it -- the pin list for
``new_horsewords.clean`` and a check that a fix actually took.

    python scratch/audit_oov.py                       # every filelists_2/*.txt
    python scratch/audit_oov.py --filelists filelists_2/genshin_full_eng_text_linux.txt
    python scratch/audit_oov.py --top 60 --examples 2

Reads raw-audio filelists (``path|sid|transcript``); stage-2 filelists carry no
transcript, so pass ``--source-filelist`` style inputs.  No GPU, no aligner.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from horsephonemizer import (  # noqa: E402
    _COMPOUND_SEP, _KEEP_PUNCT, _split_punctuation, g2p)

#: Mirrors g2p_en's sanitisation: everything except letters and ``'`` is
#: dropped before the cmudict lookup.
_WORD_CHARS = re.compile(r"[^a-z']")


def _cmu_key(token: str) -> str:
    from horsephonemizer import _APOSTROPHES

    token = str(token).translate(_APOSTROPHES)
    _prefix, word, _suffix = _split_punctuation(token.strip())
    return _WORD_CHARS.sub("", word.lower())


def _oov(hp, token: str, tag):
    """``[token]`` if ``token`` would reach the neural fallback, else ``[]``."""
    if not token or not any(ch.isalpha() for ch in token):
        return []
    if hp._dictionary_pronunciation(token, tag) is not None:
        return []
    # A compound is phonemized part by part (horsephonemizer), so audit the
    # parts -- itself a regression check that the split is still in place.
    _prefix, word, _suffix = _split_punctuation(token.strip())
    parts = _COMPOUND_SEP.split(word)
    if len(parts) >= 2:
        out = []
        for part in parts:
            if part:
                out.extend(_oov(hp, part, tag))
        return out
    if tag is not None and hp._homograph_pronunciation(token, tag) is not None:
        return []
    key = _cmu_key(token)
    if not key:
        return []
    if key in g2p.cmu:
        return []
    return [key]


def audit(hp, transcripts, limit=0):
    """``(counts, examples, n_tokens, n_clips, n_clips_with_oov)``."""
    counts = collections.Counter()
    examples = collections.defaultdict(list)
    n_tokens = n_clips = n_oov_clips = 0
    for text in transcripts:
        n_clips += 1
        words = str(text).split()
        if not words:
            continue
        tags = hp.tag_words(words)
        hit = []
        for word, tag in zip(words, tags):
            n_tokens += 1
            hit.extend(_oov(hp, word, tag))
        if hit:
            n_oov_clips += 1
            for tok in hit:
                counts[tok] += 1
                if len(examples[tok]) < 3:
                    examples[tok].append(" ".join(words)[:120])
        if limit and n_clips >= limit:
            break
    return counts, examples, n_tokens, n_clips, n_oov_clips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelists", nargs="*", default=None,
                    help="raw-audio filelists (path|sid|transcript); default: "
                         "filelists_2/*_linux.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N clips per filelist (0 = all)")
    ap.add_argument("--top", type=int, default=40,
                    help="tokens to list per filelist (default 40)")
    ap.add_argument("--examples", type=int, default=1,
                    help="example sentences per token (default 1)")
    args = ap.parse_args(argv)

    from horsephonemizer import HorsePhonemizer

    hp = HorsePhonemizer()
    paths = args.filelists or sorted(glob.glob("filelists_2/*_linux.txt"))
    if not paths:
        raise SystemExit("no filelists found; pass --filelists")

    total_oov = total_tok = 0
    worst = collections.Counter()
    for path in paths:
        texts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3:
                    continue
                texts.append(parts[-1])
        counts, examples, n_tok, n_clips, n_bad = audit(hp, texts,
                                                        args.limit)
        total_oov += sum(counts.values())
        total_tok += n_tok
        worst.update(counts)
        print(f"\n=== {os.path.basename(path)}")
        print(f"    {n_clips} clips, {n_tok} tokens, "
              f"{sum(counts.values())} OOV ({100 * sum(counts.values()) / max(n_tok, 1):.2f}%), "
              f"{n_bad} clips with >=1 OOV "
              f"({100 * n_bad / max(n_clips, 1):.1f}%)")
        for token, count in counts.most_common(args.top):
            ex = ""
            if args.examples and examples.get(token):
                ex = f"   e.g. {examples[token][0]}"
            print(f"    {count:6d}  {token}{ex}")

    print(f"\nTOTAL: {total_oov} OOV token occurrence(s) over {total_tok} "
          f"tokens ({100 * total_oov / max(total_tok, 1):.2f}%)")
    if worst:
        print("top across all filelists: "
              + ", ".join(f"{t}x{c}" for t, c in worst.most_common(20)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
