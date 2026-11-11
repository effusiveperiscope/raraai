"""Level (bucket) ids for the continuous conditioning hints.

One implementation shared by everything that touches a level id:

* ``hints/build.py`` fits and writes the edges (``hints.json``),
* ``dataset_codes.py`` buckets each clip's continuous value at load time,
* ``FeatureExtractor.extract_hints`` / an inference UI does the same from a
  live value.

Continuity is the point: ``.cond`` stores a *float*, so changing how many levels
are used (or where the cuts sit) is a config edit, never a re-run of the corpus.
The edges are equal-frequency quantiles over the corpus because the DSP
intensity regression is compressed (measured: ``dsp = 2.07 + 0.512 * label`` on
the upstream val slice), so fixed-width bins would leave most of the table
unused.

Level ids::

    0        hint absent (not provided, or dropped by hint dropout)
    1 .. N   quantile bin, ascending in the hint value

This module only needs numpy, so training code can import it without pulling in
scikit-learn / parselmouth.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np


def bucket_levels(values, edges: Optional[Sequence[float]]):
    """Map continuous hint value(s) to level ids ``0..N``.

    Args:
        values: scalar or array-like of floats; non-finite entries become 0.
        edges: the ``N - 1`` interior quantiles from ``hints.json``, ascending.

    Returns:
        An int (scalar input) or ``int64`` array (array input) in ``[0, N]``,
        where 1..N are the bins and 0 means "absent".

    Raises:
        ValueError: ``edges`` is missing, or not ascending.
    """
    if edges is None or len(edges) == 0:
        raise ValueError(
            "bucket_levels needs the fitted edges from hints.json "
            "(run hints/build.py without --no-edges)")
    e = np.asarray(list(edges), dtype=np.float64)
    if e.ndim != 1 or np.any(np.diff(e) < 0):
        raise ValueError(f"edges must be a 1-D ascending sequence, got {e.tolist()}")
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    # Replace non-finite entries before the search (avoids NaN comparisons) and
    # zero their ids afterwards: id 0 is the trained "not provided" row.
    ids = np.searchsorted(e, np.where(finite, v, e[0]), side="right") + 1
    ids = np.where(finite, ids, 0).astype(np.int64)
    return int(ids) if np.ndim(values) == 0 else ids


def load_levels(path: str) -> dict:
    """Read ``hints.json``'s ``levels`` block -> ``{name: {n_levels, edges}}``."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f) or {}
    out = {}
    for name, spec in (doc.get("levels") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        edges = spec.get("edges")
        out[name] = {
            "n_levels": int(spec.get("n_levels")
                            or ((len(edges) + 1) if edges else 0)),
            "edges": list(edges) if edges else None,
        }
    return out


def load_level_edges(path: str, name: str):
    """``(edges, n_levels)`` for one hint. Raises if it was never fitted."""
    levels = load_levels(path)
    if name not in levels or not levels[name]["edges"]:
        raise ValueError(
            f"{path} has no fitted edges for hint {name!r} "
            f"(found: {sorted(levels)}). Re-run hints/build.py over the whole "
            f"corpus without --no-edges.")
    return levels[name]["edges"], levels[name]["n_levels"]


#: Default cuts for the **word pitch** hint, in semitones relative to the
#: utterance's voiced median.  Two semitones per bin, symmetric, saturating --
#: 7 voiced bins (``< -6``, ``-6..-4``, ... , ``>= 6``).
#:
#: Fixed rather than corpus quantiles on purpose: the value is *relative to the
#: utterance*, and an inference UI has to be able to say "raise this word a
#: major third" and mean it.  Quantile edges would renumber the bins per corpus
#: and make a stored level meaningless outside it.  (Intensity and register are
#: absolute scales, so those stay equal-frequency from ``hints.json``.)
DEFAULT_PITCH_EDGES = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)

#: Number of **word slot** buckets: log2 buckets of the word's duration in code
#: frames (1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128+).  Also fixed: it is a
#: duration, and "8 frames" has to mean 8 frames.
DEFAULT_SLOT_BUCKETS = 8


def word_pitch_levels(pitch_st, voiced, edges=None):
    """Word pitch levels: ``0`` absent, ``1..N`` voiced bins, ``N+1`` unvoiced.

    ``pitch_st`` is the word's voiced-mean f0 in semitones relative to the
    utterance (``FeatureExtractor.hint_words`` -> ``pitch_word_st``, NaN where
    the word has no voiced frames) and ``voiced`` its ``pitch_word_voiced``
    flag.  The explicitly-unvoiced class is the last row of the table rather
    than ``0`` so that "not provided" and "this word is unvoiced" stay
    distinguishable -- the whole point of the dropout design.
    """
    edges = list(DEFAULT_PITCH_EDGES if edges is None else edges)
    lv = np.asarray(bucket_levels(pitch_st, edges), dtype=np.int64)
    v = np.asarray(voiced, dtype=bool)
    lv = np.where(v, lv, len(edges) + 1)
    return int(lv) if np.ndim(voiced) == 0 else lv


def slot_bucket(slots, n_buckets: int = DEFAULT_SLOT_BUCKETS):
    """Log2 duration buckets for word slots (code frames); ``0`` = absent.

    ``slot = 1`` -> 1, ``2..3`` -> 2, ``4..7`` -> 3, ... , ``>= 2**(n-1)`` -> n.
    Values below 1 or non-finite map to 0 (absent).
    """
    s = np.asarray(slots, dtype=np.float64)
    finite = np.isfinite(s) & (s >= 1.0)
    b = np.floor(np.log2(np.where(finite, s, 1.0))).astype(np.int64) + 1
    b = np.clip(b, 1, int(n_buckets))
    b = np.where(finite, b, 0)
    return int(b) if np.ndim(slots) == 0 else b
