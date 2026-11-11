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
