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
    """Read ``hints.json``'s ``levels`` block.

    Returns ``{name: {n_levels, edges, speaker_means, unit}}``.  The last two
    are only meaningful for ``register`` and are ``None``/``""`` when absent,
    so a file written before the relative-register change still loads.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f) or {}
    out = {}
    for name, spec in (doc.get("levels") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        edges = spec.get("edges")
        means = spec.get("speaker_means")
        out[name] = {
            "n_levels": int(spec.get("n_levels")
                            or ((len(edges) + 1) if edges else 0)),
            "edges": list(edges) if edges else None,
            # ``{str(sid): semitones}`` -- present only for a *relative*
            # register corpus (see ``register_rel_st`` below).
            "speaker_means": ({str(k): float(v) for k, v in means.items()}
                              if isinstance(means, dict) and means else None),
            "unit": str(spec.get("unit") or ""),
        }
    return out


def register_rel_st(value, sid, speaker_means):
    """Per-utterance register deviation from the speaker's frozen mean.

    ``hints.json`` -> ``levels.register.speaker_means`` holds one semitone
    offset per speaker id, fitted over the *training* split by
    ``hints/build.py --relative-register``.  Subtracting it makes the register
    channel orthogonal to ``sid`` by construction: the speaker's absolute
    level is already carried by ``spk_emb``, and what is left is the part a
    single per-speaker vector cannot express (this utterance is higher or
    lower than this speaker normally speaks).

    An unseen ``sid`` (or an old, absolute corpus) maps to an offset of 0.0,
    i.e. "typical for this speaker" -- the same meaning as level 0 elsewhere.
    """
    if speaker_means is None:
        return float(value)
    offset = speaker_means.get(str(int(sid)))
    if offset is None:
        offset = speaker_means.get(int(sid), 0.0)
    return float(value) - float(offset)


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
#: **8** voiced bins (``< -6``, ``-6..-4``, ... , ``>= 6``; ``N = len(edges)+1``),
#: then the unvoiced class at ``N + 1`` and "absent" at 0.

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
    flag.  ``N`` is the number of voiced bins, i.e. ``len(edges) + 1`` (the
    outermost bins saturate), so the unvoiced class is ``len(edges) + 2``.

    The explicitly-unvoiced class is the last row of the table rather than
    ``0`` so that "not provided" and "this word is unvoiced" stay
    distinguishable -- the whole point of the dropout design.

    Note: this id must be *above* every voiced bin.  It used to be
    ``len(edges) + 1``, which collides with the top voiced bin (``>= +6 st``),
    so unvoiced words were indistinguishable from the highest pitch bin and the
    table's real unvoiced row (``n_pitch_levels + 1``) never received a
    gradient (measured on transformer_v0.9: row 9 exactly 0.0).
    """
    edges = list(DEFAULT_PITCH_EDGES if edges is None else edges)
    lv = np.asarray(bucket_levels(pitch_st, edges), dtype=np.int64)
    v = np.asarray(voiced, dtype=bool)
    lv = np.where(v, lv, len(edges) + 2)
    return int(lv) if np.ndim(voiced) == 0 else lv



#: Number of **pitch anchors** stored per word (Hints V2).  A word is reduced to
#: at most this many ``(position, value)`` points; see ``detect_pitch_anchors``.
#: The value is repeated over the word's phones, so it is the *word*'s contour
#: that is stored, not a per-phone sample.
DEFAULT_PITCH_ANCHORS = 3

#: Deviation (semitones) below which a span counts as straight.  A word whose
#: whole contour stays inside this band becomes **one** anchor (the "flat word"
#: case); otherwise the greedy split runs until it has ``max_anchors`` points.
PITCH_ANCHOR_TOLERANCE_ST = 0.2

#: Median-filter width, in f0 frames (10 ms each).  **Spike removal only**:
#: a 3-point median deletes a one-frame octave error (12 st) without rounding a
#: real 2-3 frame peak, which a wider window would shrink (and, on a ramp, bias
#: the endpoints inwards -- both measured while building this).
PITCH_ANCHOR_SMOOTH_FRAMES = 3

#: Octave-error guard: a frame further than this from its 5-frame local median is
#: treated as a tracker octave jump and replaced by that median.  A 1-frame spike
#: is already gone after the median filter, but a 2-frame jump survives it and
#: becomes a real anchor (measured: a `-11.4 st` anchor in a smoke clip).  Real
#: f0 movements within one word are far below 6 st per 50 ms, so this is targeted.
PITCH_ANCHOR_OCTAVE_ST = 6.0


def _masked_median(y: np.ndarray, k: int) -> np.ndarray:
    """Median-filter ``y`` ignoring NaNs.  ``NaN`` where a window is all-NaN.

    A plain filter would either propagate NaN or treat unvoiced frames as 0 Hz,
    both of which invent an extremum at every voice boundary.

    The first/last ``k // 2`` frames are left unsmoothed: a truncated window
    biases the median inwards (for a ramp the edge value becomes its neighbour),
    which invents a spurious anchor at each end.
    """
    n = y.shape[0]
    half = max(int(k) // 2, 1)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo, hi = i - half, i + half + 1
        if lo < 0 or hi > n:
            out[i] = y[i]
            continue
        w = y[lo:hi]
        w = w[~np.isnan(w)]
        if w.size:
            out[i] = np.median(w)
    return out


def _guard_octaves(y: np.ndarray, threshold: float) -> np.ndarray:
    """Replace frames whose 5-frame local median is > ``threshold`` away.

    Targets the survivable octave error: the 3-point median removes a one-frame
    jump, but a two-frame one is still an anchor and would tell the model to drop
    a word an octave.  A genuine peak is kept because its *local median* is close
    to it when it lasts more than a couple of frames.
    """
    n = y.shape[0]
    if n < 3:
        return y
    out = y.copy()
    for i in range(n):
        if not np.isfinite(y[i]):
            continue
        lo, hi = max(0, i - 2), min(n, i + 3)
        w = y[lo:hi]
        w = w[np.isfinite(w)]
        if w.size < 3:
            continue
        med = float(np.median(w))
        if abs(float(y[i]) - med) > float(threshold):
            out[i] = med
    return out


def _optimal_knots(pos: np.ndarray, val: np.ndarray, max_anchors: int):
    """Exact best piecewise-linear fit with at most ``max_anchors`` knots.

    The endpoints are always knots (a word's contour has to say where it starts
    and ends); the interior knots are chosen by dynamic programming to minimise
    the squared error of the line segments between them.  This is the same
    objective the quality gate measures, so the detector cannot lose to a better
    placement: measured on 1 068 Genshin words, a greedy splitter lands at
    0.90 st and this at **0.81 st** (3 knots, code-frame contours).

    Costs are O(1) per candidate segment via six prefix sums, so the DP is
    ``O(K*n^2)``; ``n`` is a word's code-frame length (median 6, p95 28).

    Returns:
        Sorted indices into ``pos``/``val`` (at least one).
    """
    n = int(pos.shape[0])
    if n <= 1:
        return [0]
    k_max = min(int(max_anchors), n) - 1          # number of segments
    if k_max <= 0:
        return [0]
    if k_max == 1:
        return [0, n - 1]

    x = pos.tolist()
    y = val.tolist()
    p1 = [0.0] * (n + 1)
    px = [0.0] * (n + 1)
    py = [0.0] * (n + 1)
    pxx = [0.0] * (n + 1)
    pxy = [0.0] * (n + 1)
    pyy = [0.0] * (n + 1)
    for t in range(n):
        xt, yt = x[t], y[t]
        p1[t + 1] = p1[t] + 1.0
        px[t + 1] = px[t] + xt
        py[t + 1] = py[t] + yt
        pxx[t + 1] = pxx[t] + xt * xt
        pxy[t + 1] = pxy[t] + xt * yt
        pyy[t + 1] = pyy[t] + yt * yt

    def cost(i: int, j: int) -> float:
        """SSE of the line through knots ``i``/``j`` on the points between."""
        if j <= i + 1:
            return 0.0
        lo, hi = i + 1, j
        m = p1[hi] - p1[lo]
        if m <= 0.0:
            return 0.0
        sx = px[hi] - px[lo]
        sy = py[hi] - py[lo]
        sxx = pxx[hi] - pxx[lo]
        sxy = pxy[hi] - pxy[lo]
        syy = pyy[hi] - pyy[lo]
        dx = x[j] - x[i]
        slope = (y[j] - y[i]) / dx if dx > 0.0 else 0.0
        icpt = y[i] - slope * x[i]
        sse = (syy - 2.0 * icpt * sy - 2.0 * slope * sxy
               + icpt * icpt * m + 2.0 * icpt * slope * sx
               + slope * slope * sxx)
        return sse if sse > 0.0 else 0.0

    inf = float("inf")
    dp = [[inf] * n for _ in range(k_max + 1)]
    parent = [[-1] * n for _ in range(k_max + 1)]
    dp[0][0] = 0.0
    for k in range(1, k_max + 1):
        for j in range(1, n):
            best, best_i = inf, -1
            for i in range(0, j):
                prev = dp[k - 1][i]
                if prev == inf:
                    continue
                v = prev + cost(i, j)
                if v < best:
                    best, best_i = v, i
            dp[k][j] = best
            parent[k][j] = best_i
    if dp[k_max][n - 1] == inf:                   # unreachable; keep endpoints
        return [0, n - 1]
    knots = []
    j = n - 1
    for k in range(k_max, 0, -1):
        knots.append(j)
        j = parent[k][j]
    knots.append(0)
    knots.reverse()
    return knots


def _fit_sse(pos: np.ndarray, val: np.ndarray, knots) -> float:
    """Squared error of the piecewise-linear fit through ``knots``."""
    k = list(knots)
    if not k:
        return float("inf")
    if len(k) == 1:
        return float(np.sum((val - val[k[0]]) ** 2))
    rec = np.interp(pos, pos[k], val[k])
    return float(np.sum((val - rec) ** 2))


def _prune_knots(pos: np.ndarray, val: np.ndarray, knots, tolerance: float):
    """Drop interior knots the fit does not need.

    The exact DP fills the budget even on a straight ramp (any knot on the line
    gives zero error, so the tie-break just picks one), and a redundant knot is
    not free: it is another slot the model has to learn.  Remove an interior
    knot while doing so raises the RMS error by no more than ``tolerance``.
    """
    k = list(knots)
    limit = (float(tolerance) ** 2) * max(len(pos), 1)
    while len(k) > 2:
        cur = _fit_sse(pos, val, k)
        best = None
        for i in range(1, len(k) - 1):
            cand = k[:i] + k[i + 1:]
            inc = _fit_sse(pos, val, cand) - cur
            if best is None or inc < best[0]:
                best = (inc, cand)
        if best is None or best[0] > limit:
            break
        k = best[1]
    return k


def detect_pitch_anchors(contour_hz, ref_hz: float,
                         max_anchors: int = DEFAULT_PITCH_ANCHORS,
                         tolerance_st: float = PITCH_ANCHOR_TOLERANCE_ST,
                         smooth_frames: int = PITCH_ANCHOR_SMOOTH_FRAMES,
                         octave_st: float = PITCH_ANCHOR_OCTAVE_ST):
    """Reduce one word's f0 contour to at most ``max_anchors`` points.

    This is the Hints V2 replacement for the single word-mean: a word's pitch is
    stored as `(position, semitones)` anchors, and the *model* decides how to
    use them.  Measured on Genshin (words >= 8 code frames): the word mean
    reconstructs the contour to ~2.1 st, 3 anchors to ~0.7 st, and anchors
    chosen at detected positions beat uniformly spaced samples at equal count.

    Args:
        contour_hz: 1-D f0 in Hz over the word span, 0 (or non-finite) where
            unvoiced.  From ``FeatureExtractor.extract``'s ``pitch_contour``.
        ref_hz: the utterance's reference pitch, i.e. the median of its voiced
            word means.  Values are stored in semitones relative to this, which
            keeps the hint independent of the speaker and of the absolute key.
        max_anchors: budget (``transformer.hint_pitch_anchors``).
        tolerance_st: a span flatter than this contributes no extra anchor.
        smooth_frames: median filter width, in f0 frames.
        octave_st: a frame this far from its local median is treated as a
            tracker octave error and replaced (see `_guard_octaves`).

    Returns:
        ``(values, positions, n)`` where ``values`` and ``positions`` are
        ``float32`` arrays of length ``max_anchors`` (zero-padded) and ``n`` is
        the number of valid entries.  ``n == 0`` means the word has no voiced
        frames at all -- a distinct state from "no hint provided", which the
        `(B, n_phone)` state tensor keeps separate.
    """
    vals = np.zeros(int(max_anchors), dtype=np.float32)
    poss = np.zeros(int(max_anchors), dtype=np.float32)
    c = np.asarray(contour_hz, dtype=np.float64).reshape(-1)
    if c.size == 0 or not np.isfinite(ref_hz) or ref_hz <= 0:
        return vals, poss, 0
    voiced = np.isfinite(c) & (c > 0)
    if not voiced.any():
        return vals, poss, 0

    st = np.full(c.shape, np.nan, dtype=np.float64)
    st[voiced] = 12.0 * np.log2(c[voiced] / float(ref_hz))
    st = _masked_median(st, smooth_frames)
    st = _guard_octaves(st, octave_st)

    keep = voiced & np.isfinite(st)
    frames = np.flatnonzero(keep)
    if frames.size == 0:
        return vals, poss, 0
    n_frames = int(c.size)
    # Normalised position inside the word: 0 at the first frame, 1 at the last.
    denom = max(n_frames - 1, 1)
    pos = frames.astype(np.float64) / denom
    y = st[frames]

    if y.size == 1:
        chosen = [0]
    else:
        # A genuinely flat word collapses to one point at its mean.  The test
        # is the contour's own *range*, not the deviation from the endpoint
        # line: a straight 12 st ramp has zero deviation but is not flat.
        if float(np.ptp(y)) <= tolerance_st:
            chosen = []
        else:
            chosen = _prune_knots(
                pos, y, _optimal_knots(pos, y, int(max_anchors)),
                float(tolerance_st))
    if not chosen:
        n = 1
        vals[0] = float(np.mean(y))
        poss[0] = 0.5
        return vals, poss, n
    n = min(len(chosen), int(max_anchors))
    for i, j in enumerate(chosen[:n]):
        vals[i] = float(y[j])
        poss[i] = float(min(max(pos[j], 0.0), 1.0))
    return vals, poss, int(n)


def load_pitch_anchors(path: str) -> Optional[int]:
    """``hints.json`` -> ``words.pitch_anchors`` (the anchor count), or ``None``.

    The count is part of the checkpoint contract: the same anchor slot index
    means a different point of the contour if it changes, so the model compares
    this against ``transformer.hint_pitch_anchors``.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f) or {}
    value = (doc.get("words") or {}).get("pitch_anchors")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def load_word_bert_dim(path: str) -> Optional[int]:
    """``hints.json`` -> ``words.word_bert.dim`` (the per-word state width).

    Like the pitch-anchor count, the width is part of the checkpoint contract:
    a projection trained on 768-d states cannot be fed 384-d ones, and a
    *different* 768-d model would make every dimension mean something else, so
    the file records both the width and the model id (see
    ``load_word_bert_model``).
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f) or {}
    node = (doc.get("words") or {}).get("word_bert") or {}
    try:
        dim = node.get("dim")
        return int(dim) if dim is not None else None
    except (TypeError, ValueError):
        return None


def load_word_bert_model(path: str) -> Optional[str]:
    """``hints.json`` -> ``words.word_bert.model`` (provenance, e.g. bert-base)."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f) or {}
    node = (doc.get("words") or {}).get("word_bert") or {}
    model = node.get("model")
    return str(model) if model else None


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
