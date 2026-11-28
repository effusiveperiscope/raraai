"""Word-rate plan: the control surface between text and the renderer.

A plan is a sequence of **word** and **silence** segments at the 25 Hz code
rate, plus a 100 Hz pitch contour and a per-word level.  It is derived from the
stored ``.cond`` forced alignment and the stage-1 ``.spec`` / ``.f0`` tensors;
nothing here needs a model or a GPU.

    from plan import load_plan

    plan = load_plan("data/.../clip.wav.phone_ids")
    plan.check()                        # hard invariants, raises PlanError
    for seg in plan.segments():         # ('sil', n) / ('word', w, n)
        ...

Why silence is explicit
-----------------------
``word_slot_f`` is **start-to-start**, and the last slot runs to the end of the
clip.  A word's slot therefore contains the pause *after* it, and the last
slot contains the trailing silence (measured: median 7.8 code frames, 98 % of
clips have >= 2).  The first slot can likewise contain the leading silence.
Expanding a word's phones evenly over such a slot would stretch its edge phones
into the silence -- the pacing failure this project exists to remove.  A small
energy VAD over the stage-1 magnitude spectrogram finds the speech end of every
slot; the remainder becomes an explicit ``<sil>`` segment with its own
duration.  ``lead + sum(word_dur + pause_after) == n_frames`` is a hard
invariant, so a plan is a complete length specification, exactly matching the
code tensors it has to render.

Units
-----
* durations: code frames at :data:`CODE_FPS` (25 Hz, one frame = 40 ms)
* pitch: semitones relative to the clip's voiced median (:attr:`Plan.register_st`
  carries that median in semitones re 100 Hz), sampled at the 100 Hz decoder
  rate; ``pitch_hz = 100 * 2 ** ((register_st + pitch_st) / 12)``
* level: dB relative to the clip's median word energy (the stage-1 features are
  peak-normalised per clip, so only within-clip level is recoverable)
* style: :attr:`Plan.intensity_rel`, a speaker-relative z-score of the
  subjective 1-9 DSP score (0 = the character's usual delivery).  This is the
  value a UI slider should set; it is *style*, not loudness -- per-word
  emphasis is :attr:`Plan.word_db`.

The word timings are word-level; phone durations inside a word are an even
split.  That is the one supervised-but-noisy part of the plan (see AGENTS.md
§4 gotcha 3), so adherence is measured at the word level.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import torch

#: Code frame rate.  One code frame is 40 ms; ``spec_codes`` are ``(Q, T)``.
CODE_FPS = 25.0
#: Decoder / f0 frame rate.  The stage-1 tensors are four times the code rate.
PITCH_FPS = 100.0
#: f0 reference for ``register_st`` (``hints/build.py``: ``F0_REF_HZ``).
F0_REF_HZ = 100.0
#: A silent run shorter than this stays inside the word: micro-pauses below
#: ~80 ms are speech rhythm, and trimming them only perturbs phone placement.
MIN_PAUSE_F = 2
#: The magnitude spectrogram floor in every corpus built so far.
SPEC_FLOOR = 1e-3
#: Where ``scratch/fit_intensity_stats.py`` writes the speaker references.
ROOT = Path(__file__).resolve().parent
INTENSITY_STATS = ROOT / "intensity_stats.json"
#: Dataset cache layout; bump when the fields below or the derivation changes.
PLAN_CACHE_VERSION = 1
#: Per-code-frame conditioning built from the plan (order is part of the API).
FRAME_FEATURES = ("pitch_st", "voiced_frac", "word_db", "register_st",
                  "intensity_rel", "is_silence")
#: VAD: a code frame is silent when its mean magnitude is this far below the
#: clip's 95th percentile.  Measured on 280 Genshin clips: pauses >= 2 frames
#: are found in 12 % of words (median 7 frames), which matches the audible gaps.
SILENCE_REL_DB = -35.0
SPEC_BINS = 769


class PlanError(ValueError):
    """The stored clip cannot be expressed as a valid word-rate plan."""


# --------------------------------------------------------------------------- #
# the plan object
# --------------------------------------------------------------------------- #

@dataclass
class Plan:
    """A word-rate, frame-exact description of one utterance.

    All durations are code frames (:data:`CODE_FPS`).  The sequence is
    ``lead_sil`` frames of silence, then for each word ``word_dur[w]`` frames of
    speech followed by ``pause_after[w]`` frames of silence.  Only the last word
    of a run sharing a start frame may carry a pause (the aligner could not
    separate that run, so its speech span is split between the words).
    """

    # text -> phones -> frames
    phone_ids: np.ndarray            # (N,) int32, g2pen ids
    phone_word: np.ndarray           # (N,) int32, phone -> word
    phone_dur: np.ndarray            # (N,) int32, 100 Hz frames per phone
    word_start: np.ndarray           # (W,) int32, speech onset, code frames
    word_dur: np.ndarray             # (W,) int32, speech frames
    pause_after: np.ndarray          # (W,) int32, silence after the word
    lead_sil: int                    # silence before the first word
    # prosody
    pitch_st: np.ndarray             # (4T,) float32, st re the clip's median
    voiced: np.ndarray               # (4T,) bool
    register_st: float               # the clip's voiced median, st re 100 Hz
    word_db: np.ndarray              # (W,) float32, dB re the clip's median word
    #: Clip-level *subjective* performance intensity from the DSP score (1-9).
    #: This is a style descriptor, not the level knob: on Genshin it correlates
    #: 0.62 with `register_st` (0.45 within speaker) and only 0.08 with
    #: peak-normalised level.  Use :attr:`word_db` for per-word emphasis.
    intensity: float
    #: `intensity` as a z-score against the *speaker's* reference (see
    #: ``scratch/fit_intensity_stats.py``), using the global std and a
    #: shrunk speaker mean.  This is the field a downstream UI slider should
    #: set: 0 = the character's usual delivery, +1 = more intense than usual.
    #: NaN when no reference is available.  Neutral UI wording is deliberate
    #: ("Delivery"/"Energy", not "intensity") -- the score is partly pitch.
    intensity_rel: float = float("nan")
    # provenance
    speaker: int = -1
    n_frames: int = 0                # T, code frames; = len(spec_codes[-1])
    has_spec: bool = False           # False = no performance audio, slots as-is
    grouped_words: int = 0           # words sharing a start frame in this clip
    meta: dict = field(default_factory=dict)

    # -- derived views ----------------------------------------------------- #

    @property
    def n_phones(self) -> int:
        return int(self.phone_ids.size)

    @property
    def n_words(self) -> int:
        return int(self.word_start.size)

    def pitch_hz(self) -> np.ndarray:
        """Absolute f0 in Hz at 100 Hz; 0 where unvoiced."""
        if not math.isfinite(self.register_st):
            return np.zeros_like(self.pitch_st, dtype=np.float32)
        hz = F0_REF_HZ * np.exp2((self.register_st + self.pitch_st) / 12.0)
        return np.where(self.voiced, hz, 0.0).astype(np.float32)

    def segments(self) -> Iterator[tuple]:
        """``('sil', n)`` and ``('word', w, n)`` in chronological order."""
        if self.lead_sil:
            yield ("sil", int(self.lead_sil))
        for w in range(self.n_words):
            yield ("word", w, int(self.word_dur[w]))
            if self.pause_after[w]:
                yield ("sil", int(self.pause_after[w]))

    def phone_frames(self) -> list[tuple[int, int, int]]:
        """``(phone_index, start, end)`` in 100 Hz frames.

        Phones are finer than the 25 Hz code rate on purpose: a 40 ms word with
        three phones cannot give each phone a code frame.  The renderer expands
        the phone sequence at 100 Hz and pools the hidden state to the code
        rate, so no phone is dropped.  The spans skip the silence between
        words, exactly like :meth:`segments`.
        """
        counts = np.bincount(self.phone_word, minlength=self.n_words) \
            if self.n_phones else np.zeros(self.n_words, dtype=int)
        out, p = [], 0
        for w in range(self.n_words):
            f = 4 * int(self.word_start[w])
            for _ in range(int(counts[w])):
                d = int(self.phone_dur[p])
                out.append((p, f, f + d))
                f += d
                p += 1
        return out

    def phone_index_100(self) -> np.ndarray:
        """``(4T,)`` phone index at 100 Hz; ``-1`` on silence."""
        idx = np.full(4 * int(self.n_frames), -1, dtype=np.int32)
        for p, s, e in self.phone_frames():
            idx[s:e] = p
        return idx

    def word_table(self) -> list[dict]:
        """Per-word rows, for inspection and for authoring a JSON plan."""
        rows, p = [], 0
        counts = np.bincount(self.phone_word, minlength=self.n_words)
        for w in range(self.n_words):
            n = int(counts[w])
            rows.append({
                "word": w,
                "start_f": int(self.word_start[w]),
                "dur_f": int(self.word_dur[w]),
                "pause_after_f": int(self.pause_after[w]),
                "phones": self.phone_ids[p:p + n].tolist(),
                "db": round(float(self.word_db[w]), 2),
            })
            p += n
        return rows

    # -- invariants -------------------------------------------------------- #

    def check(self) -> None:
        """Raise :class:`PlanError` unless the plan is internally consistent.

        This is deliberately a hard check: a plan that does not partition the
        frame axis exactly cannot be trained on without silent misalignment.
        """
        W, N, T = self.n_words, self.n_phones, int(self.n_frames)
        if T <= 0:
            raise PlanError("n_frames must be positive")
        if self.phone_ids.shape != (N,) or self.phone_word.shape != (N,):
            raise PlanError("phone arrays disagree on N")
        if self.phone_dur.shape != (N,):
            raise PlanError("phone_dur shape")
        if self.word_start.shape != (W,) or self.word_dur.shape != (W,):
            raise PlanError("word arrays disagree on W")
        if self.pause_after.shape != (W,) or self.word_db.shape != (W,):
            raise PlanError("word vectors disagree on W")
        if self.pitch_st.shape != (4 * T,) or self.voiced.shape != (4 * T,):
            raise PlanError(f"pitch must be ({4 * T},), got "
                            f"{self.pitch_st.shape}/{self.voiced.shape}")
        if N and (self.phone_word.min() < 0 or self.phone_word.max() >= W):
            raise PlanError("phone_word out of range")
        if N and np.any(np.diff(self.phone_word) < 0):
            raise PlanError("phone_word must be non-decreasing")
        if W and np.any(np.bincount(self.phone_word, minlength=W) == 0):
            raise PlanError("a word has no phones")
        if np.any(self.phone_dur <= 0):
            raise PlanError("every phone needs >= 1 frame (100 Hz)")
        if np.any(self.word_dur <= 0):
            raise PlanError("every word needs >= 1 frame")
        if np.any(self.pause_after < 0):
            raise PlanError("negative pause")
        total = int(self.lead_sil) + int(self.word_dur.sum()) \
            + int(self.pause_after.sum())
        if total != T:
            raise PlanError(f"segments cover {total} frames, expected {T}")
        if int(self.phone_dur.sum()) != 4 * int(self.word_dur.sum()):
            raise PlanError("phone durations do not cover the word durations")
        if not np.isfinite(self.pitch_st).all():
            raise PlanError("non-finite pitch contour")
        for p, s, e in self.phone_frames():
            if s < 0 or e > 4 * T or e <= s:
                raise PlanError(f"phone {p} span {s}:{e} outside the frame axis")


# --------------------------------------------------------------------------- #
# derivation from .cond + .spec
# --------------------------------------------------------------------------- #

def _as_1d(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _load_tensor(path: Path | str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("f0", "spec", "data", "tensor"):
            if key in obj:
                return obj[key]
        obj = next(iter(obj.values()))
    return obj


def _spec_bins(spec: np.ndarray) -> np.ndarray:
    """Return ``(frames, bins)`` from either orientation seen in the corpora."""
    a = np.asarray(spec, dtype=np.float32)
    if a.ndim == 3:
        a = a.squeeze(0)
    if a.ndim != 2:
        raise PlanError(f"spec must be 2-D, got {a.shape}")
    if a.shape[0] == SPEC_BINS and a.shape[1] != SPEC_BINS:
        a = a.T
    if a.shape[1] != SPEC_BINS:
        raise PlanError(f"spec has {a.shape[1]} bins, expected {SPEC_BINS}")
    return a


def speech_mask_from_spec(spec: np.ndarray, n_code: int,
                          rel_db: float = SILENCE_REL_DB) -> np.ndarray:
    """Boolean ``(n_code,)`` speech activity from a stage-1 magnitude spec.

    The stage-1 features are peak-normalised per clip, so the threshold is
    relative to the clip's own 95th percentile.  A code frame is silent when
    at least half of its four 100 Hz frames are below the threshold; using the
    mean rather than every frame keeps a 10 ms gap from splitting a frame.
    """
    a = _spec_bins(spec)
    e = np.maximum(a.mean(axis=1), SPEC_FLOOR)
    db = 20.0 * np.log10(e)
    ref = float(np.percentile(db, 95))
    thr = max(float(db.min()) + 3.0, ref + float(rel_db))
    silent_audio = db < thr
    silent = np.ones(n_code, dtype=bool)
    n = min(n_code, len(silent_audio) // 4)
    if n:
        blocks = silent_audio[:4 * n].reshape(n, 4)
        silent[:n] = blocks.mean(axis=1) >= 0.5
    return ~silent


def _group_starts(starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct start frames and the word-index bounds of each run."""
    uniq, gid = np.unique(starts, return_inverse=True)
    bounds = np.searchsorted(gid, np.arange(uniq.size + 1), side="left")
    bounds[-1] = len(starts)
    return uniq, bounds


def _edge_run(mask: np.ndarray, at_end: bool) -> int:
    """Length of the True run at one end of ``mask``."""
    m = mask[::-1] if at_end else mask
    n = 0
    for v in m:
        if not v:
            break
        n += 1
    return n


def _allocate(total: int, weights: Sequence[float]) -> np.ndarray:
    """Split ``total`` integer frames over ``len(weights)`` bins, min 1 each."""
    n = len(weights)
    if n == 0:
        raise PlanError("cannot allocate frames to zero words")
    if total < n:
        raise PlanError(f"cannot split {total} frames over {n} bins")
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n, dtype=np.float64)
    w = w / w.sum()
    d = np.maximum(1, np.floor(total * w).astype(np.int64))
    diff = int(total - d.sum())
    if diff > 0:
        frac = total * w - np.floor(total * w)
        for i in np.argsort(-frac):
            if diff == 0:
                break
            d[i] += 1
            diff -= 1
        # floor() + the fractional pass can still fall short when n is large
        # relative to total; top up in order.
        i = 0
        while diff > 0:
            d[i % n] += 1
            diff -= 1
            i += 1
    elif diff < 0:
        for i in np.argsort(-d):
            if diff == 0:
                break
            if d[i] > 1:
                d[i] -= 1
                diff += 1
        if diff < 0:
            raise PlanError("cannot keep every word at >= 1 frame")
    return d


def _allocate_min(total: int, mins: Sequence[int],
                  weights: Sequence[float]) -> np.ndarray:
    """Split ``total`` frames ensuring ``d[i] >= mins[i]``, weighted above that."""
    mins = np.asarray(mins, dtype=np.int64)
    n = len(mins)
    if n == 0:
        raise PlanError("cannot allocate frames to zero bins")
    if total < int(mins.sum()):
        raise PlanError(f"cannot give {n} bins their minimum "
                        f"({int(mins.sum())}) from {total} frames")
    extra = int(total - mins.sum())
    if extra == 0:
        return mins
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n, dtype=np.float64)
    w = w / w.sum()
    add = np.floor(extra * w).astype(np.int64)
    left = int(extra - add.sum())
    for i in np.argsort(-(extra * w - add)):
        if left == 0:
            break
        add[i] += 1
        left -= 1
    return mins + add


def _anchors(cond: dict, w: int, n_anchor: int) -> tuple[np.ndarray, np.ndarray, int]:
    """``(positions, semitones, n_valid)`` for one word.

    Falls back to the v1 layout (one anchor at ``pos=0.5`` holding
    ``pitch_word_st``) when the anchor keys are absent, exactly as
    ``dataset_codes._load_pitch_anchors`` does.
    """
    voiced_arr = cond.get("pitch_word_voiced")
    voiced = True if voiced_arr is None else bool(_as_1d(voiced_arr)[w])
    if not voiced:
        return np.array([0.0]), np.array([0.0]), 0
    if "pitch_anchors_st" in cond and "pitch_anchor_n" in cond:
        st = np.asarray(cond["pitch_anchors_st"], dtype=np.float32)
        pos = np.asarray(cond["pitch_anchors_pos"], dtype=np.float32)
        n = int(_as_1d(cond["pitch_anchor_n"])[w])
        if st.ndim == 2 and w < st.shape[0] and n > 0:
            return pos[w, :n], st[w, :n], n
    word_st = cond.get("pitch_word_st")
    mean = 0.0 if word_st is None else float(_as_1d(word_st)[w])
    mean = 0.0 if not np.isfinite(mean) else mean
    return np.array([0.5]), np.array([mean]), 1


def _interp_anchor(st: np.ndarray, pos: np.ndarray, n: int, n_sample: int) -> np.ndarray:
    """Resample a word's anchor contour onto ``n_sample`` frames."""
    if n <= 0:
        return np.zeros(n_sample, dtype=np.float32)
    order = np.argsort(pos[:n])
    p = np.clip(pos[:n][order], 0.0, 1.0)
    v = st[:n][order]
    if p.size == 1:
        return np.full(n_sample, v[0], dtype=np.float32)
    # Deduplicate equal positions; np.interp needs a strictly increasing xp.
    keep = np.concatenate(([True], np.diff(p) > 1e-6))
    p, v = p[keep], v[keep]
    if p.size == 1:
        return np.full(n_sample, v[0], dtype=np.float32)
    x = (np.arange(n_sample, dtype=np.float32) + 0.5) / max(n_sample, 1)
    return np.interp(x, p, v).astype(np.float32)


def plan_from_cond(cond: dict, phone_ids, n_frames: int, *,
                   spec: Optional[np.ndarray] = None,
                   min_pause: int = MIN_PAUSE_F,
                   speaker: int = -1,
                   intensity_ref: Optional[float] = None,
                   intensity_scale: Optional[float] = None) -> Plan:
    """Derive a :class:`Plan` from a ``.cond`` record and the code length.

    Args:
        cond: the loaded ``.cond`` dict (word fields required).
        phone_ids: stored phone ids, ``(N,)``.
        n_frames: length of ``spec_codes`` (T, code frames).
        spec: stage-1 magnitude spectrogram (``S x 769`` or transposed); when
            absent, the silence mask is all-speech and no pauses are split.
        min_pause: silent runs shorter than this stay in the word.
        speaker: stage-2 speaker id, carried through for the renderer/stats.
        intensity_ref: the speaker's intensity reference, in raw score units;
            when given, ``intensity_rel = (intensity - ref) / scale``.
        intensity_scale: global std for the z-score; defaults to 1.0.

    Raises:
        PlanError: the record does not describe this phone sequence, or a word
            run cannot be given one frame each.
    """
    phone_ids = np.asarray(phone_ids, dtype=np.int64).reshape(-1)
    n_phone = int(phone_ids.size)
    n = int(n_frames)
    if n <= 0:
        raise PlanError("n_frames must be positive")
    if cond.get("n_phone") is not None and int(cond["n_phone"]) != n_phone:
        raise PlanError(f"stale .cond: n_phone {cond['n_phone']} != {n_phone}")
    need = ("phone_word_idx", "word_start_f", "word_slot_f")
    if any(k not in cond for k in need):
        raise PlanError("no word fields (hints/build.py --words)")

    pwi = np.asarray(cond["phone_word_idx"]).reshape(-1).astype(np.int64)
    starts = np.asarray(cond["word_start_f"]).reshape(-1).astype(np.int64)
    slots = np.asarray(cond["word_slot_f"]).reshape(-1).astype(np.int64)
    W = int(starts.size)
    if W == 0 or slots.size != W or pwi.size != n_phone:
        raise PlanError("word/phone field lengths disagree")
    if pwi.min() < 0 or pwi.max() >= W or np.any(np.diff(pwi) < 0):
        raise PlanError("phone_word_idx is not a monotone word partition")
    if np.any(np.diff(starts) < 0):
        raise PlanError("word starts are not non-decreasing")

    # A word with no phones cannot be rendered and contributes no text; the
    # corpus has them (114/400 Genshin val clips carry at least one, usually a
    # punctuation token).  Drop them and renumber, keeping a map back to the
    # .cond row so the pitch anchors still resolve.
    old_of = np.arange(W, dtype=np.int64)
    counts_all = np.bincount(pwi, minlength=W)
    if np.any(counts_all == 0):
        old_of = np.where(counts_all > 0)[0]
        remap = -np.ones(W, dtype=np.int64)
        remap[old_of] = np.arange(old_of.size)
        pwi = remap[pwi]
        starts, slots = starts[old_of], slots[old_of]
        W = int(old_of.size)
        if W == 0:
            raise PlanError("no word has any phones")

    if spec is not None:
        speech = speech_mask_from_spec(spec, n)
    else:
        speech = np.ones(n, dtype=bool)
    silent = ~speech

    counts = np.bincount(pwi, minlength=W)
    uniq, bounds = _group_starts(starts)

    lead = int(min(max(uniq[0], 0), n))
    word_start = np.zeros(W, dtype=np.int64)
    word_dur = np.zeros(W, dtype=np.int64)
    pause = np.zeros(W, dtype=np.int64)

    for gi, u in enumerate(uniq):
        s = int(np.clip(u, 0, n))
        ext_end = int(np.clip(uniq[gi + 1], 0, n)) if gi + 1 < len(uniq) else n
        e = ext_end
        if e <= s:
            raise PlanError(f"word run {gi} has no frame span")
        # leading silence inside the first slot (the aligner pins start 0 even
        # when the first phone starts 1-2 frames later)
        if gi == 0:
            k = _edge_run(silent[s:e], at_end=False)
            if k >= 1 and k < e - s:
                s += k
                lead += k
        # trailing silence is a pause, not part of the word
        last = gi == len(uniq) - 1
        words = np.arange(bounds[gi], bounds[gi + 1], dtype=np.int64)
        k = _edge_run(silent[s:e], at_end=True)
        # Keep enough code frames for one 100 Hz frame per phone and one per
        # word: a VAD that hears a short run as silence must not make it
        # unrepresentable.  Anything still short is a broken alignment.
        # one 100 Hz frame per phone is the hard floor; words above it share
        # the rest in proportion to their phone counts
        mins = -(-counts[words] // 4)
        need = int(mins.sum())
        if k >= min_pause or (last and k >= 1):
            e -= min(k, max(0, (e - s) - need))
        if e - s < need:
            raise PlanError(f"word run {gi}: {e - s} frames for "
                            f"{len(words)} words/{int(counts[words].sum())} phones")
        durs = _allocate_min(e - s, mins, counts[words])
        word_start[words] = s + np.concatenate(([0], np.cumsum(durs)[:-1]))
        word_dur[words] = durs
        pause[words[-1]] = ext_end - (s + int(durs.sum()))
        if pause[words[-1]] < 0:
            raise PlanError("pause went negative")

    # Per-phone durations at 100 Hz, not 25 Hz: a 2-code-frame word (80 ms) can
    # hold three or four phones, and forcing one code frame each would stretch
    # the word or drop a phone.  The word duration stays the authority; the
    # phones are a sub-frame split of it, and the renderer pools 4:1.
    phone_dur = np.zeros(n_phone, dtype=np.int64)
    for w in range(W):
        idx = np.where(pwi == w)[0]
        phone_dur[idx] = _allocate(4 * int(word_dur[w]), np.ones(idx.size))

    # pitch contour
    register = float(_as_1d(cond.get("register_st", [np.nan]))[0])
    if not math.isfinite(register):
        register = float("nan")
    pitch = np.zeros(4 * n, dtype=np.float32)
    voiced = np.zeros(4 * n, dtype=bool)
    for w in range(W):
        pos, st, n_a = _anchors(cond, int(old_of[w]), 0)
        if n_a == 0:
            continue
        s4 = int(word_start[w]) * 4
        n4 = int(word_dur[w]) * 4
        if n4 <= 0:
            continue
        pitch[s4:s4 + n4] = _interp_anchor(st, pos, n_a, n4)
        voiced[s4:s4 + n4] = True

    # per-word level, dB re the clip's median word energy
    word_db = np.zeros(W, dtype=np.float32)
    if spec is not None:
        e = np.maximum(_spec_bins(spec).mean(axis=1), SPEC_FLOOR)
        e_code = np.ones(n, dtype=np.float32)
        m = min(n, len(e) // 4)
        if m:
            e_code[:m] = e[:4 * m].reshape(m, 4).mean(axis=1)
        for w in range(W):
            s, d = int(word_start[w]), int(word_dur[w])
            word_db[w] = 20.0 * math.log10(max(float(e_code[s:s + d].mean()), SPEC_FLOOR)
                                           if d else SPEC_FLOOR)
        ref = float(np.median(word_db))
        if math.isfinite(ref):
            word_db -= ref

    intensity = float(_as_1d(cond.get("intensity", [np.nan]))[0])
    intensity_rel = float("nan")
    if intensity_ref is not None and math.isfinite(intensity):
        scale = float(intensity_scale or 1.0)
        if scale > 0:
            intensity_rel = (intensity - float(intensity_ref)) / scale
    grouped = int(W - uniq.size)

    plan = Plan(
        phone_ids=phone_ids.astype(np.int32),
        phone_word=pwi.astype(np.int32),
        phone_dur=phone_dur.astype(np.int32),
        word_start=word_start.astype(np.int32),
        word_dur=word_dur.astype(np.int32),
        pause_after=pause.astype(np.int32),
        lead_sil=int(lead),
        pitch_st=pitch,
        voiced=voiced,
        register_st=register,
        word_db=word_db,
        intensity=intensity,
        intensity_rel=intensity_rel,
        speaker=int(speaker),
        n_frames=n,
        has_spec=spec is not None,
        grouped_words=grouped,
    )
    plan.check()
    return plan


# --------------------------------------------------------------------------- #
# dataset cache
# --------------------------------------------------------------------------- #

def frame_features(plan: "Plan") -> np.ndarray:
    """``(T, len(FRAME_FEATURES))`` conditioning at the 25 Hz code rate.

    This is what the renderer reads at frame *t* in addition to the expanded
    phone memory.  ``pitch_st`` and ``voiced_frac`` summarise the four 100 Hz
    pitch samples of the frame; ``word_db`` is repeated over its word's frames;
    pauses are ``is_silence=1`` with zero prosody.
    """
    T = int(plan.n_frames)
    out = np.zeros((T, len(FRAME_FEATURES)), dtype=np.float32)
    pitch = plan.pitch_st.reshape(T, 4) if plan.pitch_st.size == 4 * T else None
    voiced = plan.voiced.reshape(T, 4) if plan.voiced.size == 4 * T else None
    if pitch is not None:
        v = np.maximum(voiced, 0).astype(np.float32)
        denom = np.maximum(v.sum(axis=1), 1)
        out[:, 0] = (pitch * v).sum(axis=1) / denom
        out[:, 1] = v.mean(axis=1)
    silence = np.ones(T, dtype=bool)
    for w in range(plan.n_words):
        s, d = int(plan.word_start[w]), int(plan.word_dur[w])
        if d <= 0:
            continue
        silence[s:s + d] = False
        out[s:s + d, 2] = float(plan.word_db[w])
    out[:, 3] = float(plan.register_st) if math.isfinite(plan.register_st) else 0.0
    out[:, 4] = float(plan.intensity_rel) if math.isfinite(plan.intensity_rel) else 0.0
    out[:, 5] = silence.astype(np.float32)
    return out


def save_plan_cache(plan: "Plan", path: Path | str) -> None:
    """Write a compact cache (~2-3 KB) of the derivation, not the `.spec`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" when the name does not already end with it, so
    # the temp name must; otherwise the rename looks for a file that is not
    # there (FileNotFoundError).
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(
        tmp,
        version=np.int64(PLAN_CACHE_VERSION),
        n_frames=np.int64(plan.n_frames),
        n_phone=np.int64(plan.n_phones),
        n_words=np.int64(plan.n_words),
        phone_dur=plan.phone_dur.astype(np.int16),
        phone_word=plan.phone_word.astype(np.int16),
        word_start=plan.word_start.astype(np.int32),
        word_dur=plan.word_dur.astype(np.int32),
        pause_after=plan.pause_after.astype(np.int32),
        lead_sil=np.int64(plan.lead_sil),
        pitch_st=plan.pitch_st.astype(np.float16),
        voiced=np.packbits(plan.voiced),
        register_st=np.float64(plan.register_st),
        word_db=plan.word_db.astype(np.float16),
        intensity=np.float64(plan.intensity),
        intensity_rel=np.float64(plan.intensity_rel),
        has_spec=np.bool_(plan.has_spec),
        grouped_words=np.int64(plan.grouped_words),
    )
    tmp.replace(p)


def load_plan_cache(path: Path | str, phone_ids, speaker: int = -1) -> Optional["Plan"]:
    """Read :func:`save_plan_cache`; ``None`` on version/shape mismatch."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            if int(z["version"]) != PLAN_CACHE_VERSION:
                return None
            n_phone = int(z["n_phone"])
            if n_phone != int(np.asarray(phone_ids).size):
                return None
            T = int(z["n_frames"])
            voiced = np.unpackbits(z["voiced"])[: 4 * T].astype(bool)
            phone_ids = np.asarray(phone_ids, dtype=np.int64).reshape(-1)
            plan = Plan(
                phone_ids=phone_ids.astype(np.int32),
                phone_word=z["phone_word"].astype(np.int32),
                phone_dur=z["phone_dur"].astype(np.int32),
                word_start=z["word_start"].astype(np.int32),
                word_dur=z["word_dur"].astype(np.int32),
                pause_after=z["pause_after"].astype(np.int32),
                lead_sil=int(z["lead_sil"]),
                pitch_st=z["pitch_st"].astype(np.float32),
                voiced=voiced,
                register_st=float(z["register_st"]),
                word_db=z["word_db"].astype(np.float32),
                intensity=float(z["intensity"]),
                intensity_rel=float(z["intensity_rel"]),
                speaker=int(speaker),
                n_frames=T,
                has_spec=bool(z["has_spec"]),
                grouped_words=int(z["grouped_words"]),
            )
    except Exception:  # noqa: BLE001
        return None
    try:
        plan.check()
    except PlanError:
        return None
    return plan


# --------------------------------------------------------------------------- #
# loading from disk
# --------------------------------------------------------------------------- #

def _stage1_siblings(spec_codes: Path) -> tuple[Optional[Path], Optional[Path]]:
    """``<base>.spec_codes`` -> the stage-1 ``.spec`` / ``.f0`` paths.

    Corpora live side by side as ``*_stage2`` (codes, cond) and ``*_stage1``
    (wave, spec, f0), so the mapping is the directory name.  Corpora without a
    stage-1 directory (expresso, vctk so far) return ``(None, None)`` and the
    plan falls back to "the whole slot is speech".
    """
    s = str(spec_codes)
    if "_stage2/" in s:
        s = s.replace("_stage2/", "_stage1/")
    base = s[: -len(".spec_codes")] if s.endswith(".spec_codes") else str(Path(s).with_suffix(""))
    spec, f0 = Path(base + ".spec"), Path(base + ".f0")
    return (spec if spec.exists() else None,
            f0 if f0.exists() else None)


def code_length(path: Path | str, n_books: int = 8) -> int:
    """Frames in an ``(Q, T)`` code tensor, tolerating a transposed file."""
    c = _load_tensor(path)
    a = torch.as_tensor(c)
    if a.ndim != 2:
        raise PlanError(f"codes must be 2-D, got {tuple(a.shape)}")
    if a.shape[0] == n_books:
        return int(a.shape[1])
    if a.shape[1] == n_books:
        return int(a.shape[0])
    raise PlanError(f"codes {tuple(a.shape)} do not have {n_books} books")


def load_intensity_stats(path: Path | str | None = None) -> dict | None:
    """Load ``intensity_stats.json`` if it exists (``None`` otherwise)."""
    p = Path(path) if path is not None else INTENSITY_STATS
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def intensity_reference(stats: dict | None, speaker: int | None) -> tuple[float | None, float | None]:
    """``(reference, scale)`` for the speaker-relative intensity slider.

    The speaker mean is shrunk towards the global mean with ``shrink_k``
    pseudo-clips, so a character with 3 lines does not get a wild reference;
    the z-score uses the global std because per-speaker std is noisier.
    """
    if not stats:
        return None, None
    g = stats.get("global") or {}
    g_mean = float(g.get("mean", float("nan")))
    g_std = float(g.get("std", 1.0) or 1.0)
    if not math.isfinite(g_mean) or not math.isfinite(g_std) or g_std <= 0:
        return None, None
    k = float(stats.get("shrink_k", 0.0) or 0.0)
    row = (stats.get("speakers") or {}).get(str(speaker)) if speaker is not None else None
    if row is None:
        return g_mean, g_std
    n = float(row.get("n", 0) or 0)
    mean = float(row.get("mean", g_mean))
    ref = (n * mean + k * g_mean) / (n + k) if n + k > 0 else g_mean
    return ref, g_std


def load_plan(phone_ids_path: Path | str, *,
              spec_codes_path: Optional[Path | str] = None,
              with_spec: bool = True,
              speaker: Optional[int] = None,
              intensity_stats: dict | str | Path | None = None) -> Plan:
    """Load one clip's plan from ``<base>.phone_ids`` (+ ``.cond``/``.spec``).

    ``speaker`` is the stage-2 speaker id from the filelist.  When
    ``intensity_stats`` (a dict or a path) is supplied -- or the default
    ``intensity_stats.json`` exists -- the plan also carries
    ``intensity_rel``, the slider-ready z-score.  Pass ``intensity_stats=False``
    to disable the lookup.
    """
    p = Path(phone_ids_path)
    base = str(p)[: -len(".phone_ids")] if str(p).endswith(".phone_ids") else str(p.with_suffix(""))
    cond_path = Path(base + ".cond")
    if not cond_path.exists():
        raise PlanError(f"no .cond next to {p}")
    codes = Path(spec_codes_path) if spec_codes_path else Path(base + ".spec_codes")
    if not codes.exists():
        raise PlanError(f"no spec_codes next to {p}")
    cond = torch.load(cond_path, map_location="cpu", weights_only=False)
    if not isinstance(cond, dict):
        raise PlanError(f"{cond_path} is not a dict")
    n = code_length(codes)
    spec = None
    spec_path: Optional[Path] = None
    f0_path: Optional[Path] = None
    if with_spec:
        spec_path, f0_path = _stage1_siblings(codes)
        if spec_path is not None:
            spec = _load_tensor(spec_path)
    stats = None
    if intensity_stats is not False:
        stats = (intensity_stats if isinstance(intensity_stats, dict)
                 else load_intensity_stats(intensity_stats))
    ref, scale = intensity_reference(stats, speaker)
    plan = plan_from_cond(cond, _load_tensor(p), n, spec=spec,
                          speaker=-1 if speaker is None else int(speaker),
                          intensity_ref=ref, intensity_scale=scale)
    plan.meta = {"cond": str(cond_path), "codes": str(codes),
                 "spec": None if spec_path is None else str(spec_path),
                 "f0": None if f0_path is None else str(f0_path)}
    return plan


__all__ = [
    "Plan", "PlanError", "plan_from_cond", "load_plan", "code_length",
    "load_intensity_stats", "intensity_reference", "speech_mask_from_spec",
    "frame_features", "save_plan_cache", "load_plan_cache",
    "FRAME_FEATURES", "PLAN_CACHE_VERSION",
    "CODE_FPS", "PITCH_FPS", "F0_REF_HZ",
]
