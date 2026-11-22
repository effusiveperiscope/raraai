"""Dataset / collation for the stage-2 discrete-code transformer.

Reads the filelists produced by ``2_preprocess.py``, whose lines have the
form::

    phone_ids_path|spec_codes_path|pitch_codes_path|sid

with

    phone_ids    (n_phone,)                 long
    spec_codes   (Q_spec,  T_codes)         long
    pitch_codes  (Q_pitch, T_codes)         long

The collator pads to the longest item in the batch and additionally reports
``phone_lengths`` and ``code_lengths`` (number of valid code frames).  The
transformer uses those lengths to build the delay pattern and its loss mask,
so padded entries never contribute.  It also reports ``complete``, a boolean
per item: ``True`` when the (possibly cropped) segment still ends where the
clip ends, so the transformer may put an end-of-utterance target there
(``transformer.eos_token``); random training crops are usually mid-clip and
must not get one.

With ``transformer.fim`` the collator also emits ``fim`` (bool), ``fim_prefix_len``
and ``fim_mid_len`` per item.  ``CodeDataset`` draws them for a fraction
``train.fim_rate`` of *training* items (never validation), and the transformer
reorders those items to ``prefix <SUF> suffix <MID> middle`` at loss time.  A
plain AR item (``fim=False``) is left untouched, so AR and FIM samples share a
batch.

Word-boundary cropping (``train.word_crop``)
--------------------------------------------
A clip longer than ``train.max_frames`` used to be cut to a random audio window
while its **whole** transcript stayed in the sample, so most of its text had no
audio at all and the model learned to skip utterance-initial words (measured at
23.9 % of training frames for ``max_frames: 256``).  The crop is now snapped
inwards to whole words: ``.cond`` carries ``phone_word_idx`` /
``word_start_f`` / ``word_slot_f`` (``hints/build.py --words``), so the window
starts at a word onset and ends where the first dropped word starts, and the
returned phone slice contains exactly the words in the audio.  The per-phone
hint arrays are sliced by the same span.  A clip whose ``.cond`` has no word
timings (or one whose whole-word window would be empty) falls back to the old
behaviour and is counted in ``n_word_crop_fallback``.

Conditioning hints (``transformer.hints``)
------------------------------------------
``hints/build.py`` writes a sibling ``<base>.cond`` next to each
``<base>.phone_ids`` holding *continuous* values (intensity 1-9, register in
semitones, a 384-d sentence vector, and -- once the alignment pass has run --
per-word pitch/timing).  ``intensity`` is *perceptual* intensity (emotional /
phonation effort, loudness-independent by construction -- its DSP features
peak-normalise) and is a deliberately coarse, clip-level, end-user control; see
``configs/transformer.yaml``.  With ``hints=True`` this dataset reads that file
and emits level ids, so the *bucket count* stays a config change and never needs
the corpus rebuilt:


    hint_int      (B,)      long   intensity level,      0 = absent
    hint_reg      (B,)      long   register level,       0 = absent
    hint_pitch    (B, N)    long   per-phone word-pitch level
                                   (0 absent, 1..P voiced bins, P+1 unvoiced)
    hint_slot     (B, N)    long   per-phone word-duration bucket (0 absent)
    hint_pitch_val(B, N, A) float  word's pitch-anchor values, repeated over
                                   its phones (Hints V2; semitones)
    hint_pitch_pos(B, N, A) float  anchor positions, fraction of the word
    hint_pitch_n  (B, N)    long   valid anchors for this phone's word
                                   (0 = the word has no voiced frames)
    hint_pitch_state(B, N)  long   0 absent, 1 voiced, 2 unvoiced
    hint_phone_frac(B, N)   float  phone's position inside its word, 0..1
    hint_word     (B, N, D) float  word's BERT state, repeated over its phones
                                   (Hints V2.1; D = 768)
    hint_word_valid(B, N)   bool   False -> the model uses its trained
                                   ``word_null`` vector (absent / dropped)

The per-phone arrays are gathered from the word-level fields via
``phone_word_idx``, and cropped in lockstep with ``max_phone_length``.  Edges for
intensity/register come from ``hints.json`` (equal-frequency quantiles fitted by
``hints/build.py``); word pitch and word slots use *fixed* cuts
(``hints.levels.DEFAULT_PITCH_EDGES`` / ``DEFAULT_SLOT_BUCKETS``) so a level id
means the same thing in every corpus and in an inference UI.

**Pitch is stored twice on purpose.**  ``hint_pitch`` is the v1 level id
(``transformer.hint_pitch_mode: table``); ``hint_pitch_val``/``pos``/``n`` are
the V2 anchors, repeated over each phone because the encoder input is per phone
and a word's contour cannot be represented by one scalar per phone (measured:
3 anchors projected to one value per phone is no better than 2).  Every phone
of a word carries the whole anchor vector, and ``hint_phone_frac`` makes the
phone's place in that contour explicit.  A v1 ``.cond`` (no anchor keys) is
read as **one anchor** at ``pos=0.5`` with the old word mean, so old corpora
keep training.

Dropout is what keeps this from becoming an oracle/voice-conversion renderer:
``hint_dropout`` zeroes each channel independently per item and
``hint_null_all`` zeroes *all* of them, both only on training items.  Level 0 is
therefore a genuinely trained "no hint" state, not an untrained row.

Length bucketing
----------------
``LengthBucketBatchSampler`` (used by ``2_training.py`` via
``train.bucket_batches``) orders the epoch so a batch contains similarly long
clips and is therefore padded only to *its own* width.  ``compute_lengths``
provides the per-item frame counts (cached in ``<filelist>.lengths.npz``, keyed
by tensor path).  See the class docstring for the measurements: ~60 % of the
historical step was padding.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import speaker_map as speaker_map_lib
from hints.levels import (DEFAULT_PITCH_EDGES, DEFAULT_SLOT_BUCKETS,
                          bucket_levels, load_levels, register_rel_st,
                          slot_bucket, word_pitch_levels)

#: Hint channels that ``hint_dropout`` can switch off independently.
HINT_CHANNELS = ("intensity", "register", "pitch", "timing", "word_sem")

#: ``.cond`` suffix, matching ``hints.build`` (sibling of ``<base>.phone_ids``).
COND_SUFFIX = ".cond"


def cond_path_for(phone_ids_path: str) -> str:
    """``.../clip.wav.phone_ids`` -> ``.../clip.wav.cond``."""
    return phone_ids_path[: -len(".phone_ids")] + COND_SUFFIX


def _norm_path(path: str) -> str:
    """Path key for blocklists: separators normalised, no trailing spaces."""
    return os.path.normpath(str(path).strip()).replace("\\", "/")


class CodeDataset(Dataset):
    def __init__(
        self,
        filelist: str,
        is_train: bool = True,
        max_frames: Optional[int] = None,
        max_phone_length: Optional[int] = None,
        random_seed: int = 42,
        speaker_map=None,
        fim: bool = False,
        fim_rate: float = 0.0,
        fim_min_frames: int = 8,
        fim_mid_frac: float = 0.5,
        hints: bool = False,
        hints_json: Optional[str] = None,
        hint_dropout: Optional[Dict[str, float]] = None,
        hint_null_all: float = 0.0,
        hint_pitch_edges=None,
        n_slot_buckets: int = DEFAULT_SLOT_BUCKETS,
        word_crop: bool = True,
        alignment_blocklist: Optional[str] = None,
        stage_direction_blocklist: Optional[str] = None,
    ):
        self.is_train = is_train
        self.max_frames = max_frames
        self.max_phone_length = max_phone_length
        # Crop the *text* in lock-step with a random audio window (see
        # ``_crop_frames``).  Only matters for clips longer than ``max_frames``;
        # without it those items pair a random 10 s window with the whole
        # transcript, which teaches the model that the start of the text does
        # not have to be voiced (measured: 24 % of training frames).  Needs the
        # ``phone_word_idx`` / ``word_start_f`` / ``word_slot_f`` fields that
        # ``hints/build.py --words`` writes into the sibling ``.cond``; a clip
        # without them falls back to the old behaviour and is counted in
        # ``n_word_crop_fallback``.
        self.word_crop = bool(word_crop)
        self.n_word_crop_fallback = 0
        # Optional blocklists, dropped at filelist-build time.  The first comes
        # from ``scratch/audit_word_alignment.py`` (a word given < 1 code frame
        # per phone => ~40 ms/phone, which teaches the model to swallow it);
        # the second from ``scratch/find_stage_directions.py`` (``*sigh*``-style
        # annotations phonemized as words until the corpus is rebuilt with
        # ``features.strip_stage_directions``).
        self.n_blocklisted = 0
        blocklist = set()
        for path in (alignment_blocklist, stage_direction_blocklist):
            if not path:
                continue
            with open(str(path), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        blocklist.add(_norm_path(line))
        self.rng = random.Random(random_seed)
        # Length table for the batch sampler (see ``compute_lengths``): the
        # filelist is kept because the cache path is derived from it.
        self.filelist = str(filelist)
        self._lengths: Optional[List[int]] = None
        # Fill-in-the-middle span sampling (see ``modeling/transformer.py``).
        # Per train item, ``fim_rate`` is the probability of returning a
        # prefix/suffix/middle split instead of the plain AR sample.  Spans are
        # drawn here (not in the model) so the model stays a pure function and
        # the split is reproducible with the dataset seed.
        self.fim = bool(fim)
        self.fim_rate = float(fim_rate)
        self.fim_min_frames = int(fim_min_frames)
        self.fim_mid_frac = float(fim_mid_frac)

        # ---- conditioning hints (see the module docstring) ----------------- #
        self.hints = bool(hints)
        self.hint_null_all = float(hint_null_all or 0.0)
        self.hint_dropout = {c: 0.0 for c in HINT_CHANNELS}
        for channel, prob in (dict(hint_dropout) if hint_dropout else {}).items():
            if channel not in HINT_CHANNELS:
                raise ValueError(
                    f"unknown hint channel {channel!r}; expected one of "
                    f"{HINT_CHANNELS}")
            self.hint_dropout[channel] = float(prob)
        self.pitch_edges = [float(e) for e in
                            (hint_pitch_edges or DEFAULT_PITCH_EDGES)]
        self.n_slot_buckets = int(n_slot_buckets)
        self.intensity_edges: Optional[List[float]] = None
        self.register_edges: Optional[List[float]] = None
        # ``{str(sid): semitones}`` when this corpus was built with
        # ``hints/build.py --relative-register`` (HINTS_PLAN.md §8.5): the
        # register value is then bucketed as its deviation from the speaker's
        # own median, which is orthogonal to ``sid`` by construction.  ``None``
        # means the historical absolute level.  The switch lives in
        # ``hints.json``, so the dataset and the model read it from one place.
        self.register_speaker_means: Optional[Dict[str, float]] = None
        # ``+1`` for the "absent" level 0; pitch additionally has the explicit
        # unvoiced class, so its table is ``len(edges) + 2`` rows (see
        # ``hints.levels.word_pitch_levels``).
        self.n_intensity_levels = 0
        self.n_register_levels = 0
        self.n_pitch_levels = len(self.pitch_edges) + 1
        # Counters for the two "the .cond file is not usable" cases, so a
        # mismatched corpus prints a diagnosis instead of silently training
        # without hints.
        self.n_cond_missing = 0
        self.n_cond_stale = 0
        if self.hints:
            if not hints_json or not os.path.exists(str(hints_json)):
                raise FileNotFoundError(
                    "transformer.hints is on but transformer.hints_json "
                    f"({hints_json!r}) does not exist. Run hints/build.py over "
                    "this corpus first: it writes the equal-frequency edges that "
                    "turn a continuous hint into a level id.")
            levels = load_levels(str(hints_json))
            for name, attr, count in (("intensity", "intensity_edges",
                                       "n_intensity_levels"),
                                      ("register", "register_edges",
                                       "n_register_levels")):
                spec = levels.get(name) or {}
                if not spec.get("edges"):
                    raise ValueError(
                        f"{hints_json} has no fitted edges for hint {name!r} "
                        f"(found: {sorted(levels)}). hints/build.py only fits "
                        "them once the whole corpus is covered -- re-run it "
                        "without --no-edges.")
                setattr(self, attr, [float(e) for e in spec["edges"]])
                setattr(self, count, len(spec["edges"]) + 1)
            self.register_speaker_means = (levels.get("register") or {}).get(
                "speaker_means")

        # ``speaker_map`` is the same JSON structure used by 2_preprocess
        # (see speaker_map.py).  Preprocessed filelists no longer contain
        # the audio path, so only its ``sid_map`` (raw id -> canonical id)
        # can be applied here; path-based rules must be applied during
        # preprocessing.  Passed as a path or an already-loaded dict.
        if isinstance(speaker_map, str):
            speaker_map = speaker_map_lib.load_speaker_map(speaker_map)
        self.speaker_map = speaker_map

        self.items: List[dict] = []
        with open(filelist, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 4:
                    continue
                if blocklist and _norm_path(parts[0]) in blocklist:
                    self.n_blocklisted += 1
                    continue
                sid = int(parts[3])
                if self.speaker_map is not None:
                    resolved = speaker_map_lib.resolve_sid(
                        self.speaker_map, path=None, raw_sid=sid)
                    if resolved is None:
                        continue
                    sid = resolved
                self.items.append(
                    {
                        "phone_ids": parts[0],
                        "spec_codes": parts[1],
                        "pitch_codes": parts[2],
                        "sid": sid,
                        # Pre-``sid_map`` id: the one ``hints/build.py`` keyed
                        # its per-speaker register means on, so the lookup is
                        # unambiguous even when a speaker map remaps ``sid``.
                        "sid_raw": int(parts[3]),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def speaker_ids(self) -> List[int]:
        return [it["sid"] for it in self.items]

    def compute_lengths(self, cache: bool = True) -> List[int]:
        """Per-item frame count (post-``max_frames`` clamp) for length bucketing.

        The value is what the item contributes to the collated batch width, so
        ``min(spec_frames, max_frames)``.  It is an upper bound when
        ``word_crop`` trims a long clip inwards to a word boundary (the crop
        only ever shortens), which is exactly what a batch sampler wants:
        grouping by the bound keeps every batch's true width close to its
        group's length.

        Two ways to read it, and the expensive one is only used for items not
        already in the cache: ``torch.load(..., mmap=True)`` opens the header
        and reads the shape without paging the payload (measured 0.23 ms/item,
        ~13 s for the 146 k-item corpus), versus ``os.path.getsize`` which is
        ~200x faster but would encode the container's header size as a magic
        constant.

        The cache is ``<filelist>.lengths.npz`` and is keyed by **tensor path**,
        not by filelist position: it stores the *raw* spec frame count per
        ``.spec_codes`` path, and the ``max_frames`` clamp is applied on read.
        That makes it independent of everything the dataset filters on
        (``max_frames``, the alignment/stage-direction blocklists, the speaker
        map) -- a dataset that drops 456 clips and one that keeps them both hit
        the same cache instead of invalidating each other on every start.  A
        path that is missing from the cache is loaded and added, so extending a
        filelist only pays for the new items.

        It is *not* invalidated by re-running ``2_preprocess.py`` over the same
        paths with different content: that would also leave ``phone_ids``
        describing different audio, i.e. it is a corpus replacement, not a
        cache problem.
        """
        if self._lengths is not None:
            return self._lengths
        cache_path = f"{self.filelist}.lengths.npz"
        raw: Dict[str, int] = {}
        if cache:
            try:
                with np.load(cache_path, allow_pickle=False) as z:
                    paths = z["paths"].tolist()
                    values = z["frames"].tolist()
                raw = {str(p): int(v) for p, v in zip(paths, values)}
            except (OSError, ValueError, KeyError):
                raw = {}
        need = [it["spec_codes"] for it in self.items
                if str(it["spec_codes"]) not in raw]
        for path in need:
            raw[str(path)] = int(torch.load(
                path, weights_only=True, mmap=True).shape[-1])
        lengths = [raw[str(it["spec_codes"])] for it in self.items]
        if self.max_frames is not None:
            lengths = [min(n, int(self.max_frames)) for n in lengths]
        self._lengths = lengths
        if cache and need:
            # Rewrite only when something was actually missing, so a normal
            # start is one np.load and no write.
            try:
                np.savez(cache_path,
                         paths=np.asarray([str(it["spec_codes"])
                                           for it in self.items]),
                         frames=np.asarray([raw[str(it["spec_codes"])]
                                            for it in self.items],
                                           dtype=np.int32))
            except OSError as exc:      # read-only corpus dir: just don't cache
                print(f"Length cache: could not write {cache_path} ({exc}); "
                      "lengths will be recomputed on the next start.")
        return lengths

    def _read_cond(self, phone_ids_path: str,
                   n_phone: int) -> "tuple[Optional[Dict[str, object]], bool]":
        """Load the sibling ``.cond`` for one item, plus its staleness.

        Counts the two "not usable" cases so a mismatched corpus prints a
        diagnosis instead of silently training without hints (or without
        word-aligned cropping).  A *stale* file (``n_phone`` differs from this
        phone sequence, i.e. the phonemizer changed since it was built) is
        still returned: its clip-level channels stay valid, only the per-word
        fields and the word-boundary crop must be dropped.
        """
        path = cond_path_for(phone_ids_path)
        if not os.path.exists(path):
            self.n_cond_missing += 1
            return None, False
        try:
            cond = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            self.n_cond_stale += 1
            return None, False
        stale = int(cond.get("n_phone", -1)) != int(n_phone)
        if stale:
            self.n_cond_stale += 1
        return cond, stale

    def _load_hints(self, cond: Optional[Dict[str, object]], n_phone: int,
                    sid: Optional[int] = None,
                    stale: bool = False) -> Dict[str, object]:
        """Bucketed hints for one item, plus per-channel dropout.

        ``cond`` is the ``.cond`` from :meth:`_read_cond` (``None`` = missing or
        unreadable -> every level ``0``), so one item loads the file once even
        when it is needed for both the hints and the word-boundary crop.  With
        ``stale=True`` the per-word fields are dropped but the clip-level
        channels are kept (they do not depend on the phonemizer).

        ``sid`` is the item's *raw* (pre-``sid_map``) speaker id, used only by
        the relative-register path: the speaker's mean offset is looked up from
        ``hints.json`` and subtracted before bucketing, so the level means "this
        utterance is N semitones above/below this speaker's usual level" rather
        than "this speaker is a high voice" (HINTS_PLAN.md §8.5).

        Dropout is applied here, and only for training items: each channel is
        zeroed independently with its ``hint_dropout`` probability, and
        ``hint_null_all`` zeroes all of them.  Level 0 is a real trained state,
        which is what lets the model run without hints at inference.
        """
        out: Dict[str, object] = {
            "hint_int": 0, "hint_reg": 0,
            "hint_pitch": None, "hint_slot": None,
            # Hints V2 pitch anchors (repeat-over-phones representation).
            "hint_pitch_val": None, "hint_pitch_pos": None,
            "hint_pitch_n": None, "hint_pitch_state": None,
            "hint_phone_frac": None,
            # Hints V2.1 per-word BERT state, repeated over the word's phones.
            "hint_word": None, "hint_word_valid": None,
        }
        if cond is None:
            return out

        if "intensity" in cond:
            out["hint_int"] = bucket_levels(
                float(cond["intensity"].reshape(-1)[0]), self.intensity_edges)
        if "register_st" in cond:
            value = float(cond["register_st"].reshape(-1)[0])
            if np.isfinite(value):     # a fully unvoiced clip has no register
                out["hint_reg"] = bucket_levels(
                    register_rel_st(value, sid, self.register_speaker_means),
                    self.register_edges)
        # Word-level span: only usable if it describes *this* phone sequence.
        # A stale file keeps its clip-level channels (intensity/register) --
        # those are a whole-clip property and do not depend on the phonemizer.
        if "phone_word_idx" in cond and not stale:
            word = cond["phone_word_idx"].long().reshape(-1)
            if word.numel() == n_phone:
                pitch_w = word_pitch_levels(
                    cond["pitch_word_st"].float(),
                    cond["pitch_word_voiced"].bool(), self.pitch_edges)
                slot_w = slot_bucket(cond["word_slot_f"].float(),
                                     self.n_slot_buckets)
                # ``word_slot_f`` is start-to-start, and the aligner gives
                # adjacent words the same start for ~5.5 % of words (mostly
                # "to/a/the").  The first of such a pair then carries a
                # 1-frame "slot", which is a boundary artefact, not a 40 ms
                # word: report it as *absent* (level 0, a trained state) rather
                # than as the fastest timing bucket.  The crop groups equal
                # starts separately (``_word_spans``); this only affects the
                # per-phone hint channel.
                starts = cond.get("word_start_f")
                if (starts is not None and slot_w.ndim == 1
                        and starts.numel() == slot_w.size and slot_w.size > 1):
                    same = (starts.long().reshape(-1).numpy()[1:]
                            == starts.long().reshape(-1).numpy()[:-1])
                    slot_w = slot_w.copy()
                    slot_w[:-1][same] = 0
                idx = word.numpy()
                out["hint_pitch"] = torch.as_tensor(pitch_w[idx].astype(np.int64))
                out["hint_slot"] = torch.as_tensor(slot_w[idx].astype(np.int64))
                self._load_pitch_anchors(cond, idx, out)
                self._load_word_bert(cond, idx, out)

        if self.is_train:
            self._dropout_hints(out)
        return out

    def _load_pitch_anchors(self, cond, idx, out) -> None:
        """Expand a word's pitch anchors over its phones (Hints V2).

        Every phone of a word receives the word's whole anchor vector
        (``(n_phone, A)``), plus its own normalised position inside the word, so
        the encoder sees both the contour and where in it the phone sits.  One
        scalar per phone cannot carry it: measured on Genshin, projecting 3
        anchors to one value per phone is no better than projecting 2, because a
        320-440 ms word averages ~2.5 phones.

        A v1 ``.cond`` (no anchor keys) becomes one anchor at ``pos=0.5`` with
        the old word mean, which is exactly the v1 behaviour.
        """
        n_word = int(idx.max()) + 1 if idx.size else 0
        voiced = cond.get("pitch_word_voiced")
        voiced_w = (voiced.bool().reshape(-1).numpy() if voiced is not None
                    else np.zeros(n_word, dtype=bool))
        if "pitch_anchors_st" in cond and "pitch_anchor_n" in cond:
            a_st = cond["pitch_anchors_st"].float().numpy()
            a_pos = cond["pitch_anchors_pos"].float().numpy()
            a_n = cond["pitch_anchor_n"].long().reshape(-1).numpy()
        else:
            # v1 fallback: the word mean is the single anchor.
            v1 = cond["pitch_word_st"].float().reshape(-1).numpy()
            v1 = np.where(np.isfinite(v1), v1, 0.0)
            a_st = v1[:, None].astype(np.float32)
            a_pos = np.full_like(a_st, 0.5)
            a_n = voiced_w.astype(np.int64)
        if (a_st.ndim != 2 or a_st.shape[0] < n_word or a_pos.shape != a_st.shape
                or a_n.shape[0] < n_word):
            # A word field that does not describe this clip is dropped, not
            # guessed (the n_phone check above already catches most of these).
            return

        counts = np.bincount(idx, minlength=n_word)
        offsets = np.concatenate(([0], np.cumsum(counts)))[:-1]
        rank = np.arange(idx.size) - offsets[idx]
        frac = (rank + 0.5) / np.maximum(counts[idx], 1)
        # state: 0 absent (dropped later), 1 voiced, 2 unvoiced.  Unvoiced is a
        # *trained* state, distinct from "no hint provided" (see v1's unvoiced
        # level; the whole point of the dropout design).
        state = np.where(voiced_w[idx], 1, 2).astype(np.int64)
        out["hint_pitch_val"] = torch.from_numpy(a_st[idx].astype(np.float32))
        out["hint_pitch_pos"] = torch.from_numpy(a_pos[idx].astype(np.float32))
        out["hint_pitch_n"] = torch.from_numpy(a_n[idx].astype(np.int64))
        out["hint_pitch_state"] = torch.from_numpy(state)
        out["hint_phone_frac"] = torch.from_numpy(frac.astype(np.float32))

    def _load_word_bert(self, cond, idx, out) -> None:
        """Expand each word's BERT state over its phones (Hints V2.1).

        The word vector is *repeated* over the word's phones, exactly like the
        pitch anchors: the encoder input is per phone, and a word has ~2.5
        phones, so there is no meaningful per-phone projection to compute here.
        Absence is a per-phone mask (``hint_word_valid``), which the model maps
        to its own trained ``word_null`` vector -- an all-zero BERT output is a
        legitimate vector and must not be overloaded to mean "no hint".

        A ``.cond`` without ``word_bert`` (pre-V2.1, or a clip whose word fields
        were dropped) simply leaves both keys ``None``.
        """
        wb = cond.get("word_bert")
        if wb is None:
            return
        vec = wb.float()
        need = (int(idx.max()) + 1) if idx.size else 0
        if vec.dim() != 2 or vec.shape[0] < need:
            return
        out["hint_word"] = vec[idx]
        out["hint_word_valid"] = torch.ones(int(idx.size), dtype=torch.bool)

    def _dropout_hints(self, out: Dict[str, object]) -> None:
        """Zero hint channels in-place (training only; see ``_load_hints``)."""
        if self.hint_null_all > 0.0 and self.rng.random() < self.hint_null_all:
            off = set(HINT_CHANNELS)
        else:
            off = {c for c in HINT_CHANNELS
                   if self.hint_dropout.get(c, 0.0) > 0.0
                   and self.rng.random() < self.hint_dropout[c]}
        if "intensity" in off:
            out["hint_int"] = 0
        if "register" in off:
            out["hint_reg"] = 0
        if "pitch" in off:
            if out["hint_pitch"] is not None:
                out["hint_pitch"] = torch.zeros_like(out["hint_pitch"])
            # Hints V2: state 0 is the model's "absent" path; the value/position
            # tensors are then ignored (they are left as-is, like a level id
            # that a table would still index but never reach).
            if out["hint_pitch_state"] is not None:
                out["hint_pitch_state"] = torch.zeros_like(
                    out["hint_pitch_state"])
        if "timing" in off and out["hint_slot"] is not None:
            out["hint_slot"] = torch.zeros_like(out["hint_slot"])
        if "word_sem" in off and out["hint_word_valid"] is not None:
            # Keep the vectors; the mask is what the model reads, so switching
            # the channel back on is free.
            out["hint_word_valid"] = torch.zeros_like(out["hint_word_valid"])

    def _word_spans(self, cond: Optional[Dict[str, object]], n_frames: int,
                    n_phone: int, stale: bool = False):
        """Word boundaries for the crop, or ``None`` when the file cannot say.

        Returns ``(starts, ends, word_bounds, phone_word_idx)`` in code frames /
        word indices / phone indices.  ``ends[g]`` is the start of the next
        *group*, the last group ends at the clip end.

        The aligner occasionally gives two adjacent words the **same** start
        frame (measured: ~20 % of Genshin clips have at least one such pair).
        ``word_slot_f`` is start-to-start, so the first of the pair gets a
        zero-length span.  Treating a run of equal starts as one segment is
        what keeps the audio/phone partition contiguous: otherwise the crop
        could start at word ``i`` and drop its phones because its "end" equals
        its own start, while the audio for that word is still in the window.
        Grouping changes *crop boundaries only* -- the per-word hint channels
        still see the original individual words.

        The fields are written by ``hints/build.py --words``; a pre-words
        ``.cond`` has no ``word_start_f`` and the caller falls back to the old
        crop.
        """
        if cond is None or stale:
            return None
        need = ("phone_word_idx", "word_start_f", "word_slot_f")
        if any(k not in cond for k in need):
            return None
        pwi = cond["phone_word_idx"].long().reshape(-1).numpy()
        starts = cond["word_start_f"].long().reshape(-1).numpy()
        slots = cond["word_slot_f"].long().reshape(-1).numpy()
        n_word = int(starts.size)
        if (n_word == 0 or slots.size != n_word or pwi.size != n_phone
                or pwi.min() < 0 or pwi.max() >= n_word
                or np.any(np.diff(pwi) < 0) or np.any(np.diff(starts) < 0)):
            return None
        starts = np.clip(starts, 0, n_frames)
        # Collapse runs of equal starts into one crop segment.  ``bounds[g]`` is
        # the first word index of group g; ``bounds[g + 1]`` is the exclusive
        # end, so the phone span of a group is a contiguous slice of pwi.
        uniq, gid = np.unique(starts, return_inverse=True)
        bounds = np.searchsorted(gid, np.arange(uniq.size + 1), side="left")
        bounds[-1] = n_word
        ends = np.empty(uniq.size, dtype=np.int64)
        if uniq.size > 1:
            ends[:-1] = uniq[1:]
        ends[-1] = int(n_frames)
        ends = np.clip(np.maximum(ends, uniq), 0, n_frames)
        del slots  # only checked for length; the crop derives ends from starts
        return uniq, ends, bounds, pwi

    def _crop_frames(self, spec: torch.Tensor, pitch: torch.Tensor,
                     word_spans=None):
        """Cut the item to ``max_frames`` and report whether its end survived.

        Returns ``(spec, pitch, start, complete, phone_span)``.  ``complete`` is
        ``True`` when the kept window still contains the last frame of the clip,
        i.e. the segment is the true end of the utterance rather than a mid-clip
        random crop.  The transformer only places an end-of-utterance target on
        ``complete`` items (see ``CodeTransformer._append_eos``): an EOS in the
        middle of a clip would teach the model to fall silent after
        ``max_frames``.

        With ``word_crop`` and word timings available, the window is snapped
        **inwards to whole words**: it starts at a word onset and ends where the
        next (dropped) word starts, so the returned ``phone_span`` contains
        exactly the words that are in the audio.  Without this, a clip longer
        than ``max_frames`` was trained on a random audio window paired with the
        *whole* transcript -- i.e. most of its text had no corresponding audio,
        which teaches the model to skip utterance-initial words.  Losing up to
        one word of audio at each edge is deliberate: the alternative (keeping
        partial words) reintroduces text with no audio.  The tail of a clip is
        kept whenever the last word fits, so ``complete`` stays meaningful.

        ``phone_span`` is ``(p0, p1)`` into the original phone sequence, or
        ``None`` when the item was not cropped / had no usable word timings.
        """
        n = min(spec.shape[1], pitch.shape[1])
        spec, pitch = spec[:, :n], pitch[:, :n]
        if self.max_frames is None or n <= self.max_frames:
            return spec, pitch, 0, True, None

        if self.is_train:
            # ``random.Random.randint`` is inclusive on both ends, so this can
            # start at ``n - max_frames`` -- the one window that still contains
            # the clip's last frame (and therefore the only ``complete=True``
            # one).  Snapping to a word boundary afterwards can move it slightly
            # forward, but the window still ends at the last word.
            start = self.rng.randint(0, n - self.max_frames)
        else:
            start = 0
        end = start + self.max_frames
        phone_span = None

        if self.word_crop and word_spans is not None:
            starts, ends, bounds, pwi = word_spans
            # First group at/after ``start`` and last group ending at/before
            # ``end`` (searchsorted on the group start/end arrays).
            i0 = int(np.searchsorted(starts, start, side="left"))
            i1 = int(np.searchsorted(ends, end, side="right"))
            while i0 < i1 and ends[i0] <= starts[i0]:
                i0 += 1
            while i1 > i0 and ends[i1 - 1] <= starts[i0]:
                i1 -= 1
            if i1 > i0:
                w0, w1 = int(starts[i0]), int(ends[i1 - 1])
                if w1 > w0:
                    # ``bounds`` maps group -> first word index, so the phone
                    # span covers every word of every kept group (equal-start
                    # runs stay together).
                    p0 = int(np.searchsorted(pwi, int(bounds[i0]), side="left"))
                    p1 = int(np.searchsorted(pwi, int(bounds[i1]), side="left"))
                    if p1 > p0:
                        start, end = w0, w1
                        phone_span = (p0, p1)
            if phone_span is None:
                self.n_word_crop_fallback += 1

        spec = spec[:, start:end]
        pitch = pitch[:, start:end]
        complete = end >= n
        return spec, pitch, start, complete, phone_span

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        phone_ids = torch.load(item["phone_ids"], weights_only=True).long().reshape(-1)
        spec = torch.load(item["spec_codes"], weights_only=True).long()
        pitch = torch.load(item["pitch_codes"], weights_only=True).long()
        spec = spec.reshape(spec.shape[0], -1)
        pitch = pitch.reshape(pitch.shape[0], -1)

        n_phone = int(phone_ids.numel())
        # One ``.cond`` read serves both the conditioning hints and the
        # word-boundary crop.  ``None`` = missing/stale, i.e. neither is usable.
        cond, cond_stale = self._read_cond(item["phone_ids"], n_phone)
        word_spans = (self._word_spans(cond, min(spec.shape[1], pitch.shape[1]),
                                       n_phone, stale=cond_stale)
                      if self.word_crop else None)
        spec, pitch, _start, complete, phone_span = self._crop_frames(
            spec, pitch, word_spans)

        p0, p1 = phone_span if phone_span is not None else (0, n_phone)
        phone_ids = phone_ids[p0:p1]
        if (self.max_phone_length is not None
                and phone_ids.numel() > self.max_phone_length):
            phone_ids = phone_ids[: self.max_phone_length]

        hint: Dict[str, object] = {}
        if self.hints:
            hint = self._load_hints(
                cond, n_phone, sid=item.get("sid_raw", item["sid"]),
                stale=cond_stale)
            # Per-phone hint arrays must be cropped in lockstep with the phones,
            # otherwise they would describe a different text than the model sees.
            for key in ("hint_pitch", "hint_slot", "hint_pitch_val",
                        "hint_pitch_pos", "hint_pitch_n", "hint_pitch_state",
                        "hint_phone_frac", "hint_word", "hint_word_valid"):
                if hint.get(key) is not None:
                    hint[key] = hint[key][p0:p1][: phone_ids.numel()]

        fim_flag, prefix_len, mid_len = self._sample_fim(spec.shape[1])
        return {
            "phone_ids": phone_ids,
            "spec_codes": spec,
            "pitch_codes": pitch,
            "sid": item["sid"],
            "complete": complete,
            "fim": fim_flag,
            "fim_prefix_len": prefix_len,
            "fim_mid_len": mid_len,
            **hint,
        }

    def _sample_fim(self, n_frames: int):
        """Draw a (prefix, middle) split with probability ``fim_rate``.

        Returns ``(fim, prefix_len, mid_len)``; ``(False, 0, 0)`` for a plain AR
        item (and for every validation item: ``is_train`` gates it so the val
        split stays a fixed AR yardstick).  ``mid_len`` is at least
        ``fim_min_frames`` and the span is centred by drawing ``prefix_len``
        uniformly in ``[0, n - mid_len]`` (so the suffix can be empty only when
        the prefix is long, never a degenerate all-middle clip unless the item is
        very short).
        """
        if (not self.is_train or not self.fim or self.fim_rate <= 0.0
                or n_frames <= self.fim_min_frames):
            return False, 0, 0
        if self.rng.random() >= self.fim_rate:
            return False, 0, 0
        min_mid = self.fim_min_frames
        max_mid = max(min_mid, min(n_frames - 1,
                                   int(round(self.fim_mid_frac * n_frames))))
        mid_len = self.rng.randint(min_mid, max_mid)
        mid_len = min(mid_len, n_frames - 1)  # leave >= 1 frame for prefix+suffix
        prefix_len = self.rng.randint(0, n_frames - mid_len)
        return True, prefix_len, mid_len


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups similarly long clips, so batches stay narrow.

    Why: the collator pads every batch to its longest item, and stage-2 clips
    are very uneven -- mean 124 code frames, median 99, p90 234, max 512 on
    ``data/genshin_plus_extras``.  A uniformly shuffled batch of 16 therefore
    has a *mean* max of ~330 frames, i.e. **60 % of every forward, backward and
    cross-entropy is computed on padding** (the loss masks it, but the decoder
    still runs over it).  Measured on this repo's real step (fwd+bwd -> clip ->
    AdamW -> Muon):

    =========================  ========  =========  =========
    ordering                   mean T    step       items/h
    =========================  ========  =========  =========
    ``shuffle=True``, bs=16    ~316      358 ms     161 k
    bucketed, pool=8, bs=16    ~142      146 ms     396 k
    bucketed, pool=8, bs=24    ~151      186 ms     464 k
    =========================  ========  =========  =========

    The Muon step is batch-independent (~49 ms), which is why the larger batch
    gains more than the FLOP ratio suggests.  Nothing about *what* is trained
    changes: same items, same crops, same FIM draws, same optimiser; only the
    order (so ``max_steps`` / the LR schedule keep their meaning).

    How: shuffle the index list, cut it into pools of ``pool_batches`` batches,
    sort each pool by length and emit ``batch_size`` chunks (shuffled inside the
    batch).  A pool is a sliver of the corpus (128-192 items at the shipped
    values), so every training epoch still mixes all lengths; it only means a
    given batch is length-homogeneous.  Pools are re-drawn every epoch from a
    seed + epoch RNG, so epoch-to-epoch ordering does not repeat.

    ``drop_last`` drops the partial pool tail (a few dozen items), exactly like
    ``DataLoader(drop_last=True)`` does for a partial batch.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        pool_batches: int = 8,
        drop_last: bool = True,
        seed: int = 42,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if pool_batches < 1:
            raise ValueError(f"pool_batches must be >= 1, got {pool_batches}")
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.pool_batches = int(pool_batches)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.n_items = len(lengths)
        pool = self.batch_size * self.pool_batches
        full_pools, rem = divmod(self.n_items, pool)
        n = full_pools * self.pool_batches
        if not self.drop_last:
            n += -(-rem // self.batch_size)   # ceil: the tail is a short batch
        if n < 1:
            raise ValueError(
                f"LengthBucketBatchSampler produced 0 batches from "
                f"{self.n_items} items (batch_size={self.batch_size}, "
                f"pool_batches={self.pool_batches}, drop_last={self.drop_last})")
        self.n_batches = n
        self.n_full = full_pools * pool

    def set_epoch(self, epoch: int) -> None:
        """Pick a different (reproducible) ordering for the next epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(self.n_items))
        rng.shuffle(idx)
        get = self.lengths.__getitem__
        step = self.batch_size * self.pool_batches
        for p in range(0, self.n_full, step):
            chunk = idx[p:p + step]
            # Long clips together, short clips together: the batch's padded
            # width is then whatever its own group needs, not the corpus max.
            chunk.sort(key=get)
            for b in range(0, len(chunk) - self.batch_size + 1, self.batch_size):
                batch = chunk[b:b + self.batch_size]
                rng.shuffle(batch)
                yield batch
        if not self.drop_last:
            # Partial last pool: same treatment, then a possibly short batch.
            chunk = sorted(idx[self.n_full:], key=get)
            for b in range(0, len(chunk), self.batch_size):
                batch = chunk[b:b + self.batch_size]
                rng.shuffle(batch)
                yield batch


class CodeCollator:
    def __init__(self, config=None, phone_pad_id: int = 0, code_pad_id: int = 0):
        # Hints V2: the anchor width comes from the config, so a v1 `.cond`
        # (one anchor) and a v2 one (A anchors) can share a batch.
        self.n_pitch_anchors = 3
        if config is not None:
            tcfg = getattr(config, "transformer", None)
            if tcfg is not None:
                phone_pad_id = int(getattr(tcfg, "phone_pad_id", phone_pad_id))
                self.n_pitch_anchors = int(
                    tcfg.get("hint_pitch_anchors", 3) or 3)
        self.phone_pad_id = phone_pad_id
        self.code_pad_id = code_pad_id

    def __call__(self, batch: List[dict]) -> dict:
        max_phone = max(int(x["phone_ids"].numel()) for x in batch)
        max_frames = max(int(x["spec_codes"].shape[1]) for x in batch)

        phone_ids = torch.full(
            (len(batch), max_phone), self.phone_pad_id, dtype=torch.long
        )
        phone_lengths = torch.zeros(len(batch), dtype=torch.long)

        q_spec = batch[0]["spec_codes"].shape[0]
        q_pitch = batch[0]["pitch_codes"].shape[0]
        spec_codes = torch.full(
            (len(batch), q_spec, max_frames), self.code_pad_id, dtype=torch.long
        )
        pitch_codes = torch.full(
            (len(batch), q_pitch, max_frames), self.code_pad_id, dtype=torch.long
        )
        code_lengths = torch.zeros(len(batch), dtype=torch.long)
        sids = torch.zeros(len(batch), dtype=torch.long)
        complete = torch.zeros(len(batch), dtype=torch.bool)
        # FIM span selectors (no-op unless the model has transformer.fim).
        fim = torch.zeros(len(batch), dtype=torch.bool)
        fim_prefix_len = torch.zeros(len(batch), dtype=torch.long)
        fim_mid_len = torch.zeros(len(batch), dtype=torch.long)
        # Conditioning hints (no-op unless the dataset emits them, i.e. unless
        # transformer.hints is on).  Level 0 is the trained "not provided"
        # state, so padding with zeros is correct.
        has_hints = "hint_int" in batch[0]
        hint_int = hint_reg = None
        hint_pitch = hint_slot = None
        has_anchors = has_hints and batch[0].get("hint_pitch_val") is not None
        hint_pitch_val = hint_pitch_pos = hint_pitch_n = None
        hint_pitch_state = hint_phone_frac = None
        # A batch can mix clips with and without `word_bert` (a clip whose word
        # fields were dropped), so presence is per batch, not per first item.
        dims_word = [int(x["hint_word"].shape[1]) for x in batch
                     if x.get("hint_word") is not None] if has_hints else []
        has_word = bool(dims_word)
        word_dim = max(dims_word) if dims_word else 0
        hint_word = hint_word_valid = None
        if has_hints:
            hint_int = torch.zeros(len(batch), dtype=torch.long)
            hint_reg = torch.zeros(len(batch), dtype=torch.long)
            hint_pitch = torch.zeros(len(batch), max_phone, dtype=torch.long)
            hint_slot = torch.zeros(len(batch), max_phone, dtype=torch.long)
        if has_anchors:
            A = self.n_pitch_anchors
            hint_pitch_val = torch.zeros(len(batch), max_phone, A,
                                         dtype=torch.float32)
            hint_pitch_pos = torch.zeros(len(batch), max_phone, A,
                                         dtype=torch.float32)
            hint_pitch_n = torch.zeros(len(batch), max_phone, dtype=torch.long)
            hint_pitch_state = torch.zeros(len(batch), max_phone,
                                           dtype=torch.long)
            hint_phone_frac = torch.zeros(len(batch), max_phone,
                                          dtype=torch.float32)
        if has_word:
            hint_word = torch.zeros(len(batch), max_phone, word_dim,
                                    dtype=torch.float32)
            hint_word_valid = torch.zeros(len(batch), max_phone,
                                          dtype=torch.bool)

        for i, x in enumerate(batch):
            p = x["phone_ids"]
            phone_ids[i, : p.numel()] = p
            phone_lengths[i] = p.numel()

            spec = x["spec_codes"]
            pitch = x["pitch_codes"]
            n = min(spec.shape[1], pitch.shape[1], max_frames)
            spec_codes[i, :, :n] = spec[:, :n]
            pitch_codes[i, :, :n] = pitch[:, :n]
            code_lengths[i] = n
            sids[i] = x["sid"]
            complete[i] = bool(x.get("complete", True))
            fim[i] = bool(x.get("fim", False))
            fim_prefix_len[i] = int(x.get("fim_prefix_len", 0))
            fim_mid_len[i] = int(x.get("fim_mid_len", 0))
            if has_hints:
                hint_int[i] = int(x.get("hint_int", 0))
                hint_reg[i] = int(x.get("hint_reg", 0))
                pitch_hint = x.get("hint_pitch")
                if pitch_hint is not None and pitch_hint.numel() > 0:
                    n_h = min(int(pitch_hint.numel()), max_phone)
                    hint_pitch[i, :n_h] = pitch_hint[:n_h].long()
                    hint_slot[i, :n_h] = x["hint_slot"][:n_h].long()
                if has_anchors:
                    val = x.get("hint_pitch_val")
                    if val is not None and val.numel() > 0:
                        n_h = min(int(val.shape[0]), max_phone)
                        a_h = min(int(val.shape[1]), hint_pitch_val.shape[2])
                        hint_pitch_val[i, :n_h, :a_h] = val[:n_h, :a_h].float()
                        hint_pitch_pos[i, :n_h, :a_h] = \
                            x["hint_pitch_pos"][:n_h, :a_h].float()
                        hint_pitch_n[i, :n_h] = x["hint_pitch_n"][:n_h].long()
                        hint_pitch_state[i, :n_h] = \
                            x["hint_pitch_state"][:n_h].long()
                        hint_phone_frac[i, :n_h] = \
                            x["hint_phone_frac"][:n_h].float()
                if has_word:
                    w = x.get("hint_word")
                    if w is not None and w.numel():
                        n_h = min(int(w.shape[0]), max_phone)
                        d_h = min(int(w.shape[1]), hint_word.shape[2])
                        hint_word[i, :n_h, :d_h] = w[:n_h, :d_h].float()
                        hint_word_valid[i, :n_h] = \
                            x["hint_word_valid"][:n_h].bool()

        out = {
            "phone_ids": phone_ids,
            "phone_lengths": phone_lengths,
            "sid": sids,
            "spec_codes": spec_codes,
            "pitch_codes": pitch_codes,
            "code_lengths": code_lengths,
            # True where this segment ends the clip -> the transformer adds an
            # EOS target there (no-op unless transformer.eos_token is set).
            "complete": complete,
            # Fill-in-the-middle: which items to reorder and their span lengths
            # (no-op unless transformer.fim; AR and FIM items can share a batch).
            "fim": fim,
            "fim_prefix_len": fim_prefix_len,
            "fim_mid_len": fim_mid_len,
        }
        if has_hints:
            out.update({
                "hint_int": hint_int,
                "hint_reg": hint_reg,
                "hint_pitch": hint_pitch,
                "hint_slot": hint_slot,
            })
        if has_word:
            out.update({
                "hint_word": hint_word,
                "hint_word_valid": hint_word_valid,
            })
        if has_anchors:
            out.update({
                "hint_pitch_val": hint_pitch_val,
                "hint_pitch_pos": hint_pitch_pos,
                "hint_pitch_n": hint_pitch_n,
                "hint_pitch_state": hint_pitch_state,
                "hint_phone_frac": hint_phone_frac,
            })
        return out
