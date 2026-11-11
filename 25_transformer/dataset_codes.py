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

Conditioning hints (``transformer.hints``)
------------------------------------------
``hints/build.py`` writes a sibling ``<base>.cond`` next to each
``<base>.phone_ids`` holding *continuous* values (intensity 1-9, register in
semitones, a 384-d sentence vector, and -- once the alignment pass has run --
per-word pitch/timing).  With ``hints=True`` this dataset reads that file and
emits level ids, so the *bucket count* stays a config change and never needs the
corpus rebuilt:

    hint_int      (B,)      long   intensity level,      0 = absent
    hint_reg      (B,)      long   register level,       0 = absent
    hint_sem      (B, 384)  float  sentence embedding,   zeros where absent
    hint_sem_valid(B,)      bool   False -> the model uses its own null vector
    hint_pitch    (B, N)    long   per-phone word-pitch level
                                   (0 absent, 1..P voiced bins, P+1 unvoiced)
    hint_slot     (B, N)    long   per-phone word-duration bucket (0 absent)

The per-phone arrays are gathered from the word-level fields via
``phone_word_idx``, and cropped in lockstep with ``max_phone_length``.  Edges for
intensity/register come from ``hints.json`` (equal-frequency quantiles fitted by
``hints/build.py``); word pitch and word slots use *fixed* cuts
(``hints.levels.DEFAULT_PITCH_EDGES`` / ``DEFAULT_SLOT_BUCKETS``) so a level id
means the same thing in every corpus and in an inference UI.

Dropout is what keeps this from becoming an oracle/voice-conversion renderer:
``hint_dropout`` zeroes each channel independently per item and
``hint_null_all`` zeroes *all* of them, both only on training items.  Level 0 is
therefore a genuinely trained "no hint" state, not an untrained row.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import speaker_map as speaker_map_lib
from hints.levels import (DEFAULT_PITCH_EDGES, DEFAULT_SLOT_BUCKETS,
                          bucket_levels, load_levels, slot_bucket,
                          word_pitch_levels)

#: Hint channels that ``hint_dropout`` can switch off independently.
HINT_CHANNELS = ("intensity", "register", "pitch", "semantic", "timing")

#: ``.cond`` suffix, matching ``hints.build`` (sibling of ``<base>.phone_ids``).
COND_SUFFIX = ".cond"


def cond_path_for(phone_ids_path: str) -> str:
    """``.../clip.wav.phone_ids`` -> ``.../clip.wav.cond``."""
    return phone_ids_path[: -len(".phone_ids")] + COND_SUFFIX


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
    ):
        self.is_train = is_train
        self.max_frames = max_frames
        self.max_phone_length = max_phone_length
        self.rng = random.Random(random_seed)
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
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def speaker_ids(self) -> List[int]:
        return [it["sid"] for it in self.items]

    def _load_hints(self, phone_ids_path: str, n_phone: int) -> Dict[str, object]:
        """Bucketed hints for one item, plus per-channel dropout.

        A missing or stale ``<base>.cond`` degrades to "no hints" (every level
        ``0``) instead of raising -- one un-built clip must not kill a run -- but
        the ``n_cond_missing`` / ``n_cond_stale`` counters make it visible
        (``2_training.py`` prints them after the first epoch).

        Dropout is applied here, and only for training items: each channel is
        zeroed independently with its ``hint_dropout`` probability, and
        ``hint_null_all`` zeroes all of them.  Level 0 is a real trained state,
        which is what lets the model run without hints at inference.
        """
        out: Dict[str, object] = {
            "hint_int": 0, "hint_reg": 0,
            "hint_sem": None, "hint_sem_valid": False,
            "hint_pitch": None, "hint_slot": None,
        }
        path = cond_path_for(phone_ids_path)
        if not os.path.exists(path):
            self.n_cond_missing += 1
            return out
        try:
            cond = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            self.n_cond_stale += 1
            return out

        if "intensity" in cond:
            out["hint_int"] = bucket_levels(
                float(cond["intensity"].reshape(-1)[0]), self.intensity_edges)
        if "register_st" in cond:
            value = float(cond["register_st"].reshape(-1)[0])
            if np.isfinite(value):     # a fully unvoiced clip has no register
                out["hint_reg"] = bucket_levels(value, self.register_edges)
        if "semantic" in cond:
            vec = cond["semantic"].float().reshape(-1)
            if torch.isfinite(vec).all() and float(vec.abs().sum()) > 0:
                out["hint_sem"] = vec
                out["hint_sem_valid"] = True

        # Word-level span: only usable if it describes *this* phone sequence.
        if "phone_word_idx" in cond:
            if int(cond.get("n_phone", -1)) != int(n_phone):
                self.n_cond_stale += 1
            else:
                word = cond["phone_word_idx"].long().reshape(-1)
                if word.numel() == n_phone:
                    pitch_w = word_pitch_levels(
                        cond["pitch_word_st"].float(),
                        cond["pitch_word_voiced"].bool(), self.pitch_edges)
                    slot_w = slot_bucket(cond["word_slot_f"].float(),
                                         self.n_slot_buckets)
                    idx = word.numpy()
                    out["hint_pitch"] = torch.as_tensor(pitch_w[idx].astype(np.int64))
                    out["hint_slot"] = torch.as_tensor(slot_w[idx].astype(np.int64))

        if self.is_train:
            self._dropout_hints(out)
        return out

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
        if "semantic" in off:
            out["hint_sem_valid"] = False
        if "pitch" in off and out["hint_pitch"] is not None:
            out["hint_pitch"] = torch.zeros_like(out["hint_pitch"])
        if "timing" in off and out["hint_slot"] is not None:
            out["hint_slot"] = torch.zeros_like(out["hint_slot"])

    def _crop_frames(self, spec: torch.Tensor, pitch: torch.Tensor):
        """Cut the item to ``max_frames`` and report whether its end survived.

        Returns ``(spec, pitch, complete)``.  ``complete`` is ``True`` when the
        kept window still contains the last frame of the clip, i.e. the segment
        is the true end of the utterance rather than a mid-clip random crop.
        The transformer only places an end-of-utterance target on ``complete``
        items (see ``CodeTransformer._append_eos``): an EOS in the middle of a
        clip would teach the model to fall silent after ``max_frames``.
        """
        n = min(spec.shape[1], pitch.shape[1])
        spec, pitch = spec[:, :n], pitch[:, :n]
        complete = True
        if self.max_frames is not None and n > self.max_frames:
            if self.is_train:
                start = self.rng.randint(0, n - self.max_frames)
            else:
                start = 0
            spec = spec[:, start : start + self.max_frames]
            pitch = pitch[:, start : start + self.max_frames]
            complete = start + self.max_frames >= n
        return spec, pitch, complete

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        phone_ids = torch.load(item["phone_ids"], weights_only=True).long().reshape(-1)
        spec = torch.load(item["spec_codes"], weights_only=True).long()
        pitch = torch.load(item["pitch_codes"], weights_only=True).long()
        spec = spec.reshape(spec.shape[0], -1)
        pitch = pitch.reshape(pitch.shape[0], -1)
        spec, pitch, complete = self._crop_frames(spec, pitch)

        n_phone = int(phone_ids.numel())
        if self.max_phone_length is not None and n_phone > self.max_phone_length:
            phone_ids = phone_ids[: self.max_phone_length]

        hint: Dict[str, object] = {}
        if self.hints:
            hint = self._load_hints(item["phone_ids"], n_phone)
            # Per-phone hint arrays must be cropped in lockstep with the phones,
            # otherwise they would describe a different text than the model sees.
            for key in ("hint_pitch", "hint_slot"):
                if hint.get(key) is not None:
                    hint[key] = hint[key][: phone_ids.numel()]

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


class CodeCollator:
    def __init__(self, config=None, phone_pad_id: int = 0, code_pad_id: int = 0):
        if config is not None:
            tcfg = getattr(config, "transformer", None)
            if tcfg is not None:
                phone_pad_id = int(getattr(tcfg, "phone_pad_id", phone_pad_id))
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
        # transformer.hints is on).  Level 0 / hint_sem_valid=False is the
        # trained "not provided" state, so padding with zeros is correct.
        has_hints = "hint_int" in batch[0]
        hint_int = hint_reg = hint_sem = hint_sem_valid = None
        hint_pitch = hint_slot = None
        if has_hints:
            hint_int = torch.zeros(len(batch), dtype=torch.long)
            hint_reg = torch.zeros(len(batch), dtype=torch.long)
            hint_sem_valid = torch.zeros(len(batch), dtype=torch.bool)
            dims = [int(x["hint_sem"].numel()) for x in batch
                    if x.get("hint_sem") is not None]
            hint_sem = torch.zeros(len(batch), dims[0] if dims else 0,
                                   dtype=torch.float32)
            hint_pitch = torch.zeros(len(batch), max_phone, dtype=torch.long)
            hint_slot = torch.zeros(len(batch), max_phone, dtype=torch.long)

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
                hint_sem_valid[i] = bool(x.get("hint_sem_valid", False))
                sem = x.get("hint_sem")
                if sem is not None and hint_sem.size(1) > 0:
                    hint_sem[i, : sem.numel()] = sem.float()
                pitch_hint = x.get("hint_pitch")
                if pitch_hint is not None and pitch_hint.numel() > 0:
                    n_h = min(int(pitch_hint.numel()), max_phone)
                    hint_pitch[i, :n_h] = pitch_hint[:n_h].long()
                    hint_slot[i, :n_h] = x["hint_slot"][:n_h].long()

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
                "hint_sem": hint_sem,
                "hint_sem_valid": hint_sem_valid,
                "hint_pitch": hint_pitch,
                "hint_slot": hint_slot,
            })
        return out
