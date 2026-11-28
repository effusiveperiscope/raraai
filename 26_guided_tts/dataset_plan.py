"""Torch dataset for plan -> renderer training.

One item is a word-aligned window of a clip: the plan (phones, 100 Hz phone
durations, per-code-frame prosody, silence), the stage-1 code targets
(8 spec + 2 pitch books at 25 Hz), and the dense ``.f0`` when the corpus has
performance audio.  Crops snap to word boundaries so the plan and the code
targets can never disagree about which phones are in the window.

Plans are cached (``plans/<corpus>/<clip>.npz``, a few KB) because deriving one
needs the stage-1 ``.spec`` (~0.5 MB/clip); training must not re-read 75 GB of
spectrograms per epoch.  The cache is purely derived -- delete it any time.

    ds = PlanDataset("data/genshin_plus_extras/train_stage2.txt", max_frames=400)
    dl = DataLoader(ds, batch_size=8, collate_fn=collate_plans, num_workers=4)

Build the cache once with ``python scratch/build_plan_cache.py`` (or pass
``write_cache=True`` to fill it lazily on a single-process run).  Pass
``require_cache=True`` after building: items the builder could not plan
(degenerate alignments) are then skipped instead of raising mid-epoch.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from plan import (
    FRAME_FEATURES, Plan, PlanError, ROOT, frame_features,
    intensity_reference, load_intensity_stats, load_plan, load_plan_cache,
    save_plan_cache,
)

#: Sensible default for one 16 GB card: 400 code frames = 16 s.
DEFAULT_MAX_FRAMES = 400


# --------------------------------------------------------------------------- #
# paths / tensors
# --------------------------------------------------------------------------- #

def resolve(path: str | Path, root: Path = ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (root, root.parent / "25_transformer"):
        if (base / p).exists():
            return base / p
    return root / p


def _load(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("spec", "f0", "data", "tensor"):
            if key in obj:
                return obj[key]
        return next(iter(obj.values()))
    return obj


def _codes(path: Path, n_books: int) -> torch.Tensor:
    a = torch.as_tensor(_load(path))
    if a.ndim != 2:
        raise ValueError(f"codes {path} must be 2-D, got {tuple(a.shape)}")
    if a.shape[0] != n_books and a.shape[1] == n_books:
        a = a.T
    if a.shape[0] != n_books:
        raise ValueError(f"codes {path} do not have {n_books} books")
    return a.long()


def f0_path(codes: Path) -> Optional[Path]:
    s = str(codes)
    if "_stage2/" in s:
        s = s.replace("_stage2/", "_stage1/")
    p = Path(s[: -len(".spec_codes")] + ".f0")
    return p if p.exists() else None


def cache_path(cache_dir: Path | str, phone_path: Path | str) -> Path:
    p = Path(phone_path)
    name = p.name[: -len(".phone_ids")] if p.name.endswith(".phone_ids") else p.stem
    return Path(cache_dir) / p.parent.name / (name + ".npz")


def _apply_intensity(plan: Plan, stats: dict | None, speaker: int) -> None:
    """Refresh the slider value from the current stats (cache may be older)."""
    ref, scale = intensity_reference(stats, speaker)
    if ref is None or not np.isfinite(plan.intensity) or not scale:
        plan.intensity_rel = float("nan")
        return
    plan.intensity_rel = (float(plan.intensity) - float(ref)) / float(scale)


# --------------------------------------------------------------------------- #
# word-aligned cropping
# --------------------------------------------------------------------------- #

def crop_plan(plan: Plan, spec_codes: torch.Tensor, pitch_codes: torch.Tensor,
              f0: Optional[torch.Tensor], max_frames: Optional[int],
              train: bool, rng: random.Random) -> tuple[Plan, torch.Tensor,
                                                        torch.Tensor,
                                                        Optional[torch.Tensor]]:
    """Cut a window that starts and ends on word onsets.

    The returned plan is a complete description of the window: its segments
    partition ``[0, e-s)`` and its phone durations still sum to four times
    that.  Targets are sliced to the same frames; ``f0`` (100 Hz) is padded to
    ``4*(e-s)`` with zeros so unvoiced/absent samples are easy to mask.
    """
    T = int(plan.n_frames)
    if not max_frames or T <= max_frames:
        s, e = 0, T
    else:
        start = rng.randrange(0, T - max_frames + 1) if train else 0
        w0 = int(np.searchsorted(plan.word_start, start, side="left"))
        w0 = min(max(w0, 0), plan.n_words - 1)
        s = int(plan.word_start[w0])
        if s >= T:                      # a start beyond the code length: stale
            s, w0 = 0, 0
        w1 = int(np.searchsorted(plan.word_start, s + max_frames, side="right"))
        w1 = min(max(w1, w0 + 1), plan.n_words)
        e = int(plan.word_start[w1]) if w1 < plan.n_words else T
        if e <= s:
            e = T
    return _slice_plan(plan, spec_codes, pitch_codes, f0, s, e)


def _slice_plan(plan: Plan, spec_codes: torch.Tensor, pitch_codes: torch.Tensor,
                f0: Optional[torch.Tensor], s: int, e: int):
    T = int(plan.n_frames)
    s = int(np.clip(s, 0, T))
    e = int(np.clip(e, s + 1, T))
    if s > 0:
        # snap back to the word containing s, so the window never starts
        # mid-word; s == 0 keeps any leading silence in the window
        w0 = int(np.searchsorted(plan.word_start, s, side="right")) - 1
        w0 = min(max(w0, 0), plan.n_words - 1)
        s = int(plan.word_start[w0])
    else:
        w0, s = 0, 0
    w1 = int(np.searchsorted(plan.word_start, e, side="left"))
    w1 = min(max(w1, w0 + 1), plan.n_words)
    e = int(plan.word_start[w1]) if w1 < plan.n_words else T
    if e <= s:
        e = T

    p0 = int(np.searchsorted(plan.phone_word, w0, side="left"))
    p1 = (int(np.searchsorted(plan.phone_word, w1, side="left"))
          if w1 < plan.n_words else plan.n_phones)
    n = e - s
    words = slice(w0, w1)
    starts = plan.word_start[words].astype(np.int32) - s
    grouped = int(np.sum(np.diff(starts) == 0))
    new = Plan(
        phone_ids=plan.phone_ids[p0:p1],
        phone_word=plan.phone_word[p0:p1].astype(np.int32) - w0,
        phone_dur=plan.phone_dur[p0:p1],
        word_start=starts,
        word_dur=plan.word_dur[words],
        pause_after=plan.pause_after[words],
        # s == 0 keeps the clip's leading silence; a word-aligned crop starts
        # at a word, so only the original lead that lies before s survives
        lead_sil=max(0, int(plan.lead_sil) - s),
        pitch_st=plan.pitch_st[4 * s:4 * e],
        voiced=plan.voiced[4 * s:4 * e],
        register_st=plan.register_st,
        word_db=plan.word_db[words],
        intensity=plan.intensity,
        intensity_rel=plan.intensity_rel,
        speaker=plan.speaker,
        n_frames=n,
        has_spec=plan.has_spec,
        grouped_words=grouped,
    )
    new.check()
    spec = spec_codes[:, s:e]
    pitch = pitch_codes[:, s:e]
    f0w = None
    if f0 is not None:
        g = torch.as_tensor(f0).float().reshape(-1)[4 * s:4 * e]
        if g.numel() < 4 * n:
            g = torch.cat([g, torch.zeros(4 * n - g.numel())])
        f0w = g[: 4 * n]
    return new, spec, pitch, f0w


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

class PlanDataset(Dataset):
    """Plan/codes/f0 windows for one filelist (``phone|spec|pitch|sid``)."""

    def __init__(self, filelist: str | Path, *,
                 max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
                 train: bool = True,
                 cache_dir: str | Path | None = None,
                 write_cache: bool = False,
                 require_cache: bool = False,
                 with_spec: bool = True,
                 intensity_stats=None,
                 corpora: Optional[Sequence[str]] = None,
                 limit: int = 0,
                 seed: int = 0,
                 verbose: bool = True):
        self.filelist = resolve(filelist)
        self.max_frames = max_frames
        self.train = train
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "plans"
        self.write_cache = write_cache
        self.require_cache = require_cache
        self.with_spec = with_spec
        self.stats = intensity_stats if intensity_stats is not None \
            else load_intensity_stats()
        self.seed = int(seed)
        self._rng: Optional[random.Random] = None

        self.items: list[dict] = []
        skipped = {"no line": 0, "missing tensor": 0, "no cond/cache": 0,
                   "no cache": 0}
        seen: set[str] = set()
        for line in self.filelist.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                skipped["no line"] += 1
                continue
            phone = resolve(parts[0])
            if str(phone) in seen:
                continue
            seen.add(str(phone))
            if corpora and not any(c in str(phone) for c in corpora):
                continue
            codes = resolve(parts[1])
            pitch = resolve(parts[2])
            if not (phone.exists() and codes.exists() and pitch.exists()):
                skipped["missing tensor"] += 1
                continue
            cache = cache_path(self.cache_dir, phone)
            if not cache.exists():
                if self.require_cache:
                    skipped["no cache"] += 1
                    continue
                if not Path(str(codes)[: -len(".spec_codes")] + ".cond").exists():
                    skipped["no cond/cache"] += 1
                    continue
            self.items.append({
                "phone": phone, "codes": codes, "pitch": pitch,
                "sid": int(parts[3]),
                "corpus": phone.parent.name,
            })
            if limit and len(self.items) >= limit:
                break

        if verbose:
            print(f"PlanDataset: {len(self.items)} items from {self.filelist}"
                  + (f"  skipped {skipped}" if any(skipped.values()) else ""))

    def __len__(self) -> int:
        return len(self.items)

    def _random(self) -> random.Random:
        if self._rng is None:
            self._rng = random.Random(self.seed ^ (int(torch.initial_seed()) & 0x7FFFFFFF))
        return self._rng

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        sid = int(item["sid"])
        phone_ids = _load(item["phone"]).long().reshape(-1).numpy()
        cache = cache_path(self.cache_dir, item["phone"])
        plan = load_plan_cache(cache, phone_ids, speaker=sid)
        if plan is None:
            plan = load_plan(item["phone"], with_spec=self.with_spec,
                             speaker=sid, intensity_stats=self.stats)
            if self.write_cache:
                save_plan_cache(plan, cache)
        _apply_intensity(plan, self.stats, sid)

        spec = _codes(item["codes"], 8)
        pitch = _codes(item["pitch"], 2)
        if spec.shape[1] != plan.n_frames or pitch.shape[1] != plan.n_frames:
            raise PlanError(f"{item['phone']}: codes {tuple(spec.shape)} / "
                            f"{tuple(pitch.shape)} vs plan {plan.n_frames}")
        f0p = f0_path(item["codes"])
        f0 = torch.as_tensor(_load(f0p)).float().reshape(-1) if f0p else None

        plan, spec, pitch, f0 = crop_plan(
            plan, spec, pitch, f0, self.max_frames, self.train, self._random())

        out = {
            "phone_ids": torch.from_numpy(plan.phone_ids.astype(np.int64)),
            "phone_dur": torch.from_numpy(plan.phone_dur.astype(np.int64)),
            "phone_word": torch.from_numpy(plan.phone_word.astype(np.int64)),
            "phone_index": torch.from_numpy(plan.phone_index_100().astype(np.int64)),
            "frame_phone": torch.from_numpy(plan.frame_phone().astype(np.int64)),
            "word_start": torch.from_numpy(plan.word_start.astype(np.int64)),
            "word_dur": torch.from_numpy(plan.word_dur.astype(np.int64)),
            "pause_after": torch.from_numpy(plan.pause_after.astype(np.int64)),
            "lead_sil": torch.tensor(plan.lead_sil, dtype=torch.long),
            "pitch_st": torch.from_numpy(plan.pitch_st.astype(np.float32)),
            "voiced": torch.from_numpy(plan.voiced.astype(bool)),
            "word_db": torch.from_numpy(plan.word_db.astype(np.float32)),
            "frame_feats": torch.from_numpy(frame_features(plan)),
            "spec_codes": spec,
            "pitch_codes": pitch,
            "f0": f0 if f0 is not None else torch.zeros(4 * plan.n_frames),
            "f0_valid": (f0 > 0) if f0 is not None else torch.zeros(4 * plan.n_frames, dtype=torch.bool),
            "n_frames": torch.tensor(plan.n_frames, dtype=torch.long),
            "n_phones": torch.tensor(plan.n_phones, dtype=torch.long),
            "n_words": torch.tensor(plan.n_words, dtype=torch.long),
            "register_st": torch.tensor(plan.register_st, dtype=torch.float32),
            "intensity_rel": torch.tensor(plan.intensity_rel, dtype=torch.float32),
            "intensity": torch.tensor(plan.intensity, dtype=torch.float32),
            "sid": torch.tensor(sid, dtype=torch.long),
            "has_f0": torch.tensor(f0 is not None, dtype=torch.bool),
            "phone": str(item["phone"]),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        return out


# --------------------------------------------------------------------------- #
# collation
# --------------------------------------------------------------------------- #

def collate_plans(batch: Sequence[dict]) -> dict:
    """Pad to the batch maxima; return explicit masks for every axis."""
    B = len(batch)
    N = max(int(x["phone_ids"].numel()) for x in batch)
    W = max(int(x["word_start"].numel()) for x in batch)
    T = max(int(x["n_frames"]) for x in batch)
    P = 4 * T
    F = len(FRAME_FEATURES)

    phone = torch.zeros(B, N, dtype=torch.long)
    phone_dur = torch.zeros(B, N, dtype=torch.long)
    phone_word = torch.zeros(B, N, dtype=torch.long)
    phone_mask = torch.zeros(B, N, dtype=torch.bool)
    phone_index = torch.full((B, P), -1, dtype=torch.long)
    frame_phone = torch.full((B, T), -1, dtype=torch.long)
    word_start = torch.zeros(B, W, dtype=torch.long)
    word_dur = torch.zeros(B, W, dtype=torch.long)
    pause = torch.zeros(B, W, dtype=torch.long)
    word_db = torch.zeros(B, W, dtype=torch.float32)
    word_mask = torch.zeros(B, W, dtype=torch.bool)
    pitch_st = torch.zeros(B, P, dtype=torch.float32)
    voiced = torch.zeros(B, P, dtype=torch.bool)
    f0 = torch.zeros(B, P, dtype=torch.float32)
    f0_valid = torch.zeros(B, P, dtype=torch.bool)
    feats = torch.zeros(B, T, F, dtype=torch.float32)
    spec = torch.zeros(B, 8, T, dtype=torch.long)
    pitch = torch.zeros(B, 2, T, dtype=torch.long)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)

    scalars = {k: torch.zeros(B) for k in
               ("lead_sil", "register_st", "intensity_rel", "intensity",
                "has_f0")}
    sid = torch.zeros(B, dtype=torch.long)

    for b, x in enumerate(batch):
        n = int(x["n_frames"])
        nph = int(x["phone_ids"].numel())
        nw = int(x["word_start"].numel())
        lengths[b] = n
        frame_mask[b, :n] = True
        phone[b, :nph] = x["phone_ids"]
        phone_dur[b, :nph] = x["phone_dur"]
        phone_word[b, :nph] = x["phone_word"]
        phone_mask[b, :nph] = True
        phone_index[b, :4 * n] = x["phone_index"]
        frame_phone[b, :n] = x["frame_phone"]
        word_start[b, :nw] = x["word_start"]
        word_dur[b, :nw] = x["word_dur"]
        pause[b, :nw] = x["pause_after"]
        word_db[b, :nw] = x["word_db"]
        word_mask[b, :nw] = True
        pitch_st[b, :4 * n] = x["pitch_st"]
        voiced[b, :4 * n] = x["voiced"]
        f0[b, :4 * n] = x["f0"]
        f0_valid[b, :4 * n] = x["f0_valid"]
        feats[b, :n] = x["frame_feats"]
        spec[b, :, :n] = x["spec_codes"]
        pitch[b, :, :n] = x["pitch_codes"]
        for k in scalars:
            scalars[k][b] = x[k].float()
        sid[b] = x["sid"]

    return {
        "phone_ids": phone, "phone_dur": phone_dur, "phone_word": phone_word,
        "phone_mask": phone_mask, "phone_index": phone_index,
        "frame_phone": frame_phone,
        "word_start": word_start, "word_dur": word_dur,
        "pause_after": pause, "word_mask": word_mask, "word_db": word_db,
        "pitch_st": pitch_st, "voiced": voiced,
        "f0": f0, "f0_valid": f0_valid,
        "frame_feats": feats, "frame_mask": frame_mask,
        "spec_codes": spec, "pitch_codes": pitch,
        "lengths": lengths, "sid": sid, **scalars,
        "paths": [x["phone"] for x in batch],
        "index": torch.tensor([int(x["index"]) for x in batch]),
    }


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/genshin_plus_extras/val_stage2.txt")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--no-spec", action="store_true")
    args = ap.parse_args()
    ds = PlanDataset(args.filelist, max_frames=args.max_frames,
                     train=False, with_spec=not args.no_spec,
                     limit=args.limit)
    dl = DataLoader(ds, batch_size=min(4, len(ds)), collate_fn=collate_plans)
    for b in dl:
        print({k: tuple(v.shape) if torch.is_tensor(v) else v
               for k, v in b.items()})
        print("frames", b["lengths"].tolist(),
              "phones", b["phone_mask"].sum(1).tolist(),
              "words", b["word_mask"].sum(1).tolist(),
              "f0 voiced", b["f0_valid"].sum(1).tolist())
        break
