#!/usr/bin/env python3
"""Build per-clip conditioning tensors (``.cond``) for the stage-2 transformer.

Reads the stage-2 filelists written by ``2_preprocess.py`` and writes a sibling
``<base>.cond`` next to each ``<base>.phone_ids``.  Everything it needs already
exists in the repo:

    <wave_dir>/<base>.wave    48 kHz float32 waveform (1_preprocess.py)
    <wave_dir>/<base>.f0      100 Hz f0 in Hz, 0 = unvoiced (1_preprocess.py)
    filelists_2/*.txt         ``audio_path|sid|transcript`` (sentence vector)

so no source audio is touched and no Qwen / aligner weights are loaded.

Produced fields (all optional, all *continuous* -- the dataset buckets them at
load time so the level counts stay a config change, never a re-run):

    intensity    float32(1)   calibrated + whisper-gated DSP score, 1..9
    register_st  float32(1)   utterance voiced mean f0 in semitones re: 100 Hz
    semantic     float16(384) bge-small-en-v1.5 sentence embedding (CLS, L2)

Not produced here: ``pitch_word`` / ``phone_word_idx`` / ``word_start_f`` /
``word_slot_f``.  Those need forced-alignment word timings; see
``hints/README.md`` for the command and why that is a separate pass.

Usage::

    # spot check on 5 clips (writes <base>.cond next to the tensors)
    python hints/build.py --filelist data/genshin_plus_extras/val_stage2.txt \
        --wave-dir data/genshin_full_stage1 \
        --source-filelist filelists_2/genshin_full_eng_text_linux.txt \
        --limit 5

    # full corpus, 8 workers, bin edges fitted over the union of both splits
    python hints/build.py \
        --filelist data/genshin_plus_extras/train_stage2.txt \
                   data/genshin_plus_extras/val_stage2.txt \
        --wave-dir data/genshin_full_stage1 data/mlp_stage1 \
        --source-filelist filelists_2/genshin_full_eng_text_linux.txt \
                   filelists_2/mlp_linux.txt \
        --workers 8

Clips whose ``.wave``/``.f0`` is missing (e.g. a corpus whose stage-1 tensors
are not on this box) are still written with whatever fields *are* available;
the dataset treats a missing key as "hint not provided".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

COND_SUFFIX = ".cond"
PHONE_SUFFIX = ".phone_ids"
WAVE_SUFFIX = ".wave"
F0_SUFFIX = ".f0"

# ``dsp_features.py`` hard-codes a 16 kHz analysis grid (STFT hop and the Praat
# pitch pass), while the stage-1 ``.wave`` tensors are 48 kHz
# (``data.sampling_rate`` in configs/base.yaml).
WAVE_SR = 48000
INTENSITY_SR = 16000

#: Register reference: 100 Hz = 0 st, so an octave is 12 st.  Chosen because
#: corpus median voiced f0 sits well above it (~200-280 Hz), so stored values
#: straddle zero instead of being all-positive.
F0_REF_HZ = 100.0

DEFAULT_SEMANTIC_MODEL = "BAAI/bge-small-en-v1.5"
#: Lives *inside* the package on purpose: `pretrain/` is dockerignored (and
#: holds multi-GB checkpoints), so a model kept there would neither ship in the
#: image nor be safely `git add`-able next to the code that needs it.
DEFAULT_MODEL = os.path.join(_HERE, "models", "dsp_intensity",
                             "tier1_augmented.pkl")

HINT_KEYS = ("intensity", "register", "semantic")
#: hint name -> key in the ``.cond`` dict
COND_KEYS = {"intensity": "intensity", "register": "register_st",
             "semantic": "semantic"}



# --------------------------------------------------------------------------- #
# filelists
# --------------------------------------------------------------------------- #
def read_stage2_filelist(paths: Sequence[str]) -> List[dict]:
    """Read ``phone_ids|spec_codes|pitch_codes|sid`` lines (deduped by path)."""
    items: List[dict] = []
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "longform" in line:
                    continue
                parts = line.split("|")
                if len(parts) < 4 or parts[0] in seen:
                    continue
                seen.add(parts[0])
                items.append({"phone_ids": parts[0], "sid": parts[3]})
    return items


def read_transcripts(paths: Optional[Sequence[str]]) -> Dict[str, str]:
    """Map ``basename(audio_path) -> transcript`` from ``path|sid|text`` lists."""
    out: Dict[str, str] = {}
    for path in paths or ():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 3:
                    continue
                text = parts[2].strip()
                if text:
                    out[os.path.basename(parts[0])] = text
    return out


def read_source_audio(paths: Optional[Sequence[str]]) -> Dict[str, str]:
    """Map ``basename(audio_path) -> audio_path`` from the source filelists.

    Used as an optional fallback for clips whose stage-1 ``.wave`` is not on
    this box (e.g. corpora that were only ever code-encoded here).  The
    intensity hint is defined on the waveform, so the source file is an
    equivalent input -- ``load_audio_48k`` resamples it.
    """
    out: Dict[str, str] = {}
    for path in paths or ():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if not parts or not parts[0].strip():
                    continue
                out[os.path.basename(parts[0])] = parts[0]
    return out


def clip_base(phone_path: str) -> str:
    """``.../X.wav.phone_ids`` -> ``X.wav`` (the shared tensor stem)."""
    name = os.path.basename(phone_path)
    if name.endswith(PHONE_SUFFIX):
        return name[: -len(PHONE_SUFFIX)]
    return os.path.splitext(name)[0]


def cond_path(phone_path: str, out_dir: Optional[str] = None) -> str:
    """Sibling ``<base>.cond`` (or ``out_dir/<base>.cond``)."""
    if out_dir is None:
        stem = (phone_path[: -len(PHONE_SUFFIX)]
                if phone_path.endswith(PHONE_SUFFIX)
                else os.path.splitext(phone_path)[0])
        return stem + COND_SUFFIX
    return os.path.join(out_dir, clip_base(phone_path) + COND_SUFFIX)


def find_sidecar(dirs: Sequence[str], base: str, suffix: str) -> Optional[str]:
    """First existing ``<dir>/<base><suffix>``.

    A linear probe over a handful of directories rather than an index over
    ~1e5 files per directory: ``os.path.exists`` is cheap and the corpus spans
    one tensor directory per source corpus.
    """
    for d in dirs:
        p = os.path.join(d, base + suffix)
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# per-clip audio hints (the expensive, parallelisable part)
# --------------------------------------------------------------------------- #
_MODEL = None  # DSPIntensityModel, one per worker process


def init_worker(model_path: str) -> None:
    """Worker initialiser: make the ported modules importable, load the model."""
    for p in (_HERE, _ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    get_dsp_model(model_path)


def _load_tensor(path: str):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_audio_48k(path: str) -> np.ndarray:
    """Load a waveform: a stage-1 ``<base>.wave`` tensor or any source audio.

    Source audio is resampled to 48 kHz by librosa. The DSP features peak-
    normalise and then analyse at 16 kHz, so the fallback path yields the same
    hint as the tensor path (verified in ``hints/verify_dsp.py``).
    """
    if path.endswith(WAVE_SUFFIX):
        return np.asarray(_load_tensor(path), dtype=np.float32).reshape(-1)
    import librosa
    wave, _ = librosa.load(path, sr=WAVE_SR, mono=True)
    return np.asarray(wave, dtype=np.float32).reshape(-1)


def get_dsp_model(model_path: Optional[str] = None):
    """The process-wide DSP model, loaded on first use.

    Cached in ``_MODEL`` so the worker pool (initialised once) and an inference
    caller asking for several hints per clip load it only once.
    """
    global _MODEL
    if _MODEL is None:
        from hints.model_io import load_dsp_model
        _MODEL = load_dsp_model(model_path or DEFAULT_MODEL)
    return _MODEL


def intensity_from_wave_array(wave, sr: int = WAVE_SR) -> float:
    """Calibrated, whisper-gated DSP intensity score (1-9) for a waveform.

    ``dsp_features`` assumes a 16 kHz analysis grid, so the resample is
    mandatory.  Measured on a real stage-1 clip: the predicted score moves by
    <0.05 across ``soxr_hq`` / ``kaiser_best`` / ``polyphase`` relative to the
    score computed from the upstream 16 kHz features, against a model MAE of
    0.59 -- i.e. the resampler choice is immaterial to the hint.
    """
    import librosa
    import pandas as pd
    from dsp_features import extract

    model = get_dsp_model()
    w = np.asarray(wave, dtype=np.float32).reshape(-1)
    if w.size == 0:
        raise ValueError("empty waveform")
    if sr != INTENSITY_SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=INTENSITY_SR,
                             res_type="soxr_hq")
    feats = extract(w, tier=model.tier, all_features=True)
    return float(model.predict(pd.DataFrame([feats]))[0])


def intensity_from_wave(wave_path: str) -> float:
    """Intensity for a file: ``<base>.wave`` (48 kHz) or source audio."""
    return intensity_from_wave_array(load_audio_48k(wave_path))


def register_from_f0_array(f0) -> float:
    """Utterance register: voiced-median f0 in semitones re: :data:`F0_REF_HZ`.

    Median rather than mean, matching ``pitch_correlation.py`` upstream: the
    voiced *mean* in Hz is dominated by the highest frames, so one vibrato
    excursion or an RMVPE octave error moves it much more than it moves the
    perceptual register.  On 40 val clips the two agree closely (std 9.79 vs
    9.91 st, no clip with mean/median > 1.3), so this is a robustness choice,
    not a change of meaning.
    """
    f = np.asarray(f0, dtype=np.float64).reshape(-1)
    voiced = f[f > 0]
    if voiced.size == 0:
        return float("nan")
    return float(12.0 * math.log2(float(np.median(voiced)) / F0_REF_HZ))


def register_from_f0(f0_path: str) -> float:
    """Register for a file: a stage-1 ``<base>.f0`` (100 Hz, 0 = unvoiced)."""
    return register_from_f0_array(_load_tensor(f0_path))


def audio_hints(job):
    """Worker entry: ``(base, wave_path, f0_path) -> (base, values, errors)``."""
    base, wave_path, f0_path = job
    values: Dict[str, float] = {}
    errors: Dict[str, str] = {}
    if wave_path is not None:
        try:
            values["intensity"] = intensity_from_wave(wave_path)
        except Exception as exc:  # keep the batch going; report per field
            errors["intensity"] = f"{type(exc).__name__}: {exc}"
    if f0_path is not None:
        try:
            values["register"] = register_from_f0(f0_path)
        except Exception as exc:
            errors["register"] = f"{type(exc).__name__}: {exc}"
    return base, values, errors


# --------------------------------------------------------------------------- #
# semantic embedding (optional; needs transformers + the bge weights)
# --------------------------------------------------------------------------- #
def encode_semantics(texts: Sequence[str],
                     model_name: str = DEFAULT_SEMANTIC_MODEL,
                     device: Optional[str] = None,
                     batch_size: int = 32):
    """L2-normalised CLS embeddings, float16 -- same recipe as
    ``scratch/bge_embeddings.py`` (``model_output[0][:, 0]`` then normalise), so
    a vector computed here and one computed at inference are interchangeable."""
    if not texts:
        return None
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                  max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = F.normalize(out[0][:, 0], p=2, dim=1)
        outs.append(emb.to(torch.float16).cpu())
    del model
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------- #
# bin edges / writing
# --------------------------------------------------------------------------- #
def quantile_edges(values: Sequence[float], n_levels: int):
    """Equal-frequency edges for ``n_levels`` buckets. Returns ``(edges, stats)``.

    Equal-frequency rather than equal-width on the 1-9 scale on purpose: the DSP
    regression output is compressed (measured on the upstream val slice:
    ``dsp ~ 2.07 + 0.51 * label``, std 0.71 vs label 0.93), so fixed-width bins
    would leave most of the level table unused.  The dataset buckets with
    ``searchsorted``, so the edges only have to be monotone.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    if n_levels < 2 or v.size < n_levels:
        return None, {"n": int(v.size)}
    qs = np.linspace(0.0, 1.0, n_levels + 1)[1:-1]
    edges = [float(x) for x in np.quantile(v, qs)]
    stats = {
        "n": int(v.size),
        "min": float(v.min()),
        "p1": float(np.quantile(v, 0.01)),
        "median": float(np.quantile(v, 0.5)),
        "p99": float(np.quantile(v, 0.99)),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }
    return edges, stats


def phone_count(phone_path: str) -> Optional[int]:
    """``phone_ids`` length by header read -- used as the staleness anchor."""
    try:
        t = torch.load(phone_path, map_location="cpu", weights_only=True,
                       mmap=True)
        return int(t.numel())
    except Exception:
        return None


def cond_has(path: str, keys: Sequence[str]) -> bool:
    """True if ``path`` exists, loads, and holds every one of ``keys``."""
    if not os.path.exists(path):
        return False
    try:
        rec = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        return False
    return all(k in rec for k in keys)


def write_cond(path: str, record: dict) -> None:
    """Atomic ``torch.save`` (a killed run never leaves a truncated ``.cond``)."""
    tmp = path + ".tmp"
    torch.save(record, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filelist", nargs="+", required=True,
                   help="stage-2 filelists (phone_ids|spec_codes|pitch_codes|sid)")
    p.add_argument("--wave-dir", nargs="*", default=[],
                   help="directories holding <base>.wave / <base>.f0 (stage-1 "
                        "tensor dirs). Required for intensity/register.")
    p.add_argument("--source-filelist", nargs="*", default=[],
                   help="path|sid|transcript filelists. Required for semantic; "
                        "also the source of the audio fallback below.")
    p.add_argument("--audio-fallback", dest="audio_fallback",
                   action="store_true", default=True,
                   help="when <base>.wave/.f0 is not in --wave-dir, read the "
                        "clip's own audio from --source-filelist instead of "
                        "skipping the intensity hint (default: on)")
    p.add_argument("--no-audio-fallback", dest="audio_fallback",
                   action="store_false",
                   help="never read source audio; only use --wave-dir tensors")
    p.add_argument("--out-dir", default=None,
                   help="write every .cond here instead of next to phone_ids")
    p.add_argument("--hints", default=",".join(HINT_KEYS),
                   help="comma list of intensity,register,semantic")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="DSP intensity .pkl (see hints/README.md)")
    p.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    p.add_argument("--semantic-batch", type=int, default=32)
    p.add_argument("--device", default=None,
                   help="device for the sentence encoder (default: cuda if any)")
    p.add_argument("--hints-json", default=None,
                   help="default: <dirname(filelist[0])>/hints.json")
    p.add_argument("--n-intensity-levels", type=int, default=8)
    p.add_argument("--n-register-levels", type=int, default=5)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel processes for the audio-derived hints")
    p.add_argument("--limit", type=int, default=0, help="max clips (0 = all)")
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse a .cond that already holds every requested key")
    p.add_argument("--no-edges", action="store_true",
                   help="do not fit level edges (still writes coverage/provenance)")
    args = p.parse_args(argv)

    hints = [h.strip() for h in str(args.hints).split(",") if h.strip()]
    unknown = [h for h in hints if h not in HINT_KEYS]
    if unknown:
        p.error(f"unknown hint(s) {unknown}; choose from {list(HINT_KEYS)}")
    if ({"intensity", "register"} & set(hints)) and not args.wave_dir:
        p.error("--wave-dir is required for intensity/register "
                "(they read <base>.wave / <base>.f0)")
    if "semantic" in hints and not args.source_filelist:
        p.error("--source-filelist is required for semantic "
                "(it needs the transcript)")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    items = read_stage2_filelist(args.filelist)
    if args.limit:
        items = items[: args.limit]
    transcripts = (read_transcripts(args.source_filelist)
                   if "semantic" in hints else {})
    source_audio = (read_source_audio(args.source_filelist)
                    if (args.audio_fallback and "intensity" in hints) else {})
    print(f"{len(items)} clip(s) from {len(args.filelist)} filelist(s)")
    if source_audio:
        print(f"Audio fallback: {len(source_audio)} source file(s) available "
              f"for clips without a stage-1 .wave")

    todo, n_skipped, n_nodata, n_fallback = [], 0, 0, 0
    for it in items:
        base = clip_base(it["phone_ids"])
        wave = (find_sidecar(args.wave_dir, base, WAVE_SUFFIX)
                if "intensity" in hints else None)
        if wave is None and source_audio:
            cand = source_audio.get(base)
            if cand and os.path.exists(cand):
                wave = cand
                n_fallback += 1
        rec = {
            "base": base,
            "phone_ids": it["phone_ids"],
            "cond": cond_path(it["phone_ids"], args.out_dir),
            "wave": wave,
            "f0": (find_sidecar(args.wave_dir, base, F0_SUFFIX)
                   if "register" in hints else None),
            "text": transcripts.get(base) if "semantic" in hints else None,
        }
        want = []
        if rec["wave"]:
            want.append(COND_KEYS["intensity"])
        if rec["f0"]:
            want.append(COND_KEYS["register"])
        if rec["text"]:
            want.append(COND_KEYS["semantic"])
        if not want:
            n_nodata += 1
            continue
        if args.skip_existing and cond_has(rec["cond"], want):
            n_skipped += 1
            continue
        todo.append(rec)

    # --- audio-derived hints (parallelisable) --------------------------------
    results = {}
    audio_jobs = [r for r in todo if r["wave"] or r["f0"]]
    if audio_jobs:
        from tqdm import tqdm
        init_worker(args.model)  # the parent needs it for the sequential path
        jobs = [(r["base"], r["wave"], r["f0"]) for r in audio_jobs]
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=init_worker,
                                     initargs=(args.model,)) as ex:
                for base, values, errors in tqdm(
                        ex.map(audio_hints, jobs, chunksize=8),
                        total=len(jobs), desc="audio hints"):
                    results[base] = (values, errors)
        else:
            for job in tqdm(jobs, desc="audio hints"):
                base, values, errors = audio_hints(job)
                results[base] = (values, errors)

    # --- sentence embeddings (batched, one model load) -----------------------
    semantic = {}
    if "semantic" in hints:
        pairs = [(r["base"], r["text"]) for r in todo if r["text"]]
        if pairs:
            embs = encode_semantics([t for _, t in pairs],
                                    args.semantic_model, args.device,
                                    args.semantic_batch)
            for i, (base, _) in enumerate(pairs):
                semantic[base] = embs[i]
    # --- write ---------------------------------------------------------------
    from collections import Counter
    errs: Counter = Counter()
    collected = {"intensity": [], "register": []}
    n_written = 0
    for r in todo:
        values, errors = results.get(r["base"], ({}, {}))
        for k, msg in errors.items():
            errs[f"{k}: {str(msg).splitlines()[0][:160]}"] += 1
        rec = {"version": 1}
        nph = phone_count(r["phone_ids"])
        if nph is not None:
            rec["n_phone"] = nph
        for key in ("intensity", "register"):
            v = values.get(key)
            if v is not None and np.isfinite(v):
                rec[COND_KEYS[key]] = torch.tensor([float(v)],
                                                   dtype=torch.float32)
                collected[key].append(float(v))
        if r["base"] in semantic:
            rec["semantic"] = semantic[r["base"]]
        if not any(k in rec for k in ("intensity", "register_st", "semantic")):
            continue
        write_cond(r["cond"], rec)
        n_written += 1

    # --- provenance + bin edges ---------------------------------------------
    # Merge with any existing hints.json: a `--skip-existing` re-run computes no
    # values at all, and a semantic-only run loads no DSP model.  Overwriting
    # those fields with null would silently invalidate every level the dataset
    # buckets against, so unknown fields are inherited and only freshly computed
    # ones replace them.
    hints_json = args.hints_json or os.path.join(
        os.path.dirname(args.filelist[0]) or ".", "hints.json")
    previous = {}
    if os.path.exists(hints_json):
        try:
            with open(hints_json, "r", encoding="utf-8") as f:
                previous = json.load(f) or {}
        except Exception:
            previous = {}
    doc = {"version": 1, "created_by": "hints/build.py",
           "levels": dict(previous.get("levels") or {}),
           "dsp": previous.get("dsp"),
           "semantic": previous.get("semantic")}
    if _MODEL is not None:
        doc["dsp"] = {
            "model": os.path.relpath(args.model),
            "tier": int(_MODEL.tier),
            "cal": [float(c) for c in getattr(_MODEL, "cal", (1.0, 0.0))],
            "gate": getattr(_MODEL, "gate", None) is not None,
            "input": "48 kHz <base>.wave -> 16 kHz soxr_hq",
        }
    if "semantic" in hints:
        doc["semantic"] = {"model": args.semantic_model, "dim": 384,
                           "pooling": "cls", "normalized": True}
    if not args.no_edges:
        for name, n_levels, values in (
                ("intensity", args.n_intensity_levels, collected["intensity"]),
                ("register", args.n_register_levels, collected["register"])):
            if name not in hints:
                continue
            edges, stats = quantile_edges(values, n_levels)
            if edges is None:
                if doc["levels"].get(name, {}).get("edges"):
                    print(f"NOTE: kept the existing {name} edges from "
                          f"{hints_json} (no new values in this run)")
                    continue
                # Explicit null + the count, so a too-small run is legible in
                # hints.json instead of silently missing the section.
                doc["levels"][name] = {"n_levels": n_levels, "edges": None,
                                       "stats": stats}
                print(f"WARNING: only {stats.get('n', 0)} {name} value(s); "
                      f"cannot fit {n_levels} level edges yet")
                continue
            doc["levels"][name] = {"n_levels": n_levels, "edges": edges,
                                   "stats": stats}
    doc["coverage"] = {
        "clips_read": len(items),
        "clips_written": int(n_written),
        "skipped_existing": int(n_skipped),
        "no_data": int(n_nodata),
        "with_intensity": len(collected["intensity"]),
        "with_register": len(collected["register"]),
        "with_semantic": len(semantic),
        "audio_fallback": int(n_fallback),
        "errors": dict(errs),
    }
    if os.path.dirname(hints_json):
        os.makedirs(os.path.dirname(hints_json), exist_ok=True)
    with open(hints_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"wrote {n_written} .cond file(s) "
          f"({n_skipped} reused, {n_nodata} without any hint data)")
    print(f"  intensity: {len(collected['intensity'])}  "
          f"register: {len(collected['register'])}  "
          f"semantic: {len(semantic)}"
          + (f"  (from source audio: {n_fallback})" if n_fallback else ""))
    print(f"  wrote {hints_json}")
    if args.limit and not args.no_edges:
        print("NOTE: --limit was used, so the bin edges above are fitted on a "
              "subset; re-run without --limit before training.")
    for k, n in errs.most_common(5):
        print(f"  error ({n}x): {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


