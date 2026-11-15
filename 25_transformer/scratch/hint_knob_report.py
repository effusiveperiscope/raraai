#!/usr/bin/env python3
"""Counterfactual knob report: hint level -> properties of the decoded audio.

``val_hint_gap`` / teacher-forced CE answer *"is the channel predictive"*.  They
cannot answer *"is it a knob"*, which is the actual product question for a
conditioning hint: if the user moves intensity from level 2 to level 7, does the
synthesised voice change in the intended direction, and does it stay
intelligible?  For a *style* channel the CE gradient is tiny by construction
(``HINTS_V1_PLAN.md`` §4.1), so CE is the wrong KPI here.

Per channel and level, this script generates codes with that channel forced to
the level (everything else from the item's real ``.cond``), decodes them with
the frozen stage-1 ``Generator``, and measures the audio:

``DSP int``
    The perceptual-intensity model (``hints/build.py``, same regressor that
    produced the intensity hint).  The intensity knob is working if this
    tracks the level.  Loudness-independent by design.
``reg st``
    Voiced-median f0 of the decoded audio (RMVPE), in semitones re: 100 Hz.
``f0 Hz``
    The same in Hz, for readability.
``dur s``
    The model's own generated duration (EOS), so a duration knob (word slot,
    and indirectly timing) shows up as a shift here.
``repeat``
    Adjacent-frame repetition of spec codebook 0 (the degeneration KPI from
    ``eval_stage2_repetition.py``): guidance/levels must not buy a knob by
    pushing the model into a loop.
``WER``
    Optional (``--wer``): ASR score of the decoded audio against the item's
    reference transcript, pooled over items at that level.

Two kinds of arm share the table, because "set the knob" and "pace this clip"
are different user requests:

* ``--levels`` forces the channel to one **constant** level for every phone --
  what a clip-level control does.  For ``slot`` that flattens the rhythm to one
  duration bucket per word, which is why the high levels are destructive.
* ``--shift channel=d1,d2`` keeps the item's own pattern and adds a delta to
  every *present* entry (level 0 stays absent).  For ``hint_slot`` a delta of +1
  doubles each word's length, i.e. it is the "slower / faster" knob.  Pair it
  with ``--gen_scale > 1``: at the default 1.0 the generation cap is the item's
  own frame count, so a slowdown cannot be expressed at all.

The ``real`` row is the item's own hint levels (no override) -- the reference
point every level should be compared against.  ``slope`` is a least-squares fit
of the readout against the level; for a working knob it should be clearly
non-zero and point the right way (intensity up / register up / duration up).
Level 0 is the trained "not provided" state, so every channel's sweep starts
from the same baseline.

Read-only: writes only the ``--save_dir`` wavs if asked.

Usage::

    # is the intensity channel a knob on the finished v0.9?
    python scratch/hint_knob_report.py \
        --checkpoint checkpoints_stage2/transformer_v0.9/last.ckpt \
        --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt \
        --channel intensity --num_items 8

    # all four discrete channels, every level, with WER
    python scratch/hint_knob_report.py --channel all --levels all --wer ...

    # under guidance (Phase 3)
    python scratch/hint_knob_report.py --channel intensity,register \
        --hint_guidance 2.0 ...

    # pacing knob: shift each word's own duration bucket by +-1/2 (log2), and
    # let the model run past the reference length so "slower" is expressible
    python scratch/hint_knob_report.py --channel slot --levels none \
        --shift "slot=-2,-1,1,2" --gen_scale 2.0 --wer ...

Notes:
  * Needs ``transformer.hints: true``, a checkpoint that has the hint tables,
    and a ``.cond`` next to every val item.  The semantic channel is a 384-d
    vector, not a level table, so it cannot be swept here; use
    ``scratch/hint_channel_probe.py`` for its liveness.
  * RMVPE is the same model the hint builder used, so ``reg st`` here and the
    value stored in ``.cond`` are directly comparable.
  * ``--levels all`` is ``0..max`` for that channel's table (pitch includes the
    explicit unvoiced class as its last level).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

# fairseq (pulled in by modeling.generator) calls logging.basicConfig(DEBUG) at
# import time; install our handler first so that call becomes a no-op
# (AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402

# Same-directory helper: the WER plumbing (ASR wrapper, reference loader,
# text normalisation, generator loading) lives there and must not be duplicated.
from eval_stage2_wer import (  # noqa: E402
    Asr, build_transformer, code_frame_samples, item_basename, load_generator,
    load_references, load_transformer_weights, pick_indices,
    resolve_generator_path, to_16k, wer_summary)

#: channel name -> (batch key, func(net, levels) -> highest level id)
CHANNELS = {
    "intensity": "hint_int",
    "register": "hint_reg",
    "pitch": "hint_pitch",
    "slot": "hint_slot",
}


def level_count(net, channel: str) -> int:
    if channel == "intensity":
        return int(net.n_intensity_levels)
    if channel == "register":
        return int(net.n_register_levels)
    if channel == "pitch":
        # row ``n_pitch_levels + 1`` is the explicit unvoiced class
        return int(net.n_pitch_levels) + 1
    if channel == "slot":
        return int(net.n_slot_buckets)
    raise ValueError(f"unknown channel {channel!r}")


def hint_kwargs(config):
    tc = config.transformer
    return dict(
        hints=bool(tc.get("hints", False)),
        hints_json=tc.get("hints_json", None),
        hint_dropout=dict(tc.get("hint_dropout", {}) or {}),
        hint_null_all=0.0,
        hint_pitch_edges=tc.get("hint_pitch_edges", None),
        n_slot_buckets=int(tc.get("n_slot_buckets", 8) or 8),
    )


def repeated_fraction(codes: torch.Tensor) -> float:
    c = codes.detach().reshape(-1).to("cpu")
    if c.numel() < 2:
        return float("nan")
    return float((c[1:] == c[:-1]).float().mean())


class AudioScorer:
    """Intensity + RMVPE register of a 48 kHz waveform (lazy heavy imports)."""

    def __init__(self, device):
        self.device = device
        self._rmvpe = None
        from hints import build as hb

        self._hb = hb
        hb.get_dsp_model()

    @property
    def rmvpe(self):
        if self._rmvpe is None:
            from svc_helper.pitch.rmvpe import RMVPEModel

            self._rmvpe = RMVPEModel(device=self.device,
                                     is_half=(self.device.type == "cuda"))
        return self._rmvpe

    @torch.no_grad()
    def f0(self, wav48: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        import librosa

        w = np.asarray(wav48, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            w = librosa.resample(w, orig_sr=sample_rate, target_sr=16000)
        f0, _ = self.rmvpe.extract_pitch2(np.ascontiguousarray(w))
        if torch.is_tensor(f0):
            f0 = f0.detach().float().cpu().numpy()
        return np.asarray(f0, dtype=np.float32).reshape(-1)

    def score(self, wav48: np.ndarray, sample_rate: int = 48000) -> Dict[str, float]:
        out = {"intensity": float("nan"), "reg_st": float("nan"),
               "f0_hz": float("nan")}
        try:
            out["intensity"] = float(self._hb.intensity_from_wave_array(
                wav48, sr=sample_rate))
        except Exception as exc:                       # keep the sweep going
            print(f"  intensity readout failed: {type(exc).__name__}: {exc}")
        try:
            f0 = self.f0(wav48, sample_rate)
            out["reg_st"] = float(self._hb.register_from_f0_array(f0))
            voiced = f0[f0 > 0]
            if voiced.size:
                out["f0_hz"] = float(np.median(voiced))
        except Exception as exc:
            print(f"  f0 readout failed: {type(exc).__name__}: {exc}")
        return out


def chunked(seq, size):
    size = max(int(size), 1)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@torch.no_grad()
def run_arm(net, net_g, items, labels, args, device, collator,
            key: Optional[str], level: int,
            shift: Optional[tuple] = None) -> Dict[str, object]:
    """Generate + decode ``items`` with one hint channel forced or shifted.

    Three arms share this function:

    * ``key=None``, ``shift=None`` -- the item's real levels (reference arm).
    * ``key`` + ``level`` -- *force* the channel to one constant level, i.e.
      "set the knob to L" (what a UI would do for a clip-level channel).
    * ``shift=(key, delta, max_level)`` -- *shift* each **present** entry of the
      channel by ``delta`` and leave level 0 (absent) alone.  For a log2-duration
      channel like ``hint_slot`` this is "make every real word 2**delta times
      longer", i.e. a pace change that keeps the item's own rhythm instead of
      flattening it to a constant.

    Returns per-item waveform / frame count / repetition.
    """
    hop = code_frame_samples(net_g)
    n_spec = int(net.spec_codebook_num)
    wavs, frames, repeats = [], [], []
    for chunk in chunked(items, args.batch_size):
        batch = collator(chunk)
        if key is not None:
            batch[key] = torch.full_like(batch[key], int(level))
        elif shift is not None:
            skey, delta, smax = shift
            cur = batch[skey].long()
            batch[skey] = torch.where(
                cur > 0, (cur + delta).clamp(1, smax), cur)
        hint_arg = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()}
        # `--gen_scale` lets the model run *past* the reference length, which a
        # pacing test needs: with the default 1.0 the cap is the item's own frame
        # count, so a "slow down" request can only ever be squashed.
        lens = [min(int(round(int(it["spec_codes"].shape[1]) * args.gen_scale)),
                    args.max_frames)
                for it in chunk]
        gen, glen = net.generate(
            batch["phone_ids"].to(device), batch["phone_lengths"].to(device),
            batch["sid"].to(device), max_frames=max(lens),
            temperature=args.temperature, top_k=args.top_k,
            stop_on_eos=(False if args.ignore_eos else None),
            hints=hint_arg, return_lengths=True,
            hint_guidance=args.hint_guidance,
            hint_guidance_books=args.hint_guidance_books)
        spec = gen[:, :n_spec].transpose(0, 1).contiguous()
        pitch = gen[:, n_spec:].transpose(0, 1).contiguous()
        audio = net_g.infer_from_codes(spec, pitch, batch["sid"].to(device))
        for i in range(len(chunk)):
            n = max(min(int(glen[i]), lens[i]), 0)
            wavs.append(audio[i, 0, :n * hop].float().cpu().numpy())
            frames.append(n)
            repeats.append(repeated_fraction(gen[i, 0, :n]))
    return {"wavs": wavs, "frames": frames, "repeat": repeats,
            "labels": list(labels)}


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]                       # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares d(readout)/d(level) over the non-NaN points."""
    pts = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pts) < 2:
        return float("nan")
    x = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    return float(((x - xm) * (y - ym)).sum() / den) if den > 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--generator_checkpoint", default=None,
                    help="stage-1 Generator used to decode codes; default: "
                         "config.train.generator_checkpoint")
    ap.add_argument("--val_filelist", default=None,
                    help="default: config.train.val_filelist")
    ap.add_argument("--source_filelists", nargs="*", default=None,
                    help="transcripts for --wer; default: filelists_2/*.txt")
    ap.add_argument("--channel", default="intensity",
                    help="comma separated subset of intensity,register,pitch,"
                         "slot, or 'all'")
    ap.add_argument("--shift", default=None,
                    help="pacing/offset test instead of a constant level, e.g. "
                         "slot=-2,-1,1,2 (comma separated deltas).  Each "
                         "**present** entry is shifted by the delta and clamped "
                         "to its table, so the item keeps its own rhythm; level "
                         "0 (absent) is left alone.  For hint_slot a delta of +1 "
                         "doubles every word's length.  Runs in addition to (and "
                         "listed after) the --levels sweep.")
    ap.add_argument("--gen_scale", type=float, default=1.0,
                    help="multiply the item's own frame count when capping "
                         "generation (default 1.0).  Use >1 for a pacing test: "
                         "at 1.0 a slowdown cannot be expressed at all.")
    ap.add_argument("--levels", default="all",
                    help="'all' (0..max for that channel) or a comma separated "
                         "list, e.g. 0,2,4,6,8")
    ap.add_argument("--num_items", type=int, default=8,
                    help="val items per level (evenly spaced; 0 = all)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=1000,
                    help="upper bound on generated code frames (the item's own "
                         "length is used when shorter)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: train.sample_temperature; 0 = greedy")
    ap.add_argument("--top_k", type=int, default=None,
                    help="default: train.sample_top_k; 0 disables")
    ap.add_argument("--ignore_eos", action="store_true",
                    help="decode exactly the item's own length; note this hides "
                         "the duration knob, so the default keeps EOS")
    ap.add_argument("--wer", action="store_true",
                    help="also score each level with ASR (slow; needs the "
                         "reference transcripts)")
    ap.add_argument("--hint_guidance", "--hint-guidance", type=float,
                    default=None, dest="hint_guidance")
    ap.add_argument("--hint_guidance_books", "--hint-guidance-books",
                    default=None, dest="hint_guidance_books")
    ap.add_argument("--asr_model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--language", default="English")
    ap.add_argument("--save_dir", default=None,
                    help="write the decoded wavs here (<channel>_L<level>_<i>.wav)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.hint_guidance_books:
        args.hint_guidance_books = [float(w) for w in
                                    str(args.hint_guidance_books).split(",")
                                    if w.strip()]

    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.temperature is None:
        args.temperature = float(config.train.get("sample_temperature", 0.9))
    if args.top_k is None:
        k = config.train.get("sample_top_k", None)
        args.top_k = int(k) if k else None
    elif args.top_k == 0:
        args.top_k = None

    channels = (list(CHANNELS) if args.channel.strip() == "all"
                else [c.strip() for c in args.channel.split(",") if c.strip()])
    for c in channels:
        if c not in CHANNELS:
            raise SystemExit(f"unknown channel {c!r}; choose from "
                             f"{sorted(CHANNELS)} or 'all'")
    if not bool(config.transformer.get("hints", False)):
        raise SystemExit("this config has transformer.hints: false -- there is "
                         "nothing to sweep.")

    shift_spec: Dict[str, List[int]] = {}
    if args.shift:
        for item in str(args.shift).split(";"):
            if not item.strip():
                continue
            try:
                name, deltas = item.split("=", 1)
            except ValueError:
                raise SystemExit(f"--shift expects channel=d1,d2 (got {item!r})")
            name = name.strip()
            if name not in CHANNELS:
                raise SystemExit(f"--shift: unknown channel {name!r}; choose "
                                 f"from {sorted(CHANNELS)}")
            shift_spec[name] = [int(v) for v in deltas.split(",") if v.strip()]
        args.shift = shift_spec

    val_filelist = args.val_filelist or config.train.val_filelist
    dataset = CodeDataset(
        val_filelist, is_train=False, max_frames=None, max_phone_length=None,
        speaker_map=config.train.get("speaker_map", None), **hint_kwargs(config))
    all_indices = list(range(len(dataset)))
    picked = pick_indices(all_indices, args.num_items)
    if not picked:
        raise SystemExit(f"no val items in {val_filelist}")
    items = [dataset[i] for i in picked]
    labels = [item_basename(dataset.items[i]["spec_codes"]) for i in picked]
    if items[0].get("hint_int") is None:
        raise SystemExit(
            "the val items carry no .cond hints -- run hints/build.py over "
            "this split first (the sweep overrides one channel, so the others "
            "must come from disk).")

    gen_path = resolve_generator_path(
        args.generator_checkpoint or config.train.get("generator_checkpoint"))
    if gen_path is None:
        raise SystemExit("a stage-1 Generator checkpoint is required to score "
                         "the decoded audio; pass --generator_checkpoint")
    sample_rate = int(config.data.sampling_rate)
    print(f"checkpoint: {args.checkpoint}")
    print(f"val: {val_filelist} -> {len(items)} items  device={device}  "
          f"temp={args.temperature} top_k={args.top_k}  "
          f"guidance={args.hint_guidance}")
    print(f"generator: {gen_path}\n")

    net_g = load_generator(gen_path, config, device)
    hop = code_frame_samples(net_g)
    fps = sample_rate / max(hop, 1)

    refs = None
    if args.wer:
        sources = (args.source_filelists
                   or sorted(__import__("glob").glob("filelists_2/*.txt")))
        refs = load_references(sources)
        missing = [l for l in labels if l not in refs]
        if missing:
            print(f"NOTE: {len(missing)}/{len(items)} items have no reference "
                  f"transcript; WER covers the rest.")
    asr = Asr(args.asr_model, device, args.language) if args.wer else None
    scorer = AudioScorer(device)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    net = build_transformer(config, device)
    epoch, step = load_transformer_weights(net, args.checkpoint)
    print(f"epoch {epoch}, step {step}\n")
    collator = CodeCollator(config=config)

    def score_arm(key, level, label, shift=None):
        torch.manual_seed(args.seed)
        arm = run_arm(net, net_g, items, labels, args, device, collator,
                      key, level, shift=shift)
        wavs, frames = arm["wavs"], arm["frames"]
        if args.save_dir:
            import soundfile as sf
            tag = (f"{key[5:]}_L{level}" if key else "real")
            for i, w in enumerate(wavs):
                sf.write(os.path.join(args.save_dir, f"{tag}_{i:03d}.wav"),
                         np.clip(w, -1.0, 1.0), sample_rate)
        aud = [scorer.score(w, sample_rate) for w in wavs]
        row = {
            "label": label,
            "level": level,
            "intensity": mean([a["intensity"] for a in aud]),
            "reg_st": mean([a["reg_st"] for a in aud]),
            "f0_hz": mean([a["f0_hz"] for a in aud]),
            "dur_s": mean([n / fps for n in frames]),
            "repeat": mean(arm["repeat"]),
            "wer": float("nan"),
        }
        if asr is not None:
            keep = list(zip(labels, wavs))
            pairs, hyps = [], asr.transcribe(
                to_16k([w for _, w in keep], sample_rate))
            for (label_i, _), hyp in zip(keep, hyps):
                if label_i in refs:
                    pairs.append((refs[label_i], hyp))
            if pairs:
                row["wer"] = wer_summary(pairs)["wer"]
        return row

    COLS = ("label", "intensity", "reg_st", "f0_hz", "dur_s", "repeat", "wer")
    HEAD = {"label": "level", "intensity": "DSP int", "reg_st": "reg st",
            "f0_hz": "f0 Hz", "dur_s": "dur s", "repeat": "repeat",
            "wer": "WER"}

    for channel in channels:
        key = CHANNELS[channel]
        max_level = level_count(net, channel)
        if args.levels.strip() in ("none", "skip"):
            levels = []                       # shift-only run
        elif args.levels.strip() == "all":
            levels = list(range(0, max_level + 1))
        else:
            levels = [int(v) for v in args.levels.split(",") if v.strip()]
        levels = [lv for lv in levels if 0 <= lv <= max_level]
        if not levels and not (args.shift or {}).get(channel):
            print(f"{channel}: no valid levels (max {max_level}); skipped\n")
            continue

        # Where the corpus actually sits (clip-level channels only).
        if channel in ("intensity", "register") and items[0].get(key) is not None:
            true_lv = [int(it[key]) for it in items]
            hist = {lv: true_lv.count(lv) for lv in sorted(set(true_lv))}
            print(f"=== {channel}: true levels on these items: "
                  + " ".join(f"L{k}x{v}" for k, v in hist.items()))
        else:
            print(f"=== {channel}: {len(levels)} level(s) 0..{max_level}")

        real = score_arm(None, -1, "real")
        rows = [score_arm(key, lv, f"L{lv}") for lv in levels]

        def print_table(rows_):
            print(f"  {'lvl':<7}" + "".join(f"{HEAD[c]:>9}" for c in COLS[1:]))
            for row in rows_:
                cells = []
                for c in COLS[1:]:
                    v = row[c]
                    if v != v:
                        cells.append(f"{'-':>9}")
                    elif c == "wer":
                        cells.append(f"{v * 100:>8.2f}%")
                    else:
                        cells.append(f"{v:>9.3f}")
                print(f"  {row['label']:<7}" + "".join(cells))

        if rows:
            rows.append(real)
            print_table(rows)
            fits = []
            for c in ("intensity", "reg_st", "dur_s"):
                fits.append(f"{c} {slope([r['level'] for r in rows[:-1]],
                                         [r[c] for r in rows[:-1]]):+.4f}/lvl")
            print("  slope: " + "   ".join(fits))

        # Shift arm: keep the item's own pattern, move every present entry.
        deltas = (args.shift or {}).get(channel)
        if deltas:
            print(f"\n  shift {channel} by (each present entry, level 0 kept; "
                  f"generation cap x{args.gen_scale:g}):")
            shifted = [score_arm(None, d, f"{d:+d}",
                                 shift=(key, d, max_level)) for d in deltas]
            shifted.append(real)
            print_table(shifted)
            if any(r["dur_s"] == r["dur_s"] for r in shifted):
                print("  dur slope vs shift: "
                      + f"{slope([r['level'] for r in shifted[:-1]], [r['dur_s'] for r in shifted[:-1]]):+.4f}"
                      f" s per level   (a pacing knob should be clearly "
                      f"non-zero and same-signed)")
        print("  (a working knob moves its own readout monotonically; "
              "check `repeat` and WER\n   at the extremes for the cost)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
