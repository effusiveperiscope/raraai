#!/usr/bin/env python3
"""Quantitative KPI checks for the Hints V2 channels (pitch anchors, semantic).

Teacher-forced CE answers "is the channel *predictive*"; it does not answer "is
it a *knob*".  For a clip-level channel CE is nearly blind (a style channel can
move the audio a lot at ~1e-3 nats -- see AGENTS.md "Monotone hint encoding"),
so this script measures the behaviour instead, with paired controls:

**Pitch (anchors).**
  * *shift sweep*: add ``d`` semitones to every present anchor and decode the
    generated codes with the frozen stage-1 Generator.  The decoded f0 level
    (st re: the same clip's ``d=0`` arm) should move by ``d``: slope ~1.0.
  * *swapped anchors*: replace each clip's anchors with the next clip's.  If the
    channel carries *content* (not just a constant offset), the f0 movement
    should track the difference between the injected and the original anchor
    levels, across clips.
  * *flattened anchors*: one anchor per word at its mean.  If the anchors carry
    the contour, within-clip f0 spread should drop while the level stays put.
  * *absent*: the trained null state (what generation does by default).

**Semantic.**  Three paired arms -- ``own`` (the clip's bge vector), ``null``
(``sem_null``) and ``swapped`` (another clip's vector), same seed -- plus a
``zero`` arm (a genuine all-zero vector through ``sem_proj``).  Reports code
agreement per prefix window and the prosody readouts.  If ``swapped`` diverges
from ``null`` as much as ``own`` does, the vector's content is not being used.

Everything is read-only; the checkpoint is only loaded.

Usage::

    python scratch/hint_v2_kpi.py \
        --checkpoint checkpoints_stage2/transformer_hints_v2/last.ckpt
    python scratch/hint_v2_kpi.py --checkpoint ... --num_items 8 --frames 256 \
        --top_k 1                 # greedy (cleanest slope estimate)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# fairseq (pulled in by modeling.generator) turns the root logger to DEBUG at
# import time -- install our handler first (AGENTS.md 6.18).
logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import hint_channel_probe as H  # noqa: E402  (loader + dataset boilerplate)
from dataset_codes import CodeCollator, CodeDataset  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402

WINDOWS = (5, 10, 20, 40, None)
LOG2 = float(np.log2(12.0)) / 1.0          # unused; kept for clarity


def st(hz, ref):
    """Hz -> semitones re: `ref` (0 where either is non-positive)."""
    hz = np.asarray(hz, dtype=np.float64)
    out = np.zeros_like(hz)
    ok = (hz > 0) & (np.broadcast_to(ref, hz.shape) > 0)
    out[ok] = 12.0 * np.log2(hz[ok] / np.broadcast_to(ref, hz.shape)[ok])
    return out


# --------------------------------------------------------------------------- #
# batch surgery
# --------------------------------------------------------------------------- #
def clone_batch(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def anchor_mask(batch, A):
    """``(B, N, A)`` mask of the anchor slots that are actually present."""
    n = batch["hint_pitch_n"].clamp(0, A)
    ar = torch.arange(A, device=n.device).view(1, 1, A)
    return ar < n.unsqueeze(-1)


def branch(batch, name, delta=0.0, donor=None):
    """Return a modified copy of ``batch`` for one measurement arm."""
    out = clone_batch(batch)
    A = int(out["hint_pitch_val"].shape[-1])

    if name == "absent":
        out["hint_pitch_state"] = torch.zeros_like(out["hint_pitch_state"])
        out["hint_pitch_n"] = torch.zeros_like(out["hint_pitch_n"])
        out["hint_pitch_val"] = torch.zeros_like(out["hint_pitch_val"])
        out["hint_pitch_pos"] = torch.zeros_like(out["hint_pitch_pos"])
        return out

    if name == "flat":
        val = out["hint_pitch_val"]
        mask = anchor_mask(out, A).to(val.dtype)
        flat = (val * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)   # (B, N)
        new = torch.zeros_like(val)
        new[..., 0] = flat
        out["hint_pitch_val"] = new
        n = out["hint_pitch_n"].long()
        out["hint_pitch_n"] = (n > 0).long()          # unvoiced stays 0
        pos = torch.zeros_like(out["hint_pitch_pos"])
        pos[..., 0] = 0.5
        out["hint_pitch_pos"] = pos
        return out

    mask = anchor_mask(out, A)
    if name == "shift":
        out["hint_pitch_val"] = out["hint_pitch_val"] + mask * float(delta)
        return out

    if name == "swap":
        for key in ("hint_pitch_val", "hint_pitch_pos", "hint_pitch_n",
                    "hint_pitch_state"):
            out[key] = out[key].roll(-1, dims=0)      # donor = next clip
        return out

    if name == "sem_null":
        out["hint_sem_valid"] = torch.zeros_like(out["hint_sem_valid"])
        return out

    if name == "sem_swap":
        for key in ("hint_sem", "hint_sem_valid"):
            out[key] = out[key].roll(-1, dims=0)
        return out

    if name == "sem_zero":
        out["hint_sem"] = torch.zeros_like(out["hint_sem"])
        out["hint_sem_valid"] = torch.ones_like(out["hint_sem_valid"])
        return out

    raise ValueError(name)


def branch_contour(batch, name):
    """Contour edits that stay *in distribution* for relative anchors.

    The stored anchors are semitones re: the utterance's own voiced median, so
    their level is ~0 by construction and the absolute pitch comes from the
    speaker id.  Shifting **every** anchor therefore asks for something no
    training clip contains; the meaningful edits are shape edits:

    * ``mirror``    -- reverse the contour in time (phone axis reversed, anchor
      positions mirrored).  Alignment-free to score: the generated f0 should
      correlate with the *time-reversed* baseline f0 better than with the
      baseline as-is.
    * ``half_up`` / ``half_down`` -- raise/lower the second half of the phones
      by 3 st, so the second half should sit above/below the first.
    """
    out = clone_batch(batch)
    val = out["hint_pitch_val"]
    pos = out["hint_pitch_pos"]
    n = out["hint_pitch_n"].long()
    state = out["hint_pitch_state"]
    A = int(val.shape[-1])
    _, N, _ = val.shape

    if name == "mirror":
        out["hint_pitch_val"] = torch.flip(val, dims=(1,))
        out["hint_pitch_pos"] = torch.clamp(1.0 - torch.flip(pos, dims=(1,)),
                                            0.0, 1.0)
        out["hint_pitch_n"] = torch.flip(n, dims=(1,))
        out["hint_pitch_state"] = torch.flip(state, dims=(1,))
        return out

    # The phone axis is *padded*, so every positional edit must be built from
    # each item's real length (``phone_lengths``), not from ``N``: with a wide
    # batch, "second half of N" can fall entirely inside another item's padding
    # and the edit silently does nothing (measured: 15 of 16 clips unchanged).
    lens = out["phone_lengths"].long().clamp(min=1).view(-1, 1)
    idx = torch.arange(N, device=val.device).view(1, N)
    real = idx < lens
    mask = anchor_mask(out, A)

    if name in ("half_up", "half_down"):
        d = 3.0 if name == "half_up" else -3.0
        sel = ((idx * 2 >= lens) & real).to(val.dtype).view(-1, N, 1)
        out["hint_pitch_val"] = val + mask * sel * d
        return out

    if name in ("ramp_up", "ramp_down"):
        # A global melodic trend: +-2 st across each utterance, added to the
        # clip's own micro-contour.  Real sentences declinate, so this stays in
        # distribution, and its readout needs no phone/frame alignment (just
        # the f0 trend over time).
        t = idx.to(val.dtype) / (lens - 1).clamp(min=1).to(val.dtype)
        ramp = ((2.0 * t - 1.0) * 2.0).view(-1, N, 1)
        if name == "ramp_down":
            ramp = -ramp
        out["hint_pitch_val"] = val + mask * ramp * real.view(-1, N, 1)
        return out

    raise ValueError(name)


def interp_st(row, points=100):
    """Voiced f0 (Hz) -> st re: its own median, interpolated, resampled."""
    row = np.asarray(row, dtype=np.float64)
    voiced = row > 0
    if int(voiced.sum()) < 5:
        return None
    med = float(np.median(row[voiced]))
    y = np.full(row.shape, np.nan)
    y[voiced] = 12.0 * np.log2(row[voiced] / med)
    idx = np.arange(y.size)
    y = np.interp(idx, idx[voiced], y[voiced])
    return np.interp(np.linspace(0, y.size - 1, int(points)), idx, y)


def trend(row):
    """Melodic trend: median f0 of the last third minus the first third (st).

    Robust on purpose -- a least-squares slope over ~250 frames with a sparse
    voiced tail produced absurd values (a polyfit outlier, not a real contour),
    while medians of the two ends are stable in exactly the sparsity regime
    this probe runs in.
    """
    row = np.asarray(row, dtype=np.float64)
    v = row[row > 0]
    if v.size < 12:
        return float("nan")
    med = float(np.median(v))
    third = row.size // 3
    a, b = row[:third], row[-third:]
    a, b = a[a > 0], b[b > 0]
    if a.size < 3 or b.size < 3:
        return float("nan")
    return float(12.0 * np.log2(np.median(b) / np.median(a)))


def half_delta(row):
    """Median f0 of the second half minus the first half of a clip, in st."""
    row = np.asarray(row, dtype=np.float64)
    h = row.size // 2
    if h < 5:
        return float("nan")
    a, b = row[:h], row[h:]
    a, b = a[a > 0], b[b > 0]
    if a.size < 3 or b.size < 3:
        return float("nan")
    return float(12.0 * np.log2(np.median(b) / np.median(a)))


def anchor_level(batch):
    """Per-clip mean present anchor value in semitones, ``(B,)`` numpy."""
    val = batch["hint_pitch_val"]
    mask = anchor_mask(batch, int(val.shape[-1])).to(val.dtype)
    per_phone = (val * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)  # (B, N)
    n = batch["hint_pitch_n"].clamp(min=0)
    phone_ok = (n > 0).to(val.dtype)
    return ((per_phone * phone_ok).sum(-1) / phone_ok.sum(-1).clamp_min(1.0)
            ).detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# model driving
# --------------------------------------------------------------------------- #
def hint_arg(net, batch):
    allowed = set(net.null_hints(1, 1).keys())
    return {k: v for k, v in batch.items()
            if k in allowed and torch.is_tensor(v)}


@torch.no_grad()
def generate(net, batch, hints, frames, temp, top_k):
    torch.manual_seed(0)                  # paired draws across arms
    codes, lengths = net.generate(
        batch["phone_ids"], batch["phone_lengths"], batch["sid"],
        max_frames=frames, temperature=temp, top_k=top_k,
        return_lengths=True, hints=hints)
    return codes, lengths


@torch.no_grad()
def decode(net_g, codes, sid):
    """``(audio, f0)`` for a ``(B, K, T)`` code tensor."""
    n_spec = codes.shape[1] - 2
    spec = codes[:, :n_spec].permute(1, 0, 2).contiguous()
    pitch = codes[:, n_spec:].permute(1, 0, 2).contiguous()
    audio, f0 = net_g.infer_from_codes(spec, pitch, sid, return_pitch=True)
    return audio, f0


def f0_row_stats(row):
    """One clip's ``(median_hz, spread_st)`` over its voiced frames."""
    row = np.asarray(row, dtype=np.float64)
    v = row[row > 0]
    if v.size == 0:
        return (0.0, 0.0)
    med = float(np.median(v))
    return (med, float(np.std(12.0 * np.log2(v / med))))


def code_change(a, b, rows=slice(0, 8)):
    """Fraction of differing spec / pitch codes between two draws."""
    n = min(a.shape[-1], b.shape[-1])
    spec = float((a[:, rows, :n] != b[:, rows, :n]).float().mean())
    pit = float((a[:, 8:, :n] != b[:, 8:, :n]).float().mean())
    return spec, pit


def match(a, b, window=None):
    n = min(a.shape[-1], b.shape[-1])
    if window:
        n = min(n, window)
    return float((a[:, :n] == b[:, :n]).float().mean())


def slope_r2(x, y):
    """Least-squares slope/intercept/R^2 of ``y = a x + b``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(a), float(b), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--generator_checkpoint", default=None)
    ap.add_argument("--val_filelist", default=None)
    ap.add_argument("--num_items", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=1,
                    help="1 = greedy (cleanest slope estimate); 20 = the "
                         "sampling the repl uses")
    ap.add_argument("--deltas", default="-6,-4,-2,0,2,4,6")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip_audio", action="store_true",
                    help="CE arms only (no stage-1 decode)")
    ap.add_argument("--skip_semantic", action="store_true",
                    help="skip the semantic audio arms")
    ap.add_argument("--skip_global", action="store_true",
                    help="skip the (out-of-distribution) global shift sweep")
    args = ap.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(args.config)
    device = torch.device(args.device or
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))

    net = CodeTransformer(config).to(device)
    epoch, step = H.load_checkpoint(net, args.checkpoint)
    net.eval()
    print(f"checkpoint: {args.checkpoint}  (epoch {epoch}, step {step})")
    print(net.describe())
    if not net.hints:
        print("no hint tables -- nothing to probe")
        return 1
    anchors = getattr(net, "hint_pitch_mode", "table") == "anchors"
    print(f"pitch mode: {getattr(net, 'hint_pitch_mode', 'table')}"
          + (f" (A={net.hint_pitch_anchors})" if anchors else ""))

    ds = CodeDataset(args.val_filelist or config.train.val_filelist,
                     is_train=False,
                     max_frames=config.train.get("max_frames", None),
                     max_phone_length=config.train.get("max_phone_length", None),
                     speaker_map=config.train.get("speaker_map", None),
                     **H.hint_kwargs(config))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=CodeCollator(config=config), num_workers=0)
    batch = None
    for b in loader:
        b = H.to_device(b, device)
        if b.get("hint_int") is None:
            print("val items carry no .cond hints")
            return 1
        if anchors and b.get("hint_pitch_val") is None:
            print("val items carry no pitch anchors")
            return 1
        batch = b
        break
    B, N = int(batch["sid"].numel()), int(batch["phone_ids"].shape[1])
    if B > args.num_items:
        batch = {k: (v[:args.num_items] if torch.is_tensor(v) and
                     v.dim() and v.shape[0] == B else v)
                 for k, v in batch.items()}
        B = args.num_items
    print(f"{B} val item(s), {N} phoneme slots, "
          f"{int(batch['code_lengths'].min())}-{int(batch['code_lengths'].max())} "
          f"target code frames\n")

    # ---------------------------------------------------------------- CE arms
    print("=" * 78)
    print("teacher-forced CE (the channel's *predictive* value; not the knob)")
    print("=" * 78)

    @torch.no_grad()
    def ce(b):
        return float(net.compute_batch_loss(b, epoch=None,
                                            diagnostics=False)["loss"])

    base = ce(batch)
    print(f"  real hints                      {base:.4f}")
    ce_arms = []
    if anchors:
        ce_arms += [("pitch absent (state 0)", branch(batch, "absent")),
                    ("pitch +2 st", branch(batch, "shift", 2.0)),
                    ("pitch +4 st", branch(batch, "shift", 4.0)),
                    ("pitch flattened (1 anchor/word)", branch(batch, "flat")),
                    ("pitch swapped across clips", branch(batch, "swap"))]
    ce_arms += [("semantic absent (sem_null)", branch(batch, "sem_null")),
                ("semantic swapped across clips", branch(batch, "sem_swap")),
                ("semantic zero vector (valid)", branch(batch, "sem_zero"))]
    for label, b in ce_arms:
        v = ce(b)
        print(f"  {label:<32s} {v:.4f}   ({v - base:+.4f} vs real)")

    if args.skip_audio:
        return 0

    # ------------------------------------------------------------ audio arms
    gen_ckpt = (args.generator_checkpoint
                or config.train.get("generator_checkpoint", None))
    if not gen_ckpt:
        print("\nno stage-1 generator checkpoint (--generator_checkpoint or "
              "train.generator_checkpoint); skipping the audio KPI")
        return 0
    sys.path.insert(0, REPO)
    from importlib import util as _util
    spec = _util.spec_from_file_location("train2_kpi",
                                         os.path.join(REPO, "2_training.py"))
    mod = _util.module_from_spec(spec)
    sys.modules["train2_kpi"] = mod
    spec.loader.exec_module(mod)
    net_g = mod.load_generator(gen_ckpt, config, device)
    print(f"\nstage-1 generator: {gen_ckpt}")

    sid = batch["sid"]
    frames = args.frames
    arms = {}

    def run_arm(name, b):
        hints = hint_arg(net, b)
        codes, lengths = generate(net, b, hints, frames, args.temp, args.top_k)
        audio, f0 = decode(net_g, codes, sid)
        # ``f0`` covers the *batch's* width, which the sampler pads out to the
        # longest member (repeating that member's last frame).  Statistics on
        # the untrimmed row therefore depend on batch composition -- measured:
        # the same clip's second-half pitch moved by 4 st between a 6- and a
        # 16-item batch.  Trim every row to its own predicted frame count.
        arr = f0.detach().float().cpu().numpy()
        ratio = arr.shape[1] / max(int(codes.shape[-1]), 1)
        rows = [arr[i, :max(1, int(round(float(lengths[i]) * ratio)))]
                for i in range(arr.shape[0])]
        arms[name] = {"codes": codes, "lengths": lengths,
                      "f0": [f0_row_stats(r) for r in rows],
                      "f0_rows": rows,
                      "audio": audio.detach().float().cpu()}
        return arms[name]

    if anchors:
        print("\n" + "=" * 78)
        print(f"pitch: contour arms (greedy={args.top_k == 1}, "
              f"max_frames={frames}, same seed)")
        print("=" * 78)
        print("The anchors are semitones re: the utterance's own voiced median,")
        print("so a *global* shift is out of distribution (the absolute level")
        print("comes from the speaker id).  The meaningful edits are contour")
        print("edits: a local bump, a flattened melody, a time reversal.")

        run_arm("own", batch)
        for name, b in (("absent", branch(batch, "absent")),
                        ("flat", branch(batch, "flat")),
                        ("ramp_up", branch_contour(batch, "ramp_up")),
                        ("ramp_down", branch_contour(batch, "ramp_down")),
                        ("half_up", branch_contour(batch, "half_up")),
                        ("half_down", branch_contour(batch, "half_down")),
                        ("swap", branch(batch, "swap"))):
            run_arm(name, b)

        own = arms["own"]
        own_med = np.array([m for m, _ in own["f0"]])
        print(f"\n  {'arm':<22s} {'dF0 median st':>14s} {'spread st':>10s} "
              f"{'pitch-code chg':>15s} {'frames':>7s}")
        for name in ("absent", "flat", "ramp_up", "ramp_down", "half_up",
                     "half_down", "swap"):
            a = arms[name]
            med = np.array([m for m, _ in a["f0"]])
            spread = float(np.mean([s for _, s in a["f0"]]))
            diff = st(med, own_med)
            _, pit = code_change(a["codes"], own["codes"])
            print(f"  {name:<22s} {np.median(diff):>+14.2f} "
                  f"{spread:>10.2f} {pit:>15.3f} "
                  f"{float(a['lengths'].float().mean()):>7.1f}")
        print(f"  {'own':<22s} {0.0:>+14.2f} "
              f"{float(np.mean([s for _, s in own['f0']])):>10.2f}")

        # ---- contour control: melodic trend (alignment-free) --------------
        print("\n  melodic-trend test (a +-2 st ramp across the utterance;")
        print("  readout = f0 of the last third minus the first third, st):")
        for name in ("own", "ramp_up", "ramp_down"):
            vals = np.array([trend(arms[name]["f0_rows"][i]) for i in range(B)])
            print(f"    {name:<10s} trend = {np.nanmedian(vals):+6.2f} st")
        print("    ramp_up should be clearly positive and ramp_down clearly")
        print("    negative relative to own; both ~0 (or equal) means the")
        print("    anchors do not steer the melody, only its overall level.")

        # ---- local bump: second half of the phones up/down 3 st ----------
        print("\n  local-bump test (second half of the words +-3 st;")
        print("  readout = median f0 of the second half minus the first, st):")
        half = {}
        for name in ("own", "half_up", "half_down"):
            half[name] = np.array([half_delta(arms[name]["f0_rows"][i])
                                   for i in range(B)])
        for name in ("own", "half_up", "half_down"):
            print(f"    {name:<10s} dHalf = {np.nanmedian(half[name]):+6.2f}"
                  f"   per clip: "
                  + " ".join(f"{v:+.1f}" for v in half[name]))
        print(f"    requested -3 / +3 -> observed "
              f"{(np.nanmedian(half['half_down']) - np.nanmedian(half['own'])):+5.2f}"
              f" / {(np.nanmedian(half['half_up']) - np.nanmedian(half['own'])):+5.2f}"
              f" st")
        print("    (durations re-pace, so expect a fraction of 3, not 3.0; the")
        print("     sign and the up/down split are the signal)")

        # ---- global shift: reported as a diagnostic, not as the KPI -------
        deltas = [float(d) for d in str(args.deltas).split(",") if d.strip()]
        if args.skip_global:
            deltas = []
        for d in deltas:
            if d == 0.0:
                continue
            run_arm(f"shift{d:+.0f}", branch(batch, "shift", d))
        print("\n  global-shift diagnostic (out of distribution by design):")
        for d in deltas:
            if d == 0.0:
                print(f"    {d:+5.1f} st requested ->  0.00 st observed (baseline)")
                continue
            a = arms[f"shift{d:+.0f}"]
            diff = st(np.array([m for m, _ in a["f0"]]), own_med)
            print(f"    {d:+5.1f} st requested -> {np.median(diff):+6.2f} st "
                  f"observed")

        flat_spread = float(np.mean([s for _, s in arms["flat"]["f0"]]))
        own_spread = float(np.mean([s for _, s in own["f0"]]))
        print(f"\n  flat-melody test: within-clip f0 spread, own "
              f"{own_spread:.2f} st vs flattened {flat_spread:.2f} st "
              f"({100 * (flat_spread / own_spread - 1):+.0f}%)")
        print("    flattened keeps the level and drops the contour, so a")
        print("    meaningful drop means the anchors shape the melody.")

    if args.skip_semantic:
        return 0

    # ------------------------------------------------------- semantic arms
    print("\n" + "=" * 78)
    print("semantic: own vs null vs another clip's vector (paired seed)")
    print("=" * 78)
    for name, label in (("sem", "own"), ("sem_null", "null"),
                        ("sem_swap", "swapped"), ("sem_zero", "zero-vector")):
        run_arm(name, branch(batch, name) if name != "sem" else batch)
    own_s = arms["sem"]
    print(f"\n  free-running code agreement (%)")
    head = "  ".join(f"{'<' + str(w) if w else 'full':>10}" for w in WINDOWS)
    print(f"  {'pair':<20} {head}")
    for a, b in (("sem", "sem_null"), ("sem_swap", "sem_null"),
                 ("sem", "sem_swap")):
        vals = [match(arms[a]["codes"][i], arms[b]["codes"][i], w)
                for w in WINDOWS for i in range(B)]
        vals = [np.mean(vals[j * B:(j + 1) * B]) * 100
                for j in range(len(WINDOWS))]
        print(f"  {a + ' vs ' + b:<20} "
              + "  ".join(f"{v:>10.1f}" for v in vals))

    own_med_s = np.array([m for m, _ in own_s["f0"]])
    print(f"\n  {'arm':<14s} {'median f0 st':>13s} {'spread st':>10s} "
          f"{'dur s':>7s} {'energy cv':>10s}")
    sr = int(config.data.sampling_rate)
    for name in ("sem", "sem_null", "sem_swap", "sem_zero"):
        a = arms[name]
        med = np.array([m for m, _ in a["f0"]])
        spread = float(np.mean([s for _, s in a["f0"]]))
        wav = a["audio"]
        durs, cvs = [], []
        for row in wav:
            x = row.reshape(-1)
            durs.append(x.numel() / sr)
            nf = x.numel() // max(1, int(sr * 0.02))
            if nf:
                rms = x[:nf * int(sr * 0.02)].reshape(nf, -1).pow(2).mean(1).sqrt()
                cvs.append(float(rms.std() / rms.mean().clamp_min(1e-8)))
            else:
                cvs.append(0.0)
        print(f"  {name:<14s} {np.median(st(med, own_med_s)):>+13.2f} "
              f"{spread:>10.2f} {np.mean(durs):>7.2f} {np.mean(cvs):>10.3f}")
    print("\n  read: own-vs-null is 'is the channel live'; own-vs-swapped is "
          "'does the\n  vector's content matter'.  If swapped diverges from null "
          "as much as own\n  does, the vector is not being used as information.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
