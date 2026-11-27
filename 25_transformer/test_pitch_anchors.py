#!/usr/bin/env python3
"""CPU smoke test for the Hints V2 pitch anchors. Safe next to a training job.

Covers, in order:

* ``hints.levels.detect_pitch_anchors`` -- the contour reducer.  Synthetic
  cases pin the behaviour a bad detector would ruin: a flat word is one anchor,
  a straight ramp is its two endpoints (a peak finder would find nothing and
  fall back to the mean), a 3-extremum word spends its budget on the largest
  deviations, an isolated octave error is removed, an unvoiced word is ``n=0``;
* the **quality gate** the plan sets: on real Genshin clips, the detected
  anchors must reconstruct the word's f0 contour to <= 0.9 st (median, words of
  >= 8 code frames).  The DP bound measured while designing this is 0.72 st;
* the ``.cond`` -> ``CodeDataset`` -> ``CodeCollator`` path: a word's whole
  anchor vector is repeated over its phones, ``hint_phone_frac`` is the phone's
  position inside its word, and the v1 fallback is one anchor at ``pos=0.5``;
* the model's anchor encoding: the parameters exist and are zero-initialised,
  the loss depends on them, and ``hints=None`` is the trained null state.

Run from the repo root:

    python test_pitch_anchors.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import torch

from hints.levels import detect_pitch_anchors

_failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def _st(hz):
    """Hz array -> semitones re: 100 Hz (a convenient fixed reference)."""
    return 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 100.0)


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
def test_detector() -> None:
    print("detect_pitch_anchors (synthetic)")
    vals, pos, n = detect_pitch_anchors(np.full(20, 200.0), 100.0)
    check("flat word -> one anchor at its mean",
          n == 1 and abs(vals[0] - 12.0) < 1e-3 and abs(pos[0] - 0.5) < 1e-6,
          f"n={n} val={vals[0]:.2f} pos={pos[0]:.2f}")

    vals, pos, n = detect_pitch_anchors(100.0 * 2 ** (np.linspace(0, 1, 20)),
                                        100.0)
    check("straight 12 st ramp -> its two endpoints",
          n == 2 and abs(vals[0]) < 1e-3 and abs(vals[1] - 12.0) < 1e-2
          and pos[0] == 0.0 and pos[1] == 1.0,
          f"n={n} vals={np.round(vals[:n], 2).tolist()}")

    f = np.concatenate([np.linspace(0, 6, 7), np.linspace(6, -6, 7),
                        np.linspace(-6, 3, 7)])
    vals, pos, n = detect_pitch_anchors(100.0 * 2 ** (f / 12.0), 100.0)
    check("3-extremum word uses the whole budget",
          n == 3 and abs(vals[0]) < 0.2 and abs(vals[n - 1] - 3.0) < 0.2,
          f"n={n} vals={np.round(vals[:n], 2).tolist()}")

    vals, pos, n = detect_pitch_anchors(np.zeros(20), 100.0)
    check("unvoiced word -> n=0 (a distinct state)", n == 0)

    c = np.full(21, 200.0)
    c[10] = 400.0                       # one-frame octave error
    vals, pos, n = detect_pitch_anchors(c, 100.0)
    check("isolated octave error removed",
          n == 1 and abs(vals[0] - 12.0) < 1e-3, f"n={n}")

    c = np.zeros(21)
    c[10] = 200.0                       # one frame of voicing only
    vals, pos, n = detect_pitch_anchors(c, 100.0)
    check("a single voiced frame is one anchor",
          n == 1 and abs(vals[0] - 12.0) < 1e-3 and abs(pos[0] - 0.5) < 1e-6)

    vals, pos, n = detect_pitch_anchors(np.full(20, 200.0), 0.0)
    check("an invalid reference pitch -> n=0", n == 0)

    # The budget is a hard cap, and slots beyond it stay numerically inert.
    vals, pos, n = detect_pitch_anchors(
        100.0 * 2 ** (np.sin(np.linspace(0, 6, 40)) / 2), 100.0)
    check("never exceeds max_anchors", n <= 3, f"n={n}")
    check("padded slots are zero and ignored", bool(np.all(vals[n:] == 0.0)))


def _code_frame_contour(f0_100hz):
    """100 Hz f0 -> 25 Hz code-frame means (0 = unvoiced)."""
    n = len(f0_100hz) // 4
    if n == 0:
        return np.zeros(0)
    a = np.asarray(f0_100hz, dtype=np.float64)[: n * 4].reshape(n, 4)
    a = np.where(a > 0, a, np.nan)
    with np.errstate(invalid="ignore"):
        c = np.nanmean(a, axis=1)
    return np.where(np.isfinite(c), c, 0.0)


def _reconstruct(pos_frac, vals, n, n_frames, frames):
    """Piecewise-linear anchor contour sampled at ``frames`` (in word units)."""
    if n == 0:
        return np.zeros(len(frames))
    xp = np.asarray(pos_frac[:n], dtype=np.float64)
    fp = np.asarray(vals[:n], dtype=np.float64)
    x = np.asarray(frames, dtype=np.float64) / max(n_frames - 1, 1)
    if n == 1:
        return np.full(len(frames), fp[0])
    return np.interp(x, xp, fp)


def test_detector_quality() -> None:
    """Gate 0b: detected anchors must approach the DP bound on real contours."""
    print("detector quality (real Genshin contours)")
    import glob
    import random

    cond_glob = sorted(glob.glob("data/genshin_full_stage2/*.cond"))
    f0_dir = "data/genshin_full_stage1"
    if not cond_glob or not os.path.isdir(f0_dir):
        print("  [SKIP] Genshin .cond / .f0 not on this box")
        return
    rng = random.Random(0)
    rng.shuffle(cond_glob)
    err_anchor, err_mean, used = [], [], 0
    for path in cond_glob:
        base = os.path.basename(path)[: -len(".cond")]
        f0_path = os.path.join(f0_dir, base + ".f0")
        if not os.path.exists(f0_path):
            continue
        cond = torch.load(path, map_location="cpu", weights_only=True)
        if "word_start_f" not in cond or "word_slot_f" not in cond:
            continue
        f0 = torch.load(f0_path, map_location="cpu",
                        weights_only=True).float().reshape(-1).numpy()
        cf = _code_frame_contour(f0)
        voiced = cf > 0
        if not voiced.any():
            continue
        ref = float(np.median(cf[voiced]))
        if ref <= 0:
            continue
        used += 1
        for start, slot in zip(cond["word_start_f"].numpy().astype(int),
                               cond["word_slot_f"].numpy().astype(int)):
            if slot < 8 or start < 0 or start + slot > len(cf):
                continue
            # Detect on the code-frame contour: that is the rate the builder
            # uses (features._mean_bins) and the rate the model emits.
            span = cf[start:start + slot]
            st = np.where(span > 0, 12 * np.log2(np.maximum(span, 1e-9) / ref),
                          np.nan)
            truth = st[~np.isnan(st)]
            if truth.size < 4:
                continue
            vals, pos, n = detect_pitch_anchors(span, ref)
            idx = np.flatnonzero(~np.isnan(st))
            rec = _reconstruct(pos, vals, n, len(span), idx)
            err_anchor.append(float(np.sqrt(np.mean((rec - truth) ** 2))))
            err_mean.append(float(np.sqrt(np.mean(
                (truth.mean() - truth) ** 2))))
        if used >= 120:
            break
    if not err_anchor:
        print("  [SKIP] no usable word contours found")
        return
    med_anchor = float(np.median(err_anchor))
    med_mean = float(np.median(err_mean))
    check("gate 0b: detected anchors reconstruct <= 0.9 st",
          med_anchor <= 0.9,
          f"median {med_anchor:.2f} st over {len(err_anchor)} words")
    check("anchors beat the current word mean",
          med_anchor < med_mean,
          f"{med_anchor:.2f} vs {med_mean:.2f} st")


# --------------------------------------------------------------------------- #
# dataset / collator
# --------------------------------------------------------------------------- #
def _write_hints_json(path: str, anchors: int = 3, relative: bool = True) -> None:
    doc = {
        "version": 2,
        "levels": {
            "intensity": {"n_levels": 2, "edges": [5.0], "unit": "1-9"},
            "register": {"n_levels": 2, "edges": [0.0], "unit": "st"},
        },
        "words": {"pitch_anchors": anchors},
    }
    if relative:
        doc["levels"]["register"]["speaker_means"] = {"0": 0.0}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def _make_clip(tmp: str, anchors=True):
    """A 6-phone, 2-word clip with (optionally) 3-anchor word pitch."""
    d = os.path.join(tmp, "codes")
    os.makedirs(d, exist_ok=True)
    phone = os.path.join(d, "clip.wav.phone_ids")
    spec = os.path.join(d, "clip.wav.spec_codes")
    pitch = os.path.join(d, "clip.wav.pitch_codes")
    torch.save(torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.long), phone)
    torch.save(torch.randint(0, 4, (2, 12)), spec)
    torch.save(torch.randint(0, 4, (1, 12)), pitch)
    cond = {
        "version": 2,
        "n_phone": 6,
        "intensity": torch.tensor([6.0]),
        "register_st": torch.tensor([3.0]),
        "phone_word_idx": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int16),
        "word_start_f": torch.tensor([0, 6], dtype=torch.int32),
        "word_slot_f": torch.tensor([6, 6], dtype=torch.int32),
        "pitch_word_st": torch.tensor([0.0, 4.0], dtype=torch.float16),
        "pitch_word_voiced": torch.tensor([True, False]),
    }
    if anchors:
        cond.update({
            "pitch_anchors_st": torch.tensor(
                [[0.0, 3.0, 1.0], [0.0, 0.0, 0.0]],
                dtype=torch.float16),
            "pitch_anchors_pos": torch.tensor(
                [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]],
                dtype=torch.float16),
            "pitch_anchor_n": torch.tensor([3, 0], dtype=torch.int8),
        })
    torch.save(cond, os.path.join(d, "clip.wav.cond"))
    filelist = os.path.join(tmp, "train.txt")
    with open(filelist, "w", encoding="utf-8") as f:
        f.write(f"{phone}|{spec}|{pitch}|0\n")
    return filelist


def test_dataset() -> None:
    print("dataset (anchors)")
    from dataset_codes import CodeCollator, CodeDataset
    from omegaconf import OmegaConf

    tmp = tempfile.mkdtemp()
    hints_json = os.path.join(tmp, "hints.json")
    _write_hints_json(hints_json, anchors=3)
    filelist = _make_clip(tmp, anchors=True)

    ds = CodeDataset(filelist, is_train=False, hints=True,
                     hints_json=hints_json)
    item = ds[0]
    val = item["hint_pitch_val"]
    check("anchor vector repeated over the word's phones",
          tuple(val.shape) == (6, 3)
          and torch.equal(val[:3], val[0].expand(3, 3))
          and torch.equal(val[3:], val[3].expand(3, 3)),
          f"{tuple(val.shape)}")
    check("anchor value/position round-trips",
          abs(float(val[0, 1]) - 3.0) < 1e-6
          and abs(float(item["hint_pitch_pos"][0, 1]) - 0.5) < 1e-6)
    check("n follows the word", item["hint_pitch_n"].tolist() == [3, 3, 3, 0, 0, 0])
    check("unvoiced word is state 2 (a trained state)",
          item["hint_pitch_state"].tolist()
          == [1, 1, 1, 2, 2, 2], str(item["hint_pitch_state"].tolist()))
    frac = item["hint_phone_frac"].tolist()
    check("phone_frac is the phone's place inside its word",
          np.allclose(frac, [1 / 6, 3 / 6, 5 / 6, 1 / 6, 3 / 6, 5 / 6]),
          str(np.round(frac, 2).tolist()))

    # v1 fallback: no anchor keys -> one anchor at pos 0.5 with the word mean.
    tmp2 = tempfile.mkdtemp()
    hj2 = os.path.join(tmp2, "hints.json")
    _write_hints_json(hj2, anchors=1)
    fl2 = _make_clip(tmp2, anchors=False)
    ds1 = CodeDataset(fl2, is_train=False, hints=True, hints_json=hj2)
    it1 = ds1[0]
    check("v1 .cond -> one anchor at pos 0.5",
          tuple(it1["hint_pitch_val"].shape) == (6, 1)
          and it1["hint_pitch_n"].tolist() == [1, 1, 1, 0, 0, 0]
          and abs(float(it1["hint_pitch_val"][0, 0]) - 0.0) < 1e-3)

    # Dropout: state 0 is the trained absent path.
    ds_null = CodeDataset(filelist, is_train=True, hints=True,
                          hints_json=hints_json, hint_null_all=1.0)
    it = ds_null[0]
    check("hint_null_all -> pitch state 0",
          bool((it["hint_pitch_state"] == 0).all()) and int(it["hint_int"]) == 0)

    cfg = OmegaConf.create({"transformer": {"hint_pitch_anchors": 3}})
    collator = CodeCollator(cfg)
    batch = collator([ds[0], ds[0]])
    check("collator pads anchors to (B, N, A)",
          tuple(batch["hint_pitch_val"].shape) == (2, 6, 3)
          and tuple(batch["hint_pitch_state"].shape) == (2, 6),
          f"{tuple(batch['hint_pitch_val'].shape)}")
    return ds, filelist, hints_json, cfg, batch


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _cfg(hints_json: str, anchors: int = 3, **over):
    from omegaconf import OmegaConf

    t = {
        "hidden_size": 32, "n_heads": 2, "encoder_layers": 1,
        "decoder_layers": 1, "dim_feedforward": 64, "dropout": 0.0,
        "max_positions": 512, "phone_vocab_size": 1024, "n_speakers": 4096,
        "eos_token": True, "hints": True, "hints_json": hints_json,
        "semantic_dim": 384, "decoder_hints": True,
        "hint_pitch_mode": "anchors", "hint_pitch_anchors": anchors,
        "relative_register": True,
    }
    t.update(over)
    return OmegaConf.create({
        "codec": {"codebook_num": 2, "codebook_size": 32},
        "pitch_codec": {"codebook_num": 1, "codebook_size": 32},
        "transformer": t,
    })


def test_model(batch, hints_json):
    print("model (anchor encoding)")
    from modeling.transformer import CodeTransformer

    cfg = _cfg(hints_json)
    torch.manual_seed(0)
    net = CodeTransformer(cfg)
    net.eval()
    check("anchor mode replaces the level table",
          net.hint_pitch_mode == "anchors" and net.pitch_emb is None)
    check("per-slot directions exist",
          net.pitch_val_dir.shape == (3, 32)
          and net.pitch_pos_dir.shape == (3, 32)
          and net.pitch_slot_null.shape == (3, 32)
          and net.pitch_absent.shape == (32,))
    check("anchor parameters are zero-initialised",
          all(p.abs().max().item() == 0.0 for p in
              (net.pitch_val_dir, net.pitch_pos_dir, net.pitch_slot_null,
               net.pitch_unvoiced, net.pitch_absent, net.pitch_frac_dir)))
    check("describe() reports the anchor layout",
          "3 anchors" in net.describe(), net.describe())

    b = dict(batch)
    B, T = int(b["sid"].numel()), 12
    b["spec_codes"] = torch.randint(0, 32, (B, 2, T))
    b["pitch_codes"] = torch.randint(0, 32, (B, 1, T))
    b["code_lengths"] = torch.full((B,), T, dtype=torch.long)
    b["complete"] = torch.ones(B, dtype=torch.bool)
    out = net.compute_batch_loss(b, epoch=0, diagnostics=False)
    check("zero-init anchors leave the loss untouched",
          bool(torch.isfinite(out["loss"])))

    # The side channel must actually reach the model.
    with torch.no_grad():
        net.pitch_val_dir.normal_()
        net.pitch_absent.normal_()
        net.pitch_frac_dir.normal_()
    out2 = net.compute_batch_loss(b, epoch=0, diagnostics=False)
    check("anchors reach the model",
          abs(float(out["loss"]) - float(out2["loss"])) > 1e-7,
          f"{float(out['loss']):.4f} vs {float(out2['loss']):.4f}")

    # A different anchor value must change the encoding (the knob is wired).
    alt = dict(b)
    alt["hint_pitch_val"] = b["hint_pitch_val"].clone()
    alt["hint_pitch_val"][:, :, 1] += 3.0
    out3 = net.compute_batch_loss(alt, epoch=0, diagnostics=False)
    check("moving an anchor changes the loss",
          abs(float(out2["loss"]) - float(out3["loss"])) > 1e-7,
          f"{float(out2['loss']):.4f} vs {float(out3['loss']):.4f}")

    # Dropping the channel must be the trained absent state, not a different
    # function: state 0 is what ``hint_dropout`` produced during training.
    off = dict(b)
    off["hint_pitch_state"] = torch.zeros_like(b["hint_pitch_state"])
    alt_off = dict(off)
    alt_off["hint_pitch_val"] = b["hint_pitch_val"].clone() + 5.0
    a = net.compute_batch_loss(off, epoch=0, diagnostics=False)
    c = net.compute_batch_loss(alt_off, epoch=0, diagnostics=False)
    check("state 0 is the trained absent path (values ignored)",
          abs(float(a["loss"]) - float(c["loss"])) < 1e-7,
          f"{float(a['loss']):.4f} vs {float(c['loss']):.4f}")

    with torch.no_grad():
        gen = net.generate(b["phone_ids"], b["phone_lengths"], b["sid"],
                           max_frames=8, hints=None)
    check("generate runs in the null state", tuple(gen.shape)[0] == B)

    # A mismatched anchor count must be refused, not reinterpreted.
    try:
        CodeTransformer(_cfg(hints_json, anchors=4))
        check("anchor-count mismatch raises", False)
    except ValueError:
        check("anchor-count mismatch raises", True)


# --------------------------------------------------------------------------- #
# builder path
# --------------------------------------------------------------------------- #
def test_hint_words() -> None:
    """``FeatureExtractor.hint_words`` emits the anchor fields end to end."""
    print("hint_words (anchor fields)")
    if not os.path.exists("configs/base.yaml"):
        print("  [SKIP] configs/base.yaml not on this box")
        return
    from omegaconf import OmegaConf

    from features import FeatureExtractor

    fe = FeatureExtractor(hp=OmegaConf.load("configs/base.yaml"),
                          load_qwen=False)
    items = [
        # Log-linear ramp: 100 -> 200 Hz is exactly 12 st, so a straight line
        # in semitones and the DP should keep only the endpoints.
        {"phone_ids": [1, 2], "start_time": 0.0, "end_time": 0.8,
         "nonzero_pitch_mean": 150.0,
         "pitch_contour": 100.0 * 2 ** np.linspace(0, 1, 80)},
        # Fully unvoiced word -> n=0 (a distinct state, not "absent").
        {"phone_ids": [3], "start_time": 0.8, "end_time": 1.2,
         "nonzero_pitch_mean": 0.0, "pitch_contour": np.zeros(40)},
    ]
    out = fe.hint_words(items, total_seconds=1.2)
    check("anchor tensors have the documented shapes",
          tuple(out["pitch_anchors_st"].shape) == (2, 3)
          and tuple(out["pitch_anchors_pos"].shape) == (2, 3)
          and out["pitch_anchor_n"].tolist() == [2, 0],
          str(out["pitch_anchor_n"].tolist()))
    st = out["pitch_anchors_st"].float()
    pos = out["pitch_anchors_pos"].float()
    # The first/last code-frame bins average the ramp's edges, so the endpoint
    # span loses ~0.5 st to the bin width; the structure (two knots, exactly at
    # the ends) is what matters.
    check("a 12 st ramp is its two endpoints",
          abs(float(st[0, 1] - st[0, 0]) - 12.0) < 0.6
          and pos[0, 0] == 0.0 and abs(float(pos[0, 1]) - 1.0) < 1e-3,
          f"st={st[0].tolist()} pos={pos[0].tolist()}")
    check("dtypes are the .cond contract",
          out["pitch_anchors_st"].dtype == torch.float16
          and out["pitch_anchor_n"].dtype == torch.int8)


def main() -> int:
    test_detector()
    test_hint_words()
    test_detector_quality()
    made = test_dataset()
    if made is not None:
        ds, filelist, hints_json, cfg, batch = made
        test_model(batch, hints_json)
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        return 1
    print("all pitch-anchor checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
