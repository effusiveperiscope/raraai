#!/usr/bin/env python3
"""CPU smoke test for the conditioning-hint stack. Safe next to a training job.

Covers the pieces that need no GPU, no Qwen aligner and no RMVPE:

* ``hints.levels.bucket_levels`` -- continuous value -> level id, NaN -> 0;
* ``FeatureExtractor.hint_words`` on synthetic aligned items -- start-to-start
  slots partition the clip, pitch is relative to the utterance, and phones map
  to their word;
* ``FeatureExtractor.extract_hints`` vs ``hints/build.py`` on the same clip --
  the one-stop shop and the bulk builder must produce identical numbers, or a
  UI-built ``.cond`` and an offline one would mean different things;
* the ``.cond`` round trip through ``torch.save`` / ``weights_only=True`` load.

Run from the repo root:

    python test_hints.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import torch

import hints.build as hb
from hints.levels import bucket_levels, load_level_edges

WAVE = "data/genshin_full_stage1/Venti_English__66306.wav.wave"
TEXT = "Hydrology research."

_failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def test_levels() -> None:
    print("levels")
    edges = [1.0, 2.0, 3.0]
    check("scalar below first edge -> 1", bucket_levels(0.0, edges) == 1)
    check("scalar above last edge -> N", bucket_levels(9.9, edges) == 4)
    check("scalar on an edge -> upper bin", bucket_levels(2.0, edges) == 3)
    check("NaN -> 0 (absent)", bucket_levels(float("nan"), edges) == 0)
    arr = bucket_levels([-1.0, 1.5, 2.5, float("nan"), float("inf")], edges)
    check("array mapping", arr.tolist() == [1, 2, 3, 0, 0], str(arr.tolist()))
    try:
        bucket_levels(0.0, None)
        check("missing edges raise", False)
    except ValueError:
        check("missing edges raise", True)


def test_hint_words(fe) -> None:
    print("hint_words (synthetic alignment, no aligner)")
    items = [
        {"phone_ids": [11, 12], "start_time": 0.00, "end_time": 0.40,
         "nonzero_pitch_mean": 100.0},
        {"phone_ids": [13], "start_time": 0.40, "end_time": 0.80,
         "nonzero_pitch_mean": 200.0},
        {"phone_ids": [14, 15, 16], "start_time": 0.80, "end_time": 1.60,
         "nonzero_pitch_mean": 0.0},          # fully unvoiced word
    ]
    out = fe.hint_words(items, total_seconds=2.0)
    slots = out["word_slot_f"].tolist()
    check("slots are start-to-start @25 fps", slots == [10, 10, 30], str(slots))
    check("slots partition the clip",
          sum(slots) == int(round(2.0 * 25.0)), f"sum={sum(slots)}")
    idx = out["phone_word_idx"].tolist()
    check("phones map to their word", idx == [0, 0, 1, 2, 2, 2], str(idx))
    check("phone_word_idx dtype", out["phone_word_idx"].dtype == torch.int16)
    st = out["pitch_word_st"].to(torch.float64).tolist()
    ref = 150.0                                  # median of the voiced means
    want = [12 * math.log2(100.0 / ref), 12 * math.log2(200.0 / ref)]
    check("pitch is relative to the utterance",
          abs(st[0] - want[0]) < 1e-2 and abs(st[1] - want[1]) < 1e-2,
          f"{st[0]:.2f}, {st[1]:.2f} st")
    check("unvoiced word is NaN + flagged",
          math.isnan(st[2])
          and out["pitch_word_voiced"].tolist() == [True, True, False])
    check("empty items -> {}", fe.hint_words([]) == {})
    try:
        fe.hint_words([{"phone_ids": [1], "start_time": 1.0, "end_time": 1.2},
                       {"phone_ids": [2], "start_time": 0.0, "end_time": 0.2}])
        check("non-monotonic starts raise", False)
    except ValueError:
        check("non-monotonic starts raise", True)


def test_one_stop(fe) -> None:
    print("extract_hints vs hints/build.py (same clip)")
    if not os.path.exists(WAVE):
        print(f"  [SKIP] {WAVE} not present")
        return
    wave = hb.load_audio_48k(WAVE)
    rec = fe.extract_hints(wave, TEXT, register=False)
    offline = hb.intensity_from_wave(WAVE)
    check("intensity agrees with the bulk builder",
          abs(float(rec["intensity"]) - offline) < 1e-6,
          f"{float(rec['intensity']):.4f} vs {offline:.4f}")
    sem_offline = hb.encode_semantics([TEXT])[0]
    check("semantic agrees with the bulk builder",
          torch.equal(rec["semantic"], sem_offline))
    check("semantic is 384-d float16 L2-normalised",
          tuple(rec["semantic"].shape) == (384,)
          and rec["semantic"].dtype == torch.float16
          and abs(float(rec["semantic"].float().norm()) - 1.0) < 1e-2)
    no_text = fe.extract_hints(wave, None, register=False)
    check("no transcript -> no semantic", "semantic" not in no_text)
    check("register=False -> no register", "register_st" not in no_text)

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "clip.cond")
    fe.write_hints(path, rec)
    back = torch.load(path, map_location="cpu", weights_only=True)
    check(".cond round trip", set(back) == set(rec)
          and torch.equal(back["intensity"], rec["intensity"]))


def test_hints_json() -> None:
    print("hints.json edges")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "hints.json")
    edges, stats = hb.quantile_edges(list(np.linspace(1, 9, 100)), 8)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"levels": {"intensity": {"n_levels": 8, "edges": edges,
                                            "stats": stats}}}, f)
    got_edges, n = load_level_edges(path, "intensity")
    check("edges round trip", len(got_edges) == 7 and n == 8)
    check("edges are ascending",
          all(b >= a for a, b in zip(got_edges, got_edges[1:])))
    ids = {bucket_levels(v, got_edges) for v in np.linspace(1, 9, 100, endpoint=False)}
    check("level ids cover 1..8", ids == set(range(1, 9)), sorted(ids))
    try:
        load_level_edges(path, "register")
        check("missing hint raises", False)
    except ValueError:
        check("missing hint raises", True)


def main() -> int:
    from omegaconf import OmegaConf

    from features import FeatureExtractor

    fe = FeatureExtractor(hp=OmegaConf.load("configs/base.yaml"),
                          device="cpu", load_qwen=False)
    check("code frame rate", abs(fe.code_frame_rate() - 25.0) < 1e-9,
          f"{fe.code_frame_rate():.3f} Hz")
    test_levels()
    test_hint_words(fe)
    test_one_stop(fe)
    test_hints_json()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("all hint checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

