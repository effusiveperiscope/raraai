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


#: A stage-2 filelist with real `.phone_ids`/`.spec_codes`/`.pitch_codes`.
STAGE2_FILELIST = "data/genshin_plus_extras/train_stage2.txt"


def _make_fixture(tmp: str, n_clips: int = 12):
    """Copy a few real stage-2 clips into ``tmp`` and build their ``.cond``.

    Runs the real builder (in-process) over the copies, so this exercises the
    full ``hints/build.py`` -> ``.cond`` -> ``CodeDataset`` path rather than a
    hand-written tensor.
    """
    import shutil

    if not os.path.exists(STAGE2_FILELIST):
        return None
    picked = []
    with open(STAGE2_FILELIST, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 4:
                continue
            if all(os.path.exists(p) for p in parts[:3]):
                picked.append(parts)
            if len(picked) >= n_clips:
                break
    if len(picked) < 2:
        return None

    code_dir = os.path.join(tmp, "codes")
    os.makedirs(code_dir, exist_ok=True)
    lines = []
    for parts in picked:
        names = []
        for path in parts[:3]:
            dst = os.path.join(code_dir, os.path.basename(path))
            shutil.copyfile(path, dst)
            names.append(dst)
        lines.append("|".join(names + [parts[3]]))
    filelist = os.path.join(tmp, "train.txt")
    with open(filelist, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    hints_json = os.path.join(tmp, "hints.json")
    rc = hb.main([
        "--filelist", filelist,
        "--wave-dir", "data/genshin_full_stage1",
        "--source-filelist", "filelists_2/genshin_full_eng_text_linux.txt",
        "--hints", "intensity,register,semantic",
        "--workers", "1",
        # Fewer bins than production (8) so a 12-clip fixture can still fit
        # equal-frequency edges; the dataset/model only care that the count came
        # from this file.
        "--n-intensity-levels", "4",
        "--n-register-levels", "4",
        "--hints-json", hints_json,
    ])
    if rc != 0 or not os.path.exists(hints_json):
        return None
    return filelist, hints_json


def test_hints_dataset():
    """``CodeDataset``/``CodeCollator``: bucketing, dropout, padding."""
    print("dataset + collator (hints)")
    tmp = tempfile.mkdtemp()
    made = _make_fixture(tmp)
    if made is None:
        print("  [SKIP] no usable stage-2 clips on this box")
        return None
    filelist, hints_json = made

    from dataset_codes import CodeCollator, CodeDataset

    ds = CodeDataset(filelist, is_train=False, hints=True, hints_json=hints_json)
    check("level counts come from hints.json",
          ds.n_intensity_levels > 1 and ds.n_register_levels > 1,
          f"intensity={ds.n_intensity_levels} register={ds.n_register_levels}")
    check("pitch table = bins + unvoiced",
          ds.n_pitch_levels == len(ds.pitch_edges) + 1)
    item = ds[0]
    check("intensity bucketed", int(item["hint_int"]) >= 1,
          str(int(item["hint_int"])))
    check("register bucketed", int(item["hint_reg"]) >= 1,
          str(int(item["hint_reg"])))
    check("semantic present", tuple(item["hint_sem"].shape) == (384,)
          and bool(item["hint_sem_valid"]))
    check("no dropout on validation items",
          all(int(ds[i]["hint_int"]) >= 1 for i in range(len(ds))))

    ds_null = CodeDataset(filelist, is_train=True, hints=True,
                          hints_json=hints_json, hint_null_all=1.0)
    it = ds_null[0]
    check("hint_null_all drops every channel",
          int(it["hint_int"]) == 0 and int(it["hint_reg"]) == 0
          and not bool(it["hint_sem_valid"]))

    ds_drop = CodeDataset(filelist, is_train=True, hints=True,
                          hints_json=hints_json,
                          hint_dropout={"intensity": 1.0})
    it = ds_drop[0]
    check("per-channel dropout is independent",
          int(it["hint_int"]) == 0 and bool(it["hint_sem_valid"]))

    check("hints off -> no hint keys", "hint_int" not in CodeDataset(
        filelist, is_train=False)[0])

    collator = CodeCollator()
    batch = collator([ds[i] for i in range(min(3, len(ds)))])
    check("collator pads per-phone hints like phone_ids",
          tuple(batch["hint_pitch"].shape) == tuple(batch["phone_ids"].shape),
          f"{tuple(batch['hint_pitch'].shape)}")
    check("collator stacks clip-level hints",
          tuple(batch["hint_int"].shape) == (len(batch["sid"]),)
          and tuple(batch["hint_sem"].shape) == (len(batch["sid"]), 384)
          and batch["hint_sem_valid"].dtype == torch.bool)
    return ds, filelist, hints_json, batch


def test_word_hints_in_dataset(ds, filelist, hints_json):
    """Per-word pitch/duration: gathered to phones, stale file rejected."""
    print("dataset (per-word hints)")
    from dataset_codes import CodeDataset, cond_path_for
    from hints.levels import slot_bucket, word_pitch_levels

    path = ds.items[0]["phone_ids"]
    n_phone = int(torch.load(path, weights_only=True).numel())
    cpath = cond_path_for(path)
    cond = torch.load(cpath, map_location="cpu", weights_only=True)
    n_word = 3
    per = max(1, (n_phone + n_word - 1) // n_word)
    word = (torch.arange(n_phone) // per).clamp(max=n_word - 1).to(torch.int16)
    pitch_st = torch.tensor([3.0, -5.0, 3.0], dtype=torch.float16)
    voiced = torch.tensor([True, False, True])
    slots = torch.tensor([4, 100, 2], dtype=torch.int32)
    cond.update({"n_phone": n_phone, "phone_word_idx": word,
                 "pitch_word_st": pitch_st, "pitch_word_voiced": voiced,
                 "word_slot_f": slots})
    hb.write_cond(cpath, cond)

    ds_w = CodeDataset(filelist, is_train=False, hints=True,
                       hints_json=hints_json)
    item = ds_w[0]
    want_pitch = word_pitch_levels(pitch_st.float(), voiced,
                                   ds.pitch_edges)[word.numpy()]
    want_slot = slot_bucket(slots.float(), ds.n_slot_buckets)[word.numpy()]
    check("pitch level gathered from the word",
          item["hint_pitch"].tolist() == want_pitch.tolist(),
          f"{item['hint_pitch'].tolist()} vs {want_pitch.tolist()}")
    check("unvoiced word gets its own level",
          (len(ds.pitch_edges) + 1) in want_pitch.tolist())
    check("slot level gathered from the word",
          item["hint_slot"].tolist() == want_slot.tolist(),
          f"{item['hint_slot'].tolist()}")

    stale = dict(cond)
    stale["n_phone"] = n_phone + 1
    hb.write_cond(cpath, stale)
    ds_bad = CodeDataset(filelist, is_train=False, hints=True,
                         hints_json=hints_json)
    item = ds_bad[0]
    check("stale .cond rejected, clip-level hints kept",
          item["hint_pitch"] is None and int(item["hint_int"]) >= 1
          and ds_bad.n_cond_stale >= 1)


def test_model(batch, hints_json):
    """The model consumes the hint tensors, and they change the loss."""
    print("model (hint side channels)")
    from omegaconf import OmegaConf

    from modeling.transformer import CodeTransformer

    cfg = OmegaConf.create({
        "codec": {"codebook_num": 2, "codebook_size": 32},
        "pitch_codec": {"codebook_num": 1, "codebook_size": 32},
        "transformer": {
            "hidden_size": 32, "n_heads": 2, "encoder_layers": 1,
            "decoder_layers": 1, "dim_feedforward": 64, "dropout": 0.0,
            "max_positions": 512, "phone_vocab_size": 1024, "n_speakers": 4096,
            "eos_token": True, "fim": True, "region_cond": True,
            "length_cond": True, "hints": True, "hints_json": hints_json,
            "semantic_dim": 384, "decoder_hints": True,
        },
    })
    torch.manual_seed(0)
    net = CodeTransformer(cfg)
    net.eval()
    check("describe() reports the hint layout",
          "hints=" in net.describe() and "hints=off" not in net.describe())
    check("tables sized from hints.json",
          net.int_emb.num_embeddings == net.n_intensity_levels + 1
          and net.pitch_emb.num_embeddings == net.n_pitch_levels + 2
          and net.slot_emb.num_embeddings == net.n_slot_buckets + 1)
    check("decoder-side table present", net.dec_int_emb is not None)

    b = dict(batch)
    B, T = int(b["sid"].numel()), 12
    b["spec_codes"] = torch.randint(0, 32, (B, 2, T))
    b["pitch_codes"] = torch.randint(0, 32, (B, 1, T))
    b["code_lengths"] = torch.full((B,), T, dtype=torch.long)
    b["complete"] = torch.ones(B, dtype=torch.bool)
    b["fim"] = torch.zeros(B, dtype=torch.bool)
    out = net.compute_batch_loss(b, epoch=0, diagnostics=True)
    check("loss with hints is finite", bool(torch.isfinite(out["loss"])))

    off = dict(b)
    for key in ("hint_int", "hint_reg", "hint_pitch", "hint_slot",
                "hint_sem_valid"):
        off[key] = torch.zeros_like(b[key])
    out_off = net.compute_batch_loss(off, epoch=0, diagnostics=False)
    check("hints reach the model",
          abs(float(out["loss"]) - float(out_off["loss"])) > 1e-7,
          f"{float(out['loss']):.4f} with vs {float(out_off['loss']):.4f} without")

    codes, lengths = net.generate(b["phone_ids"], b["phone_lengths"], b["sid"],
                                  max_frames=6, hints=b, return_lengths=True)
    check("generate with hints",
          tuple(codes.shape[:2]) == (B, 3)
          and codes.shape[2] == int(lengths.max()),
          f"{tuple(codes.shape)}")
    codes_all = torch.cat([b["spec_codes"], b["pitch_codes"]], dim=1)
    mid = net.infill(b["phone_ids"], b["phone_lengths"], b["sid"],
                     codes_all[:, :, :4], codes_all[:, :, 8:],
                     middle_frames=3, hints=b)
    check("infill with hints", tuple(mid.shape) == (B, 3, 3), f"{tuple(mid.shape)}")


def test_word_index(fe):
    """``word_index_for_phones``: maps a transcript-path corpus's phones."""
    print("word_index_for_phones")
    rows = []
    with open(STAGE2_FILELIST, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 4 and parts[0].endswith(".phone_ids"):
                rows.append(parts)
                break
    if not rows:
        print("  [SKIP] no stage-2 clip")
        return
    import os

    base = os.path.basename(rows[0][0])[: -len(".phone_ids")]
    transcripts = {}
    for line in open("filelists_2/genshin_full_eng_text_linux.txt",
                     encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) >= 3:
            transcripts[os.path.basename(p[0])] = p[2]
    if base not in transcripts:
        print("  [SKIP] no transcript for the first clip")
        return
    text = transcripts[base]
    stored = torch.load(rows[0][0], weights_only=True).reshape(-1)
    words = text.split()
    items = [{"text": w, "start_time": 0.2 * i, "end_time": 0.2 * (i + 1)}
             for i, w in enumerate(words)]

    idx = fe.word_index_for_phones(stored, items, text=text)
    check("maps every stored phone",
          idx is not None and len(idx) == int(stored.numel()),
          f"{None if idx is None else len(idx)} vs {int(stored.numel())}")
    if idx is None:
        return
    check("word index is non-decreasing",
          all(b >= a for a, b in zip(idx, idx[1:])))
    check("uses every word", set(idx) == set(range(len(words))))
    check("stale phone_ids rejected",
          fe.word_index_for_phones(stored[:-1], items, text=text) is None)
    check("a word-count mismatch is rejected",
          fe.word_index_for_phones(stored, items[:-1], text=text) is None)


def main() -> int:
    from omegaconf import OmegaConf

    from features import FeatureExtractor

    fe = FeatureExtractor(hp=OmegaConf.load("configs/base.yaml"),
                          device="cpu", load_qwen=False)
    check("code frame rate", abs(fe.code_frame_rate() - 25.0) < 1e-9,
          f"{fe.code_frame_rate():.3f} Hz")
    test_levels()
    test_hint_words(fe)
    test_word_index(fe)
    test_one_stop(fe)
    test_hints_json()
    made = test_hints_dataset()
    if made is not None:
        ds, filelist, hints_json, batch = made
        test_word_hints_in_dataset(ds, filelist, hints_json)
        test_model(batch, hints_json)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("all hint checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

