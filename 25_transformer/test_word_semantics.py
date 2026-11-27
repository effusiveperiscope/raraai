#!/usr/bin/env python3
"""CPU smoke test for the per-word semantic state (Hints V2.1). No GPU, no BERT.

Covers the contract between the three sides:

* ``.cond`` -> dataset: a word's vector is repeated over its phones, an
  ``n_phone`` mismatch drops the channel (stale word fields), and the
  ``word_sem`` dropout leaves the vectors but clears the mask;
* collator: ``(B, N, D)`` float + ``(B, N)`` mask, padding is invalid;
* model: the projection starts at zero (so a warm start reproduces its source),
  the channel is live once trained, the *mask* is the absent state (values are
  ignored when it is False, and that path is ``word_null``, not ``proj(0)``),
  and the width in ``hints.json`` is enforced.

Run from the repo root:

    python test_word_semantics.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import torch

_failures = []

DIM = 16


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def _hints_json(path, dim=DIM, words=True):
    doc = {
        "version": 2,
        "levels": {
            "intensity": {"n_levels": 2, "edges": [5.0]},
            "register": {"n_levels": 2, "edges": [0.0], "speaker_means": {"0": 0.0}},
        },
        "words": {"pitch_anchors": 3},
    }
    if words:
        doc["words"]["word_bert"] = {"model": "synthetic", "dim": dim}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def _make_clip(tmp, *, word_bert=True, n_phone=6, dim=DIM, stale=False):
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
        "n_phone": n_phone if not stale else n_phone + 1,
        "intensity": torch.tensor([6.0]),
        "register_st": torch.tensor([3.0]),
        "phone_word_idx": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int16),
        "word_start_f": torch.tensor([0, 6], dtype=torch.int32),
        "word_slot_f": torch.tensor([6, 6], dtype=torch.int32),
        "pitch_word_st": torch.tensor([0.0, 4.0], dtype=torch.float16),
        "pitch_word_voiced": torch.tensor([True, False]),
        "pitch_anchors_st": torch.tensor([[0.0, 3.0, 1.0], [0.0, 0.0, 0.0]],
                                         dtype=torch.float16),
        "pitch_anchors_pos": torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]],
                                          dtype=torch.float16),
        "pitch_anchor_n": torch.tensor([3, 0], dtype=torch.int8),
    }
    if word_bert:
        vec = torch.arange(2 * dim, dtype=torch.float32).reshape(2, dim) / dim
        cond["word_bert"] = vec.to(torch.float16)
    torch.save(cond, os.path.join(d, "clip.wav.cond"))
    filelist = os.path.join(tmp, "train.txt")
    with open(filelist, "w", encoding="utf-8") as f:
        f.write(f"{phone}|{spec}|{pitch}|0\n")
    return filelist


def test_dataset():
    print("dataset (per-word state)")
    from dataset_codes import CodeCollator, CodeDataset
    from omegaconf import OmegaConf

    tmp = tempfile.mkdtemp()
    hj = os.path.join(tmp, "hints.json")
    _hints_json(hj)
    fl = _make_clip(tmp)
    ds = CodeDataset(fl, is_train=False, hints=True, hints_json=hj)
    item = ds[0]
    w, valid = item["hint_word"], item["hint_word_valid"]
    check("vector repeated over the word's phones",
          tuple(w.shape) == (6, DIM)
          and torch.equal(w[:3], w[0].expand(3, DIM))
          and torch.equal(w[3:], w[3].expand(3, DIM)),
          str(tuple(w.shape)))
    check("values round-trip", abs(float(w[0, 1]) - 1.0 / DIM) < 1e-3)
    check("every phone is valid", bool(valid.all()))

    # stale n_phone -> the whole word block (including the vector) is dropped
    tmp2 = tempfile.mkdtemp()
    hj2 = os.path.join(tmp2, "hints.json")
    _hints_json(hj2)
    fl2 = _make_clip(tmp2, stale=True)
    ds2 = CodeDataset(fl2, is_train=False, hints=True, hints_json=hj2)
    check("stale n_phone drops the channel", ds2[0]["hint_word"] is None)

    # no word_bert at all -> no channel, no crash
    tmp3 = tempfile.mkdtemp()
    hj3 = os.path.join(tmp3, "hints.json")
    _hints_json(hj3)
    fl3 = _make_clip(tmp3, word_bert=False)
    ds3 = CodeDataset(fl3, is_train=False, hints=True, hints_json=hj3)
    check("absent word_bert is None", ds3[0]["hint_word"] is None)

    # dropout: vectors survive, the mask is what the model reads
    ds4 = CodeDataset(fl, is_train=True, hints=True, hints_json=hj,
                      hint_dropout={"word_sem": 1.0})
    it = ds4[0]
    check("word_sem dropout clears the mask",
          it["hint_word"] is not None and not bool(it["hint_word_valid"].any()))

    cfg = OmegaConf.create({"transformer": {"hint_pitch_anchors": 3}})
    batch = CodeCollator(cfg)([ds[0], ds[0]])
    check("collator pads (B, N, D) + (B, N)",
          tuple(batch["hint_word"].shape) == (2, 6, DIM)
          and tuple(batch["hint_word_valid"].shape) == (2, 6)
          and bool(batch["hint_word_valid"].all()),
          str(tuple(batch["hint_word"].shape)))
    return ds, fl, hj, batch


def _cfg(hints_json, dim=DIM, over=None):
    from omegaconf import OmegaConf

    t = {
        "hidden_size": 32, "n_heads": 2, "encoder_layers": 1,
        "decoder_layers": 1, "dim_feedforward": 64, "dropout": 0.0,
        "max_positions": 512, "phone_vocab_size": 1024, "n_speakers": 4096,
        "eos_token": True, "hints": True, "hints_json": hints_json,
        "semantic_dim": 384, "decoder_hints": True,
        "hint_pitch_mode": "anchors", "hint_pitch_anchors": 3,
        "relative_register": True,
        "hint_word_sem": True, "hint_word_dim": dim,
    }
    t.update(over or {})
    return OmegaConf.create({
        "codec": {"codebook_num": 2, "codebook_size": 32},
        "pitch_codec": {"codebook_num": 1, "codebook_size": 32},
        "transformer": t,
    })


def test_model(batch, hints_json):
    print("model (per-word state)")
    from modeling.transformer import CodeTransformer

    torch.manual_seed(0)
    net = CodeTransformer(_cfg(hints_json))
    net.eval()
    check("projection exists with the configured width",
          net.word_proj is not None
          and net.word_proj.in_features == DIM
          and net.word_proj.out_features == 32)
    check("projection and null are zero-initialised",
          float(net.word_proj.weight.abs().max()) == 0.0
          and float(net.word_proj.bias.abs().max()) == 0.0
          and float(net.word_null.abs().max()) == 0.0)
    check("describe() reports the channel", f"word{DIM}" in net.describe(),
          net.describe())

    b = dict(batch)
    B, T = int(b["sid"].numel()), 12
    b["spec_codes"] = torch.randint(0, 32, (B, 2, T))
    b["pitch_codes"] = torch.randint(0, 32, (B, 1, T))
    b["code_lengths"] = torch.full((B,), T, dtype=torch.long)
    b["complete"] = torch.ones(B, dtype=torch.bool)

    out0 = net.compute_batch_loss(b, epoch=0, diagnostics=False)
    check("zero-init leaves the loss finite", bool(torch.isfinite(out0["loss"])))

    with torch.no_grad():
        net.word_proj.weight.normal_()
        net.word_proj.bias.normal_()
        net.word_null.normal_()
    out1 = net.compute_batch_loss(b, epoch=0, diagnostics=False)
    check("channel reaches the model",
          abs(float(out0["loss"]) - float(out1["loss"])) > 1e-7,
          f"{float(out0['loss']):.4f} vs {float(out1['loss']):.4f}")

    alt = dict(b)
    alt["hint_word"] = b["hint_word"] + 1.0
    out2 = net.compute_batch_loss(alt, epoch=0, diagnostics=False)
    check("changing the vector changes the loss",
          abs(float(out1["loss"]) - float(out2["loss"])) > 1e-7,
          f"{float(out1['loss']):.4f} vs {float(out2['loss']):.4f}")

    off = dict(b)
    off["hint_word_valid"] = torch.zeros_like(b["hint_word_valid"])
    alt_off = dict(off)
    alt_off["hint_word"] = b["hint_word"] + 5.0
    a = net.compute_batch_loss(off, epoch=0, diagnostics=False)
    c = net.compute_batch_loss(alt_off, epoch=0, diagnostics=False)
    check("invalid rows ignore the vector (trained null path)",
          abs(float(a["loss"]) - float(c["loss"])) < 1e-7,
          f"{float(a['loss']):.4f} vs {float(c['loss']):.4f}")

    with torch.no_grad():
        gen = net.generate(b["phone_ids"], b["phone_lengths"], b["sid"],
                           max_frames=8, hints=None)
    check("generate runs in the null state", tuple(gen.shape)[0] == B)
    null = net.null_hints(B, 6)
    check("null_hints carries the new keys",
          "hint_word" in null and null["hint_word"].shape[-1] == DIM)

    # width contract from hints.json
    tmp = tempfile.mkdtemp()
    bad = os.path.join(tmp, "hints.json")
    _hints_json(bad, dim=DIM * 2)
    try:
        CodeTransformer(_cfg(bad))
        check("hints.json width mismatch raises", False)
    except ValueError:
        check("hints.json width mismatch raises", True)


def main() -> int:
    made = test_dataset()
    test_model(made[3], made[2])
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        return 1
    print("all per-word-state checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
