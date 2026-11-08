"""CPU smoke test for the FIM (prefix-suffix-middle) stage-2 additions.

Runs entirely on CPU so it can be used next to a busy GPU job (e.g. an
in-progress stage-1 finetune).  It does not read any data filelists or
checkpoints: it builds a tiny ``CodeTransformer`` from an inline config and
checks the FIM code paths directly:

* ``apply_fim`` reordering (frames, sentinels, region tags),
* ``compute_batch_loss`` with a mixed AR + FIM batch (finite loss,
  ``fim_middle_loss`` present, gradients reach the region embedding),
* ``generate`` never emitting a sentinel / pad id,
* ``infill`` returning a valid fixed-length middle.

Usage: ``python test_fim.py`` (exits non-zero on failure).
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from modeling.transformer import CodeTransformer


def make_config():
    return OmegaConf.create({
        "transformer": {
            "hidden_size": 32, "n_heads": 4, "encoder_layers": 1,
            "decoder_layers": 1, "dim_feedforward": 64, "dropout": 0.0,
            "max_positions": 512, "phone_vocab_size": 32, "phone_pad_id": 0,
            "n_speakers": 8, "decoder_speaker_cond": True,
            "memory_norm": "temporal", "residual_scale": True,
            "eos_token": True, "fim": True, "region_cond": True,
            "fim_min_frames": 2, "fim_mid_frac": 0.5,
            "codebook_weights": None, "codebook_weight_warmup_epochs": 0,
        },
        "codec": {"codebook_num": 2, "codebook_size": 8},
        "pitch_codec": {"codebook_num": 1, "codebook_size": 8},
    })


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def test_apply_fim(model):
    K = model.codebook_num
    T = 8
    codes = torch.arange(T).view(1, 1, T).repeat(1, K, 1).long()  # frame==t
    cl = torch.tensor([T])
    out, lengths, region = model.apply_fim(
        codes, cl, torch.tensor([True]),
        torch.tensor([2]), torch.tensor([3]))  # p=2, m=3, s=T-p-m=3
    check(tuple(out.shape) == (1, K, T + 2), f"apply_fim width {out.shape}")
    check(int(lengths[0]) == T + 2, f"lengths {lengths}")
    check(torch.equal(out[:, :, :2], codes[:, :, :2]), "prefix frames moved")
    check(int(out[0, 0, 2]) == model.suf_ids[0], "SUF sentinel missing at p")
    check(torch.equal(out[:, :, 3:6], codes[:, :, 5:8]), "suffix frames moved")
    check(int(out[0, 0, 6]) == model.mid_ids[0], "MID sentinel missing")
    check(torch.equal(out[:, :, 7:10], codes[:, :, 2:5]), "middle frames moved")
    exp_reg = torch.tensor([[1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
    check(torch.equal(region, exp_reg), f"region {region.tolist()}")

    # Short item (cl < T): the reordered stream is cl+2 frames; the collator
    # padding past it must stay pad (not leak into the loss).
    cl_short = torch.tensor([6])  # T=8, so 2 trailing pad frames
    out2, len2, _ = model.apply_fim(
        codes, cl_short, torch.tensor([True]),
        torch.tensor([2]), torch.tensor([3]))  # p=2,m=3,s=1 -> len 8
    check(int(len2[0]) == 8, f"short length {len2}")
    pad0 = model.pad_ids[0]
    check(int(out2[0, 0, 8]) == pad0 and int(out2[0, 0, 9]) == pad0,
          "short FIM item leaked padding frames")
    print("  apply_fim: OK")


def make_batch(model, fim_flags):
    B = len(fim_flags)
    T = 8
    q_spec = model.spec_codebook_num
    q_pitch = model.pitch_codebook_num
    spec = torch.randint(0, 8, (B, q_spec, T))
    pitch = torch.randint(0, 8, (B, q_pitch, T))
    cl = torch.tensor([T - i for i in range(B)])
    return {
        "phone_ids": torch.randint(0, 30, (B, 5)),
        "phone_lengths": torch.tensor([5] * B),
        "sid": torch.randint(0, 8, (B,)),
        "spec_codes": spec, "pitch_codes": pitch,
        "code_lengths": cl, "complete": torch.tensor([True] * B),
        "fim": torch.tensor(fim_flags),
        "fim_prefix_len": torch.tensor([2 if f else 0 for f in fim_flags]),
        "fim_mid_len": torch.tensor([3 if f else 0 for f in fim_flags]),
    }

def test_loss(model):
    batch = make_batch(model, [True, False])
    out = model.compute_batch_loss(batch, epoch=0, diagnostics=True)
    for k in ("loss", "spec_loss", "pitch_loss"):
        check(torch.isfinite(out[k]), f"{k} not finite")
    check("fim_middle_loss" in out, "fim_middle_loss missing on mixed batch")
    check(torch.isfinite(out["fim_middle_loss"]), "fim_middle_loss not finite")
    check(abs(float(out["fim_frac"]) - 0.5) < 1e-6, "fim_frac != 0.5")
    out["loss"].backward()
    rg = model.decoder.region_emb.weight.grad
    check(rg is not None and rg.abs().sum() > 0, "region_emb got no gradient")
    model.zero_grad(set_to_none=True)

    out2 = model.compute_batch_loss(batch, epoch=0, diagnostics=False)
    check(torch.isfinite(out2["loss"]), "loss (no diag) not finite")
    check("fim_middle_loss" not in out2, "fim_middle_loss leaked w/o diagnostics")

    ar = make_batch(model, [False, False])
    out3 = model.compute_batch_loss(ar, epoch=0, diagnostics=True)
    check(torch.isfinite(out3["loss"]), "AR loss not finite")
    check(float(out3["fim_frac"]) == 0.0, "AR fim_frac != 0")
    print("  compute_batch_loss (mixed + AR): OK")


def test_generate(model):
    phone = torch.randint(0, 30, (1, 5))
    plen = torch.tensor([5])
    sid = torch.tensor([1])
    codes, lens = model.generate(phone, plen, sid, max_frames=6,
                                 temperature=1.0, return_lengths=True)
    check(codes.dim() == 3 and codes.shape[0] == 1, "generate shape")
    for k in range(model.codebook_num):
        v = model.vocab_sizes[k]
        check(int(codes[:, k].min()) >= 0 and int(codes[:, k].max()) < v,
              f"generate emitted out-of-range id in codebook {k}")
    print("  generate masks sentinels/pad: OK")


def test_infill(model):
    K = model.codebook_num
    p, s, m = 2, 2, 3
    prefix = torch.randint(0, 8, (1, K, p))
    suffix = torch.randint(0, 8, (1, K, s))
    phone = torch.randint(0, 30, (1, 6))
    plen = torch.tensor([6])
    sid = torch.tensor([2])
    mid = model.infill(phone, plen, sid, prefix, suffix, m,
                       temperature=1.0, top_k=None)
    check(tuple(mid.shape) == (1, K, m), f"infill shape {tuple(mid.shape)}")
    for k in range(K):
        v = model.vocab_sizes[k]
        check(int(mid[:, k].min()) >= 0 and int(mid[:, k].max()) < v,
              f"infill emitted out-of-range id in codebook {k}")
    print("  infill fixed-length middle: OK")


def main():
    torch.manual_seed(0)
    model = CodeTransformer(make_config())
    model.eval()
    print(model.describe())
    test_apply_fim(model)
    test_loss(model)
    test_generate(model)
    test_infill(model)
    print("ALL FIM CHECKS PASSED")


if __name__ == "__main__":
    main()
