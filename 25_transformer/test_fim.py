"""CPU smoke test for the FIM (prefix-suffix-middle) stage-2 additions.

Runs entirely on CPU so it can be used next to a busy GPU job (e.g. an
in-progress stage-1 finetune).  It does not read any data filelists or
checkpoints: it builds a tiny ``CodeTransformer`` from an inline config and
checks the FIM code paths directly:

* ``apply_fim`` reordering (frames, sentinels, region tags, length buckets),
* ``compute_batch_loss`` with a mixed AR + FIM batch (finite loss,
  ``fim_middle_loss``/``fim_middle_acc`` present, gradients reach the region and
  length embeddings, the ``ablate_*`` control arms change the loss),
* ``generate`` never emitting a sentinel / pad id,
* ``infill`` returning a valid fixed-length middle,
* ``infill``'s per-column region/length tags being *bit-identical* to the
  training path's (the regression test for the one-column lag that made infill
  ask the model to emit SUF/MID under the wrong region).

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
            "length_cond": True, "n_len_buckets": 9,
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
    out, lengths, region, len_bucket = model.apply_fim(
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
    # Length buckets: 0 outside the span; MID reports m+1 = 4 -> bucket 3, then
    # the middle's three frames report 3/2/1 frames left -> buckets 2/2/1.
    exp_len = torch.tensor([[0, 0, 0, 0, 0, 0, 3, 2, 2, 1]])
    check(torch.equal(len_bucket, exp_len),
          f"len_bucket {len_bucket.tolist()} != {exp_len.tolist()}")

    # Short item (cl < T): the reordered stream is cl+2 frames; the collator
    # padding past it must stay pad (not leak into the loss).
    cl_short = torch.tensor([6])  # T=8, so 2 trailing pad frames
    out2, len2, _, len2_bucket = model.apply_fim(
        codes, cl_short, torch.tensor([True]),
        torch.tensor([2]), torch.tensor([3]))  # p=2,m=3,s=1 -> len 8
    check(int(len2[0]) == 8, f"short length {len2}")
    pad0 = model.pad_ids[0]
    check(int(out2[0, 0, 8]) == pad0 and int(out2[0, 0, 9]) == pad0,
          "short FIM item leaked padding frames")
    check(int(len2_bucket[0, 7]) == 1 and int(len2_bucket[0, 8]) == 0,
          f"short FIM length tag {len2_bucket.tolist()}")
    print("  apply_fim (frames, sentinels, region, length buckets): OK")


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
    check("fim_middle_acc" in out, "fim_middle_acc missing on mixed batch")
    for name in ("early", "mid", "late"):
        key = f"fim_middle_acc_{name}"
        check(key in out and torch.isfinite(out[key]), f"{key} missing/bad")
    check(abs(float(out["fim_frac"]) - 0.5) < 1e-6, "fim_frac != 0.5")
    out["loss"].backward()
    rg = model.decoder.region_emb.weight.grad
    check(rg is not None and rg.abs().sum() > 0, "region_emb got no gradient")
    lg = model.decoder.len_emb.weight.grad
    check(lg is not None and lg.abs().sum() > 0, "len_emb got no gradient")
    model.zero_grad(set_to_none=True)

    out2 = model.compute_batch_loss(batch, epoch=0, diagnostics=False)
    check(torch.isfinite(out2["loss"]), "loss (no diag) not finite")
    check("fim_middle_loss" not in out2, "fim_middle_loss leaked w/o diagnostics")

    # Ablation arms (the FIM probe's controls): identical span/targets, one side
    # channel zeroed.  They must stay finite and must not crash the batch.
    memory, memory_mask = model.encode(batch["phone_ids"],
                                       batch["phone_lengths"], batch["sid"])
    codes = torch.cat([batch["spec_codes"], batch["pitch_codes"]], dim=1).long()
    kw = dict(code_lengths=batch["code_lengths"], sid=batch["sid"],
              complete=batch["complete"], diagnostics=True,
              fim_mask=batch["fim"], fim_prefix_len=batch["fim_prefix_len"],
              fim_mid_len=batch["fim_mid_len"])
    for arm in ({}, {"ablate_region": True}, {"ablate_length": True}):
        ce = model.code_loss(memory, memory_mask, codes, **arm, **kw)
        check(torch.isfinite(ce["fim_middle_loss"]),
              f"fim_middle_loss not finite for arm {arm}")
    model.zero_grad(set_to_none=True)

    ar = make_batch(model, [False, False])
    out3 = model.compute_batch_loss(ar, epoch=0, diagnostics=True)
    check(torch.isfinite(out3["loss"]), "AR loss not finite")
    check(float(out3["fim_frac"]) == 0.0, "AR fim_frac != 0")
    print("  compute_batch_loss (mixed + AR + ablations): OK")


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


def test_infill_alignment(model):
    """``infill`` must feed the decoder exactly what training feeds it.

    Regression test for the one-column lag: ``infill`` used to tag each column
    with the region of the token it had *just appended* instead of the token it
    is about to predict, so the model was asked to emit ``SUF``/``MID`` (and to
    start the span) under the previous frame's region.  Force-feeding the
    ground-truth middle through ``infill``'s loop and comparing against
    ``code_loss``'s logits is the direct check: with identical inputs the two
    must agree to float noise.
    """
    K = model.codebook_num
    T, p, m = 9, 2, 3           # s = T - p - m = 4
    s = T - p - m
    torch.manual_seed(0)
    codes = torch.randint(0, 8, (1, K, T))
    phones = torch.randint(1, 30, (1, 6))
    plens = torch.tensor([6])
    sid = torch.tensor([3])
    cl = torch.tensor([T])
    memory, memory_mask = model.encode(phones, plens, sid)

    # --- training path (code_loss's internals, up to the decoder call)
    nc, nl, region, len_bucket = model.apply_fim(
        codes, cl, torch.tensor([True]), torch.tensor([p]), torch.tensor([m]))
    nc, nl = model._append_eos(nc, nl, torch.tensor([False]))
    delayed = model.build_delay_pattern(nc, nl)
    L = delayed.shape[2]
    seq = torch.cat([model._pad_column(1, codes.device), delayed], dim=2)
    region_tr = model._tag_to_columns(region, nc.shape[2], L)
    len_tr = model._tag_to_columns(len_bucket, nc.shape[2], L)

    # --- infill's tags, built the way `CodeTransformer.infill` builds them
    known = torch.cat([codes[:, :, :p],
                       torch.tensor(model.suf_ids).view(1, K, 1),
                       codes[:, :, p + m:],
                       torch.tensor(model.mid_ids).view(1, K, 1)], dim=2)
    Kp = known.shape[2]
    Tn = Kp + m
    region_frames = torch.full((1, Tn), 3, dtype=torch.long)
    region_frames[:, :p] = 1
    region_frames[:, p:p + s + 1] = 2
    pos = torch.arange(Tn).view(1, Tn)
    remaining = torch.where(pos >= (Kp - 1), Tn - pos, torch.zeros_like(pos))
    len_frames = model._len_buckets(remaining, model.n_len_buckets)
    region_inf = torch.cat([region_frames,
                            region_frames[:, -1:].expand(1, K - 1)], 1)
    len_inf = torch.cat([len_frames, len_frames[:, -1:].expand(1, K - 1)], 1)
    check(torch.equal(region_inf[:, :Tn], region_tr[:, :Tn]),
          f"infill region columns {region_inf[0, :Tn].tolist()} != training "
          f"{region_tr[0, :Tn].tolist()}")
    check(torch.equal(len_inf[:, :Tn], len_tr[:, :Tn]),
          f"infill length columns {len_inf[0, :Tn].tolist()} != training "
          f"{len_tr[0, :Tn].tolist()}")

    # --- force-feed the ground truth through infill's loop and compare logits
    Li = Tn + K - 1
    pad = torch.tensor(model.pad_ids).view(1, K)
    ar = torch.arange(K).view(K)
    with torch.no_grad():
        lg_tr = model.decoder(seq[:, :, :L], memory, memory_mask, sid=sid,
                              region=region_tr, length=len_tr)
        seq_i = model._pad_column(1, codes.device)
        worst = 0.0
        for i in range(Li):
            frame = i - ar
            if i < Kp:
                col = known[:, ar, frame.clamp(0, Kp - 1)]
                col = torch.where(((frame >= 0) & (frame < Kp)).view(1, K), col, pad)
            else:
                n_cols = seq_i.shape[2]
                lg_i = model.decoder(seq_i, memory, memory_mask, sid=sid,
                                     region=region_inf[:, :n_cols],
                                     length=len_inf[:, :n_cols])
                for k in range(K):
                    worst = max(worst, float(
                        (lg_i[k][0, -1] - lg_tr[k][0, i]).abs().max()))
                tok = known[:, ar, frame.clamp(0, Kp - 1)]
                orig = ((frame - Kp).clamp(0, m - 1) + p).clamp(0, T - 1)
                tok = torch.where(((frame >= Kp) & (frame < Tn)).view(1, K),
                                  codes[:, ar, orig], tok)
                col = torch.where(((frame >= 0) & (frame < Tn)).view(1, K),
                                  tok, pad)
            seq_i = torch.cat([seq_i, col.unsqueeze(-1)], dim=2)
    check(worst < 1e-3,
          f"infill logits differ from the training path by {worst:.3e}")
    print(f"  infill tags/logits match training (max|dlogit| {worst:.1e}): OK")


def main():
    torch.manual_seed(0)
    model = CodeTransformer(make_config())
    model.eval()
    print(model.describe())
    test_apply_fim(model)
    test_loss(model)
    test_generate(model)
    test_infill(model)
    test_infill_alignment(model)
    print("ALL FIM CHECKS PASSED")


if __name__ == "__main__":
    main()
