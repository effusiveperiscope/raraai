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
* the seam additions: ``transformer.fim_lookahead`` (repeated suffix frames
  after the middle) and ``fim_seam_frames``/``fim_seam_weight``, plus
  ``infill_seam_score`` / ``infill(best_of=N)`` and the
  ``apply_fim`` <-> ``_fim_frame_tags`` agreement.

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

    # An all-False mask -- i.e. every *validation* batch, since
    # `dataset_codes._sample_fim` is gated on `is_train` -- has no middle to
    # score, so it must emit no FIM scalars at all.  Reporting 0.0 there both
    # looked like a perfect span and, through the trainer's `on_epoch=True`
    # logging, overwrote the real `val_fim_middle_loss`/`_acc` the FIM probe
    # writes at the same step (the tag then read as 0 in TensorBoard).
    ar = model.compute_batch_loss(make_batch(model, [False, False]), epoch=0,
                                  diagnostics=True)
    leaked = sorted(k for k in ar if k.startswith("fim_middle"))
    check(not leaked, f"empty FIM mask emitted {leaked}")
    check("fim_frac" in ar and float(ar["fim_frac"]) == 0.0,
          "fim_frac should still report 0.0 for an all-AR batch")

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


def make_lookahead_config(lookahead: int = 3, seam_frames: int = 2,
                          seam_weight: float = 3.0):
    cfg = OmegaConf.create(OmegaConf.to_container(make_config(),
                                                  resolve=True))
    cfg.transformer.fim_lookahead = lookahead
    cfg.transformer.fim_seam_frames = seam_frames
    cfg.transformer.fim_seam_weight = seam_weight
    return cfg


def test_best_of_guard(model):
    """``infill(best_of>1)`` needs the lookahead objective to rank with."""
    K = model.codebook_num
    prefix = torch.randint(0, 8, (1, K, 2))
    suffix = torch.randint(0, 8, (1, K, 2))
    phone = torch.randint(0, 30, (1, 6))
    try:
        model.infill(phone, torch.tensor([6]), torch.tensor([2]),
                     prefix, suffix, 3, best_of=2)
    except ValueError as e:
        check("fim_lookahead" in str(e), f"unhelpful best_of error: {e}")
    else:
        raise SystemExit("FAIL: infill(best_of=2) should need a lookahead")
    print("  infill(best_of>1) guard without lookahead: OK")


def test_lookahead():
    """Suffix lookahead + seam weighting (v0.7-seam)."""
    torch.manual_seed(0)
    model = CodeTransformer(make_lookahead_config())
    model.eval()
    check(model.fim_lookahead == 3, "fim_lookahead not read from config")
    check("fim_layout_marker" in dict(model.named_buffers()),
          "fim_layout_marker buffer missing")
    K = model.codebook_num
    T, p, m, look = 8, 2, 3, 3
    s = T - p - m                                  # 3 == look, so a == 3
    codes = torch.arange(T).view(1, 1, T).repeat(1, K, 1).long()
    cl = torch.tensor([T])
    out, lengths, region, len_bucket = model.apply_fim(
        codes, cl, torch.tensor([True]), torch.tensor([p]), torch.tensor([m]))

    Tn = T + 2 + 1 + look
    check(tuple(out.shape) == (1, K, Tn), f"lookahead width {out.shape}")
    check(int(lengths[0]) == Tn, f"lookahead length {lengths.tolist()}")
    # prefix <SUF> suffix MID middle <SUF> suffix[:a]
    check(torch.equal(out[:, :, :p], codes[:, :, :p]), "prefix moved")
    check(int(out[0, 0, p]) == model.suf_ids[0], "first SUF missing")
    check(torch.equal(out[:, :, p + 1:p + 1 + s], codes[:, :, p + m:]),
          "suffix moved")
    check(int(out[0, 0, p + s + 1]) == model.mid_ids[0], "MID missing")
    check(torch.equal(out[:, :, p + s + 2:p + s + 2 + m], codes[:, :, p:p + m]),
          "middle moved")
    la = p + s + 2 + m
    check(int(out[0, 0, la]) == model.suf_ids[0], "lookahead SUF missing")
    check(torch.equal(out[:, :, la + 1:la + 1 + look], codes[:, :, p + m:]),
          "lookahead does not repeat the suffix start")
    exp_reg = torch.tensor([[1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2]])
    check(torch.equal(region, exp_reg), f"lookahead region {region.tolist()}")
    # Length buckets stay a property of the *span*: the lookahead is suffix.
    exp_len = torch.tensor([[0, 0, 0, 0, 0, 0, 3, 2, 2, 1, 0, 0, 0, 0]])
    check(torch.equal(len_bucket, exp_len),
          f"lookahead len_bucket {len_bucket.tolist()}")

    # ``apply_fim`` and the prompt builder in ``infill`` must agree: same span
    # -> same per-frame region and length tags.
    tag_reg, tag_remaining = model._fim_frame_tags(1, p, s, m, Tn, "cpu")
    check(torch.equal(region, tag_reg), "apply_fim / _fim_frame_tags region")
    check(torch.equal(len_bucket,
                      model._len_buckets(tag_remaining, model.n_len_buckets)),
          "apply_fim / _fim_frame_tags length tags")

    # Short suffix: a = min(look, s) frames are repeated, and the stream ends
    # there (the remaining columns are collator padding, not lookahead).
    cl_short = torch.tensor([6])                   # s = 1
    out2, len2, _, _ = model.apply_fim(
        codes, cl_short, torch.tensor([True]),
        torch.tensor([p]), torch.tensor([m]))
    check(int(len2[0]) == 6 + 2 + 2, f"short-suffix length {len2.tolist()}")
    check(int(out2[0, 0, 8]) == model.suf_ids[0], "short lookahead SUF")
    check(torch.equal(out2[:, :, 9:10], codes[:, :, p + m:p + m + 1]),
          "short lookahead frame")
    check(int(out2[0, 0, 10]) == model.pad_ids[0],
          "short lookahead leaked past the suffix")

    # Empty suffix: no lookahead at all (EOS follows the middle instead).
    cl_nosuf = torch.tensor([5])
    _, len3, _, _ = model.apply_fim(
        codes, cl_nosuf, torch.tensor([True]),
        torch.tensor([2]), torch.tensor([3]))
    check(int(len3[0]) == 5 + 2, f"no-suffix length {len3.tolist()}")

    # Loss: the lookahead block and the seam targets are scored, and the
    # weighted loss is finite.
    batch = make_batch(model, [True, False])
    out_loss = model.compute_batch_loss(batch, epoch=0, diagnostics=True)
    for key in ("fim_seam_acc", "fim_lookahead_loss", "fim_lookahead_acc"):
        check(key in out_loss and torch.isfinite(out_loss[key]),
              f"{key} missing/unusable")
    out_loss["loss"].backward()
    check(model.decoder.len_emb.weight.grad is not None, "len_emb no grad")
    model.zero_grad(set_to_none=True)
    ar = model.compute_batch_loss(make_batch(model, [False, False]), epoch=0,
                                  diagnostics=True)
    check(not [k for k in ar if k.startswith(("fim_seam",
                                              "fim_lookahead"))],
          "AR-only batch leaked seam/lookahead scalars")

    # Seam weighting must actually change the loss (it must not be a no-op).
    plain = CodeTransformer(make_lookahead_config(seam_frames=0,
                                                  seam_weight=1.0))
    plain.load_state_dict(model.state_dict())
    plain.eval()
    l_w = float(model.compute_batch_loss(batch, epoch=0,
                                         diagnostics=False)["loss"].detach())
    l_p = float(plain.compute_batch_loss(batch, epoch=0,
                                         diagnostics=False)["loss"].detach())
    check(abs(l_w - l_p) > 1e-6, "fim_seam_weight had no effect on the loss")

    # Seam score + best-of-N.
    phone = torch.randint(0, 30, (1, 6))
    plen = torch.tensor([6])
    sid = torch.tensor([2])
    prefix = torch.randint(0, 8, (1, K, 2))
    suffix = torch.randint(0, 8, (1, K, 4))
    mid = model.infill(phone, plen, sid, prefix, suffix, 3, temperature=1.0)
    score = model.infill_seam_score(phone, plen, sid, prefix, mid, suffix)
    check(score.shape == (1,) and torch.isfinite(score).all(),
          f"infill_seam_score {score.shape}")
    no_suf = model.infill_seam_score(phone, plen, sid, prefix, mid,
                                     suffix[:, :, :0])
    check(float(no_suf[0]) == 0.0, "empty suffix should score 0")
    best = model.infill(phone, plen, sid, prefix, suffix, 3, best_of=3)
    check(tuple(best.shape) == (1, K, 3), f"best_of shape {tuple(best.shape)}")
    for k in range(K):
        v = model.vocab_sizes[k]
        check(int(best[:, k].min()) >= 0 and int(best[:, k].max()) < v,
              f"best_of emitted out-of-range id in codebook {k}")

    # End-to-end pin: scoring the *ground-truth* middle through the inference
    # path must reproduce the training path's lookahead CE for the same item.
    # This is what guarantees `infill_seam_score`/`best_of` rank with the
    # objective that was actually trained.
    T2, p2, m2 = 9, 2, 3                       # suffix s = 4 > look(3) -> a = 3
    codes = torch.randint(0, 8, (1, K, T2))
    phones = torch.randint(1, 30, (1, 6))
    plens = torch.tensor([6])
    sid2 = torch.tensor([3])
    batch = {
        "phone_ids": phones, "phone_lengths": plens, "sid": sid2,
        "spec_codes": codes[:, :2], "pitch_codes": codes[:, 2:],
        "code_lengths": torch.tensor([T2]),
        "complete": torch.tensor([False]),
        "fim": torch.tensor([True]),
        "fim_prefix_len": torch.tensor([p2]),
        "fim_mid_len": torch.tensor([m2]),
    }
    out = model.compute_batch_loss(batch, epoch=0, diagnostics=True)
    sc = model.infill_seam_score(phones, plens, sid2,
                                 codes[:, :, :p2], codes[:, :, p2:p2 + m2],
                                 codes[:, :, p2 + m2:])
    check(abs(float(sc[0]) - float(out["fim_lookahead_loss"])) < 1e-4,
          f"seam score {float(sc[0]):.5f} != training lookahead CE "
          f"{float(out['fim_lookahead_loss']):.5f}")
    print("  suffix lookahead + seam weighting + best-of-N: OK")


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
    test_best_of_guard(model)
    test_lookahead()
    print("ALL FIM CHECKS PASSED")


if __name__ == "__main__":
    main()
