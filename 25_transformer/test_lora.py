"""CPU smoke test for stage-2 LoRA fine-tuning (``2_training.py --finetune``).

Runs entirely on CPU, so it can be used next to a busy GPU job.  It builds tiny
``CodeTransformer``s from inline configs (no data, no checkpoints) and checks the
properties the mode depends on:

* injection keeps every base key (so ``--transfer_from`` loads a plain
  checkpoint into a LoRA model) and is an exact identity at init,
* ``freeze_except_adapters`` leaves only adapters (+ speaker rows) trainable,
* gradients reach ``lora_B`` immediately and ``lora_A`` once ``lora_B`` is
  non-zero, while no base tensor gets a gradient,
* ``merge_lora`` folds the deltas into the base weights and returns a *plain*
  state dict that a model built with ``finetune.enabled: false`` loads and
  reproduces exactly (including the rotary qkv adapter),
* every layout guard refuses the mismatches that would otherwise load silently:
  a LoRA checkpoint into a plain model (``commons.load_state_dict_mismatch`` and
  ``check_lora_layout``), and rank / alpha / target mismatches between a
  checkpoint and the config.

Usage: ``python test_lora.py`` (exits non-zero on failure).
"""

from __future__ import annotations

import copy

import torch
from omegaconf import OmegaConf

from commons import load_state_dict_mismatch
from modeling.lora import (check_lora_layout, freeze_except_adapters,
                           is_lora_key, lora_parameters, merge_lora,
                           resolve_config, uses_lora)
from modeling.transformer import CodeTransformer

TARGETS = ["encoder_ffn", "decoder_ffn", "decoder_attn", "decoder_qkv"]


def make_config(finetune=True, rank=4, alpha=8.0, targets=None,
                rope=True, dropout=0.0):
    cfg = OmegaConf.create({
        "transformer": {
            "hidden_size": 32, "n_heads": 4, "encoder_layers": 1,
            "decoder_layers": 1, "dim_feedforward": 64, "dropout": 0.0,
            "max_positions": 512, "phone_vocab_size": 32, "phone_pad_id": 0,
            "n_speakers": 8, "decoder_speaker_cond": True,
            "memory_norm": "temporal", "residual_scale": True,
            "eos_token": False, "fim": False, "region_cond": False,
            "length_cond": False, "rope": rope,
            "hints": False,
            "codebook_weights": None, "codebook_weight_warmup_epochs": 0,
        },
        "codec": {"codebook_num": 2, "codebook_size": 8},
        "pitch_codec": {"codebook_num": 1, "codebook_size": 8},
    })
    if finetune is not False:
        cfg["finetune"] = OmegaConf.create({
            "enabled": True,
            "targets": list(TARGETS if targets is None else targets),
            "rank": rank, "alpha": alpha, "dropout": dropout,
            "train_spk_emb": True, "train_memory_norm": False,
            "lr": 1e-3, "weight_decay": 0.0, "cosine_anneal_end": None,
        })
    return cfg


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def make_batch(model, B=2, T=8):
    return {
        "phone_ids": torch.randint(0, 30, (B, 5)),
        "phone_lengths": torch.tensor([5] * B),
        "sid": torch.randint(0, 8, (B,)),
        "spec_codes": torch.randint(0, 8, (B, model.spec_codebook_num, T)),
        "pitch_codes": torch.randint(0, 8, (B, model.pitch_codebook_num, T)),
        "code_lengths": torch.tensor([T, T - 1]),
        "complete": torch.tensor([True, False]),
    }


def prefixed(state_dict):
    return {f"net.{k}": v for k, v in state_dict.items()}


def outputs(model, batch):
    with torch.no_grad():
        mem, _ = model.encode(batch["phone_ids"], batch["phone_lengths"],
                              batch["sid"])
        loss = model.compute_batch_loss(batch, epoch=0,
                                        diagnostics=False)["loss"]
    return mem, loss


# ---------------------------------------------------------------------- #
def test_config_validation():
    cfg = resolve_config(None)
    check(cfg["rank"] == 16 and cfg["alpha"] == 32.0, "defaults changed")
    check(cfg["train_spk_emb"] is True, "spk_emb should default to trainable")
    try:
        resolve_config({"targets": ["decoder_ffn", "nope"]})
    except ValueError:
        pass
    else:
        raise SystemExit("FAIL: unknown finetune.targets entry accepted")
    try:
        resolve_config({"rank": 0})
    except ValueError:
        pass
    else:
        raise SystemExit("FAIL: rank=0 accepted")
    print("  config resolution + validation: OK")


def test_identity_and_keys():
    torch.manual_seed(0)
    base = CodeTransformer(make_config(finetune=False)).eval()
    base_sd = base.state_dict()

    torch.manual_seed(1)                      # different init: adapters must
    lora = CodeTransformer(make_config()).eval()   # *not* be what matches

    lora_sd = lora.state_dict()
    plain_keys = {k for k in lora_sd if not is_lora_key(k)}
    check(plain_keys == set(base_sd),
          "injection changed base state-dict keys: "
          f"{sorted(plain_keys ^ set(base_sd))[:4]}")
    check(uses_lora(lora_sd), "lora_marker / adapters missing from state_dict")
    check(not uses_lora(base_sd), "plain model reports LoRA")
    check(not any("rope" in k and "lora" in k for k in lora_sd),
          "unexpected key")

    # The marker records the contract (rank, alpha, #targets).
    marker = lora_sd["lora_marker"]
    check(marker.tolist()[:2] == [4.0, 8.0], f"marker {marker.tolist()}")
    check("lora_marker" not in base_sd, "plain model has a lora_marker")

    # qkv/attn adapters exist only on the rotary attention: without rope they
    # must be refused, not silently skipped.
    check(any("self_attn.lora_qkv_A" in k for k in lora_sd),
          "decoder_qkv adapter missing on a rope model")
    try:
        CodeTransformer(make_config(rope=False))
    except ValueError:
        pass
    else:
        raise SystemExit("FAIL: decoder_attn/qkv accepted on a non-rope model")
    plain_rope_off = CodeTransformer(make_config(
        rope=False, targets=["encoder_ffn", "decoder_ffn"])).eval()
    check(not any("lora_qkv" in k for k in plain_rope_off.state_dict()),
          "qkv adapter injected on a non-rotary model")

    load_state_dict_mismatch(lora, base_sd)
    batch = make_batch(lora)
    mem0, loss0 = outputs(base, batch)
    mem1, loss1 = outputs(lora, batch)
    check(torch.equal(mem0, mem1), "adapters changed the encoder memory at init")
    check(torch.equal(loss0, loss1), "adapters changed the loss at init")
    print("  injection: base keys preserved, exact identity at init: OK")


def test_freezing():
    torch.manual_seed(0)
    net = CodeTransformer(make_config()).eval()
    counts = freeze_except_adapters(net, net.config.finetune)
    check(counts["lora"] > 0, "no LoRA parameters found")
    check(counts["trainable"] == counts["lora"] + counts["spk_emb"],
          f"unexpected trainable count {counts}")
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    allowed = [n for n in trainable
               if ".lora_" in n or "spk_emb.weight" in n]
    check(len(allowed) == len(trainable),
          f"non-adapter parameters are trainable: "
          f"{sorted(set(trainable) - set(allowed))[:4]}")
    check(any("encoder.spk_emb" in n for n in trainable),
          "encoder speaker table should be trainable")
    check(all(not p.requires_grad for p in lora_parameters(net)) is False,
          "adapter parameters are frozen")
    print(f"  freezing: {counts['trainable']/1e3:.1f}k trainable of "
          f"{counts['total']/1e3:.1f}k: OK")


def test_gradients():
    torch.manual_seed(0)
    net = CodeTransformer(make_config()).eval()
    freeze_except_adapters(net, net.config.finetune)
    batch = make_batch(net)
    out = net.compute_batch_loss(batch, epoch=0, diagnostics=False)
    out["loss"].backward()

    grads = {n: p.grad for n, p in net.named_parameters() if p.grad is not None}
    check(grads, "no gradients at all")
    base_grads = [n for n in grads if ".lora_" not in n and "spk_emb" not in n]
    check(not base_grads, f"frozen parameter got a gradient: {base_grads[:3]}")
    gb = [g for n, g in grads.items() if n.endswith("lora_B")]
    check(gb and any(g.abs().sum() > 0 for g in gb),
          "lora_B received no gradient (adapters are disconnected)")

    # At init B=0, so dL/dA is exactly zero: that is why `lora_A` can only start
    # learning once `lora_B` has moved.  Give B a value and re-check.
    net.zero_grad(set_to_none=True)
    with torch.no_grad():
        for module in net.modules():
            if hasattr(module, "lora_B"):
                module.lora_B.normal_(0.0, 0.02)
            if hasattr(module, "lora_qkv_B"):
                module.lora_qkv_B.normal_(0.0, 0.02)
    net.compute_batch_loss(batch, epoch=0, diagnostics=False)["loss"].backward()
    ga = [p.grad for n, p in net.named_parameters()
          if n.endswith("lora_A") and p.grad is not None]
    check(ga and any(g.abs().sum() > 0 for g in ga),
          "lora_A still has no gradient with non-zero lora_B")
    net.zero_grad(set_to_none=True)
    print("  gradients: adapters only, A learns once B != 0: OK")


def test_merge():
    torch.manual_seed(0)
    net = CodeTransformer(make_config()).eval()
    freeze_except_adapters(net, net.config.finetune)
    with torch.no_grad():
        for module in net.modules():
            if hasattr(module, "lora_B"):
                module.lora_B.normal_(0.0, 0.05)
            if hasattr(module, "lora_qkv_B"):
                module.lora_qkv_B.normal_(0.0, 0.05)
    batch = make_batch(net)
    mem_before, loss_before = outputs(net, batch)

    merged = merge_lora(net)
    check(not any(is_lora_key(k) for k in merged),
          "merge_lora left adapter keys in the state dict")
    mem_after, loss_after = outputs(net, batch)
    # Folding `scale * B @ A` into the base weight is exact algebraically but
    # not bit-exactly in floating point (`(W + sBA)x` vs `Wx + sBA x`), so this
    # is an allclose, unlike the zero-init identity check above (where the delta
    # is exactly zero and `x + 0` is bit-exact).
    check(torch.allclose(mem_before, mem_after, atol=1e-5),
          "merge_lora changed the merged model's own output (memory)")
    check(torch.allclose(loss_before, loss_after, atol=1e-5),
          "merge_lora changed the merged model's own loss")

    plain = CodeTransformer(make_config(finetune=False)).eval()
    check(set(merged) == set(plain.state_dict()),
          "merged state dict keys != a plain model's keys")
    load_state_dict_mismatch(plain, merged)
    mem_ref, loss_ref = outputs(plain, batch)
    check(torch.allclose(mem_before, mem_ref, atol=1e-6),
          f"merged model memory differs by "
          f"{(mem_before - mem_ref).abs().max().item():.2e}")
    check(torch.allclose(loss_before, loss_ref, atol=1e-5),
          f"merged model loss differs by "
          f"{(loss_before - loss_ref).abs().item():.2e}")
    print("  merge: plain state dict reproduces the adapted model: OK")


def test_guards():
    torch.manual_seed(0)
    net = CodeTransformer(make_config()).eval()
    plain = CodeTransformer(make_config(finetune=False)).eval()
    sd = prefixed(net.state_dict())
    plain_sd = prefixed(plain.state_dict())

    def must_raise(fn, what):
        try:
            fn()
        except SystemExit:
            return
        raise SystemExit(f"FAIL: {what} was accepted")

    # Base loader: dropping adapters is never forgiven.
    must_raise(lambda: load_state_dict_mismatch(plain, net.state_dict()),
               "load_state_dict_mismatch dropping adapters")

    # --resume_from of a plain checkpoint into a LoRA model.
    must_raise(lambda: check_lora_layout(plain_sd, net, "plain.ckpt",
                                         strict=True),
               "resume_from a plain checkpoint into a LoRA model")
    # --transfer_from is the intended path (zero-init adapters = identity).
    check_lora_layout(plain_sd, net, "plain.ckpt", strict=False)

    # A LoRA checkpoint into a plain model, in either direction of `strict`.
    must_raise(lambda: check_lora_layout(sd, plain, "lora.ckpt"),
               "LoRA checkpoint into a non-LoRA model")

    # rank / alpha / targets are part of the contract.
    other_rank = CodeTransformer(make_config(rank=8)).eval()
    must_raise(lambda: check_lora_layout(prefixed(other_rank.state_dict()),
                                         net, "rank8.ckpt"),
               "rank mismatch")
    other_alpha = CodeTransformer(make_config(alpha=4.0)).eval()
    must_raise(lambda: check_lora_layout(prefixed(other_alpha.state_dict()),
                                         net, "alpha4.ckpt"),
               "alpha mismatch")
    fewer = CodeTransformer(make_config(
        targets=["decoder_ffn", "decoder_attn"])).eval()   # no encoder adapters
    must_raise(lambda: check_lora_layout(sd, fewer, "full.ckpt"),
               "checkpoint adapters missing from finetune.targets")
    # ...and the accepted case must not raise.
    same = copy.deepcopy(net)
    check_lora_layout(sd, same, "same.ckpt")
    print("  layout guards: all silent-drop paths refused: OK")


def main():
    torch.manual_seed(0)
    test_config_validation()
    test_identity_and_keys()
    test_freezing()
    test_gradients()
    test_merge()
    test_guards()
    print("ALL LORA CHECKS PASSED")


if __name__ == "__main__":
    main()
