"""Fold a stage-2 LoRA fine-tune (``--finetune``) into a plain checkpoint.

A LoRA run (``modeling/lora.py``) writes its adapters next to the base weights,
so its checkpoints need ``finetune.enabled: true`` (same rank/alpha/targets) to
be loaded at all -- by design, because dropping them would silently evaluate the
*base* model.  This script turns such a checkpoint into an ordinary one: it
folds ``scale * B @ A`` into every adapted weight (attention qkv and out_proj
included), checks that the merged state dict has exactly the keys a
``finetune.enabled: false`` model expects, and verifies on a fixed probe batch
that the merged model reproduces the adapted one before writing anything.

Usage:

    python scratch/merge_lora.py \
        --checkpoint checkpoints_stage2/<exp>/last.ckpt \
        --config configs/transformer_mlp.yaml \
        --out checkpoints_stage2/<exp>/merged.ckpt

The output is a Lightning-style ``{"state_dict": {"net.*": ...}}`` file, i.e.
what ``stage2_repl.py``, ``eval_stage2_wer.py`` and ``eval_stage2_checkpoints.py``
already read, and what ``--transfer_from`` wants.  ``--config`` must be the
config the run was trained with (it defines the model shape and the adapter
targets); ``--checkpoint`` only carries the weights, since Lightning's
``hparams.yaml`` is empty for these runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Claim the root logger before the heavy imports (fairseq turns it to DEBUG; see
# AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commons import load_submodule_prefix  # noqa: E402
from modeling.lora import (check_lora_layout, is_lora_key,  # noqa: E402
                           merge_lora, uses_lora)
from modeling.transformer import CodeTransformer  # noqa: E402


def make_probe(model, T: int = 16, B: int = 2):
    """A deterministic batch with no data dependency (shape-only probe)."""
    g = torch.Generator().manual_seed(0)
    n_spec = model.spec_codebook_num
    vocab = [int(v) for v in model.vocab_sizes]

    def codes(first: int, count: int):
        return torch.stack([
            torch.randint(0, vocab[first + i], (B, T), generator=g)
            for i in range(count)]).permute(1, 0, 2)

    sid_max = max(int(model.decoder.spk_emb.num_embeddings) - 1, 1) \
        if model.decoder.spk_emb is not None else 0
    return {
        "phone_ids": torch.randint(0, 32, (B, 6), generator=g),
        "phone_lengths": torch.tensor([6, 5]),
        "sid": torch.tensor([0, min(1, sid_max)]),
        "spec_codes": codes(0, n_spec),
        "pitch_codes": codes(n_spec, len(vocab) - n_spec),
        "code_lengths": torch.tensor([T, T - 2]),
        "complete": torch.tensor([True, False]),
    }


@torch.no_grad()
def probe_outputs(model, batch):
    memory, _ = model.encode(batch["phone_ids"], batch["phone_lengths"],
                             batch["sid"], hints=None)
    loss = model.compute_batch_loss(batch, hints=None)["loss"]
    return memory.clone(), loss.detach().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="LoRA stage-2 checkpoint (last/best-checkpoint)")
    ap.add_argument("--config", required=True,
                    help="the config the run was trained with")
    ap.add_argument("--out", required=True, help="plain checkpoint to write")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if not uses_lora(state):
        raise SystemExit(f"{args.checkpoint} has no LoRA adapters; nothing to "
                         f"merge (it is already a plain checkpoint).")

    cfg = OmegaConf.load(args.config)
    if cfg.get("finetune", None) is None:
        cfg["finetune"] = OmegaConf.create({})
    cfg.finetune.enabled = True
    # rank/alpha are recorded in the marker; take them from there so a config
    # edited since the run still merges (a *wrong* rank would change the
    # adapter shapes and check_lora_layout would refuse it anyway).
    marker = state.get("net.lora_marker")
    if marker is not None and marker.numel() >= 2:
        rank, alpha = int(marker[0]), float(marker[1])
        if int(cfg.finetune.get("rank", -1)) != rank:
            print(f"NOTE: finetune.rank {cfg.finetune.get('rank')} in the "
                  f"config != {rank} in {args.checkpoint}; using {rank}.")
            cfg.finetune.rank = rank
        if abs(float(cfg.finetune.get("alpha", -1.0)) - alpha) > 1e-6:
            cfg.finetune.alpha = alpha

    net = CodeTransformer(cfg)
    if not net.lora_enabled:
        raise SystemExit(
            "The model built from --config has no adapters: finetune.targets "
            "does not match the run.  Use the run's own config.")
    check_lora_layout(state, net, args.checkpoint)
    load_submodule_prefix(net, "net.", state)
    net.eval()

    # Merge, then compare: (a) the merged model must still be the same
    # function, and (b) a *plain* model must reproduce it from the merged state
    # dict.  (a) is exact up to fp rounding; (b) is the real target.
    probe = make_probe(net)
    mem_adapt, loss_adapt = probe_outputs(net, probe)
    merged = merge_lora(net)
    mem_merged, loss_merged = probe_outputs(net, probe)
    d_mem = float((mem_adapt - mem_merged).abs().max())
    d_loss = float((loss_adapt - loss_merged).abs())
    if not (d_mem < 1e-4 and d_loss < 1e-4):
        raise SystemExit(f"merge changed the model (max|dmem| {d_mem:.2e}, "
                         f"|dloss| {d_loss:.2e}); refusing to write {args.out}")

    cfg_plain = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_plain.finetune.enabled = False
    plain = CodeTransformer(cfg_plain).eval()
    missing = sorted(set(plain.state_dict()) ^ set(merged))
    if missing:
        raise SystemExit(f"merged state dict and a plain model disagree on "
                         f"{len(missing)} key(s), e.g. {missing[:3]}")
    load_submodule_prefix(plain, "", merged)
    mem_plain, loss_plain = probe_outputs(plain, probe)
    d_ref = float((mem_adapt - mem_plain).abs().max())
    if not torch.allclose(mem_adapt, mem_plain, atol=1e-4):
        raise SystemExit(f"plain model does not reproduce the adapted one "
                         f"(max|dmem| {d_ref:.2e}); refusing to write {args.out}")

    out_state = {f"net.{k}": v for k, v in merged.items() if not is_lora_key(k)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "state_dict": out_state,
        "global_step": int(ckpt.get("global_step", -1)),
        "epoch": int(ckpt.get("epoch", -1)),
    }, args.out)
    print(f"Merged {len(merged)} tensors -> {args.out}")
    print(f"  adapters folded, probe: max|dmem| {d_mem:.2e}, "
          f"|dloss| {d_loss:.2e}; plain model max|dmem| {d_ref:.2e}")


if __name__ == "__main__":
    main()
