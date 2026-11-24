"""Initialise a low-rank speaker layout from an existing full-table checkpoint.

``transformer.spk_rank > 0`` replaces each ``nn.Embedding(n_speakers, hidden)``
with a short per-id code plus a **shared** ``rank -> hidden`` projection.  The
shipped v2.5 experiment built that layout with ``spk_init: zero``, i.e. every
speaker started at the *same* vector, so the model first learned one averaged
voice and the codes stayed tiny (measured: speaker vectors 5x/23x smaller than
the full-table run, effective rank 8.5/12 instead of 120/305, and the CE cost of
a *random* speaker id fell from +0.289 to +0.170 nats).  That is audible as
speakers blending together.

This script writes a transfer checkpoint whose low-rank speaker parameters are
the **truncated SVD** of an existing full speaker table::

    W (n_speakers, hidden) ~= U S Vh
    code  = U[:, :r] * S[:r]          # what the new embedding holds
    proj  = Vh[:r].T                  # (hidden, r), nn.Linear(rank, hidden)
    code @ proj.T == W's best rank-r approximation

so the new run starts from the voices the old model already had instead of from
zero, and every other weight still comes from ``--base_checkpoint``.  This is
the intended warm start for a low-rank ablation: no re-preprocessing, no code
changes, just a different initialisation.

Usage::

    python scratch/init_lowrank_speakers.py \
        --base_checkpoint checkpoints_stage2/transformer_hints_v2.5/last.ckpt \
        --speaker_checkpoint checkpoints_stage2/transformer_hints_v2.4/last.ckpt \
        --rank 256 \
        --out /tmp/opencode/transformer_hints_v2.6_init.ckpt

Then set ``transformer.spk_rank: 256`` / ``spk_norm: none`` /
``spk_init: zero`` (the marker must match the model; the values themselves are
the SVD rows), a fresh ``exp_name`` and::

    python 2_training.py --config <cfg> --transfer_from <out>

The checkpoint carries ``net.spk_layout_marker`` for ``--rank`` and for the
``--norm``/``--film``/``--kl`` flags, so ``check_special_layout`` accepts it only
into a model with the same layout.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rank_approx(table: torch.Tensor, rank: int):
    """Return ``(code, proj)`` whose product is the best rank-``r`` fit."""
    W = table.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.numel())
    code = U[:, :r] * S[:r]
    proj = Vh[:r].T.contiguous()          # (hidden, r)
    return code, proj


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base_checkpoint", required=True,
                    help="weights for everything except the speaker tables")
    ap.add_argument("--speaker_checkpoint", default=None,
                    help="full-table checkpoint to decompose (default: base)")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--spk_norm", choices=("none", "l2"), default="none")
    ap.add_argument("--spk_film_memory", action="store_true")
    ap.add_argument("--spk_kl_weight", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = torch.load(args.base_checkpoint, map_location="cpu",
                      weights_only=False, mmap=True)
    sd = dict(base["state_dict"])
    spk_path = args.speaker_checkpoint or args.base_checkpoint
    spk = torch.load(spk_path, map_location="cpu", weights_only=False,
                     mmap=True)["state_dict"]

    for site in ("encoder", "decoder"):
        key = f"net.{site}.spk_emb.weight"
        if key not in spk:
            raise SystemExit(f"{spk_path} has no {key}; it is not a full-table "
                             f"speaker checkpoint")
        table = spk[key]
        code, proj = rank_approx(table, args.rank)
        sd.pop(key, None)
        sd[f"net.{site}.spk_lowrank.code.weight"] = code
        sd[f"net.{site}.spk_lowrank.proj.weight"] = proj
        # Reconstruction check on a sample of ids, so a silent axis mix-up
        # cannot ship as "the voices are back".
        idx = torch.linspace(0, table.shape[0] - 1, 64).long()
        err = (code[idx] @ proj.T - table.float()[idx]).norm()
        rel = err / table.float()[idx].norm().clamp_min(1e-6)
        print(f"{site}: rank {code.shape[1]}  relative reconstruction error "
              f"{float(rel):.4f}")

    sd["net.spk_layout_marker"] = torch.tensor([
        float(args.rank),
        1.0 if args.spk_norm == "l2" else 0.0,
        1.0,                                  # spk_init: zero in the new config
        1.0 if args.spk_film_memory else 0.0,
        1.0 if args.spk_kl_weight > 0 else 0.0,
    ])

    out = {
        "state_dict": sd,
        "epoch": base.get("epoch"),
        "global_step": base.get("global_step"),
        "pytorch-lightning_version": base.get("pytorch-lightning_version"),
        "hparams_name": base.get("hparams_name", "hparams"),
        "hyper_parameters": base.get("hyper_parameters", {}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e9:.2f} GB)")
    print(f"base weights: {args.base_checkpoint}")
    print(f"speaker tables: {spk_path}")
    print(f"marker: rank={args.rank} norm={args.spk_norm} film="
          f"{args.spk_film_memory} kl={args.spk_kl_weight}")


if __name__ == "__main__":
    main()
