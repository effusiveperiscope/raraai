#!/usr/bin/env python3
"""End-of-clip transient probe for the stage-1 code -> audio decoder.

Every convolutional block in the codec decoders and in the NSF decoder
zero-pads its right edge, so the last code frame of a clip is decoded without
the right-hand context it had inside the audio.  On a quiet/unvoiced tail that
missing context turns into a ~20 ms burst right before the waveform ends -- a
small "blip"/click at the end of *every* synthesised sample, even one whose
speech stopped well before the end.  ``Generator.infer_from_codes`` now decodes
``DECODE_TAIL_FRAMES`` extra code frames (copies of the last one) and trims the
extra audio, which removes it.

This script measures the effect with the checkpoints of a stage-2 run:

* a constant *silent* code frame repeated (pure decoder test: any tail energy is
  the decoder, not the model), and
* a few real val items, decoding both their ground-truth codes and a free-
  running sample from the stage-2 checkpoint.

Unvoiced frames are generated from random noise, so every number is averaged
over ``--seeds`` decoder draws.  Prints the mean RMS of the last code frames
for ``tail_frames=0`` (the pre-fix path) and the current default; the last
frame should be at the level of the frames before it, not above them.

Usage::

    python scratch/check_decode_tail.py \
        --stage1_checkpoint checkpoints_stage1/base_v0.7/last.ckpt \
        --stage2_checkpoint checkpoints_stage2/transformer_hints_v2.6_spk128/last.ckpt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# fairseq (pulled in by modeling.generator) sets the root logger to DEBUG at
# import time; install our handler first so that call becomes a no-op
# (AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commons import load_state_dict_mismatch, load_submodule_prefix  # noqa: E402
from dataset_codes import CodeCollator  # noqa: E402
from modeling.transformer import CodeTransformer  # noqa: E402
from stage2_wer import (  # noqa: E402
    build_wer_dataset, code_frame_samples, load_sid_map)


def load_generator(config, checkpoint, device):
    """Frozen stage-1 Generator (mirrors 2_training.py:load_generator)."""
    from modeling.generator import Generator

    net_g = Generator(hp=config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net_g.") for k in state):
        load_submodule_prefix(net_g, "net_g.", state)
    else:
        load_state_dict_mismatch(net_g, state)
    return net_g.to(device).eval()


def load_transformer(config, checkpoint, device):
    net = CodeTransformer(config).to(device)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    prefix = "net." if any(k.startswith("net.") for k in sd) else ""
    load_submodule_prefix(net, prefix, sd, quiet=True)
    return net.eval()


@torch.no_grad()
def decode(net_g, codes, sid, hop, tail):
    n_spec = int(net_g.hp.codec.codebook_num)
    audio = net_g.infer_from_codes(
        codes[:, :n_spec].transpose(0, 1),
        codes[:, n_spec:].transpose(0, 1), sid, tail_frames=tail)
    audio = audio[0, 0].float().cpu().numpy()
    n = codes.shape[-1]
    return np.array([np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2))
                     for i in range(n)])


def mean_last(net_g, codes, sid, hop, tail, seeds):
    rms = None
    for s in range(seeds):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        r = decode(net_g, codes, sid, hop, tail)
        rms = r if rms is None else rms + r
    return rms / max(seeds, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--stage1_checkpoint",
                    default="checkpoints_stage1/base_v0.7/last.ckpt")
    ap.add_argument("--stage2_checkpoint",
                    default="checkpoints_stage2/transformer_hints_v2.6_spk128/last.ckpt")
    ap.add_argument("--items", default="7,1000,4242",
                    help="val dataset indices to decode")
    ap.add_argument("--seeds", type=int, default=8,
                    help="decoder noise draws to average (unvoiced frames)")
    args = ap.parse_args()

    from modeling.generator import DECODE_TAIL_FRAMES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config)
    net_g = load_generator(config, args.stage1_checkpoint, device)
    net = load_transformer(config, args.stage2_checkpoint, device)
    dataset = build_wer_dataset(config)
    coll = CodeCollator(config=config)
    hop = code_frame_samples(net_g)
    sid_map = load_sid_map(config.train.get("stage1_speaker_map", None)) or {}
    print(f"DECODE_TAIL_FRAMES = {DECODE_TAIL_FRAMES}, "
          f"{args.seeds} seed(s) per row")

    idx = int(args.items.split(",")[0])
    gt = torch.cat([dataset[idx]["spec_codes"], dataset[idx]["pitch_codes"]],
                   dim=0)
    T = int(gt.shape[-1])
    silent = gt[:, T - 1:T].expand(-1, 100).contiguous().unsqueeze(0)
    sid = torch.tensor([0], device=device)
    print("\nconstant silent code frame x100 (pure decoder test):")
    for tail in (0, DECODE_TAIL_FRAMES):
        r = mean_last(net_g, silent, sid, hop, tail, args.seeds)
        print(f"  tail={tail:>2}: steady {r[50:90].mean():.5f}  "
              f"last {r[-1]:.5f}  ratio {r[-1] / r[50:90].mean():.2f}")

    print("\nreal val items (ground-truth codes and a free-running sample):")
    for idx in (int(i) for i in args.items.split(",")):
        batch = coll([dataset[idx]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        T = int(batch["code_lengths"][0])
        torch.manual_seed(0)
        gen, _ = net.generate(
            batch["phone_ids"], batch["phone_lengths"], batch["sid"],
            max_frames=min(T, 512), temperature=0.9, top_k=None,
            return_lengths=True)
        sid = torch.tensor(
            [sid_map.get(int(batch["sid"][0]), int(batch["sid"][0]))],
            device=device)
        gtc = torch.cat([batch["spec_codes"][0], batch["pitch_codes"][0]],
                        dim=0)[:, :T]
        for label, codes in (("GT ", gtc.unsqueeze(0)), ("GEN", gen)):
            row = []
            for tail in (0, DECODE_TAIL_FRAMES):
                r = mean_last(net_g, codes, sid, hop, tail, args.seeds)
                row.append(f"tail={tail}: last {r[-1]:.4f} "
                           f"(prev {r[-2]:.4f})")
            print(f"  item {idx:>4} {label}: " + "   ".join(row))


if __name__ == "__main__":
    main()
