#!/usr/bin/env python3
"""Semantic-only probe: does the text-derived hint channel carry the sentence?

Three paired arms, same text / speaker / seed, every *other* hint channel at
level 0 (the trained null state):

* ``null``     -- semantic absent (``sem_null``)
* ``own``      -- semantic = bge embedding of the current text
* ``swapped``  -- semantic = bge embedding of the *next* text

``own-vs-null`` answers "is the channel live at all"; ``own-vs-swapped``
answers "does the vector's content matter".  If the sentence is carried by the
channel, own should track null longer than swapped does.

Free-running code agreement is reported per prefix window because sampling
divergence compounds: the first frames are the cleanest signal.  With
``--audio`` (the default when the config has a stage-1 generator) each arm is
decoded and the prosody readouts are averaged (duration, decoded f0 level and
spread, frame-energy variance).

This is the OOD-text companion of ``hint_channel_probe.py`` (teacher-forced CE)
and ``hint_knob_report.py`` (level sweeps).  Measured result: HINTS_PLAN.md §7.

Usage
-----
    python scratch/hint_semantic_probe.py \
        --checkpoint checkpoints_stage2/hints_v1_p4_final_mono1/last.ckpt
    python scratch/hint_semantic_probe.py --no_audio --num_texts 8 \
        --seeds 1234,99
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# fairseq (imported by the vendored NSF decoder) turns the root logger to DEBUG
# at import time; install the handler before any heavy import (AGENTS.md §6.18).
logging.basicConfig(level=logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import stage2_repl as R  # noqa: E402

DEFAULT_TEXTS = [
    "I cannot believe you actually brought a penguin to the interview.",
    "Please remember to water the plants before you leave for the weekend.",
    "The committee will reconsider the proposal in early autumn.",
    "Nobody expected the lighthouse keeper to win the baking contest.",
    "She sold her bicycle and bought a very expensive espresso machine.",
    "The train was delayed because of a fallen tree near the tunnel.",
    "We should probably stop pretending that the plan is working.",
    "A curious child once asked why the moon follows the car.",
]
WINDOWS = (5, 10, 20, 40, None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/transformer.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--generator_checkpoint", default=None,
                   help="stage-1 Generator for the acoustic arm (default: "
                        "train.generator_checkpoint)")
    p.add_argument("--no_audio", action="store_true",
                   help="codes only (skip the stage-1 decode)")
    p.add_argument("--device", default=None)
    p.add_argument("--sid", type=int, default=0)
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seeds", default="1234,99",
                   help="comma-separated torch seeds (paired draws)")
    p.add_argument("--num_texts", type=int, default=len(DEFAULT_TEXTS))
    p.add_argument("--texts", nargs="*", default=None,
                   help="override the built-in OOD sentences")
    return p.parse_args(argv)


def _match(a, b, window=None):
    n = min(a.shape[-1], b.shape[-1])
    if window:
        n = min(n, window)
    return float((a[:, :n] == b[:, :n]).float().mean())


def _audio_metrics(net_g, config, codes, sid, device):
    n_spec = codes.shape[0] - 2
    s = torch.tensor([sid], dtype=torch.long, device=device)
    with torch.no_grad():
        audio, f0 = net_g.infer_from_codes(
            codes[:n_spec].unsqueeze(1), codes[n_spec:].unsqueeze(1), s,
            return_pitch=True)
    audio = audio[0].squeeze(0).float().cpu()
    f0 = f0[0].float().cpu()
    sr = int(config.data.sampling_rate)
    voiced = f0 > 0
    med = float(f0[voiced].median()) if bool(voiced.any()) else 0.0
    if bool(voiced.any()) and med > 0:
        st = torch.log2(f0[voiced] / med) * 12
        spread = float(st.std()) if st.numel() > 1 else 0.0
    else:
        spread = 0.0
    frame = max(1, int(sr * 0.02))
    nf = audio.numel() // frame
    if nf:
        rms = audio[: nf * frame].reshape(nf, frame).pow(2).mean(1).sqrt()
        cv = float(rms.std() / rms.mean()) if float(rms.mean()) > 0 else 0.0
    else:
        cv = 0.0
    return {"dur": audio.numel() / sr, "f0": med, "spread": spread, "cv": cv}


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    texts = list(args.texts or DEFAULT_TEXTS)[: args.num_texts]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    gen_ckpt = (None if args.no_audio else
                (args.generator_checkpoint
                 or config.train.get("generator_checkpoint", None)))

    repl = R.Stage2REPL(config, args.checkpoint, device,
                        generator_checkpoint=gen_ckpt,
                        audio=not args.no_audio, out_dir="/tmp/semantic_probe")
    repl.frames = args.frames
    repl.temperature = args.temp
    repl.top_k = args.top_k
    repl.hint_mode = "null"       # every other channel stays absent
    repl.load_models()
    net_g = None if args.no_audio else repl._generator()

    def sample(text, arm, seed, swap=None):
        repl.set_text(text)
        repl.hint_semantic = "null" if arm == "null" else "text"
        repl.hint_semantic_text = swap
        torch.manual_seed(seed)
        phone = torch.tensor([repl.phone_ids], dtype=torch.long, device=device)
        plen = torch.tensor([len(repl.phone_ids)], dtype=torch.long,
                            device=device)
        sid = torch.tensor([args.sid], dtype=torch.long, device=device)
        codes, _ = repl.net.generate(
            phone, plen, sid, max_frames=args.frames,
            temperature=args.temp, top_k=args.top_k, return_lengths=True,
            hints=repl._hints_for(len(repl.phone_ids)))
        codes = codes[0]
        out = {"codes": codes}
        if net_g is not None:
            out.update(_audio_metrics(net_g, config, codes, args.sid, device))
        return out

    pairs = {"own-vs-null": [], "swapped-vs-null": [], "own-vs-swapped": []}
    audio = {"null": [], "own": [], "swapped": []}
    for i, text in enumerate(texts):
        other = texts[(i + 1) % len(texts)]
        for seed in seeds:
            null = sample(text, "null", seed)
            own = sample(text, "own", seed)
            swp = sample(text, "swapped", seed, swap=other)
            pairs["own-vs-null"].append((own["codes"], null["codes"]))
            pairs["swapped-vs-null"].append((swp["codes"], null["codes"]))
            pairs["own-vs-swapped"].append((own["codes"], swp["codes"]))
            for name, m in (("null", null), ("own", own), ("swapped", swp)):
                audio[name].append(m)

    print(f"\n{len(texts)} texts x {len(seeds)} seed(s), free-running code "
          f"agreement (%)")
    head = "  ".join(f"{'<' + str(w) if w else 'full':>10}" for w in WINDOWS)
    print(f"{'pair':<18} {head}")
    for name, rows in pairs.items():
        vals = [sum(_match(a, b, w) for a, b in rows) / len(rows) * 100
                for w in WINDOWS]
        print(f"{name:<18} " + "  ".join(f"{v:>10.1f}" for v in vals))

    if net_g is not None:
        print("\nmean prosody by arm (decoded with the frozen stage-1 Generator)")
        print(f"{'arm':<10} {'dur s':>7} {'f0 Hz':>7} {'f0 st':>7} "
              f"{'energy cv':>10} {'frames':>7}")
        for name, ms in audio.items():
            f = lambda k: sum(m[k] for m in ms) / len(ms)  # noqa: E731
            n_frames = sum(m["codes"].shape[-1] for m in ms) / len(ms)
            print(f"{name:<10} {f('dur'):>7.2f} {f('f0'):>7.0f} "
                  f"{f('spread'):>7.2f} {f('cv'):>10.3f} "
                  f"{n_frames:>7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
