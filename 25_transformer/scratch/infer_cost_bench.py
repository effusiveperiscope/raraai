"""Measure/estimate inference cost of the stage-2 transformer + stage-1 Generator.

Counts parameters, MAC/FLOP counts (forward only) and wall-clock latency for
the two things inference actually runs:

  stage 2  CodeTransformer.generate(phone_ids, lens, sid, max_frames=T)
           -> (B, K=10, T) codes at 25 Hz (T code frames == T/25 s of audio)
  stage 1  Generator.infer_from_codes(spec_codes, pitch_codes, sid)
           -> (B, 1, T*4*hop) waveform at 48 kHz

Run with the raraai25 env python from the repo root.
"""
from __future__ import annotations

import time

import torch
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

from modeling.generator import Generator
from modeling.transformer import CodeTransformer

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

HOP = 480
SR = 48000
DOWN = 4
CODE_RATE = SR / HOP / DOWN  # 25 codes / second


def nparams(mod: torch.nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def flops(fn) -> int:
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def bench_gpu(fn, reps=5, warmup=2):
    if not torch.cuda.is_available():
        return float("nan")
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    cfg_t = OmegaConf.load("configs/transformer.yaml")
    cfg_g = OmegaConf.load("configs/base.yaml")

    tr = CodeTransformer.from_config(cfg_t).eval()
    gen = Generator(cfg_g).eval()

    print("=" * 72)
    print("PARAMETERS")
    print("=" * 72)
    print(f"CodeTransformer total      : {nparams(tr)/1e6:8.2f} M")
    for name, m in (("  encoder", tr.encoder), ("  decoder", tr.decoder)):
        print(f"{name:26s} : {nparams(m)/1e6:8.2f} M")
    print(f"  decoder per book embeds  : "
          f"{sum(nparams(e) for e in tr.decoder.embeds)/1e6:8.2f} M "
          f"({len(tr.decoder.embeds)} books, classes={tr.decoder.num_classes})")
    print(f"  decoder per book heads   : "
          f"{sum(nparams(h) for h in tr.decoder.heads)/1e6:8.2f} M")
    print(f"  decoder spk_emb          : {nparams(tr.decoder.spk_emb)/1e6:8.2f} M")
    print(f"Generator total            : {nparams(gen)/1e6:8.2f} M")
    for name in ("enc_q", "codec", "dec", "spk_emb", "downsample",
                 "upsample", "pitch_codec", "pitch_downsample",
                 "pitch_upsample", "pitch_out_proj", "pitch_uv_proj"):
        m = getattr(gen, name, None)
        if m is not None:
            print(f"  gen.{name:20s} : {nparams(m)/1e6:8.2f} M")
    print(f"Generator (decode path only: codec.decoder + pitch dec path + dec + "
          f"spk_emb) : "
          f"{(nparams(gen.codec.decoder) + nparams(gen.pitch_codec.decoder) + nparams(gen.dec) + nparams(gen.spk_emb) + nparams(gen.upsample) + nparams(gen.pitch_upsample) + nparams(gen.pitch_out_proj) + nparams(gen.pitch_uv_proj))/1e6:.2f} M")

    K = tr.codebook_num
    n_ph = 40  # ~40 phonemes for a short sentence
    sid = torch.zeros(1, dtype=torch.long)
    phone_ids = torch.randint(0, 100, (1, n_ph))
    phone_lens = torch.tensor([n_ph])

    print()
    print("=" * 72)
    print("FLOPs (forward only, 2 flops per MAC)")
    print("=" * 72)
    for T in (128, 256):
        d_t = flops(lambda: tr.generate(phone_ids, phone_lens, sid, T,
                                        temperature=1.0, top_k=None,
                                        stop_on_eos=False))
        secs = T / CODE_RATE
        print(f"stage2 generate T={T:4d} ({secs:5.2f} s audio): "
              f"{d_t/1e9:8.2f} GFLOP  -> {d_t/secs/1e9:7.2f} GFLOP/s-of-audio")

    for T in (32, 128):
        sc = torch.randint(0, 64, (8, 1, T))
        pc = torch.randint(0, 128, (2, 1, T))
        f_dec = flops(lambda: gen.infer_from_codes(sc, pc, sid))
        secs = T / CODE_RATE
        print(f"stage1 decode  T={T:4d} ({secs:5.2f} s audio): "
              f"{f_dec/1e9:8.2f} GFLOP  -> {f_dec/secs/1e9:7.2f} GFLOP/s-of-audio")

    print()
    print("=" * 72)
    print("WALL CLOCK (this box, RTX 5060 Ti; ~contended by a running "
          "stage-1 job)")
    print("=" * 72)
    if torch.cuda.is_available():
        tr_g = tr.cuda()
        gen_g = gen.cuda()
        pid = phone_ids.cuda()
        pl = phone_lens.cuda()
        sd = sid.cuda()
        for T in (128, 256):
            ms = bench_gpu(lambda: tr_g.generate(pid, pl, sd, T,
                                                 temperature=1.0, top_k=None,
                                                 stop_on_eos=False), reps=3)
            secs = T / CODE_RATE
            print(f"stage2 generate T={T:4d} ({secs:5.2f} s audio): "
                  f"{ms:8.1f} ms  -> RTF {ms/1000/secs:6.3f}")
        for T in (128,):
            sc = torch.randint(0, 64, (8, 1, T)).cuda()
            pc = torch.randint(0, 128, (2, 1, T)).cuda()
            ms = bench_gpu(lambda: gen_g.infer_from_codes(sc, pc, sd), reps=5)
            secs = T / CODE_RATE
            print(f"stage1 decode  T={T:4d} ({secs:5.2f} s audio): "
                  f"{ms:8.1f} ms  -> RTF {ms/1000/secs:6.3f}")
    else:
        print("no CUDA device")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n(total {time.time()-t0:.1f} s)")
