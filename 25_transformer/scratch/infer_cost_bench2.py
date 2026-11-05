"""Second pass: batch scaling, sampling-overhead split, peak memory, text front end."""
from __future__ import annotations

import time

import torch
from omegaconf import OmegaConf

from modeling.transformer import CodeTransformer
from modeling.generator import Generator

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

RATE = 25.0  # code frames per second of audio


def bench(fn, reps=3, warmup=1):
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    cfg_t = OmegaConf.load("configs/transformer.yaml")
    cfg_g = OmegaConf.load("configs/base.yaml")
    dev = "cuda"
    tr = CodeTransformer.from_config(cfg_t).eval().to(dev)
    gen = Generator(cfg_g).eval().to(dev)

    print("=" * 74)
    print("STAGE-2 generate: batch scaling (T=128 codes = 5.12 s audio)")
    print("=" * 74)
    for B in (1, 2, 4, 8):
        pid = torch.randint(0, 100, (B, 40), device=dev)
        pl = torch.full((B,), 40, device=dev)
        sid = torch.zeros(B, dtype=torch.long, device=dev)
        ms = bench(lambda: tr.generate(pid, pl, sid, 128, temperature=1.0,
                                       top_k=None, stop_on_eos=False))
        print(f"  B={B:2d}: {ms:8.1f} ms  -> {ms/1000:6.3f} ms/clip  "
              f"audio {5.12*B:6.2f} s in {ms/1000:5.2f} s  RTF {ms/1000/(5.12*B):6.3f}  "
              f"({(5.12*B)/(ms/1000):5.1f}x realtime)")

    print()
    print("=" * 74)
    print("STAGE-2 where the time goes (B=1, T=128)")
    print("=" * 74)
    pid = torch.randint(0, 100, (1, 40), device=dev)
    pl = torch.full((1,), 40, device=dev)
    sid = torch.zeros(1, dtype=torch.long, device=dev)
    # a) decoder forward on a full 137-column prefix, once
    K = tr.codebook_num
    L = 128 + K - 1
    seq = torch.zeros(1, K, L, dtype=torch.long, device=dev)
    mem, mem_mask = tr.encode(pid, pl, sid)
    ms_fwd = bench(lambda: tr.decoder(seq, mem, mem_mask, sid=sid), reps=5)
    print(f"  one decoder fwd, L={L} columns      : {ms_fwd:8.2f} ms")
    # b) encoder only (run once per clip)
    ms_enc = bench(lambda: tr.encode(pid, pl, sid), reps=5)
    print(f"  phoneme encoder (once per clip)     : {ms_enc:8.2f} ms")
    # c) sampling ops per step, K books
    logits = [torch.randn(1, 1, c, device=dev) for c in tr.decoder.num_classes]

    def sample_only():
        for lg in logits:
            lg = lg.clone()
            lg[..., -2] = float("-inf")
            probs = torch.softmax(lg, dim=-1).view(-1, lg.shape[-1])
            torch.multinomial(probs, num_samples=1)
    ms_samp = bench(sample_only, reps=20)
    print(f"  sampling ops, all {K} books, 1 step  : {ms_samp:8.3f} ms")
    print(f"  => generate step is dominated by the full-prefix decoder "
          f"forward (no KV cache)")

    print()
    print("=" * 74)
    print("STAGE-1 Generator.infer_from_codes: batch scaling")
    print("=" * 74)
    for B in (1, 4, 16):
        T = 128
        sc = torch.randint(0, 64, (8, B, T), device=dev)
        pc = torch.randint(0, 128, (2, B, T), device=dev)
        sidg = torch.zeros(B, dtype=torch.long, device=dev)
        ms = bench(lambda: gen.infer_from_codes(sc, pc, sidg), reps=3)
        print(f"  B={B:2d}: {ms:8.1f} ms  audio {5.12*B:6.2f} s -> RTF "
              f"{ms/1000/(5.12*B):6.4f}  ({(5.12*B)/(ms/1000):6.1f}x realtime)")

    print()
    print("=" * 74)
    print("PEAK GPU MEMORY (both models loaded, B=1 T=128 pipeline)")
    print("=" * 74)
    torch.cuda.reset_peak_memory_stats()
    sc = torch.randint(0, 64, (8, 1, 128), device=dev)
    pc = torch.randint(0, 128, (2, 1, 128), device=dev)
    pid = torch.randint(0, 100, (1, 40), device=dev)
    out = tr.generate(pid, pl, sid, 128, temperature=1.0, top_k=None,
                      stop_on_eos=False)
    audio = gen.infer_from_codes(out[:, :8].transpose(0, 1),
                                 out[:, 8:].transpose(0, 1), sid)
    torch.cuda.synchronize()
    print(f"  peak allocated : {torch.cuda.max_memory_allocated()/2**20:7.1f} MiB")
    print(f"  peak reserved  : {torch.cuda.max_memory_reserved()/2**20:7.1f} MiB")
    print(f"  audio out      : {audio.shape} = {audio.shape[-1]/48000:.2f} s")

    print()
    print("=" * 74)
    print("TEXT FRONT END (CPU, g2p only -- no ASR/aligner needed at inference)")
    print("=" * 74)
    try:
        from features import FeatureExtractor
        fe = FeatureExtractor(cfg_g, device="cpu", load_qwen=False)
        texts = ["The quick brown fox jumps over the lazy dog, and then it "
                 "runs away into the forest.",
                 "Hey, you don't sound too excited about this."]
        for txt in texts:
            n_words = len(txt.split())
            t0 = time.time()
            ids = fe.phonemize_text(txt)
            print(f"  {n_words:3d} words: {(time.time()-t0)*1000:7.1f} ms "
                  f"(cold, includes dictionary load) -> {len(ids)} phone ids")
            t0 = time.time()
            for _ in range(5):
                fe.phonemize_text(txt)
            dt = (time.time() - t0) * 1000 / 5
            print(f"            {dt:7.1f} ms warm "
                  f"({dt/n_words:5.1f} ms/word)")
    except Exception as exc:  # pragma: no cover - front end is optional here
        print(f"  skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
