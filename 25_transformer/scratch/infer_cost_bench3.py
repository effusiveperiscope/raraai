"""Decoder forward cost vs sequence length -- what a KV cache would buy."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from modeling.transformer import CodeTransformer

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


def bench(fn, reps=10, warmup=3):
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
    cfg = OmegaConf.load("configs/transformer.yaml")
    tr = CodeTransformer.from_config(cfg).eval().cuda()
    K = tr.codebook_num
    pid = torch.randint(0, 100, (1, 40), device="cuda")
    pl = torch.full((1,), 40, device="cuda")
    sid = torch.zeros(1, dtype=torch.long, device="cuda")
    mem, mem_mask = tr.encode(pid, pl, sid)

    print(f"{'L':>6} {'B':>3} {'ms/fwd':>9} {'tokens':>7} {'GFLOP':>9} "
          f"{'TFLOP/s':>9}")
    for B, L in [(1, 1), (1, 2), (1, 10), (1, 37), (1, 69), (1, 137),
                 (1, 265), (8, 1), (8, 137)]:
        seq = torch.zeros(B, K, L, dtype=torch.long, device="cuda")
        m = mem.expand(B, -1, -1)
        mm = mem_mask.expand(B, -1) if mem_mask is not None else None
        s = sid.expand(B)
        ms = bench(lambda: tr.decoder(seq, m, mm, sid=s))
        tokens = B * L
        # 12 layers * (4 + 4 + 2*2048/768) * 768^2 MACs per token, x2 flops
        per_tok = 2 * 12 * (4 + 4 + 2 * 2048 / 768) * 768 ** 2
        gflop = tokens * per_tok / 1e9
        print(f"{L:6d} {B:3d} {ms:9.3f} {tokens:7d} {gflop:9.1f} "
              f"{gflop/(ms/1000)/1000:9.2f}")


if __name__ == "__main__":
    main()
