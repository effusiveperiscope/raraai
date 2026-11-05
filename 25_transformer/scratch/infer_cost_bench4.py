"""Contention probe: how much of the measured latency is the busy GPU?"""
from __future__ import annotations

import torch


def bench(fn, reps=20, warmup=5):
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


a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
ms = bench(lambda: a @ b)
print(f"fp32 4096^3 matmul (tf32 {torch.backends.cuda.matmul.allow_tf32}): "
      f"{ms:.2f} ms -> {2*4096**3/(ms/1000)/1e12:.1f} TFLOP/s")
torch.backends.cuda.matmul.allow_tf32 = True
ms = bench(lambda: a @ b)
print(f"wheel tf32 on: {ms:.2f} ms -> {2*4096**3/(ms/1000)/1e12:.1f} TFLOP/s")

x = torch.randn(768, 768, device="cuda")
y = torch.randn(768, 768, device="cuda")
ms = bench(lambda: [x @ y for _ in range(100)], reps=5)
print(f"100 tiny 768^3 GEMMs: {ms:.2f} ms -> {ms/100/1000:.1f} us each")

lin = torch.nn.Linear(768, 768).cuda()
z = torch.randn(1, 768, device="cuda")
ms = bench(lambda: [lin(z) for _ in range(100)], reps=5)
print(f"100 tiny Linear(1x768) launches: {ms:.2f} ms -> {ms/100/1000:.1f} us each")
