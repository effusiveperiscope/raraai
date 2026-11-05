# Inference cost: stage-2 `CodeTransformer` + stage-1 `Generator`

Measured on this box (RTX 5060 Ti 16 GB, fp32/TF32, `torch.set_grad_enabled(False)`)
while a stage-1 training job held the GPU at ~90 % util, so the latencies below
are conservative by roughly 10-30 %. Random-init weights (timing does not depend
on the values). Reproduce with `scratch/infer_cost_bench*.py`.

## Fixed operating point

| quantity | value | source |
|---|---|---|
| audio rate | 48 kHz, hop 480 | `configs/base.yaml` |
| decoder frame rate | 100 Hz | `sr / hop_length` |
| `down_factor` | 4 | `modeling/generator.py` |
| **code rate** | **25 codes/s** (`T = audio_seconds * 25`) | same |
| codebooks `K` | 8 spec (vocab 64+2) + 2 pitch (vocab 128+2) | configs |
| transformer AR steps for T frames | `T + K - 1 = T + 9` | `generate` delay pattern |
| Stage-2 params | 135.9 M (encoder 37.0 / decoder 98.9) | measured |
| Stage-1 params | 33.7 M total, **22.8 M** on the code→audio path | measured |

## Latency / throughput (one 5.12 s utterance = 128 code frames = 137 AR steps)

| stage | B | wall clock | audio produced | RTF | real-time factor |
|---|---|---|---|---|---|
| stage 2 `generate` | 1 | 755 ms | 5.12 s | 0.148 | 6.8x |
| stage 2 `generate` | 2 | 922 ms | 10.24 s | 0.090 | 11.1x |
| stage 2 `generate` | 4 | 1064 ms | 20.48 s | 0.052 | 19.2x |
| stage 2 `generate` | 8 | 1535 ms | 40.96 s | 0.037 | 26.7x |
| stage 1 `infer_from_codes` | 1 | 74 ms | 5.12 s | 0.014 | 70x |
| stage 1 `infer_from_codes` | 16 | 1440 ms | 81.9 s | 0.018 | 57x |
| **pipeline, text -> wav** | 1 | **~830 ms** | 5.12 s | **0.163** | **6.1x** |
| text front end (g2p only, CPU) | 1 | 0.3-0.7 ms / sentence | - | - | negligible |

Stage 2 is 91 % of the pipeline. Doubling the length (T=256) costs
1.58 s -> RTF 0.154, i.e. RTF is roughly flat from ~1 s to ~10 s utterances
because per-step overhead dominates (see below).

## Compute

| stage | FLOPs (fwd) for 5.12 s | FLOPs per audio-second |
|---|---|---|
| stage 2, T=128 | 1683 GFLOP | 329 GFLOP/s |
| stage 2, T=256 | 5997 GFLOP | 586 GFLOP/s |
| stage 1 | 565 GFLOP | 110 GFLOP/s |

Effective utilisation at B=1: 1683 GFLOP / 0.755 s = **2.2 TFLOP/s of a machine
that does 16.6 TFLOP/s fp32** (measured 4096^3 matmul) -> 13 %. At B=8 it reaches
8.8 TFLOP/s (53 %). Batch-1 decoding throws away ~87 % of the GPU.

## Where stage-2 time goes (B=1, L=137)

| item | time |
|---|---|
| one decoder forward over the whole 137-column prefix | 5.20 ms |
| x 137 steps | ~712 ms (94 % of `generate`) |
| sampling (10 books: clone/top-k/softmax/multinomial) | 0.45 ms/step = 62 ms (6 %) |
| phoneme encoder (once per clip) | 0.72 ms |

`generate` re-runs the decoder on the full prefix every step (no KV cache), so
FLOPs are O(T^2). But a single-token forward still costs **3.5 ms** on this box
(measured: L=1 -> 3.5 ms, L=137 -> 5.3 ms, L=265 -> 4.6 ms), i.e. the step is
**launch/dispatch bound, not math bound**. A KV cache removes 68x of the FLOPs
at T=128 (26 GFLOP instead of 1683) but, at B=1, only the 1.8 ms marginal part
of each step, so expect ~1.5-2x wall clock unless it is combined with
`torch.compile`/CUDA graphs; batching is the cheap lever that already works.

## Money / energy per audio-hour (3600 s of speech = 90,000 code frames)

| scenario | GPU-seconds | GPU-hours | @ $0.25/GPU-h | energy @ 180 W |
|---|---|---|---|---|
| B=1 (a single stream) | 587 s | 0.163 | ~$0.04 | ~0.03 kWh |
| B=8 (server, batched) | 198 s | 0.055 | ~$0.014 | ~0.01 kWh |

So ~1 GPU-minute per 6 audio-minutes unbatched, ~1 per 18 minutes batched; the
marginal cost of a 5-second utterance is ~0.2 GPU-seconds (~5e-5 $), i.e. the
economics are dominated by **queueing/idle**, not by the arithmetic.

FLOP per audio-hour: 1.18 PFLOP (stage 2, T=128) + 0.40 PFLOP (stage 1).

## Memory

* weights fp32: transformer 544 MiB + generator 135 MiB; bf16 halves that.
* peak torch allocation for a B=1 pipeline run: **994 MiB**.
* peak *reserved*: **6.1 GiB** -- the `torch.cat` growth of the prefix plus 1370
  small sampling kernels fragment the allocator. Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (or preallocate the prefix
  buffer) before sizing a deployment, otherwise each process holds ~6 GiB.

## Not included

* ASR + forced alignment (`Qwen3-ASR-0.6B` + aligner): not needed when the text
  is known; if you transcribe input audio instead, that 0.6 B model on a few
  seconds of audio costs far more than everything above.
* `Generator.enc_q` (8.8 M params, 26 % of the stage-1 model) and
  `codec.encoder` are encode-only and are pure dead weight at synthesis time.
* No trained stage-2 checkpoint exists on this box (`checkpoints_stage2/` is
  absent), so these are cost numbers only -- nothing here says anything about
  audio quality.
