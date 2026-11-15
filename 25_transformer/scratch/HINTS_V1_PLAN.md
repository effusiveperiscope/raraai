# Conditioning hints v1.0 — plan, evidence and phases

Status: **Phase 0 in progress** (2026-11-14). `HINTS_PLAN.md` remains the spec and
status file for the *existing* (v0.8/v0.9) design; this file is the plan for the
revision, with the measurements that motivate it.

## 0. Why — the three observed problems

1. **v0.9 is perceived as less coherent than the v0.7 base.** Judged subjectively
   from the TensorBoard/REPL samples. Those samples were generated in a state the
   model was never trained in (see §1), so the comparison is not clean.
2. **Every hint channel is close to inert** in the current wiring (measured, §2),
   so the knobs do not work.
3. **The injection site cannot work for clip-level channels** (measured + algebra,
   §3), independent of whether the values are right.

## 1. The bug that contaminates every listening path

`train.hint_null_all` / `hint_dropout` train **level 0** as the "not provided"
state: `int_emb(0)`, `reg_emb(0)`, `pitch_emb(0)`, `slot_emb(0)`, `sem_null`.
Every generation path, however, passed `hints=None`, which `_enc_hints` /
`_dec_hint` interpret as "omit the vectors entirely" — a *different* function.
Measured with `scratch/hint_channel_probe.py` on the **finished**
`transformer_v0.9/last.ckpt` (step 433 455, epoch 70), 16 val items:

| state passed to `generate` | CE (teacher-forced) | Δ vs real hints |
|---|---|---|
| real `.cond` hints | 2.3142 | — |
| level-0 rows (`null`) | 2.3371 | +0.0229 |
| `None` (what every path did) | 2.4502 | **+0.1360** |

i.e. **0.113 nats of pure out-of-distribution error**, three times what the
entire trained null state is worth — and the encoder memory shifts by 52 %
(level-0 vs omitted). Greedy generation `None` vs `null` agrees on only **32 %**
of frames. (An earlier measurement at step 274 k on `best-checkpoint.ckpt` read
2.3343 / 2.3480 / 2.3850, the same ordering at a smaller magnitude.)


Paths that passed nothing: `2_training.py:_sample_and_log` (codes/audio/mel/pitch),
`_log_infill`, `_log_speaker_probe`, `_log_ood_generation`,
`scratch/stage2_repl.py`, `scratch/eval_stage2_checkpoints.py` (free-running),
`scratch/eval_stage2_wer.py` (no-hint arm).

**Phase 0 fix**: `CodeTransformer.null_hints()` + `resolve_hints()`; `None` now
means the trained null state for a hint-enabled model, `"omit"` keeps the old
behaviour for ablation. Every sample log carries the state in its tag and as
`sample/hints_mode` (0 = omit, 1 = null, 2 = real `.cond`).

## 2. Measured channel liveness (v0.9 final, step 433 455)

`scratch/hint_channel_probe.py --batches 2 --batch_size 8`, i.e. 16 val items,
teacher-forced CE with all hints = **2.3142** nats.

| channel dropped | CE | delta (nats) |
|---|---|---|
| intensity | 2.3146 | **+0.0004** |
| register | 2.3162 | +0.0020 |
| semantic | 2.3142 | **-0.0000** |
| word pitch | 2.3284 | **+0.0142** |
| word timing | 2.3199 | +0.0057 |
| all (trained null state) | 2.3371 | +0.0229 |
| omit (no vectors at all) | 2.4502 | **+0.1360** |

So on the finished model the Phase-0 bug cost **0.113 nats** — three times what
the whole trained null state is worth — which is exactly why the level-0 path
sounds better than what every earlier listening path used.

Level sweeps (same batch, CE delta vs the real levels):

- intensity `L0:+0.0456 L1:+0.0464 L2:+0.0457 L3:+0.0464 L4:+0.0466 L5:+0.0462
  L6:+0.0471 L7:+0.0483 L8:+0.0506` — monotone *worse* with level, i.e. no
  benefit from the true level at all (the table is not a knob yet).
- register `L0:+0.0464 L1:+0.0512 L2:+0.0485 L3:+0.0473 L4:+0.0460 L5:+0.0471`
  — flat/noisy, argmin at the top level.

### Is the information even in the codes?  (`scratch/hint_codespace_probe.py`)

Ridge on per-book code histograms + duration, 5-fold **GroupKFold by speaker**,
3997 clips / 719 speakers, and a leave-one-out per-speaker-mean control:

| target | R²(codes) | R²(speaker mean) |
|---|---|---|
| intensity | **0.538** | -0.100 |
| register (relative) | 0.170 | -0.000 |

Intensity is *half* the variance of the code stream beyond the speaker, yet the
channel is worth 0.0004 nats — a pure wiring failure, which is what
`hint_film_memory` / `hint_decoder_proj` and the guidance mechanism address.
The relative register is only 17 % present, which bounds that knob: it is worth
wiring, but not worth expecting a large swing. (`R²(speaker) ≈ 0` is the v0.9
relative parameterisation working as designed.)

The hint tables are *not* small: `int_emb` row norms 0.58–1.21, `pitch_emb`
0.82–1.97, `slot_emb` 1.07–2.57, `dec_int_emb` 0.73–1.68 (vs `phone_emb` 2.42) —
a lot of learned parameter for ~0 CE effect.

**Conclusion**: only the two *time-varying* (per-phone) channels do anything.
The clip-level ones are inert *at the current injection site*.


## 3. Why the clip-level channels cannot work where they are

`PhonemeEncoder` ends in `_temporal_norm` (`modeling/transformer.py:250-258`),
which standardises **each dimension over time**. That is exactly invariant to any
per-dimension affine map applied at all positions:

```
y_d(t) = g_d·(x_d(t)+c_d − mean_t(x+c)) / sqrt(var_t(x+c)+ε)
       = g_d·(x_d(t) − mean_t x_d) / sqrt(var_t(x+c)+ε)     # c_d cancels
```

So an added clip constant, a FiLM γ/β derived from it, and even a diagonal
`x ⊙ W(h)` interaction all cancel at the memory boundary; only attention
non-linearity and the std term survive. Measured: intensity 0→4 changes the
pre-norm residual by 4.0 % and the post-norm memory by 4.6 % (it *does* reach
the memory through attention) yet costs 0.0000 nats.

The decoder side is similar: the hint is added once at the input and then passes
through 12 pre-LN layers; measured, the semantic vector moves the logits by
2.6× their std but changes CE by -0.002 nats, i.e. the shift lands in a direction
the heads map to a near-uniform offset (softmax-invariant).

**Consequence**: FiLM at the same place buys nothing (same algebra). The
injection has to be applied **after** `_temporal_norm` on the encoder side, and
**after the decoder's final norm** on the decoder side.

## 4. Design decisions (and the corrections that led to them)

### 4.1 A knob needs more than availability

`val_hint_gap`/CE answers "is the channel predictive", not "is it controllable".
For a *style* attribute the CE gradient is tiny (measured), so even a correct
injection site would produce a weak knob. The mechanism that fits "coarse,
clip-level, end-user control" is **classifier-free guidance over the hint
channels at inference** (`logits = uncond + w·(cond − uncond)`, `uncond` = the
level-0 dict), which the existing per-channel dropout already trains. Per-book
weights allow guiding the pitch books without disturbing the timbre books.

Risks to measure, not assume: guidance sharpens the distribution and v0.8's
failure mode was a sharpening-induced repetition attractor, so `w` must be swept
against WER + repetition. Fallback if it underdelivers: a counterfactual margin
loss (`CE(shifted level) ≥ CE(true level) + margin`).

### 4.2 Keep all five channels

* **intensity** stays a **clip-level** channel: its purpose is a coarse,
  user-facing control of *perceptual* (emotional / phonation-effort) intensity,
  and it is deliberately **loudness-independent** (the DSP feature set
  peak-normalises). CE is the wrong KPI; the KPI is a counterfactual sweep. A
  per-word intensity channel is an *additional* option (word emphasis), not a
  replacement.
* **register** stays model-side. It is *not* independent of the spec codes —
  harmonic spacing moves with f0, so a synthesis-time f0 shift would desync
  timbre. Keep the v0.9 relative-to-speaker parameterisation (it removes the
  ICC(1)=0.79 `sid` confound) and route it into both streams. Open question to
  measure: how much register is predictable from the *ground-truth* spec codes
  alone (Phase 1.5) — that bounds the headroom any channel has.
* **semantic** stays. It was mis-classified as an oracle: it is **derived from
  the text**, so it is available at inference with no leakage. It is *inert in
  this wiring*, not redundant with the phonemes (a phoneme sequence carries
  segmental content, not syntax/sentence type/emphasis). Its real risk is a
  corpus style shortcut (HINTS_PLAN §10.2), not leakage.

### 4.3 Planned injection (decided by the Phase 2 wiring test)

| mechanism | where | why it survives |
|---|---|---|
| FiLM `mem·(1+γ(h)) + β(h)` | encoder, **after** `_temporal_norm` | the norm is what kills the current channels |
| `out + W·h` | decoder, **after** the final decoder norm, before the heads | non-cancellable, rank-H, shared across books |
| per-book level bias | heads | **ablation arm only** — a static distribution tilt; it would inflate `val_hint_gap` while doing nothing prosodic |
| per-phone pitch/timing (additive input) | encoder input | already the only live channels; they vary over time |

### 4.4 Also fixed / also on the record

* `hints/levels.py:word_pitch_levels` assigned the **unvoiced** class id
  `len(edges)+1 = 8`, which collides with the top **voiced** bin (ids are
  1..8). The real unvoiced row (id 9) was therefore dead (`pitch_emb` row 9
  measured exactly 0.0) and unvoiced words looked like "f0 ≥ +6 st". Fixed in
  Phase 0; no re-preprocessing (`.cond` stores continuous values).
* `train.lr` is 5e-5 while AGENTS.md documents 1e-4; docs to be reconciled.
* `--transfer_from` restarts the cosine at step 0 with fresh optimiser state, so
  a warm start is an LR re-warm; v0.9's judged checkpoint sits at 42 % of its
  peak LR (274 k of a 500 k anneal) and has seen ~40 % fewer samples than the
  v0.7 best it is compared to (batch 24 vs 48). Any comparison must state both.

## 5. Phases

### Phase 0 — plumbing + docs (no shape change, no GPU) — **done (2026-11-14)**
1. `CodeTransformer.null_hints()` / `resolve_hints()`; `None` → trained null
   state, `"omit"` → legacy. Applied in `encode`/`generate`/`infill`.
2. `train.sample_hints: cond | null | omit`; hint mode in every sample log tag
   (`codes_hint/…`, `codes_nohint/…`) + scalar `sample/hints_mode`; printed line.
3. `_fim_probe` passes the batch's hints to `encode` (it was using none).
4. REPL: generate/infill in the configured state + a banner line saying which.
5. `test_hints.py`: `generate(hints=None) == generate(hints="null")`, null-hint
   shapes, and the unvoiced-pitch-id fix.
6. Docs: intensity = perceptual (not loudness) in `configs/transformer.yaml`,
   `AGENTS.md`, `hints/README.md`, `dataset_codes.py`, `HINTS_PLAN.md` §1;
   replace the §8.4 loudness KPI with the counterfactual one.
7. `hints/levels.py` unvoiced id + docstrings; `pitch_emb` dead row explained.

**Verified:** `test_hints.py` (all checks, including the new null-state ones:
`generate(hints=None) == generate("null")`, `"omit"` distinct, `encode(None)`
applies level 0, unknown mode raises; unvoiced pitch id now 9 and the top voiced
bin 8), `test_fim.py` (all checks), and a one-shot
`scratch/stage2_repl.py --text 'Fine. Thanks.' --sid 0 --frames 40 --no_audio`
on `transformer_v0.9/last.ckpt`, which loads 335/335 tensors and prints
`Hints: trained null state (level 0 / sem_null)` before sampling.

Also carried over: `scratch/eval_stage2_checkpoints.py` now builds its val
dataset with the config's hint kwargs and passes them to `encode`, so a hints
checkpoint's teacher-forced loss is comparable to TensorBoard's `val_loss`
(before, it silently scored the null state); `scratch/eval_stage2_wer.py`'s
no-hint arm documents that it now runs the trained null state.



### Phase 1 — measurement (read-only, GPU free) — **scripts done, numbers in §2**
1. `eval_stage2_wer.py --num_items 50`, both hint arms, `--ignore_eos` and
   default, for v0.7 best/last, v0.8 last, v0.9 best/last.
2. `eval_stage2_checkpoints.py` on identical batches (teacher-forced, FIM, free-running).
3. Repetition probe (done: v0.9 best 0.184 vs v0.7 best 0.235, v0.8 0.290, gt 0.220).
4. `scratch/hint_channel_probe.py` — done (§2); also wired into training as
   `train.hint_channel_probe` → `val_hint_channel_delta/*` per epoch.
5. `scratch/hint_codespace_probe.py` — done (§2): intensity R² 0.54, register
   0.17, speaker control ≈ 0.

### Phase 2 — wiring test (~40 min GPU) — **switches implemented, runs pending**
Four 2 k-step runs from `5090_v0.7_fim2/best-checkpoint.ckpt` (or from
`transformer_v0.9/last.ckpt`), identical except `transformer.hint_film_memory` /
`hint_decoder_proj`: (F,F) baseline / (T,F) / (F,T) / (T,T). Decide by
`val_hint_channel_delta/*` + the level sweeps, then by the audio sweep.


### Phase 3 — guidance (no retraining) — **implemented, sweep pending**
`generate`/`infill`: `hint_guidance` / `hint_guidance_books` (CFG against the
null state), plus `--hint_guidance[_books]` on `scratch/eval_stage2_wer.py`.
`w=1` is the exact identity, `w=0` reproduces the unconditional arm, and the
second pass is skipped when the hints already *are* the null state (all three
asserted in `test_hints.py`).  Next: sweep `w` per channel against WER +
repetition and produce the knob report (level → decoded intensity / f0 /
durations / WER).


### Phase 4 — one short-anneal run from v0.7 best
`lr 5e-5`, `cosine_anneal_end = max_steps = 80k`, winning injection, all five
channels; plus a `hints: false` control of the same recipe to separate "the
recipe" from "the hints".

Acceptance: hint-free WER ≥ v0.7's; the three live-channel probes non-zero;
register knob moves decoded f0 (RMVPE) **and** keeps timbre consistent;
intensity levels move the DSP-intensity readout of the decoded audio; the
control run reproduces v0.7's coherency.

## 6. Commands

```bash
# Phase 0/1 smoke tests (CPU, safe next to a GPU job)
python test_hints.py
python test_fim.py

# Phase 1: per-channel liveness + level sweeps on a checkpoint (read-only)
python scratch/hint_channel_probe.py \
    --checkpoint checkpoints_stage2/transformer_v0.9/last.ckpt
# Phase 1.5: is the covariate even in the code space? (speaker-grouped CV, CPU)
python scratch/hint_codespace_probe.py --n 4000

# Phase 1: same val batches / same utterances across checkpoints
python scratch/eval_stage2_checkpoints.py \
    --checkpoints checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/last.ckpt
python scratch/eval_stage2_wer.py --num_items 50 \
    --checkpoints checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/best-checkpoint.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt

# Phase 2: one arm per injection-site combination (fresh exp_name each,
# --transfer_from the hints checkpoint).  Watch val_hint_channel_delta/* and
# the level sweeps before spending a full run.
python 2_training.py --config configs/transformer.yaml --exp_name h_film \
    --transfer_from checkpoints_stage2/transformer_v0.9/last.ckpt   # + edit hint_film_memory: true

# Phase 3: guidance sweep (needs --hints so cond != uncond)
python scratch/eval_stage2_wer.py --num_items 24 --hints --hint_guidance 2.0 \
    --checkpoints checkpoints_stage2/<exp>/last.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt --ignore_eos
python scratch/eval_stage2_wer.py --num_items 24 --hints \
    --hint_guidance_books 1,1,1,1,1,1,1,1,2,2 \
    --checkpoints checkpoints_stage2/<exp>/last.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt --ignore_eos
```



