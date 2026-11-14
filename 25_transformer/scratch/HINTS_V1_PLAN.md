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
Measured on `transformer_v0.9/best-checkpoint.ckpt`, 8 val items:

| state passed to `generate` | CE (teacher-forced) | ‖memory‖ rel. diff vs real hints |
|---|---|---|
| real `.cond` hints | 2.3343 | — |
| level-0 rows (`null`) | 2.3480 | 0.291 |
| `None` (what every path did) | 2.3850 | **0.518** |

and greedy generation `None` vs `null` agrees on only **32 %** of frames
(5 items, range 1.8–68 %). So the "hint-free inference" that was listened to is
+0.051 nats off the trained null state, and the encoder memory is 52 % different.

Paths that passed nothing: `2_training.py:_sample_and_log` (codes/audio/mel/pitch),
`_log_infill`, `_log_speaker_probe`, `_log_ood_generation`,
`scratch/stage2_repl.py`, `scratch/eval_stage2_checkpoints.py` (free-running),
`scratch/eval_stage2_wer.py` (no-hint arm).

**Phase 0 fix**: `CodeTransformer.null_hints()` + `resolve_hints()`; `None` now
means the trained null state for a hint-enabled model, `"omit"` keeps the old
behaviour for ablation. Every sample log carries the state in its tag and as
`sample/hints_mode` (0 = omit, 1 = null, 2 = real `.cond`).

## 2. Measured channel liveness (v0.9 best, real val batches)

Per-channel `CE(ablated) − CE(base)`, i.e. how much the model actually uses each
channel. Base CE 2.3343.

| channel | effect (nats) |
|---|---|
| intensity off | **-0.0000** |
| register off | +0.0010 |
| semantic off | -0.0024 |
| word pitch off | +0.0113 |
| word timing off | +0.0055 |
| all off (level-0 rows) | +0.0137 |
| all off (omitted) | +0.0510 |
| intensity maxed | +0.0073 |
| timing maxed | +0.0916 |

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



### Phase 1 — measurement (read-only, GPU free)
1. `eval_stage2_wer.py --num_items 50`, both hint arms, `--ignore_eos` and
   default, for v0.7 best/last, v0.8 last, v0.9 best/last.
2. `eval_stage2_checkpoints.py` on identical batches (teacher-forced, FIM, free-running).
3. Repetition probe (done: v0.9 best 0.184 vs v0.7 best 0.235, v0.8 0.290, gt 0.220).
4. `scratch/hint_channel_probe.py` (promoted from the throwaway), also wired as
   `train.hint_channel_probe` so M3b is logged per epoch.
5. `scratch/hint_codespace_probe.py`: how well each hint is predictable from the
   **ground-truth codes** (speaker-grouped CV, `sid`-only control). This decides
   per channel whether the code space can express the knob at all.

### Phase 2 — wiring test (~40 min GPU)
Four 2 k-step runs from `5090_v0.7_fim2/best-checkpoint.ckpt`, identical except
the injection site: `additive` (current) / `film_memory` / `postnorm_proj` /
`head_bias`. Decide by per-channel CE + grad norm on the new tables.

### Phase 3 — guidance (no retraining)
`generate`/`infill`: `hint_guidance` (scalar or per-book), `uncond` = null state;
sweep `w` per channel against WER + repetition; produce the knob sweep report
(level → decoded intensity / f0 / durations / WER).

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
# Phase 0 smoke tests (CPU, safe next to a GPU job)
python test_hints.py
python test_fim.py

# Phase 1: same val batches / same utterances across checkpoints
python scratch/eval_stage2_checkpoints.py \
    --checkpoints checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/last.ckpt
python scratch/eval_stage2_wer.py --num_items 50 \
    --checkpoints checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/best-checkpoint.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt

# Phase 3: knob sweep (level -> decoded intensity / f0 / WER)
python scratch/eval_stage2_wer.py --num_items 24 --hints \
    --checkpoints checkpoints_stage2/<exp>/last.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt
```


