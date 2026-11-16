# Conditioning hints v1.0 — plan, evidence and phases

Status: **closed for v1.x — the clip-level knob works** (2026-11-15, §5.9).
Phase 0 fixed two
generation-path bugs (§1, §1.2); Phase 1 has 200-utterance WER for the null and
hint arms plus the channel/code-space probes; Phase 2's four wiring arms are run
but not decisive (both new sites are neutral-to-harmful); Phase 3's guidance
sweep is complete and **negative** (w=1 is best).  The channel ablations locate
the whole WER effect in one channel: **word timing (`slot`)** — not the
clip-level ones the plan was built around.  Judged as *controls* instead, the
register knob already worked, word timing is a pace control (0.8×-1.9×
duration), and **intensity is now a monotone dial too** after the v1.1
representation change (§5.8) — the level signal was the bottleneck, not the
injection site or the step count.  §5.5 is the decision log
and supersedes the phase text where they disagree.  `HINTS_PLAN.md` remains the spec
and status file for the *existing* (v0.8/v0.9) design; this file is the plan for
the revision, with the measurements that motivate it.

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

### 1.2 The second generation-path bug: `--hints` supplied nothing (found 2026-11-15)

`scratch/eval_stage2_wer.py --hints` built its `CodeDataset` **without** the
hint kwargs (`hints=`, `hints_json=`, ...), so the dataset emitted no `hint_*`
keys; the `CodeCollator` then had nothing to pad and `generate` resolved the
empty dict to the trained **null** state.  `--hints` was therefore a silent
no-op, and so was `hint_guidance` on top of it (`_guidance_setup` returns
"off" when `cond == uncond`).

Consequences for the record, all of which have to be re-measured:

* Every "hints arm" WER produced by that script — including the
  HINTS_PLAN.md §9 claim *"regenerating with each item's real `.cond` hints
  gives identical WER, so the channels are inert"* — compared the **null state
  with itself**.  "Hints do not change WER" was never measured.  On 6 val
  utterances the corrected path reads 14.52 % (hints) vs 17.74 % (null), with
  visibly different transcripts, so this is not a formality.
* Phase 3's whole guidance sweep ran with `cond == uncond`, i.e. weights
  `w = 1.5/2/3` were never applied.  (The repetition probe *did* see guidance
  change the samples, because it builds its dataset with the hint kwargs — that
  is what exposed the inconsistency.)
* The measurement is gated now: the dataset is built with hints when `--hints`
  is passed and the script raises if the result did not actually carry them.

The null-state arm, the teacher-forced probes (`hint_channel_probe`,
`eval_stage2_checkpoints`) and `hint_knob_report` were **not** affected — they
all built the dataset with hints already.

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
1. `eval_stage2_wer.py`, both hint arms, `--ignore_eos` and default, for v0.7
   best/last, v0.8 last, v0.9 best/last.  **Done at 200 utterances** for three
   of the five checkpoints (null state, model EOS), 2026-11-15:

   | checkpoint | epoch / step | WER | S | D | I |
   |---|---|---|---|---|---|
   | `5090_v0.7_fim2/best` | 69 / 213 710 | 21.45 % | 242 | 313 | 72 |
   | `transformer_v0.9/best` | 58 / 360 195 | 21.59 % | 226 | 351 | 54 |
   | `transformer_v0.9/last` | 70 / 433 455 | **20.22 %** | 192 | 335 | 64 |

   Floor (ground-truth codes through the same vocoder + ASR): **9.89 %**
   (289/2923 words), so roughly half of each WER is the transformer and half the
   vocoder/ASR.  At this sample size the binomial 95 % interval on one WER is
   about ±1.5 pp, so the 1.2 pp gap between v0.9-last and v0.7-best is a trend,
   not a ranking; the paired teacher-forced numbers (item 2) and repetition
   (item 3) are the tie-breakers.  These are **not** comparable to the earlier
   24-utterance ~14-15 % figures (different sample size and val items) — only
   the three rows above, measured on the same 200 items, are a comparison.
   The remaining arms/checkpoints are queued in the meta script (§6).

   **Hints arm, same 200 utterances, same decoding** (§1.2 fix), floor 9.96 %:

   | checkpoint | null state | with `.cond` hints | Δ |
   |---|---|---|---|
   | `5090_v0.7_fim2/best` | 21.45 % | 21.45 % | 0.00 |
   | `5090_v0.7_fim2/last` | — | 19.81 % | — |
   | `transformer_v0.8/last` | — | 27.10 % | — |
   | `transformer_v0.9/best` | 21.59 % | 21.11 % | −0.48 |
   | `transformer_v0.9/last` | 20.22 % | **19.02 %** | **−1.20** |

   v0.7-best is the internal negative control: it has no hint tables, so
   supplying hints is a no-op and the two rows are identical to the word
   (21.45 %, and S242/D313/I72 vs S243/D312/I72 — one substitution traded for
   one deletion).  For the hint models the gain is small at 200 utterances but
   consistent in direction, and it is **deletion-driven** (v0.9-last: D 335 →
   281, S 192 → 227, I 64 → 48), i.e. hints make the model stop early less
   often.

   The paired, `--ignore_eos` comparison on 100 utterances (same items, v0.9
   last) is larger because it removes the duration confound and pairs the draws:

   | arm | WER | S | D | I | floor |
   |---|---|---|---|---|---|
   | null state | 14.36 % | 105 | 100 | 23 | 7.87 % |
   | `.cond` hints | **11.40 %** | 95 | 56 | 30 | 7.68 % |

   −2.96 pp, again almost entirely deletions (−44) with +7 insertions.  Which
   channel carries it is measured by `--drop_hints` (100 utterances, paired,
   floor 7.6-7.9 %):

   | arm | WER | Δ vs all hints | S | D | I |
   |---|---|---|---|---|---|
   | all hints | **11.40 %** | — | 95 | 56 | 30 |
   | drop `register` | 11.08 % | −0.32 | 94 | 59 | 23 |
   | drop `pitch` | 12.97 % | +1.57 | 104 | 75 | 27 |
   | drop `semantic` | 13.35 % | +1.95 | 101 | 81 | 30 |
   | drop `intensity` | 13.73 % | +2.33 | 99 | 84 | 35 |
   | drop `slot` | 14.11 % | +2.71 | 93 | 103 | 28 |
   | null state (all five) | 14.36 % | +2.96 | 105 | 100 | 23 |

   Every channel except `register` carries part of the gain, and they are
   **redundant**: the single-drop deltas sum to +8.2 pp against +2.96 for all
   five at once, so no one channel is load-bearing (see the "only one channel"
   arm below).  `register` is neutral-to-slightly-negative on intelligibility,
   which is consistent with it being a *knob* rather than a cue.  At 100
   utterances each arm's standard error is ~1 pp, so the ordering among
   pitch/semantic/intensity/slot is not resolved; "register ≈ 0, the rest
   non-zero" is.

   This is the measurement that matters for the design: the channels reduce
   **free-running** errors (compounding drift / early stops) while being worth
   ~0 nats teacher-forced.  Both `val_hint_gap` and the per-channel CE probe are
   teacher-forced, so they are structurally blind to it — which is why the plan
   concluded "inert".

   **Keeping only one channel** (same 100 utterances, same paired draws) shows
   that the entire effect is the word-timing channel, and only that one:

   | arm | WER | S | D | I | floor |
   |---|---|---|---|---|---|
   | all hints | 11.40 % | 95 | 56 | 30 | 7.68 % |
   | **only `slot`** | **11.27 %** | 83 | 64 | 32 | 7.81 % |
   | only `pitch` (the other per-phone channel) | 14.80 % | 107 | 100 | 28 | 8.06 % |
   | only `intensity` | 15.24 % | 106 | 108 | 28 | 7.75 % |
   | only `semantic` | 15.49 % | 101 | 108 | 37 | 7.68 % |
   | null state | 14.36 % | 105 | 100 | 23 | 7.87 % |
   | only `register` | **16.25 %** | 119 | 115 | 24 | 7.75 % |

   `hint_slot` alone recovers the whole gain (the 0.13 pp difference from all
   five is noise).  **No other channel does** — not the other *per-phone* one
   (`pitch` 14.80 %, i.e. the null state), so this is specifically a **duration**
   effect, not a generic "time-varying channel" effect.  `register` alone is
   *worse* than the null state (16.25 %), consistent with its single-drop result:
   it is a prosodic control that misleads when supplied without an alignment
   cue.  Two details make this a duration effect rather than a length effect: the
   slots are start-to-start word lengths that partition the clip, and every
   measurement here uses `--ignore_eos`, so the utterance length is forced and
   only the per-word *placement* differs.  So v0.9's hint stack is, in practice,
   a coarse **duration plan**.

   Consequence for the design: this is a **pacing** signal, and pacing is
   something a user can legitimately ask for ("slower", "stretch this word",
   "pause here"), so it is a plausible control knob rather than only an oracle.
   What it is *not* is evidence for the plan's clip-level knob story: the
   intensity/register injection sites and the CFG mechanism were built for
   channels that turn out to be worth ~0 WER, and the one channel with a
   measured effect is per-phone and was already wired in v0.8.  (Predicting the
   slot from text would make the model better, but it is a model-quality track,
   not a knob — see §5.5.)
2. `eval_stage2_checkpoints.py` on identical batches — **done** (8 val batches,
   6 sampled items, teacher-forced):

   | checkpoint | val_loss | acc | FIM middle CE | free-run spec match |
   |---|---|---|---|---|
   | `v0.7/best` | 2.3061 | 0.3278 | 2.2555 | 0.0269 |
   | `v0.7/last` | 2.3786 | 0.3242 | 2.2845 | 0.0571 |
   | `v0.8/last` | 2.3090 | 0.3242 | 2.2685 | 0.0377 |
   | `v0.9/best` | **2.2878** | 0.3293 | 2.2678 | 0.0392 |
   | `v0.9/last` | 2.3138 | **0.3308** | 2.2710 | 0.0395 |

   The teacher-forced and end-to-end rankings **disagree**: val_loss puts
   v0.9-best first and v0.9-last third, while WER puts v0.9-last first (19.02 %)
   and v0.9-best third (21.11 %).  Same inversion AGENTS.md documents for the
   300 k anneal, so "best-checkpoint" is not the model to ship for intelligibility
   — read the two metrics as answering different questions, as this file has to.
3. Repetition probe — `scratch/eval_stage2_repetition.py` (adjacent-frame
   `repeat` on spec codebook 0, with a fixed ground-truth control), 6 items,
   null state:

   | checkpoint | repeat | uniq | maxrun | lag2 | mean length |
   |---|---|---|---|---|---|
   | `v0.9/last` | **0.170** | 0.432 | 0.054 | 0.097 | 0.914 |
   | `v0.8/last` | 0.185 | 0.423 | 0.062 | 0.106 | 0.929 |
   | `v0.7/last` | 0.221 | 0.414 | 0.103 | 0.151 | 0.921 |
   | `v0.7/best` | 0.234 | 0.391 | 0.105 | 0.161 | 0.936 |
   | `v0.9/best` | 0.254 | 0.404 | 0.112 | 0.147 | 0.942 |
   | ground truth | 0.266 | 0.392 | 0.116 | 0.171 | 1.000 |

   Every checkpoint repeats *less* than the ground truth, so none of them is
   degenerate in this sense.  In particular **v0.8 is not the repetition
   attractor here** (0.185, second lowest) despite its 27.10 % WER — its errors
   are deletions (D 375 vs 2923 words), i.e. it stops/skips, not loops.  The
   earlier ad-hoc numbers (v0.8 0.290, v0.7 0.235) came from an undocumented
   definition; use this table.
4. `scratch/hint_channel_probe.py` — done (§2); also wired into training as
   `train.hint_channel_probe` → `val_hint_channel_delta/*` per epoch.
5. `scratch/hint_codespace_probe.py` — done, and **reproduced at
   `--n 4000`** (3997 clips / 719 speakers): intensity R² 0.538, register
   0.170, speaker control −0.100 / −0.000.  So the codes *do* carry intensity,
   which is what makes its dead channel a wiring problem rather than missing
   information — the premise §4.1 is built on is confirmed.

### Phase 2 — wiring test (~40 min GPU) — **first run done 2026-11-15: not decisive**
Four 2 k-step runs from `5090_v0.7_fim2/best-checkpoint.ckpt` (or from
`transformer_v0.9/last.ckpt`), identical except `transformer.hint_film_memory` /
`hint_decoder_proj`: (F,F) baseline / (T,F) / (F,T) / (T,T). Decide by
`val_hint_channel_delta/*` + the level sweeps, then by the audio sweep.

For the in-training probe (and `last.ckpt`, which is written on epoch end) to
exist at all, a 2 k-step run needs `train.limit_train_batches = 2000`: a run
that stops at `max_steps` inside its first epoch never validates and writes no
checkpoint.  `2_training.py` now reads that key (see the comment next to
`train.val_interval` in `configs/transformer.yaml`).

> **The gates below are invalid for the three `film`/`proj` arms** — they were
> run with the base config, which dropped the new modules (see **§5.6**).  The
> *training* is valid; the offline numbers are "baseline architecture, arm's
> weights".  Re-run with `run_hints_v1.py phase2 --tag w1 --only gate`, which now
> passes each arm's own config.

**First run** (`--tag w1`, four arms × 2 000 steps from
`transformer_v0.9/last.ckpt`, `lr 5e-5` cosine to `1e-6`, one epoch):

| arm | val_loss | int Δ | reg Δ | sem Δ | pitch Δ | slot Δ |
|---|---|---|---|---|---|---|
| (F,F) baseline | 2.3157 | +0.0010 | +0.0016 | −0.0006 | +0.0127 | +0.0068 |
| (T,F) film | **2.3763** | +0.0000 | +0.0021 | +0.0007 | +0.0108 | +0.0054 |
| (F,T) proj | 2.3201 | +0.0006 | +0.0017 | −0.0005 | +0.0103 | +0.0068 |
| (T,T) film+proj | **2.3748** | +0.0005 | +0.0020 | +0.0028 | +0.0103 | +0.0088 |

Decoded-audio slopes (4 val items, levels 0..8/0..5): intensity +0.081 baseline
/ +0.060 film / +0.027 proj / +0.077 film+proj; register +1.09 / +0.95 / +0.76
/ +1.06 st per level.  Codebook-0 repetition stayed at or below the
ground-truth control in every arm (0.22–0.27 vs 0.266), so nothing degenerated.

What the run does and does not say (only the `baseline` row of the table above
is valid — its architecture is the base config; every other row is the arm's
weights in the baseline model):

* **The CE gate is blind here.** `int` sits at ~0.001 nats in all four arms,
  exactly as §4.1 predicts for a style channel — `val_hint_channel_delta/int`
  cannot see whether the new injection works.
* **FiLM costs ~0.06 nats of val_loss in 2 k steps**, and its weights move a
  long way at this LR (‖γ‖ 9.0, ‖β‖ 5.3 out of zero init).  On a converged
  model it is a large perturbation, not a neutral add-on.
* **No arm beats the baseline on the audio knob.** The existing (pre-norm,
  additive) wiring already moves intensity by +0.08/level and register by
  ~1 st/level; the new sites do not improve either at this scale.

So a 2 k-step screen is **not a decision procedure**: too short (the new modules
see ~1 k steps at a meaningful LR), wrong KPI (CE), and 4 items against a
readout whose per-item noise is ~0.5 (the DSP intensity model's MAE is 0.59).
Either do it properly (≥16–24 items, paired per-item deltas, report the slope
with an interval) or skip it and let Phase 4's 80 k run + knob report answer the
question at a scale where it can be answered.  What it *did* establish: both
switches add parameters, transfer, train, validate and sample without breaking
anything, and neither degenerates the model.


### Phase 3 — guidance (no retraining) — **implemented; the 2026-11-15 sweep was a no-op**
`generate`/`infill`: `hint_guidance` / `hint_guidance_books` (CFG against the
null state), plus `--hint_guidance[_books]` on `scratch/eval_stage2_wer.py`.
`w=1` is the exact identity, `w=0` reproduces the unconditional arm, and the
second pass is skipped when the hints already *are* the null state (all three
asserted in `test_hints.py`).

The first sweep (`w = 1/1.5/2/3`, 24 utterances, per-book `1,…,1,2,2`)
reported the *same* WER (17.27 %, S30 D40 I6) and identical transcripts for
every weight: `--hints` was supplying nothing, so `cond == uncond` and guidance
was skipped (§1.2).  The repetition probe did move (0.293/0.307/0.297/0.239 at
w = 1/1.5/2/3), because that script builds its dataset with the hint kwargs —
that is what exposed the inconsistency.  After the fix, 6 utterances give
17.74 % (null) / 14.52 % (hints, w=1) / 12.90 % (hints, w=3) with clearly
different transcripts, so the mechanism now reaches the audio.

Next: re-sweep `w` against WER + repetition **and** produce the knob report
(level → decoded intensity / f0 / durations / WER), at ≥100 utterances per
weight.  The knob report is `scratch/hint_knob_report.py` (generates each level,
decodes with the frozen stage-1 Generator, measures DSP intensity / RMVPE
register / duration / codebook-0 repetition / optional WER, plus a
least-squares slope per readout).

**Corrected sweep, 2026-11-15** (`transformer_v0.9/last.ckpt`, 100 utterances,
`--hints --ignore_eos`, same items and seed for every weight):

| arm | WER | S | D | I | floor |
|---|---|---|---|---|---|
| `w = 1.0` (identity) | **11.40 %** | 95 | 56 | 30 | 7.68 % |
| `w = 1.5` | 12.72 % | 107 | 63 | 32 | 7.81 % |
| `w = 2.0` | 14.04 % | 120 | 78 | 25 | 7.68 % |
| `w = 3.0` | 13.79 % | 116 | 76 | 27 | 7.75 % |
| per-book `1,…,1,2,2` | 13.22 % | 103 | 75 | 32 | 7.87 % |

**Guidance is a net negative.**  `w = 1` (the conditional model, no steering) is
the best arm; every guided weight is worse, and the errors are
substitution-dominated (S 95 → 120) rather than the v0.8 repetition failure —
repetition actually *falls* with `w` (book-0 `repeat` 0.293 / 0.307 / 0.297 /
0.239 at w = 1/1.5/2/3, ground truth 0.266, 6 items).  So `logits = uncond +
w(cond − uncond)` sharpens the model into confident *wrong* codes: the hint
direction is not separating what we want steered from what we want preserved.

Knob report on the same checkpoint (8 items, no guidance), i.e. what the hint
channels do through the *existing* wiring:

| channel | slope | shape | WER at the extremes |
|---|---|---|---|
| intensity | +0.045 levels/level | non-monotone (L0 4.16, L1 3.96, L8 4.41) | 8.70–15.65 % (no trend) |
| register | **+0.955 st/level** | monotone L1..L5: 8.5, 9.6, 10.6, 11.6, **14.8** st | L1–L4 8.70–9.57 %, **L5 20.00 %** |

So: **register is a real knob** (~6 st over its table) that the existing
pre-norm wiring already delivers.  Its top level is also the only one with a
visible WER cost (20.0 % vs 8.7-9.6 % for L1-L4) — but that is 8 utterances per
level, so treat it as a flag to re-measure, not a number.  **Intensity is not a
knob** — 0.25 total against a readout whose MAE is 0.59.

**Word timing as a pacing knob** (`--shift`, 8 items, `--gen_scale 2.0`, `--wer`;
each arm keeps the item's own rhythm and shifts every *present* word's log2
duration bucket, so +1 means "twice as long per word"; paired within the run):

| shift | decoded duration | vs real | repeat | WER |
|---|---|---|---|---|
| −2 | 3.97 s | 0.81× | 0.183 | 13.91 % |
| −1 | 3.79 s | 0.77× | 0.155 | 18.26 % |
| real | 4.90 s | 1.00× | 0.232 | **13.04 %** |
| +1 | 7.57 s | 1.54× | 0.301 | 23.48 % |
| +2 | 9.18 s | 1.87× | 0.433 | **59.13 %** |

Pacing is a **real, monotone knob in the slow direction** (4.9 → 7.6 → 9.2 s)
and a compressed one in the fast direction (floor ≈ 0.8×, and −1/−2 not ordered).
But it degrades fast: +1 costs ~10 pp WER, +2 collapses (repeat 0.433 > the 0.266
ground truth, i.e. it stretches words into loops), and the fast side is noisy at
this n.  Contrast with the *uniform* level sweep (`--channel slot --levels all`,
same checkpoint), which collapses from L5 on (57/83/86/92 % WER): that arm forces
every word to one bucket — it flattens the rhythm rather than pacing it — so it
answers a different (and useless) question.

Read together with register, the pattern is: **the knobs work mechanically but
only inside a narrow band around the corpus distribution**, because the hint
values seen in training *are* the natural distribution.  That is the real
limitation, and it is a training-data problem, not an injection-site one.

Two measurement notes: the 4-item and 8-item pacing runs pick different val
items, so only the 8-item run may be quoted (4 items gave 2.9-6.7 s for the same
range); and at 8 utterances the WER column is ±3-4 pp, so the ±1 rows are
"consistent with a real cost", not measured margins.


### Phase 4 — one short-anneal run from v0.7 best — **blocked: both mechanisms failed**
`lr 5e-5`, `cosine_anneal_end = max_steps = 80k`, winning injection, all five
channels; plus a `hints: false` control of the same recipe to separate "the
recipe" from "the hints".

**Do not run this as written.**  There is no "winning injection": FiLM costs
val_loss without moving the readout (Phase 2) and guidance costs WER (Phase 3).
Running it would anneal 80 k steps toward a mechanism that is measured not to
work.  See §5.5 for what to do instead (`hint_film_memory`/`hint_decoder_proj`
stay in the config as ablations, off by default).

Acceptance: hint-free WER ≥ v0.7's; the three live-channel probes non-zero;
register knob moves decoded f0 (RMVPE) **and** keeps timbre consistent;
intensity levels move the DSP-intensity readout of the decoded audio; the
control run reproduces v0.7's coherency.

### 5.5 Decision log — what the 2026-11-15 runs change

**Headline: the hints are not inert — they help, and the old evidence said the
opposite because it measured the null state twice.**

| comparison | null | hints | Δ |
|---|---|---|---|
| v0.9 last, 200 utts, model EOS, floor 9.96 | 20.22 % | **19.02 %** | −1.20 pp |
| v0.9 last, 100 utts, `--ignore_eos`, floor 7.87 | 14.36 % | **11.40 %** | −2.96 pp |
| v0.7 best (no hint tables), 200 utts | 21.45 % | 21.45 % | 0.00 (control) |

Identical S/D/I for the v0.7 control to within one word confirms the plumbing is
right; the gain for hint models is real and **deletion-driven**.  The two arms
mean slightly different things: with the model's own EOS, part of the deletion
drop is "stops early less often" (D 335 → 281 on v0.9 last); with
`--ignore_eos` the duration is forced, so the drop (D 100 → 56) is pure
intelligibility — words the ASR can now hear inside a fixed-length utterance.
HINTS_PLAN.md §9's "the hints are inert, identical WER in both arms" must be
struck.

**Settled / corrected**

1. **The `--hints` arm was never conditional** (§1.2).  Every hint-arm WER in
   this file and in `HINTS_PLAN.md` §9 was re-run (headline table below): the
   channels *help*.  Before the fix, "the hints are inert at inference"
   (HINTS_PLAN.md §9) rested on a comparison of the null state with itself.
2. **Guidance was never applied** in the first sweep (§1.2), because
   `cond == uncond` there.  After the fix the sweep does run — and it is
   negative (item 6), not a no-op.
3. **The CE liveness number is neither a knob metric nor a usefulness
   metric.**  After 2 k steps the intensity channel is +0.000-0.001 nats in
   *every* Phase 2 arm, yet the decoded audio already moves with the level in
   the baseline arm; the register channel is worth +0.002 nats and ~2 st of
   decoded f0 per level.  Two orders of magnitude apart, same channel.  And the
   hints cut free-running WER by ~3 pp while being worth ~0 nats teacher-forced
   — they suppress compounding drift and early stops, which a teacher-forced
   probe cannot see by construction.  `val_hint_channel_delta/*` and
   `val_hint_gap` answer "does the model need this to predict the next target
   given the true prefix", nothing else.
4. **The register knob already works** through the *existing* (pre-norm,
   additive) wiring: +0.95 st of decoded f0 per level, monotone over the table
   (~6 st from L1 to L5), measured on 8 items.  That is the strongest hint
   result in the repo and it did not need a new injection site.  Its cost looks
   like it is at the top level (register L5: 20.0 % WER vs 8.7-9.6 % for L1-L4),
   but that is 8 utterances per level and has to be re-measured properly before
   it drives a decision.

   Note the split personality: register is the one channel whose *removal* does
   not hurt intelligibility (−0.3 pp, i.e. noise), the one that *alone* is worse
   than no hints at all (16.25 % vs 14.36 %), and the one that gives a clean
   prosodic knob.  It is a **control**, not a cue.  Of the other four, only
   `slot` carries a cue on its own (11.27 %); `pitch`/`intensity`/`semantic`
   alone are the null state, so their single-drop cost is redundancy with
   `slot`, not an independent contribution.
5. ~~**FiLM is not free**: +0.06 nats of val_loss...~~ **Wrong — superseded by
   §5.6.**  That measurement dropped the module from weights trained with it; with
   the module enabled, teacher-forced CE is 2.3150 (film) vs 2.3178 (baseline),
   i.e. no cost.  The weights do move a long way (‖γ‖ 8.98), but they are not
   hurting the fit.
6. **Classifier-free guidance is rejected** (Phase 3, 100 utterances, paired):
   `w = 1` is best at 11.40 %, and every guided weight is worse (12.72 / 14.04 /
   13.79 % at w = 1.5/2/3; 13.22 % for the pitch-books-only arm).  The damage is
   substitution-dominated and repetition does not rise, so it is not v0.8's
   attractor: the `cond − uncond` direction sharpens the model into confident
   wrong codes.  §4.1's mechanism does not deliver its knob.
7. **Intensity has no knob at all** through the current site: +0.045
   levels/level, non-monotone, ~0.25 over the whole table against a readout
   whose MAE is 0.59.  With CFG rejected, intensity now has *two* failed
   mechanisms (additive input, guidance) and needs the §4.1 fallback (a
   counterfactual margin loss) or a different formulation (per-word emphasis)
   before another training run is worth 80 k steps.
8. **`v0.9/last` is not worse than `v0.7/best` on the objective measures**: 200
   utterances give 20.22 % vs 21.45 % WER (floor 9.89 %), and codebook-0
   repetition 0.25 vs 0.38.  The "less coherent than v0.7" impression (§0.1) is
   therefore not reproduced by WER or repetition; it was formed on samples drawn
   in the Phase-0 omit state.  Check it by listening to the same 200 items
   rather than by inference.

**Recommended order (revised after the channel ablations)**

The product goal is stated in §4.2 and it is **not** "lower WER": the hints exist
to be **user-facing control knobs**.  So the acceptance test for a channel is
"does moving its level move the audio the way the user asked, smoothly, at an
acceptable quality cost" — and the WER numbers above are the *cost* column, not
the objective.  Judged that way:

| channel | knob? | cue? (free-running WER) | verdict |
|---|---|---|---|
| register | **yes** (+0.95 st/level, monotone) | no (−0.3 pp when dropped) | ship as a control |
| intensity | no (+0.045 levels/level, non-monotone) | no (alone = null) | broken knob |
| semantic | n/a (text-derived, not a control) | no (alone = null) | conditioning only |
| pitch | not swept | no (alone = null) | neither |
| slot | untested as a knob | **yes** (11.27 % alone = all five) | see below |

* **Do not run Phase 4 as specified.**  There is no winning injection: FiLM
  costs CE without moving the readout and CFG costs WER, so an 80 k anneal has
  nothing to anneal toward.
* What is worth doing, in cost order — all of it knob work:
  1. **Register: productise the knob that already works.**  It needs no
     training and no new mechanism.  Measure it properly (≥24 items per level,
     report the decoded-f0 curve *and* WER per level) to fix a safe range, and
     decide whether the top level's cost is the model or the corpus's register
     tail.  This is the only channel that currently satisfies §4.2's goal.
  2. **Intensity: decide whether it is in the product at all.**  Two mechanisms
     have failed on it (additive input, guidance) and its readout is at the
     noise floor.  The counterfactual training in (4) covers it too, so there is
     one last shot; if it still does not move, cut it from the control surface
     instead of paying for it.  Either way it needs a KPI that is not the hint's
     own regressor (see the caveat below) — otherwise it scores itself.
  3. **Slot: it is a pacing knob — use it as one.**  Measured (Phase 3):
     shifting each word's own bucket gives 0.8×-1.9× decoded duration,
     monotone in the request.  So "slower / faster" is a legitimate second user
     control and the "it is an oracle" objection disappears: the *user* is the
     one supplying the pace.  Its usable band is narrow today (+1 costs ~10 pp
     WER, +2 collapses), which is the same limitation as register's top level.
  4. **The one training change that follows from all of this: counterfactual
     hint augmentation.**  Both working knobs work only near the corpus
     distribution because the training hint values *are* that distribution
     (dropout trains the null state, not the shifted ones).  Perturbing the hint
     levels during training — clip-level levels and per-word slot shifts drawn
     from a band around the true value — is a one-line dataset change aimed
     exactly at widening the usable range, and it subsumes the §4.1 margin loss
     (which is the same objective written as a loss rather than as augmentation).
     This is the run worth doing; it needs a fresh `exp_name` like any other
     hint-set change.
* **Orthogonal, and out of scope here:** a duration predictor would convert
  slot's 3 pp into a *naturalness/intelligibility* gain at inference, but that
  is a model-quality track, not a knob, and it would not make any control work
  better.  Note it and leave it — it belongs to a different plan.

**KPI caveat for the intensity channel.**  The DSP intensity model has an MAE of
~0.59 on the 1-9 scale and it is *both* the hint source and the scorer, so a
knob that moves the readout by ~0.6 over the whole level range is at the
measurement limit.  A defensible acceptance test needs paired per-item deltas
over ≥16 items plus a second, independent proxy (e.g. F0 range, energy variance,
spectral flux), or a different target (per-word emphasis rather than a
clip-level offset).

### 5.6 Diagnostic A — where the clip-level signal actually dies (2026-11-15)

**First, a measurement bug that invalidates the Phase 2 gates.**  The four
wiring arms trained correctly (each with its own generated config), but every
*offline* gate ran with `configs/transformer.yaml`, where
`hint_film_memory`/`hint_decoder_proj` are both `false`.
`load_submodule_prefix(..., quiet=True)` then silently dropped the FiLM and
decoder-projection tensors, so the gate scripts measured the **baseline
architecture on each arm's weights**.  Everything in the Phase-2 table above for
`film`/`proj`/`film+proj` is therefore "the arm's non-FiLM weights", not the arm.
In particular "no arm beats the baseline on the audio knob" was true by
construction, and the "+0.06 nats of val_loss for FiLM" never measured FiLM.

Fixed: each arm's gates now use that arm's config, `load_transformer_weights`
**raises** rather than dropping a post-norm module, and
`hint_injection_probe.py --configs` takes one config per checkpoint.

**The diagnostic itself.**  `scratch/hint_injection_probe.py`, 2 batches of 8,
intensity 0→8 with the other channels real (per-arm configs):

| arm | ‖γ‖ | ‖W_proj‖ | memory Δ | logit Δ | class-dependent | ΔCE |
|---|---|---|---|---|---|---|
| baseline | — | — | 3.8 % | 1.6 % | 95 % | +0.0024 |
| film | 8.98 | — | **5.0 %** | 1.7 % | 96 % | +0.0030 |
| proj | — | 6.80 | 4.4 % | **2.0 %** | 94 % | +0.0042 |
| film+proj | 8.81 | 6.73 | **5.2 %** | 1.9 % | 95 % | +0.0038 |

A fourth correction falls out of the same table: teacher-forced CE at level 0 is
2.3178 / 2.3150 / 2.3160 / 2.3147 for baseline / film / proj / film+proj, i.e.
**enabling the modules costs no CE at all**.  The arms' own training logs agree
(final `val_loss` 2.328 / 2.326 / 2.328 / 2.326), so this is not a probe
artifact.  The "+0.06 nats of val_loss" in the
Phase-2 table was the artifact of *dropping* them from weights that had adapted
to their presence, not a real penalty.  §5.5 item 5 ("FiLM is not free") is
wrong and is superseded by this.

Three things are now settled:

1. **The site works.**  FiLM enlarges the post-norm memory movement by ~30 %
   (3.8 → 5.0-5.2 %) and the decoder projection enlarges the logit movement by
   ~25 % (1.6 → 2.0 %).  Neither is a no-op.
2. **Nothing cancels.**  94-96 % of the logit shift is *class-dependent*, i.e.
   softmax can see it.  §3's "the heads map it to a near-uniform offset" is not
   what happens on the post-norm path.
3. **The signal is simply tiny, and it does not grow with training.**  The whole
   intensity range moves the clip-level hint vector by ‖Δh(L8−L0)‖ = **0.61** in
   a 768-d space whose post-norm memory has norm ≈ 27.7 (~2 %), and FiLM's
   contribution is ‖W·Δh‖ ≈ 0.18.  Decisively: **v0.9 after 433 k steps has the
   same 0.61** as the 2 k arms (0.56-0.62).  Compare `register`, whose
   ‖Δh(L5−L0)‖ is 1.18 and which *does* have an audio knob (+0.95 st/level), and
   the per-phone tables (`pitch_emb` row norms 0.84-1.95), which are the channels
   with a measured effect.

The **audio** end of the chain, with each arm's own config (8 items, levels
0/4/8, so only the slope is meaningful):

| arm | intensity slope | reg slope |
|---|---|---|
| baseline | +0.021 /level | +0.096 |
| film | +0.028 /level | +0.177 |
| proj | +0.035 /level | +0.135 |
| film+proj | +0.013 /level | +0.186 |

That does **not** rank the arms: the same baseline checkpoint measured +0.081 on
a 4-item 9-level sweep and +0.021 here, and `film+proj` (largest memory shift)
has the smallest slope — all four per-level curves are non-monotone.  The readout
is too noisy at 8 items to adjudicate anything, which is the point: the internal
metric is the usable gate, the audio readout is not.  What the audio numbers *do*
say is that the knob is weak (~0.15-0.3 over the whole range) in every arm.

So **the answer to "was 2 k enough?" is that the step count was never the
limiting factor.**  With CE as the objective the level is almost
loss-irrelevant, so the free 9-row lookup never has to grow: the rows end up
small, mutually arbitrary and unordered, i.e. the "level" is a tiny random
perturbation rather than a monotone dial.  More steps at the same objective
leave it exactly where it is.

What follows from that (and it is a representation change, not another site):

* encode the level as a **monotone direction** — one learned vector times the
  normalised **continuous** value that `.cond` already stores, instead of a free
  per-level row.  A dial is then monotone by construction, and the gradient is
  shared across the range instead of split 9 ways.
* and/or **counterfactual hint augmentation** (§5.5 item 4), which removes the
  "level is loss-irrelevant" property that keeps the table small.

Neither needs a new injection site; both are compatible with keeping
intensity/register.

### 5.7 v1.1: monotone encoding + counterfactual margin — implemented, run in flight

Two changes, both aimed at the *representation* bottleneck §5.6 identified (no
new injection site, no re-preprocessing, no dataset change):

1. **`transformer.hint_monotone`** — the intensity/register level is
   `direction * z(level)` with `z` linear on [-1, 1], plus a trained "absent"
   vector, replacing the free `(n_levels+1, H)` lookup.  Monotone by
   construction, one gradient for the whole range.  `pitch_emb`/`slot_emb` keep
   their tables (categorical, and already live).
2. **`train.counterfactual_margin`** (`_weight`, `_rate`, `_channels`) — on 25 %
   of steps, score the same batch with every sample's level replaced by a
   *different* one and add `relu(margin - (CE_shift - CE_true))`.

**Correction to §5.5 item 4 / the earlier "hint augmentation" idea.**  Plain
augmentation is *wrong* here and I am not implementing it: with the target codes
fixed, a perturbed hint is best *ignored*, so augmenting the hint alone trains
**invariance** — the exact opposite of a knob.  The margin is what turns the same
perturbation into a dependence objective (it is §4.1's fallback, not a new
mechanism).

Implementation notes that cost real time to find:

* The counterfactual arm runs as a **second backward after the main one**.  With
  both graphs live it OOMs immediately (fp32, batch 24 x 256 frames, ~13 GB of
  15.5 GB — measured at the first counterfactual step).  Two backwards accumulate
  into the same `.grad` buffers, so the objective is unchanged.
* `counterfactual_weight` must be well below 1: the penalty's gradient is a
  full-strength *ascent* on the shifted arm regardless of the margin (it is
  `-dCE_shift/dθ`, not scaled by how far the gap is from the margin).  0.2 at
  rate 0.25 ≈ 5 % average.
* `check_special_layout` and `eval_stage2_wer.load_transformer_weights` now refuse
  **both** monotone directions (a trained direction dropped into a table config
  leaves the clip-level channels *inert*, which would read as "intensity does
  nothing").

**Hyperparameter check** (400 steps, monotone with and without the margin,
identical otherwise, transfer from v0.9 last):

| arm | val@199 | val@399 |
|---|---|---|
| monotone only | 2.4423 | 2.3866 |
| monotone + margin | 2.4457 | 2.3885 |

So the margin costs **0.002 nats** and does not cause the early transient — that
is the ordinary `--transfer_from` LR re-warm (fresh Adam state on a model that
had annealed to 1e-6; the no-margin arm shows the same +0.13 rise and the same
decay, ending at 2.39 and still falling towards the 2.3284 the 2 k arms reach).

**Run in flight** (`scratch/hints_v1_runs/20261115-v11/`, detached):

```
exp_name hints_v1_p4_final_mono1   80 000 steps, cosine 80 k, lr 5e-5
hints=true, hint_monotone=true, film/dec_proj=false
cf margin 0.05, weight 0.2, rate 0.25, channels intensity,register
limit_train_batches 8000 (10 epochs), transfer_from transformer_v0.9/last.ckpt
~12 h; last.ckpt every epoch, best-checkpoint by val_loss
```

Read, in order, when it lands:

1. `train_cf_gap_step` / `train_cf_gap_epoch` — it must **climb from ~0 towards
   0.05**.  A flat 0 means the margin gradient is losing and nothing will have
   changed.
2. `val_loss` — the yardstick is the source's 2.3138 (and 2.3284 for a 2 k
   fine-tune).  A permanent +0.05 means the counterfactual arm is costing fit.
3. `scratch/hint_injection_probe.py --config <run config> --checkpoint ...` —
   `||Δh(L8-L0)||` is the acceptance metric: **0.61 today, want ≳2** (≥8 % of the
   memory norm), with the memory/logit shifts up in proportion.
4. `scratch/hint_knob_report.py` (intensity + register, ≥16 items) — the knob
   must be monotone and clearly larger than +0.045/level.  This is the *costly*
   number; do not use it to adjudicate until Δh has moved.
5. WER (`--ignore_eos`, 100-200 utts) — must not regress from the 11.40 % / 14.36 %
   null reference; the hint-arm gain should survive.

If `Δh` grows but the audio does not, the remaining gap is the decoder ignoring
the channel, and the next lever is the (measured CE-neutral) `hint_decoder_proj`
on top of the monotone encoding — not another representation change.

### 5.8 v1.1 result — the representation was the bottleneck (2026-11-15)

`hints_v1_p4_final_mono1`, 80 k steps from `transformer_v0.9/last.ckpt`, monotone
encoding + counterfactual margin (margin 0.05, weight 0.2, rate 0.25), no
post-norm injection sites.  Gates: `scratch/hints_v1_runs/20261115-v11/`.

**The level signal grew, as designed.**  `||Delta h||` between the two ends of the
dial (L1 -> L8) is **2.41** (one direction, `||int_dir||` = 1.21), against **0.61**
for the free table on v0.9 — and that 0.61 was *unchanged* after 433 k steps.
`||Delta h||(L8 - L0)` is 0.84 because level 0 is its own trained "absent" vector,
so the number to quote for a user moving the dial is the 2.41.

**The intensity knob works now, and it is monotone:**

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| DSP intensity | 4.374 | **3.708** | 3.846 | 4.122 | 4.252 | 4.329 | 4.456 | 4.588 | **4.645** |
| WER (8 utts/level) | 7.8 | 14.8 | 12.2 | 8.7 | 9.6 | 12.2 | 7.8 | 12.2 | 6.1 |

Monotone over the whole present range (3.71 -> 4.65, **+0.94** on the 1-9 scale)
versus v0.9's non-monotone 0.25 total; slope +0.086/level (was +0.045).  The
per-level WER has no trend (8 utterances per level = ±several pp), so the dial
does not buy its span with intelligibility at these extremes.

Two calibration notes, not defects: the dial is **asymmetric about "absent"**
(L0 = 4.374 sits between L7 and L8), so most of the travel is at the low end; and
the teacher-forced CE profile is strongly one-sided (L1 +1.09 nats, L2 +0.65, L3
+0.25, L4-L8 <= 0.005) because the val items' true levels cluster around L5 —
it is distance from the truth, not an anomaly at any level.  (My first read of
this as an "L2 spike" was an artefact of a probe that skipped L1.)

**Everything else held or improved:**

| metric | v0.9 last | v1.1 mono1 |
|---|---|---|
| teacher-forced val_loss | 2.3138 | **2.2631** |
| WER, null state (200 utts) | 20.22 % | **18.82 %** |
| WER, `.cond` hints (200 utts) | 19.02 % | **18.58 %** |
| register knob | +0.955 st/level | +0.897 st/level |
| slot / pacing knob | +0.064 s/level | +0.088 s/level |
| per-channel CE delta: int / reg / pitch / slot | +0.0004 / +0.0020 / +0.0125 / +0.0062 | +0.0002 / +0.0015 / +0.0138 / +0.0074 |

So the counterfactual margin cost nothing (val_loss *fell* 0.05, WER fell ~1.4 pp
in both arms) and the CE liveness numbers are unchanged at ~0 for the clip-level
channels — which is the point: **CE was never the metric; the dial is.**  The hint
*arm gap* shrank (1.20 -> 0.24 pp), consistent with the model no longer needing
the oracle channels as much; at 200 utts that is ~1 pp of noise, so read it as
"the hint-free model improved", not as a loss of hint value.

`train_cf_gap` overshot the margin by ~8x (0.05 -> 0.43 nats by step 80 k, with
the penalty itself -> 0.0007, i.e. the hinge is satisfied).  The extra sensitivity
is not the hinge doing it — it is the model using the level on its own now that
the encoding makes it usable.  Worth knowing: the hinge bounds nothing above.

**Open, in priority order**

1. The intensity *readout* is still coarse: a 0.94-level span against the DSP
   model's 0.59 MAE.  The dial is now monotone, which is the property a user
   perceives, but a paired A/B discrimination at L2 vs L7 (or an independent
   proxy: f0 range, energy variance, spectral flux) is the right KPI before
   quoting a magnitude.
2. `best-checkpoint.ckpt` is step ~73 k (val 2.2707 vs last 2.2743) — re-run the
   knob on it if the 0.004 matters.
3. The asymmetry about "absent" is a calibration question: if a UI should offer
   "lower / same / higher" around the clip's own level, the level-0 vector and
   the corpus's level distribution need to agree.
4. Still untested: `semantic`'s value on OOD text (unchanged from §5.5).

### 5.9 Cap — scope closed for v1.x (2026-11-15)

Deliberate stop.  The enable/capability question this plan was written to answer
is settled, the run that settles it is in the books, and everything left is
either a *measurement* refinement or a different project.  No v1.2 unless one of
the triggers below fires.

**What the plan set out to do, and where it landed**

| question (§0) | answer |
|---|---|
| Is v0.9 worse than the v0.7 base? | No.  Both Phase-0 generation bugs fixed (§1, §1.2); WER 20.22 % -> 18.82 % null / 19.02 % -> 18.58 % with hints (v0.7 base is 21.45 %). |
| Are the channels inert? | Four of five are live as controls: `slot` (pace, +0.088 s/level), `register` (+0.897 st/level), `intensity` (monotone, +0.086/level) and `pitch` (word-level f0).  Only the *clip-level CE deltas* read ~0, which is expected — CE was never the metric. |
| Can the clip-level channels work at their old site? | Not there (§3, §5.6): the temporal norm cancels a per-clip constant.  The two post-norm sites fix the algebra and still do not improve the knob, so they ship **off**. |
| What actually blocked intensity? | The *representation*, not the site or the step count: a free per-level row trained for 433 k steps stayed 0.61 in a 768-d memory of norm ~27.7 (§5.6).  Monotone encoding took it to 2.41 (§5.8). |

**Shipped state**

* Code: `transformer.hint_monotone` and `train.counterfactual_*` in
  `modeling/transformer.py` / `2_training.py` / `configs/transformer.yaml`, with
  both-direction guards in `check_special_layout` and
  `eval_stage2_wer.load_transformer_weights` (a dropped direction reads as
  "intensity does nothing").  `test_hints.py` covers the encoding;
  `test_fim.py` is green.
* Model: `checkpoints_stage2/hints_v1_p4_final_mono1/{last,best-checkpoint}.ckpt`
  — this is the stage-2 model to use for hint-capable inference.
* Recipe: `scratch/hints_v1_runs/20261115-v11/configs/hints_v1_p4_final_mono1.yaml`
  (hints + monotone, film/proj off, margin 0.05 @ weight 0.2 @ rate 0.25).  The
  shipped `configs/transformer.yaml` keeps `hint_monotone: false` on purpose: the
  flag changes the parameter set, so leaving it false keeps every older
  checkpoint resumable.  Copy the run config for a new hints run.
* Diagnostics that answered the question: `scratch/hint_injection_probe.py`
  (where the signal dies), `scratch/hint_knob_report.py` (does the dial move),
  `scratch/hint_channel_probe.py` (per-channel CE liveness).  All read-only.

**Explicitly not doing (and why), so they are not re-litigated**

* **Hint augmentation** — wrong objective.  Targets fixed, so a perturbed hint is
  best ignored; it trains invariance.  The margin (§5.7) is the correct form and
  is already in.
* **Classifier-free guidance over hints** — measured negative (§5.5 item 3):
  `w=1` best at 11.40 %, every guided weight worse (12.7-14.0 %), and it shifts
  errors into substitutions and kills repetition.
* **FiLM / decoder-projection as the fix** — they carry the signal (+30 % memory,
  +25 % logit motion, 95 % class-dependent) but the level was 2 % of the memory
  norm.  Neutral on val_loss once each arm was evaluated with its own config
  (§5.5's gate correction).  Requires a fresh run each time; not worth it.
* **A duration predictor** — out of scope: hints are user controls, not a
  reconstruction aid.  `slot` already gives the pacing knob a user would want.
* **Re-preprocessing** — nothing here touches the codes; the codes stay valid.

**If it is ever reopened, these are the two triggers and the first step**

1. *"Is the 0.94-level span actually perceptible?"* — the open question in §5.8
   is measurement, not model: paired A/B at L2 vs L7, or an independent proxy
   (f0 range, energy variance, spectral flux), scored against the DSP model's
   0.59 MAE.  Only this turns the knob into a quotable magnitude.
2. *"Should the UI offer lower/same/higher around the clip's own level?"* — a
   calibration change: the level-0 ("absent") vector currently sits at 4.374,
   between L7 and L8, so the dial is asymmetric about absent.  Fixing it needs
   the level-0 vector to agree with the corpus level distribution, i.e. a
   monotone run with an explicit "current level" target — not a config edit.

Also unmeasured, and cheap whenever a reason appears: `semantic` on OOD text
(§5.5) — **measured 2026-11-15, see `HINTS_PLAN.md` §7**: the channel is live
(presence moves the codes and raises decoded f0 spread) but not content-
discriminative, since another sentence's vector is as far from the true one as
the null state is. And the knob re-run on `best-checkpoint.ckpt` (val 2.2707 vs
2.2743, i.e. irrelevant unless something else is being decided at the same
time).

## 6. Commands

```bash
# Phase 0/1 smoke tests (CPU, safe next to a GPU job)
python test_hints.py
python test_fim.py

# ---- the whole plan, one command ------------------------------------------ #
# Every Phase 1-4 test/ablation with the right checkpoints, hint arms and
# per-arm config: preview first, then run a phase at a time.  Read-only tasks
# run; training arms need --run-training.  Logs + a WER table land in
# scratch/hints_v1_runs/<UTC>/ (results.md, manifest.json, logs/, configs/).
python scratch/run_hints_v1.py all --dry-run
python scratch/run_hints_v1.py phase1            # 200-utt WER matrix + probes
python scratch/run_hints_v1.py phase1 --quick --only wer_nohint_eos
python scratch/run_hints_v1.py phase2 --tag w1   # writes the 4 arm configs
python scratch/run_hints_v1.py phase2 --tag w1 --run-training
python scratch/run_hints_v1.py phase3 --hints_checkpoint checkpoints_stage2/<exp>/last.ckpt
python scratch/run_hints_v1.py phase4 --injection both   # + --run-training

# The individual commands it runs (kept here because each is useful alone):
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
python scratch/eval_stage2_wer.py --num_items 200 \
    --checkpoints checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/best-checkpoint.ckpt \
                  checkpoints_stage2/transformer_v0.9/last.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt

# Phase 1: repetition / degeneration, with the ground-truth control arm
python scratch/eval_stage2_repetition.py --num_items 8 --checkpoints \
    checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt \
    checkpoints_stage2/transformer_v0.8/last.ckpt \
    checkpoints_stage2/transformer_v0.9/last.ckpt

# Phase 2: one arm per injection-site combination.  The config carries
# exp_name and the two switches (2_training.py has no --exp_name), so copy
# configs/transformer.yaml per arm, set hint_film_memory / hint_decoder_proj,
# and for a short run also set
#   max_steps = cosine_anneal_end = limit_train_batches = 2000
# so validation and last.ckpt actually happen.  Watch
# val_hint_channel_delta/* and the level sweeps before spending a full run:
python 2_training.py --config <arm>.yaml \
    --transfer_from checkpoints_stage2/transformer_v0.9/last.ckpt

# Phase 3: guidance sweep (needs --hints so cond != uncond)
python scratch/eval_stage2_wer.py --num_items 24 --hints --hint_guidance 2.0 \
    --checkpoints checkpoints_stage2/<exp>/last.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt --ignore_eos
python scratch/eval_stage2_wer.py --num_items 24 --hints \
    --hint_guidance_books 1,1,1,1,1,1,1,1,2,2 \
    --checkpoints checkpoints_stage2/<exp>/last.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt --ignore_eos
# ...and the knob report: level -> decoded intensity / f0 / duration / WER
python scratch/hint_knob_report.py \
    --checkpoint checkpoints_stage2/<exp>/last.ckpt \
    --generator_checkpoint checkpoints_stage1/base_v0.7/last.ckpt \
    --channel intensity,register --levels all --num_items 8 --wer
```



