# 26_guided_tts — agents guide

A rewrite of 25_transformer stage 2 as **plan → renderer**: a word-rate *plan* (durations,
pitch contour, intensity, nonverbal tokens) is expanded to frame rate and
rendered by a **non-autoregressive** decoder over the stage-1 RVQ codes.
Stage 1 (codec + NSF vocoder) is reused unchanged from `../25_transformer`.

Read `../25_transformer/AGENTS.md` for the stage-1 codec, the `.cond` format,
`features.py` and the phonemizer. This file covers only what is new, plus the
measurements that motivated the rewrite.

---

## 1. Why this project exists

Four stage-2 arms were measured on `25_transformer` (120 val utterances, ASR
WER, floor = ground-truth codes through the same vocoder = 6.1 %):

| arm | what changed | WER | notes |
|---|---|---|---|
| `transformer_hints_v2.4` | – (baseline) | **15.75 %** | S117 D161 I25 |
| `transformer_word_progress_v1` | word-clock auxiliary loss, 136 k steps | 14.76 % | CI [−3.5, +1.3] ⇒ no effect |
| `transformer_duration_v1` | oracle phone-progress as a decoder *input* | +0.1 pp vs its paired control | channel never learned (`prog_dir` norm 0.27 vs `input_bias` 13.2) |
| `transformer_align_v2` | cross-attention supervised to the alignment | +2.3 pp | CI [−0.35, +5.1] ⇒ no effect |

Three findings drive this project:

1. **Guidance channels do not reach the model.** `hint_slot` +0.004 nats,
   `hint_int` −0.0005, `hint_reg` +0.003 when dropped. The cause is structural,
   not a weight: `memory_norm: temporal` standardises every memory dimension
   *over time*, which subtracts any constant added to all phones, so clip-level
   conditioning is erased before the decoder sees it.
2. **There is no alignment to pace with.** Cross-attention entropy / log N is
   **0.96–0.99 in all 12 layers** (1.0 = uniform), on every checkpoint, with a
   healthy memory (phone vectors pairwise cosine ≈ 0). The decoder sequences
   from an acoustic prior. Consequence: length is not a function of the text
   (halving the phoneme sequence leaves the generated length unchanged).
3. **The failure survives the obvious fixes.** ~1 word in 5 of the final 20 %
   of each sentence is dropped; forcing the exact code length makes WER
   *worse* (14.8 → 20.3 %), and it stays unchanged when the attention map is
   made to track the alignment exactly (`|pos − oracle| = 0.015`). Supervising a
   map changes what it looks like; it does not change what the decoder is
   driven by.

Conclusion: in an AR codec decoder, *time is a decision* that the code CE never
trains, and soft side channels are ignorable. Both have to become structure.

## 2. Goal

A strongly guidable voice-imitation TTS for voice acting / content creation:

- **coarse, human-settable prosody** that the model interprets into natural
  micro-prosody: per-word duration, register/pitch contour, intensity/emphasis;
- **phoneme input**, so OOV words and names can be corrected (the baseline
  mispronounces exactly those: Paimon→Python, Mora→Laura);
- **no skipped or stretched words**;
- **nonverbal phenomena** (sighs, laughs, breaths) under explicit control;
- **small**: consumer GPU, ideally CPU;
- trainable on ~100 h / one 16 GB card, and useful *without* guidance in
  low-resource settings.

Not voice conversion: prosody must be settable independently of any source
audio, so the control surface is symbolic/numeric, never extracted from a
reference's acoustics.

## 3. Architecture

```
plan (word rate)                           renderer (non-AR, one pass)
  segments: <sil> / <word>                  phones → phoneme encoder
  dur_frames[word]                   ──→    → expand at 100 Hz by phone duration
  dur_frames[phone] @100 Hz                 + per-frame plan features
  register + pitch anchors (st)             → frame decoder (L = Σ word + pause)
  word level (dB)                           → 8 parallel spec heads
  emphasis / <sigh> / <laugh>               → 2 depth-AR pitch heads
                                            → stage-1 Generator (reused) → audio
```

- **Length is computed, never decided**: `L = lead + Σ(word + pause)`, no EOS,
  no delay pattern. The plan partitions the code frame axis exactly; this is
  what removes the tail truncation by construction.
- **Every frame is built from its phone** (the expanded memory *is* the input),
  not from an attention map the model may ignore. Phone durations are stored at
  **100 Hz**, not 25 Hz: a 40–80 ms word can hold 3–5 phones, and one code frame
  each would either stretch the word or drop a phone. The renderer expands at
  100 Hz and pools the hidden state 4:1 to the code rate.
- **Global context, not just convolutions.** The dilated-conv frame stack has a
  receptive field of `1 + 4·(1+2+4+8+1+2) = 73` code frames (~2.9 s), and the
  phone convs see ~33 phones. Both are too local for a sentence-level plan.
  The renderer adds **1 phone-attention layer + 2 frame-attention layers**
  (non-causal, key-padding masks), which makes every frame see the whole plan.
  The v2 default is 28.7 M params.
- **The content path must be forced, not hoped for.** On the first conv-only
  baseline (v1, 24 epochs, WER 99 % at its best checkpoint) zeroing *all* phone
  input moved spec CE only **3.53 → 3.74 nats**, and zeroing the plan
  3.53 → 3.67. The model was a speaker-conditioned code prior: good f0/level
  (corr ~0.77) with no words. The fix is a **phone-identity auxiliary head**
  (predict the plan's dominant phone per code frame, weight 0.5), which makes
  the hidden retain content; `val_phone_acc` is the diagnostic to watch.
- **Codebook heads: measured, not assumed.** Within-frame dependence is weak in
  this codec: conditioning on all previous books saves ~0.4 bits per spec book
  (marginal 5.6 bits, 64 codes), falling pairwise as 0.38 → 0.13 bits from book
  0→1 to 6→7 (937 k frames). So the 8 spec books get **parallel heads**, with a
  `depth_ar_spec_books` knob (2 steps buys ~0.8 of the ~2.7 bits) as the
  insurance ablation. The **pitch** books are different: `MI(c0,c1) = 1.23` of
  ~6 bits, so they get depth-AR (one extra step).
- **The plan has three sources, one interface**:
  1. *authored* — sliders/markup/an editable per-word JSON;
  2. *derived from a reference recording* — align it, extract per-word duration,
     coarse f0 and energy → a plan (prosody transfer; keep it coarse, never pass
     the source's spectral detail or this becomes VC);
  3. *predicted* — variance adaptors inside the renderer infer the plan from
     text + speaker, and the user overrides the fields they care about. Same
     slots, predicted or set.
- Plan units: **absolute register** (`register_st`, semitones re 100 Hz) plus a
  **contour relative to it** (anchors, semitones re the clip's voiced median),
  word duration in code frames, level in dB re the clip's median word. A
  whole-line "raise it by 2 st" is the register field; it must not be
  re-normalised against the realised median or the guidance erases itself (the
  same failure as `memory_norm: temporal`).
- **Two intensity axes, kept apart** (measured on 1,200 Genshin clips):
  `word_db` is per-word level in dB re the clip median — the clean emphasis
  control, speaker- and clip-normalised. `cond.intensity` is the subjective
  1–9 DSP score; it is **not loudness** (corr 0.08 with peak-normalised level)
  and is strongly tied to pitch/voice type (corr 0.62 with `register_st`, 0.45
  within speaker; ~⅓ of its variance is between speakers, within-speaker std
  0.40). Keep it as a **clip-level style condition**, but expose it as a
  **slider** (user decision: it is cross-influenced with register/pitch yet it
  demonstrably does something they want). `Plan.intensity_rel` is the slider
  value: `(raw - shrunk speaker mean) / global std`, so 0 = this character's
  usual delivery, +1 = more intense than usual. Label it **"Delivery"**
  (or "Energy") in any UI, not "intensity" — the score is partly pitch, and
  the name should not promise loudness. Sweep its adherence separately after
  training; if it only moves f0 it duplicates register and should be demoted
  to a predicted style input.

## 4. Data contract (all read from the symlinked `data/`)

Per clip (`<base>.{phone_ids,spec_codes,pitch_codes,cond}`):

| tensor | shape | meaning |
|---|---|---|
| `phone_ids` | (N,) | g2pen phone ids, the text |
| `spec_codes` | (8, T) | RVQ spec codes @25 Hz |
| `pitch_codes` | (2, T) | RVQ pitch codes @25 Hz |
| `cond.phone_word_idx` | (N,) | phone → word |
| `cond.word_start_f`, `word_slot_f` | (W,) | word start / start-to-start, in code frames |
| `cond.pitch_anchors_st`, `pitch_anchors_pos`, `pitch_anchor_n` | (W,A),(W,A),(W,) | f0 contour per word, semitones re the utterance's voiced median, with positions in the word |
| `cond.pitch_word_st`, `pitch_word_voiced` | (W,) | word mean f0 / voicing |
| `cond.register_st`, `intensity` | (1,) | clip-level register (semitones re 100 Hz) and perceptual intensity 1–9 |
| `cond.word_bert` | (W,768) | per-word bert-base state |

The plan is **derived at load time** from these — no new preprocessing pass.
Word durations are the slot minus the trailing silence found by an energy VAD
over the stage-1 `.spec` (the features are peak-normalised per clip, so the
threshold is relative to the clip's 95th percentile; a code frame is silent when
≥ half of its four 100 Hz frames are below `p95 − 35 dB`). Per-phone durations
are an even split of the word at **100 Hz**; per-word pitch is the anchors
interpolated across the word, with the level supplied by `register_st`.

Five gotchas, all of which have already caused bugs elsewhere:

1. **A clip's code frames include lead/tail silence**, and `word_slot_f` is
   start-to-start with the last slot running to the clip end, so the edge
   silence lives *inside* the first/last word's slot. The plan splits it out
   with the VAD; without that, expanding the edge word's phones over its slot
   stretches them into the silence. Measured on 284 val clips: lead median 1
   frame, tail median 2, pauses 9.1 % of all frames; 13.3 % of words have a
   pause ≥ 2 frames.
2. **Stage directions are stripped** by the current corpus build
   (`features.strip_stage_directions`), so `*sigh*` is *not* in `phone_ids`.
   Nonverbals need a reserved token id written at phonemization time; this is a
   corpus decision, not a model one.
3. **Word timings are word-level**; phone splits within a word are even. Treat
   per-phone durations as a supervised but noisy target, and measure adherence
   at the word level.
4. **`word_slot_f` is start-to-start** (a word ends where the next starts), and
   adjacent words occasionally share a start frame; collapse equal starts into
   one run before deriving durations (3.8 % of words in the val sample).
5. **Zero-phone words exist** (114/400 Genshin val clips carry at least one,
   usually a punctuation token) and so do degenerate runs (2 frames for 3
   words). `plan.py` drops the former and renumbers; a run that cannot give one
   100 Hz frame per phone is rejected (`PlanError`) and the clip is skipped —
   5.5 % of the val sample. Do not "fix" those by stretching: they are broken
   alignments, not fast speech.

## 5. Reuse map

Symlinked (data and artefacts only — never copy, `data/` is 417 GB):

| link | target |
|---|---|
| `data`, `filelists_1`, `filelists_2`, `hints` | `../25_transformer/*` (all corpora, `.cond`, alignments) |
| `checkpoints_stage1`, `checkpoints_stage2` | stage-1 codec + the baselines to compare against |
| `speakers.json`, `speaker_map_genshin.json` | stage-2 → stage-1 speaker id mapping |

Imported by path (`shared.py` appends `../25_transformer` to `sys.path`):
`features`, `hints`, `stage2_wer`, `commons`, `utils`, `speaker_map`,
`horsephonemizer`, `modeling.generator`, `dataset_codes` (for reference only).

**Not reused:** `modeling/transformer.py`, `dataset_codes`' hint plumbing,
`2_training.py`. The AR stage 2 is the thing being replaced; the old runs stay
useful only as baselines and as an expressiveness reference.

**Reserved module names** — do not create files called `features.py`,
`utils.py`, `commons.py`, `stage2_wer.py`, `speaker_map.py`, `hints/` or a
`modeling/generator.py`: the local project root comes first on `sys.path`, so a
same-named file would shadow the shared one for every import here. New modules
are named for what they do (`plan.py`, `renderer.py`, `dataset_plan.py`, ...).

`shared.py` is the only place that knows about the old tree:

```python
from shared import DATA, CORPORA, STAGE1, BASELINE_WER, build_stage1
net_g, cfg = build_stage1()          # frozen 33.7 M codec + vocoder
```

## 6. Metrics

Judge on these, in this order — none of them is CE, which cannot see prosody
(a per-frame code CE is flat across pitch/intensity changes because they move
probability mass between equally plausible codes).

1. **Plan adherence** (primary). Realised vs requested: per-word duration error
   (%), frame-level f0 correlation with the requested contour, energy response
   to the intensity knob, and monotonicity across a level sweep. A guidance
   channel that does not move these is decoration, regardless of CE.
2. **Intelligibility** — `stage2_wer` (ASR WER), same 120-utterance set and
   audit as the baselines: floor **6.08 %**, current best **15.75 %**.
3. **Deletions**, by position (head/mid/tail) — the symptom this rewrite targets;
   the baseline loses 18–23 % of the words in the final 20 %.
4. **Nonverbals** — a listening rubric: present when requested, absent when not.
5. **Cost** — params, step time, and CPU real-time factor.

## 7. Status

- Project created; `shared.py` verified: resolves every artefact and loads the
  stage-1 Generator (33.7 M params, 8+2 books, 3 frames → 5760 samples at
  48 kHz).
- **`plan.py` implemented** (CPU only, no model): word/silence segments at
  25 Hz, phone durations at 100 Hz, 100 Hz pitch contour (`register + anchors`),
  per-word dB, exact partition invariant. `test_plan.py` guards the edge cases
  (equal starts, zero-phone words, edge silence, stale `.cond`).
- `scratch/validate_plan.py`, 1,000 Genshin val clips: partition exact on all
  946 accepted clips, 9.4 % of frames explicit silence, word-level plan pitch
  vs `.f0` **corr 0.977 / slope 1.007 / 67 cents median**, frame-level after
  offset 66 cents (the anchors are a coarse 1–3 points per word; the renderer is
  meant to add the micro-prosody). 5.4 % of clips are rejected: 3.6 % degenerate
  word runs and 1.7 % word/phone length mismatches. On MLP (noisier corpus)
  the word-level figures are 0.927 / 0.967 / 98 cents.
- **Head design, corrected by measurement** (see §3): 8 spec heads and 2
  depth-AR pitch heads, but the spec objective changes with v3. Pre-quantisation
  `z` is still rejected (needs ~18 GB of reprocessed latents; the frozen decoder
  only saw `zq`). `zq` itself is free from the stored codes, and the 64-way CE
  is **metric-blind**: on 3 clips the vocoder log-mel L1 is **3.81** for a random
  wrong code, **1.37** for the nearest other code (same CE loss!), **1.70** for
  nearest-other across all 8 books, and only **0.16–0.26** for 1–30 % Gaussian
  noise on `zq`. A linear probe from the v2 hidden to `zq` gets L1 0.31
  (R² 0.49) and decodes closer than the sampled codes (mel 1.96/2.33/1.80 vs
  2.59/2.74/1.86). v3 therefore uses a **continuous `zq` head** with L1 weight
  5; the 64-way CE is a 0.1 auxiliary. Pitch keeps its codes: round trip on
  `base_v0.7` is **0.22 cents median / 0.6 p90**.
- **Dataset implemented** (`dataset_plan.py`): word-aligned crops, 100 Hz
  phone index for the expansion, per-frame plan features, masked collator, and
  a compact plan cache (`plans/<corpus>/<clip>.npz`, ~8.6 KB/clip) built by
  `scratch/build_plan_cache.py` (287 clips/s). Plans are cached because the
  derivation needs the 0.5 MB stage-1 `.spec`; training must not read 75 GB of
  spectrograms per epoch. Use `--require-cache` after building so degenerate
  alignments are skipped rather than raising mid-epoch.
- **Renderer implemented** (`renderer.py`): phone conv encoder, 100 Hz weighted
  expansion pooled 4:1, dilated-conv frame decoder (non-causal), 8 parallel
  spec heads, 2 depth-AR pitch heads (parallel ablation flag), optional f0
  regression head, **global phone/frame attention**, and a **phone-identity
  auxiliary head**. **18.7 M** conv-only, **28.7 M** with attention + aux at
  512/4+6, **~38 M** at 576/5+7 + attention. A real 4-item batch
  (≤161 frames) is 0.25 s forward+backward conv-only on the 5060 Ti.
- **Training loop implemented** (`train_renderer.py`): masked code CE +
  optional f0 L1, bf16 autocast, warmup/cosine, grad clip, val CE/accuracy,
  `best.ckpt`/`last.ckpt`, resume. Smoke-tested for 4 steps with a 1.7 M
  throwaway model; checkpoints go to the gitignored `checkpoints/`.
- **Intensity slider wired**: `scratch/fit_intensity_stats.py` fitted on the
  combined train split (146,441 clips, 2,704 speakers; global mean 4.328,
  std 0.822) writes `intensity_stats.json` (gitignored, regenerate with
  `python scratch/fit_intensity_stats.py`). `Plan.intensity_rel` uses the
  shrunk speaker mean and the global std; the validator sees it centred at 0
  (p10 −0.59, p90 +0.68). The file is optional: without it the loader falls
  back to the global mean, and a fully authored plan can set the field itself.
- Environment: conda `raraai25` (same as `25_transformer`).

- **Plan cache built** for genshin+mlp train+val: **115,115 plans**, 6,713
  skipped (5.7 %: 4,864 degenerate runs, 1,762 length mismatches, 87 no word
  fields), 428 s. `plans/` is gitignored.
- **Eval path implemented** (`eval_renderer.py`): generated codes -> frozen
  stage-1 -> 48 kHz, ASR WER, plan-vs-generated f0, true-f0-vs-generated f0,
  per-word level, optional oracle decode (codec ceiling), and
  `--register-shift`/`--level-shift` for later adherence sweeps. Validated with
  the oracle: true f0 vs oracle decode **corr 0.996, 25 c median**, oracle WER
  10 % on a 6-item probe. The 400-step smoke model gives WER 108 % and f0 corr
  0.41 -- expected at that budget, and exactly the gap the first real run is
  meant to close.
- **v1 (conv-only) stopped at epoch 24** as a negative baseline: val spec CE
  3.49 nats, WER **99 %** on 12 items (oracle 9.3 %), f0 plan-corr 0.77.
  Checkpoints remain in `checkpoints/renderer_v1/`.
- **v2 finished (conv + attention + phone-aux)**: val spec CE **3.4912** vs
  v1's 3.4867, `val_phone_acc` **0.959**, WER **102.9 %** (oracle 9.3 %). The
  attention and the phone head did not make the content path load-bearing; the
  phone head solved its own task in a subspace the code head ignored.
- **v3 launched**: the same attention/phone-aux model plus a continuous `zq`
  head, L1 weight 5, spec CE auxiliary 0.1, 20 epochs, 400-item val,
  TensorBoard audio every 5 epochs. This is the arm the sensitivity numbers
  call for.
- **Batch-padding decode bug fixed**: the collator pads codes to the batch max;
  the decoder must be called per item with `lengths[i]`, or short items get a
  tail of zero-code frames (an audible drone and extra length). Both the eval
  and TensorBoard loggers now use the per-item path.

Next: read `renderer_v1`'s val curves, then run `eval_renderer.py` on
`best.ckpt` with 120 items + oracle, and start the guidance sweeps
(register/level shifts) once the base WER is credible.

Still open, before the first training run:

1. **Nonverbal tokens**: reserve ids in the existing g2pen vocabulary, or add a
   small parallel token stream? Needs a `--only phones` re-phonemization either
   way.
2. **Pitch plan strength**: a hard target the decoder must hit (concatenative
   feel) or a bias the decoder may deviate from. Add an auxiliary f0-space loss
   through the frozen pitch decoder so the code CE is not the only pitch signal.
3. **Renderer size**: start ~40 M; the point is that ~100 h trains it, not that
   it can absorb more.
4. **Corpus mix**: dense 100 Hz `.f0` and `.spec` exist for genshin (106 k) and
   mlp (28 k), but not for expresso/vctk, which carry only `.cond` word anchors.
   `.cond` itself covers genshin 102,815, mlp 19,088 of 33,765 train items,
   expresso 6,182 and vctk 23,568 — and **mlp's val split has no `.cond` at
   all** (0 of 1,777), so it cannot be used for plan training or validation
   until `hints/build.py --words` is re-run on it. Restrict the first run to
   the two corpora with performance audio, or backfill the alignments.
