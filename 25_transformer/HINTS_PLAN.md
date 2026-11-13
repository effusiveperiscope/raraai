# Stage-2 conditioning hints — plan and status

**Status (2026-11-12):** M0–M4 are done and validated, the corpus is built
(151 653 `.cond`, §7) and the v0.8 run has been evaluated. **v0.8 regressed**
(free-running WER 42 % vs 15 % for the checkpoint it was warm-started from;
§8.6), so v0.9 replaces it: relative register, zero-initialised hint tables and
`fim_rate: 0.2`. M3b (per-channel probes) and M5 (inference ergonomics) are not
started.

**Register objection — resolved in v0.9.** The absolute register channel was
~79 % between-speaker variance (ICC(1) = 0.794), i.e. redundant with `spk_emb`
and confounded with it. It is now bucketed on the **deviation from the
speaker's own median** (§8.5), which is orthogonal to `sid` by construction.
The ablations that would *confirm* the benefit are still deferred — they need a
coherent model.

Deep reference material lives in `AGENTS.md` (§4 "Conditioning hints",
§6.20/§6.21) and `hints/README.md`; this file is the plan, the status, and the
evidence.

---

## 1. Goal

Give the stage-2 `CodeTransformer` **optional, coarse control knobs** over how a
sentence is spoken — loudness, register (pitch level), per-word pitch, per-word
pacing, and a sentence-level semantic vector — without touching the stage-1 code
space.

Two things this buys:

1. **Inference-time control.** Today a stage-2 model maps text + speaker id to
   codes and nothing else; the only way to change *how* something is said is to
   change the speaker or resample. The hints are extra conditioning a UI can
   set ("more intensity", "raise this word a major third", "make this word 12
   frames").
2. **Prosody control without a new code space.** Stage 1 owns the code space and
   takes months, so steering *how* a fixed voice speaks should not require
   re-training it. A hint-enabled stage 2 can lean louder/quieter or flatter/
   wider within an existing voice; a genuinely new timbre still needs stage 1
   (see `--acoustic_only`).

The constraint that shapes everything: **the codes must not move.** Every hint is
an extra input, so a hint-enabled model is a new parameter set but the same
`spec_codes`/`pitch_codes` — no re-preprocessing, and the existing corpus stays
valid.

---

## 2. Design in one page

Five additive side channels, all owned by `CodeTransformer`:

| channel | levels | source | applied to |
|---|---|---|---|
| intensity | `n_levels` corpus quantiles | DSP regressor on the 48 kHz wave (calibrated + whisper-gated, 1–9) | encoder + decoder (constant per clip) |
| register | `n_levels` corpus quantiles | voiced-median f0, semitones re: 100 Hz — **absolute; under review, see §8.5** | encoder (constant per clip) |
| semantic | 384-d continuous | `bge-small-en-v1.5` CLS, L2-normalised | encoder + decoder (`Linear(384, H)`) |
| word pitch | 7 voiced bins + 1 unvoiced | word's voiced-mean f0, semitones re: the utterance's voiced median | encoder, per phone |
| word timing | 8 log2 buckets | word's start-to-start duration in code frames | encoder, per phone |

Decisions worth remembering:

- **`.cond` stores continuous values; bucketing happens at load.** The number of
  levels is therefore a config change, not a re-run. Only the *edges* live in
  `hints.json`, and `hints.json` is part of the checkpoint contract (a different
  file means different level semantics, so `check_special_layout` refuses a
  mismatch on row counts).
- **Level 0 is a trained "not provided" state** in every table, so hint-free
  inference is a supported mode rather than a hole. For word pitch the
  *unvoiced* class is the **last** row, not 0 — "this word has no f0" and "no
  hint" are different.
- **Intensity/register use corpus quantiles; pitch/timing use fixed edges.**
  The former are absolute scales (equal-frequency edges make the table usable
  across corpora); the latter are relative or physical, and an inference UI has
  to be able to say "a major third" and mean it.
- **Register is the one channel whose *scale* is under review.** It is absolute
  (semitones re 100 Hz) and therefore largely a speaker property — measured
  ICC(1) = 0.79 against `sid` — which makes it redundant with `spk_emb` and, at
  inference, a voice-swap dial rather than a pitch dial. The plan is to condition
  on the *deviation from the speaker's mean* instead; see §8.5. Word pitch is
  already relative for exactly this reason.
- **Dropout per channel** plus `hint_null_all` on training items, because every
  hint is derivable from the target audio — at train time they are an oracle.
- The pitch/timing fields need forced-alignment word timestamps, hence a
  separate `--words` pass; the three clip-level hints need no alignment.

---

## 3. Milestones

| id | scope | status |
|---|---|---|
| **M0** | Port the DSP intensity stack from `20_intensity_annotation` into `hints/` (sklearn 1.9 re-fit, `verify_dsp.py` acceptance test), plus `hints/levels.py`, `model_io.py`, README | **done** |
| **M1** | `dataset_codes.py`: read sibling `<base>.cond`, bucket with the shared `hints.json`, emit `hint_*` batch keys, per-channel dropout + `hint_null_all`, stale/missing counters | **done** |
| **M2** | `modeling/transformer.py`: hint tables, encoder clip + per-phone channels, decoder clip channels, threaded through `encode`/`code_loss`/`generate`/`infill`/`compute_batch_loss`; `check_special_layout` guards | **done** |
| **M3** | Training-time observability: `val_loss_nohint` / `val_hint_gap` oracle probe | **done** |
| **M3b** | Per-channel counterfactual probes (drop *one* channel, re-score) | **todo** |
| **M3c** | Register re-parameterisation: condition on `register_st − speaker mean` instead of the absolute level (§8.5) | **done (v0.9)** |
| **M4** | `hints/build.py --words` (aligner pass), `--fit-edges-only`, `features.py` one-stop API, resumability | **done** |
| **M5** | Inference ergonomics: CFG/guidance knob, `stage2_repl.py` controls, docs for a UI | **todo** |
| **M6** | Train a real v0.8 run on the hinted corpus; evaluate | **blocked on the aligner pass** |

---

## 4. Two prerequisites that came out of M4

The `--words` pass forced two things that are useful well beyond the hints.

### 4.1 One tokenization for both `phone_ids` paths

`--phones_from aligner` tokenized **per word**; `--phones_from transcript`
tokenized the whole sentence. The g2pen tokenizer merges phones across word
boundaries, so the same 20-word clip gave **71 ids** on one path and **44** on
the other — which made `phone_word_idx` impossible to derive consistently.
`features.phonemize_items(items, transcript=...)` now builds the sentence string,
tokenizes **once**, and gives each item a `phone_span` slice, so:

    itertools.chain(it["phone_ids"] for it in items) == phonemize_text(transcript)

`word_index_for_phones` then only has to *check* that the stored ids are exactly
those slices (returning `None` when the corpus predates the current
phonemizer/dictionary).

Verified: **200/200** stored tensors across the four corpora reproduce exactly;
a real `2_preprocess.py --only phones --phones_from aligner` run now produces
**44 and 26** ids for two clips, **identical** to the transcript path and to the
existing corpus. Remaining divergence is honest: in ASR mode (no caller
transcript) the aligner's own token list is the source, so a word it cannot place
is genuinely absent.

### 4.2 The word pass had to become resumable

`hints/build.py` used to accumulate everything and write at the end, so an
interruption after hours wrote **nothing**. Every stage now persists as it goes:

- the audio stage flushes in **waves of `FLUSH_EVERY` (2000) clips** — at ~13
  clips/s that caps an interruption at ~2.5 min of recompute;
- the sentence-embedding stage flushes all clip-level records before the word
  pass starts;
- the word loop flushes each clip immediately after its words are computed.

`--skip-existing` resumes from whatever is on disk (it only redoes clips whose
word fields are missing), `--device` is validated before any model loads, and a
partial run refuses to refit the corpus-wide level edges (see §6). A clip can
therefore be written twice — `coverage.clips_written` counts **files**,
`coverage.writes` counts writes.

---

## 5. Evidence

| what | measured |
|---|---|
| parameters, `hints: false` | 135.95 M (bit-identical to the pre-v0.8 code) |
| parameters, `hints: true` | 136.28 M (**+0.24 %**) |
| oracle probe (2-step smoke run) | `val_loss_nohint = 4.5637`, `val_hint_gap = +0.0054` — the tables are alive but barely used at step 2, which is the expected starting point |
| `.cond` size | **4–10 KB**/clip (tensors are a few hundred bytes; the rest is `torch.save` container overhead) |
| `.cond` corpus (final) | **151 653 files, 0.74 GB**, avg 4.8 KB, max 9.7 KB — no file over 1 MB |
| intensity / semantic | 151 653 / 151 653 (100 %; 39 394 via the source-audio fallback) |
| register | 112 231 / 151 653 (74 %) — VCTK + Expresso have no stage-1 `.f0` |
| **register vs `sid`** | **ICC(1) = 0.794** (Genshin train split, 2 705 speakers, 22 821 clips with a finite register): between-speaker std **6.94 st**, pooled within-speaker **3.52 st**; a speaker's mean bin predicts the clip's bin **66.7 %** of the time (mean error 0.36 bins); 55 % of speakers with ≥5 clips span ≥2 bins — see §8.5 |
| speaker skew | top 10 % of speakers = **85 %** of clips; 257 speakers (≥100 clips) = 84 %; 1 474 speakers (≤4 clips) = **2 %** |
| word fields | 147 991 / 151 653 (**97.6 %**); 3 546 stale `phone_ids` + 116 aligner failures keep their clip-level hints only |
| level edges | intensity 8 levels from 151 653 values; register 5 levels from 112 231 — corpus-wide, not a subset |
| codes ↔ checkpoint | re-encoded 2 clips with `checkpoints_stage1/base_v0.7/last.ckpt`: `spec_codes` and `pitch_codes` **bit-identical** to the corpus |
| word pass throughput | ~5 clips/s documented on the full corpus; 13–22 it/s measured on short warm clips ⇒ ~8–9 h |
| stale `phone_ids` rate | 3/200 Genshin clips (1.5 %) — dropped, not guessed |
| tokenization agreement | 200/200 stored tensors; 44 vs 44 and 26 vs 26 on the end-to-end rerun |
| tests | `test_hints.py` **59 PASS**, `test_fim.py` passes, `test_phonemize.py` 24/10/32 PASS |

---

## 6. Bugs found while doing this (all fixed, all guarded)

1. **Every `.cond` was 116.5 MB → 440 GB → disk full.** `torch.save` serialises a
   tensor's *whole storage*, and the sentence embedding was stored as a **row**
   of the batched `(n_clips, 384)` matrix. The size is proportional to corpus
   size, so small fixtures looked perfect (6 KB at 8 clips) — it is only visible
   at scale. `write_cond` now clones every tensor and refuses anything over
   4 MB; 4 regression checks in `test_hints.py`.
2. **A killed run wrote nothing.** Fixed by per-stage writes (§4.2). Verified:
   killing a 200-clip run at 139 left 139 valid files, and a resume finished the
   rest; with the audio-stage flush waves, files appear *while* that stage runs
   (100 files at 195/400 with `FLUSH_EVERY = 100`).
3. **A mistyped `--device` failed minutes in.** Now validated before any model
   loads (`cpu` / `cuda[:N]`, checked against `torch.cuda.device_count()`).
4. **A resumed run refitted the level edges from the few clips it still had to
   do**, which would have silently renumbered every level — the one failure mode
   that corrupts an otherwise-fine corpus. Partial runs now keep the existing
   edges and print the `--fit-edges-only` remedy.

---

## 7. Where we are right now

- **Code:** committed (`d5a69c2`); `test_hints.py` 59 PASS, `test_fim.py` and
  `test_phonemize.py` green, `hints: false` is bit-identical to pre-v0.8.
- **Corpus: done.** 151 653 `.cond` files (0.74 GB) beside the stage-2 tensors,
  with `hints.json` fitted on the whole corpus. Verified on the real data: 40/40
  sampled clips have `n_phone` matching their `phone_ids`, unit-norm semantics
  and self-consistent word fields (slots chain into the clip length, NaN exactly
  where unvoiced); the val dataset yields non-zero levels for 60/60 sampled
  items. Rates and the two tolerated failure classes are in §5.
- **Config: flipped.** `exp_name: transformer_v0.8`, `hints: true`,
  `hints_json: data/genshin_plus_extras/hints.json`, and
  `train.generator_checkpoint` now points at
  `checkpoints_stage1/base_v0.7/last.ckpt` — whose codes were verified
  bit-identical to the corpus, so sampled audio is in the right voice.
- **Disk:** `/mnt/data` at 90 % (410 GB free); the hints cost 0.74 GB.

---

## 8. Remaining work

### 8.1 M3b — per-channel counterfactual probes

`val_hint_gap` currently drops **all** channels at once, which conflates them: a
flat gap could be one dead table or five weak ones. The fix is the same probe per
channel, reusing `_hints_off` with a single key:

    val_hint_gap/{int, reg, sem, pitch, slot} = CE(drop that one) - CE(all)

Cost: 5 extra forwards per epoch on the 2 probe batches (12 forward passes, no
backward) — negligible, and diagnostics-only. It answers "which knob is doing
work", which is exactly what M5's UI needs to know before exposing sliders.

For `reg` specifically the probe needs a **second control arm**: replace the
channel with the *speaker's mean bin* instead of dropping it. Dropping it removes
speaker information the model may have learned to read from there, so a channel
that is pure speaker redundancy still shows a healthy gap — the plain drop probe
cannot tell "useful prosody" from "speaker shortcut" (§8.5).

### 8.2 M5 — inference ergonomics

- **CFG / guidance.** Today `generate`/`infill` accept `hints=`, i.e. plain
  conditional sampling. A guidance knob (`logits = (1+w)·logits_hint −
  w·logits_no_hint`) doubles the forward cost per frame at inference and needs a
  decision on the default; a cheaper alternative is "strength = probability of
  dropping each channel", reusing the trained level-0 state. Recommend shipping
  the drop-probability form first (free, matches training), guidance later.
- **`stage2_repl.py` has no hint support at all** (0 occurrences of `hint`). It
  needs: read the val item's `.cond`, per-channel sliders (intensity 0–8,
  register 0–4, word pitch −3..+3 st, slot scale), and CLI `--hint_*` overrides
  so a one-shot `--text ... --sid N --hint_intensity 7` works.
- **A UI contract.** `features.extract_hints` (lazy `hints/` import) is the
  intended entry point: audio + text → a `.cond`-shaped dict. Document the
  level ids per channel there.

### 8.3 M6 — the actual run

Fresh `exp_name` (`transformer_v0.8`), `--transfer_from
checkpoints_stage2/transformer_v0.7_fim2/last.ckpt`, `hints: true`,
`hints_json: data/genshin_plus_extras/hints.json`, `train.hint_probe_batches: 2`.
Watch, in order: `val_hint_gap` (should grow from ~0), `val_loss_nohint` (must
*not* rise — that is the VC failure mode), and `val_loss`.

### 8.4 Evaluation plan (numeric, not just listening)

| channel | measurable prediction |
|---|---|
| intensity | decoded RMS / loudest-frame dBFS should move monotonically with the level |
| register | voiced-median f0 of the decoded audio vs level (RMVPE) — **necessary but not sufficient**: a voice swap also moves f0, so pair it with the identity check in §8.5 |
| word pitch | per-word f0 delta ≈ the bin's semitones |
| word timing | word durations follow the slot bucket |
| semantic | hardest; check that swapping two sentences' vectors in the *same* text changes prosody, and that the gap probe is non-zero |
| regression | hint-free WER (`scratch/eval_stage2_wer.py`) ≥ v0.7's, against the ground-truth-code floor |

### 8.5 Register is confounded with the speaker embedding (open objection)

**The objection (raised 2026-11-11).** `spk_emb` is a learned per-speaker vector,
so it can already represent a speaker's mean f0. A *global* register channel is
therefore redundant with it — and worse, confounded: the model can read register
as a second, coarser speaker id.

**Measured, and the objection is right for the bulk of the signal.** Genshin
train split, 2 705 speakers, 26 361 clips sampled (22 821 with a finite
register), bucketed with the shipped edges:

| quantity | value |
|---|---|
| overall register std | 7.45 st |
| between-speaker std of per-speaker means | **6.94 st** |
| pooled within-speaker std | **3.52 st** |
| **ICC(1) = between / total** | **0.794** |
| bin accuracy from the speaker's mean bin | **66.7 %** |
| mean \|bin error\| of that predictor | 0.36 bins |
| speakers with ≥5 clips spanning ≥2 bins | **55 %** (median range 2 bins) |

So ~79 % of the register variance is a speaker property, and one number per
speaker reproduces two thirds of the bins exactly. But the residual 21 %
(3.3 st within-speaker) is real per-utterance prosody — excited vs calm delivery,
shouting, whispering — and a single per-speaker vector provably cannot express
it. **The concept survives; the absolute parameterisation does not.**

**Why redundancy is the lesser problem and confounding is the real one.**
Register is an oracle derived from the target audio (`hints/build.py:264-283`,
from the same `.f0` the pitch codec is trained on), so it is a cheap 5-way
summary of "who is this" that is *easier to fit* than a 4 096-row speaker table.
The model can learn "bin 5 ⇒ this timbre/formant set", and then the knob is a
**voice-swap dial**: set register 5 on a low-voiced speaker and you do not get
that speaker an octave up, you get dragged toward the high-register speakers in
the training set. That is the failure mode that looks fine on the training
distribution and breaks exactly when a user turns the knob.

`val_hint_gap` **cannot see this.** Dropping register removes speaker information
the model had learned to read from there, so a channel that is pure speaker
redundancy still produces a healthy gap. The existing probe answers "is the model
using hints", not "is this hint redundant with `sid`".

**The corpus skew makes it worse, not better.** Top 10 % of speakers = 85 % of
clips; 257 speakers (≥100 clips) = 84 % of clips; 1 474 speakers (≤4 clips) =
2 % of clips. For the 84–85 % of the training signal where `spk_emb` is well
estimated, register is ~79 % redundant. The only place it carries information
`sid` lacks is the 2 % long tail — so the channel is paid for by everyone and
earns its keep on 2 % of the data.

**And structurally it is a weak knob anyway.** Register is **encoder-only**:
`_clip_hints` adds `reg_emb` to the phoneme memory
(`modeling/transformer.py:1072-1074`), while `_dec_hint` (`:1097-1106`) feeds
only intensity + semantic to the decoder. The actual f0 comes from the
pitch-codebook heads in the decoder, so register can only move f0 *indirectly*
through cross-attention — the same weak channel the codebase already documents as
easily averaged away. If the goal is a real global register control, the current
wiring is the wrong place for it; if the goal is "tell the encoder about the
voice's pitch range", `sid` already does that.

#### Implemented (v0.9): condition on the deviation, not the level

The value that gets bucketed is `register_st − speaker_mean_register_st`. This
resolves the objection by construction:

- the between-speaker 79 % is delegated to `spk_emb`, where it belongs;
- the channel carries only the within-speaker residual (3.3 st, ~0.6 bin);
- the inference knob becomes "**+2 st for this line**", which is both more useful
  and not a voice swap;
- level 0 becomes "**typical for this speaker**", a far better default than
  "unknown";
- it makes the design **internally consistent**: word pitch is already relative
  to the utterance's voiced median (`hints/levels.py:91-100`, whose docstring
  argues for relative-over-quantile), and register is the one channel that broke
  that rule.

**As built — the corpus was *not* rewritten.** The plan said to write
`register_rel_st` into each `.cond`; that would have been a 151 k-file / 0.74 GB
rewrite for a value that is an exact function of two things the loader already
has. Instead the float stays absolute in `.cond` (so one corpus serves both
parameterisations) and the subtraction happens at **load time**:

1. `hints/build.py --relative-register` collects `(sid, register_st)` pairs,
   freezes `{str(sid): mean_st}` into `hints.json` →
   `levels.register.speaker_means`, and fits the edges on the residual. The
   means come from the filelist you pass, so pass the **training** split:

   ```bash
   python hints/build.py --filelist data/genshin_plus_extras/train_stage2.txt \
       --hints intensity,register \
       --hints-json data/genshin_plus_extras/hints_rel.json \
       --fit-edges-only --relative-register
   ```

   Measured: **2 591 offsets**, residual std **3.58 st**, mean 0.00, edges
   `[-2.48, -0.98, 0.34, 2.38]` st (vs `6.43…23.06` absolute), 107 093 residuals
   from 146 513 clips. `unit` is recorded as "semitones re the speaker's median
   f0", so the file is self-describing.
2. `hints/levels.py:register_rel_st(value, sid, speaker_means)` does the
   subtraction; an unseen `sid` maps to 0.0 = "typical for this speaker" (the
   same meaning as level 0 elsewhere).
3. `dataset_codes.py` looks the offset up with the item's **raw, pre-`sid_map`**
   id (`item["sid_raw"]`), which is the id `build.py` keyed on, so a speaker map
   cannot make the two disagree. Presence of `speaker_means` *is* the switch.
   Verified: the level histogram over 300 val clips is 52/63/57/68/60 across
   bins 1–5, i.e. the equal-frequency fit survives, with no clip in "absent".
4. `modeling/transformer.py` — **no shape change** (same `n_levels + 1` rows), so
   `--transfer_from` still works. Because the ids change meaning without
   changing any shape, `transformer.relative_register` is asserted against the
   file in `CodeTransformer.__init__` (both directions raise). This is the one
   guard `check_special_layout` structurally cannot provide.
5. `features.extract_hints` is **unchanged**: it writes the absolute
   `register_st`, which is what the loader subtracts from. An inference UI that
   wants a relative value calls `register_rel_st` with the speaker's offset.

Cost: no `.cond` rewrite, no re-preprocessing of the codes, and the refit is a
read-only pass over the existing files (~1 min for 146 k clips).

#### Deferred: the ablations that would decide it

Three cheap tests, all of which need a **trained** model, so they are blocked
until the current run produces coherent speech (a loss delta on an incoherent
model measures nothing) — and the run cannot be restarted right now:

1. **Redundancy** — val_loss with register dropped vs. with register *replaced by
   the speaker's mean bin*. If the second ≈ the first, register carries no
   within-speaker information and should go; if it ≈ the full model, it does.
2. **Confound** — hold `sid` fixed, sweep register 1→5, and measure the f0 shift
   **and** speaker similarity (ECAPA cosine, or formant drift). If identity moves
   more than f0, it is a voice-swap dial.
3. **Leak** — evaluate with register shuffled *across speakers*. A model that
   learned "register = speaker" degrades sharply; one that learned prosody barely
   moves.

(1) is a per-channel extension of M3b with a speaker-mean control arm (§8.1);
that is the probe to add, and it is worth having regardless of which way this
goes.

#### Interim decision (no code change now)

Leave the shipped **absolute** register in place for the current run, because
(a) the run is already going and cannot be restarted, (b) the channel is dropped
on 10 % of items (`hint_dropout.register`) plus `hint_null_all: 0.10`, so the
model is forced to work without it, and (c) the relative form needs a `.cond`
rewrite plus an edge refit, which is a fresh-run change anyway. Treat the current
run's register channel as an experiment whose result the ablations above will
interpret — not as a validated knob.

**Doc nit found while measuring:** `hints/build.py:18` still says "utterance
voiced **mean** f0", while the code (`:264-278`), `hints/README.md:158` and
`AGENTS.md` all say **median**. The header comment is stale; fix it with the
M3c change.

### 8.6 The v0.8 regression, and what v0.9 changes

The v0.8 model stutters/repeats words. Measured with
`scratch/eval_stage2_wer.py` on the **same 24 val utterances**, paired seed, same
vocoder (Qwen3-ASR; the "floor" arm decodes the ground-truth codes):

| checkpoint | step | WER | S / D / I |
|---|---|---|---|
| `5090_v0.7_fim2/best-checkpoint` | 213 710 | **15.00 %** | 18 / 44 / 4 |
| `5090_v0.7_fim2/last` ← v0.8's warm start | 335 830 | **15.23 %** | 26 / 36 / 5 |
| `transformer_v0.8/model-step=200000` | 200 000 | **43.18 %** | 61 / 118 / 11 |
| `transformer_v0.8/last` | 231 990 | **42.05 %** | 72 / 109 / 4 |
| floor (ground-truth codes, same vocoder) | — | 5.5–6.8 % | — |

So v0.8 degraded **15 % → 42 % during its own training**, and was already 43 % at
step 200 k. Its teacher-forced `val_loss` meanwhile *improved* (2.295 vs v0.7's
2.306) — the two metrics decoupled, which is the fact to remember.

**Ruled out, with measurements:**

- **The hints.** Regenerating with each item's real `.cond` hints supplied gives
  *identical* WER (42.05 % both arms, identical transcripts); `val_hint_gap` is
  0.021 nats. The channels were **inert**, not harmful. (New
  `scratch/eval_stage2_wer.py --hints` measures both arms; before this nothing
  in the repo passed hints at inference at all, so the mismatch could not be
  seen.)
- **EOS / duration.** `--ignore_eos` (force the reference length) and normal EOS
  give the same WER.
- **Undertraining.** Step 200 k = step 232 k, and val_loss is below v0.7's best.
- **Broken weights.** All 327 common tensors load, no missing keys, only the 8
  expected new hint tables.
- **Dead text conditioning.** `val_text_delta` = 0.41 nats (v0.4, the documented
  failure, was +0.003); encoder memory spread 1.27 (v0.7: 1.33); the best
  monotonic cross-attention head reaches corr **+0.97** (v0.7: +0.59). The text
  channel is *sharper* than v0.7's.
- **Decoding knobs.** Greedy collapses to a constant frame (97 % periodic, 3 %
  unique); temperature 1.0 / no top-k is worse (58.6 %). The shipped 0.9 / 20 is
  the best of the three.

**Mechanism.** Deletion-dominated (D 109 vs S 72 vs I 4): the model says a
phrase, re-says it, never advances past that point, so the rest of the sentence
is never spoken. Adjacent-frame repetition in the generated codebook-0 stream is
**0.18–0.20 for v0.8 vs 0.11 for v0.7**. Transcripts show it directly —
*"Why? Why would Walter steal? Steal something that can't sell off?"*,
*"Don't give up! Don't give up."* (dropping "You'll get the hang of it"). It is a
**repetition attractor**, not an alignment collapse: v0.8 is sharper and more
confident, and a sharp attention makes a repetition self-reinforcing.

**v0.9 changes**, cheapest-first by evidence:

1. **Zero-initialised hint tables** (`modeling/transformer.py`). `encode` adds
   `int_emb + reg_emb + sem_proj(sem)` to *every* phoneme, so a randomly
   initialised table is a large random perturbation of a converged text memory,
   applied to 90 % of training items — at exactly the moment a `--transfer_from`
   model's alignment is most fragile. With zeros, a warm start reproduces the
   source model exactly and the tables must earn their effect.
   `test_hints.py` now checks both halves (zeros at init, real effect once
   perturbed).
2. **`fim_rate: 0.5 → 0.2`** (`configs/transformer.yaml`). Half of v0.8's
   gradient went to `prefix <SUF> suffix <MID> middle`, which teaches "continue
   from a given context", not "advance monotonically through the text". Plain AR
   is what free-running generation does.
3. **Warm start from `.../best-checkpoint.ckpt`**, not `.../last.ckpt` — the
   latter is the post-anneal checkpoint v0.8 started from, which §4 already flags
   as the worse one.
4. **Relative register** (§8.5).

`exp_name: transformer_v0.9`. Not yet trained, so none of this is a measured
improvement yet — the entry criterion is WER back to ≤ v0.7's ~15 % on the same
24 utterances.

---

## 9. Runbook (once the aligner pass finishes)

```bash
# 1. the pass itself (resumable: add --skip-existing if it stops)
python hints/build.py \
    --filelist data/genshin_plus_extras/train_stage2.txt data/genshin_plus_extras/val_stage2.txt \
    --wave-dir data/genshin_full_stage1 \
    --source-filelist filelists_2/genshin_full_eng_text_linux.txt \
    --hints intensity,register,semantic --words --device cuda:0

# 2. refit the level edges over every .cond (only needed after a sharded/resumed run)
python hints/build.py --filelist data/genshin_plus_extras/train_stage2.txt --fit-edges-only

# 3. flip the switch (fresh exp_name -- the tables are new parameters)
#    transformer.hints: true
#    transformer.hints_json: data/genshin_plus_extras/hints.json
python 2_training.py --config configs/transformer.yaml \
    --transfer_from checkpoints_stage2/transformer_v0.7_fim2.ckpt
```

---

## 10. Risks and open questions

1. **Oracle leakage.** Every hint is computed from the target audio, so a strong
   model can learn to read the answer instead of the text. The gap probe is the
   guard; if `val_hint_gap` grows while `val_loss_nohint` rises, lower the
   per-channel dropout or the hint capacity. `hint_null_all: 0.10` is a starting
   point, not a measurement.
2. **Semantic ⇒ prosody smuggling.** A 384-d sentence vector carries more than
   "what is said"; it can leak speaking style from the recording. If the
   semantic channel dominates the gap, that is the signal.
3. **Stale `phone_ids`.** ~1.5 % of clips are dropped by the word pass because
   their ids predate the current phonemizer. Fixing them means
   `2_preprocess.py --only phones` and re-running the pass for those clips.
4. **`hints.json` is a checkpoint contract.** Row counts must match the file the
   run was trained with; keep it with the run.
5. **Two stages, two id spaces, unchanged.** Hints do not affect the speaker-id
   canonicalisation (`speaker_map.py`); that footgun is untouched.
6. **Not yet wired:** hints at inference in the REPL, guidance, and any
   evaluation of a *trained* hints model — everything so far is validated on the
   untrained tables (loss finite, shapes right, hints reach the model).
7. **Resume cost is bounded by the *remaining* clips.** A resume reuses nothing
   from a `.cond` that still lacks its word fields, so the audio hints for those
   clips are recomputed. In practice that is the remaining tail (e.g. 20 % of the
   corpus ≈ 40 min), not the whole pass — but a crash *exactly* between the audio
   stage and the word pass would redo the audio stage in full. Reusing the
   clip-level values from disk would remove that; it is a deliberate omission
   rather than an oversight (it would add an untested path right before an
   unattended 8 h run).
8. **Register is confounded with `spk_emb` (§8.5).** Measured ICC(1) = 0.79
   against `sid`, so the absolute register channel is mostly a second, coarser
   speaker id — and because it is an oracle, the model can use it as a speaker
   shortcut, which turns the inference knob into a voice-swap dial. `val_hint_gap`
   cannot detect this (dropping the channel removes speaker information too). The
   plan is a relative register (`register_st − speaker mean`); until that lands,
   treat the channel as unvalidated and do not expose it as a UI slider.

---

## 11. Reproducing the evidence

```bash
# the whole hint stack on CPU (levels, word math, one-stop API == builder,
# dataset bucketing/dropout, model side channels, .cond size guards) -- 59 checks
python test_hints.py

# FIM / span regeneration still fine next to a GPU job
python test_fim.py

# both phone_ids paths agree (transcript path needs no GPU/weights)
python test_phonemize.py --skip_audio
python test_phonemize.py            # ...plus ASR mode, informational

# a real .cond is 4-10 KB; anything bigger means a tensor is sharing storage
python -c "import torch,glob,os; p=sorted(glob.glob('data/*_stage2/*.cond'))[:5];
[print(os.path.getsize(q)//1024,'KB',os.path.basename(q)) for q in p]"

# per-tensor storage vs shape: on a healthy file storage is ~shape x dtype.
# The bug looked like shape=(384,) with storage=0.31 MB and offset=8832 (a row
# of the 400-clip matrix); at 151k clips that was 116 MB in every file.
python -c "
import torch,glob,os
p=max(glob.glob('data/*_stage2/*.cond'), key=os.path.getsize)
for k,v in torch.load(p, weights_only=True).items():
    if hasattr(v,'untyped_storage'):
        print(k, tuple(v.shape), v.dtype, round(v.untyped_storage().size()/1e6,2),'MB')"

# the hint tables are alive: run 2 steps with hints: true and read the probe
python 2_training.py --config configs/transformer.yaml
#   -> val_loss_nohint (all channels dropped) and val_hint_gap (~0 at step 2,
#      grows as the model starts using them)
```



