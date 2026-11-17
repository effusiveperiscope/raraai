# Stage-2 conditioning hints — spec and status

**Status (2026-11-15): shipped, and the v1 scope is closed.** The design below
is what runs; `scratch/HINTS_V1_PLAN.md` §5.9 owns the measurement record and
the reasons the scope is closed. This file is the durable spec: what the
channels are, what the `.cond` contract is, and what is left undone.

**Model to use for hint-capable inference:**
`checkpoints_stage2/hints_v1_p4_final_mono1/{last,best-checkpoint}.ckpt`
(hints + monotone encoding + counterfactual margin, post-norm injection sites
off). The recipe is
`scratch/hints_v1_runs/20261115-v11/configs/hints_v1_p4_final_mono1.yaml`.

Deep reference material: `AGENTS.md` §4 (wiring) and §6.20–6.22 (footguns),
`hints/README.md` (offline extraction), `scratch/HINTS_V1_PLAN.md` (evidence).

---

## 1. Goal

Give the stage-2 `CodeTransformer` **optional, coarse control knobs** over how a
sentence is spoken — perceptual intensity (emotional / phonation effort, *not*
loudness), register (pitch level), per-word pitch, per-word pacing, and a
sentence-level semantic vector — without touching the stage-1 code space.

Two things this buys:

1. **Inference-time control.** A UI can set "more intensity", "raise this word a
   major third", "make this phrase slower".
2. **Prosody control without a new code space.** Stage 1 owns the code space and
   takes months; steering *how* a fixed voice speaks must not require retraining
   it. A genuinely new timbre still needs stage 1 (`--acoustic_only`).

The constraint that shapes everything: **the codes must not move.** Every hint
is an extra input, so a hint-enabled model is a new parameter set but the same
`spec_codes`/`pitch_codes` — no re-preprocessing, and the existing corpus stays
valid.

---

## 2. Design in one page

Five side channels, all owned by `CodeTransformer`:

| channel | levels | source | applied to |
|---|---|---|---|
| intensity | `n_levels` corpus quantiles | DSP regressor on the 48 kHz wave (calibrated + whisper-gated, 1–9) | encoder + decoder (constant per clip) |
| register | `n_levels` corpus quantiles | voiced-median f0 **minus the speaker's median** (§8.5) | encoder (constant per clip) |
| semantic | 384-d continuous | `bge-small-en-v1.5` CLS, L2-normalised | encoder + decoder (`Linear(384, H)`) |
| word pitch | 8 voiced bins + 1 unvoiced | word's voiced-mean f0, semitones re: the utterance's voiced median | encoder, per phone |
| word timing | 8 log2 buckets | word's start-to-start duration in code frames | encoder, per phone |

Decisions worth remembering:

- **`.cond` stores continuous values; bucketing happens at load.** The number of
  levels is a config change, not a re-run. Only the *edges* live in
  `hints.json`, and `hints.json` is part of the checkpoint contract (a different
  file means different level semantics, so `check_special_layout` refuses a row
  count mismatch).
- **Level 0 is a trained "not provided" state** in every table, so hint-free
  inference is a supported mode rather than a hole. For word pitch the
  *unvoiced* class is the **last** row, not 0 — "this word has no f0" and "no
  hint" are different.
- **Intensity/register use corpus quantiles; pitch/timing use fixed edges.** The
  former are absolute scales (equal-frequency edges stay usable across corpora);
  the latter are relative or physical, and a UI has to be able to say "a major
  third" / "12 frames" and mean it.
- **Register is relative to the speaker** (§8.5); word pitch is relative to the
  utterance's voiced median for the same reason.
- **Dropout per channel** plus `hint_null_all` on training items, because every
  hint is derivable from the target audio — at train time they are an oracle.
- **Clip-level channels need a post-norm injection to be knobs.** `hint_monotone`
  (v1.1) encodes the level as one learned direction times a linear `z(level)`,
  monotone by construction; `train.counterfactual_*` adds the margin that makes
  the level loss-relevant. `hint_film_memory` / `hint_decoder_proj` exist as
  ablations but ship **off** (`AGENTS.md` §4 "Monotone hint encoding").
- The pitch/timing fields need forced-alignment word timestamps, hence a
  separate `--words` pass; the three clip-level hints need no alignment.

---

## 3. Data contract

`hints/build.py` writes a sibling `<base>.cond` next to each
`<base>.phone_ids`, from artefacts already on disk (no source audio, no ASR, no
aligner): `<base>.wave` + `<base>.f0` from `1_preprocess.py`, and the filelist
transcript for the semantic vector.

| key | type | meaning |
|---|---|---|
| `version` | int | 1 |
| `n_phone` | int | `phone_ids.numel()`, the staleness anchor |
| `intensity` | float32(1) | perceptual intensity prediction (1–9), loudness-independent |
| `register_st` | float32(1) | voiced-median f0, semitones re: 100 Hz (**absolute**; the loader subtracts the speaker offset) |
| `semantic` | float16(384) | bge-small-en-v1.5 CLS embedding, L2-normalised |

Word-level `pitch_word` / `phone_word_idx` / `word_start_f` / `word_slot_f` are
added by the **`--words` pass** (it merges into the existing `.cond`). Both
`phone_ids` paths tokenize the words as one sentence, so the stored ids are
exactly the per-item slices and `word_index_for_phones` only has to verify them.

A healthy file is **4–10 KB**. Anything near 4 MB means a tensor shares storage
with a batched array — `write_cond` clones and refuses that (see §5).

---

## 4. Corpus facts

| what | measured |
|---|---|
| parameters, `hints: false` | 135.95 M (bit-identical to pre-v0.8) |
| parameters, `hints: true` | 136.28 M (**+0.24 %**) |
| `.cond` corpus (final) | **151 653 files, 0.74 GB**, avg 4.8 KB, max 9.7 KB |
| intensity / semantic | 151 653 (100 %; 39 394 via the source-audio fallback) |
| register | 112 231 (74 %) — VCTK + Expresso have no stage-1 `.f0` |
| word fields | 147 991 (97.6 %); 3 546 stale `phone_ids` + 116 aligner failures keep clip-level hints only |
| level edges | intensity 8 levels from 151 653 values; register 5 from 112 231 — corpus-wide, not a subset |
| codes ↔ checkpoint | re-encoded 2 clips with `base_v0.7/last.ckpt`: both code tensors **bit-identical** to the corpus |
| speaker skew | top 10 % of speakers = 85 % of clips; 1 474 speakers (≤4 clips) = 2 % |
| stale `phone_ids` rate | 3/200 Genshin clips (1.5 %) — dropped, not guessed |
| tests | `test_hints.py` 59 PASS; `test_fim.py` and `test_phonemize.py` green |

---

## 5. Bugs found while building it (all fixed, all guarded)

1. **Every `.cond` was 116.5 MB → 440 GB → disk full.** `torch.save`
   serialises a tensor's whole storage and the sentence embedding was stored as
   a *row* of the batched matrix. `write_cond` now clones every tensor and
   refuses anything over 4 MB.
2. **A killed run wrote nothing.** Every stage now persists as it goes (audio
   flushes in waves of 2000, per-clip writes in the word pass);
   `--skip-existing` resumes.
3. **A mistyped `--device` failed minutes in.** Validated before any model
   loads.
4. **A resumed run refitted the level edges from the subset it still had to do**,
   silently renumbering every level. Partial runs keep the existing edges and
   print the `--fit-edges-only` remedy.
5. **The unvoiced pitch id collided with the top voiced bin.** `word_pitch_levels`
   emitted `len(edges)+1` (= 8) for unvoiced, which is the last *voiced* bin, so
   row 9 was dead and unvoiced words read as "f0 ≥ +6 st". Now `len(edges)+2`;
   no re-preprocessing.

---

## 6. Shipped state

- `transformer.hints: true`, `hints_json: data/genshin_plus_extras/hints_rel.json`
  (relative-register file), `decoder_hints: true`.
- `transformer.hint_monotone` and `train.counterfactual_*` are implemented and
  used by the v1.1 recipe; the shipped `configs/transformer.yaml` keeps
  `hint_monotone: false` so every older checkpoint stays resumable.
- `scratch/HINTS_V1_PLAN.md` §5.8/§5.9 records the knob results: intensity is a
  monotone dial (+0.086/level), register +0.897 st/level, slot a pace control
  (0.8×–1.9× duration), and the hints cut free-running WER by 1.2–3.0 pp.
- Turning hints on needs a fresh `exp_name` (new parameters) and a
  `--transfer_from` warm start; **no re-preprocessing**.

---

## 7. Remaining work

1. **The register control arm** — the per-channel probe drops each channel; the
   register-specific control (replace it with the *speaker's mean bin* rather
   than dropping it) is not implemented. It is the probe that distinguishes "the
   model reads prosody" from "the model reads a second, coarser speaker id".

M5 (REPL ergonomics) is **done**: `scratch/stage2_repl.py` has per-channel
controls (`e` intensity, `r` register, `j` word pitch, `u` word slot,
`d` semantic), one-shot `--hint_*` overrides, reads a loaded val item's `.cond`
as the `cond` base, and derives the hint/monotone/FiLM layout from the
checkpoint. It also has `--seed` so paired A/B draws are comparable.

### Semantic-only on OOD text (measured 2026-11-15)

The one channel flagged "untested" in `scratch/HINTS_V1_PLAN.md` §5.8.
`hints_v1_p4_final_mono1/last.ckpt`, every other channel at level 0, semantic
from the current sentence (bge-small CLS), `temperature 0.9 / top_k 20`:

| pair | first 5 frames | first 10 | first 20 | first 40 | full |
|---|---|---|---|---|---|
| own vs null | 59.1 % | 46.9 % | 36.4 % | 30.0 % | 24.4 % |
| swapped vs null | 65.6 % | 55.5 % | 42.9 % | 32.5 % | 27.0 % |
| **own vs swapped** | **59.9 %** | **49.3 %** | **36.3 %** | **27.0 %** | **23.0 %** |

8 OOD sentences × 2 seeds; "swapped" injects the *next sentence's* vector.
The channel is **live**: against the null state only ~59 % of the first five
frames agree, i.e. supplying a vector changes the generation immediately. But
**the vector's content is not discriminative at this scale** — swapping in
another sentence's vector is as far from "own" as the null state is (59.9 % vs
59.1 % in the first five frames), and slightly *closer* to the null arm than
own is.

The acoustic arm (4 sentences, decoded through `base_v0.7`) agrees: a semantic
vector raises f0 spread (2.97 → 3.65 st) and leaves duration essentially
unchanged (2.94 → 2.95 s), but own and swapped are indistinguishable
(f0 spread 3.65 vs 3.60 st, median f0 132 vs 136 Hz, 74.2 vs 73.8 frames).
So on OOD text the channel behaves as a trained *"semantic state present"* flag
rather than a sentence-specific conditioner. Caveat: free-running agreement
compounds sampling noise and 8×2 items give a wide interval, so this bounds the
effect rather than proving it zero — the 4-text / 1-seed subset of the same
probe reads the other way round (own-vs-swapped 49.5 % vs own-vs-null 53.5 % in
the first five frames), which is exactly the noise level involved. It is
consistent with §5.5's "semantic alone = the null state" on WER.

Reproduce with `python scratch/hint_semantic_probe.py --checkpoint
checkpoints_stage2/hints_v1_p4_final_mono1/last.ckpt` (`--no_audio` for the
codes-only table, `--num_texts` / `--seeds` to size it).


---

## 8. Background

(Subsection numbers are kept from the pre-2026-11-15 version so existing code
comments that cite `HINTS_PLAN.md §8.5` / `§9` still resolve.)

### 8.5 Register is relative to the speaker

The original register channel was **absolute** (semitones re 100 Hz). Measured
on the Genshin train split (2 705 speakers, 22 821 clips with a finite
register): between-speaker std **6.94 st**, pooled within-speaker **3.52 st**,
**ICC(1) = 0.794**, and a speaker's mean bin predicts the clip's bin **66.7 %**
of the time. So ~79 % of the channel is a speaker property — redundant with
`spk_emb` and, being an oracle, confounded with it, which turns the inference
knob into a voice-swap dial. `val_hint_gap` cannot detect that.

v0.9 conditions on `register_st − speaker median` instead. The corpus was **not**
rewritten: `hints/build.py --relative-register` freezes one semitone offset per
speaker into `hints.json` (`levels.register.speaker_means`) and fits the edges on
the residual (std 3.58 st, edges `[-2.48, -0.98, 0.34, 2.38]` st), and
`dataset_codes` subtracts the offset at load time using the item's raw
pre-`sid_map` id. Presence of `speaker_means` *is* the switch;
`transformer.relative_register` asserts that the config agrees with the file,
since the level ids are identical either way and no shape guard can catch it. A
register level now means "this utterance is N semitones above/below this
speaker's usual level".

---

## 9. The v0.8 regression, and what it changed

v0.8 degraded **15 % → 42 % WER** during its own training on the same 24
utterances (paired seeds, same vocoder), while its teacher-forced `val_loss`
*improved* — the two metrics decoupled. The mechanism was a **repetition
attractor**: deletion-dominated errors and adjacent-frame repetition 0.18–0.20
vs 0.11 for v0.7.

An earlier note in this file claimed the failure was unrelated to the hints
because "regenerating with real `.cond` hints gives identical WER, so the
channels are inert". That measurement was wrong: `eval_stage2_wer.py --hints`
built its dataset without the hint kwargs, so both arms ran the trained null
state (`scratch/HINTS_V1_PLAN.md` §1.2). Once fixed, the channels **help**
(−1.2 pp at 200 utterances, −3.0 pp paired with `--ignore_eos`), and the v0.7
warm-start control reproduces its WER to within one word.

v0.9 therefore changed, cheapest-first:

1. **Relative register** (§8.5).
2. **Zero-initialised hint tables** — a random table is added to every phoneme,
   perturbing a warm-started encoder exactly when its text alignment is most
   fragile.
3. **`fim_rate: 0.5 → 0.2`** — half of v0.8's gradient went to the
   prefix/suffix/middle objective, which teaches "continue from a context", not
   "advance through the text".
4. **Warm start from `best-checkpoint.ckpt`**, not the post-anneal `last.ckpt`.

---

## 10. Hints V2 — per-word pitch anchors

**Status (2026-11-16): implemented, not yet trained.** Switches:
`transformer.hint_pitch_mode: anchors` + `hint_pitch_anchors: 3` (the shipped
`configs/transformer.yaml`), a `.cond` v2 corpus (`hints/build.py --words
--pitch-anchors 3`), and the phonemizer fixes below.

### Why the representation changed

Measured on 300 Genshin clips (`<base>.cond` + stage-1 `<base>.f0`), 1 609 words
of >= 8 code frames (>= 320 ms), residual in semitones against the word's f0
contour:

| N | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| best piecewise-constant fit | 2.12 | 1.18 | 0.72 | 0.49 | 0.32 |
| uniformly spaced samples | 2.12 | 1.58 | 1.22 | 0.94 | — |

- v1 stored **one voiced mean per word** — the worst case. Per-word pitch
  range: median 3.5 st, 6.5 st for long words, p90 11–15 st; 19 % of words are
  within 0.5 st of constant (a genuinely flat word) and 40 % within 1 st.
- Anchors at detected positions beat uniform sampling at equal count (uniform
  N=4 is worse than optimal N=3).
- **But projecting anchors to one value per phone is worth nothing beyond 2**
  (1.72 st at N=2, 1.62 at N=3/N=4) because a 320–440 ms word has ~2.5 phones
  (median 2). Hence the **vector**: every phone of a word carries the word's
  whole anchor set, plus its own position inside the word.

The detector (`hints.levels.detect_pitch_anchors`) is an exact DP over knot
positions (piecewise-linear, endpoints fixed, redundant knots pruned). On the
885-word gate it reconstructs to **0.87 st median** vs **2.09 st** for the word
mean (a greedy splitter reaches only 0.90, hence the DP).

### Contract

`.cond` v2 adds, next to the v1 fields (which are kept):

```
pitch_anchors_st    float16(n_word, A)  semitones re: the utterance voiced median
pitch_anchors_pos   float16(n_word, A)  position as a fraction of the word span
pitch_anchor_n      int8(n_word)        valid anchors; 0 = the word is unvoiced
```

`A = transformer.hint_pitch_anchors` and is part of the checkpoint contract;
`hints.json` records it (`words.pitch_anchors`) and the model refuses a
mismatch. A v1 `.cond` (no anchor keys) is read as **one anchor at pos 0.5**
with the old word mean, so a half-rebuilt corpus still trains.

Per phone the dataset emits `hint_pitch_val (N, A)`, `hint_pitch_pos (N, A)`,
`hint_pitch_n (N)`, `hint_pitch_state (N)` (0 absent / 1 voiced / 2 unvoiced)
and `hint_phone_frac (N)`. The model encodes it as

```
sum_a  mask_a * ( dir_val[a] * tanh(st_a / 6) + dir_pos[a] * (pos_a - 0.5) )
     + (~mask_a) * null[a]
     + dir_frac * (phone_frac - 0.5)
     + where(state == 2, unvoiced, 0) + where(state == 0, absent, 0)
```

so the three trained states stay distinct (the dropout design depends on it)
and a level is continuous: "raise this word a major third" is 3 st, not a bin.
`hint_pitch_mode: table` keeps the v1 `pitch_emb` for old checkpoints; the two
layouts are refused in both directions by `check_special_layout`.

Pitch stays **encoder-only** (the decoder keeps the clip-level channels), and
the timing channel is **unchanged** — the whole V2 hint stack is in practice a
duration plan (`slot` alone recovers the v1 WER gain), so that was left alone
deliberately.

### Phonemizer fixes (same corpus rebuild)

- **Hyphenated compounds** went through g2p_en's character-level neural
  fallback because `TweetTokenizer` keeps `top-notch` as one token and it is
  absent from cmudict: `well-known` -> `WEH1L K AO1 N ER0`,
  `self-esteem` -> `S EH1L F S IY1 T S IY0`. 3.9 % of Genshin clips, 2.1 %
  Expresso, 1.3 % VCTK, 0.8 % MLP contain such a token. `_phonemize_word` now
  tries the whole token against the hand dictionary first, then splits on
  hyphens (`Top-notch` -> `T AA1 P - N AA1 CH`).
- **Typographic apostrophes** are not in g2p_en's `[a-z']` sanitiser, so
  `don’t` became `dont` and missed both the dictionary and cmudict (1.9 % of
  Expresso tokens). Normalised to ASCII before every lookup.
- **`READ/VB`** -> `R IY1 D` (g2p_en's homograph table is inverted, so the
  present tense always lost), **`FOR`** -> `F AO0 R` (it was identical to
  `FOUR`, both `F AO1 R`).
- `scratch/audit_oov.py` lists every token that still reaches the neural
  fallback, per filelist, so the pin list is generated rather than guessed.

`test_phonemize.py --regress` and `test_pitch_anchors.py` guard all of it.

### Corpus rebuild: the aligner path vs the transcript path

Rebuilding the corpus surfaced a second, larger defect than any of the above:
**the two phonemizer paths disagreed on 2.3 % of clips**, and every disagreeing
clip lost its word-level hints (pitch anchors *and* timing). Two independent
causes, both fixed:

- `features.attach_punctuation` now records which whitespace tokens of the
  transcript each aligned item covers (`seg_idx`), and `phonemize_items`
  phonemizes *those* instead of the aligner's stripped ones. The aligner's
  `clean_token` deletes internal punctuation, so `top-notch` arrived as
  `topnotch` -- OOV, neural fallback -- while the transcript path (which wrote
  `phone_ids`) produced `T AA1 P - N AA1 CH`. The old code only used the
  transcript's tokens when the whitespace counts matched exactly, which a single
  standalone `—` / `--` / `...` token breaks.
- `word_index_for_phones`'s safety check counted `text.split()`, so the same
  standalone punctuation token made it reject an alignment that was correct.
  It now counts `features.transcript_words(text)` (word-bearing segments only).

Measured result: `phonemize_text` == `align` + `phonemize_items` == stored
`phone_ids` on all checked clips, word-hint coverage 3 546/151 653 missing ->
**115/151 653 (0.08 %)**, and the coverage gate passes. The residual failures are
clips with no audio/transcript in this environment and tokenizer edge cases
(`*emphasis*` / `"quoted"` phone strings decode with `[UNK]`, which
`_token_spans` refuses to slice).

### Reload cost

`phone_ids` change, so `2_preprocess.py --only phones` (cheap, no codec) and
then one `hints/build.py --words` pass (the old word fields go stale). No
stage-1 re-preprocessing. A new `exp_name` + `--transfer_from` is required.

