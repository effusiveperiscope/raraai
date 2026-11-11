# `hints/` — offline conditioning-hint extraction

This directory is a **port** of `/mnt/data/Code/raraai/20_intensity_annotation`
(commit `a83bcb8`) plus a per-clip builder that writes the stage-2 `.cond`
tensors. It exists so the stage-2 transformer can be conditioned on a small set
of coarse, *optional* prosody hints without the *training* path depending on
parselmouth (Praat) or on loading the DSP / embedding models: `2_training.py`
and `dataset_codes.py` only read the tensors it produces.

Plan / status / measurements for the model side live in **`../HINTS_PLAN.md`**;
this file documents the extraction package.

Two ways in:

* `hints/build.py` — offline, bulk, parallel, no source audio needed (see below);
* `FeatureExtractor.extract_hints` / `hint_intensity` / `hint_register` /
  `hint_semantic` / `hint_words` in `features.py` — the same helpers, callable
  on an in-memory waveform, which is the entry point for an inference UI.  Those
  wrappers import this package lazily and produce the identical `.cond` record
  (verified in `test_hints.py`).

```
hints/dsp_common.py     copied verbatim (numpy/os/re only)
hints/dsp_features.py   copied verbatim (librosa + parselmouth)
hints/dsp_intensity.py  copied verbatim (DSPIntensityModel + its fit CLI)
hints/whisper_gate.py   copied verbatim (the pkl embeds a WhisperGate)
hints/dedupe_annotations.py  copied verbatim (needed by the re-fit `--dedupe`)
hints/model_io.py       safe loader + version-skew message
hints/build.py          NEW: per-clip .cond builder
hints/verify_dsp.py     NEW: acceptance test against an upstream features csv
hints/models/dsp_intensity/tier1_augmented.pkl   shipping model (322 KB)
```

Upstream scripts that were *not* ported (annotation fitting, window profiles,
reports, candidate mining, the whisper model) are irrelevant here: the hint is a
*label-free* prediction, so only the feature extractor and the fitted regressor
are needed.

## Two deviations from upstream, both deliberate

### 1. The model was re-fitted under scikit-learn 1.9

The upstream `tier1_augmented.pkl` was written by scikit-learn **1.7.2** and
cannot be unpickled by this repo's environment (1.9.1):

```
ModuleNotFoundError: No module named '_loss'
```

A joblib pickle is bound to the version that wrote it, so the shipped copy was
re-fitted from the upstream feature cache with the same recipe:

```bash
python hints/dsp_intensity.py --features-csv <upstream>/dsp_features_augmented.csv \
    --dedupe --tier 1 --cv group --fit-gate \
    --save hints/models/dsp_intensity/tier1_augmented.pkl
```

It reproduces the published held-out numbers (val, n=492):

| | published (`report/dsp_tier1_augmented_metrics.json`) | this re-fit |
|---|---|---|
| pooled r | 0.772 | **0.772** |
| MAE | 0.588 | **0.585** |
| R² | 0.591 | **0.590** |
| within-speaker r | 0.581 | **0.582** |

`hints/model_io.py` turns the version-skew ImportError into that command instead
of a bare traceback.

### 2. Register uses the **median** voiced f0, not the mean

The register hint (not an upstream feature) is
`12 * log2(median(f0_voiced) / 100 Hz)`. Median for the same reason
`pitch_correlation.py` uses it: a voiced *mean* in Hz is dominated by the highest
frames, so one vibrato excursion or an RMVPE octave error moves it much more than
it moves the perceptual register. On 40 val clips the two agree closely
(std 9.79 vs 9.91 st; no clip with mean/median > 1.3), so this is robustness, not
a change of meaning.


## Levels: which hints get fitted edges, and which do not

`hints.json` carries equal-frequency `edges` for the two **absolute-scale** hints
(`intensity`, `register`). That continuous -> level-id mapping is read by
`dataset_codes.py`, `CodeTransformer` and an inference UI from the *same* file, so
a level id cannot mean two different things.

The **word-pitch** and **word-timing** hints deliberately do *not* use fitted
edges. Word pitch is in semitones relative to the utterance's voiced median
(`DEFAULT_PITCH_EDGES`: +-2 st cuts, i.e. 7 voiced bins, plus one explicit
"unvoiced" bin) and word timing is a log2 bucket of the word's duration in code
frames (`DEFAULT_SLOT_BUCKETS` = 8). Both are defined by their units, and a UI
has to be able to say "raise this word a major third" or "make it 12 frames" and
have that mean the same thing in every corpus -- quantile edges would renumber
the bins per corpus and make a stored level meaningless outside it. See
`hints/levels.py`.

## Why the level counts are *not* baked into the tensors

`.cond` stores **continuous** values; `build.py` only writes bin *edges* into
`hints.json`. The dataset buckets at load time, so changing the number of levels
is a config edit, never a re-run. That matters because the DSP regression output
is compressed — measured on the upstream 120-clip val slice,
`dsp = 2.07 + 0.512 * label` (pred std 0.71 vs label std 0.93; labels 3 to 6 are
squeezed into about 1.5 points). Fixed-width bins on the 1-9 scale would
therefore leave most of the level table unused, which is why the edges are
**equal-frequency quantiles** of the corpus.

## Acceptance evidence (`hints/verify_dsp.py`)

The port changes the audio path in two ways the upstream pipeline did not have:
the stage-1 `.wave` tensors are 48 kHz (so there is a resample), and they were
peak-normalised by `data.audio_norm`. Measured rather than assumed:

* **Resampler choice is immaterial.** For one clip, predicted intensity from the
  upstream 16 kHz features is 4.169; recomputed from the 48 kHz `.wave` it is
  4.128 (`soxr_hq`), 4.151 (`kaiser_best`), 4.182 (`polyphase` / scipy). Spread
  of +-0.03 against a model MAE of ~0.59.
* **Feature values agree** on clips without digital silence: `alpha` 0.002,
  `cpp` 0.004, `log2_f0` 0.001, `voiced_frac` 0.001, `hfr` 0.32 dB, `tilt` 0.12,
  `mfcc` 0.14 (absolute units, n=38).
* **Prediction agreement** (n=60 Genshin clips): on clips with no exact-zero
  samples, mean |delta| = 0.080, max 0.391 -> PASS. Clips containing digital
  silence differ more (mean 0.114, max 0.567); see the next section.

```bash
python hints/verify_dsp.py \
    --csv /mnt/data/Code/raraai/20_intensity_annotation/dsp_features_augmented.csv \
    --wave-dir data/genshin_full_stage1 --n 60
```

### Known soft spot: digital silence and the frame means

`dsp_features` computes per-frame features and then takes *means* over frames,
and a digitally silent frame gets `alpha = 10*log10(0/0)` -> **-120 dB**. Our
source audio contains exact zeros in its pauses, while the upstream cached `.pt`
files do not (the augmented cache has a noise floor, so those frames score ~0 dB).
A clip that is, say, 60 % digital silence therefore gets `alpha_mean` near -68
instead of ~0, and its intensity can move by up to ~0.5.

This is a property of the upstream feature definitions, not of the port -- the
same thing would happen to any clean-audio caller. It affects mostly-silent
clips, which stage 2 already drops by default (`2_preprocess.py`'s
`--min_rms_db`), and the hint is a coarse bucket that is dropped 20 % of the time
anyway. `verify_dsp.py` reports the two groups separately rather than averaging
the problem away.

## Data contract

`build.py` writes `<base>.cond` next to `<base>.phone_ids`:

| key | type | meaning |
|---|---|---|
| `version` | int | 1 |
| `n_phone` | int | `phone_ids.numel()` -- the staleness anchor |
| `intensity` | float32(1) | calibrated + whisper-gated DSP score, 1..9 |
| `register_st` | float32(1) | voiced-median f0, semitones re: 100 Hz |
| `semantic` | float16(384) | bge-small-en-v1.5 CLS embedding, L2-normalised |

`hints.json` (written next to the first `--filelist`, or at `--hints-json`) holds
provenance (`dsp.tier`, `dsp.cal`, gate on/off, semantic model id), coverage, and
the level edges per hint.

**Not produced by the default run:** `pitch_word`, `phone_word_idx`,
`word_start_f`, `word_slot_f`. Word-level timing and word pitch need
forced-alignment word timestamps, so they come from the separate `--words`
pass (see below), which merges into the existing `.cond` and leaves the
clip-level fields alone.

## Commands

```bash
# whole corpus (audio-derived hints), 8 worker processes
python hints/build.py \
    --filelist data/genshin_plus_extras/train_stage2.txt \
               data/genshin_plus_extras/val_stage2.txt \
    --wave-dir data/genshin_full_stage1 data/mlp_stage1 \
    --source-filelist filelists_2/genshin_full_eng_text_linux.txt \
               filelists_2/mlp_linux.txt \
    --workers 8

# add the sentence vector too (downloads the bge weights on first use)
#   --hints intensity,register,semantic
# clips with no stage-1 .wave fall back to their own audio file from
# --source-filelist (on by default; --no-audio-fallback disables)

# provenance only, no level edges
python hints/build.py ... --no-edges

# acceptance test against the upstream feature cache
python hints/verify_dsp.py --csv <upstream>/dsp_features_augmented.csv \
    --wave-dir data/genshin_full_stage1 --n 60
```

The same hints from a live waveform + text (the inference-UI entry point; the
output record is identical to what `build.py` writes, and `test_hints.py`
asserts that):

```python
from features import FeatureExtractor
fe = FeatureExtractor(hp=cfg, device="cuda:0")
rec = fe.extract_hints(wav_48k, "Hey. you don't sound too excited.",
                       intensity=True, register=True, semantic=True,
                       words=False)      # words=True needs the aligner
fe.write_hints("test.wav.cond", rec)     # atomic, same format as build.py
level = fe.hint_level(float(rec["intensity"]),
                      fe.load_hint_levels("hints.json")["intensity"]["edges"])
```

`.cond` files are written atomically (`torch.save` to `.tmp`, then
`os.replace`), so a killed run never leaves a truncated tensor, and
`--skip-existing` reuses a `.cond` only when it already holds every requested
key.

A healthy `.cond` is **4-10 KB** (a few hundred bytes of tensors; the rest is
`torch.save`'s container overhead). `write_cond` enforces that twice: it clones
every tensor and it refuses to write anything over 4 MB. The clone matters
because `torch.save` serialises a tensor's **whole underlying storage**, so
storing a *row* of a batched array writes the entire array -- one clip's
`semantic` taken as a row of the corpus-wide `(n_clips, 384)` matrix made every
`.cond` 116 MB on the 151 k-clip corpus (440 GB for 3 784 clips, which filled the
volume) instead of 8 KB. Measured after the fix: 400 clips = 1.94 MB total, max
7.84 KB, i.e. ~0.74 GB for the full corpus.


## Word-level hints (`--words`): the aligner pass

The three clip-level hints need no alignment. The **word** fields
(`phone_word_idx`, `word_start_f`, `word_slot_f`, `pitch_word_st`,
`pitch_word_voiced`) do, so they are a separate, optional pass:

```bash
python hints/build.py \
    --filelist data/genshin_plus_extras/train_stage2.txt \
    --wave-dir data/genshin_full_stage1 \
    --source-filelist filelists_2/genshin_full_eng_text_linux.txt \
    --hints intensity,register,semantic \
    --words --device cuda:0
# ...and if it stops for any reason, just add --skip-existing to resume:
python hints/build.py ... --words --device cuda:0 --skip-existing
```

- It runs `FeatureExtractor.extract` per clip: the **forced aligner is given the
  transcript**, so no ASR runs. `--source-filelist` is therefore mandatory (and
  its audio is also the intensity fallback).
- **Sequential on one GPU** (`--workers` is ignored, with a printed note); the
  aligner + RMVPE are single models. Measured on a 5060 Ti: **~5 clips/s once
  warm** (the first clip pays ~1.2 s for the caches), so the 151 k-clip
  `data/genshin_plus_extras` corpus is ~9-20 h. Shard by filelist if you want
  several GPUs.
- Needs `<base>.wave` (or the source audio via `--audio-fallback`) because the
  alignment runs on audio.
- **Stale clips are dropped, not guessed.** The word boundaries are the per-item
  slices of that one tokenization (`phonemize_items` attaches `phone_span`), so
  `FeatureExtractor.word_index_for_phones` only has to check that the stored
  `phone_ids` are exactly those slices; if they are not, the clip's ids predate
  the current phonemizer/dictionary and the pass skips it, printing the count
  and the remedy (`2_preprocess.py --only phones`, then re-run). Measured on 200
  Genshin clips: 197 reproduce exactly, 3 differ by 1-2 phones.
- Why this matters: the aligner path and the transcript path used to tokenize at
  different granularities (per word vs the whole sentence), and the g2pen
  tokenizer merges phones across word boundaries -- a sentence tokenizes to 44
  tokens where the same words tokenized individually give 71, and incremental
  prefix counts are not even monotonic (43 -> 42 when one word merges into the
  previous token). `phonemize_items` now tokenizes once and slices, so both
  paths produce the same `phone_ids` for the same transcript.
- **Writes are incremental, at every stage.** The audio stage flushes in waves
  of `FLUSH_EVERY` (2000) clips, the embedding stage flushes the clip-level
  records before the word pass, and the word loop flushes each clip as soon as
  its words are done — so an interruption costs minutes, not the run.
  Re-running the same command with `--skip-existing` only redoes clips whose
  `phone_word_idx` is missing. Measured: killing a 200-clip run at 139 left 139
  valid `.cond` files and the resume finished the rest; with a 100-clip wave the
  files appear while the audio stage is still running. `coverage.clips_written`
  counts files, `coverage.writes` counts writes (a clip can be written twice).
- `--device` is validated before any model loads (`cpu` / `cuda[:N]`), so a
  typo like `cuda0` or `cuda:1` on a one-GPU box exits in a millisecond with
  the valid range instead of failing part-way through a multi-hour run.
- A **partial** run never refits the level edges: with `--limit` or
  `--skip-existing` (some clips reused), the existing `hints.json` edges are
  kept and a note says so. Equal-frequency edges are corpus-wide quantiles, so
  refitting them on the few clips a resumed run still had to do would renumber
  every level for every clip. Use `--fit-edges-only` to refit them from every
  `.cond` on disk.
- `--force` (with `--skip-existing`) redoes the word fields for clips that
  already have them; a `--words`-only re-run merges into the existing `.cond`
  instead of replacing it, so the clip-level hints survive.
- `hints.json` gains a `words` block (aligner id, `code_fps`, the pitch edges and
  slot-bucket count) plus `coverage.with_words` / `word_phone_mismatch` /
  `word_failures`.

Slots are start-to-start in **code frames** (25 Hz), so they sum to the clip
length; `pitch_word_st` is each word's voiced-mean f0 in semitones relative to
the utterance's median, NaN (plus `pitch_word_voiced=False`) when the word has no
voiced frames.

