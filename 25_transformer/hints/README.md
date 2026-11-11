# `hints/` — offline conditioning-hint extraction

This directory is a **port** of `/mnt/data/Code/raraai/20_intensity_annotation`
(commit `a83bcb8`) plus a per-clip builder that writes the stage-2 `.cond`
tensors. It exists so the stage-2 transformer can be conditioned on a small set
of coarse, *optional* prosody hints without making the training/inference
environment depend on scikit-learn, Praat or an embedding model: nothing in
`2_training.py`, `dataset_codes.py` or the inference path imports this package —
they read the tensors it produces.

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

**Not produced here:** `pitch_word`, `phone_word_idx`, `word_start_f`,
`word_slot_f`. Word-level timing and word pitch need forced-alignment word
timestamps (`features.align`); the `--phones_from transcript` path has no word
boundaries at all. That is a separate pass and is intentionally not run by the
tooling here.

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

# provenance only, no level edges
python hints/build.py ... --no-edges

# acceptance test against the upstream feature cache
python hints/verify_dsp.py --csv <upstream>/dsp_features_augmented.csv \
    --wave-dir data/genshin_full_stage1 --n 60
```

`.cond` files are written atomically (`torch.save` to `.tmp`, then
`os.replace`), so a killed run never leaves a truncated tensor, and
`--skip-existing` reuses a `.cond` only when it already holds every requested
key.

