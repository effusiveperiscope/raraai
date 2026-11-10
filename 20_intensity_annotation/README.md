Trying to predict intensity annotations from whisper-base features with data augmentation
Intensity annotations are from me (my opinion), integer score 1-9.

The target is *perceptual* intensity (timbre / voice quality / emotional
intensity), deliberately independent of loudness: audio is peak-normalised
before the encoder sees it.

## What this model does and does not give you

* It predicts a **clip / window score** on the 1-9 scale. That is the calibrated
  output (trained against the annotations).
* It does **not** give a trustworthy score *within* a clip at frame resolution.
  The per-frame `intensity` head and the `attn` weights are only constrained
  through their attention-weighted mean (`sum_t attn_t * intensity_t`), so their
  time-shape is unconstrained. Measured on val data, silencing the most-attended
  25% of a clip changes the prediction by the same amount as silencing a random
  25% -- i.e. attention is not a contribution map. Run `attribution_test.py` if
  you want to re-check that after changing the model/loss.
* The honest way to get an intensity *profile over time* is **sliding-window
  scoring**: score overlapping 0.5-4 s windows, each point is an independent
  forward pass against a target the model was trained on.

## Commands

```bash
# train (length-robust, multi-scale supervision)
python train.py --config config/base.yaml [--resume_from checkpoints/<exp>/last.ckpt]

# intensity profile over time for one file (plot + per-window scores)
python eval_windows.py some.wav --ckpt checkpoints/<exp>/last.ckpt \
    --window 2.0 --hop 0.5 --out profile.png --json profile.json

# diagnostic plot: window profile + per-frame head + attention
python eval.py some.wav --ckpt checkpoints/<exp>/last.ckpt --out diag.png

# is the attention head a faithful "how much does this part contribute"? (exit 1 = no)
python attribution_test.py --filelist data/v0.5/val_filelist.txt \
    --ckpt checkpoints/<exp>/last.ckpt --n 48

# clip/window metrics on a held-out filelist (r, MAE, R^2)
python eval_val.py --filelist data/v0.5/val_filelist.txt --ckpt checkpoints/<exp>/last.ckpt --n 200

# self-contained HTML page: audio + profile + attention for a set of samples
python make_report.py --ckpt checkpoints/<exp>/best-checkpoint.ckpt \
    --compare checkpoints/test8/last.ckpt \
    --audio-dir eval_sets/eval1 --filelist data/v0.5/val_filelist.txt --n 8 \
    --out report/index.html     # then: python -m http.server -d report 8080

# mine the big unlabeled pool for standout clips (extremes, dynamics, model
# disagreement, ...) and render them; clips already in data/v0.5 are excluded
python mine_samples.py --ckpt checkpoints/<exp>/best-checkpoint.ckpt \
    --compare checkpoints/test8/last.ckpt \
    --limit 900 --per-category 3 --out report/mined.html
```

`in_channels` is read from the checkpoint and the matching whisper model is
picked automatically (`--whisper_model` to override), so a whisper-base
checkpoint can't silently be loaded into a whisper-small model.

## DSP-only alternative (CPU, no whisper)

`dsp_features.py` / `dsp_intensity.py` implement a cheap, interpretable pipeline
that predicts the same 1-9 scale from hand-crafted acoustic features. It is
useful when you want:

* **a time profile** without a forward pass per window (frame features are
  computed once and aggregated per window),
* **interpretability** (every score decomposes into phonation and spectral
  cues),
* **CPU-only inference** (~20-25 ms/clip), and
* **correct whisper handling**: whisper is a qualitative phonation mode, not a
  low point on a continuous spectral axis. A two-stage gate forces true
  whispers to 1 instead of letting the regressor smear them toward the mean.

The feature set is organised around the two axes that matter:

| tier | features | cost | what it captures |
|---|---|---|---|
| 0 | CPP, log F0, F0 std, voiced fraction, contrast2, MFCC2 | ~8 ms | phonation / voicing (whisper = no voicing) |
| 1 | tier 0 + tilt, alpha ratio, Hammarberg, HFR, MFCC3, contrast3, flatness, ZCR, RMS std, flux | ~20 ms | + spectral effort / brightness |
| 2 | tier 1 + CPP std, HNR, F0 IQR, centroid, rolloff, MFCC1/4, bandwidth, duration | ~30 ms | + a few shape features |

Everything comes from **one STFT** plus **one Praat pitch pass**; Praat HNR is
replaced by cepstral CPP because it costs ~30 ms/clip on its own. Loudness is
removed by peak-normalising before feature extraction; the raw level is used
only by the whisper gate (whisper is ~6 dB quieter than speech).

```bash
# IMPORTANT: use the fixed cache. The original data/v0.5 MareFairData rows
# are broken (filename collisions + 3x slowdown); see the repair section below.
python dsp_features.py \
    --filelist data/v0.5_fixed/train_filelist_linux.txt data/v0.5_fixed/val_filelist_linux.txt \
    --split train val --tier 1 --out-csv dsp_features_fixed.csv --workers 4

# fit + evaluate the gate (speaker-held-out AUC / precision / recall)
python whisper_gate.py --features-csv dsp_features_fixed.csv \
    --out checkpoints/dsp_intensity/whisper_gate_fixed.pkl

# speaker-grouped CV + fit on train + evaluate val + save
python dsp_intensity.py --features-csv dsp_features_fixed.csv --dedupe --tier 1 \
    --cv group --fit-gate \
    --save checkpoints/dsp_intensity/tier1_fixed.pkl \
    --report report/dsp_tier1_fixed_metrics.json

# profile one file (same output shape as eval_windows.py)
python dsp_profile.py some.wav --model checkpoints/dsp_intensity/tier1_fixed.pkl \
    --window 2.0 --hop 0.5 --out profile.png --json profile.json

# head-to-head against the whisper model on the same clips
python compare_dsp_model.py --features-csv dsp_features_fixed.csv \
    --dsp-model checkpoints/dsp_intensity/tier1_fixed.pkl \
    --ckpt checkpoints/intensity_whisper_small_local/best-checkpoint.ckpt \
    --eval-splits val --n 120

# self-contained HTML: audio + profile + phonation cues, across the 1-9 range
python make_dsp_report.py --features-csv dsp_features_fixed.csv --dedupe \
    --model checkpoints/dsp_intensity/tier1_fixed.pkl \
    --eval-splits val --per-bin 2 --out report/dsp_intensity.html
# then: python -m http.server -d report 8080
```

Measured on the **fixed** v0.5 set (tier 1, HGB, whisper gate at p>=0.8 and
`rms_dbfs < -21`):

| model | pooled r | MAE | within-speaker r |
|---|---|---|---|
| whisper-small, val whole clip (n=440) | 0.755 | 0.569 | 0.578 |
| whisper-small, val window mean | 0.824 | 0.503 | — |
| **DSP tier 1, fixed val (n=449)** | **0.708** | **0.575** | **0.536** |
| DSP tier 1, fixed group CV (n=4066) | 0.696 | 0.598 | 0.498 |

The DSP model is now essentially at the whisper model's level on MAE and
within-speaker ranking (0.536 vs 0.578), at ~20 ms/clip on CPU. The earlier
0.43/0.50 numbers were computed on the broken MareFairData cache; the repair
below recovers the missing 428 clips and turns the 3x-slowed audio into normal
speech, which is why the within-speaker correlation jumps.

The whisper gate costs a little pooled r on random samples (it caps a few quiet
clips) but it is the only way to honour `whisper = 1`: on the 5 val whisper
clips it takes MAE from ~1.7 (whisper model) to ~0.10.

The whisper rows above are from the original v0.5 split. Retrain the whisper
model on `data/v0.5_fixed/*_filelist_linux.txt` for a strict apples-to-apples
comparison; the DSP numbers use the fixed filelists.

Use DSP when you need the profile or interpretability; keep whisper when you
need the last bit of within-speaker ranking accuracy. `make_dsp_report.py`
renders a self-contained page with audio, profiles, and phonation cues so you
can calibrate your ear against the labels.





## Data repair: MareFairData preprocessing bug

The original v0.5 preprocessing had two independent bugs for MareFairData:

1. **Filename collision.** `preprocess.py` saved `Path(audio_path).stem + ".pt"`
   into one flat directory. Every `MareFairData/<character>/_046.flac` wrote the
   same `_046.pt`, so the last character processed won. 711 annotations collapsed
   to 283 `.pt` files: **428 clips were overwritten**, and 189 basenames carried
   labels from different characters.
2. **Missing resample.** The surviving `.pt` files kept the 48 kHz source frame
   count but were read at 16 kHz, so they play **3× too slow**. Before the fix:
   median F0 89 Hz, median duration 17.3 s. After: **210 Hz, 5.4 s**. Genshin,
   VCTK and expresso all have `source_frames / pt_frames = 3.0` (correct);
   MareFairData had `1.0`.

Repair (the original `data/v0.5` tree is not modified):

```bash
python fix_marefair.py --annotation-json annotations/v0.5.json \
    --out-dir data/v0.5_fixed/marefair \
    --map-out report/marefair_fixed_map.csv \
    --filelist-out data/v0.5_fixed/marefair_filelist.txt

python build_fixed_features.py --base-csv dsp_features.csv \
    --marefair-map report/marefair_fixed_map.csv \
    --out-csv dsp_features_fixed.csv \
    --out-dir data/v0.5_fixed --workers 4
```

Result: **710 unique fixed clips** (one source file is missing), and
`dsp_features_fixed.csv` has **4,515 rows / 4,515 unique paths**, with
**0 duplicates, 0 conflicting labels, 0 train/val leaks**. The fixed audit is
`report/dedupe_fixed.csv`; `report/label_conflicts_fixed.csv` is empty.

The old `report/*_broken.html/csv` artifacts are kept for reference. **Do not
re-label `label_conflicts_broken.html`**: its audio is one arbitrary
last-writer waveform with several characters' labels attached to it. The
"73 conflicts" were entirely a preprocessing artifact, not annotation noise.

## Mining high-intensity candidates (fixed data)

The DSP **regressor** still compresses the top end, so the miner is a pair of
binary classifiers trained on the fixed labels with speaker-grouped OOF
predictions:

* `clf6 = P(label >= 6)` — AUC 0.845
* `clf7 = P(label >= 7)` — AUC 0.832

Ranking by `rank(clf6) + rank(clf7) + rank(rms_dbfs)`:

| positive | base | P@25 | P@50 | P@100 | P@200 |
|---|---|---|---|---|---|
| label ≥ 7 | 1.9% | 48% | 38% | 24% | 16% |
| label ≥ 6 | 13.2% | 88% | 82% | 68% | 55% |

```bash
python mine_high_intensity.py --features-csv dsp_features_fixed.csv \
    --out-csv report/high_intensity_candidates_fixed.csv \
    --all-out report/high_intensity_all_fixed.csv
```

The listening queue has **93 low/mid-labeled candidates**:
**14 `A_low_clf7`** (`label<=4 & clf7>=0.7`), **40 `C_borderline5`**
(`label==5 & clf6>=0.9`), and **39 `B_low_clf6`**
(`label<=4 & clf6>=0.8`). VCTK and `other` are excluded (VCTK labels top out at
5 and the classifier over-scores it); whisper/`Whispering` clips are excluded by
default because breathiness can look like effort. Use `--keep-whisper` to
include them.

### Interactive re-labeling

```bash
python make_label_report.py \
    --csv report/high_intensity_candidates_fixed.csv \
    --title "High-intensity candidates (fixed v0.5)" \
    --model checkpoints/dsp_intensity/tier1_fixed.pkl \
    --out report/high_intensity_candidates.html
```

The generated `report/high_intensity_candidates.html` contains 93 candidates
(14 A, 40 C, 39 B). Click **1-9** on each card; selections are stored in the
browser (`localStorage`, keyed by path) and **Export labels CSV** downloads
`path,new_label,note,current_labels,suggested,corpus,speaker,reason`. Rows you
skip are included with a blank `new_label`, so you can see what is left. Serve
it over `python -m http.server -d report 8080` if your browser restricts
`localStorage` on `file://`.

After re-labeling, merge the exported labels, deduplicate again, and retrain
`dsp_intensity.py --features-csv dsp_features_fixed.csv --dedupe`. The high-end
MAE should improve as labels 7-9 become less rare; if it does not, the next step
is an ordinal/quantile model rather than more features.


## Training notes (config/base.yaml)

* `crop_min_sec` / `crop_max_sec` (0.5 / 4.0): every step sees a random-length
  crop, so the model is calibrated for the sliding windows used at inference.
  (Previously everything was cropped to a fixed 2 s, which is why whole-clip
  predictions degraded badly.)
* `num_windows` / `window_loss_weight` / `window_consistency_weight`: the same
  prediction is also supervised on random sub-windows of the crop, and pulled
  towards the whole-crop prediction, so that
  `clip score ~= mean of window scores`.
* `val_augment: false`: validation runs on clean audio; `data.val_crop_sec: 30`
  means validation sees whole clips (deterministically from index 0).
* Watch `val_pred_mean` in tensorboard: if it collapses towards the dataset
  mean, the model is regressing to the mean (label noise is large - the label
  std is ~1.0 on a 1-9 scale, and per-speaker means differ).

## Intensity vs pitch

`pitch_correlation.py` scores clips, extracts Praat F0, and reports pooled,
within-speaker and across-speaker correlations (plus label/loudness controls).
`PITCH_RESULTS.md` has the measured numbers: the model is weakly pitch-sensitive
**within** a speaker (r ~ 0.31; ~0.41 in the expressive corpora, ~0.49 on the
held-out val clips) but the apparent **across**-speaker correlation (r ~ 0.34)
is a corpus confound and drops to ~0.02 once corpus is controlled.


`python train.py --config config/local.yaml --max_epochs 15` (whisper-small,
4063 train / 450 val clips, RTX 5060 Ti, ~18 min; best checkpoint at epoch 10),
evaluated with `eval_val.py --n 120` on held-out val clips, 2 s windows / 0.5 s hop:

| metric | old `checkpoints/test8` (whisper-base, fixed 2 s crops) | this run (whisper-small, length-robust + multi-scale) |
|---|---|---|
| whole-clip pass r / MAE / R^2 | 0.686 / 0.706 / 0.323 | **0.755 / 0.569 / 0.566** |
| mean of window scores r / MAE / R^2 | 0.811 / 0.519 / 0.638 | **0.824 / 0.503 / 0.653** |
| per-window vs clip label r | 0.590 | 0.563 |
| within-clip std of window scores | 0.198 | 0.186 (label std 1.09) |

So the length-robust training fixed exactly the pathology it targeted: whole-clip
predictions no longer collapse (R^2 0.32 -> 0.57), while the 2 s window path stayed
as good or slightly better. Two honest caveats:

* **Attention is still not faithful after this run.** `attribution_test.py --n 48`
  gives most-attended span -0.03 +- 0.27 vs random -0.08 +- 0.21, top>random in 54%
  of clips (chance 50%). Multi-scale *pooled* supervision cannot fix per-frame
  attribution - only frame-aligned targets can (mosaic piecewise-constant labels
  or per-region annotations). Use the window profile, not the attention.
* **Within-clip contrast is small**: window scores vary with std ~0.19 while labels
  vary with std ~1.09. Most of the usable signal is the (near-constant) clip-level
  score. `window_consistency_weight` explicitly pulls each window towards the crop
  prediction, so lower it (or set `0`) if you care more about profile contrast than
  about clip score == mean of window scores.

