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

