# Is predicted intensity correlated with pitch (F0)?

Short answer: **yes within speakers, no across speakers (once corpus/source is
controlled).** The raw across-speaker correlation is a Simpson's paradox driven
by corpus differences, not by pitch.

Model: `checkpoints/intensity_whisper_small_local/best-checkpoint.ckpt`
(whisper-small, epoch 10 / step 1700). Inference path: 2 s windows, 0.5 s hop,
mean of window scores; whole-clip pass as a robustness check.

## Method

* **Pitch**: Praat autocorrelation (`parselmouth`, 10 ms frames, 60-500 Hz),
  median over voiced frames per clip. Also F0 std in semitones (pitch
  variability) and voiced fraction.
* **Data**: `data/v0.5`, train + val filelists, deduplicated by `.pt` path.
  3805 clips / 940 speakers after dropping 283 ambiguous `_NNN` clips (those
  filenames collide across source folders during preprocessing).
* **Speaker**: parsed from the corpus filename (VCTK `pXXX`, MLP
  `HH_MM_SS_Speaker_...`, Genshin `Name_English__id`).
* **Statistics**:
  * pooled Pearson / Spearman over clips;
  * across-speaker = correlation of speaker means (>= 5 clips/speaker);
  * within-speaker = fixed-effects (speaker-demeaned) correlation over clips,
    with speaker-cluster bootstrap CI and within-speaker permutation p-value;
  * partial correlations controlling label, RMS level, duration, and corpus.

## Headline results (3805 clips, 940 speakers)

| quantity | r | 95% CI / notes |
|---|---|---|
| pooled intensity vs median F0 | **+0.380** | [+0.349, +0.412], n=3803 |
| within-speaker (speaker-demeaned) | **+0.312** | [+0.214, +0.412], perm p=0.001 |
| within-speaker, 3 main corpora only | **+0.406** | [+0.350, +0.457], n=3305 |
| within-speaker, held-out val split | **+0.366** | [+0.075, +0.578], n=193 |
| within-speaker, val, main corpora only | **+0.490** | [+0.310, +0.619], n=184 clips after demeaning |
| across-speaker, raw | **+0.343** | [+0.226, +0.458], 276 speakers |
| across-speaker, controlling corpus | **+0.020** | essentially zero |
| across-speaker, controlling corpus + label | **−0.015** | essentially zero |

Per-speaker r for speakers with >= 8 clips: 150 speakers, mean +0.195,
median +0.240, **107/150 positive** (sign test p=1.8e-7).

## Why the across-speaker result is misleading

Speaker-mean F0 and speaker-mean intensity both differ by corpus:

| corpus | speakers | mean F0 (Hz) | mean intensity | mean label | across-speaker r (F0) |
|---|---|---|---|---|---|
| VCTK (read speech) | 96 | 163 | 3.56 | 3.32 | +0.138 (ns) |
| Genshin (acted) | 136 | 211 | 4.43 | 4.20 | −0.044 (ns) |
| MLP (acted/singing) | 39 | 289 | 5.12 | 4.74 | +0.135 (ns) |

Higher-pitched corpora also happen to be scored more intense, which produces
the raw across-speaker r=+0.34. Within each corpus the relation is ~0; partial
r after corpus dummies is +0.02. The same confound is present in the human
labels (across-speaker label~F0 r=+0.238 raw, **+0.034** after corpus control).

## Within-speaker breakdown

Fixed-effects (speaker-demeaned) correlations:

| corpus | n (within) | pitch (F0) | loudness (RMS) | pitch \| RMS+duration | pitch \| label+RMS+duration |
|---|---|---|---|---|---|
| Genshin | 1510 | **+0.472** | +0.188 | +0.449 | +0.320 |
| MLP | 941 | **+0.388** | +0.435 | +0.290 | +0.225 |
| VCTK | 854 | +0.072 | +0.247 | +0.086 | +0.082 |
| all clips | 3390 | **+0.312** | +0.322 | +0.255 | +0.228 |

Pitch variability (F0 std in semitones) is also positive within speaker, but
much weaker in speech than in singing: all clips +0.071, singing +0.284,
non-singing speech +0.050.

## Is it just the label or just loudness?

* pooled intensity~label r=+0.814 (the model does predict the annotation);
* within-speaker label~F0 r=+0.200;
* within-speaker intensity~F0 controlling the label r=+0.245;
* within-speaker intensity~F0 controlling RMS+duration r=+0.255.

So pitch adds information beyond both the (single, noisy) human label and the
raw level, but the effect is modest: r≈0.3 means ~10% of within-speaker
variance, r≈0.4-0.47 in expressive corpora means ~15-22%. The model is mostly a
speaker/corpus identifier across speakers: the between-speaker share of variance
(one-way eta²) is 0.76 (0.88 in the val split), yet those speaker offsets do not
line up with pitch.

The older whisper-base checkpoint (`checkpoints/test8`) on the same val split
gives a weaker, non-significant within-speaker effect (r=+0.209, 95% CI
[−0.166, +0.481], p=0.115), so the current whisper-small model is more
pitch-sensitive, though the CIs overlap.

## Caveats

* Correlations are descriptive, not causal. A pitch-shift manipulation would be
  needed to prove the model *uses* F0 rather than a correlate of F0 (vocal
  effort, tension, brightness).
* Speaker IDs come from filenames, not ground-truth identity; actor/character
  aliases (e.g. "Ashleigh Ball" vs "Applejack") split the same voice.
* The full-set numbers include training clips (in-sample); the val-only rows
  are the clean check, but val has only 3 speakers with >= 8 clips.
* `data/v0.5/_NNN.pt` files collide across source folders and are excluded.
* F0 is a median over voiced frames; singing with background music and
  whispery/noisy clips can produce octave errors.

## Reproduce

```bash
python pitch_correlation.py --ckpt checkpoints/intensity_whisper_small_local/best-checkpoint.ckpt \
    --filelist data/v0.5/train_filelist_linux.txt data/v0.5/val_filelist_linux.txt \
    --out-csv pitch_all.csv

python pitch_correlation.py --from-csv pitch_all.csv     # re-analyse, no GPU
python plot_pitch.py --csv pitch_all.csv --out report/pitch_correlation.png
```
