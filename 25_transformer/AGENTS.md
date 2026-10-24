# Agents Guide

This file is a guide for AI coding agents (and future humans) working in this
repository. It explains what the project does, how the pipeline is wired
together, and the conventions and footguns you must respect before making
changes.

---

## 1. What this project is

A **two-stage, neural text-to-speech (TTS) system**. Audio is compressed into
discrete RVQ (residual vector quantisation) codec codes, then a second model
learns to predict those codes from text (phonemes) plus a speaker id.

- **Stage 1** — a VITS/GAN-style **`Generator`** (`modeling/generator.py`)
  trained as a GAN (with MPD/MRD and pitch discriminators) to reconstruct
  audio. It owns the frozen speaker-embedding table and two RVQ codecs: a
  **spectrogram codec** (`codec`, 8 codebooks) and a **pitch codec**
  (`pitch_codec`, 2 codebooks) over the f0 contour. This is where actual
  synthesis happens.
- **Stage 2** — a **`CodeTransformer`** (`modeling/transformer.py`) that
  predicts the spectrogram + pitch codes autoregressively from phoneme tokens
  and a speaker id, using a MusicGen-style **delay pattern** across codebooks.

The scripts are numbered accordingly: `1_*` = stage 1, `2_*` = stage 2.

---

## 2. Repository layout

```
1_preprocess.py      Stage-1 feature extraction -> data/base/{train,val}.txt
1_training.py        Stage-1 GAN training (Generator + Discriminators)
2_preprocess.py      Stage-2 code extraction (frozen G -> codes) -> train_stage2.txt
2_training.py        Stage-2 transformer training
commons.py           Shared helpers (timers, tensor checks, checkpoint loaders)
utils.py             Small helpers (count_parameters, win_longpath)
features.py          FeatureExtractor (Qwen3-ASR forced alignment, RMVPE pitch,
                     HorsePhonemizer, g2pen tokenizer)
dataset_ac.py        Stage-1 dataset (audio/spec/f0)
dataset_codes.py     Stage-2 dataset + collator (phone/spec/pitch codes)
speaker_map.py       Speaker-id canonicalisation (stage-2 id -> stage-1 id)
horsephonemizer.py   G2P phonemiser
ultimate_xc.py       (imported for side effects; see footguns)

configs/
  base.yaml          Stage-1 hyperparameters (Generator, codecs, discriminators)
  transformer.yaml   Stage-2 hyperparameters (CodeTransformer, decoding)

modeling/
  generator.py       Generator (codec + pitch codec + NSF decoder)
  transformer.py     CodeTransformer (PhonemeEncoder + DelayCodebookDecoder)
  vevo_repcodec.py   RVQ codec implementation
  pitch.py, intensity.py, pitch_discriminator.py
  rvc/               NSF decoder / f0 predictor (vendored RVC)
  vits/              Vendored VITS modules (models, attentions, discriminator)
  vits_decoder/      Generator + discriminator building blocks (mpd/mrd/msd/nsf)

vits_extend/         STFT / multi-resolution STFT losses
filelists_1/         Stage-1 source filelists
filelists_2/         Stage-2 source filelists
data/                Preprocessed features + generated train/val filelists
checkpoints/         Per-experiment model checkpoints (gitignored)
logs/                TensorBoard logs (gitignored)
```

`modeling/{vits,rvc,vits_decoder}` and `vits_extend/` are **vendored /
third-party** code — do not restyle or "fix" them unless a bug genuinely
requires it.

---

## 3. End-to-end pipeline

Each stage is `preprocess` → `train`.

### Stage 1

```bash
# 1. Extract features (wave/spec/f0) and split into train/val filelists
python 1_preprocess.py \
    --filelist filelists_1/base_linux.txt \
    --output_dir data/base \
    --val_fraction 0.05

# 2. Train the GAN
python 1_training.py --config configs/base.yaml
```

`1_preprocess.py` writes per-file `.wave`, `.spec`, `.f0` tensors into
`output_dir` and emits `output_dir/train.txt` / `val.txt` lines of the form:

```
<base>.wave|<base>.spec|<base>.f0|sid
```

### Stage 2

Stage 2 depends on a trained stage-1 checkpoint, which is used (frozen) as an
encoder to produce discrete codes.

```bash
# 1. (Optional but recommended) build a speaker map
python speaker_map.py \
    --filelist filelists_2/25_stage2_base1.txt \
    --stage1-filelist filelists_1/base_linux.txt \
    --out speakers.json

# 2. Extract phone_ids + spec_codes + pitch_codes with the frozen Generator
python 2_preprocess.py \
    --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 \
    --checkpoint checkpoints/base_v0.4/last.ckpt \
    --config configs/base.yaml \
    --speaker_map speakers.json

# 3. Train the code transformer
python 2_training.py --config configs/transformer.yaml
```

`2_preprocess.py` writes per-file `.phone_ids`, `.spec_codes`, `.pitch_codes`
and emits `train_stage2.txt` / `val_stage2.txt` lines:

```
<base>.phone_ids|<base>.spec_codes|<base>.pitch_codes|sid
```

Tensor shapes (see `dataset_codes.py` docstring):

- `phone_ids`   `(n_phone,)`               long
- `spec_codes`  `(Q_spec, T_codes)`        long (Q_spec = `codec.codebook_num`)
- `pitch_codes` `(Q_pitch, T_codes)`       long (Q_pitch = `pitch_codec.codebook_num`)

---

## 4. Configuration

Configs are YAML loaded via `omegaconf.OmegaConf`. Hyperparameters are accessed
with **dotted attribute access** (`hp.data.hop_length`) *and* **dict-style**
access (`config.train.get("max_frames", None)`). Both patterns are used — keep
this in mind for new code; the `.get` style is preferred for optional keys.

Two config files exist with **non-overlapping responsibilities**:

- `configs/base.yaml` — stage-1 model + shared `data`/`codec`/`pitch_codec`
  blocks.
- `configs/transformer.yaml` — stage-2 transformer + a copy of the `codec`,
  `pitch_codec`, `data`, `vits` and `gen` blocks needed for validation-time
  audio sampling.

> **Critical invariant:** the `codec` and `pitch_codec` blocks in
> `transformer.yaml` **must exactly match** the stage-1 `Generator` checkpoint
> that produced the codes. Changing these numbers requires re-running
> `2_preprocess.py`.

`exp_name` drives checkpoint and TensorBoard paths:
`checkpoints/{exp_name}/` and `logs/{exp_name}/`.

---

## 5. Conventions

- **Python version/style:** modern Python (3.10+), type hints on new code,
  `from __future__ import annotations`. No formatter is enforced, but follow
  the surrounding file's style. Newer modules (`dataset_codes.py`,
  `transformer.py`, `speaker_map.py`) use Google-style docstrings; older ones
  are less consistent.
- **Filelists** are newline-delimited, `|`-separated. Speaker id is always the
  last field. Preprocessing **skips empty lines and any line containing the
  substring `longform`**.
- **Windows long paths:** audio paths may come from Windows. Always route
  paths through `utils.win_longpath()` before `os.path.exists` / `load`.
- **Checkpoint loading:** use helpers in `commons.py`
  (`load_state_dict_mismatch`, `load_submodule_prefix`). Lightning checkpoints
  nest state under `state_dict`, and module params may be prefixed (`net_g.`,
  `net_d.`, `net_dp.`, `net.`). Always load full checkpoints with
  `weights_only=False`.
- **Devices:** default to `cuda:0` when available, else `cpu`. Stage-1 and
  stage-2 training run with PyTorch Lightning `accelerator='gpu'`,
  `precision='bf16-mixed'`.
- **`mp.set_start_method('spawn', force=True)`** appears at the top of both
  training `__main__` blocks and is required on Linux; don't remove it.
- **Logging:** `logs/` holds TensorBoard runs (`train.log_interval` controls
  step logging). Validation audio for stage 2 is optionally logged by loading a
  frozen stage-1 Generator (`train.generator_checkpoint`); it is non-fatal if
  the checkpoint is missing.

---

## 6. Footguns & gotchas

1. **Speaker-id spaces differ between stages.** The stage-1 `Generator` owns a
   frozen speaker-embedding table, so the *only* id that synthesizes the right
   voice is a **stage-1 id**. Stage-2 filelists may use unrelated ids.
   Canonicalise with `speaker_map.py`:
   - `2_preprocess.py` applies path **rules** (`--speaker_map`).
   - `dataset_codes.py` applies the **`sid_map`** fallback (no audio path
     exists in preprocessed filelists).
   - Both honour `unmapped: "skip" | "keep" | <int>`. Read the `speaker_map.py`
     docstrings before touching this.
   - `2_training.py` prints a warning if any speaker id is out of range of
     `transformer.n_speakers`.
2. **`1_training.py` sets
   `warnings.filterwarnings('error', category=RuntimeWarning)`.** A benign
   RuntimeWarning (e.g. from numpy/pyworld) becomes a hard crash in stage 1.
   Don't casually add code that emits RuntimeWarnings there.
3. **`ultimate_xc` is imported for side effects** in both training scripts
   (`import ultimate_xc`) and never used by name. Do not "clean up" the
   apparently-unused import.
4. **Vendored code:** `modeling/vits`, `modeling/rvc`, `modeling/vits_decoder`,
   `vits_extend/` are third-party. They contain their own duplicate
   `commons.py`/`utils.py` — be careful *which* `commons` an import resolves to
   (`import commons` = repo root; `from modeling.vits import commons` =
   vendored).
5. **`skip_if_one_exists` quirk:** in both `*_preprocess.py`, the "already
   exists" check returns from the *whole loop* (not just the current file) on
   first hit. Keep this in mind if you rely on it for incremental runs.
6. **Features require heavy downloads** on first use: `Qwen3-ASR-0.6B`,
   `Qwen3-ForcedAligner-0.6B`, RMVPE weights, `therealvul/g2pen_tokenizer_clean`.
   `FeatureExtractor.__init__` accepts `load_qwen=True` default; code that only
   needs spec/pitch can avoid loading Qwen when not needed.
7. **Code rates:** spec and pitch codes must have identical lengths per sample
   (both are produced at the decoder frame rate / `down_factor`). The Generator
   asserts this in `infer_from_codes`.
8. **`torch.load(..., weights_only=True)`** is used in `dataset_codes.py`, but
   `weights_only=False` is required for full checkpoints (see §5).

---

## 7. Useful commands

```bash
# Stage-1 preprocessing / training
python 1_preprocess.py --filelist filelists_1/base_linux.txt --output_dir data/base
python 1_training.py --config configs/base.yaml            # auto-resumes last.ckpt

# Resume / transfer for stage 1
python 1_training.py --config configs/base.yaml --resume_from checkpoints/base_v0.4.2/last.ckpt
python 1_training.py --config configs/base.yaml --transfer_from some.ckpt   # partial weight transfer

# Speaker map bootstrap
python speaker_map.py --filelist filelists_2/25_stage2_base1.txt \
    --stage1-filelist filelists_1/base_linux.txt --out speakers.json

# Stage-2 preprocessing / training
python 2_preprocess.py --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 --checkpoint checkpoints/base_v0.4/last.ckpt \
    --config configs/base.yaml --speaker_map speakers.json
python 2_training.py --config configs/transformer.yaml

# Ad-hoc smoke tests (many modules have an `if __name__ == "__main__"` block)
python modeling/transformer.py    # constructs a CodeTransformer, runs fwd + generate
python modeling/generator.py      # constructs a Generator, runs infer
```

There is no test suite in this repo; validation is done by running the ad-hoc
`__main__` blocks in `modeling/*.py` and by training on a small filelist.