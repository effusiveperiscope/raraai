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
  (`pitch_codec`, 2 codebooks) over the f0 contour. The pitch codec also owns
  a small **voicing (V/UV) head** (`pitch_uv_proj`) so its codes carry the
  voiced/unvoiced decision. This is where actual synthesis happens.
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
docker/              vast.ai image helpers (prefetch_models.py, fetch_shards.sh,
                     boot/), see DOCKER.md
data/                Preprocessed features + generated train/val filelists
checkpoints_stage1/  Stage-1 checkpoints, per experiment (gitignored)
logs_stage1/         Stage-1 TensorBoard runs (gitignored)
checkpoints_stage2/  Stage-2 checkpoints, per experiment (gitignored)
logs_stage2/         Stage-2 TensorBoard runs (gitignored)
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

Two extra flags make re-runs cheap and drop clips that cannot be trained on:

- `--skip_existing` — reuse tensors already in `output_dir` (no audio load, no
  feature extraction, and `FeatureExtractor` is built lazily, i.e. only if some
  file still needs extracting) and just re-validate them.
- `--min_frames N` — omit clips whose `.spec` has fewer than `N` frames from
  `train.txt`/`val.txt` (default `data.segment_size // data.hop_length` = 16,
  i.e. one training segment; `0` disables it). Too-short clips are still
  written to disk, so later `--skip_existing` runs do not re-extract them.
  The train/val split is computed over the *kept* files.

```bash
# Rebuild the genshin filelists without re-extracting anything, dropping clips
# too short for a training segment
python 1_preprocess.py --filelist filelists_1/genshin_full_eng_linux.txt \
    --output_dir data/genshin_full_stage1 --skip_existing
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
    --checkpoint checkpoints_stage1/base_v0.4/last.ckpt \
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

**Silent / near-silent clips are dropped by default.** Before extracting
anything, `2_preprocess.py` runs a level check on the loaded 48 kHz audio
(`audio_level_stats` / `is_silent` in the script): a clip is skipped when the
RMS of its loudest 20 ms frame is below `--min_rms_db` (default `-50` dBFS), or
— only when `--min_speech_frac > 0` — when fewer than that fraction of its
20 ms frames reach `--min_rms_db` (drops clips that are almost all silence).
Dropped clips are not written to disk and their reason (loudest frame / peak /
active-frame fraction, all dBFS) is printed. Notes:

- `--min_rms_db none` disables the filter; raise it (e.g. `-35`) to also drop
  very quiet recordings. Nothing about the filter is stored on disk, so
  changing the thresholds only needs a re-run — with `--skip_existing` the
  level check still runs (audio is loaded, no features are re-extracted), so
  silent clips can be dropped from an already-encoded corpus, but clips that
  pass are reused as-is.
- `train_stage2.txt` / `val_stage2.txt` are split over the **kept** clips, so
  dropping clips never shifts items out of train into val. If *nothing*
  survives, the script raises instead of writing empty filelists.
- `--skip_existing` reuses tensors already in `--output_dir` instead of
  re-running the ASR/aligner + codec. Reuse is per file *and per tensor*, and
  each tensor is header-checked on its own (`key_shape_ok`: `phone_ids`
  non-empty 1-D, `spec_codes`/`pitch_codes` 2-D with a non-zero frame count;
  `code_lengths_agree` for the `T` they share), so a corrupt entry is rebuilt
  rather than entering the filelist. `FeatureExtractor` and the stage-1
  `Generator` are built lazily *and only as heavy as needed* — a codes-only run
  never loads Qwen3-ASR, and RMVPE is built on first pitch use — so a pure
  reuse run touches neither. `--checkpoint` is therefore only required when
  code tensors are (re)built.
- `--only phone_ids|spec_codes|pitch_codes` (shorthands `phones`, `codes`,
  `all`) selects which tensors are (re)built; the others are reused as-is, and
  a clip whose reused tensor is missing/unusable is skipped with a printed hint
  rather than silently re-encoded. This is what makes the two common re-runs
  cheap: `--only phones` after a phonemizer/dictionary change (no codec at
  all; add `--phones_from transcript` to also skip audio + ASR) and
  `--only codes` after a new stage-1 checkpoint (no ASR/aligner weights).
  `spec_codes` and `pitch_codes` always move together — one `encode_codes`
  call produces both and they must share `T`. `--force` rebuilds the `--only`
  keys even when `--skip_existing` would reuse them.

**Clips with no phonemes are dropped too.** When the ASR + forced alignment
return an empty phone sequence (typically a 0.4–0.9 s interjection such as
"hmm"), the codes are still valid but there is no text to condition on, so the
clip could only train the stage-2 transformer as an *unconditional* sample. The
guard sits right after the phones are built and before the codes are encoded, so
neither the filelist entry nor the tensors are written (the count is printed in
the final summary). The header checks reject an empty `phone_ids` too, so
`--skip_existing` rebuilds and then re-drops it — i.e. a corpus that still
contains them (e.g. `data/genshin_full_stage2`, 1.0 % of the clips) should be
rebuilt with this guard before a fresh stage-2 run.

Tensor shapes (see `dataset_codes.py` docstring):

- `phone_ids`   `(n_phone,)`               long
- `spec_codes`  `(Q_spec, T_codes)`        long (Q_spec = `codec.codebook_num`)
- `pitch_codes` `(Q_pitch, T_codes)`       long (Q_pitch = `pitch_codec.codebook_num`)

**Heteronyms (homographs).** g2p_en picks between a heteronym's two
pronunciations (`read` past/present, `refuse` verb/noun, `present`, `record`,
…) from the **Penn tag of the word inside its sentence** — it runs
`nltk.pos_tag` over whatever string it is given. `HorsePhonemizer.phonemize`
is called once per forced-alignment token (see `features.extract`), so a naive
per-word call tags each word in isolation and always returns the same
pronunciation (~1 % of the `25_stage2_base1` utterances are affected, mostly
`use`/`used`/`close`/`read`). The fix, already wired in, is:

- `FeatureExtractor.phonemize_items` (used by `extract`) tags the whole
  utterance once (`HorsePhonemizer.tag_words` over the aligner's items) and
  passes each tag down as `phonemize(word, tag=...)`, so the heteronym branch of
  g2p_en's homograph table (`g2p_en.homograph2features`, re-implemented in
  `_homograph_pronunciation`) sees real context. `phonemize(text)` with
  `tag=None` still reproduces the old context-free behaviour, and
  `phonemize_sentence` (what `--phones_from transcript` uses) tags the whole
  transcript at once.
- `new_horsewords.clean` accepts POS-qualified keys (`USE/VB  Y UW1 Z`,
  `LEAD/VB  L IY1 D`) for the cases g2p_en *still* gets wrong even in context
  (its tagger is a small perceptron: "use the tool" is tagged `IN`, and `lead`
  is not in its homograph table at all, so it falls back to `cmudict[0]` =
  the metal). Matching is exact tag first, then two-letter prefix, so
  `USE/VB` covers `VB/VBD/VBP/VBZ`; POS-qualified > plain word > g2p_en.

Because `phone_ids` are baked into the tensors, a phonemizer change only takes
effect on a run that rebuilds them — which is now cheap and touches nothing
else:

```bash
# re-phonemize only; spec/pitch codes are reused and the codec is never loaded
python 2_preprocess.py --filelist ... --output_dir data/base2 --only phones
# same, but also skip audio + ASR: phonemize the filelist transcript directly
python 2_preprocess.py --filelist ... --output_dir data/base2 \
    --only phones --phones_from transcript
```

Both write `phone_ids` per clip from scratch (the `--only` keys are rebuilt by
default; `--skip_existing` would instead reuse the old ones — combine it with
`--force` if you want to keep the reuse contract for the other keys). The
filelists are rewritten as usual, from the same seeded shuffle.

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

`exp_name` drives checkpoint and TensorBoard paths, and **each stage owns its
own tree**:

| stage | checkpoints | TensorBoard |
|---|---|---|
| 1 (`1_training.py`) | `checkpoints_stage1/{exp_name}/` | `logs_stage1/{exp_name}/` |
| 2 (`2_training.py`) | `checkpoints_stage2/{exp_name}/` | `logs_stage2/{exp_name}/` |

The split is deliberate: a stage-1 GAN run is months long, so `rm -rf`-ing a
failed stage-2 experiment must never be able to take it with it (and vice
versa). The roots are literals in the two `train()` functions
(`checkpoints_dir` / `log_dir`) — never reintroduce a bare `'checkpoints'` or
`'logs'` path there. The only knob is `train.log_dir` in the config, which
defaults to (and in the shipped configs is set to) the stage's own root, so a
stale `log_dir: logs` in an old config would silently merge the stages again;
`train()` prints the resolved checkpoint and TensorBoard paths at startup so
this is visible on the first line of a run. `exp_name` colliding across stages
is fine (`base_v0.5` and `transformer_v0.5` would still land in different
trees).

**This does not move existing runs.** Checkpoints and logs written before the
split live in `checkpoints/` and `logs/`; `1_training.py` / `2_training.py`
only look in their own tree, so a pre-split run is *not* auto-resumed and
starts from scratch in the new location. Either move it (`mv
checkpoints/base_v0.5 checkpoints_stage1/base_v0.5`, same for `logs/`) or pass
`--resume_from checkpoints/base_v0.5/last.ckpt` (the rest of the run then
continues into `checkpoints_stage1/`).

Use conda env raraai25 for environment with dependencies including GPU pytorch.

### Audio level normalisation (preprocessing, both stages)

`features.FeatureExtractor.extract_features_ac` levels every clip **before** it
computes the stage-1 `wave`/`spec`/`f0` and, because `2_preprocess.py` uses the
same extractor, before the stage-2 codes are encoded. So one setting keeps both
stages (and the codec's training and inference inputs) at the same operating
point. It is controlled by an optional `data.audio_norm` block, present in
`configs/base.yaml` **and** `configs/transformer.yaml` (they must match — codes
encoded at a different level than the codec was trained on decode at the wrong
loudness):

```yaml
data:
  audio_norm:
    mode: peak        # none | peak | rms
    target_db: -1.0   # dBFS the reference is scaled to
    max_gain_db: 30.0 # cap on boosting (never on attenuation)
    frame_ms: 20.0    # window for mode: rms
```

Notes:

- `none` reproduces the historical behaviour, i.e. every tensor and checkpoint
  produced before this key existed was built from raw audio. `peak` scales the
  loudest sample to `target_db`; `rms` scales the loudest `frame_ms` window
  (a closer proxy for perceived loudness, and less sensitive to a single click,
  without a `pyloudnorm` dependency).
- Keep `target_db` **below 0**: the stage-1 decoder ends in `tanh`, so a target
  at full scale is unreachable and would bias the output layer. `max_gain_db`
  stops a clip whose peak is a click or a noise blip from being boosted by tens
  of dB; near-silent input is returned unchanged.
- The gain is **not** stored on disk, and it is applied inside
  `extract_features_ac`, so `--skip_existing` cannot detect a changed setting.
  Changing it requires rebuilding the stage-1 tensors (their `spec`/`wave` were
  built at the old level; `1_preprocess.py` has no `--force`, so use a fresh
  `--output_dir`) and then `2_preprocess.py --only codes --force`. The
  `phone_ids` are unaffected.
- `f0` is gain-invariant, so pitch tracking and the pitch codec are unchanged.
- `2_preprocess.py`'s silence filter still runs on the **raw** audio before
  `extract_features_ac`, so normalising does not resurrect quiet/noisy clips
  that the level check is meant to drop.
- `normalize_waveform`/`normalize_audio` are idempotent, which is why
  `test_stage1_codes.py` can normalise the input for its metrics and then still
  feed it to `extract_features_ac`; it prints the applied gain.
- Measured reference points: Genshin is already peak-normalised to ~0 dBFS
  (median peak −0.2 dBFS, 62 % of clips within 0.5 dB of full scale), so
  `peak`/`-1.0` changes it by well under 1 dB; the MLP filelists are ~10 dB
  quieter (median peak −10.7, RMS −28.3 dBFS).

### Stage-1 step cost, single-pass critics, and torch.compile

An audited step of `1_training.py` at `batch_size: 32` (`base_v0.4.7d`,
0.805 s/step on a 5060 Ti) splits roughly:

| component | ms |
|---|---|
| `enc_q` + `codec` fwd+bwd (the dataset hands over 8 s of spec, `T=800`) | 136 |
| `dec` (NSF) fwd+bwd (`segment_size/hop_length` = 16 frames = 0.16 s) | 99 |
| discriminator work **as originally written** (4 fwd + 3 graph-bwds) | 511 |
| pitch discriminator, mel + multi-res STFT, 3 × AdamW, Lightning/data | ~57 |

The discriminator dominated because `step()` called `net_d` **four** times per
iteration: `d(fake)` and `d(audio)` for the generator losses, then
`d(fake.detach())` and `d(audio)` again for the D loss, with the G-phase
backward traversing both graphs. Since then:

- **One real pass per step.** `step()` computes `disc_real = net_d(audio)` once,
  uses its feature maps (`.detach()`ed — they are only a target for the
  feature-matching loss) for the generator loss and its scores for the D loss.
  `loss_g` now back-propagates through `disc_fake` only, so the real graph is
  still intact when the D update runs and no `retain_graph` is needed. The
  weights are identical in both uses (the generator step does not touch
  `net_d`), so the losses are bit-identical and the generator's gradient is
  unchanged; the only difference is that the gradients `loss_g` used to push
  into `net_d` (thrown away by `optim_d.zero_grad()` a few lines later) are
  never computed. `net_dp` gets the same treatment. Measured: 511 → 324 ms for
  the discriminator work, 0.848 → 0.709 s/step end to end. **If you add another
  loss that reads discriminator output, take it from `disc_real` /
  `disc_real_feats` instead of calling `net_d` again.**
- **`train.compile_dec`** (default `true` in `configs/base.yaml`) wraps
  `net_g.dec.forward` in `torch.compile(..., dynamic=True)` from
  `on_train_start` (after the initial `test()`). Only the decoder: it is ~12 %
  of a step and runs 1.14x compiled, while `net_d` compiles cleanly (1 graph, 0
  breaks) to **1.0x** — it is bound by its fp32 cuFFT spectrogram STFTs (that
  is why `Discriminator.forward` casts with `x.float()`; `cuFFT doesn't support
  tensor of type: BFloat16`) and cuDNN convs — and `net_g` as a whole only
  reaches 1.1x (7 graphs, 6 breaks: 4 × "Dynamic slicing with Tensor arguments"
  from `rand_slice_segments_with_pitch`, 2 resume-point breaks in the
  quantiser's EMA buffer updates). Details that matter:
  - It patches `dec.forward`, **not** the module. `torch.compile(module)` returns
    an `OptimizedModule` whose `_orig_mod` child renames every decoder key in
    `state_dict()` (`net_g.dec._orig_mod.*`), which breaks checkpoint loading.
    Verified: 497 keys before and after, no `_orig_mod`, values untouched, and
    a fresh `Generator` loads the compiled-state `state_dict` with no mismatch.
  - The warm-up call compiles up front (~15-100 s) and a failure there falls
    back to eager with a printed message instead of dying at step 0. Its RNG
    side effects are undone, so `compile_dec: true`/`false` differ only in speed.
  - `dynamic=True` because validation batches have a different `T` (val skips
    `RandomSubsample`); `train.drop_last: true` keeps training at one shape.
- **`train.stft_range_check`** (default `false`) drops the `assert |x| <= 1`
  guards in `TacotronSTFT.mel_spectrogram`/`linear_spectrogram` — each is a full
  reduction plus a GPU→CPU sync, and training calls it twice per step. The
  generator output is tanh-bounded and the targets are level-normalised below
  full scale, so the guard is only a data sanity check; `check_range=True`
  (the `TacotronSTFT` default) restores it. Note the `except AssertionError:
  return None` in `step()` ("mel_real error, skip") can therefore no longer fire
  — set `stft_range_check: true` if you are debugging a corpus with over-range
  samples.
- **Metric gating.** `training_step` only calls `self.log` for the auxiliary
  scalars on the steps Lightning would write anyway
  (`(batch_idx + 1) % log_every_n_steps == 0`, matching `should_update_logs`);
  `self.log` in `training_step` is `on_step=True, on_epoch=False`, so nothing
  that is aggregated per epoch is lost. `loss_g`/`loss_d` (progress bar) are
  logged every step.

PyTorch updates these numbers; the way to re-measure is a short run of the same
config for the current code and for `git stash`ed code, then median
`Δt/Δstep` from the `loss_g` scalars in the TensorBoard events.

### Codebook loss weighting (stage 2)

The per-codebook cross-entropies are **not** averaged uniformly. `CodeTransformer`
combines them as

```
loss = sum_k w_k * CE_k / sum_k w_k        w = CodeTransformer.loss_weights(epoch)
```

controlled by two optional keys in `configs/transformer.yaml`:

- `transformer.codebook_weights` — list of `codec.codebook_num +
  pitch_codec.codebook_num` floats, order `spec0..spec7, pitch8, pitch9`. Absent
  (or all `1.0`) reproduces the historical `losses.mean()`. Rationale for the
  shipped values: `spec0` is the coarse spectral envelope (errors there cost
  intelligibility and later books can't repair them), books 7 is near-inaudible
  residual detail, and the pitch books have a 2x larger vocabulary (`log 128`
  vs `log 64`) so their raw CE is much flatter.
- `transformer.codebook_weight_warmup_epochs` — soft curriculum: weights are
  linearly interpolated from uniform at epoch 0 to the list above at epoch
  `>= N`. `0` applies them immediately.

Notes:
- Only *relative* weights matter (the sum renormalises), and they only change
  the loss — **no re-preprocessing** is needed when tuning them.
- `code_loss`/`compute_batch_loss` take an `epoch` argument; `2_training.py`
  passes `self.current_epoch` for train and `None` (full target weights) for
  val so `val_loss` stays a fixed yardstick across the warmup.
- The weights are plain attributes, not buffers, so `state_dict` is unchanged
  and existing stage-2 checkpoints load with no key/shape drift.
- Logged per epoch as `train_loss_weight/{spec0..pitch9}`.

### Encoder → decoder conditioning (stage 2, v0.4.1)

The decoder only sees the text through cross-attention to the `PhonemeEncoder`
memory, and that channel can silently die: the loss keeps falling (acoustic
continuity is a strong prior) while the model ignores the phonemes entirely.
This is what happened in v0.4 — the encoder's pre-LN residual stream grew
~20 → ~4000 over six layers, one dimension carried ~4000 at every position,
the stock final `LayerNorm` divided by that dimension's magnitude, the memory's
variation across phoneme positions dropped to **0.7%** of its norm, and the
cross-attention became uniform (entropy / log N ≈ 0.99):
`CE(fixed phonemes) − CE(shuffled phonemes)` was **+0.003 nats**, i.e. zero.

Two `transformer` keys control it:

- `memory_norm` — `temporal` (default since v0.4.1) standardises every memory
  dimension over the *time* axis (masked instance norm + learnable
  `memory_gain`/`memory_bias`), so a runaway dimension can no longer suppress
  the others (measured on the same batch: the memory's per-dimension spread
  over phoneme positions goes from **0.007** to **1.01**, and the
  cross-attention entropy from 0.985 to 0.12 for the same queries). `layer`
  reproduces the historical single `LayerNorm` over the feature axis and exists
  only to load/evaluate ≤ v0.4 checkpoints; `none` returns the raw residual
  stream.
- `residual_scale` — GPT-2-style scaling of the *initial* weights of the
  residual output projections (`self_attn.out_proj`, `multihead_attn.out_proj`,
  `linear2`) by `1/sqrt(2·n_layers)` in the encoder and `1/sqrt(3·n_layers)` in
  the decoder, which stops the residual stream from growing in the first place.
  Initialisation only: no new parameters, no `state_dict` change.

Notes:

- A model built with `memory_norm: temporal` has two extra parameters
  (`encoder.memory_gain`, `encoder.memory_bias`) and **no**
  `encoder.encoder.norm.*`. Resuming a ≤ v0.4 checkpoint with
  `--resume_from` therefore fails Lightning's strict load; use
  `--transfer_from` (it reports the two unexpected keys and leaves
  `memory_gain` at 1.0 / `memory_bias` at 0.0), or evaluate the old checkpoint
  with `memory_norm: layer`.
- Setting `memory_norm: temporal` needs **no** re-preprocessing: it only
  changes how the encoder output is normalised at train time.

### End-of-utterance token (stage 2, v0.5)

By default the decoder can only ever be asked for a fixed number of frames, so
every utterance is padded/truncated to `max_frames` (too long == trailing
babble, too short == clipped words). `transformer.eos_token: true` gives every
codebook **one extra output class** at index `vocab_size + 1` (pad/BOS stays at
`vocab_size`) and teaches the model where a clip ends:

- **Training** — `CodeTransformer._append_eos` appends one EOS frame at frame
  `code_lengths[b]` (i.e. right after the last real frame) to *every* codebook's
  stream, and this happens **before** `build_delay_pattern`, so "emit EOS" is a
  normal next-token target in each book (codebook `k`'s EOS lands in column
  `n + k`). It is only added for samples flagged `complete` — `dataset_codes`
  reports whether the item ended at the clip's real end; random mid-clip crops
  (the usual training case) get no EOS, otherwise the model would learn to stop
  after `train.max_frames` frames.
- **Inference** — `generate` stops at the first EOS from codebook 0 (it leads
  the delay pattern, so its column index *is* the frame count), still decodes
  the `K - 1` delay-tail columns, trims the result to the predicted lengths and
  handles a ragged batch by **repeating each shorter sample's last frame**, never
  the pad id (`vocab_size` is out of range for the stage-1 codec). `max_frames`
  becomes an upper bound, and `generate(..., return_lengths=True)` returns the
  per-sample frame counts. `stop_on_eos=False` restores the old fixed-length
  behaviour; the EOS class is masked in book 0 then, and in *every* book except
  the leading one always (only book 0 decides the end, and an EOS id in the
  output would be an out-of-range codec index).
- **Checkpoints** — the extra class widens all codebook embeddings and heads
  (`v + 1 -> v + 2` rows, plus one bias element per head), so an EOS-less
  checkpoint can **not** be `--resume_from`'d (Lightning's strict load fails on
  ~30 shapes, and the optimizer state is positional). `2_training.py` calls
  `check_eos_layout` and raises a `SystemExit` with that message — also for the
  reverse direction (EOS checkpoint, `eos_token: false`), which has no helper.
  Warm-start with `--transfer_from` instead: `expand_state_dict_with_eos` pads
  the old rows (and biases) with zeros, so only the new EOS class is fresh (a
  neutral logit, ~`1/(v+2)` probability, which then has to be learned).
- **Logging** — the new `sample/gen_frames_{j}`, `probe/gen_frames` and
  `ood/gen_frames_{i}` scalars show how many frames the model chose; for
  `ood` that is the interesting number, because the probe deliberately allows
  more frames than any val item.
- Toggling it needs **no** re-preprocessing (codes are unchanged) — only a new
  `exp_name` (`2_training.py:train` refuses to resume an EOS-less checkpoint
  into it). `dataset_codes` always emits `complete`, so an EOS-less model
  simply ignores it.


### Optimizers & manual optimization (stage 2)

`TrainModule` splits the parameters between **two optimizers**:

- `torch.optim.Muon` on every **2-D** parameter (attention / FFN / projection
  matrices), with `lr * train.muon_lr_mult` (`200` → `0.02` with the shipped
  `lr: 1e-4`; see the comment in `configs/transformer.yaml`), and
- `torch.optim.AdamW` on everything else (biases, LayerNorm gains) with
  `train.lr` / `betas` / `eps`.

Lightning ≥ 2.0 refuses multiple optimizers in automatic optimization
(`Training with multiple optimizers is only supported with manual
optimization`), so `TrainModule.__init__` sets
`self.automatic_optimization = False` and `training_step` runs the step itself:
`zero_grad` → `self.manual_backward(loss)` → clip/log → `AdamW.step()` →
`Muon.step()` → both cosine schedulers. Consequences:

- `on_before_optimizer_step` is **never called** in manual optimization (it
  hangs off the precision plugin's automatic-mode closure), so clipping moved
  into `TrainModule._clip_and_log_grads`, which `training_step` calls
  explicitly. Do not re-add a clipping hook — it would not run.
- Only the AdamW step goes through its `LightningOptimizer` wrapper; that is
  what advances `Trainer.global_step`. Muon is stepped via
  `optim_muon.optimizer.step()`, because every wrapper `.step()` counts one
  global step and two per batch would silently halve `max_steps`,
  `cosine_anneal_end`, `every_n_train_steps` and friends. One batch = one
  step per optimizer, like the old single-optimizer runs.
- `Trainer(accumulate_grad_batches > 1)` is rejected by Lightning under manual
  optimization, so `TrainModule.__init__` raises if
  `train.accumulate_grad_batches > 1`.
- `self.optimizers()` order is `[AdamW, Muon]` and `lr_schedulers()` is
  `[sched_adamw, sched_muon]`; `training_step` unpacks them by position, so
  keep `configure_optimizers` in that order.
- Both optimizers are plain `torch.optim.Optimizer`s, so checkpoints keep two
  `optimizer_states` + two `lr_schedulers` entries; resuming from them works
  as before (a checkpoint saved by the old single-AdamW code would not match).

### Gradient clipping (stage 2)

`TrainModule._clip_and_log_grads` clips gradients before every optimizer
step (called from `training_step`; see the manual-optimization note above),
configured by two optional `train` keys in `configs/transformer.yaml`:

- `grad_clip_val` — max value; `0` (or absent) disables clipping. The
  pre-clip norm is logged either way.
- `grad_clip_algorithm` — `"norm"` (default, `clip_grad_norm_`: clip the total
  norm) or `"value"` (`clip_grad_value_`: clip each element). Only validated
  when `grad_clip_val > 0`; anything else raises `ValueError`.

Notes:

- Clipping is done in Python, **not** via `pl.Trainer(gradient_clip_val=...)`
  — Lightning forbids that with manual optimization (and it would clip twice).
- The gradient buffers are shared by both optimizers (disjoint parameters, one
  backward pass), so clipping once protects both updates.
- Logged: `train_grad_norm` (pre-clip — the spike detector, same tag/meaning
  as before clipping existed), `train_grad_clip_scale` (factor applied,
  `1.0` = no clipping), `lr` (AdamW) and `lr_muon` at `train.log_interval`,
  and per epoch
  `train_grad_norm_max` + `train_grad_clip_frac` (fraction of steps that were
  clipped). The two epoch scalars are computed on *every* step, so they expose
  spikes that fall between two logged steps; they only exist when clipping is
  enabled.
- All of it is tensor math (no `float()` on GPU tensors and no `if tensor > x`
  in the hot path), so the call adds no per-step device sync.
- Non-finite gradients are *not* repaired: Lightning's default
  `error_if_nonfinite=False` is kept, so an `inf` norm gives a clip factor of
  0 (harmless no-op step) while `nan` grads stay `nan`. A `nan`
  `train_grad_norm` means the run is already poisoned — restart from a
  checkpoint.
- Reference magnitude with `batch_size: 8` / `max_frames: 256`: the total grad
  norm is ~0.7-1.0 at init, so the shipped `grad_clip_val: 1.0` only bites on
  spikes. Clipping changes no checkpoint shape, so it needs neither
  re-preprocessing nor a fresh run — a spike can be fixed by resuming.

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
- **Devices:** default to `cuda:0` when available, else `cpu`. Both stages train
  with PyTorch Lightning `accelerator='gpu'`. **Stage 2 must run in
  `precision: 32-true` with `torch.set_float32_matmul_precision("high")`**
  (TF32 GEMMs; set in `2_training.py`'s `__main__` — keep the legacy setter,
  Lightning 2.6 still calls the legacy getter). `bf16-mixed` autocast silently
  corrupts the decoder `norm2.weight/bias` gradients by ~1e4x even at a healthy
  checkpoint, Adam amplifies that noise into a random walk of the LayerNorm
  gains, and the run explodes within a few thousand steps (this killed
  `transformer_v0.2` at step 18.8k and `v0.2.1` immediately). `precision` is
  read from `config.train.precision`, so don't delete the key. Stage 1 still
  uses `bf16-mixed`.
- **`mp.set_start_method('spawn', force=True)`** appears at the top of both
  training `__main__` blocks and is required on Linux; don't remove it.
- **Logging:** `logs_stage1/` and `logs_stage2/` hold the TensorBoard runs (one
  tree per stage — see §4; `train.log_interval` controls
  step logging). Validation audio for stage 2 is optionally logged by loading a
  frozen stage-1 Generator (`train.generator_checkpoint`); it is non-fatal if
  the checkpoint is missing, and every visual/audio log is skipped without a
  stage-1 checkpoint or TensorBoard logger.
- **Stage-2 diagnostics.** `2_training.py` logs, per codebook, `{train,val}_`
  `loss/`, `acc/`, `usage_entropy/` (`spec0..7`, `pitch8..9`) — these per-book
  curves are always **unweighted** (raw diagnostics), whereas `{train,val}_loss`
  is the weighted combination described in §4, and
  `train_loss_weight/{spec0..pitch9}` shows the effective weights per epoch.
  It also logs
  `{train,val}_{acc,spec_acc,pitch_acc}`, `train_grad_norm` /
  `train_grad_clip_scale` (plus per-epoch `train_grad_norm_max` /
  `train_grad_clip_frac` when clipping is on — see §4), and `lr` (AdamW) /
  `lr_muon` (see §4 for the two-optimizer manual-optimization setup).
  Every epoch it also logs `val_text_delta` (plus `/spec` and `/pitch`): the
  teacher-forced CE on the first `train.text_delta_batches` validation batches
  with the phonemes rotated across samples *minus* the CE with the real ones.
  This is the "is the text conditioning alive?" signal — `val_loss` cannot show
  it, because acoustic continuity alone keeps the loss falling. ~0 nats means
  the decoder ignores the encoder memory (see §4 and
  `test_stage2_conditioning.py`); `0` in `train.text_delta_batches` disables
  the probe. Every `train.sample_interval`
  epochs (and at train start) it also samples
  `sample_count` val items with `sample_temperature`/`sample_top_k` and logs
  `codes/{spec,pitch}_*` (target-vs-generated codebook heatmaps) plus the
  free-running agreement scalars `sample/{spec,pitch}_code_match_frac_{j}`,
  `mel/`, `pitch/`, and — when `train.speaker_probe > 0` and the val set has
  ≥2 speakers — `probe/*` (same phonemes re-generated per speaker, with
  `probe/{spec,pitch}_code_change_frac` and `probe/f0_delta_hz` as numeric
  "is the speaker conditioning doing anything" signals). Pitch curves come
  from `Generator.infer_from_codes(..., return_pitch=True)`, which returns f0
  in Hz at the audio frame rate (`sampling_rate / hop_length`).
  Each sampling pass also runs an **out-of-distribution text probe**
  (`train.ood_samples`, default 1; `0` disables): `DEFAULT_OOD_TEXTS` (or
  `train.ood_texts`) is phonemized on CPU via
  `FeatureExtractor.phonemize_text` with `load_qwen=False` — no ASR/aligner
  weights, ids cached after the first pass — and decoded autoregressively for
  `train.ood_frames` (default `4 * sample_frames` = 512) frames, i.e.
  deliberately longer than the val clips, which are capped at their own
  `code_len`. There is no target for unseen text, so only the generated side
  is logged: `ood/text_*`, `ood/{i}/codes_{spec,pitch}`, and (with a stage-1
  Generator) `ood/audio_*`, `ood/pitch_*`, `ood/mel_*`. Speaker is
  `train.ood_sid` or the first sampled val item's; failures are printed and
  never interrupt training.

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
5. **`--skip_existing` / `--only` / `skip_if_one_exists`:** in `2_preprocess.py`
   state is decided **per file and per tensor**, which is what lets one stage be
   re-run without the others. Each tensor is header-checked on its own
   (`tensor_shape` + `key_shape_ok`: non-empty 1-D `phone_ids`, 2-D
   `spec_codes`/`pitch_codes` with a non-zero frame count) so a truncated file is
   rebuilt instead of silently entering the filelist; the pair *reused* code
   tensors is additionally cross-checked by `code_lengths_agree`. Selected
   (`--only`) tensors are rebuilt unless `--skip_existing` finds them usable
   (`--force` overrides), while unselected tensors are reused and a clip is
   skipped with a hint if one of them is missing. A phonemizer change is
   therefore `--only phones` (no codec, no `--checkpoint`) and a new stage-1
   checkpoint is `--only codes` (no ASR/aligner weights). The old bug where the
   "already exists" check `return`ed from the *whole loop* on the first hit
   (truncating the filelists) must not come back. When the silence filter is on
   (its default — see §3) reused clips are still level-checked, which loads the
   audio but re-extracts nothing; if the source audio has disappeared the clip
   is kept as-is instead of being dropped (only a clip that still needs work is
   skipped). `FeatureExtractor` and the stage-1 `Generator` are built lazily and
   only as heavy as needed (`load_qwen=need_qwen`, RMVPE on demand), so a
   reuse-only run never constructs either and a bogus `--checkpoint` is harmless
   in that case. `1_preprocess.py` likewise skips
   per file and, with `--skip_existing`, also re-checks the clip length, so
   re-running it over an already-extracted corpus only rewrites
   `train.txt`/`val.txt` and never builds the heavy `FeatureExtractor`.
6. **Features require heavy downloads** on first use: `Qwen3-ASR-0.6B`,
   `Qwen3-ForcedAligner-0.6B`, RMVPE weights, `therealvul/g2pen_tokenizer_clean`.
   `FeatureExtractor.__init__` accepts `load_qwen=True` default; code that only
   needs spec/pitch can avoid loading Qwen when not needed.
7. **Code rates:** spec and pitch codes must have identical lengths per sample
   (both are produced at the decoder frame rate / `down_factor`). The Generator
   asserts this in `infer_from_codes`.
8. **`torch.load(..., weights_only=True)`** is used in `dataset_codes.py`, but
   `weights_only=False` is required for full checkpoints (see §5).
9. **The pitch codec predicts voicing (V/UV) itself.** `Generator.pitch_uv_proj`
   is a per-frame head trained with `c_pitch_uv` (BCE against `f0 > 0`) plus a
   small unvoiced-f0 regression weighted by `c_pitch_uv_frame`. At synthesis,
   `infer_from_codes(..., voiced=None)` thresholds these logits
   (`pitch_uv_threshold`) to decide voicing; pass an explicit mask to override.
   Consequences:
   - A stage-1 checkpoint created **before** this head existed loads it randomly
     initialized (`load_state_dict_mismatch(..., strict=False)`), so its
     predicted voicing is meaningless. Fine-tune with
     `python 1_training.py --config configs/base.yaml --pitch_only` (the head is
     in `freeze_non_pitch`'s trainable set) before relying on predicted voicing.
   - No voicing mask needs to be stored on disk for stage 2: the information
     lives in the pitch codes and is decoded by the head.
10. **Stage-2 `decoder_speaker_cond` changes the parameter set.** The decoder
    now has its own speaker embedding (`transformer.decoder_speaker_cond:
    true`, default). Checkpoints trained without it (`transformer_v0.1`) do
    **not** load strictly: set `decoder_speaker_cond: false` *and* match the
    layer counts that checkpoint used (v0.1 was `encoder_layers: 4`,
    `decoder_layers: 10`) when resuming/transferring from it.
    `load_state_dict_mismatch` only reports shape/key mismatches, so silent
    partial loads are easy to miss — always check the reported keys.
11. **Preprocessed tensors are named by *basename* only** (`data/base/<basename
    of the source>.wave`, `data/base2/<basename>.spec_codes`). Two source files
    with the same basename in different folders therefore overwrite each other
    and both filelist lines point at the same tensor. `filelists_1/base_linux.txt`
    has 188 such collisions (502 extra lines) — 474 duplicate lines in
    `data/base/train.txt` and 27 items shared with `val.txt` — while
    `filelists_2/` currently has none. Check with
    `python dataset_duration.py data/base/train.txt data/base/val.txt`, which
    reports duplicate lines and train/val overlap.
12. **Changing `data.segment_size` requires re-running `1_preprocess.py`.** The
    generator slices training segments with
    `rand_slice_segments_with_pitch(yq, f0, spec_l, segment_size // hop_length)`
    (`modeling/generator.py`), and `x_lengths` there is the **real** clip length
    from `dataset_ac`, not the batch-padded `T`. For a clip shorter than the
    segment, `ids_str_max = spec_l - segment` goes negative, so
    `slice_segments` gets a negative start index and the slice comes back empty
    → `RuntimeError` on the assignment in `modeling/vits/commons.py`. Padding
    does not save you (`PadGroup(to_length=[257])` only lengthens the tensor).
    Because such clips are rare (a handful per 100 k), the crash lands a few
    hundred steps into a run instead of at startup. `1_preprocess.py` already
    filters them — its `min_frames` default is `segment / hop_length` — so just
    re-run it with `--skip_existing` after any `segment_size` change:

    ```bash
    python 1_preprocess.py --filelist filelists_1/genshin_full_eng_linux.txt \
        --output_dir data/genshin_full_stage1 --skip_existing
    ```

    This only re-validates lengths and rewrites `train.txt`/`val.txt`; nothing
    is re-extracted, and the val set stays ~identical (the shuffle is seeded
    with `--shuffle_seed 42`, and the 5 % split is the tail of the shuffled
    kept list, so dropping a few clips shifts only a handful of members).
13. **`transformer.eos_token` changes the parameter shapes**, so it is the one
    stage-2 switch that a `--resume_from` cannot absorb: every codebook
    embedding/head (weights *and* bias) grows by a row. `2_training.py` calls
    `check_eos_layout`, which exits with instructions in **both** directions
    (EOS-less -> EOS needs a new `exp_name` + `--transfer_from`, EOS -> EOS-less
    has no helper at all). Without that, Lightning's strict load would fail on
    ~30 tensors, and a `--transfer_from` would silently re-initialise them —
    `load_state_dict_mismatch` only reports shape mismatches, it does not fix
    them. `--transfer_from` runs `expand_state_dict_with_eos`, which zero-pads
    the old rows so only the new EOS class is fresh.
    `dataset_codes` emits the `complete` flag unconditionally, so EOS-less
    models and their configs need no change (see §4).
14. **The two stages write to separate checkpoint/log trees**
    (`checkpoints_stage1/` + `logs_stage1/` for `1_training.py`,
    `checkpoints_stage2/` + `logs_stage2/` for `2_training.py`). This is the
    point of the split: stage-1 runs are months long, so deleting a failed
    stage-2 experiment must never be able to reach them. Consequences:
    - The paths are set by `checkpoints_dir` / `log_dir` at the top of each
      `train()`. Adding a new `ModelCheckpoint` (or any other writer) **must**
      use those variables, not a literal `'checkpoints'`/`'logs'` — a single
      stray literal puts that artefact back in the shared tree.
    - `train.log_dir` in the config overrides only the TensorBoard root; the
      shipped configs set it to `logs_stage1` / `logs_stage2`. A config left
      with `log_dir: logs` silently writes both stages' runs into `logs/`.
    - Pre-split artefacts stay where they are. Both scripts only auto-resume
      from their own tree, so an old run needs `mv checkpoints/<exp>
      checkpoints_stage1/<exp>` (and the `logs/` equivalent) or an explicit
      `--resume_from checkpoints/<exp>/last.ckpt`.
    - Anything that *consumes* a checkpoint by path (`2_preprocess.py
      --checkpoint`, `test_stage1_codes.py`, `train.generator_checkpoint` in
      `configs/transformer.yaml`) is unaffected: those take a full path, so
      update it when you move a stage-1 run. Consumers *inside* `2_training.py`
      must read the stage-1 tree, never their own.

---

## 7. Useful commands

```bash
# Fetch the published stage-1 tensors (1204 .tar.zst shards, ~325 GB extracted)
# on a fresh vast.ai box; streams curl | zstd -dc | tar -x into /data, resumable.
# `--link-project .` symlinks data/genshin_full_stage1 -> <dest> for the configs.
docker/fetch_shards.sh --dest /data/genshin_full_stage1 --link-project .
docker/fetch_shards.sh --dry-run --dest /data/genshin_full_stage1   # plan only

# Stage-1 preprocessing / training
python 1_preprocess.py --filelist filelists_1/base_linux.txt --output_dir data/base
python 1_training.py --config configs/base.yaml            # auto-resumes last.ckpt

# Resume / transfer for stage 1
# (auto-resume looks in checkpoints_stage1/{exp_name}/last.ckpt; the explicit
#  paths below are a full path, so they can live in either tree)
python 1_training.py --config configs/base.yaml --resume_from checkpoints_stage1/base_v0.4.2/last.ckpt
python 1_training.py --config configs/base.yaml --transfer_from some.ckpt   # partial weight transfer

# Dataset duration / duplicate / train-val-overlap report
# (auto-detects stage-1, stage-2 and raw-audio filelists; --json for scripting)
python dataset_duration.py data/base2/train_stage2.txt data/base2/val_stage2.txt \
    --per-speaker --hist

# Speaker map bootstrap
python speaker_map.py --filelist filelists_2/25_stage2_base1.txt \
    --stage1-filelist filelists_1/base_linux.txt --out speakers.json

# Stage-2 preprocessing / training
python 2_preprocess.py --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 --checkpoint checkpoints_stage1/base_v0.4/last.ckpt \
    --config configs/base.yaml --speaker_map speakers.json
python 2_training.py --config configs/transformer.yaml

# Stage 2, one tensor stage at a time (see §3): re-phonemize only
python 2_preprocess.py --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 --only phones          # reuses both code tensors
python 2_preprocess.py --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 --only phones --phones_from transcript   # no audio/ASR
# ...or re-encode only (new stage-1 checkpoint; no ASR/aligner weights)
python 2_preprocess.py --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 --only codes \
    --checkpoint checkpoints_stage1/base_v0.5/last.ckpt

# Rewrite the stage-2 filelists only (no re-extraction), now also dropping
# silent/near-silent clips and clips that are <5% non-silent frames. The
# silence check is the default (-50 dBFS on the loudest 20 ms frame);
# `--min_rms_db none` disables it, `--min_rms_db -35` is stricter.
python 2_preprocess.py --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 --checkpoint checkpoints_stage1/base_v0.4/last.ckpt \
    --config configs/base.yaml --speaker_map speakers.json \
    --skip_existing --min_speech_frac 0.05

# Stage-1 codec round-trip on your own audio (spec/f0 -> codes -> audio)
python test_stage1_codes.py test.wav \
    --checkpoint checkpoints_stage1/base_v0.4.6/last.ckpt --sid 0
# ...add --roundtrip for code -> audio -> code match fractions and
# --save_tensors to dump spec/pitch codes + both f0 contours.

# Ad-hoc smoke tests (many modules have an `if __name__ == "__main__"` block)
python modeling/transformer.py    # constructs a CodeTransformer, runs fwd + generate
python modeling/generator.py      # constructs a Generator, runs infer

# Is the stage-2 text conditioning actually alive? (CPU by default, so it can
# run next to a training job; --device cuda for speed)
python test_stage2_conditioning.py \
    --checkpoint checkpoints_stage2/transformer_v0.4.1/last.ckpt --batches 4

# Phonemization smoke test: both stage-2 paths (transcript-only vs aligner),
# per-word word|tag|phones tables plus PASS/FAIL checks and a cross-path diff
python test_phonemize.py                     # default sentences + test.wav (ASR)
python test_phonemize.py --skip_audio        # transcript path only (no Qwen, CPU)
python test_phonemize.py --skip_text --audio test.wav \
    --transcript "Hey. you don't sound too excited."   # forced alignment
python test_phonemize.py --skip_audio --text "Use it." --expect YUW1S  # pin output
```

There is no test suite in this repo; validation is done by running the ad-hoc
`__main__` blocks in `modeling/*.py` and by training on a small filelist.
`test_stage1_codes.py` is the exception worth knowing about: it exercises the
exact encode (`2_preprocess.py`) / decode (stage-2 validation) path of a stage-1
checkpoint on arbitrary audio and prints mel/f0/voicing metrics plus per-codebook
usage. `test_stage2_conditioning.py` is its stage-2 counterpart: it reports the
encoder-memory scale/spread, the per-layer cross-attention entropy and the
teacher-forced loss with real vs shuffled vs random phonemes, and prints a
DEAD/WEAK/HEALTHY verdict on the text conditioning (see §4).
`test_phonemize.py` covers the text side instead: it runs both ``phone_ids``
paths (`phonemize_text` = ``--phones_from transcript`` and
`align` + `phonemize_items` = ``--phones_from aligner``), prints the per-word
word / Penn tag / phones table, checks the tensor invariants (non-empty, no
``[UNK]``, ids inside the vocab, ``decode(ids)`` round-trip) and diffs the two
paths on the same utterance, exiting non-zero on failure.
