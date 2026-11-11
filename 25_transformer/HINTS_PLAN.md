# Stage-2 conditioning hints — plan and status

**Status (2026-11-11):** M0, M1, M2, M3 (the oracle probe) and M4 are done and
validated; M3b (per-channel probes) and M5 (inference ergonomics) are not
started. The code is committed `d5a69c2`. **No corpus has `.cond` files yet** —
the word-level pass (`hints/build.py --words`, ~8–9 h) is running, and until it
finishes `transformer.hints` cannot be switched on for a real run.

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
| register | `n_levels` corpus quantiles | voiced-median f0, semitones re: 100 Hz | encoder (constant per clip) |
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
| `.cond` total | 400 clips = 1.94 MB; **≈ 0.74 GB** for the 151 653-clip corpus |
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
- **Corpus:** `data/*_stage2/` holds **no `.cond` files yet**; the word pass is
  running (~8–9 h). It writes `hints.json` next to the first `--filelist`, i.e.
  `data/genshin_plus_extras/hints.json`, when it finishes. Progress:

  ```bash
  ls data/*_stage2/*.cond | wc -l     # grows in waves of 2000, then per clip
  ```
- **Config:** `transformer.hints: false` on purpose — a hints model needs the
  `.cond` corpus *and* a fresh `exp_name` (new parameter tables).
- **Disk:** `/mnt/data` at 89 % (411 GB free); the pass itself costs ~0.74 GB.

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
| register | voiced-median f0 of the decoded audio vs level (RMVPE) |
| word pitch | per-word f0 delta ≈ the bin's semitones |
| word timing | word durations follow the slot bucket |
| semantic | hardest; check that swapping two sentences' vectors in the *same* text changes prosody, and that the gap probe is non-zero |
| regression | hint-free WER (`scratch/eval_stage2_wer.py`) ≥ v0.7's, against the ground-truth-code floor |

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



