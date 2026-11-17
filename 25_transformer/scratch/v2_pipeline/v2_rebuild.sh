#!/usr/bin/env bash
# Hints V2 corpus rebuild, phase A+B: re-phonemize every corpus from its
# transcript, then re-combine.  Idempotent: safe to re-run if interrupted.
set -u
cd /mnt/data/Code/raraai/25_transformer
PY=/home/vul/miniconda3/envs/raraai25/bin/python
LOG=logs_stage2/v2_rebuild.log
mkdir -p logs_stage2

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run() {
  say "BEGIN $*"
  if "$@" >>"$LOG" 2>&1; then
    say "OK    $*"
  else
    say "FAIL($?) $*"
    return 1
  fi
}

say "=== phase A: phones from transcript ==="
run "$PY" 2_preprocess.py \
  --filelist filelists_2/genshin_full_eng_text_linux.txt \
  --output_dir data/genshin_full_stage2 \
  --only phones --phones_from transcript || exit 10
run "$PY" 2_preprocess.py \
  --filelist filelists_2/mlp_linux.txt \
  --output_dir data/mlp_stage2 \
  --only phones --phones_from transcript || exit 11
run "$PY" 2_preprocess.py \
  --filelist filelists_2/vctk_linux.txt \
  --output_dir data/vctk_stage2 \
  --only phones --phones_from transcript || exit 12
run "$PY" 2_preprocess.py \
  --filelist filelists_2/expresso_read_linux.txt \
  --output_dir data/expresso_read_stage2 \
  --only phones --phones_from transcript || exit 13

say "=== phase B: combine ==="
run "$PY" 2_combine_corpora.py \
  --base data/genshin_full_stage2 \
  --extra data/mlp_stage2 \
  --extra data/vctk_stage2 \
  --extra data/expresso_read_stage2 \
  --output_dir data/genshin_plus_extras \
  --cap_total_multiplier 0.5 || exit 20

say "=== done: $(wc -l < data/genshin_plus_extras/train_stage2.txt) train / $(wc -l < data/genshin_plus_extras/val_stage2.txt) val ==="
