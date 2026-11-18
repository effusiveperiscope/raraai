#!/usr/bin/env bash
# Hints V2.1: per-word bert-base states into every .cond (text only).
# Resumable: `--skip-existing` only computes clips whose .cond lacks word_bert.
set -u
cd /mnt/data/Code/raraai/25_transformer
PY=/home/vul/miniconda3/envs/raraai25/bin/python
LOG=logs_stage2/v2_word_bert.log
mkdir -p logs_stage2
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
for attempt in 1 2 3; do
  extra=()
  [ "$attempt" -gt 1 ] && extra=(--skip-existing)
  say "=== word-bert attempt $attempt ==="
  if "$PY" hints/build.py \
      --filelist data/genshin_plus_extras/train_stage2.txt \
                 data/genshin_plus_extras/val_stage2.txt \
      --word-bert \
      --source-filelist filelists_2/genshin_full_eng_text_linux.txt \
                        filelists_2/mlp_linux.txt \
                        filelists_2/vctk_linux.txt \
                        filelists_2/expresso_read_linux.txt \
      --hints-json data/genshin_plus_extras/hints_rel.json \
      --hints "" --device cuda:0 "${extra[@]}" >>"$LOG" 2>&1; then
    say "OK word-bert attempt $attempt"
    exit 0
  fi
  say "FAIL word-bert attempt $attempt"
  sleep 20
done
say "ABORT: word-bert failed 3x"
exit 1
