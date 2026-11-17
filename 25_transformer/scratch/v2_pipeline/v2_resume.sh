#!/usr/bin/env bash
# Resume the Hints V2 pipeline after a reboot / crash.
#
# The corpus is already built (phones from transcript on all four corpora, then
# one combine, then .cond v2 with per-word pitch anchors, 99.92 % coverage), so
# by default this only resumes *training*.  Set RUN_PHONES=1 to re-check the
# phone tensors and RUN_HINTS=1 to refill any `.cond` still missing anchors
# (both are idempotent; the hints pass always uses --skip-existing).
set -u
cd /mnt/data/Code/raraai/25_transformer
PY=/home/vul/miniconda3/envs/raraai25/bin/python
LOG=logs_stage2/v2_pipeline.log
mkdir -p logs_stage2
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run() { say "BEGIN $*"; if "$@" >>"$LOG" 2>&1; then say "OK    $*"; else say "FAIL($?) $*"; return 1; fi; }

if [ "${RUN_PHONES:-0}" = "1" ]; then
  say "=== phones (idempotent) ==="
  run "$PY" 2_preprocess.py --filelist filelists_2/genshin_full_eng_text_linux.txt \
    --output_dir data/genshin_full_stage2 --only phones --phones_from transcript || exit 10
  run "$PY" 2_preprocess.py --filelist filelists_2/mlp_linux.txt \
    --output_dir data/mlp_stage2 --only phones --phones_from transcript || exit 11
  run "$PY" 2_preprocess.py --filelist filelists_2/vctk_linux.txt \
    --output_dir data/vctk_stage2 --only phones --phones_from transcript || exit 12
  run "$PY" 2_preprocess.py --filelist filelists_2/expresso_read_linux.txt \
    --output_dir data/expresso_read_stage2 --only phones --phones_from transcript || exit 13
  run "$PY" 2_combine_corpora.py --base data/genshin_full_stage2 \
    --extra data/mlp_stage2 --extra data/vctk_stage2 --extra data/expresso_read_stage2 \
    --output_dir data/genshin_plus_extras --cap_total_multiplier 0.5 || exit 20
fi

if [ "${RUN_HINTS:-0}" = "1" ]; then
  say "=== hints refill (--skip-existing) ==="
  run "$PY" hints/build.py \
    --filelist data/genshin_plus_extras/train_stage2.txt data/genshin_plus_extras/val_stage2.txt \
    --wave-dir data/genshin_full_stage1 data/mlp_stage1 \
    --source-filelist filelists_2/genshin_full_eng_text_linux.txt \
                      filelists_2/mlp_linux.txt \
                      filelists_2/vctk_linux.txt \
                      filelists_2/expresso_read_linux.txt \
    --hints intensity,register --words --pitch-anchors 3 --relative-register \
    --hints-json data/genshin_plus_extras/hints_rel.json \
    --device cuda:0 --skip-existing || exit 30
fi

say "=== training (auto-resumes last.ckpt) ==="
exec bash scratch/v2_pipeline/v2_train_only.sh
