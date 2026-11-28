#!/usr/bin/env bash
# Word-progress / duration fine-tune of the stage-2 transformer.
#
# Warm-starts from the current best hints checkpoint.  The new
# `net.word_progress_head` is fresh; every other weight transfers.  A run with
# this exp_name auto-resumes from checkpoints_stage2/transformer_word_progress_v1/last.ckpt
# on a second invocation, so use --transfer_from only for the first start (or a
# new exp_name for a fresh arm).
set -euo pipefail
cd "$(dirname "$0")/.."

python 2_training.py \
    --config configs/transformer_word_progress.yaml \
    --transfer_from checkpoints_stage2/transformer_hints_v2.4/last.ckpt
