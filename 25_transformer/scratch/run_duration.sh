#!/usr/bin/env bash
# Oracle duration-conditioning prototype (see configs/transformer_duration.yaml).
#
# The decoder gets the true per-frame phone progress from the .cond alignment,
# so the test is "does telling it exactly where it is in the text fix the
# swallowed final words / stretched pacing?".  The paired control is the same
# checkpoint with the schedule absent, logged as `wer/generated_nodur`.
#
# Warm start: the word-progress run (same parameter set plus the two new
# channel vectors, which load as zeros = exact identity).
set -euo pipefail
cd "$(dirname "$0")/.."

python 2_training.py \
    --config configs/transformer_duration.yaml \
    --transfer_from checkpoints_stage2/transformer_word_progress_v1/last.ckpt
