#!/usr/bin/env bash
# Alignment supervision (guided attention), see configs/transformer_align.yaml.
#
# The cross-attention of `align_layers` (8 and 11) is scored against the forced-alignment
# phone schedule, because every checkpoint measured so far attends to the
# phoneme memory uniformly (entropy/log N 0.96-0.99) and therefore paces and
# sequences from an acoustic prior.  Watch `align_entropy`, not WER.
#
# Warm start: v2.4, the common ancestor of the word-progress and duration arms
# (its 120-utterance offline baseline is 15.75 % against a 6.08 % floor).
set -euo pipefail
cd "$(dirname "$0")/.."

# The supervised cross-attention retains one (B, heads, L, phones) map per
# layer; expandable segments keeps the allocator from fragmenting 16 GB away.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python 2_training.py \
    --config configs/transformer_align.yaml \
    --transfer_from checkpoints_stage2/transformer_hints_v2.4/last.ckpt
