#!/usr/bin/env bash
# Hints V2 phase D: the corpus is built (phones from transcript, .cond v2 with
# per-word pitch anchors), so this only starts training.  Safe to re-run: the
# script auto-resumes checkpoints_stage2/transformer_v2.0_anchors/last.ckpt.
set -u
cd /mnt/data/Code/raraai/25_transformer
PY=/home/vul/miniconda3/envs/raraai25/bin/python
LOG=logs_stage2/v2_pipeline.log
mkdir -p logs_stage2
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== phase C check: anchor coverage ==="
if ! "$PY" - <<'PY' >>"$LOG" 2>&1
import torch
fl = "data/genshin_plus_extras/train_stage2.txt"
lines = [l.split("|")[0] for l in open(fl) if l.strip()]
step = max(1, len(lines) // 4000)
sample = lines[::step]
ok = miss = 0
for phone in sample:
    cond = phone[:-len(".phone_ids")] + ".cond"
    try:
        c = torch.load(cond, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        miss += 1
        continue
    ok += ("pitch_anchors_st" in c and "pitch_anchor_n" in c)
    miss += not ("pitch_anchors_st" in c and "pitch_anchor_n" in c)
print(f"anchor coverage on {len(sample)} sampled train clips: {ok} ok / {miss} missing")
raise SystemExit(0 if ok >= 0.99 * len(sample) else 1)
PY
then
  say "ABORT: anchor coverage below 99%"
  exit 31
fi
say "anchor coverage OK"

say "=== phase D: training ==="
"$PY" 2_training.py --config configs/transformer.yaml \
  --transfer_from checkpoints_stage2/hints_v1_p4_final_mono1/last.ckpt \
  >>logs_stage2/v2_train.out 2>&1
say "=== training exited rc=$? ==="
