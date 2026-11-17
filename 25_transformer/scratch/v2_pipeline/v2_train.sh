#!/usr/bin/env bash
# Hints V2, phase C+D: rebuild the per-word pitch anchors + word fields for the
# combined corpus, then start training.  Resumable: the .cond files are written
# per clip, and a failed pass is retried with --skip-existing.
set -u
cd /mnt/data/Code/raraai/25_transformer
PY=/home/vul/miniconda3/envs/raraai25/bin/python
LOG=logs_stage2/v2_pipeline.log
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

say "=== phase C: hints (words + anchors) ==="
HINT_ARGS=(
  --filelist data/genshin_plus_extras/train_stage2.txt
             data/genshin_plus_extras/val_stage2.txt
  --wave-dir data/genshin_full_stage1 data/mlp_stage1
  --source-filelist filelists_2/genshin_full_eng_text_linux.txt
                    filelists_2/mlp_linux.txt
                    filelists_2/vctk_linux.txt
                    filelists_2/expresso_read_linux.txt
  --hints intensity,register
  --words --pitch-anchors 3 --relative-register
  --hints-json data/genshin_plus_extras/hints_rel.json
  --device cuda:0
)
ok=0
for attempt in 1 2 3; do
  extra=()
  [ "$attempt" -gt 1 ] && extra=(--skip-existing)
  if run "$PY" hints/build.py "${HINT_ARGS[@]}" "${extra[@]}"; then ok=1; break; fi
  say "hints attempt $attempt failed; retrying"
  sleep 30
done
[ "$ok" -eq 1 ] || { say "ABORT: hints pass failed 3x"; exit 30; }

say "=== phase C check: anchors coverage ==="
if ! "$PY" - <<'PY' >>"$LOG" 2>&1
import os, torch
fl = "data/genshin_plus_extras/train_stage2.txt"
lines = [l.split("|")[0] for l in open(fl) if l.strip()]
step = max(1, len(lines)//4000)
sample = lines[::step]
ok = miss = 0
for phone in sample:
    cond = phone[:-len(".phone_ids")] + ".cond"
    try:
        c = torch.load(cond, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        miss += 1
        continue
    if "pitch_anchors_st" in c and "pitch_anchor_n" in c:
        ok += 1
    else:
        miss += 1
print(f"anchor coverage on {len(sample)} sampled train clips: {ok} ok / {miss} missing")
raise SystemExit(0 if ok >= 0.99 * len(sample) else 1)
PY
then
  say "ABORT: anchor coverage below 99%"
  exit 31
fi
say "anchor coverage OK"

say "=== phase D: training ==="
run "$PY" 2_training.py --config configs/transformer.yaml \
  --transfer_from checkpoints_stage2/hints_v1_p4_final_mono1/last.ckpt
rc=$?
say "=== training exited rc=$rc ==="
exit "$rc"
