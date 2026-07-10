#!/bin/bash
# S2 driver — T4 W-sweep freeze-safe (coffee, n=3) + T5 weighted-vs-unit ABAB (W=50).
# Runs INSIDE WSL. One harness invocation per W-group / T5 arm-round so a crash
# loses at most one group. Progress in $BASE/driver.log; per-group summary.csv
# appears when the group completes (Windows side polls and commits incrementally).
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
BASE=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local
mkdir -p "$BASE"
LOG=$BASE/driver.log

# Dedicated lock: the user's UI server (port 8000) holds /tmp/ds4.lock — do not touch it.
export DS4_LOCK_FILE=/tmp/ds4_s2.lock
# IO-path speedup only (page-cache backed streaming + prefetch threads), same as
# the M1/w100 SOTA_LOCAL_3060_TIMED profile; no routing/quality semantics.
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_REAP_PREFETCH_THREADS=16
export DS4_REAP_PREFETCH_LOCK=1

COMMON="--binary /root/ds4/ds4 --model /root/models/ds4-2bit.gguf --cache 256 --ctx-p1 4096 --ctx-p2 4096 --total 1200 --keep-k 23 --timeout 3600"
cd "$REPO" || exit 1
echo "[driver] start $(date -Is)" >> "$LOG"

# ---- T4 core: W-sweep, weighted mask, n=3 per W ----
for W in 30 50 70 90 110 130 150; do
  OUT=$BASE/t4_W$(printf %03d "$W")
  if [ -f "$OUT/summary.csv" ]; then
    echo "[driver] T4 W=$W already done, skip $(date -Is)" >> "$LOG"
    continue
  fi
  echo "[driver] T4 W=$W start $(date -Is)" >> "$LOG"
  python3 scripts/run_w_sweep_freeze_safe.py $COMMON \
    --w-values "$W" --runs 3 --mask-mode weighted --outdir "$OUT" >> "$LOG" 2>&1
  echo "[driver] T4 W=$W done rc=$? $(date -Is)" >> "$LOG"
done

# ---- T5: weighted vs unit at W=50, ABAB rounds, n=3 per arm ----
for R in 0 1 2; do
  for MODE in weighted unit; do
    OUT=$BASE/t5_${MODE}_r$R
    if [ -f "$OUT/summary.csv" ]; then
      echo "[driver] T5 $MODE r$R already done, skip $(date -Is)" >> "$LOG"
      continue
    fi
    echo "[driver] T5 $MODE r$R start $(date -Is)" >> "$LOG"
    python3 scripts/run_w_sweep_freeze_safe.py $COMMON \
      --w-values 50 --runs 1 --mask-mode "$MODE" --outdir "$OUT" >> "$LOG" 2>&1
    echo "[driver] T5 $MODE r$R done rc=$? $(date -Is)" >> "$LOG"
  done
done

echo "[driver] ALL DONE $(date -Is)" >> "$LOG"
