#!/bin/bash
# S2 driver v2 — REORDERED per coordinator (decisional value first):
#   T4: W50 (pavimento storico) -> W30 (sotto-pavimento) -> W130 (knife-edge replay)
#   T5: weighted-vs-unit ABAB (W=50, 3 rounds) BEFORE the tail of the grid
#   T4 tail: W90 -> W70 -> W110 -> W150
# One harness invocation per group; resume via summary.csv marker.
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
BASE=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local
mkdir -p "$BASE"
LOG=$BASE/driver.log

export DS4_LOCK_FILE=/tmp/ds4_s2.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_REAP_PREFETCH_THREADS=16
export DS4_REAP_PREFETCH_LOCK=1

COMMON="--binary /root/ds4/ds4 --model /root/models/ds4-2bit.gguf --cache 256 --ctx-p1 4096 --ctx-p2 4096 --total 1200 --keep-k 23 --timeout 3600"
cd "$REPO" || exit 1
echo "[driver-v2] start $(date -Is) order: T4(50,30,130) T5(ABABx3) T4(90,70,110,150)" >> "$LOG"

run_t4 () {
  W=$1
  OUT=$BASE/t4_W$(printf %03d "$W")
  if [ -f "$OUT/summary.csv" ]; then
    echo "[driver-v2] T4 W=$W already done, skip $(date -Is)" >> "$LOG"
    return
  fi
  echo "[driver-v2] T4 W=$W start $(date -Is)" >> "$LOG"
  python3 scripts/run_w_sweep_freeze_safe.py $COMMON \
    --w-values "$W" --runs 3 --mask-mode weighted --outdir "$OUT" >> "$LOG" 2>&1
  echo "[driver-v2] T4 W=$W done rc=$? $(date -Is)" >> "$LOG"
}

run_t5 () {
  MODE=$1; R=$2
  OUT=$BASE/t5_${MODE}_r$R
  if [ -f "$OUT/summary.csv" ]; then
    echo "[driver-v2] T5 $MODE r$R already done, skip $(date -Is)" >> "$LOG"
    return
  fi
  echo "[driver-v2] T5 $MODE r$R start $(date -Is)" >> "$LOG"
  python3 scripts/run_w_sweep_freeze_safe.py $COMMON \
    --w-values 50 --runs 1 --mask-mode "$MODE" --outdir "$OUT" >> "$LOG" 2>&1
  echo "[driver-v2] T5 $MODE r$R done rc=$? $(date -Is)" >> "$LOG"
}

for W in 50 30 130; do run_t4 "$W"; done
for R in 0 1 2; do run_t5 weighted "$R"; run_t5 unit "$R"; done
for W in 90 70 110 150; do run_t4 "$W"; done

echo "[driver-v2] ALL DONE $(date -Is)" >> "$LOG"
