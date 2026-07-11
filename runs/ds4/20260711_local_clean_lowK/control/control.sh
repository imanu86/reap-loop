#!/bin/bash
# Coordinator-requested CONTROL: repeat the exact same config twice to measure
# baseline non-determinism from async prefetch races (independent of cache size).
# Bonus: try DS4_REAP_PREFETCH=0 to see if it restores determinism.
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUT=$REPO/runs/ds4/20260711_local_clean_lowK/control
PROMPT=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local/t4_W050/W050/r00/p2prompt.txt
LOG=$OUT/control.log
mkdir -p "$OUT"

export DS4_LOCK_FILE=/tmp/ds4_clean_lowK.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1
export DS4_SPEX_STATS=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_REAP_MASK_FILE=$REPO/runs/ds4/20260711_local_clean_lowK/masks/sessK12.txt

run_one () {
  NAME=$1
  CACHE=$2
  NOPREFETCH=$3  # 1 or empty
  D="$OUT/$NAME"
  mkdir -p "$D"
  if [ -n "$NOPREFETCH" ]; then
    export DS4_REAP_PREFETCH=0
  else
    unset DS4_REAP_PREFETCH
  fi
  echo "[ctrl] START $NAME cache=$CACHE prefetch_off=${NOPREFETCH:-0} $(date -Is)" >> "$LOG"
  T0=$(date +%s)
  timeout 1800 /root/ds4/ds4 -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$CACHE" -c 4096 --nothink --temp 0.0 -n 300 \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  RC=$?
  T1=$(date +%s)
  PF=$(grep -o "prefill: [0-9.]* t/s, generation: [0-9.]* t/s" "$D/diag.txt" | tail -1)
  echo "[ctrl] DONE $NAME rc=$RC wall=$((T1-T0))s perf=[$PF] $(date -Is)" >> "$LOG"
}

# control pair: identical config (cache32, prefetch default-on) x2 — cache32
# chosen over 1024 purely for wall-clock budget (same fixed ~5min warmup tax,
# faster decode), still directly tests the SAME async-prefetch non-determinism
# hypothesis regardless of cache size.
run_one "ctrl_32_a" 32 ""
run_one "ctrl_32_b" 32 ""

echo "[ctrl] === DIFFS ===" >> "$LOG"
diff "$OUT/ctrl_32_a/gen.txt" "$OUT/ctrl_32_b/gen.txt" > "$OUT/diff_ctrl_32_a_vs_b.txt"
FIRSTDIFF=$(diff "$OUT/ctrl_32_a/gen.txt" "$OUT/ctrl_32_b/gen.txt" | head -1)
echo "CONTROL identical-config cache32 A vs B: $(wc -l < "$OUT/diff_ctrl_32_a_vs_b.txt") diff lines. first=[$FIRSTDIFF]" >> "$LOG"

echo "CONTROL_COMPLETE" >> "$LOG"
