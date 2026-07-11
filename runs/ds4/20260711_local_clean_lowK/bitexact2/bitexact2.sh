#!/bin/bash
# Pivoted bit-exact + speed probe per coordinator instructions:
# reserve=1 FIXED (parsing-safe), vary cache size 1024 (hit-high, contains K12
# working set) vs 32 (hit-low, near-total miss). 2x2 with/without
# DS4_CUDA_NO_Q8_F16_CACHE. Plus K23@cache1024 for the K12-vs-K23 speed head-to-head.
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUT=$REPO/runs/ds4/20260711_local_clean_lowK/bitexact2
PROMPT=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local/t4_W050/W050/r00/p2prompt.txt
LOG=$OUT/bitexact2.log
mkdir -p "$OUT"

export DS4_LOCK_FILE=/tmp/ds4_clean_lowK.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1
export DS4_SPEX_STATS=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1

run_one () {
  NAME=$1
  K=$2
  CACHE=$3
  Q8=$4      # 1 or empty
  D="$OUT/$NAME"
  mkdir -p "$D"
  if [ -n "$Q8" ]; then
    export DS4_CUDA_NO_Q8_F16_CACHE=1
  else
    unset DS4_CUDA_NO_Q8_F16_CACHE
  fi
  export DS4_REAP_MASK_FILE=$REPO/runs/ds4/20260711_local_clean_lowK/masks/sessK$K.txt
  echo "[bx2] START $NAME K=$K cache=$CACHE q8_uniform=${Q8:-0} $(date -Is)" >> "$LOG"
  T0=$(date +%s)
  timeout 1800 /root/ds4/ds4 -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$CACHE" -c 4096 --nothink --temp 0.0 -n 300 \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  RC=$?
  T1=$(date +%s)
  CAPLINE=$(grep -E "streaming expert cache (capped|disabled)" "$D/diag.txt" | tail -1)
  SPEXLINE=$(grep -E "SPEX stats:" "$D/diag.txt" | tail -1)
  PF=$(grep -o "prefill: [0-9.]* t/s, generation: [0-9.]* t/s" "$D/diag.txt" | tail -1)
  echo "[bx2] DONE $NAME rc=$RC wall=$((T1-T0))s cap=[$CAPLINE] spex=[$SPEXLINE] perf=[$PF] $(date -Is)" >> "$LOG"
}

run_one "K12_cache1024_q8off" 12 1024 1
run_one "K12_cache32_q8off"   12 32   1
run_one "K12_cache1024_q8on"  12 1024 ""
run_one "K12_cache32_q8on"    12 32   ""
run_one "K23_cache1024_q8off" 23 1024 1

echo "[bx2] === DIFFS ===" >> "$LOG"
diff "$OUT/K12_cache1024_q8off/gen.txt" "$OUT/K12_cache32_q8off/gen.txt" > "$OUT/diff_q8off_hi_vs_lo.txt"
echo "q8off (uniform 2bit cache) hit-HIGH(1024) vs hit-LOW(32): $(wc -l < "$OUT/diff_q8off_hi_vs_lo.txt") diff lines" >> "$LOG"
diff "$OUT/K12_cache1024_q8on/gen.txt" "$OUT/K12_cache32_q8on/gen.txt" > "$OUT/diff_q8on_hi_vs_lo.txt"
echo "q8on (q8/f16 cache active) hit-HIGH(1024) vs hit-LOW(32): $(wc -l < "$OUT/diff_q8on_hi_vs_lo.txt") diff lines" >> "$LOG"

echo "BITEXACT2_COMPLETE" >> "$LOG"
