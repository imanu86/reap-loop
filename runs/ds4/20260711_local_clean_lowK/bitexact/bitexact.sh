#!/bin/bash
# Priority-zero correctness probe: does the SSD-streaming expert cache change the
# generated tokens (greedy, temp0) vs cache-off (direct-load every expert)?
# 2x2: {NO_Q8_F16_CACHE=1, unset} x {cache-ON reserve=1, cache-OFF reserve=16}
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUT=$REPO/runs/ds4/20260711_local_clean_lowK/bitexact
PROMPT=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local/t4_W050/W050/r00/p2prompt.txt
MASK=$REPO/runs/ds4/20260711_local_clean_lowK/masks/sessK12.txt
LOG=$OUT/bitexact.log
mkdir -p "$OUT"

export DS4_LOCK_FILE=/tmp/ds4_clean_lowK.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1
export DS4_SPEX_STATS=1
export DS4_REAP_MASK_FILE=$MASK

run_one () {
  NAME=$1
  Q8=$2      # 1 or unset(empty string means do not export)
  RESERVE=$3 # 1 (on) or 16 (off/buggy-default)
  D="$OUT/$NAME"
  mkdir -p "$D"
  if [ -n "$Q8" ]; then
    export DS4_CUDA_NO_Q8_F16_CACHE=1
  else
    unset DS4_CUDA_NO_Q8_F16_CACHE
  fi
  export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=$RESERVE
  echo "[bitexact] START $NAME q8_uniform=${Q8:-0} reserve=$RESERVE $(date -Is)" >> "$LOG"
  timeout 1800 /root/ds4/ds4 -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts 256 -c 4096 --nothink --temp 0.0 -n 300 \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  RC=$?
  CAPLINE=$(grep -E "streaming expert cache (capped|disabled)" "$D/diag.txt" | tail -1)
  SPEX=$(grep -E "SPEX stats:" "$D/diag.txt" | tail -1)
  PF=$(grep -o "prefill: [0-9.]* t/s, generation: [0-9.]* t/s" "$D/diag.txt" | tail -1)
  echo "[bitexact] DONE $NAME rc=$RC cap=[$CAPLINE] spex=[$SPEX] perf=[$PF] $(date -Is)" >> "$LOG"
}

run_one "A_q8off_cacheON"  1 1
run_one "A_q8off_cacheOFF" 1 16
run_one "B_q8on_cacheON"   "" 1
run_one "B_q8on_cacheOFF"  "" 16

echo "[bitexact] === DIFFS ===" >> "$LOG"
diff "$OUT/A_q8off_cacheON/gen.txt" "$OUT/A_q8off_cacheOFF/gen.txt" > "$OUT/diff_A_q8off_ON_vs_OFF.txt"
echo "A (NO_Q8_F16_CACHE=1) ON vs OFF: $(wc -l < "$OUT/diff_A_q8off_ON_vs_OFF.txt") diff lines" >> "$LOG"
diff "$OUT/B_q8on_cacheON/gen.txt" "$OUT/B_q8on_cacheOFF/gen.txt" > "$OUT/diff_B_q8on_ON_vs_OFF.txt"
echo "B (q8/f16 cache active) ON vs OFF: $(wc -l < "$OUT/diff_B_q8on_ON_vs_OFF.txt") diff lines" >> "$LOG"

echo "BITEXACT_COMPLETE" >> "$LOG"
