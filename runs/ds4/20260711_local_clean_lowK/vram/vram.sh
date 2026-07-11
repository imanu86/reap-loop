#!/bin/bash
# Quick VRAM-pressure check: short runs (n=60) at cache=32 vs cache=1024,
# polling nvidia-smi memory.used every 2s in parallel, to see actual peak VRAM
# (not just the startup "available/reserve" accounting).
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUT=$REPO/runs/ds4/20260711_local_clean_lowK/vram
PROMPT=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local/t4_W050/W050/r00/p2prompt.txt
LOG=$OUT/vram.log
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
  D="$OUT/$NAME"
  mkdir -p "$D"
  echo "[vram] START $NAME cache=$CACHE $(date -Is)" >> "$LOG"
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$D/vram_trace.csv"
      sleep 2
    done ) &
  POLLER=$!
  T0=$(date +%s)
  timeout 900 /root/ds4/ds4 -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$CACHE" -c 4096 --nothink --temp 0.0 -n 60 \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  RC=$?
  T1=$(date +%s)
  kill "$POLLER" 2>/dev/null
  PEAK=$(sort -n "$D/vram_trace.csv" | tail -1)
  PF=$(grep -o "prefill: [0-9.]* t/s, generation: [0-9.]* t/s" "$D/diag.txt" | tail -1)
  CAPLINE=$(grep -E "streaming expert cache (capped|disabled)" "$D/diag.txt" | tail -1)
  echo "[vram] DONE $NAME rc=$RC wall=$((T1-T0))s peak_vram_mib=$PEAK cap=[$CAPLINE] perf=[$PF] $(date -Is)" >> "$LOG"
}

run_one "v_cache32"   32
run_one "v_cache516"  516
run_one "v_cache1024" 1024

echo "VRAM_COMPLETE" >> "$LOG"
