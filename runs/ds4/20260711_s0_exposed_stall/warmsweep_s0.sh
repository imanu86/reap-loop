#!/bin/bash
# S0 warm sweep in ONE wsl session (warmth persists only within a session on WSL2).
# Warms the keep-set into RAM page-cache (K23 covers K12 since K12 keeps subset K23),
# then measures, WARM, across cache sizes + masks to test whether raising VRAM hit
# above 0 beats the warm hit0 baseline. If more residency does NOT raise t/s, the
# H2D expert-load copy is NOT the exposed bottleneck at warm-RAM => async has little to hide.
set -u
BASE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_s0_exposed_stall
PROMPT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/prompt_cyberpunk_wide.txt
BIN=/root/ds4_pin/ds4
MODEL=/root/models/ds4-2bit.gguf
K23=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_local_clean_lowK/masks/sessK23.txt
K12=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_local_clean_lowK/masks/sessK12.txt
PROG="$BASE/warmsweep_progress.log"
RES="$BASE/warmsweep_results.log"
: > "$PROG"; : > "$RES"

export DS4_LOCK_FILE=/tmp/ds4_s0_stall.lock
export DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_SPEX_STATS=1 DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1
unset DS4_PACE_PIN DS4_SPEX_PREFETCH_NEXT_LAYER DS4_SPEX_HIDDEN_GPU_PREFETCH DS4_SPEX_FILE DS4_SELECTED_UPLOAD_EVENT

# background GPU/mem sampler
( echo "iso,gpu_used_mb,buffcache_mb" > "$BASE/warmsweep_mem.log"
  while true; do
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1)
    bc=$(free -m|awk '/Mem:/{print $6}')
    echo "$(date -Is),$g,$bc" >> "$BASE/warmsweep_mem.log"; sleep 2
  done ) & SPID=$!

run () { # LABEL MASK CACHE NTOK
  local L=$1 M=$2 C=$3 N=$4; local D="$BASE/$L"; mkdir -p "$D"
  export DS4_REAP_MASK_FILE=$M
  local bc0=$(free -m|awk '/Mem:/{print $6}')
  echo "[$(date +%H:%M:%S)] START $L mask=$(basename $M) cache=$C ntok=$N buffcache=${bc0}MB" | tee -a "$PROG"
  local T0=$(date +%s)
  timeout 800 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$C" -c 2048 --nothink --temp 0.0 -n "$N" \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  local RC=$?; local T1=$(date +%s)
  local pk=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m+0}' "$BASE/warmsweep_mem.log")
  local perf=$(grep -oiE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt"|tail -1)
  local spex=$(grep -oE "hit_rate=[0-9.]+ .*sync_ms_per_batch=[0-9.]+" "$D/diag.txt"|tail -1)
  local hid=$(grep -oE "ds4: SPEX hidden-gpu prefetch stats.*" "$D/diag.txt"|tail -1)
  local md5=$(md5sum "$D/gen.txt"|cut -c1-12)
  echo "[$(date +%H:%M:%S)] DONE  $L rc=$RC wall=$((T1-T0))s peakGPU=${pk}MB md5=$md5 [$perf]" | tee -a "$PROG"
  { echo "== $L cache=$C mask=$(basename $M) ntok=$N rc=$RC wall=$((T1-T0))s peakGPU=${pk}MB genmd5=$md5"
    echo "   [$perf]"; echo "   SPEX hit/copy/sync: $spex"; echo "   HID(prefetch): ${hid:-<empty=>overlap machinery inactive>}"; } >> "$RES"
}

# 1) warmup (discard) — loads keep-set into RAM cache; K23 covers K12
run WS_warmup_K23_c32 "$K23" 32 90
# 2) warm measured sweep
run WS_K23_c32  "$K23" 32  140
run WS_K23_c256 "$K23" 256 140
run WS_K23_c516 "$K23" 516 140
run WS_K12_c256 "$K12" 256 140
run WS_K12_c516 "$K12" 516 140

kill "$SPID" 2>/dev/null
echo "[$(date +%H:%M:%S)] SWEEP COMPLETE" | tee -a "$PROG"
