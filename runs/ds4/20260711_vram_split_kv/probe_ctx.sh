#!/bin/bash
# VRAM split probe: isolate fixed non-expert weights (intercept) vs KV+arena (ctx slope)
# by measuring peakGPU at increasing context with a MINIMAL expert cache (32).
# Own lock; never touches UI:8000 or the calibration lock. GPU must be exclusive.
# Arg: CTX
set -u
CTX=$1
BASE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_vram_split_kv
D="$BASE/ctx_$CTX"; mkdir -p "$D"
MASK=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_domain_calibration/masks/sessCyber_K23.txt
PROMPT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/prompt_cyberpunk_wide.txt
BIN=/root/ds4_pin/ds4
MODEL=/root/models/ds4-2bit.gguf

export DS4_LOCK_FILE=/tmp/ds4_vram_split.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_SPEX_STATS=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.5
export DS4_REAP_MASK_FILE=$MASK

# VRAM sampler (fast 1s cadence to catch transient peaks)
( echo "iso,gpu_used_mb" > "$D/mem.log"
  while true; do
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "$(date -Is),$g" >> "$D/mem.log"; sleep 1
  done ) &
SPID=$!

echo "[probe] START ctx=$CTX $(date -Is)"
T0=$(date +%s)
timeout 900 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 32 -c "$CTX" --nothink --temp 0.0 -n 4 \
  --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
RC=$?
T1=$(date +%s)
kill "$SPID" 2>/dev/null

PKGPU=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m+0}' "$D/mem.log")
echo "[probe] DONE ctx=$CTX rc=$RC wall=$((T1-T0))s peakGPU_MB=$PKGPU $(date -Is)"
echo "ctx=$CTX rc=$RC wall=$((T1-T0))s peakGPU_MB=$PKGPU" >> "$BASE/results.log"
