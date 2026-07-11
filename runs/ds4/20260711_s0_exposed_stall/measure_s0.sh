#!/bin/bash
# S0 — exposed expert-load-stall baseline on local 3060 (sm_86).
# Measures, per regime, the DS4_SPEX_STATS copy_ms/sync_ms breakdown (the exposed
# H2D expert-load cost) + hidden-prefetch stats (must be ~0 => overlap==0), to
# decide if the async prefetch pipeline has any stall to hide.
#
# NO prefetch, NO pin (S0 = stage-0 counters only, 0001). Reactive load path only.
# Coexists with UI:8000 via its OWN lock. Warm-RAM lever. Clean 2-bit cache path.
# Args: LABEL MASK CACHE NTOK CTX
set -u
LABEL=$1; MASK=$2; CACHE=$3; NTOK=$4; CTX=$5
BASE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_s0_exposed_stall
D="$BASE/$LABEL"; mkdir -p "$D"
PROMPT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/prompt_cyberpunk_wide.txt
BIN=/root/ds4_pin/ds4
MODEL=/root/models/ds4-2bit.gguf

# --- config (vaccinations): own lock, clean 2-bit cache, warm RAM, greedy, reserve parse-safe ---
export DS4_LOCK_FILE=/tmp/ds4_s0_stall.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_REAP_MASK_FILE=$MASK
# --- S0 instrumentation: stage-0 counters + hidden-prefetch stats (prefetch OFF) ---
export DS4_SPEX_STATS=1
export DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1
# --- explicitly NO pin, NO prefetch ---
unset DS4_PACE_PIN DS4_SPEX_PREFETCH_NEXT_LAYER DS4_SPEX_HIDDEN_GPU_PREFETCH DS4_SPEX_FILE DS4_SELECTED_UPLOAD_EVENT

# VRAM/mem sampler (2s)
( echo "iso,gpu_used_mb,mem_used_mb" > "$D/mem.log"
  while true; do
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    mu=$(free -m | awk '/^Mem:/{print $3}')
    echo "$(date -Is),$g,$mu" >> "$D/mem.log"; sleep 2
  done ) &
SPID=$!

echo "[s0] START $LABEL mask=$(basename $MASK) cache=$CACHE ntok=$NTOK ctx=$CTX $(date -Is)"
T0=$(date +%s)
timeout 900 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts "$CACHE" -c "$CTX" --nothink --temp 0.0 -n "$NTOK" \
  --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
RC=$?
T1=$(date +%s)
kill "$SPID" 2>/dev/null

PKGPU=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m+0}' "$D/mem.log")
PF=$(grep -oiE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt" | tail -1)
SPEX=$(grep -oE "ds4: SPEX stats:.*" "$D/diag.txt" | tail -1)
HID=$(grep -oE "ds4: SPEX hidden-gpu prefetch stats.*" "$D/diag.txt" | tail -1)
CHARS=$(wc -c < "$D/gen.txt")
GENHASH=$(md5sum "$D/gen.txt" | cut -c1-12)
echo "[s0] DONE  $LABEL rc=$RC wall=$((T1-T0))s peakGPU_MB=$PKGPU gen_chars=$CHARS genmd5=$GENHASH $(date -Is)"
echo "  perf=[$PF]"
echo "  SPEX =[$SPEX]"
echo "  HID  =[$HID]"
{
  echo "== $LABEL rc=$RC wall=$((T1-T0))s peakGPU_MB=$PKGPU gen_chars=$CHARS genmd5=$GENHASH"
  echo "   mask=$MASK cache=$CACHE ntok=$NTOK ctx=$CTX"
  echo "   perf=[$PF]"
  echo "   SPEX=[$SPEX]"
  echo "   HID =[$HID]"
} >> "$BASE/results.log"
