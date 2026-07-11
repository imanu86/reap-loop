#!/bin/bash
# FIT=SPEED + real fit-boundary sweep on local 3060 (warm RAM, cyberpunk, correct reserve).
# Triangulates the real VRAM slot boundary: over-request cache (c900) on 3 masks of
# increasing working-set (K12~480, K16~640, K23~920); resident = hit_rate x working_set.
# Also measures t/s vs hit to test whether residency buys speed locally.
set -u
BASE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_token_breakdown/fit_sweep
mkdir -p "$BASE"
PROMPT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/prompt_cyberpunk_wide.txt
BIN=/root/ds4_pin/ds4
MODEL=/root/models/ds4-2bit.gguf
MDIR=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_local_clean_lowK/masks
PROG="$BASE/progress.log"; RES="$BASE/results.log"; : > "$PROG"; : > "$RES"

export DS4_LOCK_FILE=/tmp/ds4_fitsweep.lock
export DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_SPEX_STATS=1
unset DS4_PACE_PIN DS4_SPEX_PREFETCH_NEXT_LAYER DS4_SPEX_HIDDEN_GPU_PREFETCH DS4_SPEX_FILE DS4_SELECTED_UPLOAD_EVENT
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }

( echo "iso,gpu_used_mb" > "$BASE/mem.log"
  while true; do echo "$(date -Is),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1)" >> "$BASE/mem.log"; sleep 1; done ) & SPID=$!

run(){ # LABEL MASKNAME CACHE NTOK
  local L=$1; local MN=$2; local C=$3; local N=$4; local D="$BASE/$L"; mkdir -p "$D"
  export DS4_REAP_MASK_FILE=$MDIR/$MN.txt
  : > "$BASE/mem.log"; echo "iso,gpu_used_mb" > "$BASE/mem.log"
  log "START $L mask=$MN cache=$C n=$N"
  local T0=$(date +%s)
  timeout 900 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$C" -c 2048 --nothink --temp 0.0 -n "$N" \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  local RC=$?; local T1=$(date +%s)
  local pk=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m+0}' "$BASE/mem.log")
  local perf=$(grep -oiE "generation: [0-9.]+ t/s" "$D/diag.txt"|tail -1)
  local hit=$(grep -oE "hit_rate=[0-9.]+" "$D/diag.txt"|tail -1)
  local sel=$(grep -oE "selected_experts=[0-9]+ cache_hits=[0-9]+ cache_misses=[0-9]+" "$D/diag.txt"|tail -1)
  local sync=$(grep -oE "sync_ms_per_batch=[0-9.]+" "$D/diag.txt"|tail -1)
  local oom=$(grep -ciE "out of memory|oom|cudaError|failed" "$D/diag.txt")
  log "DONE $L rc=$RC wall=$((T1-T0))s peakGPU=${pk}MB $perf $hit sync=$sync oomflags=$oom"
  { echo "== $L mask=$MN cache=$C n=$N rc=$RC wall=$((T1-T0))s peakGPU=${pk}MB"
    echo "   $perf | $hit | $sel | $sync | oomflags=$oom"; } >> "$RES"
}

log "=== WARMUP (discard) ==="
run WARMUP sessK23 32 60
log "=== baseline hit0 ==="
run K23_c32   sessK23 32   90
log "=== over-request c900 (triangulate boundary) ==="
run K12_c900  sessK12 900  90
run K16_c900  sessK16 900  90
run K23_c900  sessK23 900  90
log "=== K12 exact-fit attempt c516 ==="
run K12_c516  sessK12 516  90
kill "$SPID" 2>/dev/null
log "=== SWEEP DONE ==="
