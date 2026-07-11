#!/bin/bash
# Per-token time breakdown of ds4 decode on local RTX 3060 (sm_86, WSL2).
# Goal: decompose the ~296 ms/token (warm K23 cache32, cyberpunk-wide, 3.38 t/s) into
#   (a) GPU compute (kernel), (b) H2D expert copy (exposed vs hidden behind compute),
#   (c) blocking-sync, (d) CPU/launch/orchestration overhead (= wall - GPU-busy union).
# Method: nsys CUDA trace (kernel+memcpy timeline) + differential (n=40 vs n=120 cancels
#   model-load/prefill) + a clean non-nsys wall/SPEX run for the honest wall time.
# Vaccinations: own lock (UI:8000 untouched, CLI binds no port), clean 2-bit path,
#   warm-RAM levers, reserve parse-safe (=1 NOT 16). GPU exclusive (only job).
set -u
BASE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_token_breakdown
PROMPT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/prompt_cyberpunk_wide.txt
BIN=/root/ds4_pin/ds4
MODEL=/root/models/ds4-2bit.gguf
K23=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_local_clean_lowK/masks/sessK23.txt
PROG="$BASE/progress.log"
: > "$PROG"

# --- clean config (vaccinations) ---
export DS4_LOCK_FILE=/tmp/ds4_tokbreak.lock
export DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_REAP_MASK_FILE=$K23
export DS4_SPEX_STATS=1 DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1
unset DS4_PACE_PIN DS4_SPEX_PREFETCH_NEXT_LAYER DS4_SPEX_HIDDEN_GPU_PREFETCH DS4_SPEX_FILE DS4_SELECTED_UPLOAD_EVENT

CACHE=32; CTX=2048
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }

# background GPU/mem sampler
( echo "iso,gpu_used_mb,buffcache_mb" > "$BASE/mem.log"
  while true; do
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1)
    bc=$(free -m|awk '/Mem:/{print $6}')
    echo "$(date -Is),$g,$bc" >> "$BASE/mem.log"; sleep 2
  done ) & SPID=$!

clean_run(){ # LABEL NTOK
  local L=$1; local N=$2; local D="$BASE/$L"; mkdir -p "$D"
  log "START clean $L n=$N buffcache=$(free -m|awk '/Mem:/{print $6}')MB"
  local T0=$(date +%s.%N)
  timeout 900 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$CACHE" -c "$CTX" --nothink --temp 0.0 -n "$N" \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  local RC=$?; local T1=$(date +%s.%N)
  local wall=$(echo "$T1 - $T0"|bc)
  local perf=$(grep -oiE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt"|tail -1)
  local spex=$(grep -oE "hit_rate=[0-9.]+ .*sync_ms_per_batch=[0-9.]+" "$D/diag.txt"|tail -1)
  local md5=$(md5sum "$D/gen.txt"|cut -c1-12)
  log "DONE  clean $L rc=$RC wall=${wall}s md5=$md5 [$perf]"
  echo "   SPEX: $spex" | tee -a "$PROG"
}

nsys_run(){ # LABEL NTOK
  local L=$1; local N=$2; local D="$BASE/$L"; mkdir -p "$D"
  log "START nsys $L n=$N"
  local T0=$(date +%s.%N)
  timeout 1200 nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$D/trace" \
    "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$CACHE" -c "$CTX" --nothink --temp 0.0 -n "$N" \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  local RC=$?; local T1=$(date +%s.%N)
  local wall=$(echo "$T1 - $T0"|bc)
  local perf=$(grep -oiE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt"|tail -1)
  log "DONE  nsys $L rc=$RC wall=${wall}s [$perf] (nsys wall is inflated; use for GPU-busy only)"
  log "EXPORT sqlite $L ..."
  nsys export --type sqlite --force-overwrite=true -o "$D/trace.sqlite" "$D/trace.nsys-rep" >>"$D/diag.txt" 2>&1
  log "EXPORT done $L ($(ls -la $D/trace.sqlite 2>/dev/null|awk '{print $5}') bytes)"
}

log "=== WARMUP (discard) ==="
clean_run WARMUP 60
log "=== CLEAN wall/SPEX reps ==="
clean_run CLEAN_A 120
clean_run CLEAN_B 120
log "=== NSYS differential ==="
nsys_run NSYS_n40 40
nsys_run NSYS_n120 120

kill "$SPID" 2>/dev/null
log "=== ALL DONE ==="
