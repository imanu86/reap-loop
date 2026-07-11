#!/bin/bash
# FASE 3 — 0031 pin-keep VRAM-residency velocity measurement on local 3060 (sm_86).
# Coexists with any UI via its OWN lock; never touches UI:8000 or the calibration lock.
# Args: LABEL PIN CACHE_EXPERTS RESERVE_GB BUDGET_MB WARMUP ROTATE NTOK CTX
set -u
LABEL=$1; PIN=$2; CACHE=$3; RESGB=$4; BUDGET=$5; WARMUP=$6; ROTATE=$7; NTOK=$8; CTX=$9
BASE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_0031_local_velocity
D="$BASE/$LABEL"; mkdir -p "$D"
MASK=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_domain_calibration/masks/sessCyber_K23.txt
PROMPT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/prompt_cyberpunk_wide.txt
BIN=/root/ds4_pin/ds4
MODEL=/root/models/ds4-2bit.gguf

# --- config (vaccinations): own lock, clean 2-bit cache path, warm RAM lever, greedy ---
export DS4_LOCK_FILE=/tmp/ds4_pin_velocity.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_SPEX_STATS=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=$RESGB
export DS4_REAP_MASK_FILE=$MASK
# --- 0031 pin knobs ---
export DS4_PACE_PIN=$PIN
export DS4_PACE_PIN_BUDGET_MB=$BUDGET
export DS4_PACE_PIN_WARMUP=$WARMUP
export DS4_PACE_PIN_ROTATE=$ROTATE
export DS4_PACE_PIN_LOG="$D/pin.jsonl"

# VRAM/mem sampler
( echo "iso,gpu_used_mb,mem_used_mb" > "$D/mem.log"
  while true; do
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    mu=$(free -m | awk '/^Mem:/{print $3}')
    echo "$(date -Is),$g,$mu" >> "$D/mem.log"; sleep 3
  done ) &
SPID=$!

echo "[measure] START $LABEL PIN=$PIN cache=$CACHE reserveGB=$RESGB budgetMB=$BUDGET warmup=$WARMUP rotate=$ROTATE ntok=$NTOK ctx=$CTX $(date -Is)"
T0=$(date +%s)
timeout 1200 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts "$CACHE" -c "$CTX" --nothink --temp 0.0 -n "$NTOK" \
  --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
RC=$?
T1=$(date +%s)
kill "$SPID" 2>/dev/null

PKGPU=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m+0}' "$D/mem.log")
PF=$(grep -oiE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt" | tail -1)
HIT=$(grep -oiE "hit[_ ]?rate[^0-9]*[0-9.]+" "$D/diag.txt" | tail -1)
CHARS=$(wc -c < "$D/gen.txt")
FREEZE=$(grep -c pin_freeze "$D/pin.jsonl" 2>/dev/null || echo 0)
ROT=$(grep -c pin_rotate "$D/pin.jsonl" 2>/dev/null || echo 0)
GENHASH=$(md5sum "$D/gen.txt" | cut -c1-12)
echo "[measure] DONE  $LABEL rc=$RC wall=$((T1-T0))s peakGPU_MB=$PKGPU perf=[$PF] hit=[$HIT] gen_chars=$CHARS pin_freeze=$FREEZE pin_rotate=$ROT genmd5=$GENHASH $(date -Is)"
echo "$LABEL rc=$RC wall=$((T1-T0))s peakGPU_MB=$PKGPU perf=[$PF] hit=[$HIT] gen_chars=$CHARS pin_freeze=$FREEZE pin_rotate=$ROT genmd5=$GENHASH" >> "$BASE/results.log"
