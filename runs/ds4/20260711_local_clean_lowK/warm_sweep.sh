#!/bin/bash
# Warm-controlled velocity-vs-K at cache32, cyberpunk prompt. Resolves the cold-start
# confound: the batch's K16/K23/K38 happened to run warm (~3.1 t/s) while K12 ran cold
# (1.77). Here: one discard warm-up, then K12/K16/K23/K38 measured BACK-TO-BACK so the
# 86GB model's expert pages stay hot for all four -> apples-to-apples K comparison at
# fixed warmth. No routing trace (pure speed; routes already emitted, cache/warmth-indep).
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
MASKS=$REPO/runs/ds4/20260711_local_clean_lowK/masks
WARM=$REPO/runs/ds4/20260711_local_clean_lowK/curve/warm
PROMPT=$REPO/runs/ds4/20260711_masked_route_traces/prompt_cyberpunk_wide.txt
LOG=$WARM/warm_progress.log
mkdir -p "$WARM"

export DS4_LOCK_FILE=/tmp/ds4_clean_lowK_curve.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_SPEX_STATS=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1

run_one () {
  NAME=$1; K=$2
  D="$WARM/$NAME"; mkdir -p "$D"
  export DS4_REAP_MASK_FILE=$MASKS/sessK$K.txt
  echo "[warm] START $NAME K=$K $(date -Is)" >> "$LOG"
  T0=$(date +%s)
  timeout 1800 /root/ds4/ds4 -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts 32 -c 4096 --nothink --temp 0.0 -n 300 \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  RC=$?
  T1=$(date +%s)
  PF=$(grep -oE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt" | tail -1)
  echo "[warm] DONE $NAME rc=$RC wall=$((T1-T0))s perf=[$PF] $(date -Is)" >> "$LOG"
}

run_one "warmup_K12" 12      # discard (cold prime)
run_one "w_K12" 12
run_one "w_K16" 16
run_one "w_K23" 23
run_one "w_K38" 38
echo "[warm] WARM_COMPLETE $(date -Is)" >> "$LOG"
