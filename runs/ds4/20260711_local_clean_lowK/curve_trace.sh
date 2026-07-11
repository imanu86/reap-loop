#!/bin/bash
# Clean velocity-vs-K curve @ cache32 (+ K23@cache516) with REAL masked route traces.
# Continues aec293's clean probe (bitexact2.sh): reserve=1 (parse-safe, avoids 256->154
# default-cap and the reserve=16 abort), DS4_CUDA_NO_Q8_F16_CACHE=1 (clean 2-bit path),
# --ssd-streaming-cold, greedy temp0, ctx4096, -n 300, cache-experts=32 (fastest config).
# ONE run per K yields BOTH the gen t/s (curve) AND the post-mask route.csv (verified
# post-mask: 0/587760 picks land on a pruned expert on the reference cyber K12 run).
# Route CSV staged on native ext4 (avoids 9p per-token latency contaminating t/s) then
# copied to the repo deliverable dir. Frozen REAP mask active => e0..e5 are the REAL
# post-mask selected experts (no proxy). Distro Ubuntu-24.04, /root/ds4/ds4.
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
MASKS=$REPO/runs/ds4/20260711_local_clean_lowK/masks
CURVE=$REPO/runs/ds4/20260711_local_clean_lowK/curve
TRACES=$REPO/runs/ds4/20260711_masked_route_traces
PROMPT_CYBER=$TRACES/prompt_cyberpunk_wide.txt
PROMPT_COFFEE=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local/t4_W050/W050/r00/p2prompt.txt
STAGE=/root/clean_lowK_curve_stage
LOG=$CURVE/curve_trace_progress.log
mkdir -p "$CURVE" "$TRACES" "$STAGE"

export DS4_LOCK_FILE=/tmp/ds4_clean_lowK_curve.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1
export DS4_SPEX_STATS=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_SPEX_TRACE_ROUTING_WEIGHTS=1

run_one () {
  NAME=$1; K=$2; CACHE=$3; PROMPT=$4; LABEL=$5
  D="$CURVE/$NAME"; mkdir -p "$D"
  export DS4_REAP_MASK_FILE=$MASKS/sessK$K.txt
  export DS4_SPEX_TRACE_ROUTING=$STAGE/route_masked_K${K}_${LABEL}.csv
  rm -f "$DS4_SPEX_TRACE_ROUTING"
  echo "[curve] START $NAME K=$K cache=$CACHE label=$LABEL $(date -Is)" >> "$LOG"
  T0=$(date +%s)
  timeout 1800 /root/ds4/ds4 -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts "$CACHE" -c 4096 --nothink --temp 0.0 -n 300 \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  RC=$?
  T1=$(date +%s)
  DEST=$TRACES/route_masked_K${K}_${LABEL}.csv
  ROWS=0
  if [ -s "$DS4_SPEX_TRACE_ROUTING" ]; then cp "$DS4_SPEX_TRACE_ROUTING" "$DEST"; ROWS=$(wc -l < "$DEST"); fi
  CAP=$(grep -E "streaming expert cache (capped|disabled)" "$D/diag.txt" | tail -1)
  PF=$(grep -oE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt" | tail -1)
  HR=$(grep -oE "hit_rate=[0-9.]+" "$D/diag.txt" | tail -1)
  echo "[curve] DONE $NAME rc=$RC wall=$((T1-T0))s perf=[$PF] $HR cap=[$CAP] route_rows=$ROWS -> $DEST $(date -Is)" >> "$LOG"
}

# Front-load the two most important cells (K12, K23 @ cache32) for phase-seg + curve anchor.
run_one "K12_cache32_cyber"  12 32  "$PROMPT_CYBER"  cyberpunk
run_one "K23_cache32_cyber"  23 32  "$PROMPT_CYBER"  cyberpunk
run_one "K16_cache32_cyber"  16 32  "$PROMPT_CYBER"  cyberpunk
run_one "K38_cache32_cyber"  38 32  "$PROMPT_CYBER"  cyberpunk
run_one "K23_cache516_cyber" 23 516 "$PROMPT_CYBER"  cyberpunk
echo "[curve] CORE_COMPLETE $(date -Is)" >> "$LOG"
# Contrast: coffee-shop compact (narrow) at the two anchor Ks (budget permitting).
run_one "K12_cache32_coffee" 12 32  "$PROMPT_COFFEE" coffee
run_one "K23_cache32_coffee" 23 32  "$PROMPT_COFFEE" coffee
echo "[curve] ALL_COMPLETE $(date -Is)" >> "$LOG"
