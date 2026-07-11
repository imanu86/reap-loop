#!/bin/bash
# Single measured ds4 run. Clean cache32 regime, greedy temp0. EXACT config of d046a71 (domain_calibration).
# Args: $1 NAME  $2 MASKPATH(none=full)  $3 NTOK  $4 CTX  $5 PROMPTFILE  $6 OUTDIR  [$7 TRACE(1=emit route csv)]
set -u
NAME=$1; MASK=$2; NTOK=$3; CTX=$4; PROMPT=$5; OUTDIR=$6; TRACE=${7:-0}
LOG=$OUTDIR/progress.log
D="$OUTDIR/$NAME"; mkdir -p "$D"
export DS4_LOCK_FILE=/tmp/ds4_k12q.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_SPEX_STATS=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
if [ "$MASK" = "none" ]; then unset DS4_REAP_MASK_FILE; else export DS4_REAP_MASK_FILE=$MASK; fi
if [ "$TRACE" = "1" ]; then export DS4_SPEX_TRACE_ROUTING="$D/route.csv"; export DS4_SPEX_TRACE_ROUTING_WEIGHTS=1; else unset DS4_SPEX_TRACE_ROUTING; fi
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
echo "[run] START $NAME mask=$(basename "$MASK") ntok=$NTOK ctx=$CTX trace=$TRACE $(date -Is)" >> "$LOG"
T0=$(date +%s)
timeout 5400 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 32 -c "$CTX" --nothink --temp 0.0 -n "$NTOK" \
  --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
RC=$?
T1=$(date +%s)
PF=$(grep -oE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt" | tail -1)
CHARS=$(wc -c < "$D/gen.txt")
BODY=$(grep -c "<body" "$D/gen.txt")
CLOSE=$(grep -c "</html>" "$D/gen.txt")
ROWS=$( [ -f "$D/route.csv" ] && wc -l < "$D/route.csv" || echo 0 )
echo "[run] DONE $NAME rc=$RC wall=$((T1-T0))s perf=[$PF] gen_chars=$CHARS body=$BODY close_html=$CLOSE route_rows=$ROWS $(date -Is)" >> "$LOG"
echo "DONE_$NAME rc=$RC" > "$D/done"
