#!/bin/bash
# Generic single measured ds4 run (clean cache32 regime). Args:
#   $1 NAME  $2 K(0=no mask)  $3 NTOK  $4 CTX  $5 PROMPTFILE  $6 OUTDIR  [$7 TRACE(1=emit route csv)]
# Appends a one-line result to $OUTDIR/progress.log and samples mem/swap/gpu to $OUTDIR/$NAME/mem.log
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
RUN=$REPO/runs/ds4/20260711_highK_sweetspot
MASKS=$RUN/masks
NAME=$1; K=$2; NTOK=$3; CTX=$4; PROMPT=$5; OUTDIR=$6; TRACE=${7:-0}
LOG=$OUTDIR/progress.log
D="$OUTDIR/$NAME"; mkdir -p "$D"

export DS4_LOCK_FILE=/tmp/ds4_highK_sweetspot.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_SPEX_STATS=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
if [ "$K" = "0" ]; then unset DS4_REAP_MASK_FILE; else export DS4_REAP_MASK_FILE=$MASKS/sessK$K.txt; fi
if [ "$TRACE" = "1" ]; then export DS4_SPEX_TRACE_ROUTING="$D/route.csv"; export DS4_SPEX_TRACE_ROUTING_WEIGHTS=1; else unset DS4_SPEX_TRACE_ROUTING; fi

BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf

( echo "iso,mem_used_mb,swap_used_mb,gpu_used_mb" > "$D/mem.log"
  while true; do
    mu=$(free -m | awk '/^Mem:/{print $3}')
    sw=$(free -m | awk '/^Swap:/{print $3}')
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "$(date -Is),$mu,$sw,$g" >> "$D/mem.log"; sleep 10
  done ) &
SPID=$!

echo "[gen] START $NAME K=$K ntok=$NTOK ctx=$CTX trace=$TRACE $(date -Is)" >> "$LOG"
T0=$(date +%s)
timeout 7200 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 32 -c "$CTX" --nothink --temp 0.0 -n "$NTOK" \
  --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
RC=$?
T1=$(date +%s)
kill "$SPID" 2>/dev/null
PF=$(grep -oE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt" | tail -1)
PKSWAP=$(awk -F, 'NR>1{if($3>m)m=$3}END{print m+0}' "$D/mem.log")
PKMEM=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m+0}' "$D/mem.log")
CHARS=$(wc -c < "$D/gen.txt")
CLOSE=$(grep -c "</html>" "$D/gen.txt")
ROWS=$( [ -f "$D/route.csv" ] && wc -l < "$D/route.csv" || echo 0 )
echo "[gen] DONE $NAME rc=$RC wall=$((T1-T0))s perf=[$PF] peakMemMB=$PKMEM peakSwapMB=$PKSWAP gen_chars=$CHARS close_html=$CLOSE route_rows=$ROWS $(date -Is)" >> "$LOG"
