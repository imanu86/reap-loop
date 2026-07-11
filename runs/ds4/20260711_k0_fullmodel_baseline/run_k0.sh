#!/bin/bash
# K0 FULL model, NO mask, NO PACE, clean path, greedy temp0, with TRUE route trace.
# Full router: all 256 experts/layer eligible, native top-6 (unbiased -> trace captures all 256).
set -u
BIN=/root/bin/ds4
MODEL=/root/models/ds4-2bit.gguf
OUT=/root/k0
cd "$OUT"
export DS4_CUDA_NO_Q8_F16_CACHE=1   # clean uniform-2bit cache path

run_one() {  # $1=name $2=promptfile $3=ntok $4=ctx
  local name=$1 prompt=$2 ntok=$3 ctx=$4
  echo "=== START $name ntok=$ntok ctx=$ctx $(date -u +%H:%M:%S) ===" >> "$OUT/progress.log"
  DS4_SPEX_TRACE_ROUTING="$OUT/route_k0_${name}.csv" \
  DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
  "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts 1024 -c "$ctx" --nothink --temp 0 \
    -n "$ntok" --prompt-file "$prompt" \
    > "$OUT/gen_k0_${name}.txt" 2> "$OUT/diag_k0_${name}.txt"
  local rc=$?
  echo "=== END $name rc=$rc gen_chars=$(wc -c < $OUT/gen_k0_${name}.txt) route_rows=$(wc -l < $OUT/route_k0_${name}.csv 2>/dev/null) $(date -u +%H:%M:%S) ===" >> "$OUT/progress.log"
  grep -iE "prefill:.*generation:" "$OUT/diag_k0_${name}.txt" | tail -1 >> "$OUT/progress.log"
}

run_one cyberpunk "$OUT/cyberpunk_prompt.txt" 4000 6144
run_one coffee    "$OUT/frontpage_prompt.txt" 1000 3072
echo "ALL_DONE $(date -u +%H:%M:%S)" >> "$OUT/progress.log"
echo DONE > "$OUT/k0.done"
