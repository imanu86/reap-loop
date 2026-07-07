#!/bin/bash
# Trace routing+weights dei 20 prompt surrogati. Uso: bash run_trace.sh CANDDIR OUTDIR
# CANDDIR = /root/prod  (contiene prompts/pNN_*.txt)
set -u
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
CAND=$1
OUT=$2
mkdir -p "$OUT"
COMMON="--cuda --ssd-streaming -c 4096 --nothink --temp 0"

# warm-up scartata (valida colonne pesi)
DS4_SPEX_TRACE_ROUTING=$OUT/warmup.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
  $BIN -m $MODEL $COMMON -n 8 -p "ciao" > $OUT/warmup.log 2>&1
echo "warmup done $(date -u +%FT%TZ)"

for f in "$CAND"/prompts/p*.txt; do
  b=$(basename "$f" .txt)
  case "$b" in *explain*) N=512;; *) N=320;; esac
  DS4_SPEX_TRACE_ROUTING=$OUT/trace_${b}.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
    $BIN -m $MODEL $COMMON -n $N --prompt-file "$f" > $OUT/gen_${b}.log 2>&1
  echo "traced $b ($(wc -l < $OUT/trace_${b}.csv) rows) $(date -u +%FT%TZ)"
done
echo "TRACE_ALL_DONE $(date -u +%FT%TZ)"
