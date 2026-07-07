#!/bin/bash
# JOB1 standalone — validazione appaiata ds4-eval OFF vs ON.
# Fix vs night_run.sh: niente gate sull'exit code (ds4-eval esce rc=1 se una
# domanda fallisce: non e' un errore di flag). rc atteso: 0/1 = ok, 124 = timeout.
set -u
EVAL=/root/ds4-dspark/ds4-eval
MODEL=/root/models/ds4-2bit.gguf
MTP=/root/models/ds4-mtp.gguf
OUT=/root/out_night
PROG="$OUT/progress.log"
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

say "JOB1(retry) OFF start - 20 domande, 384 tok, cache 250"
env DS4_SPEX_STATS=1 timeout 14400 \
  "$EVAL" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 250 \
  --questions 20 --tokens 384 --temp 0 --nothink --plain \
  --trace "$OUT/eval_off.trace" > "$OUT/eval_off.log" 2>&1
say "JOB1(retry) OFF rc=$? (0/1=ok 124=timeout)"

say "JOB1(retry) ON start - stesse 20 domande, unlock+probe"
env DS4_SPEX_STATS=1 DS4_MTP_STREAMING=1 DS4_MTP_PROBE=1 timeout 14400 \
  "$EVAL" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 250 \
  --mtp "$MTP" --questions 20 --tokens 384 --temp 0 --nothink --plain \
  --trace "$OUT/eval_on.trace" > "$OUT/eval_on.log" 2>&1
say "JOB1(retry) ON rc=$?"

say "JOB1(retry) DONE"
echo DONE > "$OUT/JOB1_DONE"
