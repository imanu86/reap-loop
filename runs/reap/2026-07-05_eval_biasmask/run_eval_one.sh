#!/bin/bash
# Track REAP-ds4 — eval bias-mask, UNA config per pod (parallelo).
# Uso: bash run_eval_one.sh {full|reap|rand}
set -u
CFG=$1
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
MASK=/root/reap_mask_ds4_domain.json
OUT=/root/eval
mkdir -p $OUT
CTX=4096
COMMON="--cuda --ssd-streaming -c $CTX"

ppl_run() { # chunkfile ntok
  local f=$1 n=$2 b
  b=$(basename "$1" .txt)
  echo "=== ppl $CFG $b -n $n $(date -u +%FT%TZ)"
  $BIN -m $MODEL $COMMON --perplexity-file "$f" -n "$n" > "$OUT/ppl_${CFG}_${b}.log" 2>&1
  local rc=$?
  local line
  line=$(grep '^tokens=' "$OUT/ppl_${CFG}_${b}.log" | tail -1)
  echo "$CFG,$b,rc=$rc,$line" | tee -a $OUT/results_raw.csv
}

if [ "$CFG" != "full" ]; then
  WHICH=$CFG
  [ "$CFG" = "rand" ] && WHICH=random
  echo "=== APPLY $WHICH $(date -u +%FT%TZ)"
  python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --maskfile $MASK --apply $WHICH 2>&1 | tee -a $OUT/biasmask.log
fi

echo "=== warm-up scartata $(date -u +%FT%TZ)"
$BIN -m $MODEL $COMMON --nothink --temp 0 -n 8 -p "ciao" > $OUT/warmup.log 2>&1

if [ "$CFG" = "reap" ]; then
  echo "=== V0 mechanism check $(date -u +%FT%TZ)"
  DS4_SPEX_TRACE_ROUTING=$OUT/v0_reap_trace.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
    $BIN -m $MODEL $COMMON --nothink --temp 0 -n 48 --prompt-file /root/v0_prompt.txt > $OUT/v0_reap_gen.log 2>&1
  python3 - <<'EOF' 2>&1 | tee -a $OUT/biasmask.log
import csv, json
mask = json.load(open("/root/reap_mask_ds4_domain.json"))
keep = {int(l): set(v) for l, v in mask["keep"].items()}
bad = tot = 0
for row in csv.DictReader(open("/root/eval/v0_reap_trace.csv")):
    l = int(row["layer"]); n = int(row["n"])
    if l not in keep:
        continue
    for s in range(n):
        e = int(row[f"e{s}"]); tot += 1
        if e not in keep[l]:
            bad += 1
print(f"V0 selections checked={tot} violations={bad}", "V0_OK" if bad == 0 else "V0_FAIL")
EOF
fi

for c in 0 1 2 3; do ppl_run /root/corpus/dom_chunk$c.txt 850; done
for c in 0 1; do ppl_run /root/corpus/gen_chunk$c.txt 800; done
echo "ALL_DONE_$CFG $(date -u +%FT%TZ)"
