#!/bin/bash
# Eval ppl bias-mask sul corpus surrogato pubblico. Config sequenziali su un pod.
# Uso: bash run_eval.sh MASKFILE CORPUSDIR OUTDIR CONFIG [CONFIG...]
#   CONFIG: full | reap_k50 | rand50_s0 | rand50_s1 | rand50_s2
set -u
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
MASK=$1; CORPUS=$2; OUT=$3; shift 3
mkdir -p "$OUT"
COMMON="--cuda --ssd-streaming -c 4096"
BM=/root/scripts/reap_bias_mask_ds4.py

apply_cfg() {
  case "$1" in
    full) return 0;;
    reap_k50) python3 $BM --gguf $MODEL --maskfile $MASK --apply reap;;
    rand50_s*) local s=${1##*_s}; python3 $BM --gguf $MODEL --maskfile $MASK --apply random --random-seed "$s";;
    *) echo "config sconosciuta $1"; exit 1;;
  esac
}
restore_cfg() { [ "$1" = "full" ] && return 0; python3 $BM --gguf $MODEL --restore; }

ppl_run() {
  local cfg=$1 f=$2 b
  b=$(basename "$2" .txt)
  $BIN -m $MODEL $COMMON --perplexity-file "$f" -n 600 > "$OUT/ppl_${cfg}_${b}.log" 2>&1
  local rc=$?
  local line
  line=$(grep '^tokens=' "$OUT/ppl_${cfg}_${b}.log" | tail -1)
  echo "$cfg,$b,rc=$rc,$line" | tee -a $OUT/results_raw.csv
}

echo "=== warm-up $(date -u +%FT%TZ)"
$BIN -m $MODEL $COMMON --nothink --temp 0 -n 8 -p "ciao" > $OUT/warmup.log 2>&1

for CFG in "$@"; do
  echo "=== CONFIG $CFG $(date -u +%FT%TZ)"
  apply_cfg "$CFG" 2>&1 | tee -a $OUT/biasmask.log
  if [ "$CFG" = "reap_k50" ]; then
    echo "=== V0 mechanism check $CFG"
    DS4_SPEX_TRACE_ROUTING=$OUT/v0_trace.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
      $BIN -m $MODEL $COMMON --nothink --temp 0 -n 48 --prompt-file "$CORPUS/../prompts/p00_extract.txt" > $OUT/v0_gen.log 2>&1
    python3 - "$OUT/v0_trace.csv" "$MASK" <<'EOF' 2>&1 | tee -a $OUT/biasmask.log
import csv, json, sys
trace, maskf = sys.argv[1], sys.argv[2]
mask = json.load(open(maskf))
keep = {int(l): set(v) for l, v in mask["keep"].items()}
bad = tot = 0
for row in csv.DictReader(open(trace)):
    l = int(row["layer"]); n = int(row["n"])
    if l not in keep: continue
    for s in range(n):
        e = int(row[f"e{s}"]); tot += 1
        if e not in keep[l]: bad += 1
print(f"V0 checked={tot} violations={bad}", "V0_OK" if bad == 0 else "V0_FAIL")
EOF
  fi
  for c in 0 1 2 3 4 5 6 7 8 9; do ppl_run "$CFG" "$CORPUS/dom_chunk$c.txt"; done
  restore_cfg "$CFG" 2>&1 | tee -a $OUT/biasmask.log
  echo "=== DONE_CONFIG $CFG $(date -u +%FT%TZ)"
done
echo "ALL_DONE $(date -u +%FT%TZ)"
