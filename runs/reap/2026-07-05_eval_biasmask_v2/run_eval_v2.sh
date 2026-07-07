#!/bin/bash
# Track REAP-ds4 — eval ppl v2 (mandato paper-grade): N config sequenziali su un pod.
# Uso: bash run_eval_v2.sh CONFIG [CONFIG...]
#   CONFIG: full | reap_k50 | reap_k25 | reap_k70 | rand50_s0 | rand50_s1 | rand50_s2
# 10 chunk dom (-n 600) + 10 chunk gen (-n 600) per config. Restore tra config.
set -u
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
OUT=/root/eval
mkdir -p $OUT
CTX=4096
COMMON="--cuda --ssd-streaming -c $CTX"

apply_cfg() { # config
  case "$1" in
    full) return 0;;
    reap_k50) python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --maskfile /root/reap_mask_ds4_domain.json --apply reap;;
    reap_k25) python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --maskfile /root/reap_mask_ds4_domain_k25.json --apply reap;;
    reap_k70) python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --maskfile /root/reap_mask_ds4_domain_k70.json --apply reap;;
    rand50_s*) local s=${1##*_s}; python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --maskfile /root/reap_mask_ds4_domain.json --apply random --random-seed "$s";;
    *) echo "config sconosciuta $1"; exit 1;;
  esac
}

restore_cfg() { # config
  [ "$1" = "full" ] && return 0
  python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --restore
}

ppl_run() { # config chunkfile
  local cfg=$1 f=$2 b
  b=$(basename "$2" .txt)
  echo "=== ppl $cfg $b $(date -u +%FT%TZ)"
  $BIN -m $MODEL $COMMON --perplexity-file "$f" -n 600 > "$OUT/ppl_${cfg}_${b}.log" 2>&1
  local rc=$?
  local line
  line=$(grep '^tokens=' "$OUT/ppl_${cfg}_${b}.log" | tail -1)
  echo "$cfg,$b,rc=$rc,$line" | tee -a $OUT/results_raw.csv
}

echo "=== warm-up scartata $(date -u +%FT%TZ)"
$BIN -m $MODEL $COMMON --nothink --temp 0 -n 8 -p "ciao" > $OUT/warmup.log 2>&1

for CFG in "$@"; do
  echo "=== CONFIG $CFG $(date -u +%FT%TZ)"
  apply_cfg "$CFG" 2>&1 | tee -a $OUT/biasmask.log
  if [ "$CFG" = "reap_k50" ]; then
    echo "=== V0 mechanism check $CFG"
    DS4_SPEX_TRACE_ROUTING=$OUT/v0_${CFG}_trace.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
      $BIN -m $MODEL $COMMON --nothink --temp 0 -n 48 --prompt-file /root/v0_prompt.txt > $OUT/v0_${CFG}_gen.log 2>&1
    python3 - "$OUT/v0_${CFG}_trace.csv" /root/reap_mask_ds4_domain.json <<'EOF' 2>&1 | tee -a $OUT/biasmask.log
import csv, json, sys
trace, maskf = sys.argv[1], sys.argv[2]
mask = json.load(open(maskf))
keep = {int(l): set(v) for l, v in mask["keep"].items()}
bad = tot = 0
for row in csv.DictReader(open(trace)):
    l = int(row["layer"]); n = int(row["n"])
    if l not in keep:
        continue
    for s in range(n):
        e = int(row[f"e{s}"]); tot += 1
        if e not in keep[l]:
            bad += 1
print(f"V0 {trace}: checked={tot} violations={bad}", "V0_OK" if bad == 0 else "V0_FAIL")
EOF
  fi
  for c in 0 1 2 3 4 5 6 7 8 9; do ppl_run "$CFG" /root/corpus/dom_chunk$c.txt; done
  for c in 0 1 2 3 4 5 6 7 8 9; do ppl_run "$CFG" /root/corpus/gen_chunk$c.txt; done
  restore_cfg "$CFG" 2>&1 | tee -a $OUT/biasmask.log
  echo "=== DONE_CONFIG $CFG $(date -u +%FT%TZ)"
done
echo "ALL_DONE $(date -u +%FT%TZ)"
