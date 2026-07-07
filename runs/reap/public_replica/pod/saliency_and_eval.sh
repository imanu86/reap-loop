#!/bin/bash
# Dopo la trace: n_eff equivalenza -> (se valida) saliency K50 + random control -> eval ppl.
# Uso: bash saliency_and_eval.sh TRACEDIR CORPUSDIR OUTBASE LABEL
set -u
export PATH=/usr/local/cuda/bin:$PATH
TRACEDIR=$1; CORPUS=$2; OUT=$3; LABEL=$4
mkdir -p "$OUT"

echo "=== n_eff equivalenza ($LABEL) ==="
python3 /root/scripts/reap_neff.py --tracedir "$TRACEDIR" --label "$LABEL" --out "$OUT/neff.json" | tee "$OUT/neff_stdout.txt"

VALID=$(python3 -c "import json;print(json.load(open('$OUT/neff.json'))['equivalence_verdict']['valid'])")
echo "EQUIVALENCE_VALID=$VALID"
if [ "$VALID" != "True" ]; then
  echo "SURROGATE_INVALID: n_eff fuori range, STOP (prova altro corpus)"; exit 2
fi

echo "=== saliency g-only -> mask K50 (+random seed0) ==="
python3 /root/scripts/reap_saliency_ds4.py --tracedir "$TRACEDIR" --keep-frac 0.5 --seed 0 \
  --tag "${LABEL}" --out "$OUT/reap_mask_${LABEL}_k50.json" | tee "$OUT/saliency_stdout.txt"

echo "=== eval ppl full/reap_k50/rand50_s0/s1/s2 ==="
bash /root/run_eval.sh "$OUT/reap_mask_${LABEL}_k50.json" "$CORPUS" "$OUT/eval" \
  full reap_k50 rand50_s0 rand50_s1 rand50_s2 > "$OUT/eval_run.log" 2>&1
echo "EVAL_EXIT=$?"
tail -5 "$OUT/eval_run.log"
echo "SALIENCY_EVAL_ALL_DONE $(date -u +%FT%TZ)"
