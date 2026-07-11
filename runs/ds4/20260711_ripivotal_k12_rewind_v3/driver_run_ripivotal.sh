#!/bin/bash
# RI-PIVOTAL: K12 + rewind-v3 (0022 v3), two arms, cyberpunk W50 two-phase, ctx8192,
# fase2 4000, greedy temp0, GARBAGE=0.80 CKPT_DEPTH=8 WARMUP=40, cache 1024.
#  ARM 1: K12 + rewind-v3, n=3 (WARMUP re-freezes to K12 = model-predicted 3.99 config).
#  ARM 2: K-ESCALATION (script-side, sanctioned): restart phase-2 from the healthy frozen
#         W50 anchor at K = 12,20,28,36,44,48; early-stop the ladder at first </html> close;
#         then n=3 confirm at the closing K. Each rung = "a collapse bumps K, retry from a
#         healthy point" -- engine has no per-rewind escalation env, so it is done per-run.
set -u
cd /root/pivotal
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
PROMPT=/root/pivotal/cyberpunk_prompt.txt
OUT=/root/ripivotal
mkdir -p "$OUT"
COMMON="--binary $BIN --model $MODEL --prompt-file $PROMPT --w-values 50 --total 4050 \
 --mask-mode weighted --n-expert 256 --cache 1024 --ctx-p1 2048 --ctx-p2 8192 \
 --temp 0 --timeout 3600"

mkenv() {  # $1 = operating keep K
  local K=$1
  printf '%s' "{\"DS4_PACE\":\"1\",\"DS4_PACE_S1\":\"1\",\"DS4_PACE_WARMUP\":\"50\",\"DS4_PACE_KEEP\":\"$K\",\"DS4_PACE_KEEP_MIN\":\"$K\",\"DS4_PACE_KEEP_MAX\":\"$K\",\"DS4_PACE_ROTATE\":\"0\",\"DS4_PACE_RELEARN\":\"0\",\"DS4_PACE_BREATH_EVERY\":\"999999\",\"DS4_PACE_DRIFT\":\"2.0\",\"DS4_PACE_WRAP\":\"1\",\"DS4_PACE_WRAP_ROTATE_DELTA\":\"1\",\"DS4_PACE_DEBUG\":\"1\",\"DS4_PACE_WEIGHTED_SELECTED\":\"1\",\"DS4_PACE_LOG\":\"{rundir}/pace_events.jsonl\",\"DS4_PACE_REWIND\":\"1\",\"DS4_SPEX_TRACE_TOKENS\":\"{rundir}/tokens.csv\",\"DS4_PACE_REWIND_GARBAGE\":\"0.80\",\"DS4_PACE_REWIND_CKPT_DEPTH\":\"8\",\"DS4_PACE_REWIND_WARMUP\":\"40\"}"
}

closed() {  # $1 = run dir; true if phase-2 closed the document
  grep -qi '</html>' "$1/deliverable.html" 2>/dev/null
}

echo "=== ARM 1: K12 + rewind-v3, n=3 ===" | tee "$OUT/progress.log"
export PIVOTAL_P2_ENV="$(mkenv 12)"
python3 scripts/run_pivotal_arm.py $COMMON --keep-k 12 --runs 3 \
  --outdir "$OUT/arm1_k12_rewind_v3" > "$OUT/arm1.out" 2>&1
echo "ARM1_RC=$?" | tee -a "$OUT/progress.log"
for r in "$OUT"/arm1_k12_rewind_v3/W050/r*/; do
  c=NO; closed "$r" && c=YES
  n=$(grep -c '"ev":"rewind"' "$r/pace_events.jsonl" 2>/dev/null || echo 0)
  echo "  arm1 $(basename $r): close=$c rewinds=$n" | tee -a "$OUT/progress.log"
done

echo "=== ARM 2: K-escalation ladder (n=1, early-stop on </html>) ===" | tee -a "$OUT/progress.log"
FINAL_K=""
for K in 12 20 28 36 44 48; do
  export PIVOTAL_P2_ENV="$(mkenv $K)"
  python3 scripts/run_pivotal_arm.py $COMMON --keep-k $K --runs 1 \
    --outdir "$OUT/arm2_esc/K$K" > "$OUT/arm2_K$K.out" 2>&1
  D="$OUT/arm2_esc/K$K/W050/r00"
  n=$(grep -c '"ev":"rewind"' "$D/pace_events.jsonl" 2>/dev/null || echo 0)
  if closed "$D"; then
    echo "  arm2 K=$K: close=YES rewinds=$n  <== LADDER HOLDS" | tee -a "$OUT/progress.log"
    FINAL_K=$K; break
  else
    echo "  arm2 K=$K: close=NO rewinds=$n" | tee -a "$OUT/progress.log"
  fi
done
echo "ARM2_LADDER_FINAL_K=${FINAL_K:-none}" | tee -a "$OUT/progress.log"

if [ -n "$FINAL_K" ]; then
  echo "=== ARM 2 confirm: n=3 at holding K=$FINAL_K ===" | tee -a "$OUT/progress.log"
  export PIVOTAL_P2_ENV="$(mkenv $FINAL_K)"
  python3 scripts/run_pivotal_arm.py $COMMON --keep-k "$FINAL_K" --runs 3 \
    --outdir "$OUT/arm2_confirm_K$FINAL_K" > "$OUT/arm2_confirm.out" 2>&1
  echo "ARM2_CONFIRM_RC=$? K=$FINAL_K" | tee -a "$OUT/progress.log"
else
  echo "=== ARM 2: no ladder rung closed </html> up to K=48; skipping n=3 confirm ===" | tee -a "$OUT/progress.log"
fi

echo "ALL_DONE" | tee -a "$OUT/progress.log"
echo DONE > "$OUT/ripivotal.done"
