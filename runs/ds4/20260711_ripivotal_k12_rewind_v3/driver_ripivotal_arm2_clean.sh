#!/bin/bash
# ARM 2 launcher: waits for the (already-running) ARM 1 n=3 to finish, then runs the
# K-escalation ladder + n=3 confirm. Same recipe as ARM 1 but K sweeps 12..48 restarting
# phase-2 from the healthy frozen W50 anchor; early-stop at first </html> close.
set -u
cd /root/pivotal
BIN=/root/ds4/ds4; MODEL=/root/models/ds4-2bit.gguf; PROMPT=/root/pivotal/cyberpunk_prompt.txt
OUT=/root/ripivotal
COMMON="--binary $BIN --model $MODEL --prompt-file $PROMPT --w-values 50 --total 4050 \
 --mask-mode weighted --n-expert 256 --cache 1024 --ctx-p1 2048 --ctx-p2 8192 --temp 0 --timeout 3600"

mkenv() { local K=$1
  printf '%s' "{\"DS4_PACE\":\"1\",\"DS4_PACE_S1\":\"1\",\"DS4_PACE_WARMUP\":\"50\",\"DS4_PACE_KEEP\":\"$K\",\"DS4_PACE_KEEP_MIN\":\"$K\",\"DS4_PACE_KEEP_MAX\":\"$K\",\"DS4_PACE_ROTATE\":\"0\",\"DS4_PACE_RELEARN\":\"0\",\"DS4_PACE_BREATH_EVERY\":\"999999\",\"DS4_PACE_DRIFT\":\"2.0\",\"DS4_PACE_WRAP\":\"1\",\"DS4_PACE_WRAP_ROTATE_DELTA\":\"1\",\"DS4_PACE_DEBUG\":\"1\",\"DS4_PACE_WEIGHTED_SELECTED\":\"1\",\"DS4_PACE_LOG\":\"{rundir}/pace_events.jsonl\",\"DS4_PACE_REWIND\":\"1\",\"DS4_SPEX_TRACE_TOKENS\":\"{rundir}/tokens.csv\",\"DS4_PACE_REWIND_GARBAGE\":\"0.80\",\"DS4_PACE_REWIND_CKPT_DEPTH\":\"8\",\"DS4_PACE_REWIND_WARMUP\":\"40\"}"; }
closed() { grep -qi '</html>' "$1/deliverable.html" 2>/dev/null; }
plog() { echo "$@" | tee -a "$OUT/progress.log"; }

# --- wait for ARM 1 to finish (summary_median.csv written + no ds4 running) ---
plog "[arm2] waiting for arm1 to complete ..."
while true; do
  if [ -f "$OUT/arm1_k12_rewind_v3/summary_median.csv" ] && ! pgrep -f '[d]s4/ds4' >/dev/null; then break; fi
  sleep 20
done
plog "[arm2] arm1 done; arm1 summary:"
cat "$OUT/arm1_k12_rewind_v3/summary_median.csv" | tee -a "$OUT/progress.log"

plog "=== ARM 2: K-escalation ladder (n=1, early-stop on </html>) ==="
FINAL_K=""
for K in 12 20 28 36 44 48; do
  export PIVOTAL_P2_ENV="$(mkenv $K)"
  python3 scripts/run_pivotal_arm.py $COMMON --keep-k $K --runs 1 --outdir "$OUT/arm2_esc/K$K" > "$OUT/arm2_K$K.out" 2>&1
  D="$OUT/arm2_esc/K$K/W050/r00"
  n=$(grep -c '"ev":"rewind"' "$D/pace_events.jsonl" 2>/dev/null || echo 0)
  if closed "$D"; then plog "  arm2 K=$K: close=YES rewinds=$n  <== LADDER HOLDS"; FINAL_K=$K; break
  else plog "  arm2 K=$K: close=NO rewinds=$n"; fi
done
plog "ARM2_LADDER_FINAL_K=${FINAL_K:-none}"

if [ -n "$FINAL_K" ]; then
  plog "=== ARM 2 confirm: n=3 at holding K=$FINAL_K ==="
  export PIVOTAL_P2_ENV="$(mkenv $FINAL_K)"
  python3 scripts/run_pivotal_arm.py $COMMON --keep-k "$FINAL_K" --runs 3 --outdir "$OUT/arm2_confirm_K$FINAL_K" > "$OUT/arm2_confirm.out" 2>&1
  plog "ARM2_CONFIRM_RC=$? K=$FINAL_K"
else
  plog "=== ARM 2: no rung closed </html> up to K=48; skipping n=3 confirm ==="
fi
plog "ALL_DONE"; echo DONE > "$OUT/ripivotal.done"
