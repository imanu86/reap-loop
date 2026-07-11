#!/bin/bash
# ARM 2 n=3 confirm at the escalation CEILING K=48 (final K reached; no rung held).
set -u
cd /root/pivotal
BIN=/root/ds4/ds4; MODEL=/root/models/ds4-2bit.gguf; PROMPT=/root/pivotal/cyberpunk_prompt.txt
OUT=/root/ripivotal
COMMON="--binary $BIN --model $MODEL --prompt-file $PROMPT --w-values 50 --total 4050 \
 --mask-mode weighted --n-expert 256 --cache 1024 --ctx-p1 2048 --ctx-p2 8192 --temp 0 --timeout 3600"
K=48
export PIVOTAL_P2_ENV="{\"DS4_PACE\":\"1\",\"DS4_PACE_S1\":\"1\",\"DS4_PACE_WARMUP\":\"50\",\"DS4_PACE_KEEP\":\"$K\",\"DS4_PACE_KEEP_MIN\":\"$K\",\"DS4_PACE_KEEP_MAX\":\"$K\",\"DS4_PACE_ROTATE\":\"0\",\"DS4_PACE_RELEARN\":\"0\",\"DS4_PACE_BREATH_EVERY\":\"999999\",\"DS4_PACE_DRIFT\":\"2.0\",\"DS4_PACE_WRAP\":\"1\",\"DS4_PACE_WRAP_ROTATE_DELTA\":\"1\",\"DS4_PACE_DEBUG\":\"1\",\"DS4_PACE_WEIGHTED_SELECTED\":\"1\",\"DS4_PACE_LOG\":\"{rundir}/pace_events.jsonl\",\"DS4_PACE_REWIND\":\"1\",\"DS4_SPEX_TRACE_TOKENS\":\"{rundir}/tokens.csv\",\"DS4_PACE_REWIND_GARBAGE\":\"0.80\",\"DS4_PACE_REWIND_CKPT_DEPTH\":\"8\",\"DS4_PACE_REWIND_WARMUP\":\"40\"}"
python3 scripts/run_pivotal_arm.py $COMMON --keep-k $K --runs 3 --outdir "$OUT/arm2_confirm_K48" > "$OUT/arm2_confirm.out" 2>&1
echo "ARM2_CONFIRM_K48_RC=$?" >> "$OUT/progress.log"
echo DONE > "$OUT/confirm_k48.done"
