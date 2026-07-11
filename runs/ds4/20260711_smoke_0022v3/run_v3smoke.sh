#!/bin/bash
# v3 S1-guided rewind mechanism smoke: reproduce pivotal arm2 (K12 static weighted,
# cyberpunk, W50 two-phase, ctx_p2 8192) on the v3 binary, phase-2-only PACE+rewind
# env via the wrapper, PLUS the three v3 levers: GARBAGE=0.80, CKPT_DEPTH=8, WARMUP=40.
set -u
cd /root/pivotal
ARGS="--binary /root/ds4/ds4 --model /root/models/ds4-2bit.gguf \
 --prompt-file /root/pivotal/cyberpunk_prompt.txt \
 --w-values 50 --total 800 --keep-k 12 --mask-mode weighted --n-expert 256 \
 --cache 1024 --ctx-p1 2048 --ctx-p2 8192 --temp 0 --timeout 3600 --runs 1"
export PIVOTAL_P2_ENV='{"DS4_PACE":"1","DS4_PACE_S1":"1","DS4_PACE_WARMUP":"50","DS4_PACE_KEEP":"12","DS4_PACE_KEEP_MIN":"12","DS4_PACE_KEEP_MAX":"12","DS4_PACE_ROTATE":"0","DS4_PACE_RELEARN":"0","DS4_PACE_BREATH_EVERY":"999999","DS4_PACE_DRIFT":"2.0","DS4_PACE_WRAP":"1","DS4_PACE_WRAP_ROTATE_DELTA":"1","DS4_PACE_DEBUG":"1","DS4_PACE_WEIGHTED_SELECTED":"1","DS4_PACE_LOG":"{rundir}/pace_events.jsonl","DS4_PACE_REWIND":"1","DS4_SPEX_TRACE_TOKENS":"{rundir}/tokens.csv","DS4_PACE_REWIND_GARBAGE":"0.80","DS4_PACE_REWIND_CKPT_DEPTH":"8","DS4_PACE_REWIND_WARMUP":"40"}'
python3 scripts/run_pivotal_arm.py $ARGS --outdir /root/pivotal/v3_smoke > /root/pivotal/v3_smoke.out 2>&1
echo "V3SMOKE_RC=$?" | tee /root/pivotal/v3_smoke.done
