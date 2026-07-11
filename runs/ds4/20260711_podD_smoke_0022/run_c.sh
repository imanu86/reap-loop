#!/bin/bash
cd /root/ds4; source /root/smoke_0022/common_env.sh
export DS4_PACE_LOG=/root/smoke_0022/c_events.jsonl
export DS4_PACE_REWIND=1
# default detector: ARM(k0.5,h4) FIRE(k1.0,h8) CALWIN128 — do NOT override
rm -f "$DS4_PACE_LOG"
t0=$(date +%s)
./ds4 $CLI > /root/smoke_0022/c_gen.txt 2> /root/smoke_0022/c_run.log
echo "GATE_C_RC=$? elapsed=$(( $(date +%s)-t0 ))s" | tee /root/smoke_0022/c_status.txt
