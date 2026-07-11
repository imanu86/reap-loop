#!/bin/bash
cd /root/ds4; source /root/smoke_0022/common_env.sh
export DS4_PACE_LOG=/root/smoke_0022/b_events.jsonl
export DS4_PACE_REWIND=0
rm -f "$DS4_PACE_LOG"
t0=$(date +%s)
./ds4 $CLI > /root/smoke_0022/b_gen.txt 2> /root/smoke_0022/b_run.log
echo "GATE_B_RC=$? elapsed=$(( $(date +%s)-t0 ))s" | tee /root/smoke_0022/b_status.txt
