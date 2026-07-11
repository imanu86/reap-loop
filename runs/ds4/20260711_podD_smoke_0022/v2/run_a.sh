#!/bin/bash
cd /root/ds4; source /root/smoke_0022/common_env.sh
export DS4_PACE_LOG=/root/smoke_0022_v2/a_events.jsonl
export DS4_PACE_REWIND=1
export DS4_PACE_REWIND_ARM_K=-20 DS4_PACE_REWIND_ARM_H=0.1
export DS4_PACE_REWIND_FIRE_K=-20 DS4_PACE_REWIND_FIRE_H=0.1
export DS4_TOKEN_TIMING=1
rm -f "$DS4_PACE_LOG"
t0=$(date +%s)
./ds4 $CLI > /root/smoke_0022_v2/a_gen.txt 2> >(python3 /root/smoke_0022_v2/ts.py > /root/smoke_0022_v2/a_run.log)
rc=$?
wait
echo "GATE_A_RC=$rc elapsed=$(( $(date +%s)-t0 ))s" | tee /root/smoke_0022_v2/a_status.txt
