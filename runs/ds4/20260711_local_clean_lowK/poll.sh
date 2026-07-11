#!/bin/bash
# poll.sh <max_iters> : heartbeat every 30s, break when a new DONE line appears.
S=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_local_clean_lowK/curve
L=$S/curve_trace_progress.log
MAX=${1:-18}
base=$(grep -c DONE "$L" 2>/dev/null); base=${base:-0}
i=0
while [ "$i" -lt "$MAX" ]; do
  now=$(grep -c DONE "$L" 2>/dev/null); now=${now:-0}
  cur=$(grep -E "START|DONE|COMPLETE" "$L" 2>/dev/null | tail -1)
  # current run dir = last START name
  name=$(grep START "$L" 2>/dev/null | tail -1 | sed -E 's/.*START ([^ ]+).*/\1/')
  gt=$(wc -l < "$S/$name/gen.txt" 2>/dev/null || echo 0)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
  echo "[$(date +%H:%M:%S)] iter=$i done=$now genlines=$gt gpu=$mem :: $cur"
  if [ "$now" -gt "$base" ]; then echo "=== NEW DONE ==="; grep DONE "$L" | tail -1; break; fi
  i=$((i+1)); sleep 30
done
echo "--- poll end (i=$i) ---"
grep -E "DONE|COMPLETE" "$L" 2>/dev/null | tail -3
