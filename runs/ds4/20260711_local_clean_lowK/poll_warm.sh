#!/bin/bash
L=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_local_clean_lowK/curve/warm/warm_progress.log
MAX=${1:-18}
i=0
while [ "$i" -lt "$MAX" ]; do
  cur=$(tail -1 "$L" 2>/dev/null)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
  echo "[$(date +%H:%M:%S)] iter=$i gpu=$mem :: $cur"
  if grep -q WARM_COMPLETE "$L" 2>/dev/null; then echo "=== WARM_COMPLETE ==="; break; fi
  i=$((i+1)); sleep 30
done
echo "--- results ---"
grep DONE "$L" 2>/dev/null
