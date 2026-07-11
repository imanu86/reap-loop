#!/bin/bash
# Phase-2 status: traces + coffee + k0 progress logs, newest gen tail, mem/gpu.
R=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot
echo "===== TRACES ====="; cat "$R/traces/progress.log" 2>/dev/null
echo "===== COFFEE ====="; cat "$R/coffee/progress.log" 2>/dev/null
echo "===== K0 ====="; cat "$R/k0/progress.log" 2>/dev/null
echo "===== phase2.log ====="; cat "$R/phase2.log" 2>/dev/null
echo "===== live ds4 ====="; pgrep -af "ds4 -m" | grep -v pgrep || echo "(none)"
NEWD=$(ls -dt "$R"/traces/*/ "$R"/coffee/*/ "$R"/k0/*/ 2>/dev/null | head -1)
echo "===== newest=$NEWD bytes=$(wc -c < "$NEWD/gen.txt" 2>/dev/null) ====="
echo "-- tail 400 --"; tail -c 400 "$NEWD/gen.txt" 2>/dev/null
echo; echo "-- swap/gpu --"; free -m | grep Swap:; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; date +%H:%M:%S
