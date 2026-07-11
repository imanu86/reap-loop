#!/bin/bash
RUN=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot
echo "===== progress.log (sweep) ====="
cat "$RUN/sweep/progress.log" 2>/dev/null
echo "===== ls sweep ====="
ls -la "$RUN/sweep" 2>/dev/null
echo "===== live ds4 ====="
pgrep -af "ds4 -m" | grep -v pgrep | head -1 || echo "(none)"
echo "===== newest run dir diag tail ====="
NEWD=$(ls -dt "$RUN"/sweep/*/ 2>/dev/null | head -1)
echo "newest=$NEWD"
if [ -n "$NEWD" ]; then
  echo "-- gen bytes: $(wc -c < "$NEWD/gen.txt" 2>/dev/null) --"
  echo "-- diag tail --"
  tr '\r' '\n' < "$NEWD/diag.txt" 2>/dev/null | grep -viE "prefill layer" | tail -6
  echo "-- mem.log tail --"
  tail -3 "$NEWD/mem.log" 2>/dev/null
fi
echo "===== mem/gpu now ====="
free -m | grep -E "Mem:|Swap:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
