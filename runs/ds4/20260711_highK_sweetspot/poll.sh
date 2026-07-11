#!/bin/bash
# Poll until DONE-count reaches $1 (or timeout ~$2 iterations of 30s), then print status.
set -u
LOG=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/sweep/progress.log
NEED=${1:-2}
ITERS=${2:-18}
for i in $(seq 1 "$ITERS"); do
  dn=$(grep -c "DONE" "$LOG" 2>/dev/null); dn=${dn:-0}
  cc=$(grep -c "COMPLETE" "$LOG" 2>/dev/null); cc=${cc:-0}
  if [ "$dn" -ge "$NEED" ] || [ "$cc" -ge 1 ]; then break; fi
  sleep 30
done
echo "=== progress.log ==="
cat "$LOG"
echo "=== live ds4 ==="
pgrep -af "ds4 -m" | grep -v pgrep | head -1 || echo "(no ds4 running)"
echo "=== mem/gpu now ==="
free -m | awk '/^Mem:/{print "mem_used_MB="$3" free="$4" buffcache="$6} /^Swap:/{print "swap_used_MB="$3}'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
