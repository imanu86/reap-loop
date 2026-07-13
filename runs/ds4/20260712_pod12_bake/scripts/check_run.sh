#!/usr/bin/env bash
# Quick status dump for a run dir. Usage: check_run.sh <run_dir>
OUT="$1"
echo "=== $OUT ==="
ls -la "$OUT" 2>&1
echo "--- RUN_META.txt ---"
cat "$OUT/RUN_META.txt" 2>/dev/null
echo "--- STOP_REASON.txt ---"
cat "$OUT/STOP_REASON.txt" 2>/dev/null
echo "--- server.pid alive? ---"
if [[ -f "$OUT/server.pid" ]]; then
  pid=$(cat "$OUT/server.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "PID $pid ALIVE"; else echo "PID $pid DEAD"; fi
fi
echo "--- server.stderr.log (tail 30) ---"
tail -30 "$OUT/server.stderr.log" 2>/dev/null
echo "--- ram_log.txt (tail 5) ---"
tail -5 "$OUT/ram_log.txt" 2>/dev/null
echo "--- stream_live.txt (size + tail) ---"
wc -c "$OUT/stream_live.txt" 2>/dev/null
tail -c 400 "$OUT/stream_live.txt" 2>/dev/null
echo
echo "--- warmup_curl.err ---"
cat "$OUT/warmup_curl.err" 2>/dev/null
echo "--- current mem/gpu ---"
free -h
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
