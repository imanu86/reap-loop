#!/usr/bin/env bash
set -uo pipefail

LABEL=${1:?label required}
ARM=${2:?off or on required}
[[ "$ARM" == "off" || "$ARM" == "on" ]] || exit 2

ROOT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_v2_zerocopy
SRC=/root/ds4-v2-work
BIN=$SRC/ds4-server
MODEL=/root/models/ds4-2bit.gguf
MASK=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_virtual_bake/masks/mask60_self.txt
OUT=$ROOT/$LABEL
PORT=8096
RAM_FLOOR_MB=8192

mkdir -p "$OUT"
exec 9>/tmp/ds4-gpu.lock
flock 9

cleanup() {
  if [[ -f "$OUT/server.pid" ]]; then
    pid=$(cat "$OUT/server.pid" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  if [[ -n "${ram_mon_pid:-}" ]]; then kill "$ram_mon_pid" 2>/dev/null || true; fi
  flock -u 9 2>/dev/null || true
}
trap cleanup EXIT

(
  while true; do
    avail=$(free -m | awk '/^Mem:/{print $7}')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) MemAvailable_MB=$avail" >> "$OUT/ram_log.txt"
    if [[ -n "$avail" && "$avail" -lt "$RAM_FLOOR_MB" ]]; then
      echo "ram_floor_breach available_mb=$avail" > "$OUT/ABORT.txt"
      [[ -f "$OUT/server.pid" ]] && kill "$(cat "$OUT/server.pid")" 2>/dev/null || true
      break
    fi
    sleep 15
  done
) &
ram_mon_pid=$!

export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
export DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1
export DS4_PACE=0
export DS4_REAP_MASK_FILE="$MASK"
export DS4_REAP_PREFETCH=1
export DS4_REAP_PREFETCH_THREADS=16
if [[ "$ARM" == "on" ]]; then
  export DS4_CUDA_STREAM_FROM_RAM_MASKED="$MASK"
  export DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=24
  export DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1
else
  unset DS4_CUDA_STREAM_FROM_RAM_MASKED
  unset DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB
  unset DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/start_utc.txt"
env | LC_ALL=C sort > "$OUT/server_env.txt"
git -C "$SRC" rev-parse HEAD > "$OUT/ds4_base_commit.txt"
git -C "$SRC" diff --stat > "$OUT/ds4_diff_stat.txt"
sha256sum "$SRC/ds4.c" "$SRC/ds4_cuda.cu" "$SRC/ds4_gpu.h" "$BIN" > "$OUT/build_hashes.txt"
cp "$ROOT/request_warmup.json" "$OUT/request_warmup.json"
cp "$ROOT/request_measured.json" "$OUT/request_measured.json"

"$BIN" -m "$MODEL" --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 400 --prefill-chunk 512 \
  -c 2048 -n 128 --host 127.0.0.1 --port "$PORT" --cors \
  >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
pid=$!
echo "$pid" > "$OUT/server.pid"

ready=0
for _ in $(seq 1 180); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  kill -0 "$pid" 2>/dev/null || break
  [[ -f "$OUT/ABORT.txt" ]] && break
  sleep 5
done
echo "ready=$ready" > "$OUT/status.txt"
[[ "$ready" == 1 ]] || exit 1

curl -fsS -m 1200 -H "Content-Type: application/json" \
  -d @"$OUT/request_warmup.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  > "$OUT/warmup_response.json" 2> "$OUT/warmup_curl.err"
warm_rc=$?
echo "warmup_rc=$warm_rc" >> "$OUT/status.txt"
[[ "$warm_rc" == 0 ]] || exit "$warm_rc"

curl -fsS -m 1800 -w 'http_code=%{http_code}\ntime_total=%{time_total}\n' \
  -H "Content-Type: application/json" -d @"$OUT/request_measured.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -o "$OUT/response.json" > "$OUT/curl_timing.txt" 2> "$OUT/curl.err"
rc=$?
echo "measured_rc=$rc" >> "$OUT/status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/end_utc.txt"
sleep 2
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
exit "$rc"
