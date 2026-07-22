#!/usr/bin/env bash
# Profile-A protocol on the 0050k safe async selected-load transport.
# Uses bounded masked mmap pinning without a REAP/PACE/SPEX compute mask.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 DS4_SERVER OUTROOT REPEATS [PIN_PLAN]" >&2
    echo "       PIN_PLAN may instead be supplied through the environment" >&2
    exit 2
fi

BIN=$1
OUTROOT=$2
REPEATS=$3
PIN_PLAN=${4:-${PIN_PLAN:-}}
MODEL=${MODEL:-/workspace/models/ds4-2bit.gguf}
PORT=${PORT:-8096}
MAX_TOKENS=${MAX_TOKENS:-256}
WARMUP_TOKENS=${WARMUP_TOKENS:-256}
PIN_BUDGET_GB=${PIN_BUDGET_GB:-72}
PIN_MODE=${PIN_MODE:-on}
STAGE_DEPTH=${STAGE_DEPTH:-8}
TRANSPORT_DEBUG=${TRANSPORT_DEBUG:-0}
DIAGNOSTICS=${DIAGNOSTICS:-1}
GPU_LOCK=${GPU_LOCK:-/tmp/ds4-gpu.lock}
if [[ "$PIN_MODE" != on && "$PIN_MODE" != off ]]; then
    echo "PIN_MODE must be on or off" >&2
    exit 3
fi
if [[ "$DIAGNOSTICS" == 1 ]]; then
    RUN="$OUTROOT/R_0050k_profileA_async_${PIN_MODE}_diag"
elif [[ "$DIAGNOSTICS" == 0 ]]; then
    RUN="$OUTROOT/R_0050k_profileA_async_${PIN_MODE}_clean"
else
    echo "DIAGNOSTICS must be 0 or 1" >&2
    exit 3
fi
mkdir -p "$RUN"

[[ "$PIN_MODE" == off || -n "$PIN_PLAN" ]] || {
    echo "PIN_PLAN is required when PIN_MODE=on" >&2
    exit 3
}
for required in "$BIN" "$MODEL"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 3; }
done
[[ "$PIN_MODE" == off || -f "$PIN_PLAN" ]] || {
    echo "missing required file: $PIN_PLAN" >&2
    exit 3
}
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 3 )) || {
    echo "invalid repeats (n>=3 required): $REPEATS" >&2
    exit 3
}
[[ "$PIN_BUDGET_GB" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid PIN_BUDGET_GB: $PIN_BUDGET_GB" >&2
    exit 3
}
[[ "$STAGE_DEPTH" =~ ^[1-9][0-9]*$ ]] && (( STAGE_DEPTH <= 16 )) || {
    echo "invalid STAGE_DEPTH (expected 1..16): $STAGE_DEPTH" >&2
    exit 3
}

exec 9>"$GPU_LOCK"
flock -w 60 9 || { echo "could not acquire $GPU_LOCK" >&2; exit 4; }
pgrep -x ds4-server >/dev/null 2>&1 && { echo "another ds4-server is active" >&2; exit 4; }

SERVER_ENV=(
    "CUDA_VISIBLE_DEVICES=0"
    "DS4_ASYNC_PIPELINE=1"
    "DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1"
    "DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER=1"
    "DS4_CUDA_SELECTED_STAGE_DEPTH=$STAGE_DEPTH"
    "DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1"
    "DS4_CUDA_NO_DIRECT_IO=1"
    "DS4_CUDA_KEEP_MODEL_PAGES=1"
    "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256"
    "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=4"
    "HOME=/root"
    "LANG=C.UTF-8"
    "LC_ALL=C.UTF-8"
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    "LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64"
    "TMPDIR=/tmp"
)
if [[ "$PIN_MODE" == on ]]; then
    SERVER_ENV+=(
        "DS4_CUDA_STREAM_FROM_RAM_MASKED=$PIN_PLAN"
        "DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=$PIN_BUDGET_GB"
    )
fi
if [[ "$DIAGNOSTICS" == 1 ]]; then
    SERVER_ENV+=(
        "DS4_CUDA_TRANSPORT_DIAG=1"
        "DS4_SPEX_STATS=1"
        "DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1"
    )
fi
if [[ "$TRANSPORT_DEBUG" == 1 ]]; then
    SERVER_ENV+=("DS4_CUDA_TRANSPORT_DEBUG=1")
fi
SERVER_CLI=(
    -m "$MODEL" --cuda --ssd-streaming
    --ssd-streaming-cache-experts 8GB
    --prefill-chunk 1024
    --ctx 32768 -n "$(((MAX_TOKENS > WARMUP_TOKENS ? MAX_TOKENS : WARMUP_TOKENS) + 32))"
    --host 127.0.0.1 --port "$PORT" --cors
)

printf '%s\n' "${SERVER_ENV[@]}" | LC_ALL=C sort > "$RUN/server_env.txt"
printf '%q ' "$BIN" "${SERVER_CLI[@]}" > "$RUN/server_argv.txt"
printf '\n' >> "$RUN/server_argv.txt"
if [[ "$PIN_MODE" == on ]]; then
    sha256sum "$BIN" "$PIN_PLAN" "$0" > "$RUN/input.sha256"
else
    sha256sum "$BIN" "$0" > "$RUN/input.sha256"
fi
stat -c 'path=%n size=%s mtime=%y' "$MODEL" > "$RUN/model.stat"
model_bytes=$(stat -c '%s' "$MODEL")
[[ "$model_bytes" == 86720111488 ]] || {
    echo "unexpected model size: $model_bytes" >&2
    exit 3
}
printf '%s  %s\n' \
    efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668 \
    "$MODEL" > "$RUN/model.expected.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max \
    --format=csv,noheader > "$RUN/gpu.txt"
cat /sys/fs/cgroup/memory.max > "$RUN/cgroup_memory_max.txt"
cat /sys/fs/cgroup/memory.current > "$RUN/cgroup_memory_before.txt"

python3 - "$RUN" "$WARMUP_TOKENS" "$MAX_TOKENS" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
warmup_tokens = int(sys.argv[2])
max_tokens = int(sys.argv[3])
prompt = (
    "Genera una pagina HTML5 minima e VALIDA per una caffetteria Bean & Brew: "
    "doctype, head con title, body con nav (Home, Menu, Contatti), un h1 e un "
    "bottone Ordina. Chiudi TUTTI i tag fino a </html>."
)
base = {
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "think": False,
    "thinking": {"type": "disabled"},
    "stream": False,
}
(run / "request_warmup.json").write_text(
    json.dumps(dict(base, max_tokens=warmup_tokens), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(run / "request_measured.json").write_text(
    json.dumps(dict(base, max_tokens=max_tokens), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

env -i "${SERVER_ENV[@]}" "$BIN" "${SERVER_CLI[@]}" \
    >"$RUN/server.stdout.log" 2>"$RUN/server.stderr.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN/server.pid"

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$SERVER_PID" 2>/dev/null && kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 1800); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
[[ "$ready" == 1 ]] || { echo server_not_ready > "$RUN/STOP_REASON.txt"; exit 5; }

if [[ "$PIN_MODE" == on ]]; then
python3 - "$RUN/server.stderr.log" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
m = re.findall(
    r"stream-from-RAM masked window: .*?(\d+)/(\d+) ranges? registered, ([0-9.]+) GiB zero-copy",
    text,
)
if not m:
    raise SystemExit("missing masked registration summary")
registered, requested, gib = m[-1]
if (int(registered), int(requested)) != (120, 120) or float(gib) < 67.0:
    raise SystemExit(
        f"insufficient masked registration: {registered}/{requested}, {gib} GiB"
    )
PY
fi

curl -fsS --max-time 3600 -H 'Content-Type: application/json' \
    --data @"$RUN/request_warmup.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    > "$RUN/response_warmup.json" 2> "$RUN/curl_warmup.err"

for i in $(seq 1 "$REPEATS"); do
    printf -v rep '%02d' "$i"
    start_ns=$(date +%s%N)
    curl -fsS --max-time 3600 -H 'Content-Type: application/json' \
        --data @"$RUN/request_measured.json" \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        > "$RUN/response_r${rep}.json" 2> "$RUN/curl_r${rep}.err"
    end_ns=$(date +%s%N)
    python3 - "$RUN" "$rep" "$start_ns" "$end_ns" <<'PY'
import hashlib
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
rep = sys.argv[2]
start_ns, end_ns = map(int, sys.argv[3:5])
response = json.loads((run / f"response_r{rep}.json").read_text(encoding="utf-8"))
content = response["choices"][0]["message"]["content"]
raw = content.encode("utf-8")
(run / f"content_r{rep}.txt").write_bytes(raw)
meta = {
    "rep": int(rep),
    "wall_seconds": (end_ns - start_ns) / 1e9,
    "content_bytes": len(raw),
    "content_sha256": hashlib.sha256(raw).hexdigest(),
    "usage": response.get("usage", {}),
    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
}
(run / f"result_r{rep}.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
done

python3 - "$RUN" "$REPEATS" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
rows = [json.loads((run / f"result_r{i:02d}.json").read_text())
        for i in range(1, int(sys.argv[2]) + 1)]
hashes = [row["content_sha256"] for row in rows]
summary = {
    "repeats": len(rows),
    "byte_identical": len(set(hashes)) == 1,
    "distinct_content_hashes": sorted(set(hashes)),
    "runs": rows,
    "scope": "same-node 0050k safe-async Profile A protocol with bounded pinning and no compute mask",
    "diagnostics_enabled": (run.name.endswith("_diag")),
    "pin_mode": ("on" if "_on_" in run.name else "off"),
}
(run / "exactness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

nvidia-smi --query-gpu=timestamp,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader > "$RUN/gpu_after.txt"
cat /sys/fs/cgroup/memory.current > "$RUN/cgroup_memory_after.txt"
cat /sys/fs/cgroup/memory.peak > "$RUN/cgroup_memory_peak.txt" 2>/dev/null || true

echo complete > "$RUN/STOP_REASON.txt"
