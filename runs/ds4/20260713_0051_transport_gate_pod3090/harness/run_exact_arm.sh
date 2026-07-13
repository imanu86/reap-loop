#!/usr/bin/env bash
# Repeated deterministic mechanism gate for one frozen transport arm.
# Starts one server, performs one warmup and N measured requests, then stops
# only the recorded PID. This is an exactness/repeatability gate, not a
# cold-start or cross-arm performance benchmark.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 ARM BIN COMPUTE_MASK OUTROOT [REPEATS=3]" >&2
    exit 2
fi

ARM=$1
BIN=$2
COMPUTE_MASK=$3
OUTROOT=$4
REPEATS=${5:-3}

MODEL=${MODEL:-/workspace/models/ds4-2bit.gguf}
PORT=${PORT:-8096}
CACHE_EXPERTS=${CACHE_EXPERTS:-400}
CTX=${CTX:-2048}
PREFILL_CHUNK=${PREFILL_CHUNK:-512}
MAX_TOKENS=${MAX_TOKENS:-60}
GPU_LOCK=${GPU_LOCK:-/tmp/ds4-gpu.lock}
PIN_PLAN=${PIN_PLAN:-}
PIN_BUDGET_GB=${PIN_BUDGET_GB:-24}
ENABLE_ASYNC=${ENABLE_ASYNC:-0}
ENABLE_PREFILL_S1=${ENABLE_PREFILL_S1:-0}

RUN="$OUTROOT/$ARM"
mkdir -p "$RUN"

for required in "$BIN" "$MODEL" "$COMPUTE_MASK"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required file: $required" >&2
        exit 3
    fi
done
if [[ -n "$PIN_PLAN" && ! -f "$PIN_PLAN" ]]; then
    echo "missing pin plan: $PIN_PLAN" >&2
    exit 3
fi
if [[ ! "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS must be a positive integer" >&2
    exit 3
fi

exec 9>"$GPU_LOCK"
if ! flock -w 60 9; then
    echo "could not acquire $GPU_LOCK" >&2
    exit 4
fi
if pgrep -x ds4-server >/dev/null 2>&1; then
    echo "another ds4-server is already running" >&2
    exit 4
fi

SERVER_ENV=(
    "CUDA_VISIBLE_DEVICES=0"
    "DS4_CUDA_NO_DIRECT_IO=1"
    "DS4_CUDA_KEEP_MODEL_PAGES=1"
    "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1"
    "DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE=1"
    "DS4_CUDA_NO_Q8_F16_CACHE=1"
    "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256"
    "DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1"
    "DS4_PACE=0"
    "DS4_REAP_MASK_FILE=$COMPUTE_MASK"
    "DS4_SPEX_STATS=1"
    "HOME=/root"
    "LANG=C.UTF-8"
    "LC_ALL=C.UTF-8"
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    "LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64"
    "TMPDIR=/tmp"
)

if [[ "$ENABLE_ASYNC" == 1 ]]; then
    SERVER_ENV+=("DS4_ASYNC_PIPELINE=1" "DS4_CUDA_SELECTED_STAGE_DEPTH=8")
fi
if [[ "$ENABLE_PREFILL_S1" == 1 ]]; then
    SERVER_ENV+=("DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1")
fi
if [[ -n "$PIN_PLAN" ]]; then
    SERVER_ENV+=(
        "DS4_CUDA_STREAM_FROM_RAM_MASKED=$PIN_PLAN"
        "DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=$PIN_BUDGET_GB"
        "DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1"
    )
fi

SERVER_CLI=(
    -m "$MODEL" --cuda --ssd-streaming
    --ssd-streaming-cache-experts "$CACHE_EXPERTS"
    --prefill-chunk "$PREFILL_CHUNK"
    -c "$CTX" -n "$((MAX_TOKENS + 32))"
    --host 127.0.0.1 --port "$PORT" --cors
)

printf '%s\n' "${SERVER_ENV[@]}" | LC_ALL=C sort > "$RUN/server_env.txt"
printf '%q ' "$BIN" "${SERVER_CLI[@]}" > "$RUN/server_argv.txt"
printf '\n' >> "$RUN/server_argv.txt"
sha256sum "$BIN" "$COMPUTE_MASK" > "$RUN/input.sha256"
if [[ -n "$PIN_PLAN" ]]; then
    sha256sum "$PIN_PLAN" >> "$RUN/input.sha256"
fi
stat -c 'path=%n size=%s mtime=%y' "$MODEL" > "$RUN/model.stat"
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max \
    --format=csv,noheader > "$RUN/gpu.txt"

python3 - "$RUN" "$MAX_TOKENS" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
max_tokens = int(sys.argv[2])
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
warmup = dict(base, max_tokens=16)
measured = dict(base, max_tokens=max_tokens)
(run / "request_warmup.json").write_text(
    json.dumps(warmup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(run / "request_measured.json").write_text(
    json.dumps(measured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

SERVER_STDOUT="$RUN/server.stdout.log"
SERVER_STDERR="$RUN/server.stderr.log"
env -i "${SERVER_ENV[@]}" "$BIN" "${SERVER_CLI[@]}" \
    >"$SERVER_STDOUT" 2>"$SERVER_STDERR" &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN/server.pid"

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 900); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    echo "server_not_ready" > "$RUN/STOP_REASON.txt"
    exit 5
fi

curl -fsS --max-time 1800 -H 'Content-Type: application/json' \
    --data @"$RUN/request_warmup.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    > "$RUN/response_warmup.json" 2> "$RUN/curl_warmup.err"

for i in $(seq 1 "$REPEATS"); do
    printf -v rep '%02d' "$i"
    start_ns=$(date +%s%N)
    curl -fsS --max-time 1800 -H 'Content-Type: application/json' \
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
(run / f"result_r{rep}.json").write_text(
    json.dumps(meta, indent=2) + "\n", encoding="ascii"
)
PY
done

python3 - "$RUN" "$REPEATS" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
rows = [json.loads((run / f"result_r{i:02d}.json").read_text())
        for i in range(1, repeats + 1)]
hashes = [row["content_sha256"] for row in rows]
summary = {
    "repeats": repeats,
    "byte_identical": len(set(hashes)) == 1,
    "distinct_content_hashes": sorted(set(hashes)),
    "runs": rows,
    "scope": "same-process deterministic repeatability; not a performance verdict",
}
(run / "exactness_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="ascii"
)
print(json.dumps(summary, indent=2))
PY

echo "complete" > "$RUN/STOP_REASON.txt"
