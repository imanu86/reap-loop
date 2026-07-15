#!/usr/bin/env bash
# Functional quality gate for Windows bake candidate masks.
# Linux is used only as a model oracle; throughput is not a Windows result.
set -euo pipefail

REPO=${REPO:-/root/reap-loop}
BIN=${BIN:-/root/ds4-fullstack/ds4-server}
MODEL=${MODEL:-/root/models/ds4-2bit.gguf}
LEARN=${LEARN:-$REPO/runs/ds4/20260712_pod12_bake/coding_mass_prefill_seed_20260715}
OUT=${OUT:-$REPO/runs/ds4/20260712_pod12_bake/windows_bake_quality_ab_v2_20260715}
PORT=${PORT:-18083}
GPU_LOCK=${GPU_LOCK:-/tmp/ds4-gpu.lock}
MAX_TOKENS=${MAX_TOKENS:-3200}

test -x "$BIN"
test -s "$MODEL"
test -f "$LEARN/masks/k60_coding.txt"
test -f "$LEARN/masks/k65_coding.txt"
mkdir -p "$OUT"

exec 9>"$GPU_LOCK"
flock -w 60 9 || { echo "could not acquire GPU lock" >&2; exit 4; }

PROMPT='Crea una dashboard HTML5/CSS/JavaScript single-file e COMPATTA per monitorare build di software. Deve includere nav, hero con stato generale, tabella di job, filtro testuale, pulsante per rilanciare una build, modulo per aggiungere un job e popup di conferma JavaScript. Tutte le interazioni devono funzionare senza librerie esterne. Mantieni il CSS essenziale, completa tutto entro circa 1800 token, chiudi </html> e restituisci soltanto HTML valido.'

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup_server EXIT

run_arm() {
    local arm=$1 mask=$2
    local dir="$OUT/$arm"
    mkdir -p "$dir"
    if pgrep -f '/root/ds4-fullstack/ds4-server' >/dev/null; then
        echo "another DS4 server is active" >&2
        exit 4
    fi

    export DS4_CUDA_NO_DIRECT_IO=1
    export DS4_CUDA_KEEP_MODEL_PAGES=1
    export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
    export DS4_CUDA_NO_Q8_F16_CACHE=1
    export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
    export DS4_PACE=0
    unset DS4_SPEX_TRACE_ROUTING DS4_SPEX_TRACE_ROUTING_WEIGHTS
    unset DS4_SPEX_TRACE_PREFILL_ROUTING
    if [[ "$mask" == NONE ]]; then
        unset DS4_REAP_MASK_FILE
    else
        export DS4_REAP_MASK_FILE="$mask"
    fi

    env | LC_ALL=C sort > "$dir/server_env.txt"
    printf '%q ' "$BIN" -m "$MODEL" --cuda --ssd-streaming \
        --ssd-streaming-cache-experts 1024 --prefill-chunk 512 \
        -c 4096 -n "$((MAX_TOKENS + 128))" --host 127.0.0.1 --port "$PORT" --cors \
        > "$dir/server_argv.txt"
    printf '\n' >> "$dir/server_argv.txt"
    sha256sum "$BIN" > "$dir/binary.sha256"
    if [[ "$mask" != NONE ]]; then sha256sum "$mask" > "$dir/mask.sha256"; fi
    stat -c 'path=%n size=%s mtime=%y' "$MODEL" > "$dir/model.stat"

    "$BIN" -m "$MODEL" --cuda --ssd-streaming \
        --ssd-streaming-cache-experts 1024 --prefill-chunk 512 \
        -c 4096 -n "$((MAX_TOKENS + 128))" --host 127.0.0.1 --port "$PORT" --cors \
        >"$dir/server.stdout.log" 2>"$dir/server.stderr.log" &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$dir/server.pid"

    local ready=0
    for _ in $(seq 1 240); do
        if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            ready=1
            break
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 2
    done
    [[ "$ready" == 1 ]] || { echo "$arm server not ready" >&2; exit 5; }

    local temps=(0 0.2 0.7)
    for idx in 0 1 2; do
        local run=$((idx + 1)) temp=${temps[$idx]}
        python3 - "$dir/request_r0${run}.json" "$PROMPT" "$MAX_TOKENS" "$temp" <<'PY'
import json
import sys
path, prompt, max_tokens, temp = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
request = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "Rispondi direttamente e senza ragionamento visibile."},
        {"role": "user", "content": prompt},
    ],
    "max_tokens": max_tokens,
    "temperature": temp,
    "stream": True,
    "think": False,
    "thinking": {"type": "disabled"},
    "stream_options": {"include_usage": True},
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(request, f, ensure_ascii=False, indent=2)
PY
        python3 "$REPO/scripts/stream_stop_guard.py" \
            --url "http://127.0.0.1:$PORT/v1/chat/completions" \
            --request "$dir/request_r0${run}.json" \
            --response "$dir/response_r0${run}.json" \
            --events "$dir/events_r0${run}.jsonl" \
            --live-text "$dir/content_r0${run}.txt" \
            --timeout 1800 --stop-html-close --stop-repeat \
            --repeat-ngram 3 --repeat-window 120 --repeat-count 3
        python3 "$REPO/scripts/functional_grade.py" frontpage \
            "$dir/content_r0${run}.txt" --json > "$dir/grade_r0${run}.json"
        echo "$arm r0${run}: $(cat "$dir/grade_r0${run}.json")"
    done
    cleanup_server
}

run_arm k0 NONE
run_arm k60 "$LEARN/masks/k60_coding.txt"
run_arm k65 "$LEARN/masks/k65_coding.txt"

python3 - "$OUT" <<'PY'
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
summary = {}
for arm in ("k0", "k60", "k65"):
    rows = []
    for run in range(1, 4):
        grade = json.load(open(out / arm / f"grade_r0{run}.json"))
        response = json.load(open(out / arm / f"response_r0{run}.json"))
        rows.append({
            "run": run,
            "grade": grade,
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            "client_stop": response.get("client_stop"),
            "elapsed_s": response.get("elapsed_s"),
            "usage": response.get("usage"),
        })
    summary[arm] = rows
with open(out / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY

date -u +%FT%TZ > "$OUT/DONE"
