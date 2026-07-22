#!/usr/bin/env bash
# Phase-1 server transport safety/fault matrix. Each arm starts a fresh server
# because the fault hooks and stage depth are process-scoped environment state.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 DS4_SERVER MODEL COMPUTE_MASK PROMPT_FILE OUTROOT [NTOK=16]" >&2
    exit 2
fi

BIN=$1
MODEL=$2
COMPUTE_MASK=$3
PROMPT_FILE=$4
OUTROOT=$5
NTOK=${6:-16}
CTX=${CTX:-2048}
PORT=${PORT:-8097}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-2400}
GPU_LOCK=${GPU_LOCK:-/tmp/ds4-gpu.lock}

for required in "$BIN" "$MODEL" "$COMPUTE_MASK" "$PROMPT_FILE"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 3; }
done
[[ "$NTOK" =~ ^[1-9][0-9]*$ ]] || { echo "NTOK must be positive" >&2; exit 3; }

mkdir -p "$OUTROOT"
sha256sum "$BIN" "$COMPUTE_MASK" "$PROMPT_FILE" "$0" > "$OUTROOT/inputs.sha256"
stat -c 'path=%n size=%s mtime=%y' "$MODEL" > "$OUTROOT/model.stat"
exec 9>"$GPU_LOCK"
flock -w 60 9 || { echo "could not acquire $GPU_LOCK" >&2; exit 4; }
pgrep -x ds4-server >/dev/null 2>&1 && {
    echo "another ds4-server is active" >&2
    exit 4
}

python3 - "$PROMPT_FILE" "$OUTROOT/request.json" "$NTOK" <<'PY'
import json
import pathlib
import sys

prompt = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip()
request = {
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "think": False,
    "thinking": {"type": "disabled"},
    "stream": False,
    "max_tokens": int(sys.argv[3]),
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

depths=(1 8)
all_faults=(stage_event_create stage_event_record upload_event_create upload_event_record upload_event_wait)
declare -A fault_env=(
    [stage_event_create]=DS4_CUDA_FAULT_STAGE_EVENT_CREATE
    [stage_event_record]=DS4_CUDA_FAULT_STAGE_EVENT_RECORD
    [stage_event_wait]=DS4_CUDA_FAULT_STAGE_EVENT_WAIT
    [upload_event_create]=DS4_CUDA_FAULT_UPLOAD_EVENT_CREATE
    [upload_event_record]=DS4_CUDA_FAULT_UPLOAD_EVENT_RECORD
    [upload_event_wait]=DS4_CUDA_FAULT_UPLOAD_EVENT_WAIT
)
declare -A fault_counter=(
    [stage_event_create]=stage_event_create_failures
    [stage_event_record]=stage_event_record_failures
    [stage_event_wait]=war_wait_failures
    [upload_event_create]=upload_event_create_failures
    [upload_event_record]=upload_event_record_failures
    [upload_event_wait]=upload_event_wait_failures
)
counter_names=(stage_event_create_failures stage_event_record_failures war_wait_failures upload_event_create_failures upload_event_record_failures upload_event_wait_failures)

diag_field() {
    sed -n "s/.*[[:space:]]$2=\([0-9][0-9]*\).*/\1/p" <<<"$1"
}
require_eq() {
    local got
    got=$(diag_field "$1" "$2")
    [[ -n "$got" && "$got" == "$3" ]] || {
        echo "expected $2=$3, got ${got:-missing}" >&2
        return 1
    }
}
require_gt_zero() {
    local got
    got=$(diag_field "$1" "$2")
    [[ -n "$got" && "$got" -gt 0 ]] || {
        echo "expected $2>0, got ${got:-missing}" >&2
        return 1
    }
}
require_sum_eq() {
    local total left right
    total=$(diag_field "$1" "$2")
    left=$(diag_field "$1" "$3")
    right=$(diag_field "$1" "$4")
    [[ -n "$total" && -n "$left" && -n "$right" && "$total" -eq $((left + right)) ]] || {
        echo "expected $2=$3+$4, got ${total:-missing}=${left:-missing}+${right:-missing}" >&2
        return 1
    }
}

server_pid=0
server_stop_failed=0
stop_server() {
    local forced=0 status=0
    server_stop_failed=0
    if [[ "$server_pid" -gt 0 ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            forced=1
            kill -KILL -- "-$server_pid" 2>/dev/null || true
        fi
    fi
    if [[ "$server_pid" -gt 0 ]]; then
        set +e
        wait "$server_pid" 2>/dev/null
        status=$?
        set -e
        if [[ -n "${run:-}" && -d "$run" ]]; then
            printf 'status=%s forced_sigkill=%s\n' "$status" "$forced" \
                > "$run/server_exit_status.txt"
        fi
        if [[ "$forced" == 1 || ( "$status" != 0 && "$status" != 143 ) ]]; then
            server_stop_failed=1
        fi
    fi
    server_pid=0
}
trap stop_server EXIT
trap 'stop_server; exit 130' INT
trap 'stop_server; exit 143' TERM

printf 'depth\tfault\tcontent_sha256\tcontent_bytes\tfinish_reason\tcompletion_tokens\tdiag\n' > "$OUTROOT/summary.tsv"
reference_hash=
reference_finish=
reference_tokens=

for depth in "${depths[@]}"; do
    faults=(baseline)
    [[ "$depth" == 1 ]] && faults+=("${all_faults[@]}")
    [[ "$depth" == 8 ]] && faults+=(stage_event_wait)
    for fault in "${faults[@]}"; do
        arm="depth${depth}_${fault}"
        run="$OUTROOT/$arm"
        mkdir -p "$run"

        envset=(
            "CUDA_VISIBLE_DEVICES=0"
            "DS4_ASYNC_PIPELINE=1"
            "DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1"
            "DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER=1"
            "DS4_CUDA_SELECTED_STAGE_DEPTH=$depth"
            "DS4_CUDA_TRANSPORT_DIAG=1"
            "DS4_CUDA_TRANSPORT_DEBUG=1"
            "DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1"
            "DS4_CUDA_NO_DIRECT_IO=1"
            "DS4_CUDA_KEEP_MODEL_PAGES=1"
            "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1"
            "DS4_CUDA_NO_Q8_F16_CACHE=1"
            "DS4_CUDA_WEIGHT_CACHE_VERBOSE=1"
            "DS4_REAP_MASK_FILE=$COMPUTE_MASK"
            "DS4_PACE=0"
            "HOME=/root"
            "LANG=C.UTF-8"
            "LC_ALL=C.UTF-8"
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            "LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64"
            "TMPDIR=/tmp"
        )
        [[ "$fault" == baseline ]] || envset+=("${fault_env[$fault]}=1")
        argv=(-m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 6
              --prefill-chunk 512 --ctx "$CTX" -n "$NTOK"
              --host 127.0.0.1 --port "$PORT" --cors)

        printf '%s\n' "${envset[@]}" | LC_ALL=C sort > "$run/env.txt"
        printf '%q ' "$BIN" "${argv[@]}" > "$run/argv.txt"
        printf '\n' >> "$run/argv.txt"

        setsid env -i "${envset[@]}" "$BIN" "${argv[@]}" \
            >"$run/server.stdout.log" 2>"$run/server.stderr.log" &
        server_pid=$!
        echo "$server_pid" > "$run/server.pid"

        ready=0
        for _ in $(seq 1 "$TIMEOUT_SECONDS"); do
            kill -0 "$server_pid" 2>/dev/null || break
            if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
                ready=1
                break
            fi
            sleep 1
        done
        [[ "$ready" == 1 ]] || { echo "server not ready: $arm" >&2; exit 10; }

        curl -fsS --max-time "$TIMEOUT_SECONDS" -H 'Content-Type: application/json' \
            --data @"$OUTROOT/request.json" \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            > "$run/response.json" 2> "$run/curl.err" || {
                echo "request failed: $arm" >&2
                exit 10
            }
        stop_server
        [[ "$server_stop_failed" == 0 ]] || {
            echo "server cleanup failed: $arm" >&2
            exit 10
        }

        python3 - "$run/response.json" "$run/content.txt" "$run/result.json" <<'PY'
import hashlib
import json
import pathlib
import sys

response = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
content = response["choices"][0]["message"]["content"]
raw = content.encode("utf-8")
pathlib.Path(sys.argv[2]).write_bytes(raw)
result = {
    "content_bytes": len(raw),
    "content_sha256": hashlib.sha256(raw).hexdigest(),
    "finish_reason": response["choices"][0].get("finish_reason"),
    "usage": response.get("usage", {}),
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
PY

        grep -F "CUDA streaming selected stage ring depth=$depth" "$run/server.stderr.log" >/dev/null
        diag=$(grep -F "CUDA transport safety final" "$run/server.stderr.log" | tail -n 1 || true)
        [[ -n "$diag" ]] || { echo "missing final diagnostic: $arm" >&2; exit 11; }
        require_gt_zero "$diag" stage_chunks
        require_gt_zero "$diag" slot_reuses
        require_eq "$diag" fallback_sync_failures 0
        require_eq "$diag" cleanup_device_sync_failures 0
        require_eq "$diag" staged_h2d_failures 0
        require_eq "$diag" stale_slot_reads 0
        require_sum_eq "$diag" fallback_syncs recovery_syncs sync_mode_syncs

        if [[ "$fault" == baseline ]]; then
            require_eq "$diag" sync_stage_mode 0
            require_eq "$diag" fallback_syncs 0
            require_eq "$diag" recovery_syncs 0
            require_eq "$diag" sync_mode_syncs 0
            require_eq "$diag" sync_mode_latches 0
            require_gt_zero "$diag" cursor_advances
            require_gt_zero "$diag" war_waits
            for counter in "${counter_names[@]}"; do require_eq "$diag" "$counter" 0; done
        else
            grep -F "CUDA transport fault injected: $fault" "$run/server.stderr.log" >/dev/null
            grep -F "CUDA transport safe synchronous staging mode latched" "$run/server.stderr.log" >/dev/null
            require_eq "$diag" sync_stage_mode 1
            require_eq "$diag" sync_mode_latches 1
            require_gt_zero "$diag" fallback_syncs
            require_gt_zero "$diag" recovery_syncs
            require_gt_zero "$diag" sync_mode_syncs
            for counter in "${counter_names[@]}"; do
                expected=0
                [[ "$counter" == "${fault_counter[$fault]}" ]] && expected=1
                require_eq "$diag" "$counter" "$expected"
            done
        fi

        read -r content_hash content_bytes finish_reason completion_tokens < <(
            python3 - "$run/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print(r["content_sha256"], r["content_bytes"], r["finish_reason"],
      r.get("usage", {}).get("completion_tokens", -1))
PY
        )
        [[ "$content_bytes" -gt 0 ]] || { echo "empty output: $arm" >&2; exit 12; }
        if [[ -z "$reference_hash" ]]; then
            reference_hash=$content_hash
            reference_finish=$finish_reason
            reference_tokens=$completion_tokens
        elif [[ "$content_hash" != "$reference_hash" ||
                "$finish_reason" != "$reference_finish" ||
                "$completion_tokens" != "$reference_tokens" ]]; then
            echo "output canary mismatch: $arm" >&2
            exit 13
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$depth" "$fault" "$content_hash" "$content_bytes" "$finish_reason" \
            "$completion_tokens" "$diag" \
            >> "$OUTROOT/summary.tsv"
    done
done

echo "PASS: baseline depth 1/8, five event faults at depth 1, stage WAR-wait fault at depth 8, no stale reads"
