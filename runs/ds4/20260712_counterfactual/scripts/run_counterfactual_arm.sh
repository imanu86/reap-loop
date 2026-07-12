#!/usr/bin/env bash
# Counterfactual-admission (0046) fail-fast protocol runner.
# Arm A: adaptive-K (0045) + counterfactual admission fix ON, SPEX off.
# Arm B: fixed K23 + soft-mask bias, SPEX off (informational comparison).
set -euo pipefail

ARM=${1:-}
RUN_N=${2:-}
if [[ "$ARM" != "A" && "$ARM" != "B" ]] || [[ -z "$RUN_N" ]]; then
  echo "usage: $0 A|B RUN_N" >&2
  exit 2
fi

REPO=${REAP_LOOP_REPO:-/mnt/c/Users/imanu/source/repos/reap-loop}
OUTROOT="$REPO/runs/ds4/20260712_counterfactual"
RUN_ID="arm${ARM}_run${RUN_N}"
OUT="$OUTROOT/$RUN_ID"
BIN=${DS4_BIN:-/root/ds4-cf-work/ds4-server}
MODEL=${DS4_MODEL:-/root/models/ds4-2bit.gguf}
SOURCE_REQUEST=${DS4_REQUEST:-$REPO/runs/ds4/20260709_k23_unit_vs_weighted_cache256_html800/html_local_k23_weighted_warmup_cache256_r01/request_measured.json}
PORT=8073
MAX_TOKENS=${MAX_TOKENS:-4000}
CTX=${CTX:-6144}

if pgrep -x ds4-server >/dev/null; then
  echo "ds4-server is already running; refusing to disturb it" >&2
  exit 1
fi

mkdir -p "$OUT"

cat > "$OUT/RUN_META.txt" <<META
run_id=$RUN_ID
arm=$ARM
patch_chain=0011,...,0040,0041,0042,0043,0044,0045,0046-counterfactual-admission
binary=$BIN
model=$MODEL
port=$PORT
max_tokens=$MAX_TOKENS
ctx=$CTX
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
watchdog_kill_file=$OUT/WATCHDOG_KILL.txt
tripwire_stop_file=$OUT/TRIPWIRE_STOP.json
server_pid_file=$OUT/server.pid
live_stream_file=$OUT/stream_live.txt
META

if [[ "$ARM" == "A" ]]; then
  cat >> "$OUT/RUN_META.txt" <<'META'
config=adaptive-K 0045 + counterfactual admission fix (DS4_PACE_ADAPTIVE_CF_ADMIT=1), SPEX off (add0)
K=16..50 threshold=0.15 gain=0.5 step=+4/-1 update_every=2 deadband=2
META
else
  cat >> "$OUT/RUN_META.txt" <<'META'
config=fixed K23 + soft-mask bias -2.0, livemask pin-by-mass, rotation neutralized (HYST=999), SPEX off
K=23 fixed, DS4_REAP_MASK_SOFT_BIAS=-2.0
META
fi

python3 - "$SOURCE_REQUEST" "$OUT/request.json" "$MAX_TOKENS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as src:
    request = json.load(src)
request["max_tokens"] = int(sys.argv[3])
request["stream"] = True
request["stream_options"] = {"include_usage": True}
with open(sys.argv[2], "w", encoding="utf-8") as dst:
    json.dump(request, dst, ensure_ascii=False, indent=2)
PY

# ---- baseline env (task GPU section) ----
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_PACE=0

# ---- LIVEMASK shared knobs ----
export DS4_PACE_LIVEMASK=1
export DS4_PACE_LIVEMASK_BOOTSTRAP=16
export DS4_PACE_LIVEMASK_WINDOW=10
export DS4_PACE_LIVEMASK_OBSERVE_TOP=16
export DS4_PACE_LIVEMASK_X=11
export DS4_PACE_LIVEMASK_MAX_SWAPS=1
export DS4_PACE_LIVEMASK_COOLDOWN=16
export DS4_PACE_LIVEMASK_HYST=999
export DS4_PACE_LIVEMASK_LOG="$OUT/livemask.jsonl"
export DS4_REAP_PIN_BY_MASS=1
export DS4_REAP_PREFETCH_THREADS=16
export DS4_REAP_PREFETCH_LOCK=1

# ---- SPEX off for both arms ----
export DS4_PACE_LIVEMASK_SPEX_ADD=0
export DS4_PACE_LIVEMASK_SPEX_CADENCE=2
export DS4_PACE_LIVEMASK_SPEX_LOG="$OUT/spex_mask.jsonl"
export DS4_PACE_LIVEMASK_SPEX_PIN_LOG="$OUT/spex_pin.jsonl"
export DS4_REAP_PREFETCH_DELTA=0
export DS4_PACE_LIVEMASK_SPEX_WRAP=0
export DS4_SPEX_HIDDEN_GPU_LOAD=0
export DS4_SPEX_HIDDEN_GPU_SCORE=0
export DS4_SPEX_HIDDEN_GPU_PREFETCH=0
export DS4_EXPERT_TIERING=observe
export DS4_EXPERT_TIERING_LOG=
export DS4_EXPERT_TIERING_LOG_IDS=0

if [[ "$ARM" == "A" ]]; then
  export DS4_PACE_LIVEMASK_K=16
  export DS4_PACE_LIVEMASK_K_ADAPTIVE=1
  export DS4_PACE_LIVEMASK_K_MIN=16
  export DS4_PACE_LIVEMASK_K_MAX=50
  export DS4_PACE_LIVEMASK_KNOCK_THRESHOLD=0.15
  export DS4_PACE_LIVEMASK_KNOCK_GAIN=0.5
  export DS4_PACE_LIVEMASK_KNOCK_STEP_UP=4
  export DS4_PACE_LIVEMASK_KNOCK_STEP_DOWN=1
  export DS4_PACE_LIVEMASK_KNOCK_DEADBAND=2
  export DS4_PACE_LIVEMASK_KNOCK_UPDATE_EVERY=2
  export DS4_PACE_LIVEMASK_KNOCK_MIN_HISTORY=2
  export DS4_PACE_LIVEMASK_KNOCK_PREFETCH=0
  export DS4_PACE_ADAPTIVE_CF_ADMIT=1
  export DS4_PACE_ADAPTIVE_CF_ADMIT_WINDOW=3
  unset DS4_REAP_MASK_SOFT_BIAS
else
  export DS4_PACE_LIVEMASK_K=23
  export DS4_PACE_LIVEMASK_K_ADAPTIVE=0
  export DS4_PACE_ADAPTIVE_CF_ADMIT=0
  export DS4_REAP_MASK_SOFT_BIAS=-2.0
fi

env | LC_ALL=C sort > "$OUT/server_env.txt"
git -C "$(dirname "$BIN")" rev-parse HEAD > "$OUT/ds4_work_tree_base_commit.txt" 2>&1 || true
git -C "$REPO" rev-parse HEAD > "$OUT/reap_loop_commit.txt" 2>&1 || true

"$BIN" -m "$MODEL" --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 400 --prefill-chunk 512 \
  -c "$CTX" -n "$((MAX_TOKENS + 256))" --host 127.0.0.1 --port "$PORT" --cors \
  >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
srv_pid=$!
echo "$srv_pid" > "$OUT/server.pid"
cleanup() {
  if kill -0 "$srv_pid" 2>/dev/null; then
    kill "$srv_pid" 2>/dev/null || true
    wait "$srv_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 150); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$srv_pid" 2>/dev/null; then
    echo "server died before becoming ready" >&2
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "server not ready" >&2
  echo "server_not_ready" > "$OUT/STOP_REASON.txt"
  exit 1
fi

# Start the tripwire monitor in the background; it self-stops on
# response.json, WATCHDOG_KILL.txt, or its own trigger.
python3 "$OUTROOT/scripts/cf_tripwire_monitor.py" \
  --run-dir "$OUT" \
  --pid-file "$OUT/server.pid" \
  --livemask-log "$OUT/livemask.jsonl" \
  --events-log "$OUT/stream_events.jsonl" \
  --response "$OUT/response.json" \
  --watchdog-file "$OUT/WATCHDOG_KILL.txt" \
  --out "$OUT/tripwire_summary.json" \
  --stop-file "$OUT/TRIPWIRE_STOP.json" \
  >"$OUT/tripwire_monitor.log" 2>&1 &
mon_pid=$!

set +e
python3 "$REPO/scripts/stream_stop_guard.py" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --request "$OUT/request.json" \
  --response "$OUT/response.json" \
  --events "$OUT/stream_events.jsonl" \
  --live-text "$OUT/stream_live.txt" \
  --timeout 7200 \
  --stop-html-close \
  --stop-repeat \
  --repeat-ngram 3 \
  --repeat-window 120 \
  --repeat-count 3
guard_rc=$?
set -e

kill "$mon_pid" 2>/dev/null || true
wait "$mon_pid" 2>/dev/null || true

# ---- determine motivo di stop ----
if [[ -f "$OUT/WATCHDOG_KILL.txt" ]]; then
  echo "killed_by_external_watchdog" > "$OUT/STOP_REASON.txt"
elif [[ -f "$OUT/TRIPWIRE_STOP.json" ]]; then
  python3 -c "import json,sys; print('tripwire:' + json.load(open(sys.argv[1]))['reason'])" \
    "$OUT/TRIPWIRE_STOP.json" > "$OUT/STOP_REASON.txt" || echo "tripwire:unknown" > "$OUT/STOP_REASON.txt"
elif [[ -f "$OUT/response.json" ]]; then
  python3 -c "
import json
r = json.load(open('$OUT/response.json'))
cs = r.get('client_stop')
se = r.get('stream_error')
if se:
    print('stream_error:' + se)
elif cs:
    print('client_stop:' + cs.get('reason', 'unknown'))
else:
    print('finished_or_budget')
" > "$OUT/STOP_REASON.txt"
else
  echo "guard_rc=$guard_rc no_response_file" > "$OUT/STOP_REASON.txt"
fi

cat "$OUT/STOP_REASON.txt"
echo "run dir: $OUT"
