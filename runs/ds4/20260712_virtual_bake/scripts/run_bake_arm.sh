#!/usr/bin/env bash
# Virtual-bake (2026-07-12) V1 measured-run driver: static REAP mask + full RAM/page-cache
# residency, existing ds4-server binary, no new code.
#
# Usage: run_bake_arm.sh <arm_name> <mask_file_or_NONE> <run_n> [max_tokens] [ctx]
#
# Contract (external watchdog + mission spec):
#   <OUT>/RUN_META.txt, server.pid, stream_live.txt, response.json, STOP_REASON.txt
# Kill discipline: ONLY `kill $(cat "$OUT/server.pid")`, never pkill.
# GPU serialization: flock on /tmp/ds4-gpu.lock for the whole GPU-resident phase.
# RAM safety (orchestrator hard constraint): MemAvailable must never drop below
# RAM_FLOOR_MB during the run; a background monitor logs free -m every 30s and
# kills the server immediately (via pid file) if the floor is breached.
set -uo pipefail

ARM=${1:?arm name}
MASK=${2:?mask file path or NONE}
RUN_N=${3:?run number}
MAX_TOKENS=${4:-4000}
CTX=${5:-4096}

REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUTROOT="$REPO/runs/ds4/20260712_virtual_bake"
RUN_ID="arm_${ARM}_run${RUN_N}"
OUT="$OUTROOT/$RUN_ID"
BIN=/root/ds4-fullstack/ds4-server
MODEL=/root/models/ds4-2bit.gguf
PORT=8081
GPU_LOCK=/tmp/ds4-gpu.lock
RAM_FLOOR_MB=7168   # ~7 GiB hard floor, orchestrator instruction

if pgrep -x ds4-server >/dev/null; then
  echo "ds4-server already running; refusing to start a second one" >&2
  exit 1
fi

mkdir -p "$OUT"
PROMPT_TEXT='Crea una landing page HTML/CSS/JS single-file per un negozio di programmazione AI in stile cyberpunk. Deve avere un modulo contatti e un popup JS che dice richiesta inviata. Codice valido e compatto.'

python3 - "$OUT/request.json" "$MAX_TOKENS" "$PROMPT_TEXT" <<'PY'
import json, sys
out, max_tokens, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
req = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "Rispondi in modo diretto, utile e senza ragionamento visibile."},
        {"role": "user", "content": prompt},
    ],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
    "think": False,
    "thinking": {"type": "disabled"},
    "stream_options": {"include_usage": True},
}
json.dump(req, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY

python3 - "$OUT/warmup_request.json" "$PROMPT_TEXT" <<'PY'
import json, sys
out, prompt = sys.argv[1], sys.argv[2]
req = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "Rispondi in modo diretto, utile e senza ragionamento visibile."},
        {"role": "user", "content": prompt},
    ],
    "max_tokens": 40,
    "temperature": 0,
    "stream": False,
    "think": False,
    "thinking": {"type": "disabled"},
}
json.dump(req, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY

cat > "$OUT/RUN_META.txt" <<META
run_id=$RUN_ID
arm=$ARM
mask=$MASK
binary=$BIN
model=$MODEL
port=$PORT
max_tokens=$MAX_TOKENS
ctx=$CTX
ram_floor_mb=$RAM_FLOOR_MB
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
server_pid_file=$OUT/server.pid
live_stream_file=$OUT/stream_live.txt
META

git -C "$REPO" rev-parse HEAD > "$OUT/reap_loop_commit.txt" 2>&1 || true

echo "[$RUN_ID] waiting for GPU lock..."
exec 9>"$GPU_LOCK"
flock 9
echo "[$RUN_ID] GPU lock acquired $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/RUN_META.txt"

cleanup() {
  if [[ -f "$OUT/server.pid" ]]; then
    pid=$(cat "$OUT/server.pid" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -0 "$pid" 2>/dev/null && sleep 2 || true
    fi
  fi
  if [[ -n "${ram_mon_pid:-}" ]]; then
    kill "$ram_mon_pid" 2>/dev/null || true
  fi
  flock -u 9 2>/dev/null || true
}
trap cleanup EXIT

# ---- RAM safety monitor: free -m every 30s, log, hard-kill on floor breach ----
(
  while true; do
    line=$(free -m | awk '/^Mem:/{print $7}')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) MemAvailable_MB=$line" >> "$OUT/ram_log.txt"
    if [[ -n "$line" && "$line" -lt "$RAM_FLOOR_MB" ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RAM FLOOR BREACHED ($line < $RAM_FLOOR_MB) - killing server" >> "$OUT/ram_log.txt"
      echo "ram_floor_breach available_mb=$line" > "$OUT/RAM_KILL.txt"
      if [[ -f "$OUT/server.pid" ]]; then
        kill "$(cat "$OUT/server.pid")" 2>/dev/null || true
      fi
      break
    fi
    sleep 30
  done
) &
ram_mon_pid=$!

# ---- baseline V1 env (mission spec) ----
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_PACE=0
if [[ "$MASK" != "NONE" ]]; then
  export DS4_REAP_MASK_FILE="$MASK"
else
  unset DS4_REAP_MASK_FILE
fi
env | LC_ALL=C sort > "$OUT/server_env.txt"

"$BIN" -m "$MODEL" --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 400 --prefill-chunk 512 \
  -c "$CTX" -n "$((MAX_TOKENS + 256))" --host 127.0.0.1 --port "$PORT" --cors \
  >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
srv_pid=$!
echo "$srv_pid" > "$OUT/server.pid"

ready=0
for _ in $(seq 1 150); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
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
echo "[$RUN_ID] server ready $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- pre-warm (same prompt, short budget, non-stream) BEFORE measured ----
warmup_t0=$(date -u +%Y-%m-%dT%H:%M:%SZ); warmup_s0=$(date +%s.%N)
curl -fsS -m 1800 -H "Content-Type: application/json" \
  -d @"$OUT/warmup_request.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  > "$OUT/warmup_response.json" 2> "$OUT/warmup_curl.err"
warmup_rc=$?
warmup_s1=$(date +%s.%N); warmup_t1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
warmup_dur=$(python3 -c "print(round($warmup_s1 - $warmup_s0, 1))")
{
  echo "warmup_started_utc=$warmup_t0"
  echo "warmup_finished_utc=$warmup_t1"
  echo "warmup_duration_s=$warmup_dur"
  echo "warmup_curl_rc=$warmup_rc"
} >> "$OUT/RUN_META.txt"
if [[ "$warmup_rc" != 0 ]]; then
  echo "warmup_failed_rc=$warmup_rc" > "$OUT/STOP_REASON.txt"
  exit 1
fi
echo "[$RUN_ID] warmup done in ${warmup_dur}s"

# ---- measured stream ----
python3 "$REPO/scripts/stream_stop_guard.py" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --request "$OUT/request.json" \
  --response "$OUT/response.json" \
  --events "$OUT/stream_events.jsonl" \
  --live-text "$OUT/stream_live.txt" \
  --timeout 3600 \
  --stop-html-close \
  --stop-repeat \
  --repeat-ngram 3 \
  --repeat-window 120 \
  --repeat-count 3
guard_rc=$?

kill "$ram_mon_pid" 2>/dev/null || true
wait "$ram_mon_pid" 2>/dev/null || true
ram_mon_pid=""

if [[ -f "$OUT/RAM_KILL.txt" ]]; then
  echo "ram_floor_breach" > "$OUT/STOP_REASON.txt"
elif [[ -f "$OUT/response.json" ]]; then
  python3 -c "
import json
r = json.load(open('$OUT/response.json'))
cs = r.get('client_stop'); se = r.get('stream_error')
if se: print('stream_error:' + se)
elif cs: print('client_stop:' + cs.get('reason', 'unknown'))
else: print('finished_or_budget')
" > "$OUT/STOP_REASON.txt"
else
  echo "guard_rc=$guard_rc no_response_file" > "$OUT/STOP_REASON.txt"
fi

# ---- grade + summarize ----
python3 -c "
import json
r = json.load(open('$OUT/response.json'))
content = r['choices'][0]['message']['content']
open('$OUT/content.txt', 'w', encoding='utf-8').write(content)
usage = r.get('usage') or {}
print(json.dumps({'chars': len(content), 'usage': usage, 'elapsed_s': r.get('elapsed_s')}))
" > "$OUT/content_stats.json" 2>>"$OUT/server.stderr.log" || true

python3 "$REPO/scripts/functional_grade.py" frontpage "$OUT/content.txt" --json > "$OUT/grade.json" 2>>"$OUT/server.stderr.log" || true
python3 "$REPO/scripts/functional_grade.py" frontpage "$OUT/content.txt" > "$OUT/grade.txt" 2>>"$OUT/server.stderr.log" || true

echo "[$RUN_ID] STOP_REASON: $(cat "$OUT/STOP_REASON.txt" 2>/dev/null)"
echo "[$RUN_ID] GRADE: $(cat "$OUT/grade.txt" 2>/dev/null | head -1)"
echo "[$RUN_ID] run dir: $OUT"
