#!/usr/bin/env bash
# keep46 zero-copy test (2026-07-13) - verifies whether a keep46 mass mask
# (118/layer, 46.1%) fits entirely inside the 31 GiB WDDM-pinnable host arena
# (zero cold tier) and, if so, whether it renders (quality) and is faster
# than keep60 (which has a documented ~9 GiB cold tier that stalls decode).
#
# Modeled on:
#  - runs/ds4/20260713_decisive_residency/scripts/run_arm.sh (env/DIAG wiring)
#  - runs/ds4/20260712_virtual_bake/scripts/run_bake_arm.sh (grading pipeline)
#
# Uses an ISOLATED copy of the 0050-series zero-copy binary at
# /root/ds4-keep46-work (copied from /root/ds4-v2-work at the start of this
# test to avoid another agent's concurrent rebuild changing it underfoot).
#
# Kill discipline: ONLY `kill $(cat "$OUT/server.pid")`, never pkill.
# GPU serialization: flock /tmp/ds4-gpu.lock for the whole GPU-resident phase.
set -uo pipefail

CACHE_N=${1:-450}
WARMUP_TOK=${2:-400}
MEASURED_TOK=${3:-1600}
BUDGET_GB=${4:-31}

REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUTROOT="$REPO/runs/ds4/20260713_keep46_zerocopy"
RUN_ID="arm_keep46_zerocopy_cache${CACHE_N}_budget${BUDGET_GB}g"
OUT="$OUTROOT/arms/$RUN_ID"
BIN=/root/ds4-keep46-work/ds4-server
MODEL=/root/models/ds4-2bit.gguf
MASK="$OUTROOT/masks/mask46_self.txt"
PORT=8097
CTX=4096
PREFILL_CHUNK=512
GPU_LOCK=/tmp/ds4-gpu.lock
RAM_FLOOR_MB=7168

if pgrep -x ds4-server >/dev/null; then
  echo "ABORT: ds4-server already running" >&2
  exit 1
fi

mkdir -p "$OUT"
PROMPT_TEXT='Crea una landing page HTML/CSS/JS single-file per un negozio di programmazione AI in stile cyberpunk. Deve avere un modulo contatti e un popup JS che dice richiesta inviata. Codice valido e compatto.'

python3 - "$OUT/request.json" "$MEASURED_TOK" "$PROMPT_TEXT" <<'PY'
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

python3 - "$OUT/warmup_request.json" "$WARMUP_TOK" "$PROMPT_TEXT" <<'PY'
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
    "stream": False,
    "think": False,
    "thinking": {"type": "disabled"},
}
json.dump(req, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY

md5sum "$BIN" > "$OUT/binary.md5"
stat -c 'path=%n size=%s mtime=%y' "$MODEL" > "$OUT/model.stat" 2>/dev/null || true
( cd /root/ds4-keep46-work && git rev-parse HEAD ) > "$OUT/src_head.txt" 2>/dev/null || true
wc -l "$MASK" > "$OUT/mask_lines.txt"

cat > "$OUT/RUN_META.txt" <<META
run_id=$RUN_ID
arm=keep46_zerocopy
mask=$MASK
mask_keep_per_layer=118
mask_keep_pct=46.1
binary=$BIN
binary_source=/root/ds4-v2-work (isolated copy at test start)
model=$MODEL
port=$PORT
cache_experts=$CACHE_N
warmup_tokens=$WARMUP_TOK
measured_tokens=$MEASURED_TOK
budget_gb=$BUDGET_GB
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
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  if [[ -n "${ram_mon_pid:-}" ]]; then
    kill "$ram_mon_pid" 2>/dev/null || true
  fi
  flock -u 9 2>/dev/null || true
}
trap cleanup EXIT

# ---- RAM safety monitor ----
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

# ---- env: zero-copy masked host arena + pin-by-mass (rating-only) ----
SERVER_ENV=(
  "CUDA_VISIBLE_DEVICES=0"
  "DS4_CUDA_NO_DIRECT_IO=1"
  "DS4_CUDA_KEEP_MODEL_PAGES=1"
  "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1"
  "DS4_CUDA_NO_Q8_F16_CACHE=1"
  "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256"
  "DS4_PACE=0"
  "DS4_REAP_MASK_FILE=$MASK"
  "DS4_SPEX_STATS=1"
  "DS4_CUDA_STREAM_FROM_RAM_MASKED=$MASK"
  "DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=$BUDGET_GB"
  "DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1"
  "DS4_REAP_PIN_BY_MASS=1"
  "DS4_PACE_LIVEMASK=1"
  "DS4_PACE_LIVEMASK_RATING_ONLY=1"
  "HOME=/root" "LANG=C.UTF-8" "LC_ALL=C.UTF-8"
  "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
  "TMPDIR=/tmp"
)
printf '%s\n' "${SERVER_ENV[@]}" | LC_ALL=C sort > "$OUT/server_env.txt"

cd /root/ds4-keep46-work
env -i "${SERVER_ENV[@]}" "$BIN" -m "$MODEL" --cuda --ssd-streaming \
  --ssd-streaming-cache-experts "$CACHE_N" --prefill-chunk "$PREFILL_CHUNK" \
  -c "$CTX" -n "$((MEASURED_TOK + WARMUP_TOK + 256))" \
  --host 127.0.0.1 --port "$PORT" --cors \
  >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
srv_pid=$!
echo "$srv_pid" > "$OUT/server.pid"
echo "[$RUN_ID] server pid=$srv_pid"

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

# ---- registration/DIAG snapshot right after ready, before any request ----
sleep 2
grep -E 'masked|REAP mask applied|registered|pinned' "$OUT/server.stderr.log" > "$OUT/diag_post_ready.txt" 2>/dev/null || true

# ---- warmup (same prompt, WARMUP_TOK budget, non-stream) ----
warmup_t0=$(date -u +%Y-%m-%dT%H:%M:%SZ); warmup_s0=$(date +%s.%N)
curl -fsS -m 3600 -H "Content-Type: application/json" \
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

# baseline offset: DIAG lines AFTER this point are the WARM measured window
wc -c < "$OUT/server.stderr.log" > "$OUT/stderr_offset_baseline.txt"
tail -n 5 "$OUT/server.stderr.log" > "$OUT/diag_post_warmup.txt"

# ---- measured stream (WARM state) ----
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

# ---- final DIAG snapshot (whole run) ----
grep -E 'masked zero-copy diag final|SPEX stats' "$OUT/server.stderr.log" > "$OUT/diag_final.txt" 2>/dev/null || true
python3 - "$OUT/server.stderr.log" "$OUT/stderr_offset_baseline.txt" "$OUT/diag_measured_window.txt" <<'PY'
import sys
log_path, offset_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
offset = int(open(offset_path).read().strip())
with open(log_path, 'rb') as f:
    f.seek(offset)
    data = f.read()
with open(out_path, 'wb') as f:
    f.write(data)
PY
grep -E 'masked zero-copy diag|REAP mask applied|SPEX stats' "$OUT/diag_measured_window.txt" > "$OUT/diag_measured_window_filtered.txt" 2>/dev/null || true

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
