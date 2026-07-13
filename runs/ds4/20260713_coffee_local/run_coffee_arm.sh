#!/usr/bin/env bash
# Coffee-mask promo+prune zero-copy measurement on the REAL RTX 3060 / WSL.
# Arms: k83_promo (dynamic promotion ON), k83_staticpin (promotion OFF),
#       k65_promo, k100_promo.
# Serialized under /tmp/ds4-gpu.lock. Kills only its own pid. n=1 by design.
set -uo pipefail

ARM="${1:?usage: run_coffee_arm.sh <arm>}"

REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUTROOT="$REPO/runs/ds4/20260713_coffee_local"
MASKS="$OUTROOT/masks"
OUT="$OUTROOT/$ARM"
BIN=/root/ds4-v2-work/ds4-server
MODEL=/root/models/ds4-2bit.gguf
GUARD="$REPO/scripts/stream_stop_guard.py"
LOCK=/tmp/ds4-gpu.lock
PORT=8097
CTX=2048
MAXTOK=1000
NPRED=$((MAXTOK + 256))

case "$ARM" in
  k83_promo|k83_staticpin) MASK="$MASKS/mask_coffee_k83.txt";  BUDGET=30 ;;
  k65_promo)               MASK="$MASKS/mask_coffee_k65.txt";  BUDGET=24 ;;
  k100_promo)              MASK="$MASKS/mask_coffee_k100.txt"; BUDGET=31 ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

mkdir -p "$OUT"

# ---- serialize on the shared GPU lock (coexist with keep46) ----
exec 9>"$LOCK"
if ! flock -w 180 9; then
  echo "could not acquire $LOCK within 180s; aborting to avoid disturbing others" >&2
  exit 1
fi

if pgrep -x ds4-server >/dev/null 2>&1; then
  echo "another ds4-server is running; refusing to disturb it (lock held but a stray proc exists)" >&2
  exit 1
fi

# ---- build requests (coffee shop HTML, temp0) ----
PROMPT='Genera una pagina HTML5 minima e VALIDA per una caffetteria Bean & Brew: doctype, head con title, body con nav (Home, Menu, Contatti), un h1 e un bottone Ordina. Chiudi TUTTI i tag fino a </html>.'
python3 - "$OUT/request.json" "$OUT/warmup_request.json" "$MAXTOK" "$PROMPT" <<'PY'
import json, sys
resp, warm, maxtok, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
base = {"model":"deepseek-v4-flash",
        "messages":[{"role":"user","content":prompt}],
        "temperature":0,"think":False,"thinking":{"type":"disabled"}}
measured = dict(base); measured.update(max_tokens=maxtok, stream=True,
                                       stream_options={"include_usage":True})
warmup = dict(base); warmup.update(max_tokens=80, stream=False)
json.dump(measured, open(resp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(warmup, open(warm,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
PY

# ---- env: baseline zero-copy + pruning (bake mask) ----
COMMON_ENV=(
  HOME=/root USER=root LOGNAME=root SHELL=/bin/bash
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib
  PWD=/root/ds4-v2-work TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8
  CUDA_VISIBLE_DEVICES=0
  DS4_CUDA_NO_DIRECT_IO=1
  DS4_CUDA_KEEP_MODEL_PAGES=1
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
  DS4_CUDA_NO_Q8_F16_CACHE=1
  DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
  DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1
  DS4_PACE=0
  DS4_REAP_MASK_FILE="$MASK"
  DS4_REAP_PREFETCH=1
  DS4_REAP_PREFETCH_THREADS=16
  DS4_SPEX_STATS=0
  # zero-copy 0050
  DS4_CUDA_STREAM_FROM_RAM_MASKED="$MASK"
  DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB="$BUDGET"
  DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1
  # expert-cache hit/miss profiling
  DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE=1
)

# ---- promotion levers (rating-only publishes mass; bake mask authoritative) ----
PROMO_ENV=(
  DS4_PACE_LIVEMASK=1
  DS4_PACE_LIVEMASK_RATING_ONLY=1
  DS4_REAP_PIN_BY_MASS=1
  DS4_PACE_LIVEMASK_BOOTSTRAP=16
  DS4_PACE_LIVEMASK_WINDOW=10
  DS4_PACE_LIVEMASK_OBSERVE_TOP=16
  DS4_PACE_LIVEMASK_K=32
  DS4_PACE_LIVEMASK_X=11
  DS4_PACE_LIVEMASK_MAX_SWAPS=1
  DS4_PACE_LIVEMASK_COOLDOWN=16
  DS4_PACE_LIVEMASK_HYST=1
  DS4_PACE_LIVEMASK_PRESSURE=1
  DS4_PACE_LIVEMASK_PRESSURE_X=6
  DS4_PACE_LIVEMASK_PRESSURE_KNOCK=1
  DS4_PACE_LIVEMASK_PRESSURE_COOLDOWN=8
  DS4_PACE_LIVEMASK_PRESSURE_MAX_SWAPS=2
  DS4_PACE_LIVEMASK_LOG="$OUT/livemask.jsonl"
)

ENVSET=("${COMMON_ENV[@]}")
if [[ "$ARM" == *_promo ]]; then
  ENVSET+=("${PROMO_ENV[@]}")
fi

# record intended config
{ printf '%s\n' "${ENVSET[@]}"; } | LC_ALL=C sort > "$OUT/server_env.txt"
{
  echo "arm=$ARM"
  echo "binary=$BIN"
  echo "binary_md5=$(md5sum "$BIN" | awk '{print $1}')"
  echo "model=$MODEL"
  echo "mask=$MASK"
  echo "budget_gb=$BUDGET"
  echo "ctx=$CTX max_tokens=$MAXTOK n_predict=$NPRED port=$PORT"
  echo "cache_experts=400 prefill_chunk=512"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/RUN_META.txt"

# ---- launch server ----
env -i "${ENVSET[@]}" "$BIN" -m "$MODEL" --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 400 --prefill-chunk 512 \
  -c "$CTX" -n "$NPRED" --host 127.0.0.1 --port "$PORT" --cors \
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

# ---- GPU memory sampler (own file) ----
( while kill -0 "$srv_pid" 2>/dev/null; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"
    sleep 3
  done ) > "$OUT/gpu_mem.log" 2>/dev/null &
sampler_pid=$!

# ---- wait for ready ----
ready=0
for _ in $(seq 1 200); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$srv_pid" 2>/dev/null; then echo "server died before ready" >&2; break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "server_not_ready" > "$OUT/STOP_REASON.txt"
  kill "$sampler_pid" 2>/dev/null || true
  exit 1
fi
echo "ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/RUN_META.txt"

# ---- warmup (non-stream, same prompt) ----
w0=$(date +%s.%N)
curl -fsS -m 3000 -H "Content-Type: application/json" \
  -d @"$OUT/warmup_request.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  > "$OUT/warmup_response.json" 2> "$OUT/warmup_curl.err"
wrc=$?
w1=$(date +%s.%N)
echo "warmup_rc=$wrc warmup_s=$(python3 -c "print(round($w1-$w0,1))")" >> "$OUT/RUN_META.txt"

# ---- measured stream (guard stops on </html> or repeat degeneration) ----
m0=$(date +%s.%N)
python3 "$GUARD" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --request "$OUT/request.json" \
  --response "$OUT/response.json" \
  --events "$OUT/stream_events.jsonl" \
  --live-text "$OUT/stream_live.txt" \
  --timeout 2400 \
  --stop-html-close --stop-repeat --repeat-ngram 3 --repeat-window 120 --repeat-count 3 \
  > "$OUT/guard_out.json" 2> "$OUT/guard_err.txt"
m1=$(date +%s.%N)
echo "measured_wall_s=$(python3 -c "print(round($m1-$m0,1))")" >> "$OUT/RUN_META.txt"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/RUN_META.txt"

kill "$sampler_pid" 2>/dev/null || true
cat "$OUT/guard_out.json"
echo "OUT=$OUT"
