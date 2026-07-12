#!/bin/bash
# prefill-overlap A/B test harness. Usage: run_test.sh <label> <extra_env_kv...>
# extra_env_kv are NAME=VALUE pairs appended to the baseline env before
# launching ds4-server. Writes: <OUTDIR>/<label>.server.log, <label>.response.json
# COLD=1 in the caller env: sync + drop_caches before launching (cold page cache).
# MAXTOK=N in the caller env: override max_tokens (default 60).
set -u
BIN=/root/ds4-prefill-work
OUTDIR=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_prefill_overlap
LABEL="$1"; shift
PORT=8071
MODEL=/root/models/ds4-2bit.gguf
MAXTOK="${MAXTOK:-60}"

# make sure nothing else is holding the port / process name
pkill -x ds4-server 2>/dev/null
for i in $(seq 1 30); do
  ss -tln 2>/dev/null | grep -q ":$PORT " || break
  sleep 0.5
done

if [ "${COLD:-0}" = "1" ]; then
  sync
  echo 3 > /proc/sys/vm/drop_caches
  echo "COLD: page cache dropped" > "$OUTDIR/${LABEL}.server.log.pre"
fi

cd "$BIN" || exit 1

env \
  DS4_CUDA_NO_DIRECT_IO=1 \
  DS4_CUDA_KEEP_MODEL_PAGES=1 \
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 \
  DS4_CUDA_NO_Q8_F16_CACHE=1 \
  DS4_PACE=0 \
  "$@" \
  ./ds4-server --cuda --ssd-streaming --ssd-streaming-cache-experts 400 \
    --prefill-chunk 512 -c 4096 \
    -m "$MODEL" --port $PORT --host 127.0.0.1 \
    > "$OUTDIR/${LABEL}.server.log" 2>&1 &
SERVER_PID=$!

# wait for port to come up
READY=0
for i in $(seq 1 240); do
  if curl -s -o /dev/null -m 1 "http://127.0.0.1:$PORT/v1/models"; then
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo "SERVER_NOT_READY" > "$OUTDIR/${LABEL}.FAILED"
  kill "$SERVER_PID" 2>/dev/null
  exit 1
fi

PROMPT='Genera una pagina HTML5 minima e VALIDA per una caffetteria Bean & Brew: doctype, head con title, body con nav (Home, Menu, Contatti), un h1 e un bottone Ordina. Chiudi TUTTI i tag fino a </html>.'

T0=$(date +%s.%N)
curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"deepseek-v4-flash\",\"temperature\":0,\"max_tokens\":$MAXTOK,\"think\":false,\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT")}]}" \
  > "$OUTDIR/${LABEL}.response.json"
T1=$(date +%s.%N)
echo "wall_clock_request_seconds $(echo "$T1 - $T0" | bc)" >> "$OUTDIR/${LABEL}.server.log"

sleep 1
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null

grep "prompt done" "$OUTDIR/${LABEL}.server.log" | tail -5
echo "---content---"
python3 -c "import json; d=json.load(open('$OUTDIR/${LABEL}.response.json')); print(d['choices'][0]['message']['content'])" 2>&1
