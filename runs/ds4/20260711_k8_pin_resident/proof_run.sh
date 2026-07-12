#!/bin/bash
# Instrumented residency proof. RUN M = warmup+measured (pin ON) with /proc/io
# snapshots isolating prefill vs decode SSD. RUN W = warmup-only (pin ON) so
# measured-delta SPEX = M - W. Server never killed before measured resp non-empty.
set -u
OUT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k8_pin_resident
MASK=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_keep8_fit_repro/masks/sessK8_coffee.txt
MODEL=/root/models/ds4-2bit.gguf
PROMPT_FILE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k0_fullmodel_baseline/frontpage_prompt.txt
BIN=/root/ds4-pin/ds4-server
PORT=8016
CACHE=400
CTX=4096
PROG="$OUT/proof_progress.log"
: > "$PROG"

body () {
python3 - "$PROMPT_FILE" "$1" <<'PY'
import json,sys
print(json.dumps({"model":"deepseek-v4-flash","messages":[
 {"role":"system","content":"Rispondi in modo diretto, utile e senza ragionamento visibile."},
 {"role":"user","content":open(sys.argv[1]).read()}],
 "max_tokens":int(sys.argv[2]),"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}}))
PY
}

start_server () {  # $1 = pin_events file (empty to disable), $2 = stderr log
  pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 3
  local PINLOG="$1"; local ERRLOG="$2"
  : > "${PINLOG:-/dev/null}" 2>/dev/null || true
  ( cd /root/ds4-pin && DS4_LOCK_FILE=/tmp/ds4_k8pin.lock DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 \
    DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_PACE=0 \
    DS4_PACE_PIN=1 DS4_PACE_PIN_BUDGET_MB=3500 DS4_PACE_PIN_WARMUP=512 \
    DS4_PACE_PIN_LOG="$PINLOG" \
    DS4_SPEX_STATS=1 DS4_REAP_MASK_FILE="$MASK" \
    "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts "$CACHE" \
      --prefill-chunk 512 -c "$CTX" -n 2048 --host 127.0.0.1 --port "$PORT" --cors \
    ) > "$OUT/proof_stdout.log" 2> "$ERRLOG" &
  for i in $(seq 1 300); do
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  pgrep -f "ds4-server.*--port $PORT" | head -1
}

io_field () { grep -m1 "^$2:" "/proc/$1/io" 2>/dev/null | awk '{print $2}'; }

########## RUN M : warmup + measured (instrumented) ##########
echo "[M] start $(date -Is)" >> "$PROG"
SRV=$(start_server "$OUT/pin_events.jsonl" "$OUT/proofM.stderr.log")
echo "[M] server pid=$SRV $(date -Is)" >> "$PROG"

body 320 > "$OUT/warmup_req.json"
echo "[M] WARMUP fired $(date -Is)" >> "$PROG"
curl -s -m 3000 -H "Content-Type: application/json" -d @"$OUT/warmup_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/warmupM_resp.json" 2>>"$PROG"
echo "[M] WARMUP done bytes=$(wc -c < "$OUT/warmupM_resp.json") $(date -Is)" >> "$PROG"

# io snapshot BEFORE measured
echo "IO_BEFORE epoch=$(date +%s) read_bytes=$(io_field $SRV read_bytes) rchar=$(io_field $SRV rchar) syscr=$(io_field $SRV syscr)" >> "$PROG"
PD0=$(grep -ac "prompt done" "$OUT/proofM.stderr.log")

body 512 > "$OUT/measured_req.json"
echo "[M] MEASURED fired $(date -Is)" >> "$PROG"
T0=$(date +%s)
curl -s -m 3000 -H "Content-Type: application/json" -d @"$OUT/measured_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/measuredM_resp.json" 2>>"$PROG" &
CURL=$!

# watch for measured prompt-done -> snapshot io at prefill/decode boundary
PDDONE=0
while kill -0 $CURL 2>/dev/null; do
  if [ "$PDDONE" = "0" ] && [ "$(grep -ac 'prompt done' "$OUT/proofM.stderr.log")" -gt "$PD0" ]; then
    echo "IO_PROMPTDONE epoch=$(date +%s) read_bytes=$(io_field $SRV read_bytes) rchar=$(io_field $SRV rchar) syscr=$(io_field $SRV syscr)" >> "$PROG"
    PDDONE=1
  fi
  sleep 2
done
wait $CURL 2>/dev/null
T1=$(date +%s)
echo "IO_AFTER epoch=$(date +%s) read_bytes=$(io_field $SRV read_bytes) rchar=$(io_field $SRV rchar) syscr=$(io_field $SRV syscr)" >> "$PROG"
MB=$(wc -c < "$OUT/measuredM_resp.json")
echo "[M] MEASURED done wall=$((T1-T0))s bytes=$MB $(date -Is)" >> "$PROG"

if [ "$MB" -ge 5 ]; then
  python3 - "$OUT/measuredM_resp.json" "$OUT/measuredM_content.txt" <<'PY'
import json,sys
try:
    r=json.load(open(sys.argv[1])); open(sys.argv[2],"w").write(r["choices"][0]["message"]["content"])
    print("finish",r["choices"][0].get("finish_reason"),"usage",json.dumps(r.get("usage",{})))
except Exception as e: print("ERR",e)
PY
  MEAS_CHUNKS=$(grep -aoE 'gen=[0-9]+ decoding chunk=[0-9.]+ t/s avg=[0-9.]+ t/s' "$OUT/proofM.stderr.log" | tr '\n' '|')
  echo "[M] MEAS_CHUNKS: $MEAS_CHUNKS" >> "$PROG"
  pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 3
else
  echo "[M] WARNING measured empty; leaving server up" >> "$PROG"
fi
grep -aoE 'SPEX stats:.*' "$OUT/proofM.stderr.log" | tail -1 | sed 's/^/[M] SPEX_M: /' >> "$PROG"
echo "[M] pin_freeze=$(grep -ac pin_freeze "$OUT/pin_events.jsonl") pin_rotate=$(grep -ac pin_rotate "$OUT/pin_events.jsonl")" >> "$PROG"
grep -a pin_freeze "$OUT/pin_events.jsonl" | tail -1 | sed 's/^/[M] pin_freeze_last: /' >> "$PROG"

########## RUN W : warmup only (baseline for SPEX subtraction) ##########
echo "[W] start $(date -Is)" >> "$PROG"
SRVW=$(start_server "" "$OUT/proofW.stderr.log")
echo "[W] server pid=$SRVW $(date -Is)" >> "$PROG"
body 320 > "$OUT/warmupW_req.json"
echo "[W] WARMUP fired $(date -Is)" >> "$PROG"
curl -s -m 3000 -H "Content-Type: application/json" -d @"$OUT/warmupW_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/warmupW_resp.json" 2>>"$PROG"
echo "[W] WARMUP done bytes=$(wc -c < "$OUT/warmupW_resp.json") $(date -Is)" >> "$PROG"
pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 3
grep -aoE 'SPEX stats:.*' "$OUT/proofW.stderr.log" | tail -1 | sed 's/^/[W] SPEX_W: /' >> "$PROG"
echo "[PROOF] COMPLETE $(date -Is)" >> "$PROG"
