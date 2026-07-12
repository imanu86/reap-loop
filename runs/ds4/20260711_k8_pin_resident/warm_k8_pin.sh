#!/bin/bash
# PHASE 3 -- K8 pin-resident warm test. Binary=/root/ds4-pin/ds4-server, port 8016.
# WARMUP (~320 tok) then MEASURED (512 tok). Server NOT killed until measured resp non-empty.
set -u
OUT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k8_pin_resident
MASK=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_keep8_fit_repro/masks/sessK8_coffee.txt
MODEL=/root/models/ds4-2bit.gguf
PROMPT_FILE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k0_fullmodel_baseline/frontpage_prompt.txt
BIN=/root/ds4-pin/ds4-server
PORT=8016
CACHE=400
CTX=4096
PROG="$OUT/warm_progress.log"
: > "$PROG"
: > "$OUT/pin_events.jsonl"

body () {
python3 - "$PROMPT_FILE" "$1" <<'PY'
import json,sys
print(json.dumps({"model":"deepseek-v4-flash","messages":[
 {"role":"system","content":"Rispondi in modo diretto, utile e senza ragionamento visibile."},
 {"role":"user","content":open(sys.argv[1]).read()}],
 "max_tokens":int(sys.argv[2]),"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}}))
PY
}

pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 3
echo "[pin] server-start $(date -Is)" >> "$PROG"
( cd /root/ds4-pin && DS4_LOCK_FILE=/tmp/ds4_k8pin.lock DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 \
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_PACE=0 \
  DS4_PACE_PIN=1 DS4_PACE_PIN_BUDGET_MB=3500 DS4_PACE_PIN_WARMUP=512 \
  DS4_PACE_PIN_LOG="$OUT/pin_events.jsonl" \
  DS4_SPEX_STATS=1 DS4_REAP_MASK_FILE="$MASK" \
  "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts "$CACHE" \
    --prefill-chunk 512 -c "$CTX" -n 2048 --host 127.0.0.1 --port "$PORT" --cors \
  ) > "$OUT/server.stdout.log" 2> "$OUT/server.stderr.log" &
SRV=$!
echo "[pin] server pid $SRV" >> "$PROG"

up=0
for i in $(seq 1 300); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[pin] SERVER DIED during load $(date -Is)" >> "$PROG"; break; }
  sleep 2
done
echo "[pin] models up=$up $(date -Is)" >> "$PROG"

body 320 > "$OUT/warmup_req.json"
echo "[pin] WARMUP fired $(date -Is)" >> "$PROG"
curl -s -m 3000 -H "Content-Type: application/json" -d @"$OUT/warmup_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/warmup_resp.json" 2>>"$PROG"
echo "[pin] WARMUP done bytes=$(wc -c < "$OUT/warmup_resp.json") $(date -Is)" >> "$PROG"
echo "===MEASURED_BOUNDARY===" >> "$OUT/server.stderr.log"

body 512 > "$OUT/measured_req.json"
echo "[pin] MEASURED fired $(date -Is)" >> "$PROG"
T0=$(date +%s)
curl -s -m 3000 -H "Content-Type: application/json" -d @"$OUT/measured_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/measured_resp.json" 2>>"$PROG"
T1=$(date +%s)
MB=$(wc -c < "$OUT/measured_resp.json")
echo "[pin] MEASURED done wall=$((T1-T0))s bytes=$MB $(date -Is)" >> "$PROG"

if [ "$MB" -lt 5 ]; then
  echo "[pin] WARNING measured empty -- NOT killing, leaving server up" >> "$PROG"
else
  python3 - "$OUT/measured_resp.json" "$OUT/measured_content.txt" <<'PY'
import json,sys
try:
    r=json.load(open(sys.argv[1])); c=r["choices"][0]["message"]["content"]
    open(sys.argv[2],"w").write(c)
    print("finish",r["choices"][0].get("finish_reason"),"usage",json.dumps(r.get("usage",{})))
except Exception as e: print("ERR",e)
PY
fi

# bit-exact source: dedicated greedy 64-tok request under PIN (identical body to Phase 4)
if [ "$MB" -ge 5 ]; then
  body 64 > "$OUT/greedy64_req.json"
  echo "[pin] GREEDY64 fired $(date -Is)" >> "$PROG"
  curl -s -m 600 -H "Content-Type: application/json" -d @"$OUT/greedy64_req.json" \
       "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/pin_on_64_resp.json" 2>>"$PROG"
  python3 - "$OUT/pin_on_64_resp.json" "$OUT/pin_on_64.txt" <<'PY'
import json,sys
try:
    r=json.load(open(sys.argv[1])); open(sys.argv[2],"w").write(r["choices"][0]["message"]["content"])
    print("pin64 finish",r["choices"][0].get("finish_reason"),"usage",json.dumps(r.get("usage",{})))
except Exception as e: print("ERR",e)
PY
  echo "[pin] GREEDY64 done bytes=$(wc -c < "$OUT/pin_on_64.txt" 2>/dev/null) $(date -Is)" >> "$PROG"
fi

MEAS_CHUNKS=$(awk '/===MEASURED_BOUNDARY===/{f=1} f' "$OUT/server.stderr.log" | grep -aoE 'gen=[0-9]+ decoding chunk=[0-9.]+ t/s avg=[0-9.]+ t/s' | tr '\n' '|')
SPEX_FINAL=$(grep -aoE 'SPEX stats:.*' "$OUT/server.stderr.log" | tail -1)
{
  echo "[RESULT-PIN] K=8 cache=$CACHE PIN=1 budget=3500MB"
  echo "  measured_wall=$((T1-T0))s bytes=$MB"
  echo "  MEASURED chunks: $MEAS_CHUNKS"
  echo "  SPEX_FINAL: $SPEX_FINAL"
  echo "  pin_freeze=$(grep -c pin_freeze "$OUT/pin_events.jsonl" 2>/dev/null)"
  echo "  pin_rotate=$(grep -c pin_rotate "$OUT/pin_events.jsonl" 2>/dev/null)"
} >> "$PROG"

if [ "$MB" -ge 5 ]; then pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 2; fi
echo "[pin] COMPLETE $(date -Is)" >> "$PROG"
