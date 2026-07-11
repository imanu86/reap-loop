#!/bin/bash
# Decisive warm keep-8 test (coordinator config): KEEP_MODEL_PAGES=1, NO drop_caches,
# a real warmup request first, then measure. If the pre-warmed measured request STARTS
# fast (>10 t/s at tok 1-50) -> residency buys speed. If it starts at the ~3.5 plateau
# -> 3.5 is the resident ceiling. Reads per-chunk t/s per request + SPEX hit_rate.
set -u
RD=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_keep8_fit_repro
MASK=$RD/masks/sessK8_coffee.txt
MODEL=/root/models/ds4-2bit.gguf
PROMPT_FILE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k0_fullmodel_baseline/frontpage_prompt.txt
PORT=8014
CACHE=${CACHE:-400}
CTX=4096
D=$RD/K8_warm; mkdir -p "$D"
PROG=$D/warm_progress.log
: > "$PROG"

body () { python3 - "$PROMPT_FILE" "$1" <<'PY'
import json,sys
print(json.dumps({"model":"deepseek-v4-flash","messages":[
 {"role":"system","content":"Rispondi in modo diretto, utile e senza ragionamento visibile."},
 {"role":"user","content":open(sys.argv[1]).read()}],
 "max_tokens":int(sys.argv[2]),"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}}))
PY
}

pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 3
echo "[warm] server-start (KEEP_MODEL_PAGES=1, no drop_caches) $(date -Is)" >> "$PROG"
( cd /root/ds4 && DS4_LOCK_FILE=/tmp/ds4_keep8repro.lock \
  DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 \
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 \
  DS4_PACE=0 DS4_REAP_MASK_FILE=$MASK DS4_SPEX_STATS=1 \
  /root/ds4/ds4-server -m "$MODEL" --cuda --ssd-streaming \
    --ssd-streaming-cache-experts "$CACHE" --prefill-chunk 512 -c "$CTX" -n 2048 \
    --host 127.0.0.1 --port "$PORT" --cors ) > "$D/server.stdout.log" 2> "$D/server.stderr.log" &
SRV=$!

for i in $(seq 1 300); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 2
done
echo "[warm] models up $(date -Is)" >> "$PROG"

# WARMUP request (discard) -- fills the 400-slot VRAM cache with the 320 working set
# and warms RAM. -n 320 so every kept expert is touched.
body 320 > "$D/warmup_req.json"
echo "[warm] WARMUP fired $(date -Is)" >> "$PROG"
curl -s -m 2400 -H "Content-Type: application/json" -d @"$D/warmup_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$D/warmup_resp.json" 2>>"$PROG"
echo "[warm] WARMUP done $(date -Is)" >> "$PROG"
# marker so we can separate measured chunks from warmup chunks
echo "===MEASURED_BOUNDARY===" >> "$D/server.stderr.log"
SPEX_AFTER_WARMUP=$(grep -aoE "hit_rate=[0-9.]+ miss_per_expert=[0-9.]+" "$D/server.stderr.log" | tail -1)

# MEASURED request -- starts fully warm (cache already holds the 320)
body 512 > "$D/measured_req.json"
echo "[warm] MEASURED fired $(date -Is)" >> "$PROG"
T0=$(date +%s)
curl -s -m 2400 -H "Content-Type: application/json" -d @"$D/measured_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$D/measured_resp.json" 2>>"$PROG"
T1=$(date +%s)
echo "[warm] MEASURED done wall=$((T1-T0))s $(date -Is)" >> "$PROG"

python3 - "$D/measured_resp.json" "$D/measured_content.txt" <<'PY'
import json,sys
try:
    r=json.load(open(sys.argv[1])); c=r["choices"][0]["message"]["content"]
    open(sys.argv[2],"w").write(c)
    print("finish",r["choices"][0].get("finish_reason"),"usage",json.dumps(r.get("usage",{})))
except Exception as e: print("ERR",e)
PY

# chunks AFTER the boundary marker = the measured request's page-in->steady curve
MEAS_CHUNKS=$(awk '/===MEASURED_BOUNDARY===/{f=1} f' "$D/server.stderr.log" | grep -aoE "gen=[0-9]+ decoding chunk=[0-9.]+ t/s avg=[0-9.]+ t/s" | tr '\n' '|')
SPEX_FINAL=$(grep -aoE "cache_hits=[0-9]+ cache_misses=[0-9]+ hit_rate=[0-9.]+ miss_per_expert=[0-9.]+" "$D/server.stderr.log" | tail -1)
{
  echo "[RESULT-WARM] K=8 cache=$CACHE KEEP_MODEL_PAGES=1 no-drop"
  echo "  spex_after_warmup: $SPEX_AFTER_WARMUP"
  echo "  spex_final(cumulative): $SPEX_FINAL"
  echo "  MEASURED chunks (warm from tok1): $MEAS_CHUNKS"
} >> "$PROG"

pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 2
echo "[warm] COMPLETE $(date -Is)" >> "$PROG"
