#!/bin/bash
# PHASE 4 -- bit-exact gate. SAME binary, PIN OFF (DS4_PACE_PIN unset). Cold OK.
# Greedy 64-tok request, identical body to Phase 3 greedy64, compare byte-for-byte.
set -u
OUT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k8_pin_resident
MASK=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_keep8_fit_repro/masks/sessK8_coffee.txt
MODEL=/root/models/ds4-2bit.gguf
PROMPT_FILE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k0_fullmodel_baseline/frontpage_prompt.txt
BIN=/root/ds4-pin/ds4-server
PORT=8016
CACHE=400
CTX=4096
PROG="$OUT/bitexact_progress.log"
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

pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 3
echo "[nopin] server-start (PIN OFF) $(date -Is)" >> "$PROG"
( cd /root/ds4-pin && DS4_LOCK_FILE=/tmp/ds4_k8pin.lock DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 \
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_PACE=0 \
  DS4_SPEX_STATS=1 DS4_REAP_MASK_FILE="$MASK" \
  "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts "$CACHE" \
    --prefill-chunk 512 -c "$CTX" -n 2048 --host 127.0.0.1 --port "$PORT" --cors \
  ) > "$OUT/nopin_server.stdout.log" 2> "$OUT/nopin_server.stderr.log" &
SRV=$!
echo "[nopin] server pid $SRV" >> "$PROG"

up=0
for i in $(seq 1 300); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "[nopin] SERVER DIED during load $(date -Is)" >> "$PROG"; break; }
  sleep 2
done
echo "[nopin] models up=$up $(date -Is)" >> "$PROG"

body 64 > "$OUT/greedy64_req.json"
echo "[nopin] GREEDY64 fired $(date -Is)" >> "$PROG"
curl -s -m 900 -H "Content-Type: application/json" -d @"$OUT/greedy64_req.json" \
     "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/nopin_64_resp.json" 2>>"$PROG"
NB=$(wc -c < "$OUT/nopin_64_resp.json")
echo "[nopin] GREEDY64 done bytes=$NB $(date -Is)" >> "$PROG"

if [ "$NB" -ge 5 ]; then
  python3 - "$OUT/nopin_64_resp.json" "$OUT/nopin_64.txt" <<'PY'
import json,sys
try:
    r=json.load(open(sys.argv[1])); open(sys.argv[2],"w").write(r["choices"][0]["message"]["content"])
    print("nopin64 finish",r["choices"][0].get("finish_reason"),"usage",json.dumps(r.get("usage",{})))
except Exception as e: print("ERR",e)
PY
fi

echo "=== BIT-EXACT COMPARE (nopin_64 as prefix of pin-ON 512-tok outputs) ===" >> "$PROG"
python3 - >> "$PROG" <<'PY'
import os
OUT="/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k8_pin_resident"
def rd(p):
    try: return open(os.path.join(OUT,p),encoding="utf-8",errors="replace").read()
    except Exception as e: return None
nop=rd("nopin_64.txt")
if not nop:
    print("  nopin_64.txt missing/empty -> INCONCLUSIVE"); raise SystemExit
print(f"  nopin_64 len={len(nop)}")
for ref in ("measuredM_content.txt","measured_content.txt"):
    r=rd(ref)
    if not r: print(f"  {ref}: missing"); continue
    # longest common prefix
    n=min(len(nop),len(r)); i=0
    while i<n and nop[i]==r[i]: i+=1
    verdict="PASS (nopin_64 is a prefix of pin-ON)" if r.startswith(nop) else f"DIFFER at char {i}"
    print(f"  vs {ref}: common_prefix_chars={i} nopin_len={len(nop)} -> {verdict}")
    print(f"    nopin head: {nop[:80]!r}")
    print(f"    pinON head: {r[:80]!r}")
PY

if [ "$NB" -ge 5 ]; then pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 2; fi
echo "[nopin] COMPLETE $(date -Is)" >> "$PROG"
