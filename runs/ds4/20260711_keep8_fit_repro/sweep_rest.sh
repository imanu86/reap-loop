#!/bin/bash
# Coffee static-mask fit sweep on the local 3060. One ds4-server per K, one
# non-streaming -n 512 request; the server logs per-50-token chunk t/s natively,
# giving the within-request page-in -> steady curve (the CLAIM-008 segmented metric).
# Coexists with the UI: distinct DS4_LOCK_FILE + port 8014; only ever pkills a
# ds4-server bound to this exact port.
set -u
RD=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_keep8_fit_repro
MASKS=$RD/masks
MODEL=/root/models/ds4-2bit.gguf
PROMPT_FILE=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_k0_fullmodel_baseline/frontpage_prompt.txt
PORT=8014
CACHE=${CACHE:-400}
NTOK=${NTOK:-512}
CTX=4096
PROG=$RD/sweep_progress.log
PROMPT=$(cat "$PROMPT_FILE")

echo "[sweep] START cache=$CACHE ntok=$NTOK $(date -Is)" >> "$PROG"

req_body () {
  python3 - "$1" "$2" <<'PY'
import json,sys
prompt=open(sys.argv[1]).read()
n=int(sys.argv[2])
print(json.dumps({
  "model":"deepseek-v4-flash",
  "messages":[
    {"role":"system","content":"Rispondi in modo diretto, utile e senza ragionamento visibile."},
    {"role":"user","content":prompt}],
  "max_tokens":n,"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}}))
PY
}

run_k () {
  K=$1; TRACE=$2
  D=$RD/K${K}; mkdir -p "$D"
  MASK=$MASKS/sessK${K}_coffee.txt
  TRACEFILE=/dev/shm/route_k${K}_coffee.csv
  rm -f "$TRACEFILE"
  pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 3
  # 86GB model > 60GB RAM: KEEP_MODEL_PAGES=1 thrashes on repeated launches and
  # gives no benefit (model can't be held resident). Stream experts on demand;
  # steady-state decode is VRAM-cache bound and unaffected. Drop page cache so
  # every K starts from a comparable cold page-in.

  ENV="DS4_LOCK_FILE=/tmp/ds4_keep8repro.lock \
DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=0 \
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 \
DS4_PACE=0 DS4_REAP_MASK_FILE=$MASK DS4_SPEX_STATS=1"
  if [ "$TRACE" = "1" ]; then
    ENV="$ENV DS4_SPEX_TRACE_ROUTING=$TRACEFILE DS4_SPEX_TRACE_ROUTING_WEIGHTS=0"
  fi

  echo "[sweep] K=$K cache=$CACHE trace=$TRACE server-start $(date -Is)" >> "$PROG"
  ( cd /root/ds4 && eval "$ENV" /root/ds4/ds4-server -m "$MODEL" --cuda --ssd-streaming \
      --ssd-streaming-cache-experts "$CACHE" --prefill-chunk 512 -c "$CTX" -n 2048 \
      --host 127.0.0.1 --port "$PORT" --cors ) > "$D/server.stdout.log" 2> "$D/server.stderr.log" &
  SRV=$!

  # wait for models
  UP=0
  for i in $(seq 1 300); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then UP=1; break; fi
    sleep 2
  done
  if [ "$UP" != "1" ]; then
    echo "[sweep] K=$K SERVER_NO_MODELS $(date -Is)" >> "$PROG"
    kill $SRV 2>/dev/null; pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 2; return
  fi

  req_body "$PROMPT_FILE" "$NTOK" > "$D/request.json"
  echo "[sweep] K=$K request-fired $(date -Is)" >> "$PROG"
  T0=$(date +%s)
  curl -s -m 2400 -H "Content-Type: application/json" -d @"$D/request.json" \
       "http://127.0.0.1:$PORT/v1/chat/completions" > "$D/response.json" 2>"$D/curl.err"
  T1=$(date +%s)

  python3 - "$D/response.json" "$D/content.txt" <<'PY'
import json,sys
try:
    r=json.load(open(sys.argv[1]))
    c=r.get("choices",[{}])[0].get("message",{}).get("content","")
    open(sys.argv[2],"w").write(c)
    u=r.get("usage",{})
    print("USAGE",json.dumps(u))
    print("FINISH",r.get("choices",[{}])[0].get("finish_reason"))
except Exception as e:
    print("PARSE_ERR",e)
PY

  PF=$(grep -aoE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/server.stderr.log" | tail -1)
  PROMPTDONE=$(grep -aoE "prompt done [0-9.]+s" "$D/server.stderr.log" | tail -1)
  # per-chunk decode t/s (page-in -> steady curve)
  CHUNKS=$(grep -aoE "gen=[0-9]+ decoding chunk=[0-9.]+ t/s avg=[0-9.]+ t/s" "$D/server.stderr.log" | tr '\n' '|')
  HIT=$(grep -aoE "hit_rate=[0-9.]+" "$D/server.stderr.log" | tail -1)
  MASKLINE=$(grep -aoE "REAP mask applied: [0-9]+ layer" "$D/server.stderr.log" | tail -1)
  CHARS=$(wc -c < "$D/content.txt" 2>/dev/null)
  CLOSE=$(grep -aoic "</html>" "$D/content.txt" 2>/dev/null)

  ENF="trace-off"
  if [ "$TRACE" = "1" ] && [ -f "$TRACEFILE" ]; then
    ENF=$(python3 - "$TRACEFILE" "$MASKS/sessK${K}_coffee.json" <<'PY'
import csv,collections,json,sys
used=collections.defaultdict(set)
with open(sys.argv[1]) as f:
    r=csv.reader(f); next(r,None)
    for row in r:
        if len(row)<4: continue
        try:
            L=int(row[1]); n=int(row[2]); es=[int(row[3+i]) for i in range(n)]
        except: continue
        used[L].update(es)
pl={L:len(s) for L,s in used.items()}
keep=json.load(open(sys.argv[2]))["keep"]; keep={int(k):set(v) for k,v in keep.items()}
viol=sum(len(s-keep.get(L,set())) for L,s in used.items())
mx=max(pl.values()) if pl else -1
print(f"maxdistinct/layer={mx} violations_outside_keep={viol}")
PY
)
    gzip -c "$TRACEFILE" > "$D/route_measured.csv.gz" 2>/dev/null; rm -f "$TRACEFILE"
  fi

  WS=$(( K * 40 ))
  FIT=$( [ $WS -le $CACHE ] && echo FITS || echo OVER )
  echo "[RESULT] K=$K ws=$WS cache=$CACHE fit=$FIT wall=$((T1-T0))s $PF | promptdone=$PROMPTDONE | $HIT | mask='$MASKLINE' | enf=[$ENF] | chars=$CHARS close_html=$CLOSE | chunks=$CHUNKS" >> "$PROG"

  kill $SRV 2>/dev/null; pkill -f "ds4-server.*--port $PORT" 2>/dev/null; sleep 2
}

# K8 first with trace (enforcement + priority). Then the rest clean.
# K8 covered by K8_warm and prior runs
run_k 9 1
run_k 12 0
run_k 16 0
run_k 23 0
echo "[sweep] COMPLETE $(date -Is)" >> "$PROG"
