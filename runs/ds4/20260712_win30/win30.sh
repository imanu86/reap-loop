#!/bin/bash
# Test finestra=30 a K23 sul cyberpunk. Rilassa il clamp SUPERIORE (20->32=WIN_MAX); l'inferiore era gia' 3.
set -u
RD=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_win30
mkdir -p "$RD"; SRC=/root/ds4-fullstack; MODEL=/root/models/ds4-2bit.gguf
PROG=$RD/win30_progress.log; : > "$PROG"
echo "[w30] START $(date -Is)" >> "$PROG"
cd "$SRC" || exit 1
sed -i 's/lm_window > 20) g_pace.lm_window = 20;/lm_window > 32) g_pace.lm_window = 32;/' ds4.c
echo "[w30] upper-clamp rilassato: hit=$(grep -c 'lm_window > 32) g_pace.lm_window = 32;' ds4.c)" >> "$PROG"
make cuda CUDA_ARCH=sm_86 > $RD/build.log 2>&1
echo "[w30] build md5=$(md5sum ds4-server|cut -c1-12) err=$(grep -ci 'error:' $RD/build.log)" >> "$PROG"
[ -f ds4-server ] || { echo "[w30] BUILD FALLITA" >> "$PROG"; exit 1; }
for i in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 5; done
CYBER="Crea una landing page HTML/CSS/JS single-file per un negozio di programmazione AI in stile cyberpunk. Deve avere un modulo contatti e un popup JS che dice richiesta inviata. Codice valido e compatto."
python3 - "$RD" "$CYBER" <<'PY'
import json,sys
rd=sys.argv[1]; p=sys.argv[2]
for k,n in [("warmup",40),("measured",900)]:
    json.dump({"model":"deepseek-v4-flash","messages":[{"role":"user","content":p}],"max_tokens":n,"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}}, open(rd+"/req_"+k+".json","w"))
PY
pkill -x ds4-server 2>/dev/null; sleep 3
PORT=8061
env DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_PACE=0 \
  DS4_PACE_LIVEMASK=1 DS4_PACE_LIVEMASK_BOOTSTRAP=10 DS4_PACE_LIVEMASK_K=23 DS4_PACE_LIVEMASK_WINDOW=30 \
  DS4_REAP_PIN_BY_MASS=1 DS4_SPEX_STATS=1 \
  $SRC/ds4-server -m $MODEL --cuda --ssd-streaming --ssd-streaming-cache-experts 400 --prefill-chunk 512 -c 4096 -n 2048 --host 127.0.0.1 --port $PORT --cors > $RD/server.log 2>&1 &
SRV=$!
UP=0; for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { UP=1; break; }; sleep 2; done
echo "[w30] up=$UP $(date -Is)" >> "$PROG"
echo "[w30] finestra effettiva: $(grep -aoE 'window=[0-9]+' $RD/server.log | head -1)" >> "$PROG"
if [ "$UP" = "1" ]; then
  curl -s -m 600 -H "Content-Type: application/json" -d @$RD/req_warmup.json http://127.0.0.1:$PORT/v1/chat/completions > $RD/warmup_resp.json 2>>"$PROG"
  curl -s -m 600 -H "Content-Type: application/json" -d @$RD/req_measured.json http://127.0.0.1:$PORT/v1/chat/completions > $RD/measured_resp.json 2>>"$PROG"
fi
kill $SRV 2>/dev/null; pkill -x ds4-server 2>/dev/null; sleep 3
python3 - "$RD" <<'PY'
import json,sys
rd=sys.argv[1]
try:
    r=json.load(open(rd+"/measured_resp.json")); open(rd+"/content.txt","w").write(r.get("choices",[{}])[0].get("message",{}).get("content","") or "")
except Exception: open(rd+"/content.txt","w").write("")
PY
CLOSE=$(grep -aic "</html>" $RD/content.txt); CHARS=$(wc -c < $RD/content.txt)
TS=$(grep -aoE "avg=[0-9.]+ t/s" $RD/server.log | tail -1)
echo "[RESULT-W30] close=$CLOSE chars=$CHARS | $TS" >> "$PROG"
echo "  head: $(head -c 200 $RD/content.txt | tr '\n' ' ')" >> "$PROG"
echo "  tail: ...$(tail -c 220 $RD/content.txt | tr '\n' ' ')" >> "$PROG"
echo "[w30] COMPLETE $(date -Is)" >> "$PROG"
