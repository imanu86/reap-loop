#!/bin/bash
# Window sweep K23-cyber: la finestra e' la leva di fase-adattivita'. Clamp rilassato 10->3 per testare 5.
set -u
RD=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_winsweep
mkdir -p "$RD"
SRC=/root/ds4-fullstack; MODEL=/root/models/ds4-2bit.gguf
PROG=$RD/winsweep_progress.log; : > "$PROG"
echo "[ws] START $(date -Is)" >> "$PROG"
cd "$SRC" || exit 1
# 1. rilassa il clamp inferiore della finestra livemask (10 -> 3) per poter testare W=5
sed -i 's/lm_window < 10) g_pace.lm_window = 10;/lm_window < 3) g_pace.lm_window = 3;/' ds4.c
echo "[ws] clamp rilassato: hit=$(grep -c 'lm_window < 3) g_pace.lm_window = 3;' ds4.c)" >> "$PROG"
# 2. rebuild
make cuda CUDA_ARCH=sm_86 > $RD/build.log 2>&1
echo "[ws] build md5=$(md5sum ds4-server|cut -c1-12) err=$(grep -ci 'error:' $RD/build.log)" >> "$PROG"
[ -f ds4-server ] || { echo "[ws] BUILD FALLITA - stop" >> "$PROG"; exit 1; }
# 3. attendi GPU
for i in $(seq 1 300); do pgrep -x ds4-server >/dev/null || break; sleep 10; done
CYBER="Crea una landing page HTML/CSS/JS single-file per un negozio di programmazione AI in stile cyberpunk. Deve avere un modulo contatti e un popup JS che dice richiesta inviata. Codice valido e compatto."

run_win () {
  W=$1; PORT=$2; NAME=win${W}
  python3 - "$RD" "$NAME" "$CYBER" <<'PY'
import json,sys
rd=sys.argv[1]; name=sys.argv[2]; prompt=sys.argv[3]
for kind,n in [("warmup",40),("measured",900)]:
    json.dump({"model":"deepseek-v4-flash","messages":[{"role":"user","content":prompt}],
               "max_tokens":n,"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}},
              open(rd+"/req_"+name+"_"+kind+".json","w"))
PY
  pkill -x ds4-server 2>/dev/null; sleep 3
  env DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_PACE=0 \
    DS4_PACE_LIVEMASK=1 DS4_PACE_LIVEMASK_BOOTSTRAP=10 DS4_PACE_LIVEMASK_K=23 DS4_PACE_LIVEMASK_WINDOW=$W \
    DS4_REAP_PIN_BY_MASS=1 DS4_SPEX_STATS=1 \
    $SRC/ds4-server -m $MODEL --cuda --ssd-streaming --ssd-streaming-cache-experts 400 --prefill-chunk 512 -c 4096 -n 2048 --host 127.0.0.1 --port $PORT --cors > $RD/${NAME}_server.log 2>&1 &
  SRV=$!
  UP=0; for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { UP=1; break; }; sleep 2; done
  echo "[$NAME] up=$UP W=$W $(date -Is)" >> "$PROG"
  if [ "$UP" = "1" ]; then
    curl -s -m 600 -H "Content-Type: application/json" -d @$RD/req_${NAME}_warmup.json http://127.0.0.1:$PORT/v1/chat/completions > $RD/${NAME}_warmup_resp.json 2>>"$PROG"
    curl -s -m 600 -H "Content-Type: application/json" -d @$RD/req_${NAME}_measured.json http://127.0.0.1:$PORT/v1/chat/completions > $RD/${NAME}_resp.json 2>>"$PROG"
  fi
  kill $SRV 2>/dev/null; pkill -x ds4-server 2>/dev/null; sleep 3
  python3 - "$RD" "$NAME" <<'PY'
import json,sys
rd=sys.argv[1]; name=sys.argv[2]
try:
    r=json.load(open(rd+"/"+name+"_resp.json"))
    open(rd+"/"+name+"_content.txt","w").write(r.get("choices",[{}])[0].get("message",{}).get("content","") or "")
except Exception:
    open(rd+"/"+name+"_content.txt","w").write("")
PY
  CLOSE=$(grep -aic "</html>" $RD/${NAME}_content.txt); CHARS=$(wc -c < $RD/${NAME}_content.txt 2>/dev/null)
  TS=$(grep -aoE "avg=[0-9.]+ t/s" $RD/${NAME}_server.log | tail -1)
  SWAPS=$(grep -aoE "swap" $RD/${NAME}_server.log 2>/dev/null | wc -l)
  echo "[RESULT-$NAME] W=$W close=$CLOSE chars=$CHARS | $TS" >> "$PROG"
  echo "  tail: ...$(tail -c 160 $RD/${NAME}_content.txt | tr '\n' ' ')" >> "$PROG"
}

run_win 5  8051
run_win 8  8052
run_win 12 8053
run_win 16 8054
echo "[ws] COMPLETE $(date -Is)" >> "$PROG"
