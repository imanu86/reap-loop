#!/bin/bash
# Isolamento regressione full-stack: produttore-only vs full-stack, fattorino on/off. Serial su 1 GPU.
set -u
RD=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_isolation
mkdir -p "$RD"
MODEL=/root/models/ds4-2bit.gguf
BIN_PROD=/root/ds4-0039-work/ds4-server
BIN_FULL=/root/ds4-fullstack/ds4-server
PROG=$RD/iso_progress.log; : > "$PROG"
echo "[iso] START $(date -Is) prod=$(md5sum $BIN_PROD|cut -c1-12) full=$(md5sum $BIN_FULL|cut -c1-12)" >> "$PROG"
echo "[iso] attendo GPU libera..." >> "$PROG"
for i in $(seq 1 300); do pgrep -x ds4-server >/dev/null || break; sleep 10; done
echo "[iso] GPU libera $(date -Is)" >> "$PROG"

COFFEE="Genera una pagina HTML5 minima e VALIDA per una caffetteria Bean & Brew: doctype, head con title, body con nav (Home, Menu, Contatti), un h1 e un bottone Ordina. Chiudi TUTTI i tag fino a </html>."
CYBER="Crea una landing page HTML/CSS/JS single-file per un negozio di programmazione AI in stile cyberpunk. Deve avere un modulo contatti e un popup JS che dice richiesta inviata. Codice valido e compatto."

run_case () {
  NAME=$1; BIN=$2; K=$3; MAXTOK=$4; DELTA=$5; PROMPT=$6; PORT=$7
  python3 - "$RD" "$NAME" "$MAXTOK" "$PROMPT" <<'PY'
import json,sys
rd=sys.argv[1]; name=sys.argv[2]; mt=int(sys.argv[3]); prompt=sys.argv[4]
for kind,n in [("warmup",40),("measured",mt)]:
    json.dump({"model":"deepseek-v4-flash","messages":[{"role":"user","content":prompt}],
               "max_tokens":n,"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}},
              open(rd+"/req_"+name+"_"+kind+".json","w"))
PY
  pkill -x ds4-server 2>/dev/null; sleep 3
  DELTAENV=""; [ "$DELTA" = "1" ] && DELTAENV="DS4_REAP_PREFETCH_DELTA=1"
  env DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_PACE=0 \
    DS4_PACE_LIVEMASK=1 DS4_PACE_LIVEMASK_BOOTSTRAP=10 DS4_PACE_LIVEMASK_K=$K DS4_REAP_PIN_BY_MASS=1 $DELTAENV DS4_SPEX_STATS=1 \
    $BIN -m $MODEL --cuda --ssd-streaming --ssd-streaming-cache-experts 400 --prefill-chunk 512 -c 4096 -n 2048 --host 127.0.0.1 --port $PORT --cors > $RD/${NAME}_server.log 2>&1 &
  SRV=$!
  UP=0; for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { UP=1; break; }; sleep 2; done
  echo "[$NAME] up=$UP bin=$(basename $(dirname $BIN)) K=$K delta=$DELTA $(date -Is)" >> "$PROG"
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
  TS=$(grep -aoE "avg=[0-9.]+ t/s" $RD/${NAME}_server.log | tail -1); PD=$(grep -aoE "prompt done [0-9.]+s" $RD/${NAME}_server.log | tail -1)
  echo "[RESULT-$NAME] $PD | close=$CLOSE chars=$CHARS | $TS" >> "$PROG"
  echo "  head: $(head -c 180 $RD/${NAME}_content.txt | tr '\n' ' ')" >> "$PROG"
}

run_case T1_prod_coffee_k8       $BIN_PROD 8  400 0 "$COFFEE" 8041
run_case T2_full_coffee_k8_nodel $BIN_FULL 8  400 0 "$COFFEE" 8042
run_case T3_full_coffee_k8_full  $BIN_FULL 8  400 1 "$COFFEE" 8043
run_case T4_prod_cyber_k23       $BIN_PROD 23 800 0 "$CYBER"  8044
echo "[iso] COMPLETE $(date -Is)" >> "$PROG"
