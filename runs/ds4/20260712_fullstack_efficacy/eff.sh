#!/bin/bash
# Efficacia full-stack (0034-0043) K8-per-massa + pin + fattorino su prompt esigente.
set -u
RD=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_fullstack_efficacy
PORT=8031; BIN=/root/ds4-fullstack/ds4-server; MODEL=/root/models/ds4-2bit.gguf
PROG=$RD/efficacy_progress.log; : > "$PROG"
echo "[eff] START $(date -Is) bin=$(md5sum $BIN|cut -c1-12)" >> "$PROG"
python3 - "$RD" <<'PY'
import json,sys
rd=sys.argv[1]
p=("Genera una landing page HTML5 COMPLETA e valida per un'app di food delivery chiamata RapidoEat: "
   "header con logo e nav (Home, Ristoranti, Come funziona, Accedi); una hero con titolo, sottotitolo "
   "e bottone 'Ordina ora'; una sezione con 3 feature (consegna rapida, pagamento sicuro, tracking live), "
   "ciascuna con titolo e descrizione; un form newsletter con campo email e validazione JavaScript che "
   "mostra un popup di conferma; un footer. Includi CSS interno (flexbox, palette coerente) e uno script "
   "con la validazione. Chiudi correttamente TUTTI i tag fino a </html>.")
for name,n in [("warmup",40),("measured",650)]:
    json.dump({"model":"deepseek-v4-flash","messages":[{"role":"user","content":p}],
               "max_tokens":n,"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}},
              open(rd+"/req_"+name+".json","w"))
PY
pkill -x ds4-server 2>/dev/null; sleep 3
DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_PACE=0 \
DS4_PACE_LIVEMASK=1 DS4_PACE_LIVEMASK_BOOTSTRAP=10 DS4_PACE_LIVEMASK_K=8 \
DS4_REAP_PIN_BY_MASS=1 DS4_REAP_PREFETCH_DELTA=1 DS4_SPEX_STATS=1 \
DS4_PACE_LIVEMASK_LOG=$RD/livemask.jsonl \
  $BIN -m $MODEL --cuda --ssd-streaming --ssd-streaming-cache-experts 400 --prefill-chunk 512 -c 4096 -n 2048 --host 127.0.0.1 --port $PORT --cors > $RD/server.stderr.log 2>&1 &
SRV=$!
UP=0; for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { UP=1; break; }; sleep 2; done
echo "[eff] up=$UP $(date -Is)" >> "$PROG"
if [ "$UP" = "1" ]; then
  curl -s -m 600 -H "Content-Type: application/json" -d @$RD/req_warmup.json http://127.0.0.1:$PORT/v1/chat/completions > $RD/warmup_resp.json 2>>"$PROG"
  echo "[eff] warmup done $(date -Is)" >> "$PROG"
  curl -s -m 600 -H "Content-Type: application/json" -d @$RD/req_measured.json http://127.0.0.1:$PORT/v1/chat/completions > $RD/measured_resp.json 2>>"$PROG"
  echo "[eff] measured done $(date -Is)" >> "$PROG"
fi
kill $SRV 2>/dev/null; pkill -x ds4-server 2>/dev/null; sleep 3
python3 - "$RD" <<'PY'
import json,sys
rd=sys.argv[1]
try:
    r=json.load(open(rd+"/measured_resp.json"))
    open(rd+"/content.txt","w").write(r.get("choices",[{}])[0].get("message",{}).get("content","") or "")
except Exception:
    open(rd+"/content.txt","w").write("")
PY
CLOSE=$(grep -aic "</html>" $RD/content.txt); CHARS=$(wc -c < $RD/content.txt 2>/dev/null)
PD=$(grep -aoE "prompt done [0-9.]+s" $RD/server.stderr.log | tail -1)
HIT=$(grep -aoE "hit_rate=[0-9.]+" $RD/server.stderr.log | tail -1)
CAP=$(grep -aoE "final_cap=[0-9]+|capacity=[0-9]+" $RD/server.stderr.log | tail -2 | tr "\n" " ")
CH=$(grep -aoE "avg=[0-9.]+ t/s" $RD/server.stderr.log | tail -1)
SWAPS=$(grep -aic swap $RD/livemask.jsonl 2>/dev/null)
PINADM=$(grep -aoE "pin_admits=[0-9]+" $RD/server.stderr.log | tail -1)
echo "[RESULT-EFFICACY] $PD $HIT cap=$CAP | RENDE close=$CLOSE chars=$CHARS | swaps=$SWAPS $PINADM | $CH" >> "$PROG"
echo "--- content primi 300 char ---" >> "$PROG"; head -c 300 $RD/content.txt >> "$PROG"; echo "" >> "$PROG"
echo "[eff] COMPLETE $(date -Is)" >> "$PROG"
