#!/usr/bin/env bash
# V2 zero-copy A/B decode driver. ARM=off (pread) | on (DS4_CUDA_STREAM_FROM_RAM_MASKED).
# Same ds4-v2-work binary both arms; only the STREAM_FROM_RAM_MASKED env differs.
set -uo pipefail
ARM=${1:?arm off|on}
MAXTOK=${2:-450}
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUTROOT="$REPO/runs/ds4/20260712_v2_zerocopy"
OUT="$OUTROOT/arm_$ARM"
BIN=/root/ds4-v2-work/ds4-server
MODEL=/root/models/ds4-2bit.gguf
MASK="$REPO/runs/ds4/20260712_virtual_bake/masks/mask60_self.txt"
PORT=8097
GPU_LOCK=/tmp/ds4-gpu.lock
RAM_FLOOR_MB=7168
mkdir -p "$OUT"

if pgrep -x ds4-server >/dev/null; then echo "ds4-server already running; refuse" >&2; exit 1; fi

PROMPT='Crea una landing page HTML/CSS/JS single-file per un negozio di programmazione AI in stile cyberpunk. Deve avere un modulo contatti e un popup JS che dice richiesta inviata. Codice valido e compatto.'
python3 - "$OUT/request.json" "$MAXTOK" "$PROMPT" <<PY
import json,sys
out,mt,p=sys.argv[1],int(sys.argv[2]),sys.argv[3]
json.dump({"model":"deepseek-v4-flash","messages":[{"role":"system","content":"Rispondi in modo diretto, utile e senza ragionamento visibile."},{"role":"user","content":p}],"max_tokens":mt,"temperature":0,"stream":True,"think":False,"thinking":{"type":"disabled"},"stream_options":{"include_usage":True}},open(out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
PY
python3 - "$OUT/warmup_request.json" "$PROMPT" <<PY
import json,sys
out,p=sys.argv[1],sys.argv[2]
json.dump({"model":"deepseek-v4-flash","messages":[{"role":"system","content":"Rispondi in modo diretto, utile e senza ragionamento visibile."},{"role":"user","content":p}],"max_tokens":48,"temperature":0,"stream":False,"think":False,"thinking":{"type":"disabled"}},open(out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
PY

exec 9>"$GPU_LOCK"; flock 9
echo "[$ARM] GPU lock $(date -u +%H:%M:%S)"

cleanup(){
  if [[ -f "$OUT/server.pid" ]]; then p=$(cat "$OUT/server.pid" 2>/dev/null||true); [[ -n "${p:-}" ]] && kill "$p" 2>/dev/null||true; sleep 2; [[ -n "${p:-}" ]] && kill -9 "$p" 2>/dev/null||true; fi
  [[ -n "${rammon:-}" ]] && kill "$rammon" 2>/dev/null||true
  flock -u 9 2>/dev/null||true
}
trap cleanup EXIT

( while true; do
    mb=$(free -m|awk "/^Mem:/{print \$7}")
    echo "$(date -u +%H:%M:%S) MemAvailable_MB=$mb" >> "$OUT/ram_log.txt"
    if [[ -n "$mb" && "$mb" -lt "$RAM_FLOOR_MB" ]]; then
      echo "RAM FLOOR BREACH $mb" >> "$OUT/ram_log.txt"; echo "ram_breach=$mb">"$OUT/RAM_KILL.txt"
      [[ -f "$OUT/server.pid" ]] && kill "$(cat "$OUT/server.pid")" 2>/dev/null||true; break
    fi; sleep 10
  done ) & rammon=$!

export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_PACE=0
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
export DS4_REAP_MASK_FILE="$MASK"
export DS4_REAP_PREFETCH=1
export DS4_REAP_PREFETCH_THREADS=16
export DS4_SPEX_STATS=1
if [[ "$ARM" == "on" ]]; then
  export DS4_CUDA_STREAM_FROM_RAM_MASKED="$MASK"
  export DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1
fi
env|LC_ALL=C sort > "$OUT/server_env.txt"

"$BIN" -m "$MODEL" --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 400 --prefill-chunk 512 \
  -c 4096 -n "$((MAXTOK+256))" --host 127.0.0.1 --port "$PORT" --cors \
  >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
srv=$!; echo "$srv" > "$OUT/server.pid"

ready=0
for _ in $(seq 1 180); do
  curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$srv" 2>/dev/null || { echo "server died"; break; }
  sleep 3
done
echo "ready=$ready" | tee -a "$OUT/RUN_META.txt"
[[ "$ready" == 1 ]] || { echo "not_ready">"$OUT/STOP_REASON.txt"; exit 1; }

echo "[$ARM] warmup start $(date -u +%H:%M:%S)"
ws0=$(date +%s.%N)
curl -fsS -m 2400 -H "Content-Type: application/json" -d @"$OUT/warmup_request.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/warmup_response.json" 2>"$OUT/warmup.err"
wrc=$?
ws1=$(date +%s.%N)
echo "warmup_rc=$wrc warmup_s=$(python3 -c "print(round($ws1-$ws0,1))")" | tee -a "$OUT/RUN_META.txt"
[[ "$wrc" == 0 ]] || { echo "warmup_failed">"$OUT/STOP_REASON.txt"; exit 1; }

echo "[$ARM] measured start $(date -u +%H:%M:%S)"
python3 "$OUTROOT/scripts/measure_stream.py" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --request "$OUT/request.json" --out "$OUT/measure.json" \
  --live "$OUT/stream_live.txt" --drop 40
echo "[$ARM] measured done $(date -u +%H:%M:%S)"
echo "done" > "$OUT/STOP_REASON.txt"
