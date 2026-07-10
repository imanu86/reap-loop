#!/usr/bin/env bash
set -u
BIN=/root/bin/ds4-admit
MODEL=/root/models/ds4-2bit.gguf
OUT=/root/pod3/out
PROMPT="$1"; TAG="$2"; CTX="$3"; NTOK="$4"; NRUNS="$5"
BASE="DS4_PACE=1 DS4_PACE_WARMUP=50 DS4_PACE_KEEP=23 DS4_PACE_KEEP_MIN=23 DS4_PACE_KEEP_MAX=96 DS4_PACE_BREATH_EVERY=999999 DS4_PACE_RELEARN=0 DS4_PACE_ROTATE=0 DS4_PACE_WRAP=1 DS4_PACE_WRAP_ROTATE_DELTA=1 DS4_PACE_DEBUG=1"
ADMIT_ON="DS4_PACE_ADMIT=1 DS4_PACE_ADMIT_H=1.2 DS4_PACE_ADMIT_KDRIFT=0.02 DS4_PACE_ADMIT_PERSIST=2 DS4_PACE_ADMIT_COOLDOWN=16 DS4_PACE_ADMIT_MAX_PER_100=0"
ADMIT_OFF="DS4_PACE_ADMIT=0"
runcell(){
  local arm="$1" idx="$2"
  local d="$OUT/${TAG}_${arm}_r${idx}"; mkdir -p "$d"
  local extra="$ADMIT_OFF"; [ "$arm" = "B" ] && extra="$ADMIT_ON"
  echo "=== CELL ${TAG} ${arm} r${idx} START $(date -u +%H:%M:%S) ==="
  env $BASE $extra DS4_PACE_LOG="$d/pace.jsonl" timeout 4000 "$BIN" -m "$MODEL" \
    --cuda --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 1024 \
    -c "$CTX" --nothink --temp 0 -n "$NTOK" --prompt-file "$PROMPT" \
    > "$d/out.txt" 2> "$d/diag.txt"
  local rc=$?
  echo "=== CELL ${TAG} ${arm} r${idx} DONE rc=$rc $(date -u +%H:%M:%S) chars=$(wc -c < "$d/out.txt") admit=$(grep -c '"ev":"admit"' "$d/pace.jsonl" 2>/dev/null) ==="
}
for i in $(seq 0 $((NRUNS-1))); do
  runcell A "$i"
  runcell B "$i"
done
echo "ALL_DONE_${TAG} $(date -u +%H:%M:%S)"
