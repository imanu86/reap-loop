#!/bin/bash
# In-engine clock-breath fair trial batch (D6b). Each run = one ds4 continuous
# stream with PACE. Faithful actuator: warmup->K23 hold->clock-breath(K0)->relearn->K23.
# Static control = breath disabled (pure K23 hold). Greedy temp 0, trace off.
BIN=/root/canon/ds4
MODEL=/root/models/ds4-2bit.gguf
COFFEE=/root/prompts/frontpage_prompt.txt
CYBER=/root/prompts/cyberpunk_prompt.txt
ROOT=/root/cb_runs
PROG=$ROOT/batch_progress.log
: > "$PROG"

# arm: name|prompt|ctx|ntok|breath_every|breath_len|breath_keep|keep|runs
ARMS=(
"A1_coffee_breath|$COFFEE|4096|1200|450|70|0|23|3"
"A1_coffee_static|$COFFEE|4096|1200|9999999|70|0|23|3"
"A2_cyber_breath|$CYBER|8192|4050|450|70|0|23|3"
"A2_cyber_static|$CYBER|8192|4050|9999999|70|0|23|3"
"A2b_cyber_breath_early|$CYBER|8192|4050|128|70|0|23|2"
"A3_cyber_k38_breath|$CYBER|8192|4050|450|70|0|38|2"
)

run_one() {
  local name=$1 prompt=$2 ctx=$3 ntok=$4 be=$5 bl=$6 bk=$7 keep=$8 idx=$9
  local d=$ROOT/$name/r$(printf %02d $idx)
  mkdir -p "$d"
  local T0=$(date +%s)
  DS4_PACE=1 DS4_PACE_WARMUP=50 DS4_PACE_KEEP=$keep DS4_PACE_KEEP_MIN=$keep DS4_PACE_KEEP_MAX=$keep \
  DS4_PACE_KEEP_STEP=8 DS4_PACE_BREATH_EVERY=$be DS4_PACE_BREATH_LEN=$bl DS4_PACE_BREATH_KEEP=$bk \
  DS4_PACE_RELEARN=1 DS4_PACE_RELEARN_DECAY=0.3 DS4_PACE_DRIFT=99 DS4_PACE_HYST=200 \
  DS4_PACE_WRAP=1 DS4_PACE_DEBUG=1 DS4_PACE_LOG="$d/pace.jsonl" \
  "$BIN" --cuda -m "$MODEL" --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c "$ctx" --nothink --temp 0 -n "$ntok" --prompt-file "$prompt" \
    > "$d/gen.out" 2> "$d/gen.err"
  local rc=$? el=$(( $(date +%s) - T0 ))
  local tps=$(grep -oE "generation: [0-9.]+ t/s" "$d/gen.err" | tail -1)
  local breaths=$(grep -c "breath(clock)" "$d/pace.jsonl" 2>/dev/null)
  echo "rc=$rc elapsed=${el}s $tps breaths=$breaths chars=$(wc -c < "$d/gen.out")" > "$d/status.txt"
  echo "[$(date -u +%H:%M:%S)] $name r$idx rc=$rc ${el}s $tps breaths=$breaths" >> "$PROG"
}

for spec in "${ARMS[@]}"; do
  IFS='|' read -r name prompt ctx ntok be bl bk keep runs <<< "$spec"
  echo "[$(date -u +%H:%M:%S)] === ARM $name (runs=$runs) ===" >> "$PROG"
  for i in $(seq 0 $((runs-1))); do
    run_one "$name" "$prompt" "$ctx" "$ntok" "$be" "$bl" "$bk" "$keep" "$i"
  done
done
echo "BATCH_DONE $(date -u +%H:%M:%S)" >> "$PROG"
