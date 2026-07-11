#!/bin/bash
# usage: run_tier.sh <label> <tier 0|1> [cache] [keep] [prompt] [twarmup]
cd /root/ds4_0033
L="$1"; TIER="${2:-0}"
export CACHE="${3:-256}"; export KEEP="${4:-12}"; export PROMPT="${5:-/root/stage0033/cyberpunk_prompt.txt}"
source /root/smoke_0033/common_env0033.sh
CLI="-m $MODEL --prompt-file $PROMPT -n $NTOK -c 8192 --temp 0 --nothink --ssd-streaming --ssd-streaming-cache-experts $CACHE"
export DS4_PACE_TIER="$TIER"
export DS4_PACE_TIER_WARMUP="${6:-512}" DS4_PACE_TIER_X=3 DS4_PACE_TIER_Y=5 DS4_PACE_TIER_HYST=1.0
export DS4_PACE_TIER_VRAM_SLOTS=394 DS4_PACE_TIER_DECAY=0.98 DS4_PACE_TIER_KNOCK=1.0 DS4_PACE_TIER_COOLDOWN=64
export DS4_PACE_TIER_LOG="$D/${L}_events.jsonl"
export DS4_SPEX_TRACE_TOKENS="$D/${L}_tokens.csv"
rm -f "$DS4_PACE_TIER_LOG" "$DS4_SPEX_TRACE_TOKENS"
t0=$(date +%s)
$BIN $CLI > "$D/${L}_gen.txt" 2> "$D/${L}_run.log"
rc=$?
echo "RC=$rc elapsed=$(( $(date +%s)-t0 ))s tier=$TIER cache=$CACHE keep=$KEEP twarm=${6:-512} ntok=$NTOK prompt=$(basename $PROMPT)" | tee "$D/${L}_status.txt"
grep "SPEX stats" "$D/${L}_run.log" | tail -1 | tee "$D/${L}_run.log.stats" >/dev/null
echo -n "  seed=";    grep -c "\"tier_seed\""    "$DS4_PACE_TIER_LOG" 2>/dev/null || echo 0
echo -n "  promote="; grep -c "\"tier_promote\"" "$DS4_PACE_TIER_LOG" 2>/dev/null || echo 0
echo -n "  demote=";  grep -c "\"tier_demote\""  "$DS4_PACE_TIER_LOG" 2>/dev/null || echo 0
echo -n "  swap=";    grep -c "\"tier_swap\""    "$DS4_PACE_TIER_LOG" 2>/dev/null || echo 0
echo -n "  tokens_csv_lines="; wc -l < "$D/${L}_tokens.csv" 2>/dev/null || echo 0
