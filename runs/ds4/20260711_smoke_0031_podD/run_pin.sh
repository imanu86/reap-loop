#!/bin/bash
# usage: run_pin.sh <label> <pin 0|1> <rotate 0|1>
cd /root/ds4_pin
source /root/smoke_0031/common_env.sh
L="$1"; PIN="${2:-0}"; ROT="${3:-0}"
export DS4_PACE_PIN="$PIN" DS4_PACE_PIN_ROTATE="$ROT"
export DS4_PACE_PIN_WARMUP=${PINWARM:-512} DS4_PACE_PIN_BUDGET_MB=${PINBUDGET:-3500}
export DS4_PACE_PIN_LOG="$D/${L}_events.jsonl"
export DS4_SPEX_TRACE_TOKENS="$D/${L}_tokens.csv"
rm -f "$DS4_PACE_PIN_LOG" "$DS4_SPEX_TRACE_TOKENS"
t0=$(date +%s)
$BIN $CLI > "$D/${L}_gen.txt" 2> "$D/${L}_run.log"
rc=$?
echo "RC=$rc elapsed=$(( $(date +%s)-t0 ))s pin=$PIN rot=$ROT cache=$CACHE ntok=$NTOK" | tee "$D/${L}_status.txt"
grep 'SPEX stats' "$D/${L}_run.log" | tail -1
echo -n "pin_freeze="; grep -c '"pin_freeze"' "$D/${L}_events.jsonl" 2>/dev/null || echo 0
echo -n "pin_rotate="; grep -c '"pin_rotate"' "$D/${L}_events.jsonl" 2>/dev/null || echo 0
echo -n "gen_tokens_csv_lines="; wc -l < "$D/${L}_tokens.csv" 2>/dev/null || echo 0
