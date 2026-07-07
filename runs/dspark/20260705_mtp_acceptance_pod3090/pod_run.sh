#!/bin/bash
# DSpark track — run acceptance MTP su pod. Log per-run in /root/out.
set -u
DS4=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
MTP=/root/models/ds4-mtp.gguf
OUT=/root/out
mkdir -p "$OUT"
NTOK=200
CTX=2048

P_code="Write a Python function that parses a CSV file and returns the sum of the second column. Include error handling and a short docstring."
P_math="Compute step by step: what is the sum of all integers from 1 to 100 that are divisible by 3? Show your reasoning and give the final number."
P_chat="Give me practical, friendly advice for organizing a small team's weekly schedule. Keep it conversational."

{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; } > "$OUT/env.txt" 2>&1

run_one() {  # $1=logname $2=prompt $3=extra_env(str)  $4...=extra args
  local log="$OUT/$1.log"; shift
  local prompt="$1"; shift
  local extra_env="$1"; shift
  local t0=$(date +%s.%N)
  env $extra_env "$DS4" -m "$MODEL" --cuda -p "$prompt" -n $NTOK -c $CTX --temp 0 --nothink "$@" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")
  echo "$(basename $log .log),$rc,$wall" >> "$OUT/runtimes.csv"
  echo "done $(basename $log .log) rc=$rc ${wall}s"
}

echo "run,rc,wall_s" > "$OUT/runtimes.csv"

echo "=== warmup (scartata) ==="
run_one warmup_code_spec2 "$P_code" "DS4_MTP_CONF_LOG=1" --mtp "$MTP" --mtp-draft 2 --mtp-margin 0

for pname in code math chat; do
  eval "PROMPT=\$P_$pname"
  echo "=== $pname: probe (MTP-1 acceptance per token) ==="
  for r in 1 2; do
    run_one "${pname}_probe_r${r}" "$PROMPT" "DS4_MTP_PROBE=1" --mtp "$MTP" --mtp-draft 1
  done
  echo "=== $pname: spec2 margin0 ==="
  for r in 1 2; do
    run_one "${pname}_spec2_r${r}" "$PROMPT" "DS4_MTP_CONF_LOG=1 DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1" --mtp "$MTP" --mtp-draft 2 --mtp-margin 0
  done
  echo "=== $pname: spec4 margin0 ==="
  for r in 1 2; do
    run_one "${pname}_spec4_r${r}" "$PROMPT" "DS4_MTP_CONF_LOG=1 DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1" --mtp "$MTP" --mtp-draft 4 --mtp-margin 0
  done
done

echo "=== code: spec2 margin default 3.0 (production) ==="
for r in 1 2; do
  run_one "code_spec2m3_r${r}" "$P_code" "DS4_MTP_CONF_LOG=1 DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1" --mtp "$MTP" --mtp-draft 2
done

echo "=== code: baseline senza MTP ==="
for r in 1 2; do
  run_one "code_baseline_r${r}" "$P_code" ""
done

echo "=== riepilogo veloce ==="
for f in "$OUT"/*_probe_*.log; do
  echo "$f: $(grep -o 'hit=[0-9]*/[0-9]*' "$f" | tail -1)"
done
for f in "$OUT"/*_spec*_r*.log; do
  n=$(grep -c 'mtp conf drafted' "$f" 2>/dev/null || echo 0)
  echo "$f: conf_lines=$n"
done
cat "$OUT/runtimes.csv"
echo "=== ALL DONE ==="
