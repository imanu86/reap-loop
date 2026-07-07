#!/bin/bash
# DSpark Strada A — v3 (pod 3090 sano).
# Lezioni v2: (1) non-streaming+--mtp su 24GB = OOM VRAM (path vuole residenza device;
# ok solo su unified memory tipo GB10) -> acceptance SI MISURA IN STREAMING con la combo
# validata: DS4_MTP_STREAMING_UNSAFE + DS4_MTP_PROBE + DS4_MTP_SPEC_DISABLE + --mtp-draft 2
# (dispatch CLI ds4_cli.c:922 richiede draft>1 per passare da ds4_session_eval).
# (2) --simulate-used-memory fallisce per RLIMIT_MEMLOCK container -> per i CONTEGGI
# union-load il memlock e' irrilevante: profiler senza memlock; ulimit tentato come bonus.
set -u
DS4=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
MTP=/root/models/ds4-mtp.gguf
OUT=/root/out_v3
mkdir -p "$OUT"
NTOK=150
CTX=2048
CACHE=12GB
ENVP="DS4_MTP_PROBE=1 DS4_MTP_SPEC_DISABLE=1 DS4_MTP_STREAMING_UNSAFE=1"

P_code="Write a Python function that parses a CSV file and returns the sum of the second column. Include error handling and a short docstring."
P_math="Compute step by step: what is the sum of all integers from 1 to 100 that are divisible by 3? Show your reasoning and give the final number."
P_chat="Give me practical, friendly advice for organizing a small team's weekly schedule. Keep it conversational."

{ nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader; free -g | head -2; ulimit -l; } > "$OUT/env.txt" 2>&1
echo "run,rc,wall_s" > "$OUT/runtimes.csv"

echo "== warmup streaming (scartato, riempie page-cache+expert-cache) =="
env $ENVP "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts $CACHE \
    -p "$P_code" -n 12 -c $CTX --temp 0 --nothink --mtp "$MTP" --mtp-draft 2 \
    > "$OUT/warmup.log" 2>&1
echo "warmup,$?,0" >> "$OUT/runtimes.csv"

for pname in code math chat; do
  eval "PROMPT=\$P_$pname"
  for r in 1 2; do
    log="$OUT/accept_${pname}_r${r}.log"
    t0=$(date +%s)
    env $ENVP "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts $CACHE \
        -p "$PROMPT" -n $NTOK -c $CTX --temp 0 --nothink --mtp "$MTP" --mtp-draft 2 \
        > "$log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "accept_${pname}_r${r},$rc,$((t1-t0))" >> "$OUT/runtimes.csv"
    echo "done accept_${pname}_r${r} rc=$rc $((t1-t0))s $(tr '\r' '\n' < "$log" | grep -o 'hit=[0-9]*/[0-9]*' | tail -1)"
  done
done

echo "== union-load: prefill batch streaming con profiler (NO memlock: i conteggi valgono) =="
LONGP=$(python3 -c "print('Explain in detail how mixture-of-experts routing works in transformer models, covering the gating network, expert selection, load balancing, and the role of the router. ' * 20)")
DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE=1 DS4_CUDA_STREAMING_SELECTED_LOG=1 \
  "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts $CACHE \
  -p "$LONGP" -n 4 -c 2048 --temp 0 --nothink > "$OUT/unionload_prefill.log" 2>&1
echo "unionload_prefill,$?,0" >> "$OUT/runtimes.csv"

echo "== bonus: memlock se il container lo consente =="
if ulimit -l unlimited 2>/dev/null; then
  env $ENVP "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts $CACHE \
      --simulate-used-memory 100GB -p "$P_code" -n 30 -c $CTX --temp 0 --nothink \
      --mtp "$MTP" --mtp-draft 2 > "$OUT/accept_memlock_code.log" 2>&1
  echo "accept_memlock_code,$?,0" >> "$OUT/runtimes.csv"
else
  echo "ulimit memlock non alzabile: skip" | tee "$OUT/accept_memlock_code.log"
fi

echo "== riepilogo =="
for f in "$OUT"/accept_*.log "$OUT"/warmup.log; do
  echo "$(basename $f): $(tr '\r' '\n' < "$f" | grep -o 'hit=[0-9]*/[0-9]*' | tail -1)"
done
tr '\r' '\n' < "$OUT/unionload_prefill.log" | grep -m6 -E 'batch selected load|streaming selected'
cat "$OUT/runtimes.csv"
echo "=== ALL DONE V3 ==="
