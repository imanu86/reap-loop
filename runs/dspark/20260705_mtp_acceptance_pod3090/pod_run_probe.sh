#!/bin/bash
# DSpark — misura acceptance MTP-1 via probe, IN STREAMING (guardia bypassata).
# Scoperta chiave (ds4_cli.c:922-947 @80ebbc3): con --temp 0 e --mtp-draft 1 la CLI
# usa ds4_engine_generate_argmax che NON passa da ds4_session_eval -> probe muto.
# Combo corretta: --mtp-draft 2 + DS4_MTP_SPEC_DISABLE=1 (dispatch -> sampled loop,
# spec-dec spento, drafting attivo, verifier mai invocato) + DS4_MTP_PROBE=1.
set -u
DS4=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
MTP=/root/models/ds4-mtp.gguf
OUT=/root/out_probe
mkdir -p "$OUT"
NTOK=150
CTX=2048
CACHE=12GB
ENVP="DS4_MTP_PROBE=1 DS4_MTP_SPEC_DISABLE=1 DS4_MTP_STREAMING_UNSAFE=1"

P_code="Write a Python function that parses a CSV file and returns the sum of the second column. Include error handling and a short docstring."
P_math="Compute step by step: what is the sum of all integers from 1 to 100 that are divisible by 3? Show your reasoning and give the final number."
P_chat="Give me practical, friendly advice for organizing a small team's weekly schedule. Keep it conversational."

{ nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader; free -g | head -2; } > "$OUT/env.txt" 2>&1
echo "run,rc,wall_s" > "$OUT/runtimes.csv"

run_probe() {  # $1=logname $2=prompt
  local log="$OUT/$1.log"; local prompt="$2"
  local t0=$(date +%s)
  env $ENVP "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts $CACHE \
      -p "$prompt" -n $NTOK -c $CTX --temp 0 --nothink \
      --mtp "$MTP" --mtp-draft 2 > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "$1,$rc,$((t1 - t0))" >> "$OUT/runtimes.csv"
  echo "done $1 rc=$rc $((t1 - t0))s last=$(tr '\r' '\n' < "$log" | grep -o 'hit=[0-9]*/[0-9]*' | tail -1)"
}

for pname in code math chat; do
  eval "PROMPT=\$P_$pname"
  for r in 1 2; do
    run_probe "${pname}_probe_r${r}" "$PROMPT"
  done
done

# Verifica runtime union-load (senza MTP): prefill batch streaming con profiling
# della selected-load. Prompt lungo per forzare il path batch split_commands.
LONGP=$(python3 -c "print('Explain in detail how mixture-of-experts routing works in transformer models. ' * 30)")
DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE=1 "$DS4" -m "$MODEL" --cuda --ssd-streaming \
    --ssd-streaming-cache-experts $CACHE -p "$LONGP" -n 4 -c 2048 --temp 0 --nothink \
    > "$OUT/unionload_prefill.log" 2>&1
echo "unionload_prefill,$?,0" >> "$OUT/runtimes.csv"

echo "=== riepilogo ==="
for f in "$OUT"/*_probe_*.log; do
  echo "$f: $(tr '\r' '\n' < "$f" | grep -o 'hit=[0-9]*/[0-9]*' | tail -1)"
done
tr '\r' '\n' < "$OUT/unionload_prefill.log" | grep -m5 'batch selected load'
cat "$OUT/runtimes.csv"
echo "=== ALL DONE ==="
