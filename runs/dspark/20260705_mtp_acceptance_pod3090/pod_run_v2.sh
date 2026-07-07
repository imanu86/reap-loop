#!/bin/bash
# DSpark Strada A — misura su pod 3090 (fresh, machine gkckwlfw6fwt).
# Due fasi nettamente separate:
#
# FASE 1 — ACCEPTANCE (proprieta' del modello -> TRASFERISCE al 3060):
#   non-streaming (--cuda, modello host-mapped in RAM, experts via PCIe on-demand).
#   Nessuna patch. Combo che fa scattare il probe (scoperta ds4_cli.c:922 + :483):
#   --mtp-draft 2 + DS4_MTP_SPEC_DISABLE=1 -> dispatch a run_sampled_generation ma
#   spec-path OFF -> cade su ds4_session_eval -> eval_internal(probe=true) -> drafta
#   e conta hit/total. Questo e' l'MTP-1 baseline del paper misurato sul flusso reale.
#
# FASE 2 — UNION-LOAD (conteggi expert unici/blocco -> TRASFERISCONO; t/s NO):
#   streaming (--ssd-streaming) + --simulate-used-memory per emulare la fame di RAM
#   del 3060 (28GB) su una macchina con piu' RAM. Profiler batch-selected-load ON.
#   Verifica che la compact-load per blocco (ds4_cuda.cu:3176) scatti e misura gli
#   expert unici. NB: la banda IO del pod != NVMe del 3060 -> il guadagno t/s finale
#   resta da misurare sul 3060 reale; qui provo MECCANISMO + CONTEGGI.
set -u
DS4=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
MTP=/root/models/ds4-mtp.gguf
OUT=/root/out_v2
mkdir -p "$OUT"
NTOK=150
CTX=2048

P_code="Write a Python function that parses a CSV file and returns the sum of the second column. Include error handling and a short docstring."
P_math="Compute step by step: what is the sum of all integers from 1 to 100 that are divisible by 3? Show your reasoning and give the final number."
P_chat="Give me practical, friendly advice for organizing a small team's weekly schedule. Keep it conversational."

{ nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader; free -g | head -2; } > "$OUT/env.txt" 2>&1
echo "run,rc,wall_s" > "$OUT/runtimes.csv"

timed() {  # $1=log $2=env $3.. = cmd
  local log="$OUT/$1.log"; shift; local e="$1"; shift
  local t0=$(date +%s)
  env $e "$@" > "$log" 2>&1; local rc=$?
  local t1=$(date +%s)
  echo "$1_placeholder,$rc,$((t1-t0))" >> "$OUT/runtimes.csv"
  echo "done rc=$rc $((t1-t0))s :: $(basename $log)"
}

echo "========== FASE 1: ACCEPTANCE (non-streaming) =========="
for pname in code math chat; do
  eval "PROMPT=\$P_$pname"
  echo "--- warmup $pname (scartato) ---"
  env DS4_MTP_PROBE=1 DS4_MTP_SPEC_DISABLE=1 "$DS4" -m "$MODEL" --cuda \
      -p "$PROMPT" -n 8 -c $CTX --temp 0 --nothink --mtp "$MTP" --mtp-draft 2 \
      > "$OUT/warmup_${pname}.log" 2>&1
  for r in 1 2; do
    log="$OUT/accept_${pname}_r${r}.log"
    t0=$(date +%s)
    env DS4_MTP_PROBE=1 DS4_MTP_SPEC_DISABLE=1 "$DS4" -m "$MODEL" --cuda \
        -p "$PROMPT" -n $NTOK -c $CTX --temp 0 --nothink --mtp "$MTP" --mtp-draft 2 \
        > "$log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "accept_${pname}_r${r},$rc,$((t1-t0))" >> "$OUT/runtimes.csv"
    echo "done accept_${pname}_r${r} rc=$rc $((t1-t0))s hit=$(tr '\r' '\n' < "$log" | grep -o 'hit=[0-9]*/[0-9]*' | tail -1)"
  done
done

echo "========== FASE 2: UNION-LOAD (streaming + memlock) =========="
# Blocca RAM per forzare lo streaming: RAM_pod - ~20GB headroom cache. Su 125GB -> lock 100GB.
RAMLOCK=100GB
LONGP=$(python3 -c "print('Explain in detail how mixture-of-experts routing works in transformer models, covering the gating network, expert selection, load balancing, and the role of the router. ' * 20)")
echo "--- prefill union-load profile (no MTP, streaming, memlock=$RAMLOCK) ---"
DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE=1 \
  "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 1GB \
  --simulate-used-memory $RAMLOCK -p "$LONGP" -n 4 -c 2048 --temp 0 --nothink \
  > "$OUT/unionload_prefill.log" 2>&1
echo "unionload_prefill,$?,0" >> "$OUT/runtimes.csv"

echo "--- probe streaming (MTP+streaming, bypass guard, memlock) ---"
DS4_MTP_PROBE=1 DS4_MTP_SPEC_DISABLE=1 DS4_MTP_STREAMING_UNSAFE=1 \
  "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 1GB \
  --simulate-used-memory $RAMLOCK -p "$P_code" -n 40 -c $CTX --temp 0 --nothink \
  --mtp "$MTP" --mtp-draft 2 > "$OUT/accept_streaming_code.log" 2>&1
echo "accept_streaming_code,$?,0" >> "$OUT/runtimes.csv"

echo "========== RIEPILOGO =========="
for f in "$OUT"/accept_*.log; do
  echo "$(basename $f): hit=$(tr '\r' '\n' < "$f" | grep -o 'hit=[0-9]*/[0-9]*' | tail -1)"
done
echo "--- union-load righe profiler ---"
tr '\r' '\n' < "$OUT/unionload_prefill.log" | grep -m8 -E 'batch selected load|compact|unique' || echo "(nessuna riga profiler - controlla gating)"
cat "$OUT/runtimes.csv"
echo "=== ALL DONE ==="
