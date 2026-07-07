#!/bin/bash
# Fase C — smoke sblocco MTP+streaming sul 3060 REALE (worktree /root/ds4-dspark, patch 0009).
# DA LANCIARE SOLO DOPO HANDOFF GPU dal track SPEX-main. Un solo ds4 alla volta.
# Regime: quello DwarfStar vero (2-bit, --cuda --ssd-streaming). Etichettare cold/warm.
set -u
DS4=/root/ds4-dspark/ds4
MODEL=/root/models/ds4-2bit.gguf
MTP=/root/models/ds4-mtp.gguf   # copiare da D: se assente: /mnt/d/models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
OUT=/root/out_fase_c
mkdir -p "$OUT"
NTOK=100
CTX=2048
CACHE=6GB
P_code="Write a Python function that parses a CSV file and returns the sum of the second column. Include error handling and a short docstring."

log_run() { # nome, env-string, argomenti... (DS4_SPEX_STATS=1 sempre: richiesta SPEX-main)
  local name="$1"; shift; local envs="$1"; shift
  local t0=$(date +%s)
  env DS4_SPEX_STATS=1 $envs "$DS4" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts $CACHE \
      -c $CTX --temp 0 --nothink "$@" > "$OUT/$name.log" 2>&1
  local rc=$?; local t1=$(date +%s)
  echo "$name,$rc,$((t1-t0))" >> "$OUT/runtimes.csv"
  echo "done $name rc=$rc $((t1-t0))s :: $(tr -d '\0' < $OUT/$name.log | tr '\r' '\n' | grep -m1 'generation:' )"
}

echo "run,rc,wall_s" > "$OUT/runtimes.csv"
{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; date; } > "$OUT/env.txt"

echo "== 1. baseline streaming senza MTP (riferimento t/s 3060) =="
log_run base_r1 "" -p "$P_code" -n $NTOK
log_run base_r2 "" -p "$P_code" -n $NTOK

echo "== 2. SMOKE CRITICO: MTP+streaming con VERIFIER ATTIVO (draft 2) =="
log_run spec2_r1 "DS4_MTP_STREAMING=1 DS4_MTP_CONF_LOG=1 DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1" \
    -p "$P_code" -n $NTOK --mtp "$MTP" --mtp-draft 2 --mtp-margin 0
log_run spec2_r2 "DS4_MTP_STREAMING=1 DS4_MTP_CONF_LOG=1 DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1" \
    -p "$P_code" -n $NTOK --mtp "$MTP" --mtp-draft 2 --mtp-margin 0

echo "== 3. draft 4 (blocco piu' lungo = piu' union) =="
log_run spec4_r1 "DS4_MTP_STREAMING=1 DS4_MTP_CONF_LOG=1 DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1" \
    -p "$P_code" -n $NTOK --mtp "$MTP" --mtp-draft 4 --mtp-margin 0

echo "== 4. union-load VIVA nel verify (il log da 49%) =="
log_run spec2_verbose "DS4_MTP_STREAMING=1 DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1" \
    -p "$P_code" -n 30 --mtp "$MTP" --mtp-draft 2 --mtp-margin 0

echo "== 5. margin gate default (production behavior) =="
log_run spec2_m3 "DS4_MTP_STREAMING=1 DS4_MTP_TIMING=1" \
    -p "$P_code" -n $NTOK --mtp "$MTP" --mtp-draft 2

echo "== riepilogo =="
cat "$OUT/runtimes.csv"
for f in "$OUT"/spec*_r*.log "$OUT"/spec2_m3.log; do
  echo "-- $(basename $f)"
  tr -d '\0' < "$f" | tr '\r' '\n' | grep -cE 'mtp conf drafted' | sed 's/^/  cicli conf: /'
  tr -d '\0' < "$f" | tr '\r' '\n' | grep -m2 -E 'verifier failed|illegal|error' | sed 's/^/  ⚠ /'
  tr -d '\0' < "$f" | tr '\r' '\n' | grep -m1 'generation:' | sed 's/^/  /'
done
tr -d '\0' < "$OUT/spec2_verbose.log" | tr '\r' '\n' | grep -m6 'streaming selected layer' | sed 's/^/  UNION: /'
echo "=== SMOKE FASE C DONE ==="
