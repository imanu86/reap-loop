#!/bin/bash
# DSpark — notturna 2026-07-05/06 (finestra GPU assegnata da SPEX-main, utente ok).
# JOB1: validazione appaiata ds4-eval ON(unlock+probe) vs OFF -> prerequisito PR upstream.
#       NB: ds4-eval NON usa il verify speculativo (mtp_draft_tokens=1 hardcoded,
#       ds4_eval.c:4142): il --mtp fa probe per token. La coppia valida quindi che
#       0009+0010 con drafter ATTIVO non alterano la qualita' del decode (equivalenza
#       statistica dei punteggi). La correttezza del verify e' coperta da JOB3.
# JOB2: M8 — spec2 CON expert cache attiva (reserve 1GB, fallback 3GB) vs cache-off.
# JOB3: coppie di generazione CLI ON/OFF sui 3 domini (archivio + acceptance).
# Disciplina: sequenziale, un ds4 alla volta, DS4_SPEX_STATS=1 ovunque, log per run,
# progress timestampato. Niente pod.
set -u
DS4=/root/ds4-dspark/ds4
EVAL=/root/ds4-dspark/ds4-eval
MODEL=/root/models/ds4-2bit.gguf
MTP=/root/models/ds4-mtp.gguf
OUT=/root/out_night
mkdir -p "$OUT"
PROG="$OUT/progress.log"
say() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

P_code="Write a Python function that parses a CSV file and returns the sum of the second column. Include error handling and a short docstring."
P_math="Compute step by step: what is the sum of all integers from 1 to 100 that are divisible by 3? Show your reasoning and give the final number."
P_chat="Give me practical, friendly advice for organizing a small team's weekly schedule. Keep it conversational."

say "NOTTURNA START; gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ---------- sanity 1 domanda (fail-fast dei flag) ----------
say "sanity ds4-eval ON (1 domanda, 16 tok)"
env DS4_SPEX_STATS=1 DS4_MTP_STREAMING=1 DS4_MTP_PROBE=1 timeout 900 \
  "$EVAL" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 250 \
  --mtp "$MTP" --questions 1 --tokens 16 --temp 0 --nothink --plain \
  > "$OUT/sanity_on.log" 2>&1
RC=$?
say "sanity rc=$RC"
if [ $RC -ne 0 ] && [ $RC -ne 124 ]; then
  grep -m3 -iE 'error|invalid|unknown' "$OUT/sanity_on.log" >> "$PROG"
  say "SANITY FALLITA - salto JOB1, procedo con JOB2/JOB3"
  SKIP_EVAL=1
else
  SKIP_EVAL=0
fi

# ---------- JOB1: ds4-eval appaiato ----------
if [ "$SKIP_EVAL" = "0" ]; then
  say "JOB1 OFF start (20 domande, 384 tok)"
  env DS4_SPEX_STATS=1 timeout 12600 \
    "$EVAL" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 250 \
    --questions 20 --tokens 384 --temp 0 --nothink --plain \
    --trace "$OUT/eval_off.trace" > "$OUT/eval_off.log" 2>&1
  say "JOB1 OFF rc=$?"
  say "JOB1 ON start (stesse 20 domande, unlock+probe)"
  env DS4_SPEX_STATS=1 DS4_MTP_STREAMING=1 DS4_MTP_PROBE=1 timeout 12600 \
    "$EVAL" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts 250 \
    --mtp "$MTP" --questions 20 --tokens 384 --temp 0 --nothink --plain \
    --trace "$OUT/eval_on.trace" > "$OUT/eval_on.log" 2>&1
  say "JOB1 ON rc=$?"
fi

# ---------- JOB2: M8 cache-on ----------
m8_run() { # nome envextra argsextra...
  local name="$1"; shift; local envx="$1"; shift
  local t0=$(date +%s)
  env DS4_SPEX_STATS=1 $envx timeout 1800 "$DS4" -m "$MODEL" --cuda --ssd-streaming \
      -p "$P_code" -n 100 -c 2048 --temp 0 --nothink "$@" > "$OUT/$name.log" 2>&1
  local rc=$?; local t1=$(date +%s)
  echo "$name,$rc,$((t1-t0))" >> "$OUT/m8_runtimes.csv"
  say "JOB2 $name rc=$rc $((t1-t0))s $(tr -d '\0' < $OUT/$name.log | tr '\r' '\n' | grep -m1 'generation:')"
}
echo "run,rc,wall_s" > "$OUT/m8_runtimes.csv"
say "JOB2 M8 start (reserve 1GB)"
R1="DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1"
SPEC="DS4_MTP_STREAMING=1 DS4_MTP_CONF_LOG=1 DS4_MTP_TIMING=1"
m8_run m8_base_res1_r1 "$R1" --ssd-streaming-cache-experts 400
m8_run m8_spec2_res1_r1 "$R1 $SPEC" --ssd-streaming-cache-experts 400 --mtp "$MTP" --mtp-draft 2 --mtp-margin 0
m8_run m8_base_res1_r2 "$R1" --ssd-streaming-cache-experts 400
m8_run m8_spec2_res1_r2 "$R1 $SPEC" --ssd-streaming-cache-experts 400 --mtp "$MTP" --mtp-draft 2 --mtp-margin 0
if grep -q 'cache disabled' "$OUT/m8_spec2_res1_r1.log"; then
  say "JOB2: cache disabled con reserve 1GB (dato per 0011) - provo reserve 3GB + 250 exp"
  m8_run m8_spec2_res3_r1 "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=3 $SPEC" --ssd-streaming-cache-experts 250 --mtp "$MTP" --mtp-draft 2 --mtp-margin 0
  m8_run m8_base_res3_r1 "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=3" --ssd-streaming-cache-experts 250
fi
say "JOB2 M8 spec4 bonus"
m8_run m8_spec4_res1_r1 "$R1 $SPEC" --ssd-streaming-cache-experts 400 --mtp "$MTP" --mtp-draft 4 --mtp-margin 0

# ---------- JOB3: coppie generazione 3 domini (verify attivo) ----------
say "JOB3 coppie CLI start"
for d in code math chat; do
  eval "PP=\$P_$d"
  env DS4_SPEX_STATS=1 timeout 1200 "$DS4" -m "$MODEL" --cuda --ssd-streaming \
      --ssd-streaming-cache-experts 250 -p "$PP" -n 80 -c 2048 --temp 0 --nothink \
      > "$OUT/gen_${d}_off.log" 2>&1
  say "JOB3 ${d} OFF rc=$?"
  env DS4_SPEX_STATS=1 DS4_MTP_STREAMING=1 DS4_MTP_CONF_LOG=1 timeout 1200 "$DS4" -m "$MODEL" \
      --cuda --ssd-streaming --ssd-streaming-cache-experts 250 -p "$PP" -n 80 -c 2048 \
      --temp 0 --nothink --mtp "$MTP" --mtp-draft 2 --mtp-margin 0 \
      > "$OUT/gen_${d}_on.log" 2>&1
  say "JOB3 ${d} ON rc=$? conf=$(tr -d '\0' < $OUT/gen_${d}_on.log | tr '\r' '\n' | grep -c 'mtp conf')"
done

say "NOTTURNA DONE"
echo DONE > "$OUT/NIGHT_DONE"
