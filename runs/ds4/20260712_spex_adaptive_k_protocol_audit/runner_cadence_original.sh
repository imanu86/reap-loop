#!/usr/bin/env bash
set -euo pipefail

ROOT=${OUT_ROOT:-/mnt/c/Users/imanu/Documents/Codex/2026-07-07/cia/work/spex_cadence_smoke}
BIN=/root/ds4-fullstack/ds4-server
MODEL=/root/models/ds4-2bit.gguf
SPX1=/mnt/c/Users/imanu/source/repos/moe-aggressive-commit/runs/spex/spex_model/ds4flash_d2_nextlayer.spex
CACHE_EXPERTS=${CACHE_EXPERTS:-256}
CACHE_RESERVE_GB=${CACHE_RESERVE_GB:-0.25}
REQUEST=${REQUEST:-/mnt/c/Users/imanu/Documents/Codex/2026-07-07/cia/work/spex_cadence_smoke/request.json}
REPS=${REPS:-1}
ENABLE_SPEX=${ENABLE_SPEX:-1}
CADENCES=${CADENCES:-"2 4 8"}
SPEX_MAX_ADD=${SPEX_MAX_ADD:-4}
if [[ "$ENABLE_SPEX" == 1 ]]; then
  SPEX_DELTA=1
  SPEX_ADD="$SPEX_MAX_ADD"
  SPEX_WRAP=1
  SPEX_FILE="$SPX1"
  SPEX_GPU=1
else
  SPEX_DELTA=0
  SPEX_ADD=0
  SPEX_WRAP=0
  SPEX_FILE=
  SPEX_GPU=0
fi
mkdir -p "$ROOT"

for cadence in $CADENCES; do
  out="$ROOT/cadence_${cadence}"
  port=$((8050 + cadence))
  mkdir -p "$out"
  cp "$REQUEST" "$out/request.json"
  pkill -x ds4-server 2>/dev/null || true
  sleep 2

  DS4_CUDA_NO_DIRECT_IO=1 \
  DS4_CUDA_KEEP_MODEL_PAGES=1 \
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB="$CACHE_RESERVE_GB" \
  DS4_CUDA_NO_Q8_F16_CACHE=1 \
  DS4_PACE=1 \
  DS4_PACE_WARMUP=16 \
  DS4_PACE_KEEP=23 \
  DS4_PACE_KEEP_MIN=23 \
  DS4_PACE_KEEP_MAX=96 \
  DS4_PACE_KEEP_STEP=0 \
  DS4_PACE_BREATH_EVERY=999999 \
  DS4_PACE_BREATH_KEEP=96 \
  DS4_PACE_BREATH_LEN=80 \
  DS4_PACE_RELEARN=0 \
  DS4_PACE_DRIFT=1.0 \
  DS4_PACE_PREBREATH=0 \
  DS4_PACE_WRAP=1 \
  DS4_PACE_DEBUG=1 \
  DS4_PACE_CACHE_FLOOR=1 \
  DS4_PACE_CACHE_TARGET_SLOTS="$CACHE_EXPERTS" \
  DS4_PACE_CACHE_FLUSH=0 \
  DS4_PACE_PREFILL_APPLY=0 \
  DS4_PACE_PREFILL_WAIT_WRAP=0 \
  DS4_PACE_EXCHANGE_OBSERVE=0 \
  DS4_PACE_ROTATE=0 \
  DS4_PACE_WEIGHTED_SELECTED=0 \
  DS4_PACE_WEIGHTED_WARMUP=1 \
  DS4_PACE_WEIGHTED_RELEARN=0 \
  DS4_EXPERT_TIERING=observe \
  DS4_EXPERT_TIERING_LOG= \
  DS4_EXPERT_TIERING_LOG_IDS=0 \
  DS4_REAP_PREFETCH_THREADS=16 \
  DS4_REAP_PREFETCH_LOCK=1 \
  DS4_REAP_PREFETCH_DELTA="$SPEX_DELTA" \
  DS4_SPEX_STATS=1 \
  DS4_SPEX_HIDDEN_FILE="$SPEX_FILE" \
  DS4_SPEX_HIDDEN_PREFETCH=0 \
  DS4_SPEX_HIDDEN_GPU_LOAD="$SPEX_GPU" \
  DS4_SPEX_HIDDEN_GPU_SCORE="$SPEX_GPU" \
  DS4_SPEX_HIDDEN_GPU_PREFETCH="$SPEX_GPU" \
  DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN=0 \
  DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS="$SPEX_GPU" \
  DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS_EVERY=256 \
  DS4_SPEX_HIDDEN_CAP=27 \
  DS4_PACE_LIVEMASK_SPEX_ADD="$SPEX_ADD" \
  DS4_PACE_LIVEMASK_SPEX_WRAP="$SPEX_WRAP" \
  DS4_PACE_LIVEMASK_SPEX_CADENCE="$cadence" \
  DS4_PACE_LIVEMASK_SPEX_LOG="$out/spex_mask.jsonl" \
  DS4_PACE_LIVEMASK_SPEX_PIN_LOG="$out/spex_pin.jsonl" \
    "$BIN" -m "$MODEL" --cuda --ssd-streaming \
      --ssd-streaming-cache-experts "$CACHE_EXPERTS" --prefill-chunk 512 \
      -c 2048 -n 2048 --host 127.0.0.1 --port "$port" --cors \
      >"$out/server.stdout.log" 2>"$out/server.stderr.log" &
  pid=$!

  ready=0
  for _ in $(seq 1 150); do
    if curl -fsS -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    kill "$pid" 2>/dev/null || true
    echo "cadence=$cadence server-not-ready" >&2
    exit 1
  fi

  for rep in $(seq -w 1 "$REPS"); do
    curl -fsS -m 1800 -H 'Content-Type: application/json' \
      --data-binary @"$out/request.json" \
      "http://127.0.0.1:$port/v1/chat/completions" \
      >"$out/response_r${rep}.json"
  done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
done

pkill -x ds4-server 2>/dev/null || true
