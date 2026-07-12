#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/mnt/c/Users/imanu/Documents/Codex/2026-07-07/cia/work/adaptive_k_smoke}
BIN=/root/ds4-fullstack/ds4-server
MODEL=/root/models/ds4-2bit.gguf
REQUEST=${REQUEST:-/mnt/c/Users/imanu/Documents/Codex/2026-07-07/cia/work/spex_cadence_smoke/request.json}
PORT=${PORT:-8061}
ADAPTIVE_DELTA=${ADAPTIVE_DELTA:-0}
SPEX_MAX_ADD=${SPEX_MAX_ADD:-0}
SPEX_CADENCE=${SPEX_CADENCE:-2}
SPX1=/mnt/c/Users/imanu/source/repos/moe-aggressive-commit/runs/spex/spex_model/ds4flash_d2_nextlayer.spex
if [[ "$SPEX_MAX_ADD" -gt 0 ]]; then
  PREFETCH_DELTA=1
  SPEX_FILE="$SPX1"
  SPEX_GPU=1
  SPEX_WRAP=1
else
  PREFETCH_DELTA="$ADAPTIVE_DELTA"
  SPEX_FILE=
  SPEX_GPU=0
  SPEX_WRAP=0
fi
mkdir -p "$OUT"
cp "$REQUEST" "$OUT/request.json"
pkill -x ds4-server 2>/dev/null || true
sleep 2

DS4_CUDA_NO_DIRECT_IO=1 \
DS4_CUDA_KEEP_MODEL_PAGES=1 \
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.25 \
DS4_CUDA_NO_Q8_F16_CACHE=1 \
DS4_PACE=0 \
DS4_PACE_LIVEMASK=1 \
DS4_PACE_LIVEMASK_BOOTSTRAP=16 \
DS4_PACE_LIVEMASK_WINDOW=10 \
DS4_PACE_LIVEMASK_K=16 \
DS4_PACE_LIVEMASK_K_ADAPTIVE=1 \
DS4_PACE_LIVEMASK_K_MIN=16 \
DS4_PACE_LIVEMASK_K_MAX=50 \
DS4_PACE_LIVEMASK_KNOCK_THRESHOLD=0.15 \
DS4_PACE_LIVEMASK_KNOCK_GAIN=0.5 \
DS4_PACE_LIVEMASK_KNOCK_STEP_UP=4 \
DS4_PACE_LIVEMASK_KNOCK_STEP_DOWN=1 \
DS4_PACE_LIVEMASK_KNOCK_DEADBAND=2 \
DS4_PACE_LIVEMASK_KNOCK_UPDATE_EVERY=2 \
DS4_PACE_LIVEMASK_KNOCK_MIN_HISTORY=2 \
DS4_PACE_LIVEMASK_KNOCK_PREFETCH=0 \
DS4_PACE_LIVEMASK_OBSERVE_TOP=16 \
DS4_PACE_LIVEMASK_X=11 \
DS4_PACE_LIVEMASK_MAX_SWAPS=1 \
DS4_PACE_LIVEMASK_COOLDOWN=16 \
DS4_PACE_LIVEMASK_HYST=999 \
DS4_PACE_LIVEMASK_LOG="$OUT/livemask.jsonl" \
DS4_REAP_PIN_BY_MASS=1 \
DS4_REAP_PREFETCH_DELTA="$PREFETCH_DELTA" \
DS4_REAP_PREFETCH_THREADS=16 \
DS4_REAP_PREFETCH_LOCK=1 \
DS4_SPEX_STATS=1 \
DS4_SPEX_HIDDEN_FILE="$SPEX_FILE" \
DS4_SPEX_HIDDEN_PREFETCH=0 \
DS4_SPEX_HIDDEN_GPU_LOAD="$SPEX_GPU" \
DS4_SPEX_HIDDEN_GPU_SCORE="$SPEX_GPU" \
DS4_SPEX_HIDDEN_GPU_PREFETCH="$SPEX_GPU" \
DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN=0 \
DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS="$SPEX_GPU" \
DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS_EVERY=256 \
DS4_SPEX_HIDDEN_CAP=54 \
DS4_PACE_LIVEMASK_SPEX_ADD="$SPEX_MAX_ADD" \
DS4_PACE_LIVEMASK_SPEX_WRAP="$SPEX_WRAP" \
DS4_PACE_LIVEMASK_SPEX_CADENCE="$SPEX_CADENCE" \
DS4_PACE_LIVEMASK_SPEX_LOG="$OUT/spex_mask.jsonl" \
DS4_PACE_LIVEMASK_SPEX_PIN_LOG="$OUT/spex_pin.jsonl" \
DS4_EXPERT_TIERING=observe \
DS4_EXPERT_TIERING_LOG= \
DS4_EXPERT_TIERING_LOG_IDS=0 \
  "$BIN" -m "$MODEL" --cuda --ssd-streaming \
    --ssd-streaming-cache-experts 256 --prefill-chunk 512 \
    -c 2048 -n 2048 --host 127.0.0.1 --port "$PORT" --cors \
    >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 150); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || exit 1

curl -fsS -m 1800 -H 'Content-Type: application/json' \
  --data-binary @"$OUT/request.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  >"$OUT/response.json"
