#!/bin/bash
# Reproduce the historical W50->K23 Cyberpunk runner with one controlled delta:
# CONTROL disables SPEX; SPEX enables the additive four-expert mask/VRAM/WRAP lane.
set -euo pipefail

VARIANT=${1:-}
OUT=${2:-}
if [[ "$VARIANT" != "control" && "$VARIANT" != "spex" ]] || [[ -z "$OUT" ]]; then
  echo "usage: $0 control|spex OUTPUT_DIR" >&2
  exit 2
fi

REPO=${REAP_LOOP_REPO:-/mnt/c/Users/imanu/source/repos/reap-loop}
BIN=${DS4_BIN:-/root/ds4-fullstack/ds4-server}
MODEL=${DS4_MODEL:-/root/models/ds4-2bit.gguf}
SPX1=${DS4_SPX1:-/mnt/c/Users/imanu/source/repos/moe-aggressive-commit/runs/spex/spex_model/ds4flash_d2_nextlayer.spex}
REQUEST=${DS4_REQUEST:-$REPO/runs/ds4/20260709_k23_unit_vs_weighted_cache256_html800/html_local_k23_weighted_warmup_cache256_r01/request_measured.json}
PORT=${PORT:-8044}
REPS=${REPS:-3}

mkdir -p "$OUT"
cp "$REQUEST" "$OUT/request.json"

export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.25
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_PACE=1
export DS4_PACE_WARMUP=50
export DS4_PACE_KEEP=23
export DS4_PACE_KEEP_MIN=23
export DS4_PACE_KEEP_MAX=96
export DS4_PACE_KEEP_STEP=0
export DS4_PACE_BREATH_EVERY=999999
export DS4_PACE_BREATH_KEEP=96
export DS4_PACE_BREATH_LEN=80
export DS4_PACE_RELEARN=0
export DS4_PACE_DRIFT=1.0
export DS4_PACE_PREBREATH=0
export DS4_PACE_PREBREATH_DRIFT=0.18
export DS4_PACE_PREBREATH_EVERY=64
export DS4_PACE_PREBREATH_KEEP_MAX=96
export DS4_PACE_WRAP=1
export DS4_PACE_DEBUG=1
export DS4_PACE_CACHE_FLOOR=1
export DS4_PACE_CACHE_TARGET_SLOTS=256
export DS4_PACE_CACHE_FLUSH=0
export DS4_PACE_PREFILL_APPLY=0
export DS4_PACE_PREFILL_WAIT_WRAP=0
export DS4_PACE_EXCHANGE_OBSERVE=0
export DS4_PACE_ROTATE=0
export DS4_PACE_ROTATE_EVERY=32
export DS4_PACE_ROTATE_DECAY=0.98
export DS4_PACE_WEIGHTED_SELECTED=0
export DS4_PACE_WEIGHTED_WARMUP=1
export DS4_PACE_WEIGHTED_RELEARN=0
export DS4_EXPERT_TIERING=observe
export DS4_EXPERT_TIERING_LOG=
export DS4_EXPERT_TIERING_LOG_IDS=0
export DS4_REAP_PREFETCH_THREADS=16
export DS4_REAP_PREFETCH_LOCK=1
export DS4_SPEX_STATS=1

if [[ "$VARIANT" == "spex" ]]; then
  export DS4_REAP_PREFETCH_DELTA=1
  export DS4_PACE_LIVEMASK_SPEX_ADD=4
  export DS4_PACE_LIVEMASK_SPEX_WRAP=1
  export DS4_PACE_LIVEMASK_SPEX_LOG="$OUT/spex_mask.jsonl"
  export DS4_PACE_LIVEMASK_SPEX_PIN_LOG="$OUT/spex_pin.jsonl"
  export DS4_SPEX_HIDDEN_FILE="$SPX1"
  export DS4_SPEX_HIDDEN_PREFETCH=0
  export DS4_SPEX_HIDDEN_GPU_LOAD=1
  export DS4_SPEX_HIDDEN_GPU_SCORE=1
  export DS4_SPEX_HIDDEN_GPU_PREFETCH=1
  export DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN=0
  export DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1
  export DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS_EVERY=256
  export DS4_SPEX_HIDDEN_CAP=27
else
  export DS4_REAP_PREFETCH_DELTA=0
  export DS4_PACE_LIVEMASK_SPEX_ADD=0
  export DS4_PACE_LIVEMASK_SPEX_WRAP=0
  export DS4_SPEX_HIDDEN_PREFETCH=0
  export DS4_SPEX_HIDDEN_GPU_LOAD=0
  export DS4_SPEX_HIDDEN_GPU_SCORE=0
  export DS4_SPEX_HIDDEN_GPU_PREFETCH=0
  export DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=0
fi

env | LC_ALL=C sort > "$OUT/server_env.txt"
"$BIN" -m "$MODEL" --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 256 --prefill-chunk 512 \
  -c 2048 -n 2048 --host 127.0.0.1 --port "$PORT" --cors \
  >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 120); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo "server not ready" >&2; exit 1; }

for rep in $(seq -w 1 "$REPS"); do
  curl -fsS -m 1800 -H 'Content-Type: application/json' \
    --data-binary @"$OUT/request.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    > "$OUT/response_r${rep}.json"
done
