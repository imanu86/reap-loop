#!/usr/bin/env bash
# Short syntax/performance screen for patch 0045. This is not an L0-L3 verdict.
set -euo pipefail

MAX_ADD=${1:-}
OUT=${2:-}
if [[ ! "$MAX_ADD" =~ ^(0|1|2|4)$ ]] || [[ -z "$OUT" ]]; then
  echo "usage: $0 0|1|2|4 OUTPUT_DIR" >&2
  exit 2
fi

REPO=${REAP_LOOP_REPO:-/mnt/c/Users/imanu/source/repos/reap-loop}
BIN=${DS4_BIN:-/root/ds4-fullstack/ds4-server}
MODEL=${DS4_MODEL:-/root/models/ds4-2bit.gguf}
SPX1=${DS4_SPX1:-/mnt/c/Users/imanu/source/repos/moe-aggressive-commit/runs/spex/spex_model/ds4flash_d2_nextlayer.spex}
SOURCE_REQUEST=${DS4_REQUEST:-$REPO/runs/ds4/20260709_k23_unit_vs_weighted_cache256_html800/html_local_k23_weighted_warmup_cache256_r01/request_measured.json}
PORT=${PORT:-8070}

if pgrep -x ds4-server >/dev/null; then
  echo "ds4-server is already running; refusing to disturb it" >&2
  exit 1
fi

mkdir -p "$OUT"
python3 - "$SOURCE_REQUEST" "$OUT/request.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as src:
    request = json.load(src)
request["max_tokens"] = 80
request["stream"] = False
request.pop("stream_options", None)
with open(sys.argv[2], "w", encoding="utf-8") as dst:
    json.dump(request, dst, ensure_ascii=False, indent=2)
PY

export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.25
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_PACE=0
export DS4_PACE_LIVEMASK=1
export DS4_PACE_LIVEMASK_BOOTSTRAP=16
export DS4_PACE_LIVEMASK_WINDOW=10
export DS4_PACE_LIVEMASK_K=16
export DS4_PACE_LIVEMASK_K_ADAPTIVE=1
export DS4_PACE_LIVEMASK_K_MIN=16
export DS4_PACE_LIVEMASK_K_MAX=50
export DS4_PACE_LIVEMASK_KNOCK_THRESHOLD=0.15
export DS4_PACE_LIVEMASK_KNOCK_GAIN=0.5
export DS4_PACE_LIVEMASK_KNOCK_STEP_UP=4
export DS4_PACE_LIVEMASK_KNOCK_STEP_DOWN=1
export DS4_PACE_LIVEMASK_KNOCK_DEADBAND=2
export DS4_PACE_LIVEMASK_KNOCK_UPDATE_EVERY=2
export DS4_PACE_LIVEMASK_KNOCK_MIN_HISTORY=2
export DS4_PACE_LIVEMASK_KNOCK_PREFETCH=0
export DS4_PACE_LIVEMASK_OBSERVE_TOP=16
export DS4_PACE_LIVEMASK_X=11
export DS4_PACE_LIVEMASK_MAX_SWAPS=1
export DS4_PACE_LIVEMASK_COOLDOWN=16
export DS4_PACE_LIVEMASK_HYST=999
export DS4_PACE_LIVEMASK_LOG="$OUT/livemask.jsonl"
export DS4_REAP_PIN_BY_MASS=1
export DS4_REAP_PREFETCH_THREADS=16
export DS4_REAP_PREFETCH_LOCK=1
export DS4_PACE_LIVEMASK_SPEX_ADD="$MAX_ADD"
export DS4_PACE_LIVEMASK_SPEX_CADENCE=2
export DS4_PACE_LIVEMASK_SPEX_LOG="$OUT/spex_mask.jsonl"
export DS4_PACE_LIVEMASK_SPEX_PIN_LOG="$OUT/spex_pin.jsonl"
export DS4_EXPERT_TIERING=observe
export DS4_EXPERT_TIERING_LOG=
export DS4_EXPERT_TIERING_LOG_IDS=0

if [[ "$MAX_ADD" -gt 0 ]]; then
  export DS4_REAP_PREFETCH_DELTA=1
  export DS4_PACE_LIVEMASK_SPEX_WRAP=1
  export DS4_SPEX_STATS=1
  export DS4_SPEX_HIDDEN_FILE="$SPX1"
  export DS4_SPEX_HIDDEN_PREFETCH=0
  export DS4_SPEX_HIDDEN_GPU_LOAD=1
  export DS4_SPEX_HIDDEN_GPU_SCORE=1
  export DS4_SPEX_HIDDEN_GPU_PREFETCH=1
  export DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN=0
  export DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1
  export DS4_SPEX_HIDDEN_CAP=54
else
  export DS4_REAP_PREFETCH_DELTA=0
  export DS4_PACE_LIVEMASK_SPEX_WRAP=0
  export DS4_SPEX_HIDDEN_GPU_LOAD=0
  export DS4_SPEX_HIDDEN_GPU_SCORE=0
  export DS4_SPEX_HIDDEN_GPU_PREFETCH=0
fi

env | LC_ALL=C sort > "$OUT/server_env.txt"
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
[[ "$ready" == 1 ]] || { echo "server not ready" >&2; exit 1; }

curl -fsS -m 1800 -H 'Content-Type: application/json' \
  --data-binary @"$OUT/request.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" >"$OUT/response.json"

python3 - "$OUT/response.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    text = json.load(fh)["choices"][0]["message"]["content"]
bad = bool(re.search(r"<!DOCTYPE\s+html\s*\n\s*<html", text, re.I))
print("screen=malformed-doctype" if bad else "screen=no-obvious-prefix-error")
raise SystemExit(3 if bad else 0)
PY
