#!/usr/bin/env bash
# Decisive residency test - single arm launcher.
# Runs entirely inside WSL. Holds /tmp/ds4-gpu.lock for the whole GPU phase.
# Never uses pkill; kills only the recorded server PID.
#
# Usage:
#   run_arm.sh ARM ZEROCOPY PINMASS CACHE_N OUTDIR [WARMUP] [MEASURED]
#     ARM       label, e.g. A_baseline_pread
#     ZEROCOPY  on|off  -> sets DS4_CUDA_STREAM_FROM_RAM_MASKED
#     PINMASS   on|off  -> sets DS4_REAP_PIN_BY_MASS + livemask rating-only
#     CACHE_N   integer VRAM expert cache slots (--ssd-streaming-cache-experts)
set -u

ARM="$1"; ZEROCOPY="$2"; PINMASS="$3"; CACHE_N="$4"; OUTDIR="$5"
WARMUP="${6:-120}"; MEASURED="${7:-240}"

SRC=/root/ds4-v2-work
BIN="$SRC/ds4-server"
MODEL=/root/models/ds4-2bit.gguf
MASK=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_virtual_bake/masks/mask60_self.txt
PORT=8096
CTX=2048
PREFILL_CHUNK=512
BUDGET_GB=24
GPU_LOCK=/tmp/ds4-gpu.lock

RUN="$OUTDIR/$ARM"
mkdir -p "$RUN"
STDERR="$RUN/server.stderr.log"
STDOUT="$RUN/server.stdout.log"

echo "[$(date +%T)] ARM=$ARM zerocopy=$ZEROCOPY pinmass=$PINMASS cache=$CACHE_N warm=$WARMUP meas=$MEASURED"

# ---- provenance (no full md5 of the 86GB model: record path+size+mtime) ----
md5sum "$BIN" > "$RUN/binary.md5"
stat -c 'path=%n size=%s mtime=%y' "$MODEL" > "$RUN/model.stat" 2>/dev/null || true
( cd "$SRC" && git rev-parse HEAD ) > "$RUN/src_head.txt" 2>/dev/null || true

# ---- build env ----
SERVER_ENV=(
  "CUDA_VISIBLE_DEVICES=0"
  "DS4_CUDA_NO_DIRECT_IO=1"
  "DS4_CUDA_KEEP_MODEL_PAGES=1"
  "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1"
  "DS4_CUDA_NO_Q8_F16_CACHE=1"
  "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256"
  "DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1"
  "DS4_PACE=0"
  "DS4_REAP_MASK_FILE=$MASK"
  "DS4_SPEX_STATS=1"
  "DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=$BUDGET_GB"
  "DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1"
  "HOME=/root" "LANG=C.UTF-8" "LC_ALL=C.UTF-8" "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib" "TMPDIR=/tmp"
)
if [[ "$ZEROCOPY" == on ]]; then
  SERVER_ENV+=("DS4_CUDA_STREAM_FROM_RAM_MASKED=$MASK")
fi
if [[ "$PINMASS" == on ]]; then
  # residency: pin-by-mass consumer+producer, livemask in RATING-ONLY so the
  # static bake mask (DS4_REAP_MASK_FILE) stays authoritative.
  SERVER_ENV+=(
    "DS4_REAP_PIN_BY_MASS=1"
    "DS4_PACE_LIVEMASK=1"
    "DS4_PACE_LIVEMASK_RATING_ONLY=1"
  )
fi

SERVER_CLI=(
  -m "$MODEL" --cuda --ssd-streaming
  --ssd-streaming-cache-experts "$CACHE_N"
  --prefill-chunk "$PREFILL_CHUNK" -c "$CTX" -n "$((MEASURED+16))"
  --host 127.0.0.1 --port "$PORT" --cors
)

printf '%s\n' "${SERVER_ENV[@]}" | LC_ALL=C sort > "$RUN/server_env.txt"
printf '%q ' "$BIN" "${SERVER_CLI[@]}" > "$RUN/server_argv.txt"; echo >> "$RUN/server_argv.txt"

# ---- generate requests ----
python3 /mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260713_decisive_residency/scripts/make_requests.py "$RUN" "$WARMUP" "$MEASURED" >/dev/null

# ---- guard: no server already up ----
if pgrep -x ds4-server >/dev/null; then echo "ABORT: ds4-server already running"; exit 3; fi

# ---- launch under GPU lock ----
(
  flock -w 20 9 || { echo "ABORT: could not take GPU lock"; exit 4; }

  cd "$SRC"
  env -i "${SERVER_ENV[@]}" "$BIN" "${SERVER_CLI[@]}" >"$STDOUT" 2>"$STDERR" &
  SPID=$!
  echo "$SPID" > "$RUN/server.pid"
  echo "[$(date +%T)] server pid=$SPID, waiting for ready..."

  # ---- wait ready ----
  ready=0
  for i in $(seq 1 400); do
    if ! kill -0 "$SPID" 2>/dev/null; then echo "ABORT: server died during startup"; break; fi
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "ABORT: server not ready"; kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; exit 5
  fi
  echo "[$(date +%T)] ready. warmup ($WARMUP tok)..."

  # ---- warmup ----
  curl -fsS --max-time 1200 -H 'Content-Type: application/json' \
    --data @"$RUN/request_warmup.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > "$RUN/warmup_response.json" 2> "$RUN/warmup_curl.err"
  echo "[$(date +%T)] warmup done. marking baseline."

  # baseline: byte offset in stderr AFTER warmup -> measured-window DIAG starts here
  wc -c < "$STDERR" > "$RUN/stderr_offset_baseline.txt"

  # ---- measured (WARM) ----
  echo "[$(date +%T)] measured ($MEASURED tok)..."
  T0=$(date +%s.%N)
  curl -fsS --max-time 1800 -H 'Content-Type: application/json' \
    --data @"$RUN/request_measured.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > "$RUN/measured_response.json" 2> "$RUN/measured_curl.err"
  T1=$(date +%s.%N)
  echo "measured_wall_s=$(echo "$T1 - $T0" | bc)" > "$RUN/measured_timing.txt"
  echo "[$(date +%T)] measured done. stopping server."

  # ---- stop via recorded PID only ----
  kill "$SPID" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
  kill -0 "$SPID" 2>/dev/null && { echo "escalate SIGKILL to $SPID"; kill -9 "$SPID" 2>/dev/null; }
  wait "$SPID" 2>/dev/null
  echo "[$(date +%T)] server stopped."
) 9>"$GPU_LOCK"

echo "[$(date +%T)] ARM=$ARM complete."
