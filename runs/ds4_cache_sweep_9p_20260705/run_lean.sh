#!/bin/bash
# Lean cache sweep: single measured runs at the high-value large-cache configs.
# Page cache already hot from s156 runs. NTOK reduced to keep generation bounded.
set -u
cd <DS4_DIR>
MODEL="<MODEL_GGUF>"
OUT="<OUT_DIR>"
PROMPT="The quick brown fox jumps over the lazy dog while the sun sets slowly behind the hills."
NTOK="${NTOK:-24}"
mkdir -p "$OUT"

run_cfg () {
  label="$1"; reserve="$2"; reqN="$3"
  log="$OUT/run_${label}.log"
  echo ">>> config=$label reserve_gb=$reserve requestedN=$reqN NTOK=$NTOK -> $log"
  DS4_SPEX_STATS=1 \
  DS4_CUDA_NO_DIRECT_IO=1 \
  DS4_CUDA_KEEP_MODEL_PAGES=1 \
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB="$reserve" \
  DS4_CUDA_STREAMING_EXPERT_CACHE_N="$reqN" \
    timeout 1500 ./ds4 -m "$MODEL" --cuda --ssd-streaming -c 2048 --nothink -n "$NTOK" \
    -p "$PROMPT" > "$log" 2>&1
  ec=$?
  echo "    exit=$ec lines=$(wc -l < "$log")"
  pkill -9 -x ds4 2>/dev/null
  sleep 3
}

run_cfg "s440_r3"   3   2048   # usable ~2.9 GiB -> ~430 slots (mid)
run_cfg "s740_r1"   1   2048   # usable ~4.9 GiB -> ~740 slots (large)
run_cfg "smax_r0"   0   2048   # usable ~5.9 GiB -> ~880 slots (max safe)
echo "ALL DONE LEAN"
