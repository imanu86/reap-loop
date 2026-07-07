#!/bin/bash
# Cache-sizing sweep for ds4 SPEX expert cache on RTX 3060 (real engine, ds4-2bit IQ2XXS).
# Calibrated to REAL available VRAM ~5.9 GiB (measured): reserve=6 -> disabled; lower reserve -> more slots.
# For each config: run TWICE (rep1 warm-up discarded, rep2 measured). No WSL restart between configs (host page-cache stays warm).
set -u
cd <DS4_DIR>

MODEL="<MODEL_GGUF>"
OUT="<OUT_DIR>"
PROMPT="The quick brown fox jumps over the lazy dog while the sun sets slowly behind the hills."
NTOK="${NTOK:-48}"

mkdir -p "$OUT"
if [ ! -f "$MODEL" ]; then echo "FATAL: model not found: $MODEL"; exit 2; fi

# args: <label> <reserve_gb> <requested_N>
run_cfg () {
  label="$1"; reserve="$2"; reqN="$3"
  for rep in 1 2; do
    log="$OUT/run_${label}_${rep}.log"
    echo ">>> config=$label reserve_gb=$reserve requestedN=$reqN rep=$rep NTOK=$NTOK -> $log"
    DS4_SPEX_STATS=1 \
    DS4_CUDA_NO_DIRECT_IO=1 \
    DS4_CUDA_KEEP_MODEL_PAGES=1 \
    DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB="$reserve" \
    DS4_CUDA_STREAMING_EXPERT_CACHE_N="$reqN" \
      timeout 1200 ./ds4 -m "$MODEL" --cuda --ssd-streaming -c 2048 --nothink -n "$NTOK" \
      -p "$PROMPT" > "$log" 2>&1
    ec=$?
    echo "    exit=$ec lines=$(wc -l < "$log")"
    pkill -9 -x ds4 2>/dev/null
    sleep 2
  done
}

# label       reserve_gb  requestedN   (target slots on ~5.9 GiB available, 6.75 MiB/slot)
run_cfg "s156"   5          2048   # usable ~0.9 GiB -> ~130-160 slots (baseline regime)
run_cfg "s512"   3          2048   # usable ~2.9 GiB -> ~430 slots
run_cfg "s740"   1          2048   # usable ~4.9 GiB -> ~740 slots
run_cfg "smax"   0          2048   # usable ~5.9 GiB -> ~890 slots (max safe)

echo "ALL DONE"
