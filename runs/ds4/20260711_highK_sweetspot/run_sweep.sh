#!/bin/bash
# High-K sweet-spot sweep on the LOCAL 3060. Two tiers per K:
#  (1a) fast warm t/s probe (-n 300) -> clean warm velocity curve (decisive Q-a)
#  (1b) full-budget quality run (-n 5500) -> does high-K render/close </html>? (Q-b/c)
# cache32, cyberpunk-wide prompt, warm-controlled (one discard warmup), NO route trace
# (pure speed). RAM/swap/GPU sampled during every run -> RAM-fit answer (Q: keep-set warm?).
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
RUN=$REPO/runs/ds4/20260711_highK_sweetspot
MASKS=$RUN/masks
PROMPT=$RUN/prompt_cyberpunk_wide.txt
OUT=$RUN/sweep
LOG=$OUT/progress.log
mkdir -p "$OUT"
: > "$LOG"

export DS4_LOCK_FILE=/tmp/ds4_highK_sweetspot.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_SPEX_STATS=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1

BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf

sample_mem () {  # $1=memlog ; samples until killed
  local f=$1
  echo "iso,mem_used_mb,mem_free_mb,buffcache_mb,swap_used_mb,gpu_used_mb" > "$f"
  while true; do
    local m=$(free -m | awk '/^Mem:/{print $3","$4","$6} /^Swap:/{print $3}')
    local mu=$(echo "$m" | sed -n 1p)
    local sw=$(echo "$m" | sed -n 2p)
    local g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "$(date -Is),$mu,$sw,$g" >> "$f"
    sleep 10
  done
}

run_one () {  # $1=name $2=K(mask, 0=none) $3=ntok $4=ctx
  local NAME=$1 K=$2 NTOK=$3 CTX=$4
  local D="$OUT/$NAME"; mkdir -p "$D"
  if [ "$K" = "0" ]; then unset DS4_REAP_MASK_FILE; else export DS4_REAP_MASK_FILE=$MASKS/sessK$K.txt; fi
  echo "[sweep] START $NAME K=$K ntok=$NTOK ctx=$CTX $(date -Is)" >> "$LOG"
  sample_mem "$D/mem.log" &
  local SPID=$!
  local T0=$(date +%s)
  timeout 5400 "$BIN" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts 32 -c "$CTX" --nothink --temp 0.0 -n "$NTOK" \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  local RC=$?
  local T1=$(date +%s)
  kill "$SPID" 2>/dev/null
  local PF=$(grep -oE "prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s" "$D/diag.txt" | tail -1)
  local PKSWAP=$(awk -F, 'NR>1{if($5>m)m=$5}END{print m+0}' "$D/mem.log")
  local PKMEM=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m+0}' "$D/mem.log")
  local HIT=$(grep -oE "hit[- ]?rate[^0-9]*[0-9.]+" "$D/diag.txt" | tail -1)
  local CHARS=$(wc -c < "$D/gen.txt")
  local CLOSE=$(grep -c "</html>" "$D/gen.txt")
  echo "[sweep] DONE $NAME rc=$RC wall=$((T1-T0))s perf=[$PF] peakMemMB=$PKMEM peakSwapMB=$PKSWAP gen_chars=$CHARS close_html=$CLOSE $(date -Is)" >> "$LOG"
}

echo "[sweep] ==== PHASE 1a: warm t/s probe (-n 300) ====" >> "$LOG"
run_one warmup_K48   48 300 4096   # discard (cold prime shared weights)
run_one probe_K48    48 300 4096
run_one probe_K64    64 300 4096
run_one probe_K91    91 300 4096

echo "[sweep] ==== PHASE 1b: quality full-budget (-n 5500) ====" >> "$LOG"
run_one qual_K48     48 5500 8192
run_one qual_K64     64 5500 8192
run_one qual_K91     91 5500 8192

echo "[sweep] SWEEP_COMPLETE $(date -Is)" >> "$LOG"
