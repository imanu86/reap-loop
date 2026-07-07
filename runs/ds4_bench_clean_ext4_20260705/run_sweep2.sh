#!/bin/bash
# Sweep pulito cache-reserve: A=default(6GB) vs B=1GB. Script staccato, robusto.
cd <DS4_DIR>
D=<OUT_DIR>
PROMPT="Explain briefly how expert caching reduces latency in mixture of experts inference engines."
export DS4_SPEX_STATS=1 DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1
run() {  # $1=nome  $2=reserve_env(vuoto=default)
  if [ -n "$2" ]; then export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=$2; else unset DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB; fi
  echo "[$(date +%H:%M:%S)] start $1 (reserve=${2:-default})" >> $D/sweep2_progress.log
  ./ds4 -m <MODEL_GGUF> --cuda --ssd-streaming --ssd-streaming-cache-experts 2048 \
        -c 2048 --nothink -n 32 -p "$PROMPT" > $D/run_$1.log 2>&1
  echo "[$(date +%H:%M:%S)] end $1 exit=$?" >> $D/sweep2_progress.log
}
run A1 ""; run A2 ""; run B1 1; run B2 1
echo DONE $(date +%H:%M:%S) > $D/SWEEP2_DONE
