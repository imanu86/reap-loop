#!/bin/bash
# Probe tps low-K (esperimento #3 decision model fa35dd6): fase-2 W50 con mask
# weighted K12 e K16, 400 tok, n=1 per K, misura t/s. CLI-direct co-resident
# (server UI porta 8000 attivo): numeri CONSERVATIVI (resident-hit~0).
set -u
REPO=/mnt/c/Users/imanu/source/repos/reap-loop
OUT=$REPO/runs/ds4/20260711_local_lowK_tps
LOG=$OUT/probe.log
export DS4_LOCK_FILE=/tmp/ds4_s2.lock
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_REAP_PREFETCH_THREADS=16
export DS4_REAP_PREFETCH_LOCK=1
PROMPT=$REPO/runs/ds4/20260710_t4_t5_w_sweep_local/t4_W050/W050/r00/p2prompt.txt
echo "[probe] start $(date -Is)" >> "$LOG"
for K in 12 16; do
  D=$OUT/K$K
  mkdir -p "$D"
  echo "[probe] K=$K start $(date -Is)" >> "$LOG"
  DS4_REAP_MASK_FILE=$OUT/sessK$K.txt timeout 3600 /root/ds4/ds4 \
    -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts 256 -c 4096 --nothink --temp 0.0 -n 400 \
    --prompt-file "$PROMPT" > "$D/gen.txt" 2> "$D/diag.txt"
  echo "[probe] K=$K done rc=$? $(date -Is)" >> "$LOG"
  grep -o "prefill: [0-9.]* t/s, generation: [0-9.]* t/s" "$D/diag.txt" | tail -1 >> "$LOG"
done
echo "[probe] ALL DONE $(date -Is)" >> "$LOG"
