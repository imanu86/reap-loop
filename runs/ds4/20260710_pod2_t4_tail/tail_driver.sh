#!/bin/bash
# T4 tail driver pod2: attende la fine del sampling, poi W90->W70->W110->W150 n=3 greedy weighted.
LOG=/root/tail_driver.log
echo "[tail] wait sampling $(date -Is)" >> $LOG
while pgrep -f "outdir /root/sampling_under_mask" >/dev/null; do sleep 30; done
echo "[tail] sampling done, start tail $(date -Is)" >> $LOG
cd /root/reap-loop
export DS4_LOCK_FILE=/tmp/ds4_s2.lock DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_REAP_PREFETCH_THREADS=16 DS4_REAP_PREFETCH_LOCK=1
for W in 90 70 110 150; do
  OUT=/root/t4_tail/t4_W$(printf %03d $W)
  if [ -f "$OUT/summary.csv" ]; then echo "[tail] W=$W done, skip" >> $LOG; continue; fi
  echo "[tail] W=$W start $(date -Is)" >> $LOG
  python3 scripts/run_w_sweep_freeze_safe.py \
    --binary /root/ds4/ds4 --model /root/models/ds4-2bit.gguf \
    --cache 256 --ctx-p1 4096 --ctx-p2 4096 --total 1200 --keep-k 23 --timeout 3600 \
    --w-values $W --runs 3 --mask-mode weighted \
    --prompt-file /root/reap-loop/runs/ds4/20260710_pod_cache1024_warmup_replay/frontpage_prompt.txt \
    --outdir $OUT >> $LOG 2>&1
  echo "[tail] W=$W rc=$? $(date -Is)" >> $LOG
done
echo "[tail] ALL DONE $(date -Is)" >> $LOG
