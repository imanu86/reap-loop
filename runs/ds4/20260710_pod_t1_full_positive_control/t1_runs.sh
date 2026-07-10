#!/bin/bash
# T1 full positive control - run sequence (greedy + sampling arms)
set -u
cd /root/reap-loop
R=/root/reap-loop/runs/ds4/20260710_pod_t1_full_positive_control
P=/root/t1_progress.log
log(){ echo "[$(date -u +%H:%M:%S)] $1" >> $P; }
log START

log "step1 greedy 800 html+coffee n=3 abab"
python3 scripts/run_ds4_exchange_matrix.py --suite quick --prompts html,html_coffee   --variants no_pace --runs 3 --order abab --warmups 0 --max-tokens 800 --timeout 1800   --port 8014 --out-dir $R/greedy_800 --model /root/models/ds4-2bit.gguf   --ctx 3072 --server-max-tokens 1100 --cache-experts 1024 --prefill-chunk 128 --stream   >> $P 2>&1
log "step1 exit=$?"

log "step2 greedy 2000 html n=2"
python3 scripts/run_ds4_exchange_matrix.py --suite quick --prompts html   --variants no_pace --runs 2 --warmups 0 --max-tokens 2000 --timeout 2400   --port 8014 --out-dir $R/greedy_html2000 --model /root/models/ds4-2bit.gguf   --ctx 3072 --server-max-tokens 2300 --cache-experts 1024 --prefill-chunk 128 --stream   >> $P 2>&1
log "step2 exit=$?"

log "step3 greedy 2000 html_coffee n=1"
python3 scripts/run_ds4_exchange_matrix.py --suite quick --prompts html_coffee   --variants no_pace --runs 1 --warmups 0 --max-tokens 2000 --timeout 2400   --port 8014 --out-dir $R/greedy_coffee2000 --model /root/models/ds4-2bit.gguf   --ctx 3072 --server-max-tokens 2300 --cache-experts 1024 --prefill-chunk 128 --stream   >> $P 2>&1
log "step3 exit=$?"

log "step4 sampled 2000 html n=2 (temp0.7 top_p0.95 seed42)"
python3 scripts/run_t1_sampling.py --prompt html --runs 2 --max-tokens 2000   --temperature 0.7 --top-p 0.95 --seed 42 --timeout 2400 --port 8014   --out-dir $R/sampled_html2000 --model /root/models/ds4-2bit.gguf   --ctx 3072 --server-max-tokens 2300 --cache-experts 1024 --prefill-chunk 128   >> $P 2>&1
log "step4 exit=$?"

log "step5 sampled 800 html n=1"
python3 scripts/run_t1_sampling.py --prompt html --runs 1 --max-tokens 800   --temperature 0.7 --top-p 0.95 --seed 42 --timeout 1800 --port 8014   --out-dir $R/sampled_html800 --model /root/models/ds4-2bit.gguf   --ctx 3072 --server-max-tokens 1100 --cache-experts 1024 --prefill-chunk 128   >> $P 2>&1
log "step5 exit=$?"

log "step6 greedy 4000 html n=1"
python3 scripts/run_ds4_exchange_matrix.py --suite quick --prompts html   --variants no_pace --runs 1 --warmups 0 --max-tokens 4000 --timeout 3600   --port 8014 --out-dir $R/greedy_html4000 --model /root/models/ds4-2bit.gguf   --ctx 6144 --server-max-tokens 4400 --cache-experts 1024 --prefill-chunk 128 --stream   >> $P 2>&1
log "step6 exit=$?"
log ALL_DONE
