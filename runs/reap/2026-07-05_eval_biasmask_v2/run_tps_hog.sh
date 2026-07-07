#!/bin/bash
# Track REAP-ds4 — t/s bench 3060-proxy con RAM cappata via HOG PROCESS (no privilegi).
# mlock e cgroup-write sono bloccati nel container RunPod; un processo che occupa
# ~HOGGB di RAM anonima residente (niente swap -> non evictabile) lascia (CG_PARENT-HOG)
# GB per ds4 + page-cache del modello mmap'd -> forza streaming SSD come una workstation
# con quel budget. DELTA relativi (ON/OFF, 60/28) trasferiscono al 3060; ASSOLUTI no.
set -u
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
OUT=/root/tps
mkdir -p $OUT
HOST_GB=$(free -g | awk 'NR==2{print $2}')
CG_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
[ "$CG_MAX" = "max" ] && CG_GB=$HOST_GB || CG_GB=$(( CG_MAX / 1073741824 ))
SWAP_GB=$(free -g | awk 'NR==3{print $2}')
echo "HOST_GB=$HOST_GB CG_GB=$CG_GB SWAP_GB=$SWAP_GB" | tee $OUT/results_tps.csv
echo "scenario,avail_gb,mask,rep,rc,prefill_tps,gen_tps,hog_gb,spex" >> $OUT/results_tps.csv
COMMON="--cuda --ssd-streaming --ssd-streaming-cache-experts 156 -c 4096 --nothink --temp 0"

# Target: avail = CG_GB - hog. Per avail 60 -> hog 0 (CG e' gia' ~57). Per avail 28 -> hog = CG-28.
HOG_PID=""
start_hog() { # avail_gb
  local avail=$1
  local hog=$(( CG_GB - avail ))
  [ $hog -lt 1 ] && { echo 0; return; }
  rm -f /root/hog_ready
  python3 - "$hog" > $OUT/hog.log 2>&1 <<'PY' &
import sys, time, mmap
gb = int(sys.argv[1]); n = gb * (1024**3)
buf = mmap.mmap(-1, n)
for i in range(0, n, 4096):
    buf[i] = 1
open("/root/hog_ready", "w").close()
time.sleep(100000)
PY
  HOG_PID=$!
  local w=0
  while [ ! -f /root/hog_ready ]; do sleep 1; w=$((w+1)); [ $w -gt 120 ] && break; done
  echo $hog
}
stop_hog() { [ -n "$HOG_PID" ] && kill $HOG_PID 2>/dev/null; HOG_PID=""; rm -f /root/hog_ready; sleep 2; }

run_one() { # scenario avail mask rep hoggb
  local scen=$1 avail=$2 mask=$3 rep=$4 hog=$5
  local log=$OUT/gen_${scen}_${mask}_rep${rep}.log
  DS4_SPEX_STATS=1 $BIN -m $MODEL $COMMON -n 200 --prompt-file /root/v0_prompt.txt > "$log" 2>&1
  local rc=$?
  local pf gn st
  pf=$(grep -o "prefill: [0-9.]*" "$log" | tail -1 | cut -d' ' -f2)
  gn=$(grep -o "generation: [0-9.]*" "$log" | tail -1 | cut -d' ' -f2)
  st=$(grep -o "SPEX stats: .*" "$log" | tail -1 | tr ',' ';')
  grep -qi "killed\|out of memory\|oom\|bad_alloc" "$log" && st="OOM?:$st"
  echo "$scen,$avail,$mask,$rep,rc=$rc,pf=$pf,gen=$gn,${hog},\"$st\"" | tee -a $OUT/results_tps.csv
}

for MASKCFG in off on; do
  if [ "$MASKCFG" = "on" ]; then
    echo "=== APPLY reap_k50 $(date -u +%FT%TZ)"
    python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --maskfile /root/reap_mask_ds4_domain.json --apply reap 2>&1 | tee -a $OUT/biasmask.log
  fi
  for AVAIL in 60 28; do
    HOG=$(start_hog $AVAIL)
    echo "=== scenario avail${AVAIL}GB (hog ${HOG}GB) mask=$MASKCFG $(date -u +%FT%TZ)"
    DS4_SPEX_STATS=1 $BIN -m $MODEL $COMMON -n 32 --prompt-file /root/v0_prompt.txt > $OUT/warm_${AVAIL}_${MASKCFG}.log 2>&1
    run_one "avail${AVAIL}" "$AVAIL" "$MASKCFG" 1 "$HOG"
    run_one "avail${AVAIL}" "$AVAIL" "$MASKCFG" 2 "$HOG"
    stop_hog
  done
done
python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --restore 2>&1 | tee -a $OUT/biasmask.log
echo "ALL_DONE_TPS $(date -u +%FT%TZ)"
