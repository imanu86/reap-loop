#!/bin/bash
# Track REAP-ds4 — t/s bench 3060-proxy: mask ON/OFF x RAM disponibile {60,28} GB.
# RAM cappata con --simulate-used-memory (mmap+touch+mlock IN-PROCESS: niente stato di sistema).
# VRAM expert cache cappata a 156 expert (= cap empirico del 3060 12GB, log stage0).
# I DELTA relativi (ON/OFF, 60/28) trasferiscono al 3060; gli ASSOLUTI NO (dichiarato).
set -u
BIN=/root/ds4/ds4
MODEL=/root/models/ds4-2bit.gguf
OUT=/root/tps
mkdir -p $OUT
HOST_GB=$(free -g | awk 'NR==2{print $2}')
CGMAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
if [ "$CGMAX" != "max" ] && [ -n "$CGMAX" ]; then CG_GB=$(( CGMAX / 1073741824 )); else CG_GB=$HOST_GB; fi
TOTAL_GB=$(( CG_GB < HOST_GB ? CG_GB : HOST_GB ))
echo "HOST_GB=$HOST_GB CGROUP_GB=$CG_GB TOTAL_GB=$TOTAL_GB" | tee $OUT/results_tps.csv
echo "scenario,mask,rep,prefill_tps,gen_tps,spex_stats" >> $OUT/results_tps.csv
COMMON="--cuda --ssd-streaming --ssd-streaming-cache-experts 156 -c 4096 --nothink --temp 0"

sim_for_avail() { # avail_gb -> GB da bloccare (0 = nessun flag)
  local avail=$1
  local sim=$(( TOTAL_GB - avail ))
  [ $sim -lt 1 ] && sim=0
  echo $sim
}

run_one() { # scenario mask rep simgb
  local scen=$1 mask=$2 rep=$3 simgb=$4
  local simflag=""
  [ "$simgb" -gt 0 ] && simflag="--simulate-used-memory ${simgb}GB"
  local log=$OUT/gen_${scen}_${mask}_rep${rep}.log
  DS4_SPEX_STATS=1 $BIN -m $MODEL $COMMON $simflag -n 200 --prompt-file /root/v0_prompt.txt > "$log" 2>&1
  local rc=$?
  local tps
  tps=$(grep -o "prefill: [0-9.]* t/s, generation: [0-9.]* t/s" "$log" | tail -1)
  local pf=$(echo "$tps" | grep -o "prefill: [0-9.]*" | cut -d' ' -f2)
  local gn=$(echo "$tps" | grep -o "generation: [0-9.]*" | cut -d' ' -f2)
  local st=$(grep -o "SPEX stats: .*" "$log" | tail -1 | tr ',' ';')
  echo "$scen,$mask,$rep,rc=$rc,pf=$pf,gen=$gn,\"$st\"" | tee -a $OUT/results_tps.csv
}

# mlock gate-test: 4GB
echo "=== mlock gate-test 4GB $(date -u +%FT%TZ)"
$BIN -m $MODEL $COMMON --simulate-used-memory 4GB -n 4 -p "test" > $OUT/mlock_test.log 2>&1
if grep -qi "simulate-used-memory.*failed\|mlock" $OUT/mlock_test.log; then
  echo "MLOCK_TEST_FAIL — vedi mlock_test.log"; grep -i "memory\|mlock" $OUT/mlock_test.log | head -3
else
  echo "MLOCK_TEST_OK"
fi

for MASKCFG in off on; do
  if [ "$MASKCFG" = "on" ]; then
    echo "=== APPLY reap_k50 $(date -u +%FT%TZ)"
    python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --maskfile /root/reap_mask_ds4_domain.json --apply reap 2>&1 | tee -a $OUT/biasmask.log
  fi
  for AVAIL in 60 28; do
    SIM=$(sim_for_avail $AVAIL)
    echo "=== scenario avail${AVAIL}GB (sim ${SIM}GB) mask=$MASKCFG $(date -u +%FT%TZ)"
    # warm-up scartata (paga anche l'eviction della page cache al nuovo budget)
    DS4_SPEX_STATS=1 $BIN -m $MODEL $COMMON $( [ $SIM -gt 0 ] && echo --simulate-used-memory ${SIM}GB ) -n 32 --prompt-file /root/v0_prompt.txt > $OUT/warm_${AVAIL}_${MASKCFG}.log 2>&1
    run_one "avail${AVAIL}" "$MASKCFG" 1 "$SIM"
    run_one "avail${AVAIL}" "$MASKCFG" 2 "$SIM"
  done
done
python3 /root/reap_bias_mask_ds4.py --gguf $MODEL --restore 2>&1 | tee -a $OUT/biasmask.log
echo "ALL_DONE_TPS $(date -u +%FT%TZ)"
