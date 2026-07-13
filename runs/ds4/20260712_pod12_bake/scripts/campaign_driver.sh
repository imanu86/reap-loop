#!/bin/bash
# Pod campaign driver — virtual bake quality, n=3 protocol (run1 temp0, run2-3 temp0.7)
# Fail-fast: if run1 of an arm degenerates (repeat-stop / tag-salad), skip run2/3.
# Arms in order: self60 (x3) -> self65 (x3, only if 60 holds) -> family60 (x1) -> K0 (x1)
set -uo pipefail
cd /root/reap-loop/runs/ds4/20260712_pod12_bake
S=/root/reap-loop/runs/ds4/20260712_pod12_bake/scripts
M=/root/reap-loop/runs/ds4/20260712_virtual_bake/masks
LOG=/root/reap-loop/runs/ds4/20260712_pod12_bake/CAMPAIGN_LOG.txt
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

degenerated(){ # $1 = run dir -> 0 (true) if degenerate stop
  local sr; sr=$(cat "$1/STOP_REASON.txt" 2>/dev/null || echo missing)
  case "$sr" in
    *repeat*|*degen*) return 0;;
    *) return 1;;
  esac
}

run_one(){ # arm mask run_n temp
  local arm=$1 mask=$2 n=$3 temp=$4
  log "START arm=$arm run=$n temp=$temp"
  bash "$S/run_bake_arm_pod.sh" "$arm" "$mask" "$n" 4000 4096 "$temp" >> "$LOG" 2>&1
  local rc=$?
  log "END arm=$arm run=$n rc=$rc stop=$(cat arm_${arm}_run${n}/STOP_REASON.txt 2>/dev/null) grade=$(head -1 arm_${arm}_run${n}/grade.txt 2>/dev/null)"
  return $rc
}

# --- Arm 1: bake60_self ---
run_one self60 "$M/mask60_self.txt" 1 0
if degenerated arm_self60_run1; then
  log "FAIL-FAST: self60 run1 degenerated -> skip run2/3 and self65 arm"
  SELF60_OK=0
else
  SELF60_OK=1
  run_one self60 "$M/mask60_self.txt" 2 0.7
  run_one self60 "$M/mask60_self.txt" 3 0.7
fi

# --- Arm 2: bake65_self (only if 60 held) ---
if [[ "$SELF60_OK" == 1 ]]; then
  run_one self65 "$M/mask65_self.txt" 1 0
  if degenerated arm_self65_run1; then
    log "FAIL-FAST: self65 run1 degenerated -> skip run2/3"
  else
    run_one self65 "$M/mask65_self.txt" 2 0.7
    run_one self65 "$M/mask65_self.txt" 3 0.7
  fi
else
  log "SKIP self65 (self60 failed)"
fi

# --- Arm 3: family control (x1) ---
run_one family60 "$M/mask60_family.txt" 1 0

# --- Arm 4: K0 no-mask baseline (x1) ---
run_one k0 NONE 1 0

log "CAMPAIGN COMPLETE"
