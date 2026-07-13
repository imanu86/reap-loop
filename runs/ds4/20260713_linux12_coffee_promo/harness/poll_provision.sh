#!/usr/bin/env bash
# Poll loop: grab ONE 12GB Linux GPU on RunPod. Logs every attempt.
# Bulletproof pod detection via pod-list snapshot diff (parse-failure safe).
# On success: arms a detached 2.5h anti-orphan reaper + writes POD_ID.txt, then exits.
set -u

RUNDIR="/c/Users/imanu/source/repos/moe-aggressive-commit/runs/ds4/20260713_linux12_coffee_promo"
STATE="$RUNDIR/provision_state.log"
PODIDFILE="$RUNDIR/POD_ID.txt"
IMAGE="runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404"
export RUNPOD_API_KEY="$(tr -d '\r\n' < /c/Users/imanu/Desktop/Runapod.txt)"

# Protected pods we must NEVER touch (pre-existing)
PROTECTED="99xyqm02gke4xg 3upjyrmu6mdof4 0htxln87674tjq 7qgalm9sasqnr7 i7dk94f0y05iji u49vytysl0xyqi ysegg4bx67yvr3"

# 12GB cards in RunPod catalog, task-preference order (3080-class first, then 4070 Ti)
CARDS=("NVIDIA GeForce RTX 3080 Ti" "NVIDIA GeForce RTX 4070 Ti")

CEIL=0.9
DEADLINE=$(( $(date +%s) + 150*60 ))   # 2.5 hours
ATTEMPT=0

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATE"; }

# Snapshot of current pod IDs (space separated)
list_ids(){ runpodctl pod list 2>/dev/null | awk 'NR>1{print $1}' | tr '\n' ' '; }

log "=== POLL START ceiling=\$${CEIL}/hr deadline=2.5h image=$IMAGE ==="
log "candidates: ${CARDS[*]} (community then secure)"
BASELINE_IDS="$(list_ids)"
log "baseline pod ids: $BASELINE_IDS"

arm_and_exit(){
  local pid="$1" gpu="$2" cloud="$3"
  echo "$pid" > "$PODIDFILE"
  echo "$gpu" > "$RUNDIR/POD_GPU.txt"
  nohup bash -c "sleep 9000; export RUNPOD_API_KEY='$RUNPOD_API_KEY'; runpodctl remove pod $pid >> '$STATE' 2>&1; echo '[reaper] removed $pid after 2.5h' >> '$STATE'" >/dev/null 2>&1 &
  log "SUCCESS card='$gpu' cloud=$cloud POD_ID=$pid"
  log "anti-orphan reaper armed (removes $pid in 9000s if not sooner)"
  exit 0
}

try_create(){
  local gpu="$1" cloud="$2"
  local cloudflag; [ "$cloud" = community ] && cloudflag=--communityCloud || cloudflag=--secureCloud
  local name="ds4-l12-$(date -u +%H%M%S)"
  local before after out rc
  before="$(list_ids)"
  out=$(runpodctl create pod \
        --name "$name" \
        --gpuType "$gpu" \
        --gpuCount 1 \
        $cloudflag \
        --cost "$CEIL" \
        --imageName "$IMAGE" \
        --containerDiskSize 30 \
        --volumeSize 150 \
        --volumePath /workspace \
        --mem 40 --vcpu 8 \
        --ports '22/tcp' \
        --startSSH \
        --output json 2>&1)
  rc=$?
  # Source of truth: any NEW pod id that appeared and is not protected/baseline
  after="$(list_ids)"
  local newid=""
  for id in $after; do
    case " $before " in *" $id "*) continue;; esac
    case " $PROTECTED " in *" $id "*) continue;; esac
    newid="$id"; break
  done
  if [ -n "$newid" ]; then
    arm_and_exit "$newid" "$gpu" "$cloud"
  fi
  local reason; reason=$(echo "$out" | head -3 | tr '\n' ' ' | cut -c1-220)
  log "no-instance card='$gpu' cloud=$cloud rc=$rc :: $reason"
  return 1
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  ATTEMPT=$((ATTEMPT+1))
  log "--- attempt #$ATTEMPT ($(( (DEADLINE-$(date +%s))/60 ))min left) ---"
  for card in "${CARDS[@]}"; do try_create "$card" community; done
  for card in "${CARDS[@]}"; do try_create "$card" secure; done
  log "cycle done, sleeping 240s"
  sleep 240
done

log "=== DEADLINE reached (2.5h), no 12GB acquired. Zero pods created. ==="
exit 2
