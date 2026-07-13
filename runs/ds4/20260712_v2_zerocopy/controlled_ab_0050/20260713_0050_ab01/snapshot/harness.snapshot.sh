#!/usr/bin/env bash
# Controlled, non-destructive patch-0050 OFF/ON campaign at 24/28 GiB.
# Nothing runs without the explicit --execute mode and a new campaign id.
set -Eeuo pipefail

umask 077

usage() {
  cat <<'EOF'
Usage:
  run_0050_controlled_ab.sh --preflight
  run_0050_controlled_ab.sh --execute --campaign-id ID [--rounds EVEN_N]

--preflight checks prerequisites, the GPU lock, and active servers. It never
starts ds4-server. --execute is the only mode that starts the campaign.

The default is four measured rounds. Rounds must be even so arm and budget
order remain counterbalanced.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

MODE=""
CAMPAIGN_ID=""
ROUNDS=4

while (($#)); do
  case "$1" in
    --preflight)
      [[ -z "$MODE" ]] || die "choose exactly one mode"
      MODE=preflight
      shift
      ;;
    --execute)
      [[ -z "$MODE" ]] || die "choose exactly one mode"
      MODE=execute
      shift
      ;;
    --campaign-id)
      (($# >= 2)) || die "--campaign-id requires a value"
      CAMPAIGN_ID=$2
      shift 2
      ;;
    --rounds)
      (($# >= 2)) || die "--rounds requires a value"
      ROUNDS=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$MODE" ]] || {
  usage >&2
  exit 2
}
[[ "$ROUNDS" =~ ^[0-9]+$ ]] || die "--rounds must be an integer"
((ROUNDS >= 2 && ROUNDS <= 20 && ROUNDS % 2 == 0)) ||
  die "--rounds must be even and between 2 and 20"

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)
RUN_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
REPO=$(cd "$RUN_ROOT/../../.." && pwd -P)
OUT_ROOT="$RUN_ROOT/controlled_ab_0050"

SRC=/root/ds4-v2-work
BIN="$SRC/ds4-server"
MODEL=/root/models/ds4-2bit.gguf
MASK="$REPO/runs/ds4/20260712_virtual_bake/masks/mask60_self.txt"
MEASURE_SCRIPT="$RUN_ROOT/scripts/measure_stream.py"
PROTOCOL="$RUN_ROOT/AB_PROTOCOL.md"
GPU_LOCK=/tmp/ds4-gpu.lock
PORT=8096
CTX=2048
SERVER_MAX_TOKENS=706
CACHE_EXPERTS=400
PREFILL_CHUNK=512
WSL_RAM_FLOOR_MB=8192
WINDOWS_RAM_FLOOR_GIB=8
TELEMETRY_INTERVAL_SECONDS=2
COOLDOWN_SECONDS=20

CAMPAIGN_DIR=""
CURRENT_RUN_DIR=""
SERVER_PID=""
SERVER_START_TICKS=""
SERVER_EXE=""
WSL_MON_PID=""
WSL_MON_START_TICKS=""
WINDOWS_MON_PID=""
WINDOWS_MON_START_TICKS=""
WINDOWS_MONITOR_AVAILABLE=0
CAMPAIGN_COMPLETE=0
LOCK_HELD=0

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  local line
  line="$(utc_now) $*"
  printf '%s\n' "$line"
  if [[ -n "$CAMPAIGN_DIR" && -d "$CAMPAIGN_DIR" ]]; then
    printf '%s\n' "$line" >> "$CAMPAIGN_DIR/campaign.log"
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

proc_start_ticks() {
  local pid=$1
  [[ -r "/proc/$pid/stat" ]] || return 1
  sed -E 's/^[0-9]+ \([^)]*\) //' "/proc/$pid/stat" | awk '{print $20}'
}

proc_state() {
  local pid=$1
  [[ -r "/proc/$pid/stat" ]] || return 1
  sed -E 's/^[0-9]+ \([^)]*\) //' "/proc/$pid/stat" | awk '{print $1}'
}

proc_matches() {
  local pid=$1 expected_start=$2 expected_exe=$3
  local actual_start actual_exe
  [[ "$pid" =~ ^[0-9]+$ && -n "$expected_start" && -n "$expected_exe" ]] || return 1
  [[ -r "/proc/$pid/stat" ]] || return 1
  actual_start=$(proc_start_ticks "$pid") || return 1
  [[ "$actual_start" == "$expected_start" ]] || return 1
  actual_exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || return 1
  [[ "$actual_exe" == "$expected_exe" ]]
}

child_proc_matches() {
  local pid=$1 expected_start=$2
  local actual_start actual_state
  [[ "$pid" =~ ^[0-9]+$ && -n "$expected_start" ]] || return 1
  actual_start=$(proc_start_ticks "$pid" 2>/dev/null) || return 1
  [[ "$actual_start" == "$expected_start" ]] || return 1
  actual_state=$(proc_state "$pid" 2>/dev/null) || return 1
  [[ "$actual_state" != Z ]]
}

windows_server_pids() {
  command -v powershell.exe >/dev/null 2>&1 || return 0
  powershell.exe -NoLogo -NoProfile -NonInteractive -Command \
    '$p=@(Get-Process -Name "ds4-server" -ErrorAction SilentlyContinue); if($p.Count){$p.Id -join ","}' \
    2>/dev/null | tr -d '\r' || true
}

assert_no_server() {
  local pids native
  pids=$(pgrep -x ds4-server 2>/dev/null || true)
  [[ -z "$pids" ]] || die "ds4-server already active in WSL (PID(s): ${pids//$'\n'/,})"

  native=$(windows_server_pids)
  [[ -z "$native" ]] || die "native Windows ds4-server already active (PID(s): $native)"

  if command -v ss >/dev/null 2>&1; then
    if ss -H -ltn | awk -v port=":$PORT" '$4 ~ (port "$") { found=1 } END { exit !found }'; then
      die "TCP port $PORT is already listening"
    fi
  fi
}

check_windows_telemetry() {
  WINDOWS_MONITOR_AVAILABLE=0
  if ! command -v powershell.exe >/dev/null 2>&1 || ! command -v wslpath >/dev/null 2>&1; then
    return 0
  fi
  if powershell.exe -NoLogo -NoProfile -NonInteractive -Command \
      '$ErrorActionPreference="Stop"; $o=Get-CimInstance -ClassName Win32_OperatingSystem; if($o.FreePhysicalMemory -le 0){exit 2}' \
      >/dev/null 2>&1; then
    WINDOWS_MONITOR_AVAILABLE=1
  else
    die "PowerShell is available but Windows memory telemetry failed"
  fi
}

preflight() {
  local path
  for path in "$SRC" "$BIN" "$MODEL" "$MASK" "$MEASURE_SCRIPT" "$PROTOCOL"; do
    [[ -e "$path" ]] || die "required path missing: $path"
  done
  [[ -e "$SRC/.git" ]] || die "DS4 worktree has no .git metadata: $SRC"

  local cmd
  for cmd in awk cat cmp cp curl date env find flock git head mkdir nvidia-smi \
             pgrep python3 readlink sed sha256sum sleep sort stat tr uname wc xargs; do
    require_command "$cmd"
  done

  nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi cannot see a GPU"
  assert_no_server
  if ! flock -n "$GPU_LOCK" -c ':'; then
    die "GPU lock is active: $GPU_LOCK"
  fi
  check_windows_telemetry

  printf 'preflight=PASS\n'
  printf 'repo=%s\n' "$REPO"
  printf 'source=%s\n' "$SRC"
  printf 'binary=%s\n' "$BIN"
  printf 'model=%s\n' "$MODEL"
  printf 'mask=%s\n' "$MASK"
  printf 'gpu_lock=%s (free)\n' "$GPU_LOCK"
  printf 'windows_telemetry=%s\n' "$WINDOWS_MONITOR_AVAILABLE"
  printf 'execute_required=yes\n'
}

stop_owned_server() {
  local stop_mode=not_running i rc
  if proc_matches "${SERVER_PID:-}" "${SERVER_START_TICKS:-}" "${SERVER_EXE:-}"; then
    stop_mode=term
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for ((i=0; i<300; i++)); do
      proc_matches "$SERVER_PID" "$SERVER_START_TICKS" "$SERVER_EXE" || break
      sleep 0.1
    done
    if proc_matches "$SERVER_PID" "$SERVER_START_TICKS" "$SERVER_EXE"; then
      stop_mode=kill_after_term_timeout
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    set +e
    wait "$SERVER_PID" 2>/dev/null
    rc=$?
    set -e
    if [[ -n "$CURRENT_RUN_DIR" && -d "$CURRENT_RUN_DIR" ]]; then
      printf '%s\n' "$rc" > "$CURRENT_RUN_DIR/server.wait_rc"
      printf '%s\n' "$stop_mode" > "$CURRENT_RUN_DIR/server.stop_mode"
    fi
  fi
  SERVER_PID=""
  SERVER_START_TICKS=""
  SERVER_EXE=""
}

stop_telemetry() {
  local i rc
  if [[ -n "$CURRENT_RUN_DIR" && -d "$CURRENT_RUN_DIR" ]]; then
    : > "$CURRENT_RUN_DIR/telemetry.stop"
  fi

  if child_proc_matches "${WSL_MON_PID:-}" "${WSL_MON_START_TICKS:-}"; then
    for ((i=0; i<50; i++)); do
      child_proc_matches "$WSL_MON_PID" "$WSL_MON_START_TICKS" || break
      sleep 0.1
    done
    if child_proc_matches "$WSL_MON_PID" "$WSL_MON_START_TICKS"; then
      kill -TERM "$WSL_MON_PID" 2>/dev/null || true
    fi
  fi
  if [[ -n "${WSL_MON_PID:-}" ]]; then
    set +e
    wait "$WSL_MON_PID" 2>/dev/null
    rc=$?
    set -e
    [[ -z "$CURRENT_RUN_DIR" || ! -d "$CURRENT_RUN_DIR" ]] ||
      printf '%s\n' "$rc" > "$CURRENT_RUN_DIR/wsl_telemetry.wait_rc"
  fi
  WSL_MON_PID=""
  WSL_MON_START_TICKS=""

  if child_proc_matches "${WINDOWS_MON_PID:-}" "${WINDOWS_MON_START_TICKS:-}"; then
    for ((i=0; i<100; i++)); do
      child_proc_matches "$WINDOWS_MON_PID" "$WINDOWS_MON_START_TICKS" || break
      sleep 0.1
    done
    if child_proc_matches "$WINDOWS_MON_PID" "$WINDOWS_MON_START_TICKS"; then
      kill -TERM "$WINDOWS_MON_PID" 2>/dev/null || true
    fi
  fi
  if [[ -n "${WINDOWS_MON_PID:-}" ]]; then
    set +e
    wait "$WINDOWS_MON_PID" 2>/dev/null
    rc=$?
    set -e
    [[ -z "$CURRENT_RUN_DIR" || ! -d "$CURRENT_RUN_DIR" ]] ||
      printf '%s\n' "$rc" > "$CURRENT_RUN_DIR/windows_telemetry.wait_rc"
  fi
  WINDOWS_MON_PID=""
  WINDOWS_MON_START_TICKS=""
}

on_signal() {
  log "received signal; stopping only recorded child PIDs"
  exit 130
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  stop_owned_server
  stop_telemetry
  if [[ -n "$CAMPAIGN_DIR" && -d "$CAMPAIGN_DIR" ]]; then
    if ((CAMPAIGN_COMPLETE)); then
      printf '%s\tCOMPLETE\t0\n' "$(utc_now)" >> "$CAMPAIGN_DIR/campaign_status.tsv"
    else
      printf '%s\tINCOMPLETE\t%s\n' "$(utc_now)" "$rc" >> "$CAMPAIGN_DIR/campaign_status.tsv"
    fi
  fi
  if ((LOCK_HELD)); then
    flock -u 9 2>/dev/null || true
  fi
  exit "$rc"
}

capture_snapshot() {
  local snap="$CAMPAIGN_DIR/snapshot"
  mkdir "$snap" "$snap/ds4_sources"

  cp -- "$SCRIPT_PATH" "$snap/harness.snapshot.sh"
  cp -- "$PROTOCOL" "$snap/AB_PROTOCOL.snapshot.md"
  env -0 > "$snap/runner_env.nul"
  env | LC_ALL=C sort > "$snap/runner_env.txt"
  uname -a > "$snap/uname.txt"
  cat /proc/meminfo > "$snap/meminfo.before.txt"
  nvidia-smi -q > "$snap/nvidia_smi_q.before.txt"
  nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits > "$snap/gpu.before.csv"

  git -C "$REPO" rev-parse HEAD > "$snap/reap_loop.commit"
  git -C "$REPO" status --porcelain=v2 --branch > "$snap/reap_loop.status"
  git -C "$REPO" diff --no-ext-diff --binary HEAD > "$snap/reap_loop.worktree.patch"

  git -C "$SRC" rev-parse HEAD > "$snap/ds4_base.commit"
  git -C "$SRC" status --porcelain=v2 --branch > "$snap/ds4.status"
  git -C "$SRC" log --decorate --oneline -n 100 > "$snap/ds4.log.txt"
  git -C "$SRC" diff --no-ext-diff --binary HEAD > "$snap/ds4_patch_chain.worktree.patch"
  git -C "$SRC" diff --no-ext-diff --cached --binary HEAD > "$snap/ds4_patch_chain.index.patch"
  git -C "$SRC" diff --stat HEAD > "$snap/ds4_patch_chain.stat.txt"
  git -C "$SRC" ls-files --others --exclude-standard > "$snap/ds4.untracked.txt"

  local source_file
  for source_file in ds4.c ds4_cuda.cu ds4_gpu.h ds4.h ds4_ssd.c Makefile; do
    if [[ -f "$SRC/$source_file" ]]; then
      cp -- "$SRC/$source_file" "$snap/ds4_sources/$source_file"
    fi
  done
  find "$snap/ds4_sources" -maxdepth 1 -type f -print0 |
    sort -z | xargs -0 -r sha256sum > "$snap/ds4_sources.sha256"
  sha256sum "$snap/ds4_patch_chain.worktree.patch" \
    "$snap/ds4_patch_chain.index.patch" \
    "$snap/ds4_patch_chain.stat.txt" > "$snap/ds4_patch_chain.sha256"

  find "$REPO/patches/ds4" -maxdepth 1 -type f -name '*.patch' -print0 |
    sort -z | xargs -0 -r sha256sum > "$snap/reap_loop_patch_archive.sha256"
  find "$SRC" -maxdepth 1 -type f -name 'build_0050*.log' -print0 |
    sort -z | xargs -0 -r sha256sum > "$snap/build_0050_logs.sha256"
  [[ ! -f "$SRC/build_0050i.log" ]] || cp -- "$SRC/build_0050i.log" "$snap/build_0050i.log"

  sha256sum "$BIN" "$SRC/ds4.c" "$SRC/ds4_cuda.cu" "$SRC/ds4_gpu.h" \
    "$MASK" "$MEASURE_SCRIPT" "$snap/harness.snapshot.sh" \
    > "$snap/fixed_files.sha256"
  stat -Lc '%d:%i:%s:%Y:%y %n' "$MODEL" > "$snap/model.stat"
  sha256sum "$MODEL" > "$snap/model.sha256"
  sha256sum "$MASK" > "$snap/mask.sha256"

  {
    printf 'campaign_id=%s\n' "$CAMPAIGN_ID"
    printf 'rounds=%s\n' "$ROUNDS"
    printf 'port=%s\n' "$PORT"
    printf 'ctx=%s\n' "$CTX"
    printf 'server_max_tokens=%s\n' "$SERVER_MAX_TOKENS"
    printf 'cache_experts=%s\n' "$CACHE_EXPERTS"
    printf 'prefill_chunk=%s\n' "$PREFILL_CHUNK"
    printf 'wsl_ram_floor_mb=%s\n' "$WSL_RAM_FLOOR_MB"
    printf 'windows_ram_floor_gib=%s\n' "$WINDOWS_RAM_FLOOR_GIB"
    printf 'telemetry_interval_seconds=%s\n' "$TELEMETRY_INTERVAL_SECONDS"
    printf 'cooldown_seconds=%s\n' "$COOLDOWN_SECONDS"
    printf 'windows_monitor_available=%s\n' "$WINDOWS_MONITOR_AVAILABLE"
  } > "$snap/campaign_config.txt"
}

verify_fixed_state() {
  local snap="$CAMPAIGN_DIR/snapshot"
  local now_stat expected_stat current_hash expected_hash
  sha256sum --quiet --check "$snap/fixed_files.sha256" ||
    die "binary/source/mask/measurement helper changed during campaign"
  [[ ! -f "$CAMPAIGN_DIR/requests.sha256" ]] ||
    sha256sum --quiet --check "$CAMPAIGN_DIR/requests.sha256" ||
    die "a generated request changed during campaign"
  [[ ! -f "$CAMPAIGN_DIR/windows_telemetry.sha256" ]] ||
    sha256sum --quiet --check "$CAMPAIGN_DIR/windows_telemetry.sha256" ||
    die "the Windows telemetry helper changed during campaign"
  [[ "$(git -C "$SRC" rev-parse HEAD)" == "$(cat "$snap/ds4_base.commit")" ]] ||
    die "DS4 base commit changed during campaign"
  current_hash=$(git -C "$SRC" diff --no-ext-diff --binary HEAD | sha256sum | awk '{print $1}')
  expected_hash=$(awk 'NR==1{print $1}' "$snap/ds4_patch_chain.sha256")
  [[ "$current_hash" == "$expected_hash" ]] || die "DS4 worktree patch chain changed during campaign"
  current_hash=$(git -C "$SRC" diff --no-ext-diff --cached --binary HEAD | sha256sum | awk '{print $1}')
  expected_hash=$(awk 'NR==2{print $1}' "$snap/ds4_patch_chain.sha256")
  [[ "$current_hash" == "$expected_hash" ]] || die "DS4 index patch chain changed during campaign"
  now_stat=$(stat -Lc '%d:%i:%s:%Y:%y %n' "$MODEL")
  expected_stat=$(cat "$snap/model.stat")
  [[ "$now_stat" == "$expected_stat" ]] || die "model stat changed during campaign"
}

generate_requests() {
  python3 - "$CAMPAIGN_DIR" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
system = "Rispondi in modo diretto, utile e senza ragionamento visibile."
prompt = """Write a COMPLETE and COMPACT single-file HTML page for a coffee shop. Output ONLY the HTML, nothing else. Keep the CSS SHORT (about 10-15 rules max) - prioritize a COMPLETE, working page over elaborate styling. The page MUST be fully closed with </html> and MUST contain all of these:
1. A <nav> with three links: Home, Menu, Contact.
2. A hero <section> with <h1>Bean & Brew</h1> and a one-line subheading.
3. A <button id=\"order\">Order Now</button> wired in <script> with addEventListener that shows alert(\"Thank you for your order!\").
4. A <form action=\"/submit\"> with a name text input, an email input, a submit button, and an onsubmit handler that calls preventDefault and shows a confirmation.
5. Minimal embedded CSS in <style> and the JS in <script>.
Write the entire compact HTML document now and finish it."""

def request(max_tokens, stream):
    obj = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "think": False,
        "thinking": {"type": "disabled"},
    }
    if stream:
        obj["stream_options"] = {"include_usage": True}
    return obj

for name, max_tokens, stream in (
    ("request_gate_temp0_60.json", 60, False),
    ("request_warmup_temp0_80.json", 80, False),
    ("request_measured_temp0_450.json", 450, True),
):
    (out / name).write_text(
        json.dumps(request(max_tokens, stream), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY
  sha256sum "$CAMPAIGN_DIR"/request_*.json > "$CAMPAIGN_DIR/requests.sha256"
}

treatment_arm() {
  case "$1" in
    off24|off28) printf 'off\n' ;;
    on24|on28) printf 'on\n' ;;
    *) die "invalid treatment: $1" ;;
  esac
}

treatment_budget() {
  case "$1" in
    off24|on24) printf '24\n' ;;
    off28|on28) printf '28\n' ;;
    *) die "invalid treatment: $1" ;;
  esac
}

write_plan() {
  local phase round position treatment arm budget
  local position_global=0
  printf 'phase\tround\tposition_in_round\tposition_global\ttreatment\tarm\tbudget_gib\tmeasured\n' \
    > "$CAMPAIGN_DIR/planned_order.tsv"

  position=0
  for treatment in off24 on24 off28 on28; do
    ((position+=1, position_global+=1))
    arm=$(treatment_arm "$treatment")
    budget=$(treatment_budget "$treatment")
    printf 'gate\t0\t%s\t%s\t%s\t%s\t%s\t0\n' \
      "$position" "$position_global" "$treatment" "$arm" "$budget" \
      >> "$CAMPAIGN_DIR/planned_order.tsv"
  done

  position=0
  for treatment in on28 off28 on24 off24; do
    ((position+=1, position_global+=1))
    arm=$(treatment_arm "$treatment")
    budget=$(treatment_budget "$treatment")
    printf 'warmup\t0\t%s\t%s\t%s\t%s\t%s\t0\n' \
      "$position" "$position_global" "$treatment" "$arm" "$budget" \
      >> "$CAMPAIGN_DIR/planned_order.tsv"
  done

  for ((round=1; round<=ROUNDS; round++)); do
    position=0
    if ((round % 2)); then
      treatments=(off24 on24 off28 on28)
    else
      treatments=(on28 off28 on24 off24)
    fi
    for treatment in "${treatments[@]}"; do
      ((position+=1, position_global+=1))
      arm=$(treatment_arm "$treatment")
      budget=$(treatment_budget "$treatment")
      printf 'measured\t%s\t%s\t%s\t%s\t%s\t%s\t1\n' \
        "$round" "$position" "$position_global" "$treatment" "$arm" "$budget" \
        >> "$CAMPAIGN_DIR/planned_order.tsv"
    done
  done
}

write_windows_monitor() {
  cat > "$CAMPAIGN_DIR/windows_telemetry.ps1" <<'POWERSHELL'
param(
    [Parameter(Mandatory=$true)][string]$CsvPath,
    [Parameter(Mandatory=$true)][string]$StopPath,
    [Parameter(Mandatory=$true)][string]$AbortPath,
    [Parameter(Mandatory=$true)][string]$PidPath,
    [Parameter(Mandatory=$true)][double]$FloorGiB,
    [Parameter(Mandatory=$true)][int]$PollMilliseconds
)
$ErrorActionPreference = 'Stop'
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$ascii = [System.Text.Encoding]::ASCII
[System.IO.File]::WriteAllText($PidPath, $PID.ToString($inv) + "`r`n", $ascii)
[System.IO.File]::WriteAllText(
    $CsvPath,
    "timestamp_utc,windows_available_gib,gpu_used_mib,gpu_total_mib,gpu_util_percent,gpu_temperature_c,gpu_status,action`r`n",
    $ascii)
$nvidia = Get-Command 'nvidia-smi.exe' -CommandType Application -ErrorAction SilentlyContinue
while (-not (Test-Path -LiteralPath $StopPath)) {
    $action = ''
    $gpuStatus = if ($null -eq $nvidia) { 'unavailable' } else { 'ok' }
    $available = $null
    $gpuUsed = ''
    $gpuTotal = ''
    $gpuUtil = ''
    $gpuTemp = ''
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem
        $available = ([double]$os.FreePhysicalMemory * 1024.0) / [double](1GB)
        if ($available -lt $FloorGiB) {
            $action = 'windows_memory_floor_breach'
        }
    } catch {
        $action = 'windows_memory_monitor_failed'
    }
    if ($null -ne $nvidia) {
        try {
            $line = & $nvidia.Path `
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu' `
                '--format=csv,noheader,nounits' 2>$null | Select-Object -First 1
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($line)) {
                $parts = $line.Split(',') | ForEach-Object { $_.Trim() }
                if ($parts.Count -ge 4) {
                    $gpuUsed, $gpuTotal, $gpuUtil, $gpuTemp = $parts[0..3]
                } else {
                    $gpuStatus = 'parse_failed'
                }
            } else {
                $gpuStatus = 'query_failed'
            }
        } catch {
            $gpuStatus = 'query_failed'
        }
    }
    $timestamp = [DateTime]::UtcNow.ToString('o', $inv)
    $availableField = if ($null -eq $available) { '' } else { $available.ToString('F6', $inv) }
    $line = [string]::Format(
        $inv, '{0},{1},{2},{3},{4},{5},{6},{7}',
        $timestamp, $availableField, $gpuUsed, $gpuTotal, $gpuUtil, $gpuTemp,
        $gpuStatus, $action)
    [System.IO.File]::AppendAllText($CsvPath, $line + "`r`n", $ascii)
    if ($action) {
        [System.IO.File]::WriteAllText($AbortPath, $action + "`r`n", $ascii)
        break
    }
    Start-Sleep -Milliseconds $PollMilliseconds
}
POWERSHELL
  sha256sum "$CAMPAIGN_DIR/windows_telemetry.ps1" > "$CAMPAIGN_DIR/windows_telemetry.sha256"
}

start_windows_telemetry() {
  local run_dir=$1 ps_win csv_win stop_win abort_win pid_win
  [[ "$WINDOWS_MONITOR_AVAILABLE" == 1 ]] || {
    printf 'unavailable\n' > "$run_dir/windows_telemetry.status"
    return 0
  }
  ps_win=$(wslpath -w "$CAMPAIGN_DIR/windows_telemetry.ps1")
  csv_win=$(wslpath -w "$run_dir/windows_telemetry.csv")
  stop_win=$(wslpath -w "$run_dir/telemetry.stop")
  abort_win=$(wslpath -w "$run_dir/windows_telemetry.abort")
  pid_win=$(wslpath -w "$run_dir/windows_telemetry.windows_pid")
  powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "$ps_win" -CsvPath "$csv_win" -StopPath "$stop_win" \
    -AbortPath "$abort_win" -PidPath "$pid_win" \
    -FloorGiB "$WINDOWS_RAM_FLOOR_GIB" \
    -PollMilliseconds "$((TELEMETRY_INTERVAL_SECONDS * 1000))" \
    > "$run_dir/windows_telemetry.stdout.log" \
    2> "$run_dir/windows_telemetry.stderr.log" &
  WINDOWS_MON_PID=$!
  WINDOWS_MON_START_TICKS=$(proc_start_ticks "$WINDOWS_MON_PID")
  printf '%s\n' "$WINDOWS_MON_PID" > "$run_dir/windows_telemetry.launcher_pid"
  printf '%s\n' "$WINDOWS_MON_START_TICKS" > "$run_dir/windows_telemetry.launcher_start_ticks"
  printf 'active\n' > "$run_dir/windows_telemetry.status"
}

start_wsl_telemetry() {
  local run_dir=$1 server_pid=$2 server_start=$3 server_exe=$4
  local windows_required=$WINDOWS_MONITOR_AVAILABLE
  printf 'timestamp_utc,mem_total_mb,mem_available_mb,swap_used_mb,gpu_used_mib,gpu_total_mib,gpu_util_percent,gpu_temperature_c,gpu_power_w,action\n' \
    > "$run_dir/wsl_telemetry.csv"
  (
    local mem_total_kb mem_available_kb swap_total_kb swap_free_kb
    local mem_total_mb mem_available_mb swap_used_mb gpu_line
    local gpu_used gpu_total gpu_util gpu_temp gpu_power action
    while proc_matches "$server_pid" "$server_start" "$server_exe"; do
      mem_total_kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
      mem_available_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      swap_total_kb=$(awk '/^SwapTotal:/{print $2}' /proc/meminfo)
      swap_free_kb=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)
      mem_total_mb=$((mem_total_kb / 1024))
      mem_available_mb=$((mem_available_kb / 1024))
      swap_used_mb=$(((swap_total_kb - swap_free_kb) / 1024))
      gpu_line=$(nvidia-smi \
        --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      IFS=',' read -r gpu_used gpu_total gpu_util gpu_temp gpu_power <<< "$gpu_line"
      gpu_used=${gpu_used//[[:space:]]/}
      gpu_total=${gpu_total//[[:space:]]/}
      gpu_util=${gpu_util//[[:space:]]/}
      gpu_temp=${gpu_temp//[[:space:]]/}
      gpu_power=${gpu_power//[[:space:]]/}
      action=""
      if ((mem_available_mb < WSL_RAM_FLOOR_MB)); then
        action=wsl_memory_floor_breach
      elif [[ -f "$run_dir/windows_telemetry.abort" ]]; then
        action=$(tr -d '\r\n' < "$run_dir/windows_telemetry.abort")
      elif ((windows_required)) &&
           ! child_proc_matches "$WINDOWS_MON_PID" "$WINDOWS_MON_START_TICKS"; then
        action=windows_telemetry_exited
      fi
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(utc_now)" "$mem_total_mb" "$mem_available_mb" "$swap_used_mb" \
        "$gpu_used" "$gpu_total" "$gpu_util" "$gpu_temp" "$gpu_power" "$action" \
        >> "$run_dir/wsl_telemetry.csv"
      if [[ -n "$action" ]]; then
        printf '%s\n' "$action" > "$run_dir/ABORT.txt"
        if proc_matches "$server_pid" "$server_start" "$server_exe"; then
          kill -TERM "$server_pid" 2>/dev/null || true
        fi
        break
      fi
      sleep "$TELEMETRY_INTERVAL_SECONDS"
    done
  ) &
  WSL_MON_PID=$!
  WSL_MON_START_TICKS=$(proc_start_ticks "$WSL_MON_PID")
  printf '%s\n' "$WSL_MON_PID" > "$run_dir/wsl_telemetry.pid"
  printf '%s\n' "$WSL_MON_START_TICKS" > "$run_dir/wsl_telemetry.start_ticks"
}

build_server_contract() {
  local run_dir=$1 arm=$2 budget=$3
  SERVER_ENV=(
    "HOME=${HOME:-/root}"
    "USER=${USER:-root}"
    "LOGNAME=${LOGNAME:-root}"
    "SHELL=/bin/bash"
    "PATH=$PATH"
    "PWD=$SRC"
    "TMPDIR=/tmp"
    "LANG=C.UTF-8"
    "LC_ALL=C.UTF-8"
    "CUDA_VISIBLE_DEVICES=0"
    "DS4_CUDA_NO_DIRECT_IO=1"
    "DS4_CUDA_KEEP_MODEL_PAGES=1"
    "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1"
    "DS4_CUDA_NO_Q8_F16_CACHE=1"
    "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256"
    "DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1"
    "DS4_PACE=0"
    "DS4_REAP_MASK_FILE=$MASK"
    "DS4_REAP_PREFETCH=1"
    "DS4_REAP_PREFETCH_THREADS=16"
    "DS4_SPEX_STATS=0"
    "DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=$budget"
    "DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1"
  )
  [[ -z "${LD_LIBRARY_PATH:-}" ]] || SERVER_ENV+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
  if [[ "$arm" == on ]]; then
    SERVER_ENV+=("DS4_CUDA_STREAM_FROM_RAM_MASKED=$MASK")
  fi

  SERVER_CLI=(
    -m "$MODEL"
    --cuda
    --ssd-streaming
    --ssd-streaming-cache-experts "$CACHE_EXPERTS"
    --prefill-chunk "$PREFILL_CHUNK"
    -c "$CTX"
    -n "$SERVER_MAX_TOKENS"
    --host 127.0.0.1
    --port "$PORT"
    --cors
  )

  printf '%s\0' "${SERVER_ENV[@]}" > "$run_dir/server_env.intended.nul"
  printf '%s\n' "${SERVER_ENV[@]}" | LC_ALL=C sort > "$run_dir/server_env.intended.txt"
  printf '%s\0' "$BIN" "${SERVER_CLI[@]}" > "$run_dir/server_argv.intended.nul"
  {
    printf 'cd %q\n' "$SRC"
    printf 'exec env -i'
    printf ' %q' "${SERVER_ENV[@]}"
    printf ' %q' "$BIN" "${SERVER_CLI[@]}"
    printf '\n'
  } > "$run_dir/server_command.sh"
}

wait_for_ready() {
  local run_dir=$1 ready=0 rc attempt
  printf 'attempt\ttimestamp_utc\tcurl_rc\n' > "$run_dir/ready_attempts.tsv"
  for ((attempt=1; attempt<=180; attempt++)); do
    set +e
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" \
      -o "$run_dir/ready_response.json" 2>> "$run_dir/ready_curl.stderr.log"
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' "$attempt" "$(utc_now)" "$rc" >> "$run_dir/ready_attempts.tsv"
    if ((rc == 0)); then
      ready=1
      break
    fi
    proc_matches "$SERVER_PID" "$SERVER_START_TICKS" "$SERVER_EXE" || break
    [[ ! -f "$run_dir/ABORT.txt" ]] || break
    sleep 5
  done
  printf '%s\n' "$ready" > "$run_dir/server.ready"
  ((ready == 1))
}

validate_arm_attribution() {
  local run_dir=$1 arm=$2
  python3 - "$run_dir/server.stderr.log" "$arm" "$run_dir/arm_attribution.json" <<'PY'
import json
import re
import sys

log_path, arm, out_path = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()
registrations = re.findall(
    r"(\d+)/(\d+) ranges? registered, ([0-9.]+) GiB zero-copy", text
)
diags = re.findall(
    r"CUDA masked zero-copy diag (\w+): queries=(\d+)/[0-9.]+ MiB "
    r"covered=(\d+)/[0-9.]+ MiB dma_ok=(\d+)/[0-9.]+ MiB "
    r"dma_failed=(\d+) miss_empty=(\d+) miss_base=(\d+) "
    r"miss_before=(\d+) miss_range=(\d+)",
    text,
)
active = "CUDA masked zero-copy DMA path ACTIVE" in text
result = {
    "arm": arm,
    "active_line": active,
    "registration": None,
    "final_diag": None,
    "checks": {},
}
if registrations:
    registered, requested, gib = registrations[-1]
    result["registration"] = {
        "registered": int(registered),
        "requested": int(requested),
        "gib": float(gib),
    }
if diags:
    why, queries, covered, dma_ok, dma_failed, miss_empty, miss_base, miss_before, miss_range = diags[-1]
    result["final_diag"] = {
        "why": why,
        "queries": int(queries),
        "covered": int(covered),
        "dma_ok": int(dma_ok),
        "dma_failed": int(dma_failed),
        "miss_empty": int(miss_empty),
        "miss_base": int(miss_base),
        "miss_before": int(miss_before),
        "miss_range": int(miss_range),
    }

if arm == "on":
    reg = result["registration"] or {}
    diag = result["final_diag"] or {}
    result["checks"] = {
        "all_ranges_registered": bool(reg) and reg.get("registered", 0) > 0 and reg.get("registered") == reg.get("requested"),
        "dma_path_active": active,
        "dma_ok_positive": diag.get("dma_ok", 0) > 0,
        "dma_failed_zero": diag.get("dma_failed") == 0,
        "miss_empty_zero": diag.get("miss_empty") == 0,
        "miss_base_zero": diag.get("miss_base") == 0,
    }
else:
    diag = result["final_diag"] or {}
    result["checks"] = {
        "no_ranges_registered": not registrations,
        "dma_path_inactive": not active,
        "covered_zero": diag.get("covered") == 0,
        "dma_ok_zero": diag.get("dma_ok") == 0,
    }
result["pass"] = bool(result["checks"]) and all(result["checks"].values())
open(out_path, "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
raise SystemExit(0 if result["pass"] else 1)
PY
}

run_trial() {
  local phase=$1 round=$2 position=$3 treatment=$4 request_source=$5 mode=$6
  local arm budget run_id run_dir timeout_s client_rc attribution_rc forced_stop
  local telemetry_rc=0 wsl_telemetry_lines windows_telemetry_lines
  arm=$(treatment_arm "$treatment")
  budget=$(treatment_budget "$treatment")
  printf -v run_id '%s_r%02d_p%02d_%s' "$phase" "$round" "$position" "$treatment"
  run_dir="$CAMPAIGN_DIR/$run_id"
  [[ ! -e "$run_dir" ]] || die "refusing to reuse run directory: $run_dir"
  mkdir "$run_dir"
  CURRENT_RUN_DIR=$run_dir

  verify_fixed_state
  assert_no_server
  cp -- "$request_source" "$run_dir/request.json"
  sha256sum "$run_dir/request.json" > "$run_dir/request.sha256"
  {
    printf 'run_id=%s\n' "$run_id"
    printf 'phase=%s\n' "$phase"
    printf 'round=%s\n' "$round"
    printf 'position=%s\n' "$position"
    printf 'treatment=%s\n' "$treatment"
    printf 'arm=%s\n' "$arm"
    printf 'budget_gib=%s\n' "$budget"
    printf 'request_mode=%s\n' "$mode"
    printf 'started_utc=%s\n' "$(utc_now)"
  } > "$run_dir/RUN_META.txt"

  build_server_contract "$run_dir" "$arm" "$budget"
  start_windows_telemetry "$run_dir"

  (
    cd "$SRC"
    exec env -i "${SERVER_ENV[@]}" "$BIN" "${SERVER_CLI[@]}"
  ) > "$run_dir/server.stdout.log" 2> "$run_dir/server.stderr.log" &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$run_dir/server.pid"
  for _ in {1..50}; do
    [[ -r "/proc/$SERVER_PID/stat" ]] && break
    sleep 0.02
  done
  [[ -r "/proc/$SERVER_PID/stat" ]] || {
    stop_telemetry
    die "server exited before process identity capture: $run_id"
  }
  SERVER_START_TICKS=$(proc_start_ticks "$SERVER_PID")
  SERVER_EXE=$(readlink -f "/proc/$SERVER_PID/exe")
  [[ "$SERVER_EXE" == "$(readlink -f "$BIN")" ]] || die "launched PID is not the expected binary"
  printf '%s\n' "$SERVER_START_TICKS" > "$run_dir/server.start_ticks"
  printf '%s\n' "$SERVER_EXE" > "$run_dir/server.exe"
  tr '\0' '\n' < "/proc/$SERVER_PID/cmdline" > "$run_dir/server_argv.actual.txt"
  tr '\0' '\n' < "/proc/$SERVER_PID/environ" | LC_ALL=C sort > "$run_dir/server_env.actual.txt"

  start_wsl_telemetry "$run_dir" "$SERVER_PID" "$SERVER_START_TICKS" "$SERVER_EXE"
  if ! wait_for_ready "$run_dir"; then
    printf 'server_not_ready\n' > "$run_dir/STOP_REASON.txt"
    stop_owned_server
    stop_telemetry
    return 1
  fi

  log "$run_id request start"
  client_rc=0
  if [[ "$mode" == nonstream ]]; then
    timeout_s=1800
    set +e
    curl -sS --fail-with-body --max-time "$timeout_s" \
      -H 'Content-Type: application/json' \
      -d @"$run_dir/request.json" \
      -o "$run_dir/response.json" \
      -w 'http_code=%{http_code}\ntime_total=%{time_total}\n' \
      "http://127.0.0.1:$PORT/v1/chat/completions" \
      > "$run_dir/client_timing.txt" 2> "$run_dir/client.stderr.log"
    client_rc=$?
    set -e
  elif [[ "$mode" == measured_stream ]]; then
    set +e
    python3 "$MEASURE_SCRIPT" \
      --url "http://127.0.0.1:$PORT/v1/chat/completions" \
      --request "$run_dir/request.json" \
      --out "$run_dir/measure.json" \
      --live "$run_dir/stream_live.txt" \
      --drop 40 \
      > "$run_dir/client.stdout.log" 2> "$run_dir/client.stderr.log"
    client_rc=$?
    set -e
  else
    die "invalid request mode: $mode"
  fi
  printf '%s\n' "$client_rc" > "$run_dir/client.rc"
  log "$run_id request end rc=$client_rc"

  stop_owned_server
  stop_telemetry
  forced_stop=$(cat "$run_dir/server.stop_mode" 2>/dev/null || true)
  wsl_telemetry_lines=$(wc -l < "$run_dir/wsl_telemetry.csv" 2>/dev/null || printf '0')
  if [[ "$(cat "$run_dir/wsl_telemetry.wait_rc" 2>/dev/null || printf '1')" != 0 ]] ||
     ((wsl_telemetry_lines < 2)); then
    telemetry_rc=1
  fi
  if ((WINDOWS_MONITOR_AVAILABLE)); then
    windows_telemetry_lines=$(wc -l < "$run_dir/windows_telemetry.csv" 2>/dev/null || printf '0')
    if [[ "$(cat "$run_dir/windows_telemetry.wait_rc" 2>/dev/null || printf '1')" != 0 ]] ||
       ((windows_telemetry_lines < 2)); then
      telemetry_rc=1
    fi
  fi
  printf 'telemetry_rc=%s\nwsl_lines=%s\nwindows_lines=%s\n' \
    "$telemetry_rc" "$wsl_telemetry_lines" "${windows_telemetry_lines:-0}" \
    > "$run_dir/telemetry_validation.txt"
  attribution_rc=0
  set +e
  validate_arm_attribution "$run_dir" "$arm"
  attribution_rc=$?
  set -e
  printf '%s\n' "$attribution_rc" > "$run_dir/arm_attribution.rc"

  {
    printf 'finished_utc=%s\n' "$(utc_now)"
    printf 'client_rc=%s\n' "$client_rc"
    printf 'telemetry_rc=%s\n' "$telemetry_rc"
    printf 'arm_attribution_rc=%s\n' "$attribution_rc"
    printf 'server_stop_mode=%s\n' "$forced_stop"
  } >> "$run_dir/RUN_META.txt"

  if [[ -f "$run_dir/ABORT.txt" ]]; then
    cp -- "$run_dir/ABORT.txt" "$run_dir/STOP_REASON.txt"
    return 1
  elif ((client_rc != 0)); then
    printf 'client_failed_rc=%s\n' "$client_rc" > "$run_dir/STOP_REASON.txt"
    return 1
  elif ((telemetry_rc != 0)); then
    printf 'telemetry_invalid\n' > "$run_dir/STOP_REASON.txt"
    return 1
  elif ((attribution_rc != 0)); then
    printf 'arm_attribution_failed\n' > "$run_dir/STOP_REASON.txt"
    return 1
  elif [[ "$forced_stop" == kill_after_term_timeout ]]; then
    printf 'server_required_forced_kill\n' > "$run_dir/STOP_REASON.txt"
    return 1
  fi
  printf 'completed\n' > "$run_dir/STOP_REASON.txt"
  CURRENT_RUN_DIR=""
  return 0
}

extract_gate_content() {
  local run_dir=$1
  python3 - "$run_dir/response.json" "$run_dir/content.bin" <<'PY'
import json
import pathlib
import sys

obj = json.load(open(sys.argv[1], encoding="utf-8"))
content = obj["choices"][0]["message"]["content"]
pathlib.Path(sys.argv[2]).write_bytes(content.encode("utf-8"))
PY
  sha256sum "$run_dir/content.bin" > "$run_dir/content.sha256"
}

run_gate() {
  local treatment position run_dir baseline
  position=0
  for treatment in off24 on24 off28 on28; do
    ((position+=1))
    run_trial gate 0 "$position" "$treatment" \
      "$CAMPAIGN_DIR/request_gate_temp0_60.json" nonstream
    printf -v run_dir '%s/gate_r00_p%02d_%s' "$CAMPAIGN_DIR" "$position" "$treatment"
    extract_gate_content "$run_dir"
    sleep "$COOLDOWN_SECONDS"
  done

  baseline="$CAMPAIGN_DIR/gate_r00_p01_off24/content.bin"
  printf 'treatment\tsha256\tbytes\tequal_to_off24\n' > "$CAMPAIGN_DIR/bit_exact_gate.tsv"
  position=0
  for treatment in off24 on24 off28 on28; do
    ((position+=1))
    printf -v run_dir '%s/gate_r00_p%02d_%s' "$CAMPAIGN_DIR" "$position" "$treatment"
    local hash bytes equal=0
    hash=$(sha256sum "$run_dir/content.bin" | awk '{print $1}')
    bytes=$(stat -c '%s' "$run_dir/content.bin")
    cmp -s "$baseline" "$run_dir/content.bin" && equal=1
    printf '%s\t%s\t%s\t%s\n' "$treatment" "$hash" "$bytes" "$equal" \
      >> "$CAMPAIGN_DIR/bit_exact_gate.tsv"
    ((equal == 1)) || {
      printf 'FAIL\n' > "$CAMPAIGN_DIR/BIT_EXACT_GATE.txt"
      die "bit-exact gate mismatch for $treatment"
    }
  done
  printf 'PASS\n' > "$CAMPAIGN_DIR/BIT_EXACT_GATE.txt"
}

run_warmups() {
  local treatment position=0
  for treatment in on28 off28 on24 off24; do
    ((position+=1))
    run_trial warmup 0 "$position" "$treatment" \
      "$CAMPAIGN_DIR/request_warmup_temp0_80.json" nonstream
    sleep "$COOLDOWN_SECONDS"
  done
  printf 'PASS\n' > "$CAMPAIGN_DIR/WARMUP_PHASE.txt"
}

run_measured() {
  local round treatment position
  for ((round=1; round<=ROUNDS; round++)); do
    if ((round % 2)); then
      treatments=(off24 on24 off28 on28)
    else
      treatments=(on28 off28 on24 off24)
    fi
    position=0
    for treatment in "${treatments[@]}"; do
      ((position+=1))
      run_trial measured "$round" "$position" "$treatment" \
        "$CAMPAIGN_DIR/request_measured_temp0_450.json" measured_stream
      sleep "$COOLDOWN_SECONDS"
    done
  done
}

write_performance_manifest() {
  python3 - "$CAMPAIGN_DIR" <<'PY'
import csv
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
rows = []
pat = re.compile(r"measured_r(\d+)_p(\d+)_(off|on)(24|28)$")
for path in sorted(root.glob("measured_r*_p*_*")):
    m = pat.fullmatch(path.name)
    if not m:
        continue
    measure_path = path / "measure.json"
    if not measure_path.exists():
        continue
    measure = json.loads(measure_path.read_text(encoding="utf-8"))
    rows.append({
        "run_id": path.name,
        "round": int(m.group(1)),
        "position": int(m.group(2)),
        "arm": m.group(3),
        "budget_gib": int(m.group(4)),
        "ttft_s": measure.get("ttft_s"),
        "steady_decode_tps": measure.get("steady_decode_tps"),
        "full_decode_tps": measure.get("full_decode_tps"),
        "steady_tokens": measure.get("steady_tokens"),
        "steady_window_s": measure.get("steady_window_s"),
        "n_deltas": measure.get("n_deltas"),
    })
fields = [
    "run_id", "round", "position", "arm", "budget_gib", "ttft_s",
    "steady_decode_tps", "full_decode_tps", "steady_tokens",
    "steady_window_s", "n_deltas",
]
with open(root / "performance_manifest.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerows(rows)
PY
}

main() {
  if [[ "$MODE" == preflight ]]; then
    preflight
    exit 0
  fi

  [[ -n "$CAMPAIGN_ID" ]] || die "--execute requires --campaign-id"
  [[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "campaign id may contain only letters, digits, dot, underscore, and dash"

  preflight >/dev/null
  exec 9>"$GPU_LOCK"
  flock -n 9 || die "GPU lock became active before acquisition: $GPU_LOCK"
  LOCK_HELD=1
  assert_no_server

  mkdir -p "$OUT_ROOT"
  CAMPAIGN_DIR="$OUT_ROOT/$CAMPAIGN_ID"
  [[ ! -e "$CAMPAIGN_DIR" ]] || die "campaign directory already exists: $CAMPAIGN_DIR"
  mkdir "$CAMPAIGN_DIR"
  trap on_signal INT TERM
  trap on_exit EXIT

  printf '%s\n' "$$" > "$CAMPAIGN_DIR/harness.pid"
  printf '%s\tSTARTED\t0\n' "$(utc_now)" > "$CAMPAIGN_DIR/campaign_status.tsv"
  log "campaign $CAMPAIGN_ID started; GPU lock held non-blocking"

  capture_snapshot
  generate_requests
  write_plan
  write_windows_monitor
  log "snapshot and immutable execution plan captured"

  run_gate
  log "bit-exact temp0 gate passed across OFF/ON and 24/28 GiB"

  run_warmups
  log "separate discarded warm-up phase completed"

  run_measured
  write_performance_manifest
  verify_fixed_state
  sha256sum "$MODEL" > "$CAMPAIGN_DIR/snapshot/model.after.sha256"
  cmp -s "$CAMPAIGN_DIR/snapshot/model.sha256" \
    "$CAMPAIGN_DIR/snapshot/model.after.sha256" || die "model SHA-256 changed during campaign"
  cat /proc/meminfo > "$CAMPAIGN_DIR/snapshot/meminfo.after.txt"
  nvidia-smi -q > "$CAMPAIGN_DIR/snapshot/nvidia_smi_q.after.txt"
  CAMPAIGN_COMPLETE=1
  log "campaign $CAMPAIGN_ID complete"
}

main
