[CmdletBinding()]
param(
    [ValidateRange(0.01, 60.0)]
    [double]$TargetGiB = 0.25,

    [ValidateRange(0.01, 10.0)]
    [double]$StepGiB = 1.0,

    [ValidateSet('blocks', 'single')]
    [string]$Mode = 'blocks',

    [ValidateSet('hostalloc', 'mallochost')]
    [string]$Api = 'hostalloc',

    [ValidateRange(1.0, 64.0)]
    [double]$MinWindowsAvailableGiB = 2.0,

    [ValidateRange(1.0, 32.0)]
    [double]$WslReserveGiB = 8.0,

    [string]$Distro = 'Ubuntu-24.04',

    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot '..\arena_probe'
}
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output | Out-Null

$runningServer = & wsl.exe -d $Distro -- bash -lc 'pgrep -a -x ds4-server || true'
if ($runningServer) {
    throw "ds4-server is running in $Distro; refusing arena probe: $runningServer"
}

function Convert-ToWslPath([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $drive = $resolved.Substring(0, 1).ToLowerInvariant()
    $tail = $resolved.Substring(2).Replace('\', '/')
    return "/mnt/$drive$tail"
}

$source = Join-Path $PSScriptRoot 'cuda_pinned_arena_probe.cu'
$sourceWsl = Convert-ToWslPath $source
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$targetArg = $TargetGiB.ToString('0.######', $invariant)
$stepArg = $StepGiB.ToString('0.######', $invariant)
$reserveArg = $WslReserveGiB.ToString('0.######', $invariant)
$base = "probe_${stamp}_${Mode}_${Api}_${targetArg}g"
$stdout = Join-Path $output "$base.jsonl"
$stderr = Join-Path $output "$base.stderr.txt"
$monitor = Join-Path $output "$base.windows_memory.csv"
$runner = Join-Path $output "$base.runner.sh"
$pidFile = Join-Path $output "$base.pid"
$rcFile = Join-Path $output "$base.rc"
$runnerWsl = Convert-ToWslPath $runner
$pidFileWsl = Convert-ToWslPath $pidFile
$rcFileWsl = Convert-ToWslPath $rcFile

$bash = @"
#!/usr/bin/env bash
set -euo pipefail
/usr/local/cuda-12.8/bin/nvcc -std=c++17 -O2 -arch=sm_86 \
  -o /tmp/cuda_pinned_arena_probe '$sourceWsl'
set +e
/tmp/cuda_pinned_arena_probe \
  --mode '$Mode' --api '$Api' --target-gib '$targetArg' --step-gib '$stepArg' \
  --reserve-gib '$reserveArg' &
probe_pid=`$!
echo `$probe_pid > '$pidFileWsl'
wait `$probe_pid
probe_rc=`$?
echo `$probe_rc > '$rcFileWsl'
exit `$probe_rc
"@
[System.IO.File]::WriteAllText($runner, $bash.Replace("`r`n", "`n"),
    [System.Text.UTF8Encoding]::new($false))

'timestamp,windows_available_gib,action' | Set-Content -LiteralPath $monitor -Encoding ascii
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList @(
    '-d', $Distro, '--', 'bash', $runnerWsl
) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$guardFired = $false
try {
    while (-not $process.HasExited) {
        $os = Get-CimInstance Win32_OperatingSystem
        $available = [double]$os.FreePhysicalMemory / 1MB
        $action = ''
        if ($available -lt $MinWindowsAvailableGiB) {
            $action = 'terminate_probe'
            $guardFired = $true
        }
        [string]::Format($invariant, '{0},{1:F3},{2}',
            (Get-Date -Format o), $available, $action) |
            Add-Content -LiteralPath $monitor -Encoding ascii

        if ($guardFired) {
            if (Test-Path -LiteralPath $pidFile) {
                $probePid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
                if ($probePid -match '^\d+$') {
                    & wsl.exe -d $Distro -- kill -TERM $probePid 2>$null
                }
            }
            break
        }
        Start-Sleep -Milliseconds 250
        $process.Refresh()
    }

    if ($guardFired -and -not $process.WaitForExit(5000)) {
        if (Test-Path -LiteralPath $pidFile) {
            $probePid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
            if ($probePid -match '^\d+$') {
                & wsl.exe -d $Distro -- kill -KILL $probePid 2>$null
            }
        }
    }
    $process.WaitForExit()
} finally {
    if (-not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
}

$probeExitCode = if (Test-Path -LiteralPath $rcFile) {
    [int](Get-Content -LiteralPath $rcFile -Raw).Trim()
} elseif ($guardFired) {
    20
} else {
    21
}
$summary = [ordered]@{
    target_gib = $TargetGiB
    step_gib = $StepGiB
    mode = $Mode
    api = $Api
    min_windows_available_gib = $MinWindowsAvailableGiB
    wsl_reserve_gib = $WslReserveGiB
    guard_fired = $guardFired
    exit_code = $probeExitCode
    stdout = $stdout
    stderr = $stderr
    windows_memory_csv = $monitor
}
$summary | ConvertTo-Json
if ($guardFired) { exit 20 }
exit $probeExitCode
