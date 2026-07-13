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
