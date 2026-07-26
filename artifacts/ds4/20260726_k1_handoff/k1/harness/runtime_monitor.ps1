param(
    [Parameter(Mandatory = $true)]
    [int]$ServerPid,
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [Parameter(Mandatory = $true)]
    [string]$StopPath,
    [ValidateRange(1, 60)]
    [int]$SampleSeconds = 5
)

$ErrorActionPreference = 'Continue'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $invariant

$nativeSource = @'
using System;
using System.Runtime.InteropServices;
public static class Ds4RuntimeMemory {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public class MEMORYSTATUSEX {
        public uint dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX));
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GlobalMemoryStatusEx([In, Out] MEMORYSTATUSEX value);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetProcessIoCounters(
        IntPtr processHandle,
        out IO_COUNTERS counters);
}
'@
Add-Type -TypeDefinition $nativeSource

'timestamp,elapsed_s,avail_phys_mb,memory_load_pct,server_ws_mb,server_private_mb,server_read_mb,server_write_mb,server_page_faults,gpu_pstate,gpu_used_mb,gpu_util_pct,gpu_power_w' |
    Set-Content -LiteralPath $CsvPath -Encoding ascii
$started = Get-Date

while (-not (Test-Path -LiteralPath $StopPath)) {
    $process = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
    if (-not $process) {
        break
    }

    $memory = [Ds4RuntimeMemory+MEMORYSTATUSEX]::new()
    $memoryOk = [Ds4RuntimeMemory]::GlobalMemoryStatusEx($memory)
    $io = New-Object Ds4RuntimeMemory+IO_COUNTERS
    $ioOk = [Ds4RuntimeMemory]::GetProcessIoCounters(
        $process.Handle,
        [ref]$io)
    $gpu = (& nvidia-smi `
        --query-gpu=pstate,memory.used,utilization.gpu,power.draw `
        --format=csv,noheader,nounits 2>$null | Select-Object -First 1) -split ','
    $gpuValues = @('unknown', '-1', '-1', '-1')
    for ($i = 0; $i -lt [Math]::Min(4, $gpu.Count); $i++) {
        $gpuValues[$i] = $gpu[$i].Trim()
    }

    $row = @(
        (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffK'),
        [Math]::Round(((Get-Date) - $started).TotalSeconds, 3),
        $(if ($memoryOk) { [Math]::Round($memory.ullAvailPhys / 1MB, 1) } else { -1 }),
        $(if ($memoryOk) { $memory.dwMemoryLoad } else { -1 }),
        [Math]::Round($process.WorkingSet64 / 1MB, 1),
        [Math]::Round($process.PrivateMemorySize64 / 1MB, 1),
        $(if ($ioOk) { [Math]::Round($io.ReadTransferCount / 1MB, 1) } else { -1 }),
        $(if ($ioOk) { [Math]::Round($io.WriteTransferCount / 1MB, 1) } else { -1 }),
        $process.PageFaults,
        $gpuValues[0],
        $gpuValues[1],
        $gpuValues[2],
        $gpuValues[3]
    )
    Add-Content -LiteralPath $CsvPath -Value ($row -join ',') -Encoding ascii
    Start-Sleep -Seconds $SampleSeconds
}
