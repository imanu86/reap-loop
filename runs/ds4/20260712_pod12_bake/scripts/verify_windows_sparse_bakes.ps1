[CmdletBinding()]
param(
    [int[]]$DownloadProcessId = @(18200, 23836),
    [string]$PackRoot = 'D:\ds4-models',
    [string]$BakeRoot = 'C:\ds4-models',
    [string]$Python = '',
    [string]$ReceiptRoot = ''
)

$ErrorActionPreference = 'Stop'

$campaignRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$packer = Join-Path $repoRoot 'scripts\ds4_windows_sparse_bake.py'
if (-not $Python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        $Python = $command.Source
    }
    else {
        $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundled -PathType Leaf) {
            $Python = $bundled
        }
        else {
            throw 'No Python interpreter found; pass -Python explicitly'
        }
    }
}
if (-not $ReceiptRoot) {
    $ReceiptRoot = Join-Path $campaignRoot 'windows_sparse_verification_20260715'
}

$specs = @(
    [pscustomobject]@{
        name = 'k60_mass'
        pack_name = 'ds4-2bit-k60-mass-full-decode-5b6d9850.ds4pack'
        pack_bytes = [int64]57842530728
        pack_sha256 = '3b464ee43514c8caa841be61da70190a4a7ba3c760c22849d2d723e5da5b7d71'
        payload_sha256 = '5cb4bf69d7c6ef2aadfc8760069c3a7a89fc40504ce80b7b3735b10f9539e4b5'
        bake_name = 'ds4-2bit-k60-mass-full-decode.gguf'
    },
    [pscustomobject]@{
        name = 'k75_mass'
        pack_name = 'ds4-2bit-k75-mass-full-decode-e4b6059f.ds4pack'
        pack_bytes = [int64]68600895205
        pack_sha256 = 'f1dbb64e1c8261928b56b4fa154238559444f70e8a1796875b22e42abf455dd2'
        payload_sha256 = '9b8f67ad4f69bfcd3a2369839c936371c8a433e53ed951582d0ad491c718aa3d'
        bake_name = 'ds4-2bit-k75-mass-full-decode.gguf'
    }
)

function Write-RunLog {
    param([string]$Message)
    $line = '[{0}] {1}' -f ([DateTime]::UtcNow.ToString('o')), $Message
    Add-Content -LiteralPath $script:logPath -Value $line -Encoding UTF8
}

function Write-Receipt {
    param([System.Collections.IDictionary]$Receipt)
    $temporary = "$script:receiptPath.tmp"
    $Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $script:receiptPath -Force
}

function Get-AllocatedBytes {
    param([string]$Path)
    if (-not ('Ds4SparseFile' -as [type])) {
        Add-Type @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class Ds4SparseFile {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetCompressedFileSizeW(string fileName, out uint high);

    public static ulong GetAllocatedBytes(string path) {
        uint high;
        uint low = GetCompressedFileSizeW(path, out high);
        if (low == 0xffffffff) {
            int error = Marshal.GetLastWin32Error();
            if (error != 0) throw new Win32Exception(error);
        }
        return ((ulong)high << 32) | low;
    }
}
'@
    }
    return [uint64][Ds4SparseFile]::GetAllocatedBytes($Path)
}

New-Item -ItemType Directory -Path $ReceiptRoot -Force | Out-Null
New-Item -ItemType Directory -Path $BakeRoot -Force | Out-Null
$script:logPath = Join-Path $ReceiptRoot 'verification.log'
$script:receiptPath = Join-Path $ReceiptRoot 'verification_status.json'

$receipt = [ordered]@{
    status = 'waiting_for_downloads'
    started_utc = [DateTime]::UtcNow.ToString('o')
    completed_utc = $null
    download_process_ids = @($DownloadProcessId)
    pack_root = $PackRoot
    bake_root = $BakeRoot
    packer = $packer
    results = @()
    error = $null
}
Write-Receipt $receipt
Write-RunLog "Waiting for download processes: $($DownloadProcessId -join ', ')"

try {
    foreach ($downloadId in $DownloadProcessId) {
        while (Get-Process -Id $downloadId -ErrorAction SilentlyContinue) {
            Start-Sleep -Seconds 30
        }
        Write-RunLog "Download process $downloadId exited"
    }

    $receipt.status = 'verifying'
    Write-Receipt $receipt

    foreach ($spec in $specs) {
        $packPath = Join-Path $PackRoot $spec.pack_name
        $bakePath = Join-Path $BakeRoot $spec.bake_name
        $unpackJson = Join-Path $ReceiptRoot "$($spec.name)_unpack.json"
        $unpackError = Join-Path $ReceiptRoot "$($spec.name)_unpack.stderr.log"
        $inspectJson = Join-Path $ReceiptRoot "$($spec.name)_inspect.json"
        $inspectError = Join-Path $ReceiptRoot "$($spec.name)_inspect.stderr.log"

        if (-not (Test-Path -LiteralPath $packPath -PathType Leaf)) {
            throw "Missing completed pack: $packPath"
        }
        $measuredBytes = (Get-Item -LiteralPath $packPath).Length
        if ($measuredBytes -ne $spec.pack_bytes) {
            throw "Pack size mismatch for $($spec.name): expected $($spec.pack_bytes), measured $measuredBytes"
        }

        Write-RunLog "Hashing $packPath ($measuredBytes bytes)"
        $measuredHash = (Get-FileHash -LiteralPath $packPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($measuredHash -ne $spec.pack_sha256) {
            throw "Pack SHA-256 mismatch for $($spec.name): expected $($spec.pack_sha256), measured $measuredHash"
        }

        if (Test-Path -LiteralPath $bakePath) {
            throw "Refusing to overwrite existing bake: $bakePath"
        }

        Write-RunLog "Unpacking verified $($spec.name) to $bakePath"
        & $Python $packer unpack --pack $packPath --out $bakePath 1> $unpackJson 2> $unpackError
        if ($LASTEXITCODE -ne 0) {
            throw "Unpack failed for $($spec.name) with exit $LASTEXITCODE; see $unpackError"
        }
        $unpack = Get-Content -LiteralPath $unpackJson -Raw | ConvertFrom-Json
        if ($unpack.payload_sha256 -ne $spec.payload_sha256) {
            throw "Payload SHA-256 mismatch for $($spec.name): expected $($spec.payload_sha256), measured $($unpack.payload_sha256)"
        }

        Write-RunLog "Inspecting embedded manifest and retained-expert bitset for $($spec.name)"
        & $Python $packer inspect --bake $bakePath 1> $inspectJson 2> $inspectError
        if ($LASTEXITCODE -ne 0) {
            throw "Inspect failed for $($spec.name) with exit $LASTEXITCODE; see $inspectError"
        }
        $inspect = Get-Content -LiteralPath $inspectJson -Raw | ConvertFrom-Json
        $retained = @($inspect.retained_by_layer | ForEach-Object { [int]$_ })
        if (-not $retained.Count) {
            throw "Inspect returned no retained-expert counts for $($spec.name)"
        }

        $attributes = [System.IO.File]::GetAttributes($bakePath)
        if (($attributes -band [System.IO.FileAttributes]::SparseFile) -eq 0) {
            throw "Windows did not set SparseFile on $($spec.name): $attributes"
        }
        $sparseQuery = (& fsutil sparse queryflag $bakePath 2>&1 | Out-String).Trim()
        $logicalBytes = (Get-Item -LiteralPath $bakePath).Length
        $allocatedBytes = Get-AllocatedBytes $bakePath
        if ($allocatedBytes -ge $logicalBytes) {
            throw "Sparse allocation gate failed for $($spec.name): allocated=$allocatedBytes logical=$logicalBytes"
        }

        $ranges = @(& fsutil sparse queryrange $bakePath 2>&1 | Where-Object {
            $_ -match 'Offset:\s+0x([0-9a-fA-F]+)\s+(?:Lunghezza|Length):\s+0x([0-9a-fA-F]+)'
        })
        if ($ranges.Count -lt 2) {
            throw "Sparse range gate failed for $($spec.name): range_count=$($ranges.Count)"
        }

        $receipt.results += [ordered]@{
            name = $spec.name
            pack_path = $packPath
            pack_bytes = $measuredBytes
            pack_sha256 = $measuredHash
            bake_path = $bakePath
            bake_logical_bytes = $logicalBytes
            bake_allocated_bytes = $allocatedBytes
            sparse_range_count = $ranges.Count
            sparse_gap_count = $ranges.Count - 1
            payload_sha256 = $unpack.payload_sha256
            retained_layer_count = $retained.Count
            retained_min = ($retained | Measure-Object -Minimum).Minimum
            retained_max = ($retained | Measure-Object -Maximum).Maximum
            file_attributes = $attributes.ToString()
            sparse_query = $sparseQuery
            unpack_receipt = $unpackJson
            inspect_receipt = $inspectJson
        }
        Write-Receipt $receipt
    }

    $receipt.status = 'complete'
    $receipt.completed_utc = [DateTime]::UtcNow.ToString('o')
    Write-Receipt $receipt
    Write-RunLog 'All Windows sparse-bake verification gates completed'
}
catch {
    $receipt.status = 'failed'
    $receipt.completed_utc = [DateTime]::UtcNow.ToString('o')
    $receipt.error = $_.Exception.Message
    Write-Receipt $receipt
    Write-RunLog "FAILED: $($_.Exception.Message)"
    throw
}
