param(
    [string]$RunnerPath = (Join-Path $PSScriptRoot 'run_two_turn_lifecycle_a2.ps1'),
    [string]$ReceiptPath = (
        Join-Path (
            Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
        ) 'outputs\a2_harness_static_test_receipt.txt')
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

function Get-ResultMap {
    param([string]$Path)
    $map = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^([^=]+)=(.*)$') {
            $map[$matches[1]] = $matches[2]
        }
    }
    return $map
}

function Get-PreparedRunDirectory {
    param([object[]]$Output)
    foreach ($line in $Output) {
        if ([string]$line -match 'prepared run_dir=(.+)$') {
            return $matches[1]
        }
    }
    throw 'ValidateOnly output did not identify its run directory'
}

$tokens = $null
$syntaxErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    $RunnerPath,
    [ref]$tokens,
    [ref]$syntaxErrors) | Out-Null
Assert-True ($syntaxErrors.Count -eq 0) 'runner has PowerShell syntax errors'

$runnerText = Get-Content -LiteralPath $RunnerPath -Raw
$requiredFragments = @(
    "[string]`$G73Open = 'On'"
    '$effectiveManifestLines'
    '$g73D2hBranchGatePass'
    '$engagementGatePass'
    '$wrapOnGatePass'
    '$wrapOffGatePass'
    '$correctnessGatePass'
    '$graphGatePass'
    '$turn1GraphTps'
    '$turn2GraphTps'
    '$monitorGatePass'
    '$postflightGatePass'
    '$shutdownGatePass'
)
foreach ($fragment in $requiredFragments) {
    Assert-True ($runnerText.Contains($fragment)) "missing runner fragment: $fragment"
}
$validateIndex = $runnerText.IndexOf('if ($ValidateOnly)')
$launchIndex = $runnerText.IndexOf(
    'Start-Process -FilePath $launchFile')
Assert-True ($validateIndex -ge 0) 'ValidateOnly block missing'
Assert-True ($launchIndex -gt $validateIndex) 'ValidateOnly must precede DS4 Start-Process'

$ds4BeforeProcesses = @(
    Get-Process ds4_server -ErrorAction SilentlyContinue |
    Sort-Object Id
)
$ds4Before = $ds4BeforeProcesses.Count
$ds4BeforePids = @($ds4BeforeProcesses | ForEach-Object { $_.Id })
$onOutput = @(
    & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File $RunnerPath `
        -G73Open On `
        -ValidateOnly `
        -AllowSourceDriftForValidation
)
$onExit = $LASTEXITCODE
Start-Sleep -Milliseconds 1100
$offOutput = @(
    & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File $RunnerPath `
        -G73Open Off `
        -ValidateOnly `
        -AllowSourceDriftForValidation
)
$offExit = $LASTEXITCODE
$ds4AfterProcesses = @(
    Get-Process ds4_server -ErrorAction SilentlyContinue |
    Sort-Object Id
)
$ds4After = $ds4AfterProcesses.Count
$ds4AfterPids = @($ds4AfterProcesses | ForEach-Object { $_.Id })

Assert-True ($onExit -eq 0) "ON ValidateOnly failed: exit=$onExit"
Assert-True ($offExit -eq 0) "OFF ValidateOnly failed: exit=$offExit"
Assert-True ($ds4After -eq $ds4Before) 'ValidateOnly changed ds4_server process count'
Assert-True (
    (($ds4BeforePids -join ',') -eq ($ds4AfterPids -join ','))
) 'ValidateOnly changed the ds4_server PID set'

$onRunDirectory = Get-PreparedRunDirectory -Output $onOutput
$offRunDirectory = Get-PreparedRunDirectory -Output $offOutput
$onManifest = @(
    Get-Content -LiteralPath (Join-Path $onRunDirectory 'manifest_47.env')
)
$offManifest = @(
    Get-Content -LiteralPath (Join-Path $offRunDirectory 'manifest_47.env')
)
Assert-True ($onManifest.Count -eq 47) 'ON manifest is not 47/47'
Assert-True ($offManifest.Count -eq 47) 'OFF manifest is not 47/47'

$differentLines = @()
for ($index = 0; $index -lt 47; $index++) {
    if ($onManifest[$index] -cne $offManifest[$index]) {
        $differentLines += ($index + 1)
    }
}
Assert-True ($differentLines.Count -eq 1) 'A/B manifests differ by more than one line'
Assert-True ($differentLines[0] -eq 41) 'A/B manifest delta is not line 41'
Assert-True ($onManifest[40] -ceq 'DS4_G73_OPEN=1') 'ON line 41 is wrong'
Assert-True ($offManifest[40] -ceq 'DS4_G73_OPEN=0') 'OFF line 41 is wrong'
Assert-True ($onManifest[41] -ceq 'DS4_G73_PAGEABLE_OVERFLOW_GB=0') `
    'ON pageable overflow is not zero'
Assert-True ($offManifest[41] -ceq 'DS4_G73_PAGEABLE_OVERFLOW_GB=0') `
    'OFF pageable overflow is not zero'

$forbiddenMicroNames = @(
    'DS4_CUDA_MOE_ROUTE_PACKED_COPY'
    'DS4_CUDA_MOE_ROUTE_BATCHED_PUBLISH'
    'DS4_CUDA_G73_REUSE_HOST_SELECTED'
    'DS4_CUDA_G73_ROUTE_IO_QD'
    'DS4_CUDA_GRAPH_TENSOR_DEVICE'
    'DS4_CUDA_ALLOC_TRACE'
)
foreach ($name in $forbiddenMicroNames) {
    Assert-True (
        @($onManifest | Where-Object { $_ -match "^$name=" }).Count -eq 0
    ) "ON effective manifest contains $name"
    Assert-True (
        @($offManifest | Where-Object { $_ -match "^$name=" }).Count -eq 0
    ) "OFF effective manifest contains $name"
}

$onResult = Get-ResultMap -Path (Join-Path $onRunDirectory 'result.txt')
$offResult = Get-ResultMap -Path (Join-Path $offRunDirectory 'result.txt')
Assert-True ($onResult['validation'] -eq 'PASS') 'ON static validation did not pass'
Assert-True ($offResult['validation'] -eq 'PASS') 'OFF static validation did not pass'
Assert-True ($onResult['server_started'] -eq 'no') 'ON ValidateOnly claims a server start'
Assert-True ($offResult['server_started'] -eq 'no') 'OFF ValidateOnly claims a server start'
Assert-True (
    $onResult['normalized_pair_manifest_sha256'] -eq
    $offResult['normalized_pair_manifest_sha256']
) 'normalized A/B manifest hashes differ'
Assert-True ($onResult['manifest_diff_count'] -eq '0') 'ON diff count is not zero'
Assert-True ($offResult['manifest_diff_count'] -eq '1') 'OFF diff count is not one'

$receiptDirectory = Split-Path -Parent $ReceiptPath
New-Item -ItemType Directory -Force -Path $receiptDirectory | Out-Null
@(
    'test=PASS'
    'syntax_errors=0'
    "on_validate_exit=$onExit"
    "off_validate_exit=$offExit"
    "ds4_process_count_before=$ds4Before"
    "ds4_process_count_after=$ds4After"
    "ds4_pids_before=$($ds4BeforePids -join ',')"
    "ds4_pids_after=$($ds4AfterPids -join ',')"
    'server_started=no'
    "on_run_directory=$onRunDirectory"
    "off_run_directory=$offRunDirectory"
    'manifest_count_on=47'
    'manifest_count_off=47'
    'manifest_diff_count=1'
    'manifest_diff_line=41'
    "normalized_pair_manifest_sha256=$($onResult['normalized_pair_manifest_sha256'])"
    "source_hash_gate_pass=$($onResult['source_hash_gate_pass'])"
) | Set-Content -LiteralPath $ReceiptPath -Encoding ascii

Write-Output "PASS receipt=$ReceiptPath"
