param(
    [string]$HarnessRoot = $PSScriptRoot
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$runner = Join-Path $HarnessRoot 'run_a4_mtp_lifecycle.ps1'
$monitor = Join-Path $HarnessRoot 'runtime_monitor.ps1'
$manifest = Join-Path $HarnessRoot 'manifest_47.env'
$summarizer = Join-Path $HarnessRoot 'summarize_a4_mtp_ab.ps1'

function Assert-A4 {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if (-not $Condition) {
        throw "A4_STATIC_FAIL: $Message"
    }
}

function Assert-Parses {
    param([string]$Path)
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $Path, [ref]$tokens, [ref]$errors)
    Assert-A4 ($errors.Count -eq 0) (
        "PowerShell parse errors in $Path`: " +
        (($errors | ForEach-Object Message) -join '; '))
}

foreach ($required in @($runner, $monitor, $manifest, $summarizer)) {
    Assert-A4 (Test-Path -LiteralPath $required) "missing file: $required"
}
Assert-Parses $runner
Assert-Parses $monitor
Assert-Parses $summarizer

$source = Get-Content -LiteralPath $runner -Raw
$manifestLines = @(
    Get-Content -LiteralPath $manifest |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)
$manifestNames = @{}
foreach ($line in $manifestLines) {
    Assert-A4 ($line -match '^([A-Z0-9_]+)=(.*)$') (
        "invalid manifest line: $line")
    Assert-A4 (-not $manifestNames.ContainsKey($matches[1])) (
        "duplicate manifest variable: $($matches[1])")
    $manifestNames[$matches[1]] = $matches[2]
}

Assert-A4 ($manifestLines.Count -eq 47) 'base manifest must have 47 entries'
Assert-A4 (
    [string]$manifestNames['DS4_MODEL_SHA256'] -eq
        'efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668'
) 'main-model receipt hash drift'

$requiredLiterals = @(
    "[ValidateSet('Control', 'Batch2')]"
    'DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf'
    "'AFD481EE689DCE9037F70F39085FCDAE5A5B096D521CDAD43B19FA52BF8F4083'"
    '$expectedMtpBytes = [UInt64]3807602400'
    '$mtpDraft = if ($MtpMode -eq ''Batch2'') { 2 } else { 1 }'
    '$mtpBatchVerify = $MtpMode -eq ''Batch2'''
    "throw 'A4 performance isolation requires -TraceMode Off'"
    "'--mtp', `$mtp"
    "'--mtp-draft', [string]`$mtpDraft"
    "'--mtp-margin', '3'"
    "'DS4_MTP_STRICT=1'"
    "'DS4_MTP_BATCH_VERIFY=1'"
    "'DS4_MTP_BATCH_VERIFY=<UNSET>'"
    "'DS4_MTP_SPEC_DISABLE=<UNSET>'"
    "'DS4_MTP_TIMING=1'"
    "'DS4_MTP_SPEC_LOG=1'"
    '$env:DS4_MTP_STRICT = ''1'''
    '$env:DS4_MTP_BATCH_VERIFY = ''1'''
    "temperature = 0"
    "think = `$false"
    "seed = 12345"
    "'DS4_METAL_GRAPH_TOKEN_PROFILE=<UNSET>'"
    "'DS4_REQUEST_PHASE_TRACE=1'"
    'MTP support model loaded:'
    'mtp timing micro drafted=(\d+) committed=(\d+)'
    'mtp timing decode2 drafted=(\d+) committed=(\d+)'
    'mtp timing margin-skip drafted=(\d+) committed=(\d+)'
    'mtp timing seq drafted=(\d+) verified=(\d+)'
    'falling back to sequential'
    '$mtpAcceptanceRate'
    '$mtpCycleReconciliation'
    '$mtpUnloggedTerminalCycles'
    '$mtpMetricVisibility'
    '$sourceTreeMatchesBinaryReceipt'
    'source_tree_matches_binary_receipt='
    'ds4_a4_control_reference_v1'
    'ds4_a4_batch2_engagement_v1'
    'non-diagnostic Batch2 runtime requires -EngagementReceiptPath'
    '$correctnessGate'
    '$routeCallsPerAccepted'
    '$h2dSubmissionsPerAccepted'
    '$shutdownMode -eq ''graceful_http_verified'''
    "'temperature=0'"
    "'think=false'"
    "'seed=12345'"
    'server_started=no'
)
foreach ($literal in $requiredLiterals) {
    Assert-A4 ($source.Contains($literal)) "required marker absent: $literal"
}

$validateIndex = $source.IndexOf('if ($ValidateOnly)')
$firstStartIndex = $source.IndexOf(
    '$server = Start-Process -FilePath $launchFile')
Assert-A4 ($validateIndex -ge 0) 'ValidateOnly branch absent'
Assert-A4 ($firstStartIndex -gt $validateIndex) (
    'ValidateOnly must exit before every Start-Process')
Assert-A4 (
    $source.IndexOf("exit 0", $validateIndex) -lt $firstStartIndex
) 'ValidateOnly exit must precede Start-Process'

Assert-A4 (-not $source.Contains('temperature = 0.7')) (
    'non-greedy temperature residue')
Assert-A4 (-not $source.Contains('$expectedTurn1ContentHash')) (
    'stale turn-1 output oracle')
Assert-A4 (-not $source.Contains('$expectedTurn2ContentHash')) (
    'stale turn-2 output oracle')
Assert-A4 (-not $source.Contains("'--mtp-draft', '1'")) (
    'hard-coded draft count')
Assert-A4 (-not $source.Contains('declared_parameter_count=51')) (
    'stale baseline parameter count')

$summarySource = Get-Content -LiteralPath $summarizer -Raw
foreach ($marker in @(
    'VarianceThresholdPercent = 10.0',
    'A4 summary requires at least two Control and two Batch2 results',
    '"classification=$classification"',
    'NO_CONFIRMED_GAIN',
    'INCONCLUSIVE_UNSTABLE',
    'route_calls_per_accepted_token',
    'h2d_submissions_per_accepted_token'
)) {
    Assert-A4 ($summarySource.Contains($marker)) (
        "summarizer marker absent: $marker")
}

$requestBlocks = [regex]::Matches(
    $source,
    '(?s)\$turn([12]) = \[ordered\]@\{.*?\$turn\1Body')
Assert-A4 ($requestBlocks.Count -eq 2) (
    "expected two request bodies; got $($requestBlocks.Count)")
foreach ($requestBlock in @($requestBlocks)) {
    Assert-A4 ($requestBlock.Value -match '(?m)^\s*temperature = 0\s*$') (
        'request body is not greedy')
    Assert-A4 ($requestBlock.Value -match '(?m)^\s*think = \$false\s*$') (
        'request body does not set think=false')
    Assert-A4 ($requestBlock.Value -match '(?m)^\s*seed = 12345\s*$') (
        'request body does not pin seed=12345')
}

$batchSetIndex = $source.IndexOf('$env:DS4_MTP_BATCH_VERIFY = ''1''')
$batchGuardIndex = $source.LastIndexOf(
    'if ($mtpBatchVerify)', $batchSetIndex)
Assert-A4 ($batchGuardIndex -ge 0 -and $batchGuardIndex -lt $batchSetIndex) (
    'batch-verify runtime assignment is not conditionally guarded')

$serverStartIndex = $firstStartIndex
Assert-A4 ($serverStartIndex -gt $validateIndex) (
    'DS4 Start-Process not protected by ValidateOnly')

Write-Output (
    'A4_STATIC_PASS parser=PASS manifest_count=47 modes=Control|Batch2 ' +
    'greedy_requests=2 think_false_requests=2 fixed_seed_requests=2 ' +
    'validate_before_start=PASS')
