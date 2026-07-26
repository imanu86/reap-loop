param(
    [string]$Runner = (Join-Path $PSScriptRoot 'run_kv_phase_matrix.ps1'),
    [string]$Manifest = (Join-Path $PSScriptRoot 'manifest_47.env'),
    [string]$Comparator = (Join-Path $PSScriptRoot 'compare_kv_phase_runs.ps1')
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Require-Match {
    param([string]$Text, [string]$Pattern, [string]$Message)
    if ($Text -notmatch $Pattern) {
        throw $Message
    }
}

$text = Get-Content -LiteralPath $Runner -Raw
$comparatorText = Get-Content -LiteralPath $Comparator -Raw
$manifestLines = @(
    Get-Content -LiteralPath $Manifest |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)
$manifestHash = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash

if ($manifestLines.Count -ne 47) {
    throw "expected 47-variable base manifest, got $($manifestLines.Count)"
}
if ($manifestHash -ne
    'CF1150EC0E46197E78B7F8482F8D94A8DA0666D887B291743478AA9DD1F73609') {
    throw "manifest hash drift: $manifestHash"
}

Require-Match $text `
    '\[string\]\$KvLifecycle = ''Off''' `
    'KvLifecycle must default OFF'
Require-Match $text `
    'DS4_CUDA_KV_PERSISTENT_STAGED=<UNSET>' `
    'K0 overlay must explicitly record an unset persistent mode'
Require-Match $text `
    'DS4_CUDA_KV_PERSISTENT_STAGED=1' `
    'K1 overlay must explicitly enable persistent mode'
Require-Match $text `
    '\[string\]\$ChecksumMode = ''Off''' `
    'exact checksums must default OFF for performance measurements'
Require-Match $text `
    'DS4_CUDA_KV_PHASE_VALIDATE=1' `
    'exact-checksum correctness overlay is missing'
Require-Match $text `
    'DS4_REQUEST_PHASE_TRACE=1' `
    'suffix TTFT phase trace overlay is missing'
Require-Match $text `
    '\$env:DS4_CUDA_KV_PERSISTENT_STAGED = ''1''' `
    'runtime K1 environment assignment missing'
Require-Match $text `
    'expectedManifestHash.*CF1150EC0E46197E78B7F8482F8D94A8DA0666D887B291743478AA9DD1F73609' `
    'base manifest hash is not pinned'
Require-Match $text `
    'kvLifecycleGatePass = if \(\$KvLifecycle -eq ''On''\)' `
    'K0/K1 lifecycle gate is missing'
Require-Match $text `
    'checksumGatePass = if \(\$ChecksumMode -eq ''Exact''\)' `
    'exact logits/KV checksum gate is missing'
Require-Match $text `
    'turn2_suffix_ttft_seconds=' `
    'suffix TTFT receipt field is missing'
Require-Match $text `
    'turn1ContentHash -eq \$expectedTurn1ContentHash' `
    'turn-1 exact output hash gate is missing'
Require-Match $text `
    'turn2ContentHash -eq \$expectedTurn2ContentHash' `
    'turn-2 exact output hash gate is missing'
Require-Match $text `
    'live_position_150000=not_claimed' `
    'capacity/live-position distinction is missing'

$validateIndex = $text.IndexOf('if ($ValidateOnly)')
$launchIndex = $text.IndexOf('Start-Process -FilePath $launchFile')
if ($validateIndex -lt 0 -or $launchIndex -lt 0 -or
    $validateIndex -ge $launchIndex) {
    throw 'ValidateOnly must exit before process launch'
}
Require-Match $comparatorText `
    'exact logits checksum mismatch' `
    'pair comparator does not enforce exact logits equality'
Require-Match $comparatorText `
    'exact live-KV checksum mismatch' `
    'pair comparator does not enforce exact live-KV equality'
Require-Match $comparatorText `
    'performance regression exceeds 10%' `
    'pair comparator does not enforce the 10 percent regression gate'

Write-Output 'PASS test_kv_phase_harness_static'
