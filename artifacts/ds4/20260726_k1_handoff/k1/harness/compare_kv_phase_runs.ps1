param(
    [Parameter(Mandatory = $true)]
    [string]$K0Result,
    [Parameter(Mandatory = $true)]
    [string]$K1Result,
    [string]$OutputPath = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant

function Read-Receipt {
    param([string]$Path)
    $receipt = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        $split = $line.IndexOf('=')
        if ($split -le 0) { continue }
        $receipt[$line.Substring(0, $split)] =
            $line.Substring($split + 1)
    }
    return $receipt
}

function Require {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Number {
    param([hashtable]$Receipt, [string]$Name)
    Require $Receipt.ContainsKey($Name) "missing field $Name"
    return [double]::Parse(
        $Receipt[$Name],
        [System.Globalization.NumberStyles]::Float,
        $invariant)
}

function Relative-Delta {
    param([double]$Candidate, [double]$Baseline)
    Require ($Baseline -gt 0.0) 'baseline metric must be positive'
    return ($Candidate - $Baseline) / $Baseline
}

$k0 = Read-Receipt $K0Result
$k1 = Read-Receipt $K1Result
Require ($k0['gate_pass'] -eq 'True') 'K0 standalone gate failed'
Require ($k1['gate_pass'] -eq 'True') 'K1 standalone gate failed'
Require ($k0['kv_lifecycle'] -eq 'Off') 'K0 must have lifecycle Off'
Require ($k1['kv_lifecycle'] -eq 'On') 'K1 must have lifecycle On'
Require ($k0['checksum_mode'] -eq $k1['checksum_mode']) `
    'K0/K1 checksum modes differ'

foreach ($field in @(
    'turn1_content_sha256',
    'turn2_content_sha256')) {
    Require ($k0[$field] -eq $k1[$field]) "$field mismatch"
}
Require ((Number $k0 'kv_migration_count') -ge 2) `
    'K0 did not show the legacy next-turn migration'
Require ((Number $k0 'kv_restore_count') -ge 2) `
    'K0 did not show the legacy request-end restore'
Require ((Number $k1 'kv_migration_count') -eq 1) `
    'K1 did not eliminate the next-turn full migration'
Require ((Number $k1 'kv_retain_count') -ge 2) `
    'K1 did not retain staged KV at request end'
Require ((Number $k1 'kv_reuse_count') -ge 1) `
    'K1 did not reuse staged KV for the short suffix'
Require ((Number $k1 'kv_fallback_count') -eq 0) `
    'K1 unexpectedly fell back'

if ($k0['checksum_mode'] -eq 'Exact') {
    Require ($k0['logits_fnv1a64'] -eq $k1['logits_fnv1a64']) `
        'exact logits checksum mismatch'
    Require ($k0['kv_live_fnv1a64'] -eq $k1['kv_live_fnv1a64']) `
        'exact live-KV checksum mismatch'
}

$ttftDelta = Relative-Delta `
    (Number $k1 'turn2_suffix_ttft_seconds') `
    (Number $k0 'turn2_suffix_ttft_seconds')
$throughputDelta = Relative-Delta `
    (Number $k1 'turn2_wall_tps') `
    (Number $k0 'turn2_wall_tps')
$performancePass = $ttftDelta -le 0.10 -and $throughputDelta -ge -0.10
Require $performancePass `
    ('performance regression exceeds 10%: suffix_ttft={0:P2} throughput={1:P2}' -f
        $ttftDelta, $throughputDelta)

$classification = if (
    [Math]::Abs($ttftDelta) -lt 0.10 -and
    [Math]::Abs($throughputDelta) -lt 0.10) {
    'variance'
} else {
    'material-improvement'
}
$lines = @(
    'pair_gate_pass=True'
    "checksum_mode=$($k0['checksum_mode'])"
    'only_intended_change=DS4_CUDA_KV_PERSISTENT_STAGED'
    "turn1_content_sha256=$($k0['turn1_content_sha256'])"
    "turn2_content_sha256=$($k0['turn2_content_sha256'])"
    "logits_fnv1a64=$($k0['logits_fnv1a64'])"
    "kv_live_fnv1a64=$($k0['kv_live_fnv1a64'])"
    ('turn2_suffix_ttft_delta={0:F6}' -f $ttftDelta)
    ('turn2_wall_tps_delta={0:F6}' -f $throughputDelta)
    "performance_classification=$classification"
)
if ($OutputPath) {
    $lines | Set-Content -LiteralPath $OutputPath -Encoding ascii
}
$lines
