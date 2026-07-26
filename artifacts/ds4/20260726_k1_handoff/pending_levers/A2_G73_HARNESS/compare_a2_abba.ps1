param(
    [Parameter(Mandatory = $true)]
    [string]$On1,
    [Parameter(Mandatory = $true)]
    [string]$Off1,
    [Parameter(Mandatory = $true)]
    [string]$Off2,
    [Parameter(Mandatory = $true)]
    [string]$On2,
    [string]$OutputPath = (Join-Path $PWD 'a2_abba_result.txt')
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Read-Result {
    param([string]$RunDirectory)
    $resultPath = Join-Path $RunDirectory 'result.txt'
    if (-not (Test-Path -LiteralPath $resultPath)) {
        throw "missing result: $resultPath"
    }
    $result = @{}
    foreach ($line in Get-Content -LiteralPath $resultPath) {
        if ($line -match '^([^=]+)=(.*)$') {
            $result[$matches[1]] = $matches[2]
        }
    }
    return $result
}

function Read-Double {
    param(
        [hashtable]$Result,
        [string]$Name
    )
    if (-not $Result.ContainsKey($Name)) {
        throw "missing result field: $Name"
    }
    return [double]::Parse($Result[$Name], $invariant)
}

function Get-CombinedGraphTps {
    param([hashtable]$Result)
    $milliseconds =
        (Read-Double -Result $Result -Name 'turn1_graph_total_ms') +
        (Read-Double -Result $Result -Name 'turn2_graph_total_ms')
    if ($milliseconds -le 0) {
        throw 'invalid combined graph milliseconds'
    }
    return 160000.0 / $milliseconds
}

function Get-TwoMedian {
    param([double]$First, [double]$Second)
    return ($First + $Second) / 2.0
}

function Get-SpreadPercent {
    param([double]$First, [double]$Second)
    $median = Get-TwoMedian -First $First -Second $Second
    if ($median -le 0) {
        return [double]::PositiveInfinity
    }
    return 100.0 * [Math]::Abs($First - $Second) / $median
}

function Assert-SameManifestArm {
    param([string]$FirstRun, [string]$SecondRun, [string]$Arm)
    $first = @(Get-Content -LiteralPath (Join-Path $FirstRun 'manifest_47.env'))
    $second = @(Get-Content -LiteralPath (Join-Path $SecondRun 'manifest_47.env'))
    if ($first.Count -ne 47 -or $second.Count -ne 47) {
        throw "$Arm manifest is not 47/47"
    }
    for ($index = 0; $index -lt 47; $index++) {
        if ($first[$index] -cne $second[$index]) {
            throw "$Arm manifests differ at line $($index + 1)"
        }
    }
}

function Assert-OneChange {
    param([string]$OnRun, [string]$OffRun)
    $onManifest = @(Get-Content -LiteralPath (Join-Path $OnRun 'manifest_47.env'))
    $offManifest = @(Get-Content -LiteralPath (Join-Path $OffRun 'manifest_47.env'))
    $differences = @()
    for ($index = 0; $index -lt 47; $index++) {
        if ($onManifest[$index] -cne $offManifest[$index]) {
            $differences += ($index + 1)
        }
    }
    if ($differences.Count -ne 1 -or
        $differences[0] -ne 41 -or
        $onManifest[40] -cne 'DS4_G73_OPEN=1' -or
        $offManifest[40] -cne 'DS4_G73_OPEN=0') {
        throw 'A/B is not a one-change DS4_G73_OPEN comparison'
    }
}

$runPaths = @($On1, $Off1, $Off2, $On2)
$results = @(
    Read-Result -RunDirectory $On1
    Read-Result -RunDirectory $Off1
    Read-Result -RunDirectory $Off2
    Read-Result -RunDirectory $On2
)
$expectedArms = @('On', 'Off', 'Off', 'On')
for ($index = 0; $index -lt 4; $index++) {
    if ($results[$index]['g73_open'] -ne $expectedArms[$index]) {
        throw "ABBA order mismatch at index $index"
    }
}

Assert-SameManifestArm -FirstRun $On1 -SecondRun $On2 -Arm 'ON'
Assert-SameManifestArm -FirstRun $Off1 -SecondRun $Off2 -Arm 'OFF'
Assert-OneChange -OnRun $On1 -OffRun $Off1
Assert-OneChange -OnRun $On2 -OffRun $Off2

$identityFields = @(
    'binary_sha256'
    'model_sha256'
    'normalized_pair_manifest_sha256'
    'turn1_request_sha256'
    'turn2_request_sha256'
    'turn1_content_sha256'
    'turn2_content_sha256'
)
$identityGatePass = $true
foreach ($field in $identityFields) {
    $reference = $results[0][$field]
    if ([string]::IsNullOrWhiteSpace($reference)) {
        $identityGatePass = $false
        continue
    }
    foreach ($result in $results) {
        if ($result[$field] -ne $reference) {
            $identityGatePass = $false
        }
    }
}
$runGatePass = @(
    $results | Where-Object { $_['gate_pass'] -ne 'True' }
).Count -eq 0

$on1GraphTps = Get-CombinedGraphTps -Result $results[0]
$off1GraphTps = Get-CombinedGraphTps -Result $results[1]
$off2GraphTps = Get-CombinedGraphTps -Result $results[2]
$on2GraphTps = Get-CombinedGraphTps -Result $results[3]
$onMedianGraphTps = Get-TwoMedian -First $on1GraphTps -Second $on2GraphTps
$offMedianGraphTps = Get-TwoMedian -First $off1GraphTps -Second $off2GraphTps
$onSpreadPct = Get-SpreadPercent -First $on1GraphTps -Second $on2GraphTps
$offSpreadPct = Get-SpreadPercent -First $off1GraphTps -Second $off2GraphTps
$medianDeltaPct =
    100.0 * ($onMedianGraphTps - $offMedianGraphTps) / $offMedianGraphTps
$pair1DeltaPct =
    100.0 * ($on1GraphTps - $off1GraphTps) / $off1GraphTps
$pair2DeltaPct =
    100.0 * ($on2GraphTps - $off2GraphTps) / $off2GraphTps
$sameDirection =
    [Math]::Sign($pair1DeltaPct) -eq [Math]::Sign($pair2DeltaPct)

$classification = if (-not $runGatePass -or -not $identityGatePass) {
    'INVALID_GATE_OR_IDENTITY'
} elseif ($onSpreadPct -ge 10.0 -or $offSpreadPct -ge 10.0) {
    'INCONCLUSIVE_WITHIN_ARM_VARIANCE'
} elseif ([Math]::Abs($medianDeltaPct) -lt 10.0) {
    'VARIANCE_LT_10_PERCENT'
} elseif (-not $sameDirection) {
    'INCONCLUSIVE_ORDER_EFFECT'
} else {
    'CANDIDATE_G73_EFFECT'
}

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}
@(
    "classification=$classification"
    "run_gate_pass=$runGatePass"
    "identity_gate_pass=$identityGatePass"
    'sequence=ON,OFF,OFF,ON'
    "on1_combined_graph_tps=$([Math]::Round($on1GraphTps, 6))"
    "off1_combined_graph_tps=$([Math]::Round($off1GraphTps, 6))"
    "off2_combined_graph_tps=$([Math]::Round($off2GraphTps, 6))"
    "on2_combined_graph_tps=$([Math]::Round($on2GraphTps, 6))"
    "on_median_graph_tps=$([Math]::Round($onMedianGraphTps, 6))"
    "off_median_graph_tps=$([Math]::Round($offMedianGraphTps, 6))"
    "on_within_arm_spread_pct=$([Math]::Round($onSpreadPct, 6))"
    "off_within_arm_spread_pct=$([Math]::Round($offSpreadPct, 6))"
    "median_delta_on_vs_off_pct=$([Math]::Round($medianDeltaPct, 6))"
    "pair1_delta_pct=$([Math]::Round($pair1DeltaPct, 6))"
    "pair2_delta_pct=$([Math]::Round($pair2DeltaPct, 6))"
    "same_direction=$sameDirection"
    'rule_abs_delta_lt_10_percent=variance'
) | Set-Content -LiteralPath $OutputPath -Encoding ascii

Write-Output "classification=$classification output=$OutputPath"
if ($classification -eq 'INVALID_GATE_OR_IDENTITY') {
    exit 2
}
if ($classification -like 'INCONCLUSIVE_*') {
    exit 3
}
