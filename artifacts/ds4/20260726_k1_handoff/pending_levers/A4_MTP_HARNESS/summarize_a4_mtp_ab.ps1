param(
    [Parameter(Mandatory = $true)]
    [string[]]$ControlResults,
    [Parameter(Mandatory = $true)]
    [string[]]$Batch2Results,
    [ValidateRange(0.1, 100.0)]
    [double]$VarianceThresholdPercent = 10.0,
    [string]$OutputPath = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Read-A4Result {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "A4 result missing: $Path"
    }
    $data = [ordered]@{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2) {
            $data[$parts[0]] = $parts[1]
        }
    }
    $data['_path'] = (Resolve-Path -LiteralPath $Path).Path
    return $data
}

function Assert-A4Field {
    param(
        [System.Collections.IDictionary]$Data,
        [string]$Name,
        [string]$Expected
    )
    if (-not $Data.Contains($Name) -or
        [string]$Data[$Name] -ne $Expected) {
        throw (
            "A4 result gate failed in $($Data['_path']): " +
            "$Name expected=$Expected actual=$($Data[$Name])")
    }
}

function Get-A4Double {
    param(
        [System.Collections.IDictionary]$Data,
        [string]$Name
    )
    if (-not $Data.Contains($Name)) {
        throw "A4 metric absent in $($Data['_path']): $Name"
    }
    $value = 0.0
    if (-not [double]::TryParse(
        [string]$Data[$Name],
        [System.Globalization.NumberStyles]::Float,
        $invariant,
        [ref]$value)) {
        throw "A4 metric invalid in $($Data['_path']): $Name=$($Data[$Name])"
    }
    return $value
}

function Get-A4Median {
    param([double[]]$Values)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { throw 'cannot take median of empty array' }
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$middle]
    }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

function Get-A4SpreadPercent {
    param([double[]]$Values)
    $median = Get-A4Median $Values
    if ($median -le 0) { return [double]::PositiveInfinity }
    $minimum = ($Values | Measure-Object -Minimum).Minimum
    $maximum = ($Values | Measure-Object -Maximum).Maximum
    return 100.0 * ([double]$maximum - [double]$minimum) / $median
}

if ($ControlResults.Count -lt 2 -or $Batch2Results.Count -lt 2) {
    throw 'A4 summary requires at least two Control and two Batch2 results'
}

$controls = @($ControlResults | ForEach-Object { Read-A4Result $_ })
$candidates = @($Batch2Results | ForEach-Object { Read-A4Result $_ })

foreach ($data in $controls) {
    Assert-A4Field $data 'gate_pass' 'True'
    Assert-A4Field $data 'correctness_gate' 'True'
    Assert-A4Field $data 'mtp_engagement_gate' 'True'
    Assert-A4Field $data 'mtp_mode' 'Control'
    Assert-A4Field $data 'mtp_draft' '1'
    Assert-A4Field $data 'mtp_batch_verify' '0'
    Assert-A4Field $data 'diagnostic_mtp' '0'
    Assert-A4Field $data 'temperature' '0'
    Assert-A4Field $data 'think' 'false'
    Assert-A4Field $data 'seed' '12345'
    Assert-A4Field $data 'trace_mode' 'Off'
    Assert-A4Field $data 'shutdown_mode' 'graceful_http_verified'
}
foreach ($data in $candidates) {
    Assert-A4Field $data 'gate_pass' 'True'
    Assert-A4Field $data 'correctness_gate' 'True'
    Assert-A4Field $data 'mtp_engagement_gate' 'True'
    Assert-A4Field $data 'mtp_mode' 'Batch2'
    Assert-A4Field $data 'mtp_draft' '2'
    Assert-A4Field $data 'mtp_batch_verify' '1'
    Assert-A4Field $data 'diagnostic_mtp' '0'
    Assert-A4Field $data 'temperature' '0'
    Assert-A4Field $data 'think' 'false'
    Assert-A4Field $data 'seed' '12345'
    Assert-A4Field $data 'trace_mode' 'Off'
    Assert-A4Field $data 'shutdown_mode' 'graceful_http_verified'
    Assert-A4Field $data 'mtp_metric_visibility' 'load_only_receipt_required'
    if ([string]::IsNullOrWhiteSpace(
        [string]$data['engagement_receipt_sha256'])) {
        throw "A4 engagement receipt absent in $($data['_path'])"
    }
}

$identityFields = @(
    'exe_sha256',
    'model_sha256',
    'mtp_sha256',
    'base_manifest_sha256',
    'packed_copy',
    'batched_publish',
    'g73_host_selected',
    'route_io_qd'
)
$reference = $controls[0]
foreach ($data in @($controls) + @($candidates)) {
    foreach ($field in $identityFields) {
        Assert-A4Field $data $field ([string]$reference[$field])
    }
}

$controlTps = [double[]]@(
    $controls | ForEach-Object { Get-A4Double $_ 'accepted_output_tps' })
$batch2Tps = [double[]]@(
    $candidates | ForEach-Object { Get-A4Double $_ 'accepted_output_tps' })
$controlMedian = Get-A4Median $controlTps
$batch2Median = Get-A4Median $batch2Tps
if ($controlMedian -le 0) { throw 'A4 control throughput is not positive' }
$deltaPercent = 100.0 * ($batch2Median - $controlMedian) / $controlMedian
$controlSpread = Get-A4SpreadPercent $controlTps
$batch2Spread = Get-A4SpreadPercent $batch2Tps
$stable =
    $controlSpread -le $VarianceThresholdPercent -and
    $batch2Spread -le $VarianceThresholdPercent
$classification = if ([Math]::Abs($deltaPercent) -lt
    $VarianceThresholdPercent) {
    'variance'
} elseif ($deltaPercent -gt 0) {
    'candidate_gain'
} else {
    'candidate_regression'
}
$decision = if (-not $stable) {
    'INCONCLUSIVE_UNSTABLE'
} elseif ($classification -eq 'candidate_gain') {
    'MEASURED_GAIN'
} elseif ($classification -eq 'candidate_regression') {
    'MEASURED_REGRESSION'
} else {
    'NO_CONFIRMED_GAIN'
}

$controlRoute = [double[]]@(
    $controls |
        ForEach-Object { Get-A4Double $_ 'route_calls_per_accepted_token' })
$batch2Route = [double[]]@(
    $candidates |
        ForEach-Object { Get-A4Double $_ 'route_calls_per_accepted_token' })
$controlH2d = [double[]]@(
    $controls |
        ForEach-Object { Get-A4Double $_ 'h2d_submissions_per_accepted_token' })
$batch2H2d = [double[]]@(
    $candidates |
        ForEach-Object { Get-A4Double $_ 'h2d_submissions_per_accepted_token' })

$lines = @(
    'schema=ds4_a4_ab_summary_v1'
    "control_count=$($controls.Count)"
    "batch2_count=$($candidates.Count)"
    "variance_threshold_percent=$VarianceThresholdPercent"
    "control_median_accepted_output_tps=$([Math]::Round($controlMedian, 6))"
    "batch2_median_accepted_output_tps=$([Math]::Round($batch2Median, 6))"
    "delta_percent=$([Math]::Round($deltaPercent, 6))"
    "control_spread_percent=$([Math]::Round($controlSpread, 6))"
    "batch2_spread_percent=$([Math]::Round($batch2Spread, 6))"
    "stable=$stable"
    "classification=$classification"
    "decision=$decision"
    "control_median_route_calls_per_accepted_token=$([Math]::Round((Get-A4Median $controlRoute), 9))"
    "batch2_median_route_calls_per_accepted_token=$([Math]::Round((Get-A4Median $batch2Route), 9))"
    "control_median_h2d_submissions_per_accepted_token=$([Math]::Round((Get-A4Median $controlH2d), 9))"
    "batch2_median_h2d_submissions_per_accepted_token=$([Math]::Round((Get-A4Median $batch2H2d), 9))"
)
if ($OutputPath) {
    $lines | Set-Content -LiteralPath $OutputPath -Encoding ascii
}
$lines
