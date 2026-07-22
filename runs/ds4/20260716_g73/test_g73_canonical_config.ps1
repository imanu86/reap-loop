# G73 canonical config consistency test. No GPU, no DS4 runtime.
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$canonicalPath = Join-Path $here "G73_CANONICAL_CONFIG.json"
$repoRoot = (Resolve-Path (Join-Path $here "..\..\..")).Path
$ledgerPath = Join-Path $repoRoot "docs\EXPERIMENTS_LEDGER.md"
$sourcePath = "C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_split_fused_ab_result.json"
$runnerPath = "C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g73_split_fused_ab.ps1"

if (-not (Test-Path -LiteralPath $canonicalPath -PathType Leaf)) {
    throw "Missing canonical JSON: $canonicalPath"
}
if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "Missing G73 source result JSON: $sourcePath"
}
if (-not (Test-Path -LiteralPath $ledgerPath -PathType Leaf)) {
    throw "Missing ledger: $ledgerPath"
}
if (-not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
    throw "Missing G73 source runner: $runnerPath"
}

$canonical = Get-Content -LiteralPath $canonicalPath -Raw | ConvertFrom-Json
$source = Get-Content -LiteralPath $sourcePath -Raw | ConvertFrom-Json
$ledgerText = Get-Content -LiteralPath $ledgerPath -Raw
$ledgerFlat = ($ledgerText -replace "\s+", " ")

function Assert-Eq {
    param($Actual, $Expected, [string]$Name)
    if ($Actual -ne $Expected) {
        throw ("Mismatch {0}: actual={1} expected={2}" -f $Name, $Actual, $Expected)
    }
}

Assert-Eq $canonical.sources.windows_result.schema $source.schema "source schema"
Assert-Eq $canonical.identity.status $source.status "status"
Assert-Eq $canonical.identity.stop_reason $source.stop_reason "stop reason"
Assert-Eq $canonical.prompt.sha256 $source.prompt_sha256 "prompt sha"
Assert-Eq $canonical.prompt.expected_content_sha256 $source.expected_content_sha256 "expected content sha"
Assert-Eq $canonical.model.path $source.model "model path"
Assert-Eq $canonical.model.bytes $source.provenance.model_bytes "model bytes"
Assert-Eq $canonical.launch_contract.context $source.context "context"
Assert-Eq $canonical.launch_contract.max_tokens $source.max_tokens "max tokens"
Assert-Eq $canonical.runtime_levers.arena.expected_slots $source.expected_arena_slots "expected arena slots"
Assert-Eq $canonical.runtime_levers.expert_cache.capacity $source.expert_cache_capacity "cache capacity"
Assert-Eq $canonical.provenance.runtime_head $source.provenance.head "runtime head"
Assert-Eq $canonical.provenance.executable_sha256 $source.provenance.executable_sha256 "executable sha"
Assert-Eq $canonical.provenance.summary_runner_sha256 $source.provenance.summary_runner_sha256 "summary runner sha"
Assert-Eq ((Get-FileHash -LiteralPath $runnerPath -Algorithm SHA256).Hash.ToLowerInvariant()) $canonical.sources.windows_runner.runner_sha256 "runner file sha"
Assert-Eq $canonical.sources.windows_runner.runner_sha256 $source.provenance.summary_runner_sha256 "runner sha source"
Assert-Eq $canonical.identity.not_comparable_to_full_open $true "not comparable flag"
Assert-Eq $canonical.runtime_levers.router_semantics "request_scoped_closed" "router semantics"
Assert-Eq $canonical.identity.scope "exact 64-token cyberpunk HTML short-workload only" "scope"
Assert-Eq $canonical.identity.quality_claim $null "quality claim"
Assert-Eq $canonical.model.sha256 $null "model sha"
Assert-Eq $canonical.launch_contract.nothink $null "nothink"

foreach ($needle in @(
        "G73 used a request-scoped closed transport set after prefill",
        "4.986667 t/s result was itself full/open",
        "exact 64-token cyberpunk HTML prompt",
        "does not claim general quality or broader workload quality")) {
    if ($ledgerFlat -notmatch [regex]::Escape($needle)) {
        throw "Ledger is missing G73 canonical boundary: $needle"
    }
}

$canonRuns = @($canonical.matrix.primary_runs)
$sourcePrimary = @($source.runs | Where-Object { $source.primary_run_tags -contains $_.tag })
Assert-Eq $canonRuns.Count $sourcePrimary.Count "primary run count"

foreach ($srcRun in $sourcePrimary) {
    $canonRun = $canonRuns | Where-Object { $_.tag -eq $srcRun.tag } | Select-Object -First 1
    if ($null -eq $canonRun) { throw "Missing canonical run $($srcRun.tag)" }
    Assert-Eq $canonRun.arm $srcRun.arm "arm $($srcRun.tag)"
    Assert-Eq ([double]$canonRun.decode_tokens_per_second) ([double]$srcRun.decode_tokens_per_second) "decode $($srcRun.tag)"
    Assert-Eq ([double]$canonRun.ram_h2d_gib) ([double]$srcRun.ram_h2d_gib) "ram h2d $($srcRun.tag)"
    Assert-Eq ([int]$canonRun.snapshot_misses) ([int]$srcRun.snapshot_misses) "snapshot misses $($srcRun.tag)"
    Assert-Eq ([int]$canonRun.ssd_bytes) ([int]$srcRun.ssd_bytes) "ssd bytes $($srcRun.tag)"
    Assert-Eq ([int]$canonRun.failures) ([int]$srcRun.failures) "failures $($srcRun.tag)"
}

foreach ($srcSummary in @($source.arm_summary)) {
    $canonSummary = @($canonical.matrix.aggregates) | Where-Object { $_.arm -eq $srcSummary.arm } | Select-Object -First 1
    if ($null -eq $canonSummary) { throw "Missing canonical aggregate $($srcSummary.arm)" }
    Assert-Eq ([double]$canonSummary.decode_tokens_per_second_mean) ([double]$srcSummary.decode_tokens_per_second_mean) "aggregate mean $($srcSummary.arm)"
    Assert-Eq ([double]$canonSummary.decode_tokens_per_second_median) ([double]$srcSummary.decode_tokens_per_second_median) "aggregate median $($srcSummary.arm)"
    Assert-Eq ([int]$canonSummary.independent_processes) ([int]$srcSummary.independent_processes) "aggregate n $($srcSummary.arm)"
}

Write-Host "G73 canonical config check PASS"
