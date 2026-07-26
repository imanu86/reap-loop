param(
    [ValidateSet('Control', 'Batch2')]
    [string]$MtpMode = 'Control',
    [switch]$DiagnosticMtp,
    [ValidateSet('Off', 'Sampled', 'Nsys')]
    [string]$TraceMode = 'Off',
    [ValidateSet('Off', 'On')]
    [string]$PackedCopy = 'Off',
    [ValidateSet('Off', 'On')]
    [string]$BatchedPublish = 'Off',
    [ValidateSet('Off', 'On')]
    [string]$G73HostSelected = 'Off',
    [ValidateSet('Off', 'Qd4')]
    [string]$RouteIoQd = 'Off',
    [ValidateRange(1, 1024)]
    [int]$Turn1MaxTokens = 128,
    [ValidateRange(1, 1024)]
    [int]$Turn2MaxTokens = 32,
    [string]$SourceRoot =
        'C:\Users\imanu\Documents\Codex\2026-07-25\legg\work\wt-hot-reserve',
    [string]$ExecutablePath = '',
    [string]$PrimaryModelPath = 'C:\ds4-models\ds4-2bit.gguf',
    [string]$MtpPath =
        'C:\ds4-models\DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf',
    [string]$BaseManifestPath = '',
    [string]$MonitorScriptPath = '',
    [string]$OutputRoot = '',
    [string]$ReferenceReceiptPath = '',
    [string]$EngagementReceiptPath = '',
    [switch]$ValidateOnly
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $invariant

# Windows PowerShell can expose both Path/PATH to Start-Process. Normalize the
# runner-local environment block before starting either DS4 or the monitor.
$savedProcessPath = [Environment]::GetEnvironmentVariable('Path', 'Process')
Remove-Item Env:PATH -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable('Path', $savedProcessPath, 'Process')

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceRoot = $SourceRoot
$manifestSource = if ($BaseManifestPath) {
    $BaseManifestPath
} else {
    Join-Path $root 'manifest_47.env'
}
$monitorScript = if ($MonitorScriptPath) {
    $MonitorScriptPath
} else {
    Join-Path $root 'runtime_monitor.ps1'
}
$nsysStreamExtractor = Join-Path $root 'extract_nsys_process_streams.py'
$python = 'C:\Users\imanu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$exe = if ($ExecutablePath) {
    $ExecutablePath
} else {
    Join-Path $sourceRoot 'build2\ds4_server.exe'
}
$model = $PrimaryModelPath
$mtp = $MtpPath
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe'
$outputRoot = if ($OutputRoot) {
    $OutputRoot
} else {
    Join-Path $root 'runs'
}
$mtpDraft = if ($MtpMode -eq 'Batch2') { 2 } else { 1 }
$mtpBatchVerify = $MtpMode -eq 'Batch2'
if ($DiagnosticMtp -and $MtpMode -ne 'Batch2') {
    throw '-DiagnosticMtp is valid only with -MtpMode Batch2'
}
if ($TraceMode -ne 'Off') {
    throw 'A4 performance isolation requires -TraceMode Off'
}
if ($EngagementReceiptPath -and
    ($MtpMode -ne 'Batch2' -or $DiagnosticMtp)) {
    throw '-EngagementReceiptPath is valid only with non-diagnostic Batch2'
}
$safeTrace = $TraceMode.ToLowerInvariant()
$safePacked = $PackedCopy.ToLowerInvariant()
$safePublish = $BatchedPublish.ToLowerInvariant()
$safeHostSelected = $G73HostSelected.ToLowerInvariant()
$safeRouteIo = $RouteIoQd.ToLowerInvariant()
$safeMtpMode = $MtpMode.ToLowerInvariant()
$runId = '{0}_a4_mtp-{1}_diag-{2}_trace-{3}_packed-{4}_publish-{5}_hostsel-{6}_routeio-{7}' -f (
    Get-Date -Format 'yyyyMMdd_HHmmss'), $safeMtpMode,
    ([int][bool]$DiagnosticMtp), $safeTrace, $safePacked, $safePublish,
    $safeHostSelected, $safeRouteIo
$runDirectory = Join-Path $outputRoot $runId

$expectedExeHash = 'F2F6A25600EF0B6728BE0DAE77ACFC5CE89ABCEB1446555A7DB946DA26F11DFF'
$expectedModelHash = 'efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668'
$expectedMtpHash = 'AFD481EE689DCE9037F70F39085FCDAE5A5B096D521CDAD43B19FA52BF8F4083'
$expectedMtpBytes = [UInt64]3807602400
$expectedSourceHashes = [ordered]@{
    'ds4.c' = '2C2840FCA3F1FF7B0C745BF376E04BCB28AED248FDC6DE504093D949288415A4'
    'ds4_cuda.cu' = 'FEB2954ED2FA583C0631C4FE763B75FFA1F4D0D51598D2ED2AC6F5A87E303496'
    'ds4_gpu.h' = 'EB51FE6F3250CE023803AFF77B407CE73BBE004F212F33E2E68638969E9A8E06'
    'ds4_server.c' = 'CD0A13F839775D1FCB51B8FFE41C4B84673BBE04C0E9892C3862BF00D099273E'
    'ds4_cli.c' = 'C7E6EF486A37E7D46F2ADD0C76DA0CE3628E988BE4DB051AD06B420B13FA8C9E'
}

New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
[System.IO.File]::WriteAllText(
    (Join-Path $outputRoot 'current_run.txt'),
    $runDirectory,
    [System.Text.UTF8Encoding]::new($false))
$statusPath = Join-Path $runDirectory 'runner.status.log'
$resultPath = Join-Path $runDirectory 'result.txt'
$serverLog = Join-Path $runDirectory 'server.stderr.log'
$serverStdout = Join-Path $runDirectory 'server.stdout.log'
$monitorCsv = Join-Path $runDirectory 'monitor.csv'
$monitorStop = Join-Path $runDirectory 'monitor.stop'
$monitorOut = Join-Path $runDirectory 'monitor.stdout.log'
$monitorErr = Join-Path $runDirectory 'monitor.stderr.log'
$manifestPath = Join-Path $runDirectory 'manifest_47.env'
$instrumentationPath = Join-Path $runDirectory 'instrumentation_manifest.txt'
$experimentPath = Join-Path $runDirectory 'experiment_manifest.txt'
$mtpOverlayPath = Join-Path $runDirectory 'mtp_overlay.env'
$controlPath = Join-Path $runDirectory 'control_manifest.txt'
$provenancePath = Join-Path $runDirectory 'provenance.txt'
$controlReferencePath = Join-Path $runDirectory 'a4_control_reference.json'
$batch2EngagementPath = Join-Path $runDirectory 'a4_batch2_engagement.json'
$serverArgsPath = Join-Path $runDirectory 'server_arguments.txt'
$launcherArgsPath = Join-Path $runDirectory 'launcher_arguments.txt'
$nsysLauncherStdout = Join-Path $runDirectory 'nsys.stdout.log'
$nsysLauncherStderr = Join-Path $runDirectory 'nsys.stderr.log'
$turn1RequestPath = Join-Path $runDirectory 'turn1.request.json'
$turn1ResponsePath = Join-Path $runDirectory 'turn1.response.json'
$turn1LogPath = Join-Path $runDirectory 'turn1.server.log'
$turn2RequestPath = Join-Path $runDirectory 'turn2.request.json'
$turn2ResponsePath = Join-Path $runDirectory 'turn2.response.json'
$turn2LogPath = Join-Path $runDirectory 'turn2.server.log'
$shutdownReceiptPath = Join-Path $runDirectory 'shutdown.receipt.json'
$nsysOutputBase = Join-Path $runDirectory 'decode_position_100'
$nsysReportPath = "$nsysOutputBase.nsys-rep"
$nsysSqlitePath = "$nsysOutputBase.sqlite"

function Write-Status {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date).ToString('o'), $Message
    Add-Content -LiteralPath $statusPath -Value $line -Encoding ascii
    Write-Output $line
}

function Get-TextSha256 {
    param([string]$Text)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
        return ([BitConverter]::ToString(
            $sha256.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha256.Dispose()
    }
}

function Get-LogLineCount {
    if (-not (Test-Path -LiteralPath $serverLog)) {
        return 0
    }
    return @(Get-Content -LiteralPath $serverLog).Count
}

function Get-Ds4ListenerPid {
    $netstatLines = @(& netstat.exe -ano -p tcp 2>$null)
    foreach ($line in $netstatLines) {
        if ($line -match (
            '^\s*TCP\s+127\.0\.0\.1:8000\s+\S+\s+LISTENING\s+([0-9]+)\s*$')) {
            return [int]$matches[1]
        }
    }
    return 0
}

function Save-LogSlice {
    param(
        [int]$StartLine,
        [int]$EndLine,
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $serverLog) -or
        $EndLine -le $StartLine) {
        Set-Content -LiteralPath $Path -Value '' -Encoding utf8
        return
    }
    Get-Content -LiteralPath $serverLog |
        Select-Object -Skip $StartLine -First ($EndLine - $StartLine) |
        Set-Content -LiteralPath $Path -Encoding utf8
}

function Invoke-Ds4Request {
    param(
        [string]$RequestPath,
        [string]$ResponsePath
    )
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $httpCode = & curl.exe -sS --max-time 7200 `
        -X POST 'http://127.0.0.1:8000/v1/chat/completions' `
        -H 'Content-Type: application/json' `
        --data-binary "@$RequestPath" `
        -o $ResponsePath `
        -w '%{http_code}'
    $curlStatus = $LASTEXITCODE
    $stopwatch.Stop()
    if ($curlStatus -ne 0 -or [string]$httpCode -ne '200') {
        throw "request failed: curl=$curlStatus http=$httpCode"
    }
    [pscustomobject]@{
        elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
        http_code = [string]$httpCode
        curl_status = $curlStatus
    }
}

Write-Status "prepared run_dir=$runDirectory"

$manifestLines = @(
    Get-Content -LiteralPath $manifestSource |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)
if ($manifestLines.Count -ne 47) {
    throw "manifest source must contain exactly 47 variables; got $($manifestLines.Count)"
}
$manifestNames = @{}
foreach ($line in $manifestLines) {
    if ($line -notmatch '^([A-Z0-9_]+)=(.*)$') {
        throw "invalid manifest line: $line"
    }
    if ($manifestNames.ContainsKey($matches[1])) {
        throw "duplicate manifest variable: $($matches[1])"
    }
    $manifestNames[$matches[1]] = $matches[2]
}
if ([string]$manifestNames['DS4_MODEL_SHA256'] -ne $expectedModelHash) {
    throw 'manifest model hash drift'
}
foreach ($requiredPath in @($sourceRoot, $manifestSource, $monitorScript, $exe, $model, $mtp)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "required A4 input missing: $requiredPath"
    }
}
$actualManifestHash =
    (Get-FileHash -LiteralPath $manifestSource -Algorithm SHA256).Hash

$actualExeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
if ($actualExeHash -ne $expectedExeHash) {
    throw "binary hash drift: expected=$expectedExeHash actual=$actualExeHash"
}
$actualMtpBytes = [UInt64](Get-Item -LiteralPath $mtp).Length
if ($actualMtpBytes -ne $expectedMtpBytes) {
    throw "MTP byte-size drift: expected=$expectedMtpBytes actual=$actualMtpBytes"
}
$actualMtpHash = (Get-FileHash -LiteralPath $mtp -Algorithm SHA256).Hash
if ($actualMtpHash -ne $expectedMtpHash) {
    throw "MTP hash drift: expected=$expectedMtpHash actual=$actualMtpHash"
}
$sourceReceipt = @()
$sourceTreeMatchesBinaryReceipt = $true
foreach ($entry in $expectedSourceHashes.GetEnumerator()) {
    $path = Join-Path $sourceRoot $entry.Key
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $entry.Value) {
        $sourceTreeMatchesBinaryReceipt = $false
    }
    $sourceReceipt +=
        "$($entry.Key)_binary_receipt_SHA256=$($entry.Value)"
    $sourceReceipt +=
        "$($entry.Key)_working_tree_SHA256=$actual"
}
if ((Get-Item -LiteralPath $model).Length -ne [UInt64]86720111488) {
    throw 'model byte-size drift'
}

$referenceReceipt = $null
$referenceReceiptSha256 = ''
if ($ReferenceReceiptPath) {
    if (-not (Test-Path -LiteralPath $ReferenceReceiptPath)) {
        throw "A4 reference receipt missing: $ReferenceReceiptPath"
    }
    $referenceReceipt = Get-Content -LiteralPath $ReferenceReceiptPath -Raw |
        ConvertFrom-Json
    $referenceReceiptSha256 =
        (Get-FileHash -LiteralPath $ReferenceReceiptPath -Algorithm SHA256).Hash
    if (
        [string]$referenceReceipt.schema -ne 'ds4_a4_control_reference_v1' -or
        [string]$referenceReceipt.mode -ne 'Control' -or
        [string]$referenceReceipt.exe_sha256 -ne $actualExeHash -or
        [string]$referenceReceipt.model_sha256 -ne $expectedModelHash -or
        [string]$referenceReceipt.mtp_sha256 -ne $actualMtpHash -or
        [string]$referenceReceipt.base_manifest_sha256 -ne
            $actualManifestHash -or
        [double]$referenceReceipt.temperature -ne 0 -or
        [bool]$referenceReceipt.think -ne $false -or
        [UInt64]$referenceReceipt.seed -ne 12345 -or
        [int]$referenceReceipt.turn1_max_tokens -ne $Turn1MaxTokens -or
        [int]$referenceReceipt.turn2_max_tokens -ne $Turn2MaxTokens -or
        [string]$referenceReceipt.trace_mode -ne 'Off' -or
        [string]$referenceReceipt.packed_copy -ne $PackedCopy -or
        [string]$referenceReceipt.batched_publish -ne $BatchedPublish -or
        [string]$referenceReceipt.g73_host_selected -ne
            $G73HostSelected -or
        [string]$referenceReceipt.route_io_qd -ne $RouteIoQd -or
        [int]$referenceReceipt.mtp_margin -ne 3 -or
        [bool]$referenceReceipt.mtp_strict -ne $true
    ) {
        throw 'A4 reference receipt identity/config drift'
    }
}
if ($MtpMode -eq 'Batch2' -and -not $ValidateOnly -and -not $referenceReceipt) {
    throw 'Batch2 runtime requires -ReferenceReceiptPath from a valid Control run'
}

$engagementReceipt = $null
$engagementReceiptSha256 = ''
if ($EngagementReceiptPath) {
    if (-not (Test-Path -LiteralPath $EngagementReceiptPath)) {
        throw "A4 engagement receipt missing: $EngagementReceiptPath"
    }
    $engagementReceipt = Get-Content -LiteralPath $EngagementReceiptPath -Raw |
        ConvertFrom-Json
    $engagementReceiptSha256 =
        (Get-FileHash -LiteralPath $EngagementReceiptPath -Algorithm SHA256).Hash
    if (
        [string]$engagementReceipt.schema -ne
            'ds4_a4_batch2_engagement_v1' -or
        [string]$engagementReceipt.mode -ne 'Batch2' -or
        [bool]$engagementReceipt.diagnostic_mtp -ne $true -or
        [string]$engagementReceipt.exe_sha256 -ne $actualExeHash -or
        [string]$engagementReceipt.model_sha256 -ne $expectedModelHash -or
        [string]$engagementReceipt.mtp_sha256 -ne $actualMtpHash -or
        [string]$engagementReceipt.base_manifest_sha256 -ne
            $actualManifestHash -or
        [string]$engagementReceipt.control_reference_sha256 -ne
            $referenceReceiptSha256 -or
        [string]$engagementReceipt.trace_mode -ne 'Off' -or
        [string]$engagementReceipt.packed_copy -ne $PackedCopy -or
        [string]$engagementReceipt.batched_publish -ne $BatchedPublish -or
        [string]$engagementReceipt.g73_host_selected -ne
            $G73HostSelected -or
        [string]$engagementReceipt.route_io_qd -ne $RouteIoQd -or
        [int]$engagementReceipt.mtp_draft -ne 2 -or
        [int]$engagementReceipt.mtp_margin -ne 3 -or
        [bool]$engagementReceipt.mtp_strict -ne $true -or
        [bool]$engagementReceipt.mtp_batch_verify -ne $true -or
        [bool]$engagementReceipt.engagement_gate -ne $true -or
        [bool]$engagementReceipt.cycle_reconciliation -ne $true -or
        [Int64]$engagementReceipt.fallback_count -ne 0
    ) {
        throw 'A4 engagement receipt identity/config/gate drift'
    }
}
if (
    $MtpMode -eq 'Batch2' -and
    -not $DiagnosticMtp -and
    -not $ValidateOnly -and
    -not $engagementReceipt
) {
    throw ('non-diagnostic Batch2 runtime requires -EngagementReceiptPath ' +
        'from a passing diagnostic Batch2 run')
}

$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding ascii
$traceManifest = @(
    "TraceMode=$TraceMode"
    'DS4_METAL_GRAPH_TOKEN_PROFILE=<UNSET>'
    'DS4_REQUEST_PHASE_TRACE=1'
    'DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY=<UNSET>'
    'DS4_CUDA_DECODE_TRACE_POSITIONS=<UNSET>'
    'DS4_CUDA_DECODE_TRACE_LAYERS=<UNSET>'
)
if ($TraceMode -eq 'Sampled') {
    $traceManifest = @(
        "TraceMode=$TraceMode"
        'DS4_METAL_GRAPH_TOKEN_PROFILE=<UNSET>'
        'DS4_REQUEST_PHASE_TRACE=1'
        'DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY=<UNSET>'
        'DS4_CUDA_DECODE_TRACE_POSITIONS=13,141'
        'DS4_CUDA_DECODE_TRACE_LAYERS=0,4,20,42'
    )
}
if ($TraceMode -eq 'Nsys') {
    $traceManifest = @(
        "TraceMode=$TraceMode"
        'DS4_METAL_GRAPH_TOKEN_PROFILE=<UNSET>'
        'DS4_REQUEST_PHASE_TRACE=1'
        'DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY=<UNSET>'
        'DS4_CUDA_DECODE_TRACE_POSITIONS=100'
        'DS4_CUDA_DECODE_TRACE_LAYERS=0'
        'DS4_CUDA_NSYS_CAPTURE=1'
        'nsys_capture_range=cudaProfilerApi'
        'nsys_capture_range_end=stop'
        'nsys_trace=cuda'
        'nsys_sample=none'
        'nsys_cpuctxsw=none'
        'nsys_cuda_memory_usage=false'
        'nsys_wait=all'
        'nsys_show_output=true'
        'ds4_log_extraction=SQLite_ProcessStreams_after_exit'
        'nsys_gpu_metrics=unavailable_ERR_NVGPUCTRPERM'
    )
}
$traceManifest | Set-Content -LiteralPath $instrumentationPath -Encoding ascii
@(
    "PackedCopy=$PackedCopy"
    $(if ($PackedCopy -eq 'On') {
        'DS4_CUDA_MOE_ROUTE_PACKED_COPY=1'
    } else {
        'DS4_CUDA_MOE_ROUTE_PACKED_COPY=<UNSET>'
    })
    "BatchedPublish=$BatchedPublish"
    $(if ($BatchedPublish -eq 'On') {
        'DS4_CUDA_MOE_ROUTE_BATCHED_PUBLISH=1'
    } else {
        'DS4_CUDA_MOE_ROUTE_BATCHED_PUBLISH=<UNSET>'
    })
    "G73HostSelected=$G73HostSelected"
    $(if ($G73HostSelected -eq 'On') {
        'DS4_CUDA_G73_REUSE_HOST_SELECTED=1'
    } else {
        'DS4_CUDA_G73_REUSE_HOST_SELECTED=<UNSET>'
    })
    "RouteIoQd=$RouteIoQd"
    $(if ($RouteIoQd -eq 'Qd4') {
        'DS4_CUDA_G73_ROUTE_IO_QD=4'
    } else {
        'DS4_CUDA_G73_ROUTE_IO_QD=<UNSET>'
    })
    "MtpMode=$MtpMode"
    "mtp_path=$mtp"
    "mtp_sha256=$actualMtpHash"
    "mtp_draft=$mtpDraft"
    'mtp_margin=3'
    'temperature=0'
    'think=false'
) | Set-Content -LiteralPath $experimentPath -Encoding ascii
@(
    'DS4_MTP_STRICT=1'
    $(if ($mtpBatchVerify) {
        'DS4_MTP_BATCH_VERIFY=1'
    } else {
        'DS4_MTP_BATCH_VERIFY=<UNSET>'
    })
    'DS4_MTP_SPEC_DISABLE=<UNSET>'
    'DS4_MTP_PROBE=<UNSET>'
    'DS4_MTP_CONF_LOG=<UNSET>'
    'DS4_MTP_FULL_LOGITS=<UNSET>'
    'DS4_MTP_MIN_MARGIN=<UNSET>'
    'DS4_MTP_CAPTURE_PREFIX1=<UNSET>'
    'DS4_MTP_EXACT_REPLAY=<UNSET>'
    'DS4_MTP_FORCE_SNAPSHOT=<UNSET>'
    $(if ($DiagnosticMtp) {
        'DS4_MTP_TIMING=1'
    } else {
        'DS4_MTP_TIMING=<UNSET>'
    })
    $(if ($DiagnosticMtp) {
        'DS4_MTP_SPEC_LOG=1'
    } else {
        'DS4_MTP_SPEC_LOG=<UNSET>'
    })
) | Set-Content -LiteralPath $mtpOverlayPath -Encoding ascii

$serverArguments = @(
    '-m', $model,
    '--mtp', $mtp,
    '--cuda',
    '-c', '150000',
    '-n', '8192',
    '--mtp-draft', [string]$mtpDraft,
    '--mtp-margin', '3',
    '--host', '127.0.0.1',
    '--port', '8000'
)
$serverArguments | Set-Content -LiteralPath $serverArgsPath -Encoding ascii
@(
    "run_id=$runId"
    "run_directory=$runDirectory"
    "git_head=f64cc89baf5ba890c11317b8a0283e25ace85eae"
    $sourceReceipt
    "ds4_server.exe_SHA256=$actualExeHash"
    "ds4_server.exe_bytes=$((Get-Item -LiteralPath $exe).Length)"
    "source_tree_matches_binary_receipt=$([int]$sourceTreeMatchesBinaryReceipt)"
    "base_manifest_SHA256=$actualManifestHash"
    "model_path=$model"
    "model_bytes=$((Get-Item -LiteralPath $model).Length)"
    "model_SHA256_receipt=$expectedModelHash"
    "mtp_path=$mtp"
    "mtp_bytes=$actualMtpBytes"
    "mtp_SHA256=$actualMtpHash"
    'ctx_capacity=150000'
    "mtp_mode=$MtpMode"
    "mtp_draft=$mtpDraft"
    'mtp_margin=3'
    'DS4_MTP_STRICT=1'
    "DS4_MTP_BATCH_VERIFY=$(if ($mtpBatchVerify) { '1' } else { '<UNSET>' })"
    "diagnostic_mtp=$([int][bool]$DiagnosticMtp)"
    "reference_receipt_sha256=$referenceReceiptSha256"
    "engagement_receipt_sha256=$engagementReceiptSha256"
    'DS4_METAL_GRAPH_TOKEN_PROFILE=<UNSET>'
    'DS4_REQUEST_PHASE_TRACE=1'
    "route_packed_copy=$PackedCopy"
    "route_batched_publish=$BatchedPublish"
    "g73_host_selected=$G73HostSelected"
    "g73_route_io_qd=$RouteIoQd"
    'temperature=0'
    'seed=12345'
    'think=false'
    "turn1_max_tokens=$Turn1MaxTokens"
    "turn2_max_tokens=$Turn2MaxTokens"
    'turn1_prompt_utf8=Ciao, sai fare un bel sito?'
    'turn2_prompt_utf8=Fammi una landing page minimal, single-file HTML, molto breve.'
) | Set-Content -LiteralPath $provenancePath -Encoding utf8

if ($ValidateOnly) {
    Write-Status "validation=PASS mtp_mode=$MtpMode mtp_draft=$mtpDraft mtp_batch_verify=$([int]$mtpBatchVerify) temperature=0 think=false trace_mode=Off manifest_count=47 packed_copy=$PackedCopy batched_publish=$BatchedPublish g73_host_selected=$G73HostSelected route_io_qd=$RouteIoQd binary_mtp_hashes=PASS source_tree_match=$([int]$sourceTreeMatchesBinaryReceipt) server_started=no"
    "run_dir=$runDirectory`nvalidation=PASS`nmtp_mode=$MtpMode`nmtp_path=$mtp`nmtp_sha256=$actualMtpHash`nmtp_draft=$mtpDraft`nmtp_batch_verify=$([int]$mtpBatchVerify)`ntemperature=0`nthink=false`ntrace_mode=Off`nmanifest_count=47`nbase_manifest_sha256=$actualManifestHash`nsource_tree_matches_binary_receipt=$([int]$sourceTreeMatchesBinaryReceipt)`npacked_copy=$PackedCopy`nbatched_publish=$BatchedPublish`ng73_host_selected=$G73HostSelected`nroute_io_qd=$RouteIoQd`nserver_started=no" |
        Set-Content -LiteralPath $resultPath -Encoding ascii
    exit 0
}

$existingServers = @(Get-Process ds4_server -ErrorAction SilentlyContinue)
$listenerPid = Get-Ds4ListenerPid
if ($existingServers.Count -ne 0 -or $listenerPid -gt 0) {
    throw "dirty preflight: ds4_server=$($existingServers.Count) port8000_pid=$listenerPid"
}

Add-Type -AssemblyName Microsoft.VisualBasic
$memoryInfo = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
$availableGiB = $memoryInfo.AvailablePhysicalMemory / 1GB
$gpuText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null |
    Select-Object -First 1
if ($gpuText -notmatch '^\s*([0-9]+)') {
    throw 'cannot read GPU memory usage'
}
$gpuUsedMiB = [int]$matches[1]
if ($availableGiB -lt 28 -or $gpuUsedMiB -ge 1200) {
    throw ('preflight resource gate failed: RAM={0:F2}GiB GPU={1}MiB' -f
        $availableGiB, $gpuUsedMiB)
}
Write-Status ('preflight=PASS RAM={0:F2}GiB GPU={1}MiB' -f
    $availableGiB, $gpuUsedMiB)

Get-ChildItem Env: |
    Where-Object Name -Like 'DS4_*' |
    ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" }
foreach ($entry in $manifestNames.GetEnumerator()) {
    Set-Item -LiteralPath "Env:$($entry.Key)" -Value ([string]$entry.Value)
}
Remove-Item Env:DS4_METAL_GRAPH_TOKEN_PROFILE -ErrorAction SilentlyContinue
$env:DS4_REQUEST_PHASE_TRACE = '1'
$env:DS4_MTP_STRICT = '1'
if ($mtpBatchVerify) {
    $env:DS4_MTP_BATCH_VERIFY = '1'
}
if ($DiagnosticMtp) {
    $env:DS4_MTP_TIMING = '1'
    $env:DS4_MTP_SPEC_LOG = '1'
}
if ($TraceMode -eq 'Sampled') {
    $env:DS4_CUDA_DECODE_TRACE_POSITIONS = '13,141'
    $env:DS4_CUDA_DECODE_TRACE_LAYERS = '0,4,20,42'
}
if ($TraceMode -eq 'Nsys') {
    $env:DS4_CUDA_DECODE_TRACE_POSITIONS = '100'
    $env:DS4_CUDA_DECODE_TRACE_LAYERS = '0'
    $env:DS4_CUDA_NSYS_CAPTURE = '1'
}
if ($PackedCopy -eq 'On') {
    $env:DS4_CUDA_MOE_ROUTE_PACKED_COPY = '1'
}
if ($BatchedPublish -eq 'On') {
    $env:DS4_CUDA_MOE_ROUTE_BATCHED_PUBLISH = '1'
}
if ($G73HostSelected -eq 'On') {
    $env:DS4_CUDA_G73_REUSE_HOST_SELECTED = '1'
}
if ($RouteIoQd -eq 'Qd4') {
    $env:DS4_CUDA_G73_ROUTE_IO_QD = '4'
}

$shutdownBytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($shutdownBytes)
$rng.Dispose()
$shutdownToken = ([BitConverter]::ToString($shutdownBytes) -replace '-', '').ToLowerInvariant()
$env:DS4_G130_U1_SHUTDOWN_TOKEN = $shutdownToken
$sha = [System.Security.Cryptography.SHA256]::Create()
$tokenHash = ([BitConverter]::ToString(
    $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($shutdownToken))) -replace '-', ''
).ToLowerInvariant()
$sha.Dispose()
@(
    'shutdown_endpoint=http://127.0.0.1:8000/__g130_u1_shutdown'
    'shutdown_mode_planned=authenticated_loopback_http'
    "shutdown_token_sha256=$tokenHash"
) | Set-Content -LiteralPath $controlPath -Encoding ascii

$server = $null
$monitor = $null
$shutdownMode = 'not_started'
$turn1Receipt = $null
$turn2Receipt = $null
$runError = $null
$ds4Pid = 0

try {
    $launchFile = $exe
    $launchArguments = $serverArguments
    $launcherStdout = $serverStdout
    $launcherStderr = $serverLog
    if ($TraceMode -eq 'Nsys') {
        if (-not (Test-Path -LiteralPath $nsys)) {
            throw "Nsight Systems executable missing: $nsys"
        }
        if (-not (Test-Path -LiteralPath $nsysStreamExtractor)) {
            throw "Nsight stream extractor missing: $nsysStreamExtractor"
        }
        if (-not (Test-Path -LiteralPath $python)) {
            throw "Python runtime missing: $python"
        }
        $launchFile = $nsys
        $launcherStdout = $nsysLauncherStdout
        $launcherStderr = $nsysLauncherStderr
        $launchArguments = @(
            'profile',
            '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop',
            '--trace=cuda',
            '--sample=none',
            '--cpuctxsw=none',
            '--cuda-memory-usage=false',
            '--wait=all',
            '--show-output=true',
            '--force-overwrite=true',
            '--export=sqlite',
            '--output', $nsysOutputBase,
            $exe
        ) + $serverArguments
    }
    @(
        "executable=$launchFile"
        $launchArguments
    ) | Set-Content -LiteralPath $launcherArgsPath -Encoding utf8
    $server = Start-Process -FilePath $launchFile `
        -ArgumentList $launchArguments `
        -WindowStyle Hidden `
        -RedirectStandardOutput $launcherStdout `
        -RedirectStandardError $launcherStderr `
        -PassThru
    Write-Status "launcher_started mode=$TraceMode pid=$($server.Id)"

    $ready = $false
    for ($attempt = 1; $attempt -le 900; $attempt++) {
        if ($server.HasExited) {
            break
        }
        & curl.exe -s --max-time 2 'http://127.0.0.1:8000/health' -o NUL 2>$null
        if ($LASTEXITCODE -eq 0) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        throw "server readiness failed after $attempt attempts"
    }
    $ds4Pid = Get-Ds4ListenerPid
    if ($ds4Pid -le 0) {
        throw 'server health passed but loopback listener PID was not found'
    }
    Write-Status "server_ready attempts=$attempt ds4_pid=$ds4Pid launcher_pid=$($server.Id)"

    $monitorArgs = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$monitorScript`"",
        '-ServerPid', [string]$ds4Pid,
        '-CsvPath', "`"$monitorCsv`"",
        '-StopPath', "`"$monitorStop`"",
        '-SampleSeconds', '5'
    ) -join ' '
    $monitor = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList $monitorArgs `
        -WindowStyle Hidden `
        -RedirectStandardOutput $monitorOut `
        -RedirectStandardError $monitorErr `
        -PassThru

    $turn1 = [ordered]@{
        model = 'ds4'
        think = $false
        max_tokens = $Turn1MaxTokens
        temperature = 0
        seed = 12345
        messages = @(
            [ordered]@{
                role = 'user'
                content = 'Ciao, sai fare un bel sito?'
            }
        )
    }
    $turn1Body = $turn1 | ConvertTo-Json -Depth 8 -Compress
    [System.IO.File]::WriteAllText(
        $turn1RequestPath,
        $turn1Body,
        [System.Text.UTF8Encoding]::new($false))
    $turn1StartLine = Get-LogLineCount
    Write-Status "turn1_started max_tokens=$Turn1MaxTokens"
    $turn1Receipt = Invoke-Ds4Request `
        -RequestPath $turn1RequestPath `
        -ResponsePath $turn1ResponsePath
    Start-Sleep -Seconds 1
    $turn1EndLine = Get-LogLineCount
    Save-LogSlice -StartLine $turn1StartLine -EndLine $turn1EndLine -Path $turn1LogPath
    $turn1Response = [System.IO.File]::ReadAllText(
        $turn1ResponsePath,
        [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    $assistantContent = [string]$turn1Response.choices[0].message.content
    if ([string]::IsNullOrWhiteSpace($assistantContent)) {
        throw 'turn1 response has no assistant content'
    }
    $turn1PromptTokens = [int]$turn1Response.usage.prompt_tokens
    $turn1CompletionTokens = [int]$turn1Response.usage.completion_tokens
    Write-Status (
        'turn1_done elapsed_s={0:F3} prompt_tokens={1} completion_tokens={2} wall_tps={3:F3}' -f
        $turn1Receipt.elapsed_seconds,
        $turn1PromptTokens,
        $turn1CompletionTokens,
        ($turn1CompletionTokens / $turn1Receipt.elapsed_seconds))

    $turn2 = [ordered]@{
        model = 'ds4'
        think = $false
        max_tokens = $Turn2MaxTokens
        temperature = 0
        seed = 12345
        messages = @(
            [ordered]@{
                role = 'user'
                content = 'Ciao, sai fare un bel sito?'
            },
            [ordered]@{
                role = 'assistant'
                content = $assistantContent
            },
            [ordered]@{
                role = 'user'
                content = 'Fammi una landing page minimal, single-file HTML, molto breve.'
            }
        )
    }
    $turn2Body = $turn2 | ConvertTo-Json -Depth 8 -Compress
    [System.IO.File]::WriteAllText(
        $turn2RequestPath,
        $turn2Body,
        [System.Text.UTF8Encoding]::new($false))
    $turn2StartLine = Get-LogLineCount
    Write-Status "turn2_started max_tokens=$Turn2MaxTokens"
    $turn2Receipt = Invoke-Ds4Request `
        -RequestPath $turn2RequestPath `
        -ResponsePath $turn2ResponsePath
    Start-Sleep -Seconds 1
    $turn2EndLine = Get-LogLineCount
    Save-LogSlice -StartLine $turn2StartLine -EndLine $turn2EndLine -Path $turn2LogPath
    $turn2Response = [System.IO.File]::ReadAllText(
        $turn2ResponsePath,
        [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    $turn2CompletionTokens = [int]$turn2Response.usage.completion_tokens
    Write-Status (
        'turn2_done elapsed_s={0:F3} prompt_tokens={1} completion_tokens={2} wall_tps={3:F3}' -f
        $turn2Receipt.elapsed_seconds,
        ([int]$turn2Response.usage.prompt_tokens),
        $turn2CompletionTokens,
        ($turn2CompletionTokens / $turn2Receipt.elapsed_seconds))
} catch {
    $runError = $_.Exception.Message
    Write-Status "run_error=$runError"
} finally {
    New-Item -ItemType File -Path $monitorStop -Force | Out-Null
    if ($monitor -and -not $monitor.HasExited) {
        $null = $monitor.WaitForExit(10000)
        if (-not $monitor.HasExited) {
            Stop-Process -Id $monitor.Id -Force -ErrorAction SilentlyContinue
        }
    }

    if ($server) {
        try {
            $currentListenerPid = Get-Ds4ListenerPid
            if ($ds4Pid -le 0 -and $currentListenerPid -gt 0) {
                $ds4Pid = $currentListenerPid
                Write-Status "shutdown_recovered_ds4_pid=$ds4Pid"
            }
            $ds4Alive = $ds4Pid -gt 0 -and
                $null -ne (Get-Process -Id $ds4Pid -ErrorAction SilentlyContinue)
            $portAlive = (Get-Ds4ListenerPid) -gt 0
            if ($ds4Alive -or $portAlive) {
                $body = '{"token":"' + $shutdownToken + '"}'
                $shutdownReceipt = Invoke-RestMethod -Method Post `
                    -Uri 'http://127.0.0.1:8000/__g130_u1_shutdown' `
                    -ContentType 'application/json' `
                    -Body $body `
                    -TimeoutSec 10
                $shutdownReceipt | ConvertTo-Json -Depth 4 |
                    Set-Content -LiteralPath $shutdownReceiptPath -Encoding ascii

                $drainWatch = [System.Diagnostics.Stopwatch]::StartNew()
                do {
                    $ds4Alive = $ds4Pid -gt 0 -and
                        $null -ne (Get-Process `
                            -Id $ds4Pid `
                            -ErrorAction SilentlyContinue)
                    $portAlive = (Get-Ds4ListenerPid) -gt 0
                    if (-not $ds4Alive -and -not $portAlive) {
                        break
                    }
                    Start-Sleep -Milliseconds 250
                } while ($drainWatch.Elapsed.TotalSeconds -lt 120)
                $drainWatch.Stop()

                if ($ds4Alive -or $portAlive) {
                    if ($ds4Pid -gt 0) {
                        Stop-Process `
                            -Id $ds4Pid `
                            -Force `
                            -ErrorAction SilentlyContinue
                    }
                    $shutdownMode = 'forced_after_grace_timeout'
                } else {
                    $shutdownMode = 'graceful_http_verified'
                }
            } else {
                $shutdownMode = 'already_exited'
            }

            $server.Refresh()
            if (-not $server.HasExited) {
                $null = $server.WaitForExit(30000)
            }
            if (-not $server.HasExited) {
                Stop-Process `
                    -Id $server.Id `
                    -Force `
                    -ErrorAction SilentlyContinue
                $null = $server.WaitForExit(10000)
                if ($shutdownMode -eq 'graceful_http_verified') {
                    $shutdownMode = 'graceful_http_launcher_forced'
                }
            }
        } catch {
            if ($ds4Pid -gt 0) {
                Stop-Process `
                    -Id $ds4Pid `
                    -Force `
                    -ErrorAction SilentlyContinue
            }
            $server.Refresh()
            if (-not $server.HasExited) {
                Stop-Process `
                    -Id $server.Id `
                    -Force `
                    -ErrorAction SilentlyContinue
                $null = $server.WaitForExit(10000)
            }
            $shutdownMode = 'forced_after_grace_error'
            if (-not $runError) {
                $runError = "shutdown: $($_.Exception.Message)"
            }
        }
    }
    Write-Status "shutdown_mode=$shutdownMode"
}

if ($TraceMode -eq 'Nsys' -and (Test-Path -LiteralPath $nsysSqlitePath)) {
    & $python $nsysStreamExtractor `
        --sqlite $nsysSqlitePath `
        --stdout $serverStdout `
        --stderr $serverLog
    if ($LASTEXITCODE -ne 0) {
        if (-not $runError) {
            $runError = "Nsight ProcessStreams extraction failed: exit=$LASTEXITCODE"
        }
        Write-Status "streams_extracted=FAIL exit=$LASTEXITCODE"
    } else {
        Write-Status "streams_extracted=PASS"
    }
}

$allLog = if (Test-Path -LiteralPath $serverLog) {
    $stream = [System.IO.File]::Open(
        $serverLog,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
    $reader = [System.IO.StreamReader]::new(
        $stream, [System.Text.Encoding]::UTF8, $true)
    try {
        $reader.ReadToEnd()
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
} else {
    ''
}
$mtpLoadedMatches = [regex]::Matches(
    $allLog,
    'MTP support model loaded:[^\r\n]*\(draft=(\d+)\)')
$mtpMicroMatches = [regex]::Matches(
    $allLog,
    'mtp timing micro drafted=(\d+) committed=(\d+)[^\r\n]*')
$mtpDecode2Matches = [regex]::Matches(
    $allLog,
    'mtp timing decode2 drafted=(\d+) committed=(\d+)[^\r\n]*')
$mtpMarginSkipMatches = [regex]::Matches(
    $allLog,
    'mtp timing margin-skip drafted=(\d+) committed=(\d+)[^\r\n]*')
$mtpSequentialMatches = [regex]::Matches(
    $allLog,
    'mtp timing seq drafted=(\d+) verified=(\d+)[^\r\n]*')
$mtpFirstMissCount = [regex]::Matches(
    $allLog,
    'mtp spec miss first draft=\d+').Count
$mtpMicroFallbackCount = [regex]::Matches(
    $allLog,
    'mtp spec micro verifier failed, falling back to sequential').Count
$mtpDecode2FallbackCount = [regex]::Matches(
    $allLog,
    'mtp decode2 verifier failed, falling back to sequential').Count
$mtpVerifierHardFailureCount = [regex]::Matches(
    $allLog,
    'MTP verifier failed').Count
$mtpDraftFailureCount = [regex]::Matches(
    $allLog,
    'mtp probe draft failed').Count
$mtpDraftedTotal = [Int64]$mtpFirstMissCount
$mtpAcceptedDrafts = [Int64]0
foreach ($match in @($mtpMicroMatches) + @($mtpDecode2Matches) +
        @($mtpMarginSkipMatches)) {
    $mtpDraftedTotal += [Int64]$match.Groups[1].Value
    $mtpAcceptedDrafts += [Int64]$match.Groups[2].Value
}
foreach ($match in @($mtpSequentialMatches)) {
    $mtpDraftedTotal += [Int64]$match.Groups[1].Value
    $mtpAcceptedDrafts += [Int64]$match.Groups[2].Value
}
$mtpLoggedCycles =
    $mtpFirstMissCount +
    $mtpMicroMatches.Count +
    $mtpDecode2Matches.Count +
    $mtpMarginSkipMatches.Count +
    $mtpSequentialMatches.Count
$mtpFallbackCount =
    $mtpMicroFallbackCount +
    $mtpDecode2FallbackCount +
    $mtpSequentialMatches.Count +
    $mtpVerifierHardFailureCount +
    $mtpDraftFailureCount
$preserveMatches = [regex]::Matches(
    $allLog,
    '\[prefill-mass-wrap\] result=preserved reason=short-suffix[^\r\n]*')
$decodeRefusedCount = [regex]::Matches(
    $allLog,
    'decode refused: closed snapshot publication failed').Count
$mailboxQuarantineCount = [regex]::Matches(
    $allLog,
    'route-mailbox[^\r\n]*(quarant|mismatch|poison)').Count
$routeSummaryMatches = [regex]::Matches(
    $allLog,
    '\[gpu-resident-routes\] final[^\r\n]*')
$routeSummary = if ($routeSummaryMatches.Count -gt 0) {
    $routeSummaryMatches[$routeSummaryMatches.Count - 1].Value
} else {
    ''
}
$packedRuntimeRequested = -1L
$packedExperts = -1L
$packedSubmissions = -1L
$packedBytes = -1L
$legacySubmissions = -1L
$missExperts = -1L
$batchedPublishRuntimeRequested = -1L
$batchedPublishKernels = -1L
$batchedPublishRoutes = -1L
$legacyPublishKernels = -1L
$g73HostSelectedRuntimeRequested = -1L
$g73HostSelectedReuse = -1L
$g73HostSelectedFallbacks = -1L
$g73ClassificationD2h = -1L
$routeIoRuntimeRequested = -1L
$routeIoCalls = -1L
$routeIoRoutes = -1L
$routeIoSpans = -1L
$routeIoSubmits = -1L
$routeIoCompletions = -1L
$routeIoBytes = -1L
$routeIoFailures = -1L
$routeIoFallbacks = -1L
$routeIoMaxInflight = -1L
$routeCalls = -1L
if ($routeSummary -match '(?:^|\s)calls=(\d+)') {
    $routeCalls = [Int64]$matches[1]
}
if ($routeSummary -match 'miss_experts=(\d+)') {
    $missExperts = [Int64]$matches[1]
}
if ($routeSummary -match 'packed_copy_requested=(\d+)') {
    $packedRuntimeRequested = [Int64]$matches[1]
}
if ($routeSummary -match 'packed_copy_experts=(\d+)') {
    $packedExperts = [Int64]$matches[1]
}
if ($routeSummary -match 'packed_copy_submissions=(\d+)') {
    $packedSubmissions = [Int64]$matches[1]
}
if ($routeSummary -match 'packed_copy_bytes=(\d+)') {
    $packedBytes = [Int64]$matches[1]
}
if ($routeSummary -match 'legacy_copy_submissions=(\d+)') {
    $legacySubmissions = [Int64]$matches[1]
}
if ($routeSummary -match 'batched_publish_requested=(\d+)') {
    $batchedPublishRuntimeRequested = [Int64]$matches[1]
}
if ($routeSummary -match 'batched_publish_kernels=(\d+)') {
    $batchedPublishKernels = [Int64]$matches[1]
}
if ($routeSummary -match 'batched_publish_routes=(\d+)') {
    $batchedPublishRoutes = [Int64]$matches[1]
}
if ($routeSummary -match 'legacy_publish_kernels=(\d+)') {
    $legacyPublishKernels = [Int64]$matches[1]
}
if ($routeSummary -match 'g73_host_selected_requested=(\d+)') {
    $g73HostSelectedRuntimeRequested = [Int64]$matches[1]
}
if ($routeSummary -match 'g73_host_selected_reuse=(\d+)') {
    $g73HostSelectedReuse = [Int64]$matches[1]
}
if ($routeSummary -match 'g73_host_selected_fallbacks=(\d+)') {
    $g73HostSelectedFallbacks = [Int64]$matches[1]
}
if ($routeSummary -match 'g73_classification_d2h=(\d+)') {
    $g73ClassificationD2h = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_requested=(\d+)') {
    $routeIoRuntimeRequested = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_calls=(\d+)') {
    $routeIoCalls = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_routes=(\d+)') {
    $routeIoRoutes = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_spans=(\d+)') {
    $routeIoSpans = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_submits=(\d+)') {
    $routeIoSubmits = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_completions=(\d+)') {
    $routeIoCompletions = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_bytes=(\d+)') {
    $routeIoBytes = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_failures=(\d+)') {
    $routeIoFailures = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_fallbacks=(\d+)') {
    $routeIoFallbacks = [Int64]$matches[1]
}
if ($routeSummary -match 'route_io_qd_max_inflight=(\d+)') {
    $routeIoMaxInflight = [Int64]$matches[1]
}
$packedGatePass = if ($PackedCopy -eq 'On') {
    $packedRuntimeRequested -eq 1 -and
    $packedExperts -gt 0 -and
    $packedSubmissions -eq (2 * $packedExperts) -and
    $packedBytes -gt 0 -and
    $legacySubmissions -eq 0
} else {
    $packedRuntimeRequested -eq 0 -and
    $packedExperts -eq 0 -and
    $packedSubmissions -eq 0 -and
    $packedBytes -eq 0 -and
    $legacySubmissions -gt 0
}
$batchedPublishGatePass = if ($BatchedPublish -eq 'On') {
    $batchedPublishRuntimeRequested -eq 1 -and
    $missExperts -gt 0 -and
    $routeCalls -gt 0 -and
    $batchedPublishKernels -gt 0 -and
    $batchedPublishKernels -lt $batchedPublishRoutes -and
    $batchedPublishRoutes -eq $missExperts -and
    $legacyPublishKernels -eq 0
} else {
    $batchedPublishRuntimeRequested -eq 0 -and
    $batchedPublishKernels -eq 0 -and
    $batchedPublishRoutes -eq 0 -and
    $legacyPublishKernels -gt 0
}
$g73HostSelectedGatePass = if ($G73HostSelected -eq 'On') {
    $g73HostSelectedRuntimeRequested -eq 1 -and
    $g73HostSelectedReuse -gt 0 -and
    $g73HostSelectedFallbacks -eq 0 -and
    $g73ClassificationD2h -eq 0
} else {
    $g73HostSelectedRuntimeRequested -eq 0 -and
    $g73HostSelectedReuse -eq 0 -and
    $g73HostSelectedFallbacks -eq 0 -and
    $g73ClassificationD2h -gt 0
}
$routeIoGatePass = if ($RouteIoQd -eq 'Qd4') {
    $routeIoRuntimeRequested -eq 4 -and
    $routeIoCalls -gt 0 -and
    $routeIoRoutes -gt 0 -and
    $routeIoSpans -eq (3 * $routeIoRoutes) -and
    $routeIoSubmits -eq $routeIoSpans -and
    $routeIoCompletions -eq $routeIoSubmits -and
    $routeIoBytes -gt 0 -and
    $routeIoFailures -eq 0 -and
    $routeIoFallbacks -eq 0 -and
    $routeIoMaxInflight -eq 4
} else {
    $routeIoRuntimeRequested -eq 0 -and
    $routeIoCalls -eq 0 -and
    $routeIoRoutes -eq 0 -and
    $routeIoSpans -eq 0 -and
    $routeIoSubmits -eq 0 -and
    $routeIoCompletions -eq 0 -and
    $routeIoBytes -eq 0 -and
    $routeIoFailures -eq 0 -and
    $routeIoFallbacks -eq 0 -and
    $routeIoMaxInflight -eq 0
}
$nsysStartedCount = [regex]::Matches(
    $allLog,
    '\[nsys-capture\] result=started position=100(?:\r?\n|$)').Count
$nsysStoppedCount = [regex]::Matches(
    $allLog,
    '\[nsys-capture\] result=stopped position=100 ' +
    'reason=token-complete status=[^\r\n]+').Count
$nsysLauncherText = if (Test-Path -LiteralPath $nsysLauncherStdout) {
    Get-Content -LiteralPath $nsysLauncherStdout -Raw
} else {
    ''
}
$nsysRangeEndedCount = [regex]::Matches(
    $nsysLauncherText,
    'Capture range ended in the application\.').Count
$nsysReportBytes = if (Test-Path -LiteralPath $nsysReportPath) {
    (Get-Item -LiteralPath $nsysReportPath).Length
} else {
    0L
}
$nsysSqliteBytes = if (Test-Path -LiteralPath $nsysSqlitePath) {
    (Get-Item -LiteralPath $nsysSqlitePath).Length
} else {
    0L
}
$nsysGatePass = if ($TraceMode -eq 'Nsys') {
    $nsysStartedCount -eq 1 -and
    $nsysRangeEndedCount -eq 1 -and
    $nsysReportBytes -gt 0 -and
    $nsysSqliteBytes -gt 0
} else {
    $nsysStartedCount -eq 0 -and
    $nsysStoppedCount -eq 0
}
$chatRanges = [regex]::Matches(
    $allLog,
    'chat ctx=(\d+)\.\.(\d+):(\d+) prompt start')
$cachedPrefix = -1
$suffixTokens = -1
if ($chatRanges.Count -ge 2) {
    $cachedPrefix = [int]$chatRanges[1].Groups[1].Value
    $suffixTokens = [int]$chatRanges[1].Groups[3].Value
}
$preserveLine = if ($preserveMatches.Count -gt 0) {
    $preserveMatches[$preserveMatches.Count - 1].Value
} else {
    ''
}
$snapshotUnchanged = $false
$residentUnchanged = $false
if ($preserveLine -match 'snapshot_before=(\d+).*snapshot_after=(\d+)') {
    $snapshotUnchanged = $matches[1] -eq $matches[2]
}
if ($preserveLine -match 'resident_before=(\d+).*resident_after=(\d+)') {
    $residentUnchanged = $matches[1] -eq $matches[2]
}

$turn1Completion = 0
$turn2Completion = 0
$turn1ContentHash = ''
$turn2ContentHash = ''
$turn1FinishReason = ''
$turn2FinishReason = ''
$turn1RequestHash = if (Test-Path -LiteralPath $turn1RequestPath) {
    (Get-FileHash -LiteralPath $turn1RequestPath -Algorithm SHA256).Hash
} else {
    ''
}
$turn2RequestHash = if (Test-Path -LiteralPath $turn2RequestPath) {
    (Get-FileHash -LiteralPath $turn2RequestPath -Algorithm SHA256).Hash
} else {
    ''
}
if (Test-Path -LiteralPath $turn1ResponsePath) {
    try {
        $parsed = [System.IO.File]::ReadAllText(
            $turn1ResponsePath,
            [System.Text.Encoding]::UTF8) | ConvertFrom-Json
        $turn1Completion = [int]$parsed.usage.completion_tokens
        $turn1ContentHash = Get-TextSha256 `
            -Text ([string]$parsed.choices[0].message.content)
        $turn1FinishReason = [string]$parsed.choices[0].finish_reason
    } catch {}
}
if (Test-Path -LiteralPath $turn2ResponsePath) {
    try {
        $parsed = [System.IO.File]::ReadAllText(
            $turn2ResponsePath,
            [System.Text.Encoding]::UTF8) | ConvertFrom-Json
        $turn2Completion = [int]$parsed.usage.completion_tokens
        $turn2ContentHash = Get-TextSha256 `
            -Text ([string]$parsed.choices[0].message.content)
        $turn2FinishReason = [string]$parsed.choices[0].finish_reason
    } catch {}
}

function Get-A4DecodeElapsed {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0.0 }
    $text = Get-Content -LiteralPath $Path -Raw
    $found = [regex]::Matches(
        $text,
        '\[request-phase\] event=request-end[^\r\n]*since_decode=([0-9.]+)')
    if ($found.Count -eq 0) { return 0.0 }
    return [double]$found[$found.Count - 1].Groups[1].Value
}

$turn1DecodeElapsed = Get-A4DecodeElapsed -Path $turn1LogPath
$turn2DecodeElapsed = Get-A4DecodeElapsed -Path $turn2LogPath
$decodeElapsedTotal = $turn1DecodeElapsed + $turn2DecodeElapsed
$acceptedOutputTokens = $turn1Completion + $turn2Completion
$acceptedOutputTps = if ($decodeElapsedTotal -gt 0) {
    $acceptedOutputTokens / $decodeElapsedTotal
} else {
    0.0
}
$mtpAcceptanceRate = if ($mtpDraftedTotal -gt 0) {
    [double]$mtpAcceptedDrafts / [double]$mtpDraftedTotal
} else {
    0.0
}
$mtpAcceptedDraftTps = if ($decodeElapsedTotal -gt 0) {
    [double]$mtpAcceptedDrafts / $decodeElapsedTotal
} else {
    0.0
}
$mtpDraftedTps = if ($decodeElapsedTotal -gt 0) {
    [double]$mtpDraftedTotal / $decodeElapsedTotal
} else {
    0.0
}
$mtpExpectedCycles = $acceptedOutputTokens - $mtpAcceptedDrafts
$mtpUnloggedTerminalCycles = $mtpExpectedCycles - $mtpLoggedCycles
$mtpCycleReconciliation =
    $DiagnosticMtp -and
    $mtpUnloggedTerminalCycles -ge 0 -and
    $mtpUnloggedTerminalCycles -le 2
$mtpLoadedDraft = if ($mtpLoadedMatches.Count -gt 0) {
    [int]$mtpLoadedMatches[$mtpLoadedMatches.Count - 1].Groups[1].Value
} else {
    0
}
$mtpMetricVisibility = if ($DiagnosticMtp) {
    'per_cycle'
} elseif ($MtpMode -eq 'Control') {
    'disabled_by_draft1'
} else {
    'load_only_receipt_required'
}
$mtpEngagementGate = if ($MtpMode -eq 'Control') {
    $mtpLoadedMatches.Count -eq 1 -and
    $mtpLoadedDraft -eq 1 -and
    $mtpDraftedTotal -eq 0 -and
    $mtpAcceptedDrafts -eq 0 -and
    $mtpFallbackCount -eq 0
} elseif ($DiagnosticMtp) {
    $mtpLoadedMatches.Count -eq 1 -and
    $mtpLoadedDraft -eq 2 -and
    $mtpMicroMatches.Count -gt 0 -and
    $mtpDraftedTotal -gt 0 -and
    $mtpDecode2Matches.Count -eq 0 -and
    $mtpMarginSkipMatches.Count -eq 0 -and
    $mtpSequentialMatches.Count -eq 0 -and
    $mtpFallbackCount -eq 0 -and
    $mtpCycleReconciliation
} else {
    # Performance Batch2 leaves per-cycle timing disabled.  A paired diagnostic
    # receipt from the identical foundation proves acceptance/fallback behavior.
    $mtpLoadedMatches.Count -eq 1 -and
    $mtpLoadedDraft -eq 2 -and
    $null -ne $engagementReceipt
}

$correctnessGate = if ($referenceReceipt) {
    $turn1Completion -eq [int]$referenceReceipt.turn1_completion_tokens -and
    $turn2Completion -eq [int]$referenceReceipt.turn2_completion_tokens -and
    $turn1ContentHash -eq [string]$referenceReceipt.turn1_content_sha256 -and
    $turn2ContentHash -eq [string]$referenceReceipt.turn2_content_sha256 -and
    $turn1FinishReason -eq [string]$referenceReceipt.turn1_finish_reason -and
    $turn2FinishReason -eq [string]$referenceReceipt.turn2_finish_reason -and
    $turn1RequestHash -eq [string]$referenceReceipt.turn1_request_sha256 -and
    $turn2RequestHash -eq [string]$referenceReceipt.turn2_request_sha256
} else {
    $MtpMode -eq 'Control' -and
    $turn1Completion -eq $Turn1MaxTokens -and
    $turn2Completion -eq $Turn2MaxTokens -and
    $turn1ContentHash.Length -eq 64 -and
    $turn2ContentHash.Length -eq 64 -and
    $turn1FinishReason -eq 'length' -and
    $turn2FinishReason -eq 'length'
}

$h2dSubmissions = if ($packedSubmissions -ge 0 -and $legacySubmissions -ge 0) {
    $packedSubmissions + $legacySubmissions
} else {
    -1
}
$routeCallsPerAccepted = if ($acceptedOutputTokens -gt 0 -and $routeCalls -ge 0) {
    [double]$routeCalls / [double]$acceptedOutputTokens
} else { -1.0 }
$missExpertsPerAccepted = if ($acceptedOutputTokens -gt 0 -and $missExperts -ge 0) {
    [double]$missExperts / [double]$acceptedOutputTokens
} else { -1.0 }
$h2dSubmissionsPerAccepted =
    if ($acceptedOutputTokens -gt 0 -and $h2dSubmissions -ge 0) {
        [double]$h2dSubmissions / [double]$acceptedOutputTokens
    } else { -1.0 }
$routeIoBytesPerAccepted =
    if ($acceptedOutputTokens -gt 0 -and $routeIoBytes -ge 0) {
        [double]$routeIoBytes / [double]$acceptedOutputTokens
    } else { -1.0 }

$baseGatePass =
    -not $runError -and
    $cachedPrefix -gt 0 -and
    $suffixTokens -gt 0 -and
    $suffixTokens -lt 32 -and
    $preserveMatches.Count -ge 1 -and
    $snapshotUnchanged -and
    $residentUnchanged -and
    $decodeRefusedCount -eq 0 -and
    $mailboxQuarantineCount -eq 0 -and
    $correctnessGate -and
    $mtpEngagementGate -and
    $packedGatePass -and
    $batchedPublishGatePass -and
    $g73HostSelectedGatePass -and
    $routeIoGatePass -and
    $nsysGatePass -and
    $shutdownMode -eq 'graceful_http_verified'

$gatePass = if ($TraceMode -eq 'Nsys') {
    -not $runError -and
    $correctnessGate -and
    $mtpEngagementGate -and
    $nsysGatePass -and
    $shutdownMode -eq 'graceful_http_verified'
} else {
    $baseGatePass
}

$controlReferenceSha256 = ''
if ($gatePass -and $MtpMode -eq 'Control') {
    [ordered]@{
        schema = 'ds4_a4_control_reference_v1'
        mode = 'Control'
        exe_sha256 = $actualExeHash
        model_sha256 = $expectedModelHash
        mtp_sha256 = $actualMtpHash
        base_manifest_sha256 = $actualManifestHash
        temperature = 0
        think = $false
        seed = [UInt64]12345
        trace_mode = 'Off'
        packed_copy = $PackedCopy
        batched_publish = $BatchedPublish
        g73_host_selected = $G73HostSelected
        route_io_qd = $RouteIoQd
        mtp_draft = 1
        mtp_margin = 3
        mtp_strict = $true
        mtp_batch_verify = $false
        turn1_max_tokens = $Turn1MaxTokens
        turn2_max_tokens = $Turn2MaxTokens
        turn1_request_sha256 = $turn1RequestHash
        turn2_request_sha256 = $turn2RequestHash
        turn1_completion_tokens = $turn1Completion
        turn2_completion_tokens = $turn2Completion
        turn1_finish_reason = $turn1FinishReason
        turn2_finish_reason = $turn2FinishReason
        turn1_content_sha256 = $turn1ContentHash
        turn2_content_sha256 = $turn2ContentHash
        source_run_dir = $runDirectory
    } | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $controlReferencePath -Encoding utf8
    $controlReferenceSha256 =
        (Get-FileHash -LiteralPath $controlReferencePath -Algorithm SHA256).Hash
}

$batch2EngagementSha256 = ''
if ($gatePass -and $MtpMode -eq 'Batch2' -and $DiagnosticMtp) {
    [ordered]@{
        schema = 'ds4_a4_batch2_engagement_v1'
        mode = 'Batch2'
        diagnostic_mtp = $true
        exe_sha256 = $actualExeHash
        model_sha256 = $expectedModelHash
        mtp_sha256 = $actualMtpHash
        base_manifest_sha256 = $actualManifestHash
        control_reference_sha256 = $referenceReceiptSha256
        trace_mode = 'Off'
        packed_copy = $PackedCopy
        batched_publish = $BatchedPublish
        g73_host_selected = $G73HostSelected
        route_io_qd = $RouteIoQd
        mtp_draft = 2
        mtp_margin = 3
        mtp_strict = $true
        mtp_batch_verify = $true
        engagement_gate = [bool]$mtpEngagementGate
        cycle_reconciliation = [bool]$mtpCycleReconciliation
        drafted_total = [Int64]$mtpDraftedTotal
        accepted_drafts = [Int64]$mtpAcceptedDrafts
        acceptance_rate = [double]$mtpAcceptanceRate
        logged_cycles = [Int64]$mtpLoggedCycles
        expected_cycles = [Int64]$mtpExpectedCycles
        unlogged_terminal_cycles = [Int64]$mtpUnloggedTerminalCycles
        fallback_count = [Int64]$mtpFallbackCount
        source_run_dir = $runDirectory
    } | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $batch2EngagementPath -Encoding utf8
    $batch2EngagementSha256 =
        (Get-FileHash -LiteralPath $batch2EngagementPath -Algorithm SHA256).Hash
}

@(
    "run_dir=$runDirectory"
    "gate_pass=$gatePass"
    "exe_sha256=$actualExeHash"
    "model_sha256=$expectedModelHash"
    "mtp_mode=$MtpMode"
    "mtp_path=$mtp"
    "mtp_sha256=$actualMtpHash"
    "mtp_draft=$mtpDraft"
    "mtp_batch_verify=$([int]$mtpBatchVerify)"
    "mtp_strict=1"
    'temperature=0'
    'think=false'
    'seed=12345'
    "diagnostic_mtp=$([int][bool]$DiagnosticMtp)"
    "mtp_loaded_count=$($mtpLoadedMatches.Count)"
    "mtp_loaded_draft=$mtpLoadedDraft"
    "mtp_metric_visibility=$mtpMetricVisibility"
    "mtp_engagement_gate=$mtpEngagementGate"
    "mtp_micro_count=$($mtpMicroMatches.Count)"
    "mtp_first_miss_count=$mtpFirstMissCount"
    "mtp_decode2_count=$($mtpDecode2Matches.Count)"
    "mtp_margin_skip_count=$($mtpMarginSkipMatches.Count)"
    "mtp_sequential_count=$($mtpSequentialMatches.Count)"
    "mtp_fallback_count=$mtpFallbackCount"
    "mtp_drafted_total=$mtpDraftedTotal"
    "mtp_accepted_drafts=$mtpAcceptedDrafts"
    "mtp_acceptance_rate=$([Math]::Round($mtpAcceptanceRate, 9))"
    "mtp_logged_cycles=$mtpLoggedCycles"
    "mtp_expected_cycles=$mtpExpectedCycles"
    "mtp_unlogged_terminal_cycles=$mtpUnloggedTerminalCycles"
    "mtp_cycle_reconciliation=$mtpCycleReconciliation"
    "mtp_drafted_tps=$([Math]::Round($mtpDraftedTps, 6))"
    "mtp_accepted_draft_tps=$([Math]::Round($mtpAcceptedDraftTps, 6))"
    "accepted_output_tokens=$acceptedOutputTokens"
    "accepted_output_tps=$([Math]::Round($acceptedOutputTps, 6))"
    "correctness_gate=$correctnessGate"
    "reference_receipt_path=$ReferenceReceiptPath"
    "reference_receipt_sha256=$referenceReceiptSha256"
    "engagement_receipt_path=$EngagementReceiptPath"
    "engagement_receipt_sha256=$engagementReceiptSha256"
    "control_reference_path=$(if ($controlReferenceSha256) { $controlReferencePath } else { '' })"
    "control_reference_sha256=$controlReferenceSha256"
    "batch2_engagement_path=$(if ($batch2EngagementSha256) { $batch2EngagementPath } else { '' })"
    "batch2_engagement_sha256=$batch2EngagementSha256"
    "trace_mode=$TraceMode"
    "packed_copy=$PackedCopy"
    "batched_publish=$BatchedPublish"
    "g73_host_selected=$G73HostSelected"
    "route_io_qd=$RouteIoQd"
    'manifest_count=47'
    "base_manifest_sha256=$actualManifestHash"
    "source_tree_matches_binary_receipt=$sourceTreeMatchesBinaryReceipt"
    'ctx_capacity=150000'
    "cached_prefix_tokens=$cachedPrefix"
    "suffix_tokens=$suffixTokens"
    "short_suffix_preserve_count=$($preserveMatches.Count)"
    "snapshot_unchanged=$snapshotUnchanged"
    "resident_unchanged=$residentUnchanged"
    "decode_refused_count=$decodeRefusedCount"
    "mailbox_quarantine_count=$mailboxQuarantineCount"
    "packed_gate_pass=$packedGatePass"
    "packed_copy_runtime_requested=$packedRuntimeRequested"
    "packed_copy_experts=$packedExperts"
    "packed_copy_submissions=$packedSubmissions"
    "packed_copy_bytes=$packedBytes"
    "legacy_copy_submissions=$legacySubmissions"
    "miss_experts=$missExperts"
    "route_calls=$routeCalls"
    "batched_publish_gate_pass=$batchedPublishGatePass"
    "batched_publish_runtime_requested=$batchedPublishRuntimeRequested"
    "batched_publish_kernels=$batchedPublishKernels"
    "batched_publish_routes=$batchedPublishRoutes"
    "legacy_publish_kernels=$legacyPublishKernels"
    "g73_host_selected_gate_pass=$g73HostSelectedGatePass"
    "g73_host_selected_runtime_requested=$g73HostSelectedRuntimeRequested"
    "g73_host_selected_reuse=$g73HostSelectedReuse"
    "g73_host_selected_fallbacks=$g73HostSelectedFallbacks"
    "g73_classification_d2h=$g73ClassificationD2h"
    "route_io_qd_gate_pass=$routeIoGatePass"
    "route_io_qd_runtime_requested=$routeIoRuntimeRequested"
    "route_io_qd_calls=$routeIoCalls"
    "route_io_qd_routes=$routeIoRoutes"
    "route_io_qd_spans=$routeIoSpans"
    "route_io_qd_submits=$routeIoSubmits"
    "route_io_qd_completions=$routeIoCompletions"
    "route_io_qd_bytes=$routeIoBytes"
    "route_io_qd_failures=$routeIoFailures"
    "route_io_qd_fallbacks=$routeIoFallbacks"
    "route_io_qd_max_inflight=$routeIoMaxInflight"
    "route_calls_per_accepted_token=$([Math]::Round($routeCallsPerAccepted, 9))"
    "miss_experts_per_accepted_token=$([Math]::Round($missExpertsPerAccepted, 9))"
    "h2d_submissions=$h2dSubmissions"
    "h2d_submissions_per_accepted_token=$([Math]::Round($h2dSubmissionsPerAccepted, 9))"
    "route_io_qd_bytes_per_accepted_token=$([Math]::Round($routeIoBytesPerAccepted, 9))"
    "nsys_gate_pass=$nsysGatePass"
    "nsys_started_count=$nsysStartedCount"
    "nsys_stopped_count=$nsysStoppedCount"
    "nsys_range_ended_count=$nsysRangeEndedCount"
    "nsys_process_stream_scope=$(if ($TraceMode -eq 'Nsys') { 'capture_range_only' } else { 'not_applicable' })"
    "nsys_report_bytes=$nsysReportBytes"
    "nsys_sqlite_bytes=$nsysSqliteBytes"
    "turn1_completion_tokens=$turn1Completion"
    "turn1_finish_reason=$turn1FinishReason"
    "turn1_request_sha256=$turn1RequestHash"
    "turn1_content_sha256=$turn1ContentHash"
    "turn1_decode_elapsed_seconds=$([Math]::Round($turn1DecodeElapsed, 6))"
    "turn1_decode_tps=$(if ($turn1DecodeElapsed -gt 0) { [Math]::Round($turn1Completion / $turn1DecodeElapsed, 6) } else { 0 })"
    "turn1_wall_tps=$(if ($turn1Receipt) { [Math]::Round($turn1Completion / $turn1Receipt.elapsed_seconds, 6) } else { 0 })"
    "turn2_completion_tokens=$turn2Completion"
    "turn2_finish_reason=$turn2FinishReason"
    "turn2_request_sha256=$turn2RequestHash"
    "turn2_content_sha256=$turn2ContentHash"
    "turn2_decode_elapsed_seconds=$([Math]::Round($turn2DecodeElapsed, 6))"
    "turn2_decode_tps=$(if ($turn2DecodeElapsed -gt 0) { [Math]::Round($turn2Completion / $turn2DecodeElapsed, 6) } else { 0 })"
    "turn2_wall_tps=$(if ($turn2Receipt) { [Math]::Round($turn2Completion / $turn2Receipt.elapsed_seconds, 6) } else { 0 })"
    "shutdown_mode=$shutdownMode"
    "run_error=$(if ($runError) { $runError } else { 'none' })"
    "preserve_line=$preserveLine"
) | Set-Content -LiteralPath $resultPath -Encoding utf8

Write-Status "gate_pass=$gatePass"
if (-not $gatePass) {
    exit 2
}
