param(
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
    [ValidateSet('Off', 'On')]
    [string]$GraphTensorDevice = 'Off',
    [ValidateRange(1, 1024)]
    [int]$Turn1MaxTokens = 128,
    [ValidateRange(1, 1024)]
    [int]$Turn2MaxTokens = 32,
    [switch]$ValidateOnly
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $invariant

if ($PackedCopy -ne 'Off' -or
    $BatchedPublish -ne 'Off' -or
    $G73HostSelected -ne 'Off' -or
    $RouteIoQd -ne 'Off') {
    throw 'A1 requires the M0/P0 foundation: PackedCopy, BatchedPublish, G73HostSelected and RouteIoQd must all be Off'
}

# Windows PowerShell can expose both Path/PATH to Start-Process. Normalize the
# runner-local environment block before starting either DS4 or the monitor.
$savedProcessPath = [Environment]::GetEnvironmentVariable('Path', 'Process')
Remove-Item Env:PATH -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable('Path', $savedProcessPath, 'Process')

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent (Split-Path -Parent $root)
$sourceRoot = Join-Path $workspace 'work\wt-hot-reserve'
$manifestSource = Join-Path $root 'manifest_47.env'
$monitorScript = Join-Path $root 'runtime_monitor.ps1'
$nsysStreamExtractor = Join-Path $root 'extract_nsys_process_streams.py'
$python = 'C:\Users\imanu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$buildRoot = Join-Path $workspace 'work\build-a1-graph-tensor-device'
$exe = Join-Path $buildRoot 'ds4_server.exe'
$model = 'C:\ds4-models\ds4-2bit.gguf'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe'
$outputRoot = Join-Path $workspace 'outputs\p0p1_runtime'
$safeTrace = $TraceMode.ToLowerInvariant()
$safePacked = $PackedCopy.ToLowerInvariant()
$safePublish = $BatchedPublish.ToLowerInvariant()
$safeHostSelected = $G73HostSelected.ToLowerInvariant()
$safeRouteIo = $RouteIoQd.ToLowerInvariant()
$safeGraphTensorDevice = $GraphTensorDevice.ToLowerInvariant()
$runId = '{0}_a1_m0_p0_trace-{1}_packed-{2}_publish-{3}_hostsel-{4}_routeio-{5}_graphdev-{6}' -f (
    Get-Date -Format 'yyyyMMdd_HHmmss'), $safeTrace, $safePacked, $safePublish,
    $safeHostSelected, $safeRouteIo, $safeGraphTensorDevice
$runDirectory = Join-Path $outputRoot $runId

$expectedExeHash = '2B11AA8160191F9AC44802451CDEC2EF36334C52B06F0ECF5FCC2068990562BA'
$expectedModelHash = 'efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668'
$expectedTurn1ContentHash = '7f82253a4825191926f56073e40f10a0cff5541a721731bc81d2909dc1a4a65b'
$expectedTurn2ContentHash = '0179556c8e2dbcdc818fad315ca4df78f7537b63816dac276615b314195b13eb'
$expectedSourceHashes = [ordered]@{
    'ds4.c' = '54F5E14378B71CC039D1467B5B9D1D7F7512FDC460B1ADC49F24B963C5EC8466'
    'ds4_cuda.cu' = '53B152FA67FC8155F1391C27CB807001E7B3DBB39BBD5F863CC2338BFB684182'
    'ds4_gpu.h' = '625F477594EF2C835047A8FDD2512B8715ADA82C72C71DE1B4466EFBE76D436C'
    'ds4_metal.m' = '81612C353EFCBDEFF58619BD38A9CD9E72382D6A9FDF2AC709BA27FCCD2D6F5D'
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
$controlPath = Join-Path $runDirectory 'control_manifest.txt'
$provenancePath = Join-Path $runDirectory 'provenance.txt'
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
$allocSummaryPath = Join-Path $runDirectory 'cuda_alloc_summaries.log'
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

$actualExeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
if ($actualExeHash -ne $expectedExeHash) {
    throw "binary hash drift: expected=$expectedExeHash actual=$actualExeHash"
}
$sourceReceipt = @()
foreach ($entry in $expectedSourceHashes.GetEnumerator()) {
    $path = Join-Path $sourceRoot $entry.Key
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $entry.Value) {
        throw "source hash drift for $($entry.Key): expected=$($entry.Value) actual=$actual"
    }
    $sourceReceipt += "$($entry.Key)_SHA256=$actual"
}
if ((Get-Item -LiteralPath $model).Length -ne [UInt64]86720111488) {
    throw 'model byte-size drift'
}

$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding ascii
$traceManifest = @(
    "TraceMode=$TraceMode"
    'DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY=<UNSET>'
    'DS4_CUDA_DECODE_TRACE_POSITIONS=<UNSET>'
    'DS4_CUDA_DECODE_TRACE_LAYERS=<UNSET>'
)
if ($TraceMode -eq 'Sampled') {
    $traceManifest = @(
        "TraceMode=$TraceMode"
        'DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY=<UNSET>'
        'DS4_CUDA_DECODE_TRACE_POSITIONS=13,141'
        'DS4_CUDA_DECODE_TRACE_LAYERS=0,4,20,42'
    )
}
if ($TraceMode -eq 'Nsys') {
    $traceManifest = @(
        "TraceMode=$TraceMode"
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
$traceManifest += @(
    'DS4_CUDA_ALLOC_TRACE=1'
    'DS4_CUDA_ALLOC_TRACE_MIN_MIB=1048576'
    'cuda_alloc_event_lines=suppressed'
    'cuda_alloc_summary_phases=decode-start,request-end'
)
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
    "GraphTensorDevice=$GraphTensorDevice"
    $(if ($GraphTensorDevice -eq 'On') {
        'DS4_CUDA_GRAPH_TENSOR_DEVICE=1'
    } else {
        'DS4_CUDA_GRAPH_TENSOR_DEVICE=<UNSET>'
    })
) | Set-Content -LiteralPath $experimentPath -Encoding ascii

$serverArguments = @(
    '-m', $model,
    '--cuda',
    '-c', '150000',
    '-n', '8192',
    '--mtp-draft', '1',
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
    "model_path=$model"
    "model_bytes=$((Get-Item -LiteralPath $model).Length)"
    "model_SHA256_receipt=$expectedModelHash"
    'experiment=A1_graph_tensor_device'
    'foundation=M0_P0'
    'ctx_capacity=150000'
    'live_position_150000=not_claimed'
    'mtp_draft=1'
    "route_packed_copy=$PackedCopy"
    "route_batched_publish=$BatchedPublish"
    "g73_host_selected=$G73HostSelected"
    "g73_route_io_qd=$RouteIoQd"
    "graph_tensor_device=$GraphTensorDevice"
    'cuda_alloc_trace=1'
    'cuda_alloc_trace_min_mib=1048576'
    'temperature=0.7'
    'seed=12345'
    'think=false'
    "turn1_max_tokens=$Turn1MaxTokens"
    "turn2_max_tokens=$Turn2MaxTokens"
    'turn1_prompt_utf8=Ciao, sai fare un bel sito?'
    'turn2_prompt_utf8=Fammi una landing page minimal, single-file HTML, molto breve.'
) | Set-Content -LiteralPath $provenancePath -Encoding utf8

if ($ValidateOnly) {
    Write-Status "validation=PASS experiment=A1 foundation=M0_P0 manifest_count=47 declared_parameter_count=52 packed_copy=$PackedCopy batched_publish=$BatchedPublish g73_host_selected=$G73HostSelected route_io_qd=$RouteIoQd graph_tensor_device=$GraphTensorDevice alloc_trace=1 hashes=PASS server_started=no"
    "run_dir=$runDirectory`nvalidation=PASS`nexperiment=A1`nfoundation=M0_P0`nmanifest_count=47`ndeclared_parameter_count=52`npacked_copy=$PackedCopy`nbatched_publish=$BatchedPublish`ng73_host_selected=$G73HostSelected`nroute_io_qd=$RouteIoQd`ngraph_tensor_device=$GraphTensorDevice`nalloc_trace=1`nserver_started=no" |
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
$env:DS4_CUDA_ALLOC_TRACE = '1'
$env:DS4_CUDA_ALLOC_TRACE_MIN_MIB = '1048576'
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
if ($GraphTensorDevice -eq 'On') {
    $env:DS4_CUDA_GRAPH_TENSOR_DEVICE = '1'
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
        temperature = 0.7
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
        temperature = 0.7
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
$allocLogMatches = [regex]::Matches(
    $allLog,
    '(?m)^ds4: \[cuda-alloc-(?:summary|row|kind|balance)\]' +
    '[^\r\n]*(?=\r?$)')
$allocLogLines = @($allocLogMatches | ForEach-Object { $_.Value })
if ($allocLogLines.Count -gt 0) {
    $allocLogLines | Set-Content -LiteralPath $allocSummaryPath -Encoding ascii
} else {
    Set-Content -LiteralPath $allocSummaryPath -Value '' -Encoding ascii
}
$decodeAllocSummaryMatches = [regex]::Matches(
    $allLog,
    '(?m)^ds4: \[cuda-alloc-summary\] phase=decode-start ' +
    'sites=\d+ events=\d+ site_overflow=0 untracked_frees=0(?=\r?$)')
$requestEndAllocSummaryMatches = [regex]::Matches(
    $allLog,
    '(?m)^ds4: \[cuda-alloc-summary\] phase=request-end ' +
    'sites=\d+ events=\d+ site_overflow=0 untracked_frees=0(?=\r?$)')
$decodeAllocBalanceMatches = [regex]::Matches(
    $allLog,
    '(?m)^ds4: \[cuda-alloc-balance\] phase=decode-start ' +
    '[^\r\n]* status=ok(?=\r?$)')
$requestEndAllocBalanceMatches = [regex]::Matches(
    $allLog,
    '(?m)^ds4: \[cuda-alloc-balance\] phase=request-end ' +
    '[^\r\n]* status=ok(?=\r?$)')
$allocationFailureMatches = [regex]::Matches(
    $allLog,
    '(?m)^ds4: \[cuda-alloc-row\][^\r\n]* ' +
    'fails=([1-9]\d*)(?=\r?$)')
$genericDeviceRows = [regex]::Matches(
    $allLog,
    '(?m)^ds4: \[cuda-alloc-row\][^\r\n]* ' +
    'tag=device/ds4_gpu_tensor_alloc_backing@\d+' +
    '[^\r\n]*(?=\r?$)')
$genericDeviceRowsExact = @($genericDeviceRows | Where-Object {
    $_.Value -match ' calls=310 fails=0$'
})
$decodeStartUsedMiB = @()
$decodeStartFreeMiB = @()
foreach ($match in $decodeAllocBalanceMatches) {
    if ($match.Value -match ' free_mib=([0-9.]+)') {
        $decodeStartFreeMiB += $matches[1]
    }
    if ($match.Value -match ' used_mib=([0-9.]+)') {
        $decodeStartUsedMiB += $matches[1]
    }
}
$requestEndUsedMiB = @()
$requestEndFreeMiB = @()
foreach ($match in $requestEndAllocBalanceMatches) {
    if ($match.Value -match ' free_mib=([0-9.]+)') {
        $requestEndFreeMiB += $matches[1]
    }
    if ($match.Value -match ' used_mib=([0-9.]+)') {
        $requestEndUsedMiB += $matches[1]
    }
}
$allocTraceGatePass =
    $decodeAllocSummaryMatches.Count -eq 2 -and
    $requestEndAllocSummaryMatches.Count -eq 2 -and
    $decodeAllocBalanceMatches.Count -eq 2 -and
    $requestEndAllocBalanceMatches.Count -eq 2 -and
    $allocationFailureMatches.Count -eq 0
$graphTensorDeviceGatePass = if ($GraphTensorDevice -eq 'On') {
    $genericDeviceRows.Count -eq 4 -and
    $genericDeviceRowsExact.Count -eq 4
} else {
    $genericDeviceRows.Count -eq 0
}
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
if (Test-Path -LiteralPath $turn1ResponsePath) {
    try {
        $parsed = [System.IO.File]::ReadAllText(
            $turn1ResponsePath,
            [System.Text.Encoding]::UTF8) | ConvertFrom-Json
        $turn1Completion = [int]$parsed.usage.completion_tokens
        $turn1ContentHash = Get-TextSha256 `
            -Text ([string]$parsed.choices[0].message.content)
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
    } catch {}
}

$gatePass = if ($TraceMode -eq 'Nsys') {
    -not $runError -and
    $turn1ContentHash -eq $expectedTurn1ContentHash -and
    $turn2ContentHash -eq $expectedTurn2ContentHash -and
    $allocTraceGatePass -and
    $graphTensorDeviceGatePass -and
    $nsysGatePass -and
    $shutdownMode -eq 'graceful_http_verified'
} else {
    -not $runError -and
    $cachedPrefix -gt 0 -and
    $suffixTokens -gt 0 -and
    $suffixTokens -lt 32 -and
    $preserveMatches.Count -ge 1 -and
    $snapshotUnchanged -and
    $residentUnchanged -and
    $decodeRefusedCount -eq 0 -and
    $mailboxQuarantineCount -eq 0 -and
    $turn1ContentHash -eq $expectedTurn1ContentHash -and
    $turn2ContentHash -eq $expectedTurn2ContentHash -and
    $packedGatePass -and
    $batchedPublishGatePass -and
    $g73HostSelectedGatePass -and
    $routeIoGatePass -and
    $allocTraceGatePass -and
    $graphTensorDeviceGatePass -and
    $nsysGatePass -and
    $shutdownMode -eq 'graceful_http_verified'
}

@(
    "run_dir=$runDirectory"
    "gate_pass=$gatePass"
    "trace_mode=$TraceMode"
    "packed_copy=$PackedCopy"
    "batched_publish=$BatchedPublish"
    "g73_host_selected=$G73HostSelected"
    "route_io_qd=$RouteIoQd"
    "graph_tensor_device=$GraphTensorDevice"
    'manifest_count=47'
    'declared_parameter_count=52'
    'ctx_capacity=150000'
    'live_position_150000=not_claimed'
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
    "alloc_trace_gate_pass=$allocTraceGatePass"
    "alloc_decode_start_summary_count=$($decodeAllocSummaryMatches.Count)"
    "alloc_request_end_summary_count=$($requestEndAllocSummaryMatches.Count)"
    "alloc_decode_start_balance_count=$($decodeAllocBalanceMatches.Count)"
    "alloc_request_end_balance_count=$($requestEndAllocBalanceMatches.Count)"
    "alloc_failure_row_count=$($allocationFailureMatches.Count)"
    "alloc_generic_device_row_count=$($genericDeviceRows.Count)"
    "alloc_generic_device_exact_row_count=$($genericDeviceRowsExact.Count)"
    "alloc_decode_start_used_mib=$($decodeStartUsedMiB -join ',')"
    "alloc_decode_start_free_mib=$($decodeStartFreeMiB -join ',')"
    "alloc_request_end_used_mib=$($requestEndUsedMiB -join ',')"
    "alloc_request_end_free_mib=$($requestEndFreeMiB -join ',')"
    "alloc_summary_path=$allocSummaryPath"
    "graph_tensor_device_gate_pass=$graphTensorDeviceGatePass"
    "nsys_gate_pass=$nsysGatePass"
    "nsys_started_count=$nsysStartedCount"
    "nsys_stopped_count=$nsysStoppedCount"
    "nsys_range_ended_count=$nsysRangeEndedCount"
    "nsys_process_stream_scope=$(if ($TraceMode -eq 'Nsys') { 'capture_range_only' } else { 'not_applicable' })"
    "nsys_report_bytes=$nsysReportBytes"
    "nsys_sqlite_bytes=$nsysSqliteBytes"
    "turn1_completion_tokens=$turn1Completion"
    "turn1_content_sha256=$turn1ContentHash"
    "turn1_wall_tps=$(if ($turn1Receipt) { [Math]::Round($turn1Completion / $turn1Receipt.elapsed_seconds, 6) } else { 0 })"
    "turn2_completion_tokens=$turn2Completion"
    "turn2_content_sha256=$turn2ContentHash"
    "turn2_wall_tps=$(if ($turn2Receipt) { [Math]::Round($turn2Completion / $turn2Receipt.elapsed_seconds, 6) } else { 0 })"
    "shutdown_mode=$shutdownMode"
    "run_error=$(if ($runError) { $runError } else { 'none' })"
    "preserve_line=$preserveLine"
) | Set-Content -LiteralPath $resultPath -Encoding utf8

Write-Status "gate_pass=$gatePass"
if (-not $gatePass) {
    exit 2
}
