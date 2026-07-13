#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(0.01, 50.0)]
    [double]$TargetGiB = 0.25,

    [ValidateRange(0.01, 50.0)]
    [double]$StepGiB = 1.0,

    [ValidateSet('blocks', 'single')]
    [string]$Mode = 'blocks',

    [ValidateRange(1.0, 1024.0)]
    [double]$StagingMiB = 16.0,

    [ValidateRange(0.25, 64.0)]
    [double]$ReserveGiB = 2.0,

    [ValidateRange(0.25, 64.0)]
    [double]$MinWindowsAvailableGiB = 2.0,

    [ValidateRange(0, 128)]
    [int]$Device = 0,

    [ValidateRange(100, 5000)]
    [int]$PollMilliseconds = 250,

    [string]$OutputDirectory,

    [switch]$BuildOnly
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$ascii = [System.Text.Encoding]::ASCII
$targetExplicit = $PSBoundParameters.ContainsKey('TargetGiB')

function Convert-ToInvariantNumber([double]$Value) {
    return $Value.ToString('R', $script:invariant)
}

function Write-AsciiLine([string]$Path, [string]$Line) {
    [System.IO.File]::AppendAllText($Path, $Line + "`r`n", $script:ascii)
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot '..\arena_probe'
}
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($output) | Out-Null

$source = Join-Path $PSScriptRoot 'cuda_pinned_arena_probe_win.cpp'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Missing probe source: $source"
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss', $invariant)
$targetTag = (Convert-ToInvariantNumber $TargetGiB).Replace('.', 'p')
$base = "probe_${stamp}_win_${Mode}_${targetTag}g"
$executable = Join-Path $output "$base.exe"
$objectFile = Join-Path $output "$base.obj"
$compilerPdb = Join-Path $output "$base.compiler.pdb"
$linkerPdb = Join-Path $output "$base.pdb"
$buildCommand = Join-Path $output "$base.build.cmd"
$buildStdout = Join-Path $output "$base.build.stdout.txt"
$buildStderr = Join-Path $output "$base.build.stderr.txt"

$compiler = Get-Command 'cl.exe' -CommandType Application -ErrorAction SilentlyContinue
$vsDevCmd = $null
if ($null -eq $compiler) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw 'MSVC was not found in PATH and vswhere.exe is unavailable.'
    }
    $installations = @(& $vswhere -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath)
    if ($LASTEXITCODE -ne 0 -or $installations.Count -eq 0 -or
        [string]::IsNullOrWhiteSpace([string]$installations[0])) {
        throw 'No Visual Studio installation with the MSVC x64 toolchain was found.'
    }
    $vsDevCmd = Join-Path ([string]$installations[0]).Trim() 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
        throw "Visual Studio developer environment script not found: $vsDevCmd"
    }
}

$buildLines = [System.Collections.Generic.List[string]]::new()
$buildLines.Add('@echo off')
$buildLines.Add('setlocal')
if ($null -ne $vsDevCmd) {
    $buildLines.Add(('call "{0}" -no_logo -arch=amd64 -host_arch=amd64' -f $vsDevCmd))
    $buildLines.Add('if errorlevel 1 exit /b %errorlevel%')
    $compilerInvocation = 'cl.exe'
} else {
    $compilerInvocation = '"{0}"' -f $compiler.Path
}
$compileLine = ('{0} /nologo /std:c++17 /O2 /EHsc /W4 /permissive- /Zc:__cplusplus ' +
    '/utf-8 /Fo"{1}" /Fd"{2}" /Fe"{3}" "{4}" /link /INCREMENTAL:NO /PDB:"{5}"') -f `
    $compilerInvocation, $objectFile, $compilerPdb, $executable, $source, $linkerPdb
$buildLines.Add($compileLine)
$buildLines.Add('exit /b %errorlevel%')
[System.IO.File]::WriteAllText(
    $buildCommand,
    (($buildLines -join "`r`n") + "`r`n"),
    $utf8NoBom)

$cmdArgument = '/d /s /c ""{0}""' -f $buildCommand
$buildProcess = Start-Process -FilePath $env:ComSpec `
    -ArgumentList $cmdArgument `
    -WorkingDirectory $output `
    -WindowStyle Hidden `
    -RedirectStandardOutput $buildStdout `
    -RedirectStandardError $buildStderr `
    -PassThru -Wait
if ($buildProcess.ExitCode -ne 0 -or
    -not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "MSVC build failed with exit code $($buildProcess.ExitCode). See $buildStdout and $buildStderr"
}

if ($BuildOnly) {
    [ordered]@{
        build_only = $true
        executable = $executable
        build_command = $buildCommand
        build_stdout = $buildStdout
        build_stderr = $buildStderr
    } | ConvertTo-Json
    exit 0
}

$runningServers = @(Get-Process -Name 'ds4-server' -ErrorAction SilentlyContinue)
if ($runningServers.Count -ne 0) {
    $serverPids = ($runningServers | ForEach-Object { $_.Id }) -join ','
    throw "Native ds4-server process is running; refusing probe. PID(s): $serverPids"
}

$probeArguments = [System.Collections.Generic.List[string]]::new()
$probeArguments.Add('--mode')
$probeArguments.Add($Mode)
if ($targetExplicit) {
    $probeArguments.Add('--target-gib')
    $probeArguments.Add((Convert-ToInvariantNumber $TargetGiB))
}
$probeArguments.Add('--step-gib')
$probeArguments.Add((Convert-ToInvariantNumber $StepGiB))
$probeArguments.Add('--staging-mib')
$probeArguments.Add((Convert-ToInvariantNumber $StagingMiB))
$probeArguments.Add('--reserve-gib')
$probeArguments.Add((Convert-ToInvariantNumber $ReserveGiB))
$probeArguments.Add('--device')
$probeArguments.Add($Device.ToString($invariant))

$jsonl = Join-Path $output "$base.jsonl"
$stderr = Join-Path $output "$base.stderr.txt"
$monitor = Join-Path $output "$base.windows_memory.csv"
$commandFile = Join-Path $output "$base.command.txt"
$pidFile = Join-Path $output "$base.pid"
$exitCodeFile = Join-Path $output "$base.rc"

$displayCommand = '"{0}" {1}' -f $executable, ($probeArguments -join ' ')
[System.IO.File]::WriteAllText($commandFile, $displayCommand + "`r`n", $utf8NoBom)
[System.IO.File]::WriteAllText(
    $monitor,
    "timestamp_utc,probe_pid,windows_available_gib,action`r`n",
    $ascii)

$process = Start-Process -FilePath $executable `
    -ArgumentList $probeArguments.ToArray() `
    -WorkingDirectory $output `
    -WindowStyle Hidden `
    -RedirectStandardOutput $jsonl `
    -RedirectStandardError $stderr `
    -PassThru
$heldProcessHandle = $process.Handle
[System.IO.File]::WriteAllText(
    $pidFile,
    $process.Id.ToString($invariant) + "`r`n",
    $ascii)

$guardFired = $false
$guardReason = ''
$probeExitCode = $null
try {
    while ($true) {
        $process.Refresh()
        if ($process.HasExited) {
            break
        }

        $availableGiB = $null
        $action = ''
        try {
            $os = Get-CimInstance -ClassName Win32_OperatingSystem
            $availableGiB = ([double]$os.FreePhysicalMemory * 1024.0) / [double](1GB)
            if ($availableGiB -lt $MinWindowsAvailableGiB) {
                $guardFired = $true
                $guardReason = 'windows_available_below_floor'
                $action = 'kill_exact_process_handle'
            }
        } catch {
            $guardFired = $true
            $guardReason = 'windows_memory_monitor_failed'
            $action = 'kill_exact_process_handle_monitor_error'
        }

        $timestamp = [DateTime]::UtcNow.ToString('o', $invariant)
        $availableField = if ($null -eq $availableGiB) {
            ''
        } else {
            ([double]$availableGiB).ToString('F6', $invariant)
        }
        $csvLine = [string]::Format(
            $invariant,
            '{0},{1},{2},{3}',
            $timestamp,
            $process.Id,
            $availableField,
            $action)
        Write-AsciiLine -Path $monitor -Line $csvLine

        if ($guardFired) {
            # Kill through the held Process handle. This targets only the exact
            # process instance launched above and avoids name-wide termination.
            try {
                if (-not $process.HasExited) {
                    $process.Kill()
                }
            } catch [System.InvalidOperationException] {
                # The exact process exited between HasExited and Kill().
            }
            if (-not $process.WaitForExit(5000)) {
                throw "Probe PID $($process.Id) did not exit after exact-handle termination."
            }
            break
        }

        Start-Sleep -Milliseconds $PollMilliseconds
    }

    $process.WaitForExit()
    $probeExitCode = $process.ExitCode
} finally {
    $process.Refresh()
    if (-not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
}

[System.IO.File]::WriteAllText(
    $exitCodeFile,
    ([int]$probeExitCode).ToString($invariant) + "`r`n",
    $ascii)

$runnerExitCode = if ($guardFired) { 20 } else { [int]$probeExitCode }
$summary = [ordered]@{
    target_gib = $TargetGiB
    target_explicit = $targetExplicit
    step_gib = $StepGiB
    mode = $Mode
    staging_mib = $StagingMiB
    reserve_gib = $ReserveGiB
    min_windows_available_gib = $MinWindowsAvailableGiB
    guard_fired = $guardFired
    guard_reason = $guardReason
    probe_pid = $process.Id
    probe_exit_code = [int]$probeExitCode
    runner_exit_code = $runnerExitCode
    executable = $executable
    jsonl = $jsonl
    stderr = $stderr
    windows_memory_csv = $monitor
    command = $commandFile
    pid_file = $pidFile
    exit_code_file = $exitCodeFile
}
$summary | ConvertTo-Json
exit $runnerExitCode
