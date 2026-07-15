# Windows G55-G57 Execution Runbook

Date: 2026-07-15
Native branch: `port/windows-dynamic-arena-0051`
REAP branch: `plan/0051-transport-gate-20260713`

This runbook is fail-closed. Do not launch any command below until the bake
task sends the literal handoff `GPU/DISCO LIBERI`. A free GPU alone is not
enough: the model disk must also be quiescent.

## Preflight

Run from the native Windows checkout:

```powershell
$repo = "C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work"
Set-Location $repo

$manifest = Get-Content .\build\Release\g7_build_manifest.json -Raw |
    ConvertFrom-Json
if ($manifest.head -ne (git rev-parse HEAD).Trim()) {
    throw "Build manifest HEAD mismatch"
}

$ctest = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $ctest --test-dir build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Parser tests failed" }
```

The clean full-model sequence expects native HEAD `93cb193` or a documented
descendant, build input fingerprint
`9821365dc16fb02291101b9b9ef436336b446f90797a8aecb21e526f3d5b06aa`, and
executable SHA-256
`18b8e53627690d950ad37329fb32354a2b278a30b30df012df56b99d374419e1`.
If any value differs, regenerate the build manifest and record the new hashes
before measuring; never mix old and new rows.

## G55: Clean File-QD A/B

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File .\g55_wrap_file_qd_ab.ps1 -CandidateFileQD 8
```

This launches six independent processes in counterbalanced order
`QD1,QD8,QD8,QD1,QD1,QD8`. Do not use the earlier contaminated QD8 safety row.

Mandatory gates:

- three valid processes per arm;
- exact expected output SHA on all six rows;
- identical candidate-mask FNV-1a across all rows;
- identical executable/source/build/model provenance;
- zero contamination aborts, snapshot misses, SSD bytes and tier failures;
- QD1 async submit/completion/failure counters all zero;
- QD8 submits and completions equal source-part count, failures zero.

Promote QD8 only if both mean and median WRAP improve over QD1 without a
contradictory TTFT or resource-pressure regression. If the result is mixed,
retain QD1. Record the selected value as `$winningQD` for G51.

## G56: Layout Profile

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File .\g56_wrap_layout_profile.ps1
```

This is one functional metadata profile, not a benchmark. It must preserve the
request-scoped closed mask, expose its fingerprint, use QD1 with zero async
counters, use no-default-sync for every route, remain exact, and report zero
snapshot misses/SSD/failures. Use its 0/4 KiB/64 KiB/1 MiB projections only to
decide whether range coalescing merits implementation. It cannot promote a
throughput claim.

## G51: Prefill VRAM Seed A/B

```powershell
$winningQD = 1 # replace with 8 only if G55 passes the promotion rule
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File .\g51_prefill_vram_seed_ab.ps1 `
    -TransportFileQD $winningQD
```

This launches `control,on,on,control,control,on`, three independent processes
per arm. The only intended delta is the request-scoped upload of eight
highest-prefill-mass experts per routed layer (320 total) into VRAM before
decode.

Primary verdict fields:

- `observed_effect.route_h2d_saved_gib_mean`;
- `observed_effect.one_time_seed_h2d_gib_mean`;
- `observed_effect.net_h2d_saved_after_seed_gib_mean`;
- `observed_effect.seed_amortized_within_observed_window`;
- RAM-route and VRAM-route deltas;
- exact output and identical candidate-mask fingerprint.

Decode t/s is the performance outcome, but it is interpreted only after the
transport counters prove what changed. Total tier `ram_h2d_bytes` is not a
standalone verdict because it includes the one-time seed upload.

## G57: K60 Functional Safety

Fill `$k60Model` and optional `$k60Pack` only from the final verified handoff.
Do not infer paths from partial downloads.

The `SparseFile` attribute alone is not evidence of physical sparsity. Before
launching G57, the verified handoff must also prove all of the following:

- `fsutil sparse queryrange` reports allocated ranges separated by real holes,
  rather than one allocated range covering `[0, EOF]`;
- `GetCompressedFileSize` (or an equivalent allocation query) is materially
  smaller than the logical file size;
- payload SHA-256, manifest CRC and mask CRC still match after sparsification;
- the target volume has enough free space for the run without paging or
  emergency allocation pressure.

If any condition fails, do not launch DS4. Fix or re-unpack the sparse model
with explicit hole punching (`FSCTL_SET_ZERO_DATA`) and repeat every gate.

```powershell
$k60Model = "<verified local K60 sparse GGUF path>"
$k60Pack = "<verified local K60 ds4pack path, or empty if independently verified>"

$g57 = @{
    ModelPath = $k60Model
    BakeId = "K60"
    ExpectedPackSHA256 = "3b464ee43514c8caa841be61da70190a4a7ba3c760c22849d2d723e5da5b7d71"
    ExpectedMaskSHA256 = "5b6d98504ba830c1a50945a93d1a6017b1956bd17c56df8c0b1bdf92c1564e97"
    ExpectedPayloadSHA256 = "5cb4bf69d7c6ef2aadfc8760069c3a7a89fc40504ce80b7b3735b10f9539e4b5"
}
if ($k60Pack) { $g57.PackPath = $k60Pack }
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File .\g57_sparse_bake_safety.ps1 @g57
```

G57 is n=1 functional safety only. It must verify footer/manifest/CRC,
mask/payload SHA, positive selected-route calls/slots, zero rejected routes,
non-empty coherent temp0/nothink output, and absence of full-model WRAP, arena,
cache/tiering, SPEX or external mask options. No timing or quality-generalization
claim is permitted.

## Stop Conditions

- Stop immediately on any hash, mask-fingerprint or provenance mismatch.
- Stop on any contamination preflight/runtime abort; do not count the row.
- Stop on any snapshot miss, SSD byte, absent sparse expert read or tier error.
- Stop if an allegedly sparse model has no physical holes or allocated bytes
  equal its logical size.
- Never resume rows produced by a different runner/executable/build hash.
- Commit raw summaries and the ledger interpretation before implementing the
  next lever.
