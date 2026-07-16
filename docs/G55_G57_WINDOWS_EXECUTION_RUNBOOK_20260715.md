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
    ExpectedEmbeddedMaskSHA256 = "14ea7c79b0f4f0dcc59830a9877b8e76541e66dd7371b4c3781559639f0def71"
    ExpectedPayloadSHA256 = "5cb4bf69d7c6ef2aadfc8760069c3a7a89fc40504ce80b7b3735b10f9539e4b5"
}
if ($k60Pack) { $g57.PackPath = $k60Pack }
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File .\g57_sparse_bake_safety.ps1 @g57
```

G57 is n=1 functional safety only. It must verify footer/manifest/CRC, the
logical source-mask SHA separately from the embedded bitset SHA, payload SHA,
positive selected-route calls/slots, zero rejected routes,
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

## Execution Status Through G60 (2026-07-16)

The frozen sequence has now been executed. These rows supersede the earlier
"not yet measured" status but do not change the fail-closed rules above.

- G55 clean QD1/QD8 completed with n=3 per arm. QD8 was retained because it
  improved both mean and median WRAP/TTFT; it did not promote a new decode
  SOTA.
- G56 metadata profiling completed. Exact-contiguous coalescing projects
  `13,653 -> 7,185` reads (`-47.37%`) at byte amplification `1.0`. Threshold
  over-read adds no structural benefit and remains rejected.
- G51 VRAM seed completed with n=3 per arm. The seed saved net route H2D but
  changed decode by only `+0.145652%`; it is not promoted.
- G57 K60 and K75 each passed n=1 functional safety. Both installed the
  embedded mask, produced coherent non-empty temp0/nothink output, reported
  `route_calls=301`, `route_slots=4,902`, `rejected=0`, and left no DS4 process
  or VRAM allocation behind. These remain safety rows, not performance or
  quality claims.
- G58 bake-only performance completed with n=3 independent processes per
  bake. K60 measured `1.863333` mean / `1.86` median t/s; K75 measured `1.79`
  mean / `1.76` median t/s. The path read `34.323854` / `37.397406` GiB from
  disk on average for 64 generated tokens. Bake-only is therefore rejected as
  a SOTA transport path. It proves that physical pruning alone does not make
  the retained experts resident.
- G60 budget-preserving full/partial layer stripes completed with n=3 per arm.
  Stripe changed decode from `4.513333` to `4.53` mean t/s (`+0.369284%`) while
  mass coverage fell from `0.5874` to `0.5272` (`-10.248553%`). It is not
  promoted; long-output n>=3 L0-L3 grading is still required before any quality
  decision.

The next composed experiment must keep the bake identity fixed and add only
one resident transport mechanism. Start with K60 plus the current
request-scoped resident arena/cache, retaining embedded-mask guards and all
G58 provenance checks. Compare it against G58 K60, not against a different
mask. Do not add SPEX, PACE, dynamic compression or a second transport change
in the same first composition.

Native result commits:

- G57 K60: [`2d9cb0a`](https://github.com/imanu86/ds4-win/commit/2d9cb0a)
- G57 K75: [`2773f5b`](https://github.com/imanu86/ds4-win/commit/2773f5b)
- G58 runner/fix/results: [`40cb7a4`](https://github.com/imanu86/ds4-win/commit/40cb7a4),
  [`7bb6a9d`](https://github.com/imanu86/ds4-win/commit/7bb6a9d),
  [`f9ba227`](https://github.com/imanu86/ds4-win/commit/f9ba227)
- G60 implementation/safety/matrix/results:
  [`ea683f6`](https://github.com/imanu86/ds4-win/commit/ea683f6),
  [`2a9c47b`](https://github.com/imanu86/ds4-win/commit/2a9c47b),
  [`8acaf32`](https://github.com/imanu86/ds4-win/commit/8acaf32),
  [`63c8dd6`](https://github.com/imanu86/ds4-win/commit/63c8dd6)
