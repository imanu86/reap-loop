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

## Execution Status Through G63 (2026-07-16)

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
- G61 K60 embedded-bake plus `DynamicArenaGiB=30` and `PrefillMassWrap`
  completed one safety row and n=3 performance rows. It used the embedded K60
  mask only, with no cache, tiering, SPEX, ComposePrefillMassTiering,
  RouteNoDefaultSync or external mask. The performance rows matched the frozen
  G58 K60 output SHA-256
  `ceced6c1b481bb2c6f68bd116c06e554502017a44b40b4e5e6bc9fc5d710edc7` with
  `rejected=0`. Decode improved versus frozen G58 K60 from `1.863333` to `2.38`
  mean t/s (`+27.728109%`) and from `1.86` to `2.35` median t/s
  (`+26.344086%`), but TTFT median increased from `18.863` to `38.541` s
  (`+104.320628%`) and disk read mean increased from `34.323854` to
  `36.453855` GiB (`+6.205600%`). WRAP mean was `17.155667` s. The resident path
  accelerates decode but is not promoted as end-to-end SOTA.
- G62 K60 sparse bake plus cache-only GPU residency completed n=3 exact
  temp0/nothink rows on the same cyberpunk64, ctx256 protocol. It used K60
  sparse bake, cache GPU-only 320 LRU and GPU-resident routes, with no arena,
  tiering, SPEX or external mask. The rows matched SHA-256
  `ceced6c1b481bb2c6f68bd116c06e554502017a44b40b4e5e6bc9fc5d710edc7` with
  `contamination=0`, `rejected=0` and `errors=0`. Decode measured `2.123333`
  mean / `2.13` median t/s; TTFT `18.171` mean / `18.173` median s; load
  `10.528771` mean / `10.197356` median s; disk read mean `33.801597` GiB;
  process read mean `91.819531` GiB; dedicated GPU peak `10.76321` GiB. The
  cache reported `6430` hits, `12835` misses and `12515` evictions per run
  (`33.3765897%` direct hit rate), with `2734` worker jobs, `10082` miss
  experts and `7.385333` ms/call wait. Versus frozen G58 K60, decode improved
  `+13.953491%` mean and `+14.516129%` median; versus G61 arena-only it was
  `-10.784328%` mean and `-9.361702%` median. Cache-only pays versus bake-only,
  but the LRU rotates heavily and does not beat arena-only. It is not SOTA and
  makes no L0-L3 quality claim from 64 tokens.
- G63 applied the complete measured G46 composition to the K60 sparse bake:
  30 GiB DynamicArena, source-parts WRAP, PrefillMassWrap, sparse-aware
  ComposePrefillMassTiering, cache 320 LRU, GPU-resident routes,
  RouteNoDefaultSync and mass-LFRU tiering. The n=3 exact rows measured
  `4.553333` mean / `4.55` median t/s versus G46 full-model `4.563333` /
  `4.58` t/s. The measured mean delta was `-0.010000` t/s (`-0.219138%`).
  TTFT measured `42.652` mean / `42.792` median s versus G46 `45.211667` /
  `46.295` s; WRAP mean was `23.720000` versus `24.086333` s. All rows were
  deterministic at SHA-256
  `4aaf0f0813f4cb15ac21a88f195f4f7d2c2af797e81524935e22eea60603c6b1`,
  restored the embedded K60 base mask once, skipped/replaced `1579` absent
  ranked candidates, and reported zero restore failures, snapshot misses,
  forbidden SSD-to-VRAM transfers, tier failures or tier SSD bytes. This is a
  positive transport/composition result, not a long-output quality verdict.

The next experiment is a long-output n>=3 L0-L3 quality gate comparing G63 and
G46 under identical prompts, context and stopping rules. Do not promote G63 as
quality SOTA before that gate passes.

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
- G61 Windows runner/safety: [`27e0545`](https://github.com/imanu86/ds4-win/commit/27e0545),
  [`20ef2b7`](https://github.com/imanu86/ds4-win/commit/20ef2b7),
  [`ba545be`](https://github.com/imanu86/ds4-win/commit/ba545be)
- G62 Windows telemetry/runner/safety/n3:
  [`21d785b`](https://github.com/imanu86/ds4-win/commit/21d785b),
  [`068c522`](https://github.com/imanu86/ds4-win/commit/068c522),
  [`b887b41`](https://github.com/imanu86/ds4-win/commit/b887b41),
  [`ed074d3`](https://github.com/imanu86/ds4-win/commit/ed074d3),
  [`810cdb9`](https://github.com/imanu86/ds4-win/commit/810cdb9)
- G63 sparse-aware runtime/runner/safety/n3:
  [`74bc9d4`](https://github.com/imanu86/ds4-win/commit/74bc9d4),
  [`9066f36`](https://github.com/imanu86/ds4-win/commit/9066f36),
  [`b3f0a62`](https://github.com/imanu86/ds4-win/commit/b3f0a62),
  [`caaffcf`](https://github.com/imanu86/ds4-win/commit/caaffcf),
  [`11c5fb1`](https://github.com/imanu86/ds4-win/commit/11c5fb1)
