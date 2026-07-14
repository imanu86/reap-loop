# G27 REAP mass residency actuator

Date: 2026-07-14

## Objective

Turn the exact G26b sliding mass signal into pinned-system-RAM residency without
changing router scores, top-k, or masks. The policy:

1. keeps every valid resident in the target snapshot;
2. admits highest-mass nonresidents into logical free slots first;
3. once the logical residency limit is full, replaces the lowest-mass resident
   only when `entrant_mass > victim_mass * hysteresis + epsilon`;
4. publishes through the existing transactional WRAP path.

The physical pinned arena retains eight swap slots. Entrants are loaded into
those spare slots, the snapshot is atomically published, and the old victims
become the next spare ring. This preserves rollback; victim storage is never
overwritten before a successful publish. G25 prefill seeding observes the same
logical limit whenever G27 is enabled.

## Controls

```text
DS4_CUDA_REAP_MASS_WRAP=1
DS4_CUDA_REAP_MASS_WINDOW=16
DS4_CUDA_REAP_MASS_GROW_INTERVAL=4
DS4_CUDA_REAP_MASS_HYSTERESIS=1.25
```

`DS4_CUDA_REAP_MASS_WRAP=1` also requires the packed G26b router trace. Runtime
telemetry is fail-closed in `g7_measure.ps1` and states
`router=unbiased mask=off policy=free-then-mass-victim`.

## Negative safety finding

The first implementation filled all 303 physical slots. Initial admission
succeeded, but replacement attempts had no spare destination and aborted cleanly:

```text
tokens=4  entrants=303 victims=0  result=published
tokens=8  entrants=61  victims=61 result=failed reason=begin
tokens=12 entrants=84  victims=84 result=failed reason=begin
```

The active snapshot and resident count remained unchanged after both failures.
This measured failure motivated the transactional eight-slot swap ring; active
victim slots are not destructively reused.

## Final provenance

- Base commit: `20b05194b631b56c7033d2211339e2d837e7d42d`
- GPU: NVIDIA GeForce RTX 3060 12 GB
- Model: `D:\ds4-models\ds4-2bit.gguf`
- Executable SHA-256: `9e5a49558a93ff1b397623981969af81d87fc5688166fdabc7c9ad8363d701d5`
- `ds4_cuda.cu` SHA-256: `efc20cd0a84f3b138404a8f9d53bc9e352a48cc919e74bd33e81e586669a2c57`
- Harness SHA-256: `614fc4176c29c813c683a6d4bfaf9200dbbcf2248a90b25cf68b96fa022329d8`
- Build manifest SHA-256: `ba4c5e9e3dabea0e7e1130cfca60d7ffcfe59765c4762c223186a18f6d59edcc`
- Build fingerprint: `53c78a48ba20f29575111ccc8784b21446dfef333cae53ea9c06dcb1607281fa`
- Common launch: 16 generated tokens, context 256, 2 GiB empty pinned
  arena, 1,024 MiB CUDA reserve, 4,096 MiB Q8/F16 reserve, greedy nothink.
- Expected output SHA-256:
  `b037ce25fab7393eeb9fc5b7bf7f5b8ef70768aea476cd1c09b0ffa348323b30`

Each row below is an independent process. Order was counterbalanced as
`OFF/ON, ON/OFF, OFF/ON`. OFF keeps the packed REAP mass observer active but
does not actuate residency, isolating the WRAP policy cost.

## Final exact A/B

| Mode | Run | Server decode t/s | TTFT s | Publications | Entrants | Victims | WRAP s | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G27 ON | 1 | 2.10 | 4.780 | 4 | 319 | 24 | 2.337 | yes |
| G27 ON | 2 | 2.01 | 4.864 | 4 | 319 | 24 | 2.377 | yes |
| G27 ON | 3 | 2.11 | 4.928 | 4 | 319 | 24 | 2.370 | yes |
| G27 OFF (observe-only) | 1 | 2.44 | 4.858 | 0 | 0 | 0 | 0 | yes |
| G27 OFF (observe-only) | 2 | 2.58 | 4.908 | 0 | 0 | 0 | 0 | yes |
| G27 OFF (observe-only) | 3 | 2.52 | 4.869 | 0 | 0 | 0 | 0 | yes |

| Mode | n | Mean decode t/s | Median decode t/s | Mean TTFT s | Mean WRAP s |
|---|---:|---:|---:|---:|---:|
| G27 ON | 3 | 2.073 | 2.10 | 4.857 | 2.361 |
| G27 OFF (observe-only) | 3 | 2.513 | 2.52 | 4.878 | 0 |

All ON runs publish the same sequence: 295 initial admissions followed by three
eight-entry rotations, for 319 entrants and 24 victims. All runs report zero
fatal arena errors and match the expected output hash.

## Decision

The mechanism and transactional rotation are accepted. The 16-token gate is
not an amortization verdict: G27 pays about 2.36 seconds to populate and rotate
the arena inside a very short decode, producing a measured -17.5% mean decode
delta. The next performance gate must compose G25 prefill seed plus G27 on a
longer decode and report break-even against packed observe-only. The full-table
scan and ranking at each four-token interval also remain a profiling target;
no speed claim is made for them here.
