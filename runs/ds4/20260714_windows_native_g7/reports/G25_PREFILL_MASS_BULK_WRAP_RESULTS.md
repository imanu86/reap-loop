# G25 prefill-mass bulk WRAP

Date: 2026-07-14

## Scope

G25 adds one isolated lever: use cumulative router mass observed during the
semantic prefill to seed the existing pinned-host dynamic arena in one
transaction before decode.

This is not REAP/LiveMask and it does not change routing. There is no router
mask, no expert suppression, no SPEX, no expert cache, and no continuous
promotion/demotion. The selected candidate is ranked by the real prefill gate
mass and published only when the arena has no existing snapshot.

Runtime switch:

```text
DS4_CUDA_PREFILL_MASS_WRAP=1
```

The switch implies prefill-mass observation. Any other nonzero value is
rejected. The publication path is shared with the existing WRAP transport but
does not borrow or mutate decode-observer policy state.

## Protocol

- GPU: NVIDIA RTX 3060 12 GB, driver 596.21, native Windows/WDDM.
- Model: `C:\ds4-models\ds4-2bit.gguf`, 86,720,111,488 bytes.
- Base HEAD at measurement: `334b92d3287d90d71834d3d1369850d9d764952b`.
- Executable SHA-256:
  `2cbaaec4e01e8caaad96f461764b12fd2204025345fd4bff1f069f0875d50722`.
- Source SHA-256 at measurement:
  `93a684fc0211ade68aeb23c58b81a0f4cf5b05105f6f5cdddb0964f3d1b1f758`.
- Prompt: `Explain in one concise paragraph why Julius Caesar crossed the Rubicon.`
- Prompt SHA-256:
  `5f15d0d89e17beba40908970d1d82067bf825b23bf9453aa1fb563dbdd44d8bd`.
- Expected output SHA-256:
  `b037ce25fab7393eeb9fc5b7bf7f5b8ef70768aea476cd1c09b0ffa348323b30`.
- `-n 16 -c 256`, greedy/nothink server defaults.
- `BudgetGB=2`, `ReserveMB=1024`, `Q8F16CacheReserveMB=4096`.
- `ReapPrefetchThreads=8`, `IoQD=1`.
- Each repetition is a new server process. Order was
  WRAP/OFF/OFF/WRAP/WRAP/OFF.
- No throughput verdict uses the 2 GiB `n=1` safety run.

The 14 GiB command shape was:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\g7_measure.ps1 `
  -Prompt "Explain in one concise paragraph why Julius Caesar crossed the Rubicon." `
  -MaxTokens 16 -Context 256 -BudgetGB 2 -ReserveMB 1024 `
  -DynamicArenaGiB 14 -PrefillMassWrap -ReapPrefetchThreads 8 -IoQD 1 `
  -ExpectedContentSHA256 b037ce25fab7393eeb9fc5b7bf7f5b8ef70768aea476cd1c09b0ffa348323b30 `
  -ModelPath C:\ds4-models\ds4-2bit.gguf
```

The control replaces `-PrefillMassWrap` with `-PrefillMassObserve`. It uses the
same binary and allocates the same arena, but does not publish a snapshot.

## Results

All six 14 GiB runs produced the exact expected output hash.

| Arm | Run | Decode t/s | TTFT s | WRAP s | Resident | Arena hits/misses | Fatal |
|---|---|---:|---:|---:|---:|---:|---:|
| WRAP | wrap14_safety_n1 | 4.25 | 20.491 | 15.541 | 1,980 | 2,619 / 1,509 | 0 |
| WRAP | wrap14_n2 | 4.35 | 20.161 | 15.369 | 1,980 | 2,619 / 1,509 | 0 |
| WRAP | wrap14_n3 | 4.52 | 20.026 | 15.408 | 1,980 | 2,619 / 1,509 | 0 |
| OFF | observe14_control_n1 | 2.44 | 4.938 | 0 | 0 | 0 / 0 | 0 |
| OFF | observe14_control_n2 | 1.75 | 10.057 | 0 | 0 | 0 / 0 | 0 |
| OFF | observe14_control_n3 | 2.72 | 4.741 | 0 | 0 | 0 / 0 | 0 |

Measured aggregates:

| Metric | WRAP n=3 | OFF n=3 |
|---|---:|---:|
| decode t/s mean | 4.37 | 2.30 |
| decode t/s median | 4.35 | 2.44 |
| TTFT mean | 20.226 s | 6.579 s |
| WRAP time mean | 15.439 s | 0 s |

The prefill produced 1,980 unique `(layer, expert)` candidates. A 14 GiB arena
had 2,123 slots, so all 1,980 were published. Candidate membership covered
68.20% of selected decode IDs; runtime arena accounting measured 2,619 hits and
1,509 misses (63.44%). These are separate counters and are not substituted for
one another.

The 2 GiB safety run published 303 candidates in 2.200 s, covered 50.03% of
prefill gate mass and 25.78% of selected decode IDs, produced the exact hash,
and decoded at 2.82 t/s. It is an `n=1` correctness result only.

## Finding

The measured G25 result is positive for decode: prefill-ranked pinned-host
residency increased short-run decode throughput from a 2.30 t/s mean to a
4.37 t/s mean under the controlled 14 GiB A/B, with exact output and zero arena
fatal errors.

The 15.44 s mean WRAP cost is not yet amortized by this 16-token request. No
break-even token count is claimed here; that requires a measured longer decode.

## Limits and next gate

- G25 publishes only the first snapshot in an arena lifetime. Later requests
  skip rather than replace it.
- It does not handle domain drift or continuous mass changes.
- It does not pin the experts permanently in VRAM; the arena is pinned system
  RAM and still uploads selected spans to compact VRAM buffers.
- Quality beyond the 16-token exactness probe is unmeasured.
- The next isolated step is a longer G25 run to measure amortization and drift.
  Only after G25 is committed does the Windows port gain the existing
  continuous REAP mass controller as a separate lever.

