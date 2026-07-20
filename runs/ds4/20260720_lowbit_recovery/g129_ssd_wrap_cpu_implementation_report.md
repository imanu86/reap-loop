# G129 SSD-WRAP CPU implementation report

Date: 2026-07-20

Status: GO from CPU-only gates for one separately authorized structural safety
`n=1`. This is not runtime, quality, throughput, `n>=3`, or SOTA evidence.

## Scope

The G129 full/open candidate now contains an OFF-default SSD-WRAP path for
causal Q1_0 -> exact-IQ2 promotion. No DS4 server or GPU was started during
this implementation cycle.

The current call always continues through the resident Q1 fallback. Exact IQ2
is read into host storage, validated, and published only for a later call.
There is no permitted SSD -> VRAM current-token transition.

## Runtime contract

```text
Q1 current call
  -> REQUESTED
  -> SSD_INFLIGHT
  -> RAM_READY
  -> RAM_COMMITTING
  -> PINNED_READY | PAGEABLE_READY
  -> ELIGIBLE on a later call

PAGEABLE_READY
  -> fixed pinned bounce ring
  -> H2D_INFLIGHT
  -> VRAM ARENA_READY
```

- OFF creates no thread, handle, ring, allocation, or write.
- Queueing is bounded and deduplicated by layer/expert.
- Prefill and decode waves have distinct bounded limits.
- Source parts are ordered by offset and coalesced only when both source and
  destination ranges are contiguous and provenance-compatible.
- Admission occurs only after a host slot or victim has been reserved.
- Partial reads, invalid ranges/provenance, stale age, and incomplete terminal
  pairing fail closed.
- `first_eligible_call` must be greater than the observation call.
- Pageable IQ2 uses a fixed pinned bounce ring for H2D. There is no dynamic
  `cudaHostRegister`/`cudaHostUnregister` on the hot path.
- `QueryWorkingSetEx` runs only at init, flush, and release, never per token.

## Fixed host budget

Total exact-IQ2 host budget remains 5,902,958,592 bytes: 834 slots of
7,077,888 bytes. Four slots are reserved for the transition ring, leaving 830
stable slots.

| Split | Pinned stable | Pageable stable | Ring |
|---|---:|---:|---:|
| all-pinned control | 830 | 0 | 4 |
| 1.5 / 4.0 GiB | 223 | 607 | 4 |
| 2.0 / 3.5 GiB | 299 | 531 | 4 |

The first proposed safety split is 2.0/3.5 GiB. Its 299 pinned slots are close
to the existing cache320 scale while preserving a larger pageable reuse tier.

At 6 tokens/s, one 7,077,888-byte promotion per token requires about
45.65 MB/s, approximately 93% of the 49.1 MB/s observed promotion source rate.
The structural replay's 1,073 SplitFused misses over 64 tokens would imply up
to about 712 MB/s of host-to-device traffic. Therefore ring wait and promotion
utility are unresolved runtime gates, not assumed wins.

## CPU verification

- PowerShell 5.1 parse: PASS.
- SSD-WRAP parser positive, OFF, and negative cases: PASS.
- Runner SelfTest: PASS.
- Python contract and runtime tests: PASS.
- WhatIf control, legacy promotion, and SSD-WRAP 2.0/3.5: PASS.
- Release `sm_86` build: PASS.
- CTest: 1/1 PASS.
- `git diff --check`: PASS.
- Active `ds4_server` processes after the cycle: 0.

## Frozen identities

```text
build manifest  1893258c8406e8c668eaf1856527ba0b5aba9ea2169e7d53525ca7257686a66b
fingerprint     f80f32087ed9d7716651ea661846a4ed50082aae0746700c8fc3c5c0a75a106a
executable      e258a4fd60c7dc4dfb98cd0ec8b168f4a06e3f3f435adbe8ef89540cd2d307e5
ds4_cuda.cu     19a6e9b34c2ef38d196abc11c93f62cfca7e94d131403378958e0cdc83cdf87e
g7_measure.ps1  3ff7b51af06c4c78c406451bb821beb72c9348a87469aae92021be93d322f833
safety runner   07e9626995fe122a2403a69a775d8c25ce9ffdaccf41b4a0355c9df0c5a9e58c
contract test   23c64ebf6ea73310504625f3a7f8d04ba2b2e7ad6b304401c50be92bb536a997
runtime test    8b3341ec7bf9638201f06d7267f46c7e14acfd8c6ff5bf76ca734c642286e005
protocol        bd0623f9f7ae2196e2d005d796e17632d6009124769cf267b62b8b96074e5d7a
```

Published DS4 commits on `feature/q1-0-resident-base`:

- runtime: `16273d4a1d9b648a5878223e7d3ecd3a8d233672`;
- G129 harness/protocol/tests: `415ed980da69bad304d98e77b1851076a7ae06a6`;
- G73 two-turn runner and CPU mock evidence:
  `801a6d8fee17d1fd18fa4fe83fec3f750501fd7e`.

## Next gate

Run exactly one separately authorized `promotion_ssd_2_0` structural safety.
It must prove nonzero waves, one-to-one attempt/terminal pairing, next-call
causality, working-set accounting, and zero stale/drop/failure/backing/
forbidden/current-token SSD-to-VRAM transitions. It makes no throughput or
quality claim by itself.
