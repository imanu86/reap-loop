# 0050k transport safety fallback

Status: incremental safety patch for `0050j-transport-pipeline-gate.patch`.
GPU runtime validation is still pending. This is not a transport checksum and
does not implement the dynamic-arena 0051 work.

Apply after 0050j:

```bash
git apply --check /path/to/0050k-transport-safety-fallback.patch
git apply /path/to/0050k-transport-safety-fallback.patch
```

## Safety behavior

The selected-load transport now has one recovery rule for an event failure
after work has been queued: synchronize the selected upload stream, clear the
upload-event and staging-slot in-flight state, and continue only when that
drain succeeds. A failed drain remains a hard failure.

- Staging-event creation failure happens before the first affected H2D. It
  latches process-wide safe synchronous staging and continues with one pinned
  host staging slot; every later staged H2D is drained before slot reuse.
- Staging WAR wait and staging-event record failures use the common drain,
  latch synchronous staging, and resume the current copy only after a
  successful drain.
- Upload-event create, record, and compute-stream wait failures use the same
  drain-and-clear path. This covers staged H2D, masked direct DMA, and the
  deferred post-batch D2D/H2D completion record.
- A staged `cudaMemcpyAsync` H2D failure is counted and drains earlier queued
  work before returning failure; an incomplete destination is never accepted.
- Default success behavior is unchanged. Fault injection, final diagnostics,
  generation checks, and poison are OFF unless explicitly enabled. The cheap
  integer counters remain in memory; they print only when diagnostics are
  enabled or a failure counter is nonzero.

## Diagnostics

Set `DS4_CUDA_TRANSPORT_DIAG=1` for one final stderr line at cleanup/exit:

```text
ds4: CUDA transport safety final phase=... stage_chunks=... cursor_advances=... slot_reuses=... war_waits=... war_wait_failures=... stage_event_create_failures=... stage_event_record_failures=... upload_event_create_failures=... upload_event_record_failures=... upload_event_wait_failures=... fallback_syncs=... fallback_sync_failures=... staged_h2d_failures=... stale_slot_reads=... sync_stage_mode=...
```

`slot_reuses` counts selection of a staging slot with a prior generation.
`war_waits` counts actual per-slot event synchronizations. `stale_slot_reads`
is limited to a debug-observed attempt to reuse a staging slot whose completed
generation does not match; it is not a resident-cache payload checksum.

## One-shot fault injection

Each truthy variable injects exactly the first eligible failure in a fresh
process and then disarms:

| Operation | Environment |
|---|---|
| Staging event create | `DS4_CUDA_FAULT_STAGE_EVENT_CREATE=1` |
| Staging event record | `DS4_CUDA_FAULT_STAGE_EVENT_RECORD=1` |
| Upload event create | `DS4_CUDA_FAULT_UPLOAD_EVENT_CREATE=1` |
| Upload event record | `DS4_CUDA_FAULT_UPLOAD_EVENT_RECORD=1` |
| Upload event wait | `DS4_CUDA_FAULT_UPLOAD_EVENT_WAIT=1` |

The create/record/wait injections use the same production recovery branches as
real CUDA errors. All are OFF by default.

## Debug generation and poison

`DS4_CUDA_TRANSPORT_DEBUG=1` tracks a monotonically increasing generation per
staging slot, verifies completion before host reuse, and poisons the host slot
with `0xa5` immediately before the full `pread`. A successful `cuda_pread_full`
must overwrite the requested payload before it can become an H2D source.

This instrumentation does not copy device weights back and does not compare a
source/destination digest. It can expose staging-slot lifetime violations, but
it is not evidence of end-to-end payload integrity.

## Harness

`runs/ds4/20260713_0051_transport_gate_pod3090/harness/phase1_transport_safety.sh`
runs a fresh CLI process for every combination of stage depth `1/2/8` and
baseline plus the five one-shot event faults. It fixes the resident cache at
six experts, enables deferred prefill with next-layer prediction disabled,
requires the expected final counters/fallback signatures, and requires every
stdout SHA256 to match the first arm.

The harness was syntax-checked with Git Bash `bash -n`; it was not executed.

## Authoring and validation

The patch was generated from an isolated source tree. The shared DS4 source was
not modified.

| Checkpoint | `ds4_cuda.cu` SHA256 |
|---|---|
| Frozen available 0050 source | `dd48371b2b6d46cc956bb5c6f2ce2d22943334ddd7a7245bb4eea6265bc1b0bf` |
| After 0050j | `91c6d635099f87ab30c25d78c44366331fbd5344229ba3cac1d8a0ea4c18240a` |
| After 0050k | `10e21d340702df127e2398aba35923833b5eb47e9cc025af3a7beec32bf79fe2` |

- Fresh temporary Git tree: archived 0050 source + 0049 + 0050j + 0050k
  passed `git apply --check --whitespace=error-all` at every step.
- The fresh-chain post-0050k CUDA file is byte-identical to the authoring tree.
- `git diff --check`: pass.
- Patch file SHA256 before commit:
  `7a2411eebe6c3002fc9e6f640b886bf6701b21e1ee5efbb769d69739ada7183c`.
- CPU build was not available: this Windows host has no GNU `make`, `gcc`, or
  `clang`.
- An `sm_86` compile was attempted with CUDA 12.6 and Visual Studio 2022, but
  this Linux/POSIX translation unit stops on the unavailable `unistd.h` (and
  its GNU/POSIX dependencies). No CUDA compile pass is claimed.

The archived source reproduces the 0050j CUDA and GPU-header anchors exactly.
Its `ds4.c` becomes
`d9a825aabb53261f45a7b81a6e49c4bcd356c18e1b738c9a5fa7e4f8d10ff3ca`
after 0049, not the `c4e88c...` value recorded in the 0050j notes; the complete
0050j patch still applies cleanly. Since 0050k changes only `ds4_cuda.cu`, its
owned apply chain and endpoint are exact, but the unavailable authoring
snapshot for that `ds4.c` hash remains a provenance gap.

## Remaining gates

1. Run the harness on the intended Linux CUDA host and inspect all 18 arms.
2. Run `compute-sanitizer --tool racecheck` on baseline and fault fallback.
3. Complete native Linux CPU and `CUDA_ARCH=sm_86` builds.
4. Add an actual source-to-device transport checksum or validated device
   copyback before making any checksum claim.
