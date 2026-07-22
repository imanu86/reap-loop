# 0050l transport fail-closed

Status: incremental hardening patch for
`0050k-transport-safety-fallback.patch`. Apply it only after 0050k.
GPU runtime validation is pending. This patch does not add or claim a
transport checksum.

```bash
git apply --check --whitespace=error-all \
  /path/to/0050l-transport-fail-closed.patch
git apply --whitespace=error-all \
  /path/to/0050l-transport-fail-closed.patch
```

## Fail-closed behavior

- `cuda_transport_drain_upload_stream` clears deferred upload and staging-slot
  in-flight state only after `cudaStreamSynchronize` succeeds. A failed drain
  increments `fallback_sync_failures`, preserves the in-flight bookkeeping,
  and returns 0.
- A failed post-batch completion record/drain invalidates the selected cache
  and returns 0 unconditionally. `strict_failure == false` no longer converts
  an unknown completion state into success.
- `DS4_CUDA_FAULT_STAGE_EVENT_WAIT=1` injects one failure at the eligible
  staging WAR `cudaEventSynchronize`. The production failure branch increments
  `war_wait_failures` exactly once, latches synchronous staging, and uses the
  fail-closed recovery drain.
- The synchronous staging latch uses a compare-and-swap transition. Therefore
  `sync_mode_latches` counts process-wide transitions into sync mode rather
  than calls that observe an already-latched mode.
- `ds4_gpu_cleanup` checks `cudaDeviceSynchronize`. A failure increments
  `cleanup_device_sync_failures` and logs the CUDA error before the final
  transport report is emitted.

## Sync counters

The common drain now classifies every synchronization attempt:

- `recovery_syncs`: drains entered because an event, debug lifetime check, or
  staged H2D operation failed;
- `sync_mode_syncs`: normal per-copy drains after synchronous staging has been
  latched;
- `fallback_syncs`: compatibility total retained from 0050k.

For all calls through `cuda_transport_drain_upload_stream`:

```text
fallback_syncs == recovery_syncs + sync_mode_syncs
```

`fallback_sync_failures` remains the failure count across both classes.

The final diagnostic line now includes all of the following fields:

```text
fallback_syncs=... recovery_syncs=... sync_mode_syncs=...
sync_mode_latches=... fallback_sync_failures=...
cleanup_device_sync_failures=...
```

The existing stage, event, H2D, stale-slot, and `sync_stage_mode` fields remain
in the same report.

## One-shot WAR fault

`DS4_CUDA_FAULT_STAGE_EVENT_WAIT` follows the same truthy and one-shot rules as
the five 0050k event faults. It is OFF by default. It is evaluated immediately
before the eligible WAR event synchronization. An injected wait failure has
these direct counter effects before later sync-mode work:

```text
war_waits += 1
war_wait_failures += 1
recovery_syncs += 1
fallback_syncs += 1
```

It also causes one `sync_mode_latches` increment when sync mode was not already
latched. If the recovery drain fails, `fallback_sync_failures` increments,
deferred/in-flight state is retained, and the selected copy returns failure.

## Checksum gap

No source-to-device digest, device copyback, or payload comparison is added.
The generation and poison diagnostics inherited from 0050k can expose staging
slot lifetime mistakes, but they cannot prove end-to-end transport integrity.
An actual transport checksum remains a separate, unimplemented gate.

## Authoring and validation

The patch was generated from an indexed post-0050k source tree in a temporary
directory outside the repository. A second fresh external directory replayed
the frozen chain:

```text
archived 0050 source -> 0049 -> 0050j -> 0050k -> 0050l
```

Every step passed `git apply --check --whitespace=error-all` and applied with
the same whitespace policy. In particular, 0050l applies cleanly immediately
after `0050k-transport-safety-fallback.patch`. The replayed result is logically
identical to the authoring result.

| Checkpoint | `ds4_cuda.cu` canonical-LF SHA256 | Git blob |
|---|---|---|
| After 0050k | `10e21d340702df127e2398aba35923833b5eb47e9cc025af3a7beec32bf79fe2` | `c16489c39a08a809939f1e5e27345e418098c234` |
| After 0050l | `f47c279ac50e83100f7835b5fc65bc6292a442395bb2e5e310b537f1b37b55fe` | `ca1d5a1bc92450c762afe89a76e658288e33ca00` |

- Incremental patch stats: one source file, 63 insertions, 19 deletions.
- `git diff --check`: pass in the authoring tree.
- Patch SHA256:
  `6865b801d2aa0c21e98362b814dffe50d35d15cb4fd417941d4a49d7eb966f45`.
- Local project build: not executed because this Windows host does not provide
  the required Linux/POSIX build environment. A direct CUDA 12.6 `nvcc`
  preflight stopped at `ds4_cuda.cu:17` because `unistd.h` is unavailable; it
  is not counted as a syntax or build pass. No further toolchain search was
  performed.
- No GPU workload, WSL command, pod, or runtime harness was used.

## Remaining gates

1. Compile on the intended Linux CUDA toolchain.
2. Exercise the new WAR wait fault at stage depths that guarantee a WAR wait.
3. Inject or reproduce a failed recovery drain and verify retained in-flight
   state plus hard failure.
4. Add a real source-to-device checksum or validated copyback before making an
   end-to-end payload-integrity claim.
