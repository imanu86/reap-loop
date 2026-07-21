# Lane A redesign: resident-only bounded CPU expert lane

## Decision

The G132 lane-A worker/overlap implementation is reusable, but its admission decision is inverted. At commit `8f76a05` (`g132/lane-a-smoke`), `q1_resident` is promoted to `iq2_cpu_exact`, and the CPU worker is then pointed at the original 86 GB IQ2 mmap. `q1_resident` proves only that the Q1 fallback bytes are in the Q1 arena; it does **not** prove that the exact IQ2 gate/up/down bytes are in host RAM. The direct mmap pointer can therefore fault each expert from disk.

Lane A must instead take at most a small, bounded subset of routes already classified as exact IQ2 in host RAM (`iq2_snapshot_ram` or `iq2_tier_ram`), obtain their pointers from the validated primary-IQ2 arena slot, and leave every other route on its existing GPU/H2D or Q1 path. It must never dereference the main-model mmap on the CPU inference path.

Source anchors in this document refer to `D:\ds4_work\wt-lane-a\ds4_cuda.cu` at `8f76a05`.

## What failed in the current cut

The exact failure chain is visible in four anchors:

1. The resolver returns `q1_resident` only after determining that the tier is `SSD_COLD` and `has_2bit_ram == 0`, then finding resident **Q1** pointers (`ds4_cuda.cu:34923-34929`).
2. Lane A converts every such route to `iq2_cpu_exact`, up to the compile-time route-array capacity of eight (`ds4_cuda.cu:35273-35279`). There is no latency-derived per-layer cap.
3. `cuda_g132_cpu_lane_prepare_dispatch` obtains gate/up/down pointers with `cuda_g132_cpu_lane_source_ptr` (`ds4_cuda.cu:34575-34612`).
4. That helper explicitly points into `model_map + base_offset + expert_offset`, with the comment that the CPU consumes the main-model mmap directly (`ds4_cuda.cu:34555-34572`). The CPU kernels then scan those three expert ranges (`ds4_cuda.cu:34317-34345`).

This explains the measured behavior: approximately 40 of 43 layers per token perform CPU reads from an over-capacity, cold mmap working set, producing roughly 20-23 ms/expert page-fault stalls. The 0.83 ms standalone result measured a different condition: already-hot pages.

## 1. Residency check and authoritative gate

### The exact field

The semantic field is `cuda_moe_tier_entry::has_2bit_ram` (`ds4_cuda.cu:22756-22767`, specifically line 22764). It is set when an exact IQ2 expert has been committed into the primary RAM arena (`ds4_cuda.cu:25568-25577`, and the asynchronous promotion commit at `ds4_cuda.cu:24964-24983`) and cleared when that RAM backing is reclaimed (`ds4_cuda.cu:25540-25542`). The existing route trace prints this same field as `has_2bit_ram` (`ds4_cuda.cu:35300-35311`).

`has_2bit_ram` is necessary, but the robust current-residency test is the field plus a live arena binding. The arena can reuse a slot, so a boolean alone is not a safe pointer lease. The existing pointer resolvers already perform the required identity/generation checks:

- Snapshot RAM: `cuda_moe_tiering_snapshot_ram_ptrs` requires primary backing, matching snapshot generation, an active binding, and a `READY` slot (`ds4_cuda.cu:23970-23998`).
- Tier-owned RAM: `cuda_moe_tiering_ram_ptrs` requires the tier's `ram_slot`, matching layer/expert and `ram_generation`, and a `READY` or `STAGED` slot (`ds4_cuda.cu:24001-24019`).
- The lower-level binding validation checks binding state, snapshot generation, slot state, layer, expert, and content generation (`ds4_cuda.cu:6086-6102`).

The mixed resolver already turns those successful pointer checks into `iq2_snapshot_ram` and `iq2_tier_ram` (`ds4_cuda.cu:34909-34921`). Those are the only current host-IQ2 representations eligible for lane A. `iq2_vram` stays on GPU. `q1_resident` is never CPU-IQ2 eligible.

There is one further physical-residency distinction. A dynamic-arena slot records `cuda_dynamic_arena_slot::pageable` (`ds4_cuda.cu:1079-1088`). Arena construction assigns either `cudaHostAlloc`-backed pinned storage (`pageable == 0`) or `VirtualAlloc` pageable overflow (`pageable == 1`) (`ds4_cuda.cu:7188-7195`, `ds4_cuda.cu:7210-7226`). An exact IQ2 copy in pageable overflow can still be paged out. To make “never cold-from-disk” a hard Lane-A invariant, the CPU resolver must require a valid exact-IQ2 arena pointer **and** `slot.pageable == 0`. If later profiling proves pageable slots are locked/resident by another mechanism, that class can be admitted explicitly; it must not be assumed resident from `has_2bit_ram` alone.

Recommended helper contract:

```text
cpu_resident_iq2_ptrs(layer, expert, representation,
                      &gate, &up, &down, &slot_identity) -> bool

true only when:
  representation is iq2_snapshot_ram or iq2_tier_ram
  tier.has_2bit_ram == 1
  tier.state is RAM_PROBATION or RAM_WARM
  the corresponding existing pointer resolver succeeds
  the resolved primary-IQ2 slot has pageable == 0
```

The helper should return the arena pointers themselves. `cuda_g132_cpu_lane_prepare_dispatch` must store those pointers in its existing `gate_ptrs`, `up_ptrs`, and `down_ptrs` arrays rather than call `cuda_g132_cpu_lane_source_ptr`. Revalidate the slot identity/generation when preparing the dispatch. The existing ordering is safe for the current token: CPU dispatch is joined before the future-token promotion loop begins (`ds4_cuda.cu:35787-35790` versus `ds4_cuda.cu:35928-35946`).

### Route decision table

| Current representation | Exact IQ2 location | Lane-A action |
|---|---|---|
| `iq2_vram` | VRAM | Keep in `hot_selected`; GPU compute. |
| `iq2_snapshot_ram` | Primary exact-IQ2 snapshot arena | CPU-eligible only with valid binding, `has_2bit_ram`, pinned slot, and remaining cap; otherwise keep in `hot_selected` for existing H2D/GPU compute. |
| `iq2_tier_ram` | Tier-owned exact-IQ2 RAM slot | Same rule as snapshot RAM. |
| `q1_resident` | Q1 fallback arena; exact IQ2 is cold | Keep in `cold_selected`; current Q1 GPU fallback. Optionally enqueue future-token prefetch/promotion only. |
| unresolved/cold | No valid current representation | Preserve the existing fail-open/error or cold path; never make it a CPU job. |

## 2. Per-layer CPU cap

Add an environment-controlled cap, proposed as `DS4_G132_CPU_LANE_MAX_EXPERTS_PER_LAYER`, parsed once during `cuda_g132_cpu_lane_init` (`ds4_cuda.cu:34396-34469`). Clamp it to `0..min(CUDA_G132_CPU_LANE_MAX_ROUTES, worker_count)`. `0` is a useful no-CPU control. Use **2** as the initial default, then ship the largest value demonstrated to remain hidden by the acceptance run.

The admission rule is:

```text
N_cpu(layer) <= N_max
and
N_max * resident_cpu_expert_ms_p95 < gpu_branch_ms_p50(layer) - join_margin_ms
```

Use a 0.5 ms join/scheduling margin. With the measured hot-cache spike of 0.83 ms/expert and a 3-4 ms GPU branch, `N_max=2` is initially defensible. If in-process pinned-arena p95 is near 3 ms/expert, the cap must be reduced to 1; a nominal default must not override the measured inequality. Test 0/1/2/3 and select the highest passing value.

Apply the cap in the classification loop at `ds4_cuda.cu:35240-35332`, not in the worker after routes have been removed from GPU. Prefer `RAM_WARM` over `RAM_PROBATION`; within the same state, preserve router order for a minimal deterministic change. Every eligible resident route beyond the cap remains in `hot_selected`, so `routed_moe_launch` retains the existing H2D/GPU behavior (`ds4_cuda.cu:35573-35597`). Increment separate counters for `cpu_admitted`, `cpu_cap_rejected`, `cpu_not_pinned`, and `cpu_not_exact_iq2`; do not reuse the current `overflow_q1_routes` name because these are exact-IQ2 host routes, not Q1 overflow.

The cap is per layer call, not per token globally. The CPU worker pool and compile-time array capacity remain safety ceilings, not scheduling policy.

## 3. Nonblocking prefetch fallback

Lane A must not synchronously test or warm mmap pages in the token-critical classification/compute path. In particular, do not use `QueryWorkingSetEx` per route and do not touch one byte per page as a “check”; either operation defeats the invariant it is trying to enforce.

For a `q1_resident`/cold route that should be offered to the future promoter:

1. Execute the current token through the unchanged Q1 or existing H2D/GPU path.
2. Enqueue a bounded, deduplicated `(layer, expert, gate/up/down ranges, observation_call)` request to a background promoter/prefetch worker. Queue-full is a harmless skip.
3. On Windows, that worker issues one `PrefetchVirtualMemory(GetCurrentProcess(), 3, ranges, 0)` call for the exact main-mmap gate, up, and down expert ranges. Validate offset arithmetic exactly as `cuda_g132_cpu_lane_source_ptr` currently does, but do not expose those mmap pointers to CPU inference.
4. Return immediately to decode; do not wait for prefetch completion and do not reclassify the current token.
5. The promoter later copies the exact IQ2 triplet into a pinned primary-arena slot and commits slot identity/generation/state before setting `has_2bit_ram=1`. Only a later token may resolve it as `iq2_snapshot_ram`/`iq2_tier_ram` and become CPU-eligible.

The existing code already states the correct temporal contract: current Q1 output is queued first and promotion is future-token work (`ds4_cuda.cu:35928-35946`), and the promotion machinery uses a next-call eligibility guard (`ds4_cuda.cu:25824-25837`, `ds4_cuda.cu:25934-25937`). Preserve that ordering. However, without the SSD-wrap path the stage helper falls through to a synchronous load (`ds4_cuda.cu:25940-25959`), whose loader performs three `pread` calls (`ds4_cuda.cu:25461-25467`). Lane-A dispatch must never call that synchronous fallback, directly or through the post-join promotion loop. In Lane-A mode, require the asynchronous SSD-wrap/promoter queue for this handoff; if it is unavailable or full, record a skipped hint and leave the expert cold. A complete cold-to-pinned promoter is Lane B scope; Lane A needs only the nonblocking handoff/hint and must fail open by leaving the route on its current path.

`PrefetchVirtualMemory` is advisory, not proof of residency. Successful submission must never directly set `has_2bit_ram` or authorize CPU compute; only a completed arena commit may do that.

## 4. Existing overlap is sufficient

No new overlap mechanism is required:

- Before hot GPU launch, Lane A records `input_source_ready`; the bridge stream waits on that event so input D2H can overlap subsequent stream-0 GPU work (`ds4_cuda.cu:34666-34689`).
- The host callback marks input ready and wakes persistent CPU workers (`ds4_cuda.cu:34385-34393`).
- The hot IQ2 branch is launched on the existing GPU path (`ds4_cuda.cu:35573-35597`), then the CPU input transfer is enqueued (`ds4_cuda.cu:35620-35647`). The Q1 branch is also queued before the final join (`ds4_cuda.cu:35730-35740`).
- At join, the host waits for CPU completion only if it has not already finished; one 16 KiB partial is copied H2D on the bridge stream, `partial_copy_done` is published to stream 0, and the partial is accumulated (`ds4_cuda.cu:34699-34748`).
- The mixed join calls that routine after GPU accumulation has been queued (`ds4_cuda.cu:35778-35790`). Per-layer `cpu_ms` and `join_wait_ms` telemetry already exists (`ds4_cuda.cu:34751-34765`).

This is the gate-3 shape needed to hide CPU work. The 20 ms cold reads made the join wait unavoidable; the redesign makes the CPU workload resident and small enough to finish before the GPU branch. A near-zero `join_wait_ms`, not another stream or event, is the success signal.

## 5. Minimal diff from `g132/lane-a-smoke`

### Change

1. Replace the incorrect `q1_resident -> iq2_cpu_exact` gate at `ds4_cuda.cu:35273-35279` with exact-IQ2 host-resident admission for `iq2_snapshot_ram`/`iq2_tier_ram`, including `has_2bit_ram`, valid slot generation, and pinned-slot checks.
2. Parse/store `N_max` and enforce it during route classification. Routes rejected by the cap stay in `hot_selected`.
3. Replace `cuda_g132_cpu_lane_source_ptr` use in dispatch preparation (`ds4_cuda.cu:34602-34610`) with the already-resolved arena gate/up/down pointers. The CPU inference path must contain no pointer into `g_model_host_base`.
4. Adjust fail-open bookkeeping for the new ownership: before hot launch, return a failed CPU admission to `hot_selected`; after launch, preserve the existing correctness fallback (a separate exact-IQ2 GPU launch or resident Q1 recompute) without ever falling back to CPU mmap.
5. Add admission/cap/pinned-source telemetry. Add only a bounded asynchronous prefetch/promoter handoff for cold routes; no synchronous read or wait.

### Keep

- CPU IQ2-XXS/Q2_K kernels and per-route reduction (`ds4_cuda.cu:34150-34351`).
- Persistent worker pool and its current core reservation (`ds4_cuda.cu:34354-34463`).
- Pinned activation D2H buffer, pinned partial buffer, and one 16 KiB partial H2D (`ds4_cuda.cu:34438-34445`).
- Existing bridge stream, three CUDA events, host callback, condition variables, join, recovery trace, and fail-open behavior.
- Existing VRAM, exact-IQ2 H2D, Q1 fallback, and future-token tier/promotion policies for all routes not admitted to CPU.

No CPU kernel rewrite, new CUDA kernel, new stream topology, or model-format change is required.

### Effort estimate

- Resident-pointer plumbing, route gate, cap parsing, counters, and fail-open adjustment: 4-6 engineering hours.
- Focused static/unit tests plus correctness tracing: 2-3 hours.
- Cap sweep and throughput/page-fault acceptance run: 3-5 hours of engineering/bench time.

Total: approximately 1-1.5 engineer-days for Lane A, excluding the full Lane-B cold-to-RAM promoter. A robust background `PrefetchVirtualMemory` queue can be included in that range if it reuses the existing SSD-wrap worker; a new general promoter worker is separate Lane-B work.

## Acceptance contract

Run the same decode workload and model as the F1 baseline and report at least three steady-state runs for cap 0/1/2/3. Ship only a cap satisfying all of the following:

1. **Source invariant:** every CPU route traces as prior representation `iq2_snapshot_ram` or `iq2_tier_ram`, `has_2bit_ram=1`, valid primary-IQ2 slot generation, and `pageable=0`. Counts for CPU-from-`q1_resident`, CPU-from-SSD-cold, and CPU-from-main-mmap are exactly zero.
2. **No demand I/O:** CPU expert execution produces no material model-file read/page-fault burst. No approximately 20-23 ms expert tail remains. Prefetch/promoter I/O is separately attributed and never joined to the current token.
3. **Resident speed:** CPU expert time is in the resident range (target 1-3 ms/expert, with p50/p95 reported per layer) and satisfies the cap inequality against the corresponding GPU branch.
4. **Overlap:** per-layer `join_wait_ms` is approximately zero (target p95 <= 0.5 ms); any layer repeatedly exceeding the budget forces a lower cap. CPU utilization during admitted work rises above the observed I/O-wait regime, while GPU work remains active.
5. **No throughput regression:** steady-state decode is **at least the F1 baseline of 4.86 tokens/s**. Lane A is rejected if it adds net token time, even if its isolated CPU kernel is faster. Cap 0 must reproduce the F1 route behavior within measurement noise.
6. **Correctness/fail-open:** routed output/recovery traces remain within the already accepted G132 numerical tolerance; injected lane preparation/bridge/worker failures produce a correct GPU/Q1 result with no dropped or double-counted route.

The governing rule is simple: CPU is an opportunistic consumer of already-pinned exact IQ2 bytes, never a mechanism for making cold bytes resident. Cold-to-RAM movement belongs to the promoter and only affects later tokens.
