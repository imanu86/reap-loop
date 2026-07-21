# Live-Mask Promoter/Reaper Design

Status: implementation specification only. This document does not change runtime code and does not authorize a GPU run.

## Decision

Implement the live mask as a request-scoped, token-clocked controller over exact IQ2 experts keyed by `(layer, expert)`. The three externally visible tiers are:

1. **VRAM hot** — an exact IQ2 expert has a published device-cache slot and runs the existing IQ2 GEMM path.
2. **CPU warm** — the authoritative exact IQ2 bytes are still the primary-model mmap; the expert has completed a recent lane-A CPU forward (or an equivalent verified background warm), so its pages are expected to be in the Windows page cache.
3. **mmap cold** — the same authoritative mmap bytes exist, but no page-residency assumption is made. A selected cold expert still runs exactly on lane A for the current token and is marked as a future promotion candidate.

`CPU warm` is deliberately a logical residency state, not another full expert copy. The design adds no model mmap and does not recreate G129's multi-GiB exact-IQ2 probation arena. The mmap remains authoritative in every state. A reaped VRAM expert is not deleted; only its device slot is removed.

Two internal transactional substates are needed: `PROMOTING` and `REAP_PENDING`. They are never dispatchable representations. Dispatch uses the last fully published state until the corresponding event completes.

The controller is decode-only. Existing prefill-mass seeding may populate the initial exact-IQ2 device cache. Decode heat takes over after prefill, with an eight-token grace for prefill-seeded slots so they are not reaped before decode has enough evidence.

## Fixed invariants

- Expert identity is `(layer, expert)`, not the expert number alone. There are 43 × 256 = 11,008 entries; routed layers remain the existing 3 through 42.
- Route hits are scored against the residency snapshot **before** the current token is observed. Heat and promotions are committed after the token. This preserves the causality used by the working-set pilot.
- `P=8` is a global decode-token budget, not a per-layer budget and not a cache-size target.
- A cold route cannot enter VRAM in the call that first observes it. The current token runs on CPU; the earliest device eligibility is the next token.
- There is no cold mmap/SSD-to-VRAM shortcut. Exact bytes must first be consumed or verified into the warm path.
- A published VRAM entry is immutable until an event-fenced reap. Claimed, loading, pending, or current-token slots are not victims.
- A promotion failure, stale job, or queue-pressure drop leaves the expert CPU-dispatchable. It cannot poison exact inference or force decode to wait.
- The primary model stays on `C:`. No new model copy or cross-volume staging is introduced. Background staging uses Windows background/low-I/O priority.
- The lane-A and promoter lanes may coexist. At P=8, 8 × 7,077,888 bytes is 56.6 MB/token, matching the passed 56 MB/token contention arm. The 56–112 MB/token arms degraded lane A by 10.0–11.5%, below the 15% gate; this is a budget justification, not permission to run staging unthrottled.

## State machine

| Current state | Event/guard | Action | Next published state |
|---|---|---|---|
| `MMAP_COLD` | route selected | Run exact IQ2 lane A directly from the existing mmap; record `cold_route=1`. Do not issue H2D for this route. | `CPU_WARM` after the CPU job succeeds; otherwise remain cold/fail open per lane-A smoke policy |
| `CPU_WARM` | route selected | Run exact IQ2 lane A; refresh CPU-warm epoch and observe touch/weight. | `CPU_WARM` |
| `CPU_WARM` | top-P admission, source generation valid, next-token guard satisfied | Reserve a device slot and pinned bounce slot, enqueue exact IQ2 H2D, and charge one P-budget unit exactly once. | `PROMOTING` internally; `VRAM_HOT` only after event and map publication |
| `PROMOTING` | copy and publish event succeeds | Publish `(layer, expert, generation, offsets) -> slot`. | `VRAM_HOT` |
| `PROMOTING` | stale/drop/failure | Clear reservation, refund the charged budget exactly once, retain authoritative mmap. | `CPU_WARM` or `MMAP_COLD` according to the last successful CPU-warm epoch |
| `VRAM_HOT` | route selected | Dispatch the existing exact IQ2 GPU route; update heat, age, and claimed-slot protection. | `VRAM_HOT` |
| `VRAM_HOT` | cooled below hot definition, or qualified hotter entrant needs its slot | Record an event after all consumers, remove the key only when safe, and release/reassign the slot. | `CPU_WARM` if a CPU forward completed in the last eight tokens; otherwise `MMAP_COLD` |
| `CPU_WARM` | no successful CPU use for eight tokens | Drop only the logical warm assertion. Do not unmap or synchronously trim pages. | `MMAP_COLD` |

The controller must be single-owner at token commit. Route threads may append observations, but only the commit owner may rank candidates, charge P, choose/reap slots, or publish state. This avoids a second policy lock inside lane-A workers.

## Heat signal

### Token aggregation

Collect all routed-layer observations for a decode token, then update once at token end. For expert `e=(layer, expert)` and token `t`:

- `u[e,t] = 1` if the expert was routed at least once in token `t`, otherwise `0`;
- `w[e,t] = sum(abs(router_weight))` for that expert in token `t` (normally one route);
- `touches8[e,t] = sum(u[e,t-i], i=0..7)`;
- `max_weight8[e,t] = max(w[e,t-i], i=0..7)`.

Keep an exact eight-row ring. The EWMA decay is `beta = 2^(-1/8) = 0.917004...`; a sample reaches half-weight after eight token steps and is then dropped by the bounded window. The efficient recurrence is:

```text
touch_ewma[t]  = beta * touch_ewma[t-1]  + u[t] - beta^8 * u[t-8]
weight_ewma[t] = beta * weight_ewma[t-1] + w[t] - beta^8 * w[t-8]
heat[t]        = touch_ewma[t] + weight_ewma[t]
```

The common normalization denominator is omitted because it does not change rank or hysteresis comparisons. Touches dominate, gate mass distinguishes equally frequent experts, and beta encodes recency. This is the EWMA equivalent of the pilot's lexicographic recent-touch, mass, recency ranking.

The reported 81.7% is not an EWMA half-life: it is pooled gate mass that reappeared within 16 tokens. W=8 is retained because its prior-expert union covered 75.3% median next-token mass and because the P=8 feasibility simulation used exactly that window.

### Hot eligibility and reap threshold

An expert is promotion-eligible only when all are true:

- published state is `CPU_WARM`;
- `touches8 >= 2`;
- `max_weight8 >= 0.02`;
- exact source generation and gate/up/down offsets still match the registered primary mmap;
- it is not already resident, promoting, current-token cold, or involved in a pending reap;
- it is not excluded by a claimed/in-flight slot or request teardown.

The primary reap threshold is the inverse of the hot definition: a resident with `touches8 < 2` is cooled and is scheduled for reap at token commit. A currently claimed slot is reaped as soon as its consumer fence permits.

When the cache is full, retain the existing 1.25 hysteresis concept: choose the lowest-heat unclaimed resident and replace it only if

```text
entrant_heat > 1.25 * victim_heat
```

Equivalently, for a given entrant, the replacement reap threshold is `victim_heat < entrant_heat / 1.25`. Exact ties are stable: prefer the existing resident, then break ranking ties by newer touch, lower layer, and lower expert ID. Free-slot admission does not require the 1.25 margin.

This combines the useful tiered-hysteresis behavior (a 25% anti-ping-pong margin) with the DS4 mass-aware ranking. Do not port Colibri's integer `tier_pick_lfru` score: its frequency-dominant `(heat << 8) | recency` ordering loses gate-mass information and scans residency inefficiently for 11,008 entries.

## P=8 promotion/reap policy

At the end of each decode token:

1. Finish the W=8 ring update for all touched and expiring entries.
2. Mark residents with `touches8 < 2` as reap candidates. Fence them; do not synchronously wait.
3. Build the eligible nonresident set and sort by `heat`, last-touch token, layer, expert.
4. Consider at most the first eight missing experts. A cache hit, deduplicated pending job, or already-resident entry does not consume P. An accepted job consumes one of eight decision slots for that token even if it later drops and refunds byte/budget accounting; do not retry a ninth candidate in the same token.
5. Use empty, nonreserved device slots first. Otherwise pair each entrant with the coldest legal victim and apply the 1.25 test. If no legal victim exists, skip; P is a ceiling, not a mandate.
6. For each accepted promotion, capture immutable model generation and effective gate/up/down ranges, reserve the destination, charge one budget unit, and enqueue through the low-priority staging/H2D path.
7. Publish the key-to-slot mapping only after the copy event. A success may be used starting with token `t+1`; a completion that misses that boundary simply remains unavailable until a later route.
8. Classify stale work as an accounted drop, structural errors as fail-closed for the promoter, and internal errors as promoter-terminal for the request. In all non-success cases, exact CPU dispatch remains available.

Budget accounting follows the fixed SSD-WRAP semantics: charge immediately before publishing a queued attempt; pair every attempt with exactly one terminal result; refund a stale/drop/failure once; never double-count a worker completion and a teardown sweep.

Prefill-seeded exact slots retain their existing source and cache identities. They get an eight-decode-token grace, but they do not bypass heat ranking forever. At the end of the grace they either meet the live threshold or enter the normal reap set.

## Lane-A dispatch

The residency map is snapshotted after selection/weight D2H and before any current-token heat update. Each selected route is classified independently:

| Resolution | Current-token dispatch | Observation |
|---|---|---|
| Valid exact-IQ2 device-map hit with matching generation | Add to the GPU exact route list and protect/claim its slot through the GEMM consumer event. | `vram_hit`, route weight |
| No device hit; live state `CPU_WARM` | Add to `cpu_selected[]`; lane A resolves the existing mmap pointers and computes exact gate/up/SiLU/down. | `cpu_warm_hit`, route weight, refresh warm epoch |
| No device hit; state `MMAP_COLD` or warm assertion expired | Add to the same lane-A CPU queue. Also set `cold_route` and `promotion_candidate_observed`; this route cannot promote in the same token. | `cpu_cold`, page-fault/latency telemetry, route weight |

The hot GPU list and CPU list launch concurrently. Lane A returns one weighted partial vector, transfers only that 4,096-float partial to the device, and joins it with the GPU output after the existing bridge event. The final live path has no representation-quality difference between warm and cold CPU routes; the distinction controls only scheduling and promotion eligibility.

The lane-A smoke retains its Q1 fail-open behavior for integration safety. Once smoke passes and the live-mask gate is enabled, Q1 overflow must be counted as a degraded safety fallback, not called `cold`: it is not exact IQ2 and is excluded from exact-quality acceptance. Normal geometry has six routes and the lane queue capacity is eight, so overflow should be zero.

Do not make dispatch query `QueryWorkingSetEx`; that would put Windows residency inspection in every routed layer. Logical warmness comes from successful CPU service, while working-set and hard-fault samples remain diagnostic checkpoints outside the token loop.

## Existing seams to reuse

Line references below identify the inspected snapshots; function names are the durable anchors.

### Reuse directly

- **Tier metadata and exact cache ownership:** `cuda_moe_tier_entry` and `cuda_moe_tiering` in `C:\Users\imanu\g130i\ds4_cuda.cu:22594` and `:22607`; `(layer,expert)` indexing at `:23202`; device-map validation in `cuda_moe_tiering_has_exact_vram` at `:23860`. Keep the exact-IQ2 `g_moe_expert_cache` as the sole hot cache.
- **Mass-LFRU slot safety:** `cuda_moe_tiering_mass_lfru_score` and `cuda_moe_tiering_pick_vram_slot` at `ds4_cuda.cu:27177` and `:27188`. Reuse empty/claimed-slot checks, replacement-budget structure, and 1.25 comparison. The live controller supplies W=8 heat and a P=8 token epoch instead of the existing 430-call score clock.
- **Publish/demote plumbing:** `cuda_moe_tiering_demote_cache_entry` at `ds4_cuda.cu:25848`, the loading/old-map removal path at `:27534`, and `moe_publish_resident_route_kernel` publication at `:27625`. Preserve generation checks and make reap event-fenced before slot reuse.
- **G129 admission invariants:** gate configuration at `ds4_cuda.cu:23109`; next-call guard and `vram_eligible_after_call` in `cuda_moe_tiering_stage_observed_quant_cold_to_2bit_ram` at `:25618-25814`. Reuse the touches≥2, weight≥0.02, budget, provenance record, and no-same-call contracts. Do not reuse its synchronous cold-to-probation copy as the live-mask storage model.
- **Fixed SSD-WRAP lifecycle:** `cuda_q1_0_ssd_wrap_worker`, `cuda_q1_0_ssd_wrap_finish_one`, poll, submit, and flush at `ds4_cuda.cu:24400`, `:24710`, `:24866`, `:24894`, and `:25016`. The `g130/ssdwrap-semantics` version is mandatory: stale work is an accounted drop, charges/refunds are exactly once, and structural failures are counted per event. Reuse the fixed pinned H2D bounce/event idea; do not dynamically register mmap pages.
- **Sliding-ring mechanics and hysteresis:** `cuda_reap_mass_commit_pending_token` at `ds4_cuda.cu:10922`, entrant/victim ordering and 1.25 threshold at `:10974-11110`, and normalized selected-weight capture at `:12156-12235`. The live tracker is a separate W=8 instance because the current reap observer owns dynamic-arena snapshots and defaults to W=16.
- **Prefill prior:** `cuda_prefill_mass_observer` at `ds4_cuda.cu:2084`, `cuda_prefill_mass_observe_selected` at `:12061`, finalize at `:11688`, and `cuda_moe_prefill_vram_seed` at `:26237`. Reuse its mass/count input and existing seed; add only the bounded decode grace/hand-off.
- **F1 event ordering:** `g_q1_upload_ready` and `cuda_moe_selected_load_q1_0` in the F1/F2 lane snapshot, `D:\ds4_work\wt-lane-a\ds4_cuda.cu:31609-31890`. It records upload readiness on the upload stream and makes stream 0 wait on the event instead of host-synchronizing. Promotions and reaps use the same producer-event/consumer-wait pattern.
- **Lane-A exact CPU:** the current smoke snapshot defines its six-worker/eight-route state at `D:\ds4_work\wt-lane-a\ds4_cuda.cu:769-840`; exact IQ2 forward at `:34301`; mmap range resolution and dispatch preparation at `:34550` and `:34568`; input bridge and output join at `:34664` and `:34690`. Reuse these functions rather than adding another CPU pool, mmap, or IQ2 decoder.
- **Mixed dispatch seam:** `ds4_gpu_routed_moe_mixed_q1_0_one_tensor`; in the lane-A snapshot its three arrays and classification are at `ds4_cuda.cu:35179-35317`, CPU preparation is at `:35431`, hot exact launch at `:35517`, and Q1 safety fallback/join begins at `:35612`. Extend this classification to warm/cold CPU states and place heat observation after the immutable dispatch decision.

The F2 `cuda_q1_vram_lru` is an approximate-Q1 transport cache and is not the live mask's exact-IQ2 hot tier. Its useful generation-tagging and event-fenced reuse ideas may be copied, but its device allocation and keys must not compete with or masquerade as `g_moe_expert_cache`.

### Genuinely new code

1. `cuda_live_mask_promoter` request state: W=8 ring, touched/expiring sparse lists, per-entry logical state, last CPU-warm epoch, prefill grace, and token-local top-eight heap.
2. A true token-boundary commit hook. Existing route-worker `call_tick` is per routed-layer call and cannot implement P/token. Reuse the reap observer's layer rollover detection, but expose one authoritative decode token epoch.
3. A three-way resolver returning `VRAM`, `CPU_WARM`, or `CPU_COLD` without treating snapshot/probation RAM as the lane-A source.
4. Mmap-warm promotion transport: copy the three verified expert spans into a fixed pinned bounce slot on a background-priority worker, then H2D and publish with events. This adapts SSD-WRAP mechanics to mmap/page-cache source bytes; it is not the existing probation-arena loader.
5. Event-fenced proactive reap and transaction rollback that coordinates the live heat policy with `g_moe_expert_cache`'s host/device maps.
6. Live-mask telemetry and deterministic replay validation, including mass-weighted pre-observation hits and same-token/direct-cold violation counters.

## Minimal milestone 2

Milestone 2 starts only after lane-A smoke passes. Its scope is intentionally narrow:

- add the W=8 heat tracker and token boundary;
- classify lane-A exact routes as logical warm/cold;
- rank hot nonresident experts after each token;
- promote at most the top P=8 CPU-warm experts into the existing exact IQ2 device cache;
- reap cooled/replaced device entries with the 1.25 guard;
- preserve OFF-path behavior and lane-A Q1 fail-open safety;
- emit enough telemetry to reconstruct every token's pre-observation hot set, promotions, reaps, bytes, and violations.

No adaptive P, predictor, new mmap, new full RAM tier, router mask, cross-volume staging, or dynamic host registration belongs in M2.

### Acceptance metric

Primary acceptance is the same mass-weighted metric as the committed pilot, computed before observing each token and over steady-state tokens 16–63:

```text
vram_mass_hit(t) = sum(abs(weight) for routes whose exact IQ2 slot was
                       published before token t)
                   / sum(abs(weight) for all routes in token t)
```

Replay the same three 64-token traces with the runtime policy (or a bit-for-bit policy simulator fed by runtime trace rows). Accept M2 only if:

- pooled P=8 mean `vram_mass_hit >= 42.8%`;
- the equal-size first-16-token static baseline remains `<= 29.5%` (at least +13.3 percentage points, approximately +45% relative);
- every token has accepted/queued promotion attempts `<= 8`, successful promotions `<= 8`, and no hit is credited in its observation token;
- direct cold-to-VRAM violations, same-token eligibility violations, stale published slots, generation mismatches, and claimed-slot reaps are all zero;
- median published hot size fits the configured exact cache (pilot median was 153; current cache320 has headroom);
- in the replay1→replay2 switch, the first token reaching 90% of replay2's in-domain P=8 hit is no later than token 27 (trailing-four no later than 28).

The runtime smoke additionally requires nonzero `cpu_warm`, `cpu_cold`, `vram_hit`, promotion, and reap counts; exact route partition accounting; zero normal-path Q1 overflow; and no promoter-induced host synchronization. Throughput and quality remain separate lane-A/full-system gates and are not replaced by the trace metric.

## Required telemetry

Per request summary, gated consistently with existing profiling:

- decode tokens and authoritative token epoch;
- route count and gate mass by `vram`, `cpu_warm`, `cpu_cold`, and `q1_safety_fallback`;
- hot candidates, eligibility skips by state/touches/weight/provenance/claimed slot;
- promotion considered/accepted/queued/succeeded/dropped/failed/refunded, max per token, and bytes;
- free-slot admissions, hysteresis replacements, cooled reaps, replacement reaps, and delayed claimed reaps;
- cache capacity/count and prefill-grace count;
- staging queue depth, service time, page-fault deltas at out-of-loop checkpoints, H2D event waits, and lane-A join waits;
- exact structural counters for same-token, cold-to-VRAM, generation, source-range, publication, and reap-fence violations.

Detailed per-expert rows are O(promotions + reaps), not O(routes), and use immutable request epoch, token epoch, layer/expert, heat components, source generation/ranges, victim identity, slot, first eligible token, and terminal status.

## Risks

1. **CPU lane is still the throughput floor.** The promoter only converts a fraction of routes to GPU. A 42.8% mass hit does not imply the same count hit or a 42.8% latency reduction; lane-A joins and the slowest layer determine realized throughput.
2. **Logical warm can disagree with physical page residency.** Windows may evict file-backed pages between CPU use and promotion. A staging hard fault must remain background work; it must not turn token commit into a blocking SSD read.
3. **Token-boundary mistakes invalidate P=8 and causality.** Using the existing routed-layer `call_tick` would silently create P per layer. Token epoch and observation order need contract tests.
4. **Event lifetime/use-after-reap.** A device slot can feed stream 0, a promotion stream, and a later route. Reuse before all consumers fence is a correctness bug, not merely a cache miss.
5. **Two cache implementations can be confused.** F2's Q1 VRAM LRU and the exact IQ2 cache coexist. Telemetry, keys, and allocations must label the representation explicitly.
6. **Mmap pointer provenance.** Lane A and promoter must validate source generation, file size, tensor type, and all three effective ranges. A sidecar/model rebind invalidates pending work and resident keys.
7. **Background-priority behavior is not presently a DS4 CUDA helper.** The contention bench used Windows background threads, but production wiring is new. Priority setup and restoration need platform-specific failure handling.
8. **Budget/drop races.** The SSD-WRAP fixes show how easy it is to double charge, double refund, or publish after terminal teardown. Reuse that ownership model and test stale, cancellation, and shutdown paths.
9. **Domain switch lag.** P=8 measured about 27 tokens to recover. Do not hide that by temporarily raising P; report it. P=32 is a later, separately gated headroom experiment.
10. **Prefill bias can delay adaptation.** The seed grace must expire deterministically. Prefill entries cannot be permanently protected from live heat.
11. **DRAM/NVMe contention evidence is bounded.** The passed bench covers the measured C: device and 56–112 MB/token pacing. Different model placement, antivirus activity, memory pressure, or an unthrottled worker can invalidate it.
12. **Safety fallback can mask quality failure.** Q1 overflow or lane failure is acceptable for smoke continuity but must be zero in exact live-mask acceptance and reported separately from mmap cold.

## New-code effort ranking

Ranked highest effort first, assuming the lane-A smoke code is retained:

1. **Mmap-warm → pinned bounce → exact VRAM transaction, including failure/shutdown semantics:** high, approximately 3–4 engineering days. This carries the most concurrency and provenance risk.
2. **Three-way mixed dispatch plus event-fenced promotion/reap integration with the exact cache:** high, approximately 2–3 days. The work is concentrated but correctness-sensitive.
3. **Token boundary, W=8 heat state, top-P ranking, and prefill hand-off:** medium, approximately 1–2 days. The algorithm is small; proving token causality is the work.
4. **Telemetry, deterministic trace replay, and fault/contract tests:** medium, approximately 1–2 days. Do not defer this; it is the M2 acceptance mechanism.
5. **Configuration and lifecycle plumbing:** low, less than one day. Use one milestone gate with frozen W=8/P=8/1.25 values rather than adding a permanent matrix of semantic variants.

The critical path is items 1 and 2. Heat bookkeeping itself is not the risky part; safe movement and reuse of exact expert slots is.
