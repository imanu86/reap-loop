# DS4 MoE runtime audit and ranked implementation plan

Date: 2026-07-14

Target: native Windows, RTX 3060 12 GB, 64 GB RAM, DeepSeek-V4-Flash
IQ2XXS/Q2_K model.

This document consolidates the current native-Windows findings with a
source-level audit of the other runtimes cited during the investigation. It does not turn external
benchmarks into local findings. Labels used below:

- **local measured**: produced by our current native-Windows harness;
- **source verified**: behavior is present in audited code;
- **external claim**: number reported by another project, not reproduced here;
- **proposal**: PR/RFC code or design that is not merged upstream.

## Audited snapshot

Current DS4 worktree:

- repository: `C:/Users/imanu/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/ds4-win-work`
- branch: `port/windows-dynamic-arena-0051`
- HEAD: `c4bb45d`
- warning: `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h` and `g7_measure.ps1` are modified;
  the line references in this report describe that working tree, not only HEAD.

External snapshots audited:

| Project | Commit/status |
|---|---|
| PocketMoE | `9b223cc` |
| Colibri | `bad64d1` |
| KTransformers | `7c021b4` |
| mixtral-offloading | `ce54518` |
| Fiddler | `227715b` |
| ProMoE | `022f8b2` |
| MoE-Infinity | `6285c09` |
| ik_llama.cpp | `6d78a87` |
| llama.cpp | `a4ce259` plus PRs linked below |
| PowerInfer | `8bd56d6` |
| AirLLM | `4b589bb` |
| Pregated_MoE | `6b0a666` |
| chripell/ds4-rtx3090 | `5a854d2` |

## Findings in our current path

### 1. Pinned RAM is present, but every arena hit still uploads

`cuda_dynamic_arena_copy_expert_async()` resolves a valid pinned-RAM slot and
then performs three H2D copies, gate/up/down, into the compact GPU workspace.
It explicitly accumulates `bytes_uploaded`.

Reference: `ds4_cuda.cu:2097-2168`.

Therefore a RAM-arena hit removes disk I/O but does not remove PCIe traffic or
the upload dependency.

### 2. A VRAM-cache hit still repacks and synchronizes

`cuda_moe_expert_cache_copy_to_compact_async()` performs three D2D copies from
the persistent cache slot to the compact workspace. The selected path then
synchronizes the upload stream before launching MoE.

References:

- `ds4_cuda.cu:13633-13649`
- `ds4_cuda.cu:14429-14444`

G28 measured the resulting ceiling: 328/336 resident experts, 4,602 hits, zero
expert evictions, but only 2.01 t/s in the warm `n=1` mechanism probe. The same
run identified router-ID D2H, CPU lookup/dedup, slot-to-compact D2D and the
per-layer upload sync as the remaining path.

Reference: `G28_STATIC_K8_AND_EMBED_ROW_STAGING_RESULTS.md:60-62,97-110`.

### 3. The router metadata still crosses to CPU in the general path

When packed trace data is unavailable, selected IDs and weights are copied D2H
and stream 0 is synchronized before CPU lookup.

Reference: `ds4_cuda.cu:14073-14082`.

### 4. The hard cold-to-RAM admission invariant is not complete

The current VRAM cache can allocate an LRU slot for a first miss and load it
directly as an admission in the same call. That is a persistent first-touch
promotion, not the intended policy:

`SSD/cold -> pinned RAM -> persistent VRAM only after reuse or exceptional weight`.

References: `ds4_cuda.cu:14254-14278,14368-14377`.

Transient GPU staging for exact computation is still allowed. The invariant is
about persistent residency, not about forbidding the GPU from computing a miss.

### 5. REAP mass rotation exists and is correct, but is physically expensive

The current observer ranks nonresident entrants and resident victims by sliding
mass and applies hysteresis before transactional WRAP.

Reference: `ds4_cuda.cu:3623-3727`.

G27 is an `n=3` local measurement:

- observe-only: mean 2.513 t/s;
- WRAP on: mean 2.073 t/s;
- mean WRAP time: 2.361 s;
- exact hashes in all six final runs.

Reference: `G27_REAP_MASS_WRAP_RESULTS.md:64-95`.

The policy signal is reusable. The physical movement path must become cheaper
and slower-clocked before more sophisticated policy is useful.

### 6. Some proposed work is already implemented

DS4 already has fused routed gate+up+SwiGLU CUDA kernels. Re-implementing that
idea from ik_llama or PocketMoE is not a first-order lever.

References: `ds4_cuda.cu:11616`, `ds4_cuda.cu:11772`,
`ds4_cuda.cu:13196`.

## External implementation matrix

### PocketMoE: closest mechanical reference

Repository: <https://github.com/lvyufeng/PocketMoE>, audited at `9b223cc`.

Source-verified mechanisms:

1. It recognizes the same routed recipe used by our model: W1/W3 `IQ2_XXS` and
   W2 `Q2_K` (`cpp_engine/src/dsv4_engine.cpp:7394-7417`).
2. Resident mode allocates raw-quantized expert arrays once and uploads each
   local expert into a stable ordinal (`:7430-7507`).
3. Decode can convert router IDs to resident route slots entirely on CUDA,
   avoiding the ID D2H path (`:8321-8326`).
4. The MoE kernel receives the resident arrays plus route-slot IDs directly
   (`:8192-8203`, and the IQ2XXS path around `:4740-4751`).
5. The nonresident path uses reusable pinned staging and a copy stream
   (`:7451-7460`, `:8339-8385`).
6. Its Python path contains a dynamic raw-block slot cache with LRU identity to
   slot mapping (`src/models/deepseek_v4/runtime.py:3098-3193`).
7. Prefill groups routes by expert and keeps a staged layer arena valid across
   prompt chunks (`cpp_engine/src/dsv4_engine.cpp:8664-8743`).

What transfers:

- CUDA `expert_id -> route_slot` construction;
- direct raw-block resident kernel addressing;
- quantized-block staging without FP32 expansion;
- grouped prefill organization.

What does not transfer unchanged:

- C++ resident mode assumes TP and resident local expert slices, not our dynamic
  one-GPU tier;
- the Python cache promotes a first miss immediately, which conflicts with our
  cold-to-RAM-first policy;
- the repository uses the PolyForm Noncommercial 1.0.0 license. Treat it as an
  architectural reference unless code reuse is separately cleared.

External benchmark numbers in its README are not local evidence.

### Colibri: clearest three-tier scheduler

Repository: <https://github.com/JustVugg/colibri>, audited at `bad64d1`.

Source-verified mechanisms:

- router first, placement second (`c/glm.c:2087-2185`);
- batch union of unique experts (`:2179-2185`);
- resident GPU work submitted before cold loads (`:2210-2281`);
- cold loads use parallel workers and join before reuse (`:2271-2334`);
- persistent VRAM prefix and disjoint RAM suffix (`:4343-4446`);
- bounded LFRU/heat/recency repin (`:3554-3655`);
- PILOT hides a predicted slot until load completion (`:2539-2588`);
- Windows positional I/O, `VirtualLock`, and optional no-buffering
  (`c/compat.h:109-150,238-247`).

What transfers: three-tier state machine, disjoint tiers, batch union, launch
resident work first, safe publication, slow physical adaptation.

Caveat: its cold path can compute in CPU RAM. That is valuable only if an IQ2XXS
CPU expert kernel is competitive on our CPU.

### llama.cpp PR #24524: best exact hit/miss split

Proposal: <https://github.com/ggml-org/llama.cpp/pull/24524>.

The CPU `MUL_MAT_ID` node remains the owner. Thread 0 plans cache hits and
dispatches hit rows in one GPU batch while the other CPU threads compute miss
rows concurrently. Results join into the same destination. The proposal also
contains persistent slot pools, async pinned insertion, fused gate/up pairing,
GPU-resident down-result handoff, OOM surrender and a measured baseline bailout.

Why it matters: it structurally avoids making every miss a synchronous H2D
penalty. This is the strongest fallback design if our CPU IQ2XXS kernel passes
the local microbenchmark.

Status: closed as too large and predominantly AI-generated, not disproved by an
upstream technical rejection. All reported speedups are external claims.

### vLLM RFC #38256 / PR #37190: fixed slots and persistent mapping

RFC: <https://github.com/vllm-project/vllm/issues/38256>.

Source/design elements:

- CPU pinned backing store;
- fixed-address GPU buffers;
- persistent GPU `expert_id -> slot` tensor updated in place;
- LFRU score `frequency / age`;
- unique-expert dedup for batched prefill.

Important limitation: the current staged plan has no CPU fallback and can fail
when the requested unique set exceeds cache capacity. Use its mapping and
allocation pattern, not its exact miss semantics.

### llama.cpp PR #25294: waved prefill and disk streaming

Proposal: <https://github.com/ggml-org/llama.cpp/pull/25294>.

It uses a device-side per-layer slot cache, asynchronous I/O workers and an ID
remap op. When the prefill union exceeds capacity, it processes expert waves and
sums wave outputs, loading each touched expert once per ubatch.

Transferable: wave-partitioned prefill after batch dedup.

Do not copy blindly: its current path is primarily SSD/device streaming and the
author explicitly leaves three-tier CPU hybridization for later. `O_DIRECT` is
only a candidate after measuring page-cache pollution on our machine.

### llama.cpp PR #21067 and ik_llama: overlap, not residency

PR: <https://github.com/ggml-org/llama.cpp/pull/21067>.

It prefetches layer N+1 tensor overrides while layer N computes. It requires
pinned memory (`--no-mmap` in that implementation). This is more promising for
prefill/large ubatches than batch-1 decode.

ik_llama additionally contains urgent selected-range prefetch and low-priority
full-tensor prefetch, but its implementation is effectively Linux-only. Its
fused gate/up path is already represented in DS4.

### KTransformers and Fiddler: CPU warm tier

KTransformers ranks selected experts by current router score, runs high-weight
work first, defers the rest and allows one pending deferred CPU task across a
CUDA stream boundary:

- `kt-kernel/python/experts_base.py:347-375,411-455,479-482`
- `kt-kernel/cpu_backend/cpuinfer.h:87-117`

Fiddler keeps profiled experts on GPU and computes nonresident experts on CPU.

Transferable only after the CPU kernel gate: resident-first ordering, weighted
defer and CPU/GPU join. Static profiling is not our adaptive policy.

### ProMoE and MoE-Infinity: speculative work must lose to confirmed work

ProMoE tracks complete, current, partial and missed transfers, advances a
`num_ready` frontier and cancels/skips duplicated speculative portions. Its
trained predictor conflicts with the requirement not to train on a prompt or
mask, but the transfer scheduler is still useful.

References: `src/cpp_worker/prefetcher.cpp`, especially the task classification
near the start, the confirmed-task path around `:53-123`, and ready/cancel
handling around `:559-649`.

MoE-Infinity performs speculative prefetch and corrective fetch of actual
misses (`memory/expert_prefetcher.py:106-156`,
`distributed/expert_executor.py:167-174`). Its unified scheduler currently
waits on `future.result()` at `engine/unified_transfer_scheduler.py:250-253`, so
the abstraction alone does not prove useful overlap.

Transferable: confirmed misses preempt speculation, partial completion,
cancellation and wasted-byte telemetry. This is below the data-path work.

### mixtral-offloading and Pregated_MoE: fixed buffers, but still swap-heavy

mixtral-offloading allocates pinned host storage and fixed GPU/host buffers,
serves resident experts first and overlaps incoming H2D with outgoing D2H:
`src/expert_cache.py:67-74,142-200`.

Pregated_MoE uses double buffers and a worker/future pipeline. Both eliminate
allocation churn, but still move expert payloads on misses. They are references
for buffer ownership, not the target architecture.

### PowerInfer and AirLLM: not first-order candidates

PowerInfer provides a useful CPU/GPU split-and-merge shape, but its placement is
offline/static activation-based. AirLLM streams whole layers and frees them,
which increases exchange granularity and is opposite to persistent expert
identity. Neither should lead the DS4 implementation.

### chripell/ds4-rtx3090: pinning is not direct execution

Relevant commits:

- <https://github.com/chripell/ds4-rtx3090/commit/3fc5b217ee41b03ea117b03511613c92f23b05ba>
- <https://github.com/chripell/ds4-rtx3090/commit/5a854d2d4d496df4bb845087bce3fa82661f588d>

The fork validates aggressive host pinning on native Linux/3090, but its mapped
host approach and resident-cache repack do not remove our WDDM D2D/sync path.
Its throughput is an external hardware/software result, not our baseline.

## Ranked implementation plan

The ranking is by expected value on the measured current bottleneck, then by
ability to fail cheaply and cleanly.

| Rank | Implementation | Local reason | Main source | Acceptance gate |
|---:|---|---|---|---|
| 0 | IQ2XXS warm-tier microbenchmark | Decides CPU miss vs transient H2D architecture | PocketMoE, Colibri, #24524 | Full gate/up/down expert latency and bandwidth, batch 1 plus prompt batches |
| 1 | Device slot map plus direct resident execution | G28 measured D2H+D2D+sync ceiling with zero expert evictions | PocketMoE C++ resident path, vLLM mapping, Colibri | Exact output; resident hit performs zero expert H2D/D2D and no CPU ID lookup |
| 2 | Split resident hits from true misses, one join | Removes misses from the resident critical path | llama.cpp #24524, Colibri | Hit-only, mixed and all-miss exactness; one dependency sync; no regression in all-miss case |
| 3 | Hard three-tier admission state machine | Current first miss can become persistent VRAM immediately | Colibri, vLLM LFRU, REAP mass | Cold first touch ends in pinned RAM; persistent VRAM requires second touch, mass or exceptional weight |
| 4 | Batch-union plus waved prefill | TTFT still touches/reloads too many expert payloads | PocketMoE grouped prefill, Colibri, llama.cpp #25294 | Unique expert loaded once per layer/chunk; exact prompt output; TTFT and process-read A/B |
| 5 | GPU-resident result handoff and sync collapse | WDDM launch/sync tax remains after expert transport | llama.cpp #24524, PocketMoE event path | Stage timeline proves fewer syncs; exact hash; no hidden CPU round trip |
| 6 | Slow-clock mass/LFRU physical adaptation | G27 policy correct but -17.5% on short decode | REAP G27, Colibri LFRU | Long decode/request A/B; movement cost amortized; bounded swaps |
| 7 | Confirmed-miss priority and cancelable partial prefetch | Avoids speculative work blocking exact needs | ProMoE, MoE-Infinity | Saved foreground wait exceeds wasted bytes/time |
| 8 | SPEX -> speculative CPU IQ2XXS miss execution | Uses prediction to start 1-2 likely cold experts before the exact router confirms them | existing SPEX plus Rank 0 CPU gate | Exact router remains authoritative; k=1/k=2 recall, lead time, useful CPU work, discarded CPU work and decode interference |
| 9 | SPEX/PILOT transfer prefetch | Useful only after transfers can overlap | existing SPEX, Colibri PILOT | Recall, precision, saved wait and waste; never trained on prompt/mask |

### Rank 0 is a gate, not a detour

Measure two exact ways to execute one nonresident expert:

1. pinned RAM -> compact H2D -> existing GPU IQ2XXS kernels;
2. CPU IQ2XXS gate/up/SwiGLU/down -> deterministic result join.

If CPU loses decisively, Rank 2 uses transient bulk H2D for misses while direct
VRAM hits run first. If CPU is competitive, implement the #24524-style GPU-hit /
CPU-miss split. Do not decide this from another machine's memory bandwidth.

### Rank 1 implementation boundary

Minimal first patch:

1. Allocate stable expert slot arrays in raw GGUF quantized format.
2. Maintain a device `expert_id -> slot` map, `-1` for nonresident.
3. Build route slots from router IDs on device.
4. Teach the existing gate/up/down kernels to read resident slots directly.
5. Keep the old compact selected path as exact fallback for every miss.
6. Add counters for direct hits, fallback misses, H2D/D2D bytes and syncs.

Do not combine REAP rotation, SPEX, prefill waves or a new eviction policy in
this patch. The first A/B must isolate removal of resident-hit movement.

### Rank 1 measured checkpoint: G32

Native-Windows commit `1126211` implements the device slot map, exact GPU route
resolver, persistent direct route pointers and a separate exact miss worker
behind `DS4_CUDA_MOE_GPU_RESIDENT_ROUTES=1`. The output remained exact at
`n=3` with hash
`fda564ba3f7a0f028106d468420f674898ed99ac5bf2765ac9586206e39d73c5`.

Primed RTX 3060 A/B, cache 336, prompt `Hi`, 9 generated tokens:

| Variant | Server decode t/s | Client t/s | Exact |
|---|---:|---:|---|
| G32 off | 3.273 | 2.1625 | yes |
| G32 on | 3.223 | 2.0720 | yes |
| delta | -1.53% | -4.19% | unchanged |

Across warmup plus three measured requests, G32 saw 3,832 resident hits out of
9,072 exact routes (42.24%), but only 12 of 1,512 layers were all-hit (0.79%).
The working WDDM path still paid 2.629 ms/layer for resolver synchronization
and 3.747 ms/layer waiting for the exact miss worker. GPU spin-wait deadlocked,
ordered host-callback upload deadlocked, and CUDA stream memory waits were
reported unsupported by the 3060 WDDM driver.

Verdict: Rank 1 is exact and retained as opt-in infrastructure, but it is not a
throughput win by itself. The measured low all-hit-layer rate promotes Rank 2:
resident routes must execute separately from true misses so partial hits can
pay without requiring an entirely resident layer. Full commands, failed
mechanisms and artifacts are in native-Windows
`G32_GPU_RESIDENT_ROUTE_RESULTS.md` at commit `1126211`.

### Rank 2 exact schedule

Preferred schedule when a layer contains both hits and misses:

1. device-side partition route rows into resident and miss sets;
2. launch resident rows immediately from stable VRAM slots;
3. execute misses on CPU or bulk-stage only misses from pinned RAM;
4. join once in deterministic route order;
5. update admission telemetry after compute, never before exact execution.

This preserves model semantics. Placement may lag by one token; routing does not.

### Rank 2 measured checkpoint: G33

Native-Windows commit `e4d669e` implements the exact split schedule behind
`DS4_CUDA_MOE_SPLIT_HIT_MISS=1`. The resolver publishes resident pointers and a
stable hit mask; hit kernels run while the exact worker fetches misses; miss
kernels run after one join; a final route-ordered sum preserves greedy output.
The only new persistent allocation is a roughly 96 KiB six-route down scratch,
not another expert copy.

The exactness gate and all primed requests matched hash
`fda564ba3f7a0f028106d468420f674898ed99ac5bf2765ac9586206e39d73c5`.
Two counter-ordered A/Bs compared G32 synchronous-worker control with G33:

| Samples | Variant | Server decode t/s | Client t/s | Exact |
|---:|---|---:|---:|---|
| 3 | G32 control | 3.2233 | 2.1721 | yes |
| 3 | G33 split | 3.2633 | 2.1510 | yes |
| 5 | G33 split | 3.1860 | 2.1430 | yes |
| 5 | G32 control | 3.2060 | 2.1670 | yes |

Across eight measured requests per variant, server throughput was effectively
flat (`3.2125 -> 3.2150 t/s`, +0.08%) and client throughput regressed
(`2.1689 -> 2.1460 t/s`, -1.06%). Worker-ready wait decreased from
3.865 to 3.630 ms/layer in the n=3 pair and from 3.770 to 3.701 ms/layer in
the n=5 pair, proving that overlap occurred, but extra masked launches and the
materialized final sum consumed the gain.

Verdict: Rank 2 is exact but is not a throughput win at the measured 42.24%
route-hit distribution. Keep it opt-in as infrastructure; do not enable or
compose policy into it. Full commands and artifacts are in
`G33_SPLIT_HIT_MISS_RESULTS.md` at native-Windows commit `e4d669e`. This result
motivated the isolated G34 SPEX CPU test below before physical tiering.

### Rank 3 physical states

Use explicit states, not inferred booleans:

`SSD_COLD -> RAM_PROBATION -> RAM_WARM -> VRAM_PROTECTED`.

Allowed shortcuts:

- a high current router weight can trigger urgent RAM load;
- persistent VRAM promotion still requires measured reuse/mass unless an
  explicit emergency threshold is enabled and logged;
- demotion goes VRAM -> RAM without forcing an SSD read later;
- RAM eviction selects low mass, low weight, low frequency and old recency.

This implements the intended rule that a one-off cold expert does not displace a
hot persistent expert.

### Rank 3 measured checkpoint: G35

Native-Windows commit `083c305` implements the physical state machine behind
`DS4_EXPERT_TIERING=enforce`. A first cold touch reads the exact IQ2XXS expert
once into exclusive `cudaHostAllocDefault` pinned RAM and serves that route
through a transient GPU slab. A second touch promotes it to persistent VRAM;
VRAM eviction retains the pinned RAM copy, so later recall does not return to
SSD. Routing, masks and expert bytes remain unchanged.

The counter-ordered matrix ran `off/enforce/enforce/off`, with one discarded
warmup and `n=3` measured requests per arm. Configuration was RTX 3060 12 GB,
prompt `Hi`, context 256, max 12, cache336 LRU and an 8 GiB pinned arena. All 12
measured outputs and four warmups matched exact hash
`fda564ba3f7a0f028106d468420f674898ed99ac5bf2765ac9586206e39d73c5`.

| Metric | Control | Tiering enforce | Delta |
|---|---:|---:|---:|
| Server decode | 3.2283 t/s | 4.9750 t/s | +54.10% |
| Client throughput | 2.1986 t/s | 2.8909 t/s | +31.49% |
| Process reads/run | 60.275 GiB | 31.001 GiB | -48.57% |
| Warmup | 4.201 s | 11.381 s | +7.180 s once |

Both enforce runs recorded `cold_to_ram=1005`, `cold_to_vram=0`, 4,299 RAM
hits/promotions and 3,963 VRAM demotions, with zero tier failures. This proves
the cold-to-RAM invariant and exceeds the previous 3.4 t/s local target. It also
shows excessive physical churn: mass and LFRU are measured, but G35 still
promotes on the second touch. The next isolated test is a slow-clock mass/LFRU
admission policy with hysteresis; it must preserve exactness and the zero
cold-to-VRAM invariant while reducing promotions and demotions.

Full protocol, provenance and result hashes are in
`G35_REAL_EXPERT_TIERING_RESULTS.md` at native-Windows commit `083c305`.

### Rank 8 SPEX CPU speculation test

This is a separate test from GPU/RAM prefetch. SPEX predicts expert identity;
the existing IQ2XXS CPU implementation performs the speculative computation.
The exact router is always authoritative:

1. expose only information available before the target route is known;
2. predict at most `k=1`, then `k=2`, likely nonresident experts;
3. start CPU gate/up/SwiGLU/down without changing the active mask or placement;
4. when the real router result arrives, consume matching results and discard the
   rest; never substitute a prediction for an exact routed expert;
5. compare `off`, `k=1` and `k=2` with identical prompt, cache state and build.

Record per layer/token: predicted IDs, exact IDs, prediction lead time, CPU start
and finish timestamps, useful and discarded expert computations, foreground
wait saved, total CPU time, process reads, GPU utilization and decode t/s. The
test passes only if exact hashes match, useful work finishes before demand often
enough to reduce measured foreground wait, and CPU contention does not erase the
gain. A mechanism/exactness probe may use `n=1`; any performance verdict requires
`n>=3`.

### Rank 8 measured checkpoint: G34

Native-Windows commit `63ba10d` implements an observe-only SPEX CPU sidecar for
`k=1|2`. It copies the predicted target-layer hidden state to an eight-slot
pinned ring, quantizes it to Q8_K on one CPU worker, and evaluates predicted
IQ2XXS gate/up plus SwiGLU. The result is checksummed but never consumed; the
exact GPU router remains authoritative. The harness also pins the SPEX model
SHA-256 and fails closed on provenance or output mismatch.

The counter-ordered matrix used one discarded warmup and `n=3` measured requests
per run. All 24 outputs matched exact hash
`fda564ba3f7a0f028106d468420f674898ed99ac5bf2765ac9586206e39d73c5`.

| Width | Control server | CPU server | Delta | Dropped jobs | Useful-ready / prediction |
|---|---:|---:|---:|---:|---:|
| K1 | 3.1650 | 3.0583 | -3.37% | 0.13% | 20.40% |
| K2 | 3.1200 | 3.0900 | -0.96% | 18.06% | 0.18% |

Verdict: unconditional speculative IQ2XXS CPU execution is exact but rejected as
a throughput lever. K1 adds CPU/page-fault pressure and loses throughput; K2
overloads the single-worker queue. Keep the opt-in probe, but use SPEX next for
high-confidence RAM staging/pinning rather than unconditional CPU execution.
Full protocol and artifacts are in `G34_SPEX_CPU_OBSERVE_RESULTS.md` at native
commit `63ba10d`.

## Execution sequence

1. **P0 measurement commit (done, G29):** standalone warm-tier benchmark and ledger schema.
2. **P1 code commit (done, G32):** direct resident slots, old miss fallback unchanged.
3. **P1 A/B commit (done, G32):** `n>=3` primed state, exact hashes and route counters.
4. **P2 code/A-B commit (done, G33; exact, no speed win):** mixed hit/miss scheduler.
5. **P2-SPEX measurement commit (done, G34; exact, negative):** speculative CPU
   miss `off/k=1/k=2`; keep opt-in, do not compose into the runtime policy.
6. **P3 code/A-B commit (done, G35; exact, positive):** physical cold-to-RAM
   admission plus VRAM promotion/demotion and mass/LFRU telemetry.
7. **P3 policy commit:** slow-clock mass/LFRU admission with hysteresis, measured
   against G35 second-touch promotion.
8. **P4 code commit:** prefill union and then waves as separate toggles.
9. Only then return to physical REAP rotation and SPEX transfer composition.

Stop conditions:

- any exactness/hash failure blocks performance promotion;
- a mechanism probe may remain `n=1`, but no throughput verdict may;
- cache state must be declared cold, primed or uncontrolled;
- external project numbers never enter the local result matrix as measurements.

## Deferred or rejected for now

- K8 direct/bake: paused by user decision.
- Static domain masks: rejected.
- Learned prompt/mask predictors: rejected.
- AirLLM whole-layer streaming: wrong exchange granularity.
- Global expert LRU: G9 measured zero hits and a regression at 32/96 slots.
- Q8-F16 cache: exact tested `256 MiB / 1280 MiB reserve` configuration failed
  closed with 2.05 -> 0.38 t/s and a changed output hash; other budgets remain
  unmeasured, not globally disproved.
- Further cold-expert compression and mixed quantization: remains a later TODO,
  after direct residency and exact tier transitions work.

## Bottom line

The highest-value path is not another prefetcher and not a larger pinned arena.
It is:

`direct VRAM hit + exact separate miss path + cold-to-RAM admission + one join`.

PocketMoE provides the closest same-format implementation reference for direct
route slots and raw IQ2XXS/Q2_K execution. Colibri provides the best three-tier
scheduler. llama.cpp PR #24524 provides the best exact hit/miss split. REAP mass
and SPEX remain valuable, but they should control this data path only after the
resident hit is genuinely movement-free.
