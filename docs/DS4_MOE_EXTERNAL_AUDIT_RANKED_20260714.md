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
- HEAD: `97fae74`
- G39 was measured from base `78f50cb` plus the reviewed hardening committed as
  `5633856`; source, executable, manifest and matrix hashes pin that state.
- G40 is the exact cyberpunk composition checkpoint at native commit `6298b66`.
- G41 is the exact 30 GiB prefill bulk-seed checkpoint at native commit `97fae74`.

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

### Rank 3 policy checkpoint: G36

Native-Windows commit `b1ef49c` adds opt-in
`DS4_EXPERT_TIER_POLICY=mass-lfru` without changing the G35 cold-to-RAM data
path. It fills free VRAM slots after three observed uses, then evaluates
replacement candidates over a 430-call clock with a budget of 16 physical
changes. Candidate and victim are compared with a mass/frequency/recency score
and a 1.25 hysteresis factor. Rejected experts remain in pinned RAM and use the
exact transient route.

The final counter-ordered matrix used
`second-touch/mass-lfru/mass-lfru/second-touch`, one discarded warmup and `n=3`
measured requests per process. The runner fails closed if any process differs in
HEAD, executable, CUDA source, build manifest, compile-input fingerprint,
harness or model provenance. All 12 measured outputs and four warmups matched
the expected exact hash.

| Metric | G35 second-touch | G36 mass/LFRU | Delta |
|---|---:|---:|---:|
| Server decode | 4.9483 t/s | 5.5567 t/s | +12.29% |
| Client throughput | 2.8195 t/s | 3.0439 t/s | +7.96% |
| VRAM hits | 3,984 | 5,089 | +27.74% |
| VRAM replacements | 3,963 | 48 | -98.79% |
| RAM H2D | 34.963 GiB | 27.679 GiB | -20.83% |
| Process reads | 31.001 GiB | 31.001 GiB | unchanged |

Both G36 arms retained 336 VRAM residents, admitted 1,005 cold experts to RAM,
recorded zero cold-to-VRAM transitions and zero failures. The unchanged process
reads isolate the gain to hotset stability and lower RAM-to-GPU churn, not fewer
initial model reads.

Verdict: exact positive short-workload checkpoint. Keep the policy opt-in because
only one deterministic prompt and one parameter set are measured. The next
policy gate is a longer exact workload plus a domain switch; the next independent
transport lever remains batch-union/waved prefill. Full report:
`G36_MASS_LFRU_TIERING_RESULTS.md` at native commit `b1ef49c`.

### Rank 4 measured checkpoint: G37 batch-union and chunk amplification

Native CUDA already performs the first half of Rank 4: for every MoE
layer/chunk, `cuda_moe_selected_load()` deduplicates all `n_tokens * top_k`
routes into one compact expert union and remaps the route slots. G37 added an
opt-in, observation-only summary and an explicit harness chunk parameter; it did
not change routing, weights or execution order.

The counter-ordered matrix used the 43-token cyberpunk prompt, cache336 LRU,
GPU-resident decode routes, one discarded warmup and `n=3` measured requests per
process. All 27 measured outputs and all nine warmups matched exact hash
`aa3b17600d88d3161605db8389b5bf03d4e94debcc8eeb74dca27aed95a154ab`.

| Chunk | TTFT | Unique unions | Logical source spans | Upload syncs |
|---:|---:|---:|---:|---:|
| full (43) | 7.547 s | 11,764 | 77.506 GiB | 168 |
| 16 | 8.609 s | 17,956 | 118.263 GiB | 504 |
| 8* | 10.393 s | 22,632 | 149.087 GiB | 1,008 |

The telemetry A/B measured -0.21% TTFT, which is effectively zero within run
noise. Relative to full chunk, chunk 16 added 52.59% logical source traffic and
14.07% TTFT; chunk 8 added 92.35% logical source traffic and 37.70% TTFT.
Chunk 8 is a single-process `n=3` directional observation without a second
counter-order replication. `source_span_bytes` is a selected-loader logical
counter, not physical SSD bytes; the separate Win32 counter excludes mmap
page-ins.

Verdict: P4-A is already implemented and now measured. P4-B should not duplicate
that code. Its purpose is to retain a wide prompt batch while partitioning a
too-large compact union into exact output-summing expert waves. Full report:
`G37_PREFILL_UNION_RESULTS.md` in the native Windows repo.

### Rank 4 measured checkpoint: G38 capacity-bounded waves

G38 adds opt-in `DS4_CUDA_PREFILL_WAVES=1`. It preserves G37's complete
per-layer/chunk expert union, partitions only the compact staging work, writes
each partial down result to its original token/route slot and invokes the normal
ordered six-route sum once. The router selection and weights are not narrowed.

The final counter-ordered matrix used the 43-token cyberpunk prompt, cache336
LRU, one discarded warmup and `n=3` measured requests per process. Two processes
per arm compared production kernels, generic sorted kernels, and generic sorted
kernels with forced 31-expert waves. All 18 measured outputs and six warmups
matched the expected exact hash.

| Arm | TTFT | Client t/s | Decode t/s | Process reads | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| production | 7.319 s | 0.945 | 2.230 | 164.49 GiB | 10.900 GiB |
| generic | 11.007 s | 0.732 | 2.230 | 164.55 GiB | 10.900 GiB |
| wave31 | 10.141 s | 0.836 | 2.945 | 131.14 GiB | 10.437 GiB |

Wave31 was 7.86% faster than its same-kernel generic control and both observed
the same cumulative 11,716 union experts. Against production it regressed TTFT
38.56% and client throughput 11.47%, while reducing process reads 20.27% and
peak dedicated VRAM 4.25%. Forced-width safety probes also passed at 7 experts
(435 waves) and 1 expert (2,929 waves), with exact output and zero failures;
these `n=1` probes are mechanism evidence only.

Verdict: exact capacity mechanism, negative production performance in serial
v1. Keep opt-in. G39 should double-buffer weights and pair metadata, overlap
upload N+1 with compute N, and restore tile-capable wave kernels without changing
the full-union or final ordered-sum contracts. Full report:
`G38_PREFILL_WAVES_RESULTS.md` in the native Windows repo.

### Rank 4 measured checkpoint: G39 exact double-buffered waves

G39 adds opt-in `DS4_CUDA_PREFILL_WAVE_DOUBLE_BUFFER=1` on top of G38. Two
parity-owned weight and metadata slabs allow upload of wave N+1 after compute N
is launched. Persistent compute events fence reuse across layer boundaries. A
post-matrix review found and fixed two latent failure-path races: slab resize now
fences any prior upload, and failed mid-wave launches seal or drain already
enqueued parity work.

The accepted counter-ordered rerun used the same 43-token cyberpunk prompt,
cache336 LRU, one discarded warmup and `n=3` measured requests per process. All
18 measured outputs and six warmups matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.

| Arm | TTFT | Client t/s | Decode t/s | Process reads | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| production | 7.862 s | 0.8759 | 2.0533 | 164.45 GiB | 10.900 GiB |
| serial wave31 | 10.388 s | 0.8192 | 2.8200 | 131.32 GiB | 10.437 GiB |
| overlap wave31 | 8.583 s | 0.9308 | 2.7883 | 131.31 GiB | 10.642 GiB |

Overlap recovered 17.38% TTFT and 13.62% client throughput versus the same
serial wave path, with effectively unchanged reads. It remained 9.18% slower
than production TTFT while reading 20.15% fewer bytes. Both overlap replications
recorded 456 waves, 454 parity reuse fences, 456 compute records and zero
failures. A post-review `IoQD=2` probe was also exact across 42 layers and 114
waves; it is mechanism evidence only.

The isolated G39 decode path deliberately did not admit prefill experts into
the cache. Each four-request wave process measured only 24 all-hit calls out of
2,016 route calls, 1,992 miss-worker jobs and 7,141 missing experts. This is a
measured reason to compose G36 mass/LFRU and later test a request-scoped closed
arena; it is not evidence against the overlap mechanism. Full report:
`G39_PREFILL_WAVE_OVERLAP_RESULTS.md`, native commits `78f50cb` and `5633856`.

### Rank 6 composed checkpoint: G40 mass/LFRU reduces misses but loses transport

G40 composed production full-chunk prefill with an 8 GiB dynamic arena and the
G36 mass/LFRU actuator on the 43-token cyberpunk prompt. The matched counter-
ordered matrix used one discarded warmup plus `n=3` requests in each process;
all 18 measured outputs and six warmups matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.

Versus the matched arena control, mass/LFRU reduced missing experts from 7,444
to 6,155 (-17.32%) and raised all-hit calls from 28 to 137. It nevertheless
raised process reads from 174.99 to 433.99 GiB, route wait from 4.940 to 8.548
ms/call and reduced server decode from 1.99 to 0.463 t/s. TTFT was flat at
7.91 seconds. The useful policy signal therefore does not pay when the current
actuator performs incremental admissions during decode.

The 8 GiB arena was allocated but neither prefill-mass observation nor WRAP
publication was enabled, so it was not bulk-populated from prompt mass before
decode. The next isolated transport gate is to seed the arena once from the
ordinary unbiased prefill, with all seed cost included in TTFT, before tuning
mass/LFRU thresholds or adding semantic shard probes. Full report:
`G40_MASS_LFRU_CYBERPUNK_RESULTS.md`, native commit `6298b66`.

### Rank 4/6 transport checkpoint: G41 bulk seed passes, amortization pending

G41 isolated one ordinary-prompt prefill-mass publication from cache and
mass/LFRU. It compared observe-only with bulk WRAP using the same 30 GiB pinned
arena, three independent one-request processes per arm and balanced order. All
six outputs matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.

The cyberpunk prefill identified 2,657 unique entries; all fit the 4,551-slot
arena and covered 100% of observed prefill mass. Candidate membership covered
84.83% of selected decode IDs; runtime measured 2,443 arena hits and 653 misses
(78.91%). WRAP raised decode from 1.48 to 2.31 t/s (+56.08%) and reduced process
reads from 51.43 to 36.94 GiB (-28.17%). It cost 24.66 seconds to publish,
raising TTFT from 20.49 to 44.71 seconds, so the 12-token end-to-end result is
negative even though the isolated transport mechanism is positive.

At the measured mean rates, arithmetic projects a break-even near 100 generated
tokens; this is not a measured break-even. Next measure a long decode and patch
explicit co-ownership: immutable pinned-RAM snapshot plus a mass/LFRU-protected
VRAM subset, never cold SSD directly to VRAM. Full report:
`G41_PREFILL_BULK_SEED_CYBERPUNK_RESULTS.md`, native commit `97fae74`.

### Rank 1/6 composed checkpoint: G42 closes SSD and reaches 4.08 t/s

G42 implements the co-ownership required by G41. One unbiased full-probability
prefill ranks a 4,551-entry request-scoped snapshot: 768 entries cover every
expert in the three hash-routed layers and 3,783 semantic entries are selected
by normalized accumulated mass. The snapshot remains immutable pinned-RAM
backing while mass/LFRU protects 256 experts in VRAM. Only after both data and
metadata publish, a per-request mask excludes semantic entries outside the
snapshot; it is removed at request end and is not a reusable domain mask.

The balanced matrix used three independent one-request processes per arm,
1,024 MiB load reserve and exact 12-token cyberpunk output. All outputs matched
hash `921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.

| Metric | G41-style control | Closed G42 |
|---|---:|---:|
| TTFT | 44.451 s | 82.078 s |
| Publication | 22.982 s | 62.407 s |
| Decode | 2.277 t/s | 4.083 t/s |
| Process reads | 36.772 GiB | 23.190 GiB |
| Snapshot misses | 653 arena misses/request | 0 |
| SSD bytes | fallback possible | 0 measured |

Closed G42 improved decode 79.36% and reduced process reads 36.93%. Every
replication measured 886 VRAM hits and 2,210 pinned-RAM hits, with zero
snapshot misses, zero cold admission, zero SSD bytes and zero failures. The
remaining miss-shaped GPU gaps are RAM-to-VRAM service events, not SSD access.

A protocol A/B also exposed a critical VRAM-partition constraint: reserving
4,096 MiB left only about 4.94 GiB for hot non-expert weights and reduced both
arms to 0.14-0.15 t/s. Reserving 1,024 MiB restored a 7.21 GiB hot cache and
the accepted 4.08 t/s mean. Cache capacity and hot-weight residency must always
be measured together.

G42 does not yet pass short end-to-end throughput because the closed snapshot
costs 62.4 seconds to publish. Arithmetic projects a 194-token break-even, not
a measured one. Next priority is publication reuse/batching plus a long `n>=3`
L0-L3 gate. Minimum available RAM fell to 0.065-0.741 GiB in individual closed
runs, so production capacity also needs explicit headroom. Full report:
`G42_CLOSED_SNAPSHOT_TIERING_RESULTS.md`, native commit `4640c33`.

### Rank 1/6 publication checkpoint: G43 removes the duplicate checksum

Profiling showed that every WRAP worker already computed FNV-1a after copying a
complete expert slot, but `finish_load` then reread the entire 30 GiB arena
serially and computed the same checksum again. G43 keeps the original public
finish path unchanged and adds an opt-in worker-checksum path used only after
all copy workers join.

The balanced A/B used the exact G42 closed-snapshot configuration and three
independent one-request processes per arm. All six outputs matched the expected
hash; each run retained 886 VRAM hits, 2,210 pinned-RAM hits, zero snapshot
misses, zero SSD bytes and zero tier failures.

| Metric | Finish verify | Worker checksum |
|---|---:|---:|
| TTFT | 86.028 s | 47.355 s |
| WRAP total | 65.214 s | 27.251 s |
| Copy plus worker FNV | 32.758 s | 27.236 s |
| Finish FNV | 32.433 s | 0.001 s |
| Decode | 4.093 t/s | 4.093 t/s |

G43 cuts WRAP 58.21% and TTFT 44.95% without changing steady decode or
residency. The next ranked bottleneck is the first copy from model `mmap`, which
varied from 23.003 to 40.565 seconds in the accepted matrix and reached 253.623
seconds in a separate cold mechanism run. Next compare expert-major with a
globally source-ordered or part-major copy schedule, one lever at a time. Full
report: `G43_WRAP_CHECKSUM_RESULTS.md`, native commit `4a3b792`.

### Rank 1/6 source-parts checkpoint: G44 orders first copy by source

G44 keeps the exact G42/G43 closed-snapshot configuration: native Windows
RTX 3060, the same 4,551-entry 30 GiB pinned snapshot, cache256 mass/LFRU,
zero allowed SSD fallback and expected hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
The new opt-in source-parts path copies model `mmap` in global source-offset
order, with barriered gate/up/down phases and incremental exact canonical
full-slot FNV. The default expert-major path is unchanged.

The balanced A/B used three independent one-request processes per arm, ordered
expert A, source A, source B, expert B, expert C, source C. All six outputs
matched the expected hash. Every run retained 516 route calls, 3,096 selections,
886 VRAM hits, 2,210 pinned-RAM hits, zero snapshot misses, zero SSD bytes and
zero tiering failures.

| Metric | Expert-major | Source-parts |
|---|---:|---:|
| WRAP median | 31.947 s | 25.829 s |
| TTFT median | 52.508 s | 46.184 s |
| Decode mean | 4.04 t/s | 4.10 t/s |

G44 cuts primary WRAP median 31.947 -> 25.829 s, a 6.118 s or 19.15%
improvement, and cuts TTFT median 52.508 -> 46.184 s, a 6.324 s or 12.04%
improvement. Individual WRAP times were expert 30.109, 31.947 and 201.671 s
versus source 25.040, 25.829 and 26.301 s. The expert outlier is retained
because Windows standby cannot be purged. Decode means were 4.04 versus
4.10 t/s, so make no decode claim.

Verdict: exact positive startup optimization for the measured closed snapshot.
The next ranked lever is direct resident slots and hit/miss separation, not more
first-copy scheduling. Full report: `G44_SOURCE_PARTS_RESULTS.md`, native
implementation commit `48234f31ec5828ae094496e42eb01f498e4b87c8`.

### Rank 1/6 capacity checkpoint: G45 raises protected residents to 320

G45 keeps the G44 source-parts closed snapshot and changes only
`DS4_CUDA_STREAMING_EXPERT_CACHE_N=256 -> 320`, with the same 0.125 GiB cache
reserve, 30 GiB pinned host snapshot, mass/LFRU policy, prompt, context, binary
and expected output hash
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`.

The requested capacity is protocol-critical under WDDM. Two consecutive
336-slot attempts were not reproducible: one process received effective 336 and
the next received 321, so 336 is rejected as a default-capacity candidate. Other
exploratory capacity probes are retained only as mechanism evidence, not mixed
into the throughput verdict.

The accepted A/B used three independent processes per arm, ordered 256 A,
320 A, 320 B, 256 B, 256 C, 320 C. All runs were exact, SSD-free and retained a
closed snapshot: snapshot misses, SSD bytes and tier/route failures were zero in
both arms.

| Metric | 256 protected | 320 protected | Delta |
|---|---:|---:|---:|
| Server decode mean | 4.4367 t/s | 4.4767 t/s | +0.90% |
| Server decode median | 4.44 t/s | 4.48 t/s | +0.04 t/s |
| Decode seconds mean | 14.4247 s | 14.2963 s | -0.89% |
| Pinned-RAM H2D mean | 75.0344 GiB | 71.5803 GiB | -3.4541 GiB (-4.60%) |
| VRAM route hits mean | 5,129 | 5,653 | +524 |
| Pinned-RAM route hits mean | 11,383 | 10,859 | -524 (-4.60%) |
| Worker time | 1.6853 ms/job | 1.6207 ms/job | -3.84% |
| Worker-ready wait | 1.6727 ms/call | 1.6047 ms/call | -4.07% |
| Peak VRAM mean | 11,076.0 MiB | 11,517.3 MiB | +441.3 MiB |
| TTFT median | 44.604 s | 44.754 s | +0.150 s |
| WRAP median | 23.084 s | 24.640 s | +1.556 s |

Per-run decode was tightly grouped: 256 measured 4.44, 4.44 and 4.43 t/s;
320 measured 4.48, 4.48 and 4.47 t/s. The 64 extra slots replaced exactly 524
pinned-RAM routes with VRAM hits over 64 generated tokens and removed 3.454 GiB
of H2D traffic. This is a small but replicated decode win and a clearer direct
mechanism win, at about 441 MiB additional peak VRAM.

One 256 run reported a 377.736 s TTFT while decode stayed 4.43 t/s and the
source-parts WRAP interval was only 23.084 s. The stall is an unlocalized
prefill/WDDM event before first token, so TTFT mean is not used for the cache
verdict. The 64-token output is a deterministic prefix transport test only; it
does not claim complete HTML or L0-L3 quality.

Provenance: measured parent
`a8e48d7c4872e406f5f5a3764d45660315a0f687`; executable SHA-256
`801ea8ff8531245ff3083d71cdc5b5b55b93f0b1dc4904bee30d24d0dd653026`;
`ds4_cuda.cu` SHA-256
`be4103d78f05d0f565cf2103b0d93b2c04f517e1ac7ebd057951c6db67d34063`; build
manifest SHA-256
`100abc59ee94c04a4399a91f567235c7f341f5fca004985290a74e53c99a5fd6`; harness
SHA-256 `235d4220e3903425ae55c32cec950a01a58bf601f6b55d80c2784995aa069533`;
runner SHA-256
`23699ea6251ad4bffb5e03d077de0fdfa9be9095c55ac15171e680463132a31d`;
corrected summary runner SHA-256
`c9ceca6fc95467bd76ec65ecf9c4644a7470fb6108cf93da25214b7980629ad7`.

Verdict: 320 protected expert slots are the best measured reproducible capacity
for the current RTX 3060 configuration. Keep 336 rejected for default use. The
next isolated gate should remove the redundant default-stream synchronization in
the GPU-resident route handoff, relying on the mapped request sequence and
worker-ready publication. It must remain opt-in, no-default-sync, exact-safe
first, and only then run an `n>=3` A/B. Full report:
`G45_DIRECT_RESIDENT_CACHE_RESULTS.md`.

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
7. **P3 policy commit (done, G36; exact, positive on short workload):**
   slow-clock mass/LFRU admission reduced replacements 3,963 -> 48 and raised
   server decode 4.948 -> 5.557 t/s. Longer/domain-switch validation remains.
8. **P4-A measurement commit (done, G37):** existing per-layer/chunk union is
   exact; smaller chunks measured 52.59-92.35% more logical source traffic.
9. **P4-B code/A-B commit (done, G38; exact mechanism, negative serial v1):**
   full-union waved prefill passed exactness but regressed production TTFT
   38.56%; keep opt-in and proceed only to isolated double-buffer overlap.
10. **P4-C code/A-B commit (done, G39; exact overlap, not production TTFT):**
    double buffering recovered 17.38% versus serial wave31 but remained 9.18%
    behind production. Post-review failure ownership was hardened in `5633856`.
11. **Composed SOTA gate (done, G40; exact, transport negative):** mass/LFRU
    reduced missing experts 17.32%, but incremental admissions raised reads
    148.00% and reduced decode 76.72% versus the matched arena control.
12. **Prefill bulk-seed gate (done, G41; exact, transport positive, short-run
    end-to-end negative):** decode +56.08%, reads -28.17%, publication 24.66 s.
13. **Arena/cache co-ownership done, G42:** request-scoped 4,551-entry closed
    snapshot plus a 256-slot mass/LFRU VRAM tier measured 4.083 t/s, zero
    snapshot misses and zero SSD bytes. The 336-slot candidate crossed the VRAM
    cliff; 256 is the current measured configuration, not a universal optimum.
14. **Duplicate publication checksum done, G43:** joined workers now provide
    the accepted checksum, cutting WRAP 65.214 -> 27.251 s and TTFT 86.028 ->
    47.355 s with exact output, unchanged 4.093 t/s decode and zero SSD.
15. **First-copy source-parts done, G44:** global source-offset ordering cut
    WRAP median 31.947 -> 25.829 s and TTFT median 52.508 -> 46.184 s with
    exact output, unchanged zero SSD and no decode claim.
16. **Protected resident cache capacity done, G45:** cache320 versus cache256
    measured decode 4.4367 -> 4.4767 t/s and pinned-RAM H2D 75.0344 ->
    71.5803 GiB, with exact output, zero snapshot misses and zero SSD. The
    336-slot candidate failed reproducible capacity and is not a default.
17. **No-default-sync route handoff next:** keep the G45 cache320 closed
    snapshot as baseline, remove only the redundant default-stream sync in the
    GPU-resident handoff behind an opt-in gate, then require exact safety before
    any `n>=3` throughput verdict.
18. **P4-D next:** restore tile-capable wave kernels without changing G39's
    full-union, ordered-sum or parity-ownership contracts.
19. **Prompt-intent closed-arena follow-on:** split one request into semantic router-only
    probes, aggregate unbiased per-layer mass into a RAM/VRAM preload prior, then
    build a request-scoped closed set whose complete payload fits pinned RAM plus
    VRAM. Outside experts become ineligible only for that request. Account probe,
    build and preload time inside TTFT; compare exact output, L0-L3 quality, cold
    misses, SSD-to-RAM and RAM-to-VRAM bytes. Do not generate separate shard
    continuations or attempt to merge their KV caches.
20. Only then return to physical REAP rotation and SPEX transfer composition.

Stop conditions:

- any exactness/hash failure blocks performance promotion;
- a mechanism probe may remain `n=1`, but no throughput verdict may;
- cache state must be declared cold, primed or uncontrolled;
- external project numbers never enter the local result matrix as measurements.

## Deferred or rejected for now

- K8 direct/bake: paused by user decision.
- Reusable/cross-session static domain masks: rejected. A mask built from
  unbiased probes for one request and discarded on intent change remains a
  planned closed-arena experiment, not a reusable domain mask.
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
