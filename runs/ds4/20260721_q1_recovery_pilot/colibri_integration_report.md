# Colibri integration analysis for DS4 REAP-revisited

Date: 2026-07-21  
Method: read-only static inspection of both source trees. No build, executable, benchmark, or GPU command was run.

## Executive verdict

The best thing to steal is **Colibri's dispatch ordering, not one of its kernels**: resolve a layer's routes, start the resident GPU subset first, run the missing subset asynchronously, and join partial outputs exactly once at the layer boundary. Applied to DS4, the missing subset is an exact IQ2/Q2_K CPU job backed by the K=5,600 RAM arena, not an SSD load. This is the shortest path to proving the F1 layer-local overlap that determines whether the design exceeds 6 t/s.

Do **not** transplant Colibri's CPU int4 kernel. It operates on a different weight format, has no performance result in the inspected tree, and its exact grouped-int4 path is a simple output-row-parallel AVX2 kernel rather than a blocked expert engine. The useful CPU idea is narrower: adapt Colibri's x86 integer-dot/sign-folding and independent-accumulator techniques to DS4's existing IQ2_XXS x Q8_K microkernel, which is scalar on x86 in the inspected code.

Do **not** replace DS4's mass-LFRU, SSD-WRAP worker, or routed CUDA kernels wholesale. DS4 already has more appropriate policy, state safety, and multi-route CUDA execution in those areas.

### License correction

The supplied statement says Colibri is MIT-licensed, but this checkout is not: `D:\ds4_work\colibri_src\LICENSE:1-3` says **Apache License 2.0**, and `D:\ds4_work\colibri_src\flake.nix:85-89` declares `licenses.asl20`. None of the reviewed C/CUDA files carries a contrary per-file SPDX header. Treat copied code from this checkout as Apache-2.0 unless the owner supplies authoritative evidence for a different license.

For copied code, preserve applicable notices, include the Apache-2.0 license, and mark modifications. The safer recommendation below is usually to reimplement the mechanism against DS4's formats and interfaces while recording Colibri as design provenance. This is an engineering observation, not legal advice.

## Ranking by value toward >6 t/s times low effort/risk

| Rank | Requested item | Verdict | Value | Effort / risk |
|---:|---|---|---|---|
| **1** | **5. Hybrid dispatch** | **Adopt the phase ordering in a one-layer CPU/GPU join spike now.** | Very high: proves or kills F1 at the real 43 layer barriers. | **1-2 days for the spike**; medium risk, bounded and reversible. |
| **2** | **1. CPU expert kernel** | Adapt x86 microkernel tactics; do not port the int4 format/kernel. | Medium-high if CPU tail is on the layer critical path. | **1-2 days** for SIMD + paired-kernel A/B; numeric/per-CPU risk. |
| **3** | **3. CUDA expert group** | No port. DS4 already launches route groups; steal only the compact descriptor/telemetry idea if useful. | Low incremental value; it cannot remove cross-layer dependencies. | **0.5 day** for telemetry/API cleanup; a port would be high-risk negative value. |
| **4** | **4. Async SSD** | Keep the overlap pattern already present in DS4; no io_uring port on Windows. | Low for REAP-revisited because K-resident decode removes SSD from the current-token path. | No change; **2-4 days** only if later evidence justifies an IOCP backend. |
| **5** | **2. Tier swap/LFRU** | Do not replace mass-LFRU. Colibri's selector is simpler but loses gate mass and DS4 safety constraints. | Low or negative for quality-adjusted hit rate. | **0.5-1 day** as an A/B policy only; otherwise zero work. |

## Rank 1 — Hybrid dispatch: steal the structure

### What Colibri actually does

Colibri separates MoE execution into visible phases:

- It routes the whole input batch and updates frequency/recency at `c/glm.c:2895-3044`.
- It forms the union of selected experts at `c/glm.c:3064-3070`.
- It resolves each unique expert through pinned RAM, LRU RAM, or a miss workspace at `c/glm.c:3137-3160`.
- On Metal, it submits resident experts to the GPU **before** loading misses, expressly to overlap GPU work and `pread`, at `c/glm.c:3163-3217`.
- It dispatches missing loads asynchronously at `c/glm.c:3219-3234`, then waits only before a missed expert's data is consumed at `c/glm.c:3280-3287`.
- Compute dispatch is independent of storage resolution: eligible resident weights use CUDA, otherwise the same resolved `ESlot` runs on CPU at `c/glm.c:3292-3324`; CUDA-eligible experts are collected and issued as device groups at `c/glm.c:3326-3380`.
- Workspace slots are promoted only after all relevant per-expert waits have made reuse safe at `c/glm.c:3381-3391`.

So Colibri does **not** have a magical three-way `CPU/GPU/SSD` choice. It makes two orthogonal decisions:

1. **Where are the bytes?** pin/cache/miss load.
2. **Where can this resolved expert execute?** resident CUDA tensor or CPU fallback.

That separation is cleaner than coupling representation, storage, promotion, and execution in a single branch.

### Map to the DS4 seams

DS4 already has almost the exact splice points:

- Synchronous route readback is `ds4_cuda.cu:34008-34048`.
- Representation classification and hot/cold list construction are `ds4_cuda.cu:34056-34179`.
- The grouped exact-IQ2 hot launch is `ds4_cuda.cu:34376-34419`.
- The current cold branch is a Q1 GPU launch at `ds4_cuda.cu:34472-34515`.
- The partial-output add/join is `ds4_cuda.cu:34517-34545`.
- Existing exact CPU subset building blocks are `ds4.c:4699-4745` for all selected IQ2 gate/up mids and `ds4.c:4816-4857` for accumulated Q2_K down output. The persistent six-expert decode path is `ds4.c:6558-6605`.

The REAP-revisited change should therefore be a replacement of the cold execution adapter, not a new MoE stack:

```text
route metadata ready
        |
        v
classify exact routes: GPU-hot/resident vs CPU-tail
        |
        +---- enqueue grouped hot IQ2 work on compute stream --------+
        |                                                            |
        +---- x-ready event -> 16 KiB activation D2H -> CPU subset ---+
                                                     |               |
                                             16 KiB partial H2D       |
                                                     +-- event -------+
                                                                     v
                                                    wait once + add partial output
```

The CPU branch should accept `{layer, activation, expert_ids[], weights[], count, arena generation}` and produce one already-weighted 4,096-float partial. It should not expose per-expert completion to CUDA. This preserves one join per layer and lets the existing CPU functions batch rows across all tail experts.

### Directly actionable recommendation

Implement the one-layer spike described in the design document, using this exact ordering:

1. Leave the current route D2H/classifier in place for the spike; force 1-3 routes into `CPU_EXACT`.
2. Record an event after the layer activation is ready. Make a dedicated bridge stream wait on it and enqueue the 16 KiB activation D2H into a pinned mailbox.
3. Immediately enqueue the existing grouped exact-IQ2 hot subset on the compute stream.
4. A bounded CPU worker waits for mailbox readiness, invokes one arbitrary-subset exact kernel over pointers in the K arena, and publishes one partial output plus generation/status.
5. Enqueue the 16 KiB partial H2D on the bridge stream, record a completion event, make the compute stream wait once, and launch the existing add kernel.
6. Attribute, per layer: activation D2H, CPU queue delay, CPU compute, partial H2D, hot GPU span, and final wait. Report `max(hot_span, bridge_span)` and exposed join wait, not only sums.

Hard invariants should mirror the useful parts of Colibri's pipeline comments at `c/glm.c:2144-2158`: generation-tag every mailbox job, never reuse a destination before its consumer has joined, publish job fields before the ready flag, and drain/cancel safely on shutdown or error.

**Adopt? Yes.** Adopt the phases and GPU-first ordering; keep DS4's storage, kernels, streams, and state types.

**Effort:** 1-2 developer-days for the forced one-layer proof; 3-5 days to generalize the mailbox, arbitrary subset API, cancellation, and all-layer instrumentation, consistent with the existing design estimate.

**License note:** the cited implementation is Apache-2.0, not MIT. Reimplement the state flow in DS4 terminology; if code/comments are copied, retain Apache provenance and obligations.

## Rank 2 — CPU expert kernel: useful tactics, no drop-in

### Colibri's exact grouped-int4 path

The format is defined at `c/glm.c:445-448`: two offset-encoded 4-bit values per byte, with one f32 scale for each `gs` elements along the input dimension. Its layout is output-row-major packed nibbles plus output-row-major group scales (`c/glm.c:449-456`). Parallelism is a static OpenMP split over output rows (`c/glm.c:452-453`). Within each group, AVX2 unpacks 16 nibbles, converts them to f32, and performs two FMAs (`c/glm.c:458-472`):

```c
#ifdef __AVX2__
const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
__m256 acc=_mm256_setzero_ps();
for(; i+16<=base+glen; i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
    __m128i nib=_mm_unpacklo_epi8(lo,hi);
    __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
    __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
a+=hsum256(acc)*sc;
#endif
```

This is not cache-blocked over output rows, experts, or tokens. It has no persistent thread pool; each matmul uses an OpenMP parallel region. Its only “blocking” is the quantization group boundary needed to apply the correct scale. It does not use a special AVX2-transposed weight layout.

The exactness test checks scale indexing, partial groups, odd nibble tails, and offset encoding at `c/tests/test_i4_grouped.c:1-19` and `:166-187`. It is a correctness test, not a performance test. The file contains optional tests for a `matmul_i4_grouped_pair` at `:96-159`, but those tests are behind `COLI_HAVE_GROUPED_PAIR`; the inspected tree contains neither that macro definition nor the function implementation. Do not infer that production grouped-int4 gate/up fusion exists.

Colibri does have a separate, approximate Q8-activation integer-dot path. It quantizes activations once at `c/glm.c:630-634`, uses VNNI or AVX2 `maddubs/madd` sign-folding at `c/glm.c:649-687` and `:738-791`, keeps thread-local quantization scratch at `c/glm.c:1002-1005`, and documents independent accumulators to hide dot-product latency at `c/glm.c:692-704`. Crucially, grouped format 4 bypasses that path and always uses the exact f32 grouped kernel at `c/glm.c:1052-1054`.

### Why it cannot replace DS4's CPU tail

DS4's weights are IQ2_XXS gate/up and Q2_K down. IQ2_XXS uses codebook/grid indices, sign tables, block scales, and a Q8_K activation; Q2_K has its own scale/min encoding. Colibri's `(nibble-8) * group_scale` math is unrelated. Transcoding DS4's exact IQ2 weights to Colibri int4 would be a new approximation, increase resident memory substantially, invalidate the exact-IQ2 premise, and still leave the Q2_K down path unsolved.

There is also no apples-to-apples timing evidence in Colibri. The inspected Colibri test only checks accuracy. The inspected `cpugemv_spike_results.json:12-15` reports 1.267 ms/expert at eight threads for the current spike; if a newer 0.7-0.9 ms result exists, Colibri still provides no comparable shape/format/host benchmark. A static review cannot claim it is faster.

### What is worth adapting

The target's clearest x86 seam is `ds4.c:368-383`: `dot_iq2_pair_16` has NEON implementations but falls back to scalar C on x86. The full x86 IQ2 path consequently remains scalar at `ds4.c:2202-2288`, and even the “paired” gate/up function falls back to two complete calls at `ds4.c:2367-2368`. The standalone spike has the same scalar core at `cpugemv_spike.c:308-312` and `:360-392`.

Adapt, in this order:

1. Add an SSSE3/AVX2 `dot_iq2_pair_16` using Colibri's sign-folding pattern: `abs(grid)` as unsigned bytes, apply `sign(grid)` to Q8 bytes, `maddubs` to s16, then `madd` to s32. Verify the maximum two-product lane cannot saturate s16 for the IQ2 grid range.
2. Add a true x86 paired IQ2 kernel that walks the shared Q8_K activation once while maintaining independent gate and up accumulators. This copies the *ILP/fusion idea*, not Colibri's int4 decoder.
3. Keep DS4's persistent pool and current cross-expert row batching (`ds4.c:4699-4745`); they are already better suited to a 1-6 expert layer tail than Colibri's repeated OpenMP regions.
4. Keep the accumulated Q2_K down projection (`ds4.c:4801-4857`) so the pool is entered once and the 4,096-wide result is produced directly.
5. A/B 1, 2, 3, and 6 tail experts with the real K-arena pointers. Require numerical parity/tolerance before using throughput results in the >6 t/s model.

**Drop-in? No. Adopt? Only the x86 microkernel techniques.**

**Effort:** 0.5-1 day for the 16-byte dot and paired loop, another 0.5-1 day for saturation proof, tests, CPUID/compile guards, and real layer-tail benchmarks. Risk is medium because signed-byte SIMD and accumulation order can silently perturb output.

**License note:** direct reuse of Colibri intrinsics or comments is Apache-2.0. A fresh IQ2 implementation based on DS4's format is preferable; record Colibri as design inspiration.

## Rank 3 — CUDA expert group: real grouping, but DS4 already has it

### What the API promises

`c/backend_cuda.h:54-58` defines a one-expert fused MLP with one activation upload and one output download. `c/backend_cuda.h:68-74` defines a group of same-shaped experts whose input/output rows are packed consecutively. The Windows loader merely resolves and forwards that ABI (`c/backend_loader.c:188-199` and `:279-289`); it adds no scheduling behavior.

`coli_cuda_expert_group` builds up to 64 device-pointer descriptors and row offsets at `c/backend_cuda.cu:634-653`, copies one descriptor array and one packed input at `:654-674`, executes the expert pipelines at `:675-739`, and copies one packed output back at `:740-757`.

### Is it a grouped-GEMM in one launch?

The answer is **sometimes one multi-expert launch per projection, but not a conventional grouped-GEMM**:

- The exact low-row path uses `blockIdx.z` as expert index (`c/backend_cuda.cu:235-255` and `:263-297`). Gate/up and down each cover all experts in one grid; the dual gate/up kernel shares launch overhead and activation addressing (`:275-285`). This is genuine multi-expert launch grouping.
- The optional W4A4 Tensor Core path also covers `count` experts in its grid at `c/backend_cuda.cu:680-690`, but it quantizes activations to signed int4 and is not DS4's exact IQ2 path.
- The W4A16 branch explicitly loops over experts and launches kernels per expert at `c/backend_cuda.cu:691-722`; it only shares the stream and outer API.
- The generic kernels are one CUDA block per `(output,row,expert)` with hand-written reductions, not cuBLASLt/CUTLASS grouped GEMM (`c/backend_cuda.cu:235-255`). There is no inter-expert weight reuse.
- The call synchronizes before returning and performs host round-trips (`c/backend_cuda.cu:740-746`), so it is not an asynchronous graph node from the host caller's perspective.

### Map to DS4

DS4 already classifies all hot routes, then passes `hot_count` and compact device metadata to one `routed_moe_launch` at `ds4_cuda.cu:34376-34401`. Its routed kernels already span route/expert dimensions and include pointer-table and sorted/tiled variants; therefore porting Colibri's int4 kernels would regress format support and likely performance.

Most importantly, multi-expert launch grouping removes **intra-layer per-expert launch overhead**. It cannot remove DS4's 43 sequential layer dependencies, because each layer's joined routed output feeds HC post and the next layer. It is not a solution to F1 serialization.

The only low-risk steal is the compact `{gate,up,down,rows,offset}` descriptor boundary at `c/backend_cuda.cu:43-46` plus explicit group counters/timing at `:747-756`. DS4 could use an equivalent backend-neutral descriptor for GPU-hot and CPU-tail subsets and report group size/rows consistently.

**Adopt? No kernel port.** Optionally normalize DS4 telemetry around a small route-group descriptor after the F1 spike.

**Effort:** about 0.5 day for descriptor/telemetry cleanup. A kernel port is at least 3-5 days and has no demonstrated incremental value.

**License note:** backend header, loader, and CUDA implementation in this checkout are Apache-2.0. Copying the descriptor or kernels requires Apache handling; a DS4-native descriptor is trivial to reimplement.

## Rank 4 — Async SSD: steal the completion discipline, not io_uring

### Colibri's pattern

`c/uring.h:4-5` describes a single-thread-owned ring that queues positioned reads and reaps completions without a userspace spin loop. It:

- maps submission/completion rings at `c/uring.h:47-81`;
- bounds io-wq workers at `c/uring.h:84-87`;
- prepares reads with caller-owned `user_data` and forces cold regular-file reads to io-wq via `IOSQE_ASYNC` at `c/uring.h:89-112`;
- submits or waits for at least one completion at `c/uring.h:115-124`;
- reaps CQEs with acquire/release ordering at `c/uring.h:127-133`.

The integration assigns each read to a logical expert load (`c/glm.c:1966-1994`), coalesces contiguous gate/up/down weights into one read when possible (`:2049-2075`), queues scales separately (`:2077-2083`), and completes/finalizes each expert independently (`:2085-2131`). The caller queues a whole miss batch at `c/glm.c:2222-2231` and waits for only the expert it is about to consume at `:2243-2250`.

Windows does not need a literal port. Colibri itself uses a persistent bounded worker pool on Windows: `c/glm.c:2162-2167` makes PIPE the default, `:2180-2203` performs positioned loads on workers, and `:2219-2241` publishes a generation-tagged batch.

### Map to DS4 SSD-WRAP

DS4 already embodies the same conceptual pattern more fully:

- explicit request/inflight/ready/committing/failure states and generation/victim metadata at `ds4_cuda.cu:24120-24156`;
- a bounded condition-variable worker and job table at `:24158-24217`;
- wave selection, sorting/coalescing, read service, and per-job ready publication at `:24400-24547`;
- deduplication, backpressure, slot/ring reservation, generation capture, and enqueue at `:24894-25013`.

Thus Colibri validates DS4's architectural choice; it does not expose a missing Windows feature. If F5 remains as a legacy path, the reusable rules are: queue all independent positioned reads early, keep destinations disjoint, tag completions with job generation, coalesce physically adjacent ranges, publish each expert separately, and wait only at first use.

For REAP-revisited, K=5,600 exact experts are resident in RAM and the CPU tail executes those resident bytes. SSD must not be on the current-token route path, so an IO backend change has essentially zero direct value toward >6 t/s. The existing DS4 worker should remain for bootstrap/promotion/legacy SSD-WRAP work.

**Adopt? Concept already adopted. No new work now.** Consider Windows overlapped I/O/IOCP only if profiles later show the worker's sequential `pread` service—not NAND or admission—is the measured F5 bottleneck.

**Effort:** zero for REAP-revisited. A production IOCP backend with cancellation, short-read handling, direct-I/O alignment, and shutdown is roughly 2-4 days and should be evidence-gated.

**License note:** `uring.h` and its `glm.c` integration are Apache-2.0. Do not copy the Linux syscall wrapper into DS4; it is unusable on the target and unnecessary.

## Rank 5 — Tier swap/LFRU: simpler, not better for DS4

### Colibri policy

The frequency-only selector finds the coldest resident and hottest nonresident, then requires a 25% plus four-hit margin (`c/tier.h:6-24`). The LFRU score makes frequency lexicographically dominant over a bounded 8-bit recency tie-break:

```text
score = (heat << 8) | max(255 - age, 0)
```

That is `c/tier.h:27-33`. `tier_pick_lfru` uses the same 25% plus four-frequency hysteresis at `c/tier.h:35-54`, and `tier_decay` halves all heat at `:56-58`. Colibri invokes the selector between turns at a safe point (`c/glm.c:4881-4891` and `:4916-4923`).

This is easy to reason about, but its candidate search tests residency with a nested scan (`c/tier.h:44-47`), it treats all routed hits equally, and recency can never outweigh one frequency count. It is suitable for Colibri's modest per-layer expert set and between-turn re-pin cadence.

### Map to DS4 mass-LFRU

DS4's score is gate mass times log-frequency times a decaying recency factor at `ds4_cuda.cu:27177-27186`. Its victim selection additionally respects:

- minimum touches and representation eligibility (`ds4_cuda.cu:27188-27204`);
- empty slots and claimed/in-flight slots (`:27205-27216`, `:27295-27313`);
- per-epoch replacement budgets and adaptive pressure control (`:27218-27293`);
- configurable score hysteresis before replacement (`:27318-27327`).

Those are not accidental complications. DS4 ranks 11,008 `(layer,expert)` entries, optimizes gate-mass coverage rather than raw hit count, and must not evict a slot claimed by an in-flight request. Replacing that with Colibri's integer LFRU could improve raw simplicity while worsening quality-adjusted hit rate and safety.

For REAP-revisited, the K=5,600 RAM set should be deterministic and frozen from the offline held-out mass manifest for the run. Only the much smaller GPU-hot cache needs dynamic replacement, and the existing mass-LFRU is the better starting point there.

One small structural idea is worth keeping: a pure selector with explicit hysteresis is easy to unit-test. If DS4's policy is refactored later, separate `score`, `eligible`, `pick_victim`, and `admit` from I/O/state mutation. That is a code-organization improvement, not adoption of Colibri's score.

**Adopt? No.** Retain mass-LFRU for GPU hot storage and freeze the K-resident RAM manifest.

**Effort:** no production change. A Colibri-style frequency-LFRU A/B mode would take 0.5-1 day but is not recommended without held-out mass and churn comparisons.

**License note:** `tier.h` is Apache-2.0 in this checkout. Its formula is simple enough to rederive, but any copied function/comment still needs proper provenance.

## Final implementation order

1. **Run the one-layer hybrid spike** using the existing hot grouped CUDA launch and existing exact CPU kernels, with a single layer join and full critical-path telemetry.
2. If CPU compute is exposed, **add the x86 IQ2 signed-byte SIMD dot and true paired gate/up loop**, then rerun the same spike. Do not optimize a standalone expert benchmark without measuring join exposure.
3. If the spike passes, generalize the CPU subset mailbox across all layers, then retire the Q1 cold branch for this mode.
4. Leave CUDA grouping, SSD-WRAP, and mass-LFRU structurally intact until end-to-end profiles identify a new bottleneck.

The go/no-go metric is not “Colibri kernel faster” or aggregate CPU ms/token. It is the distribution of **exposed per-layer join wait** with 1-6 CPU-tail routes while the real hot GPU subset is active. That directly answers whether REAP-revisited can sustain more than 6 t/s on this machine.
