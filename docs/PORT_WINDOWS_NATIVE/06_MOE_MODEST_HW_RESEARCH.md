# MoE on modest hardware: revised architecture study for native Windows DS4

**Date:** 2026-07-14  
**Target:** native Windows, RTX 3060 12 GB, 64 GB system RAM, DS4 2-bit model
(about 80 GB).  
**Evidence rule:** performance conclusions require measured A/B runs. An `n=1`
probe can identify a mechanism, but it is not a throughput or quality verdict.

This revision supersedes the first version of this document. It incorporates
the native-Windows G26-G28 measurements and a source-level audit of Colibri at
commit `bad64d1b06cbda80dbdfb1f8f502370d0407bcbb`.

## Executive correction

The old headline, "efficient engines do not stream experts to the GPU", was too
absolute. The more useful rule is:

> Do not pay an unamortized storage, host-to-device copy, allocation, or
> synchronization cost for every routed expert at every token.

There are three viable ways to satisfy that rule:

1. compute warm experts where they already live in CPU RAM;
2. keep hot experts in persistent GPU slots and address those slots directly;
3. combine the two, loading only true cold misses while resident work runs.

The architecture can still be dynamic. Logical routing and masks may change at
token granularity, while expensive physical placement changes at a bounded
interval, with hysteresis and an emergency path for a high-weight miss. This is
the distinction that the earlier document was missing.

## Current measured state

The original `0.907 t/s` native-Windows number is obsolete.

| Experiment | Repetitions | Decode | What it establishes |
|---|---:|---:|---|
| G27 packed mass observer, WRAP off | 3 | mean **2.513 t/s**, median **2.52** | Current exact no-actuation baseline for that harness |
| G27 mass WRAP on | 3 | mean **2.073 t/s** | Correct transactional rotation, but 2.361 s mean WRAP cost is not amortized over 16 tokens |
| G28 full K8, embedding-row staging, warm | 1 | **2.01 t/s** | Mechanism probe only: routed universe fits and expert eviction is no longer the ceiling |

G27 used greedy `nothink` and all six final A/B runs matched the expected output
hash. G28 retained 336 routed experts (42 layers x 8), about 2.21 GiB, reached
328/336 resident entries, 4,602 hits and zero expert-cache evictions. Yet it
stopped at 2.01 t/s because every layer still paid router-ID D2H,
CPU lookup/deduplication, resident-slot-to-compact D2D copies and upload-stream
synchronization.

The K8 quality experiment is **paused by user decision**. Its coffee-learned
mask was tested with a different short prompt, so G28 carries no quality claim.
Direct K8 execution or a physical K8 bake remains a deferred TODO, not the next
active task.

Local provenance:

- `G27_REAP_MASS_WRAP_RESULTS.md`
- `G28_STATIC_K8_AND_EMBED_ROW_STAGING_RESULTS.md`
- branch `port/windows-dynamic-arena-0051`

## What Colibri actually implements

Repository audited: <https://github.com/JustVugg/colibri>, commit
`bad64d1b06cbda80dbdfb1f8f502370d0407bcbb`.

Colibri is not directly comparable in model geometry or quantization: it runs
GLM-5.2 int4 with 75 sparse layers and 21,504 routed experts. Its runtime is
still valuable because its tier boundaries are explicit in code.

### Router and placement are separate

`c/glm.c::moe()` computes the normal, unmasked router top-k first. Only after
selection does it resolve each expert through:

1. the pinned hot store (`pin[layer]`);
2. the per-layer LRU (`ecache[layer]`);
3. a true miss loaded into a working slot (`ws[]`).

Residency therefore does not alter router scores under the quality policy.
This matches our desired separation between the model's request and the
runtime's placement decision.

### VRAM is persistent, RAM is a compute tier

At startup, `pin_load()` ranks experts from recorded usage. A prefix is uploaded
as persistent `ColiCudaTensor` objects; the next-ranked suffix stays in RAM. With
`CUDA_RELEASE_HOST=1`, the host copy of a successfully uploaded expert is freed,
so VRAM and RAM hold disjoint hot and warm sets instead of duplicates.

A VRAM hit executes directly from those persistent tensor objects. A cold
expert is not copied to CUDA for one use: it is read into RAM and executed by
the CPU path. This is the central design difference from our selected-upload
path.

### Cold I/O is coalesced and deferred

`expert_load()` reads contiguous gate/up/down weights with one coalesced read
when possible and keeps tensor views inside that slab. Misses are either loaded
with an OpenMP parallel loop or dispatched to a persistent eight-worker I/O
pool. The runtime submits resident GPU work before waiting for cold loads, then
joins the cold results before completing the layer.

This does not make a miss free. It changes a serial dependency into:

`resident compute || cold read -> join`.

The profile reports disk service time separately from foreground-visible wait,
which is a useful requirement for our ledger.

### Prefill deduplicates by expert

For `S > 1`, `moe()` constructs the union of experts requested by all positions.
Each unique expert is loaded once and applied to every row that selected it.
This is a concrete answer to our slow-prefill problem: prefill should be planned
as a batch-union workload, not as many independent token fetches.

### Live re-pin is deliberately slower than routing

Colibri keeps persistent usage (`eusage`) separate from session heat and recency
(`eheat`, `elast`). At safe request boundaries it performs at most four swaps,
uses a frequency-dominant LFRU score, decays session heat, and applies hysteresis.
For a GPU swap it reuses the existing allocation and refreshes its contents via
`coli_cuda_tensor_update()`.

This supports a two-clock design for DS4:

- **fast clock:** token-level logical mask and mass accounting;
- **slow clock:** bounded physical RAM/VRAM replacement;
- **exception:** immediate admission only when a missing expert's router weight
  is high enough to justify the transfer.

### PILOT is useful, but not magic

Colibri's `PILOT` applies the next layer's router to the current layer's
post-attention state. Its documentation reports 71.6% top-8 recall versus 41.3%
for previous-token reuse. A dedicated queue issues hints off the main thread.
`PILOT_REAL` can load a predicted expert into a future layer LRU, using a
hide-until-complete publication protocol so inference never sees a partial slot.

The same documentation reports PILOT as neutral when storage is already near
saturation. Prediction only helps if there is idle transfer time or resident
compute behind which the read can hide. SPEX should therefore be evaluated as a
miss-latency overlap mechanism after the residency data path is efficient, not
as a substitute for it.

### Native Windows has no hidden memory bypass

Colibri's Windows port uses:

- `ReadFile` plus `OVERLAPPED` for positional reads;
- `VirtualLock` with best-effort working-set growth;
- `GlobalMemoryStatusEx` for available-memory planning;
- `FILE_FLAG_NO_BUFFERING` for its direct-I/O option;
- a CUDA DLL loaded through `LoadLibrary`.

This validates native Windows as a workable target, but it does not expose a
special shared-memory path or unlimited pinning. Capacity and WDDM behavior
still have to be measured on our machine.

## What transfers to DS4

### Priority 1: persistent direct expert slots

Our future hot tier needs a device-side `expert_id -> slot` map. A hit must feed
the MoE kernel from the resident slot directly. Repacking resident experts into
a compact selected buffer every layer recreates the transfer and sync tax even
when the weights already fit.

This is the strongest lesson from G28 and Colibri together. G28 measures our
remaining tax; Colibri demonstrates persistent tensor identity and in-place
slot refresh.

### Priority 2: keep the residual stream on device

Colibri's newer resident pipeline adds device-side RMSNorm, RoPE, residual add,
attention and shared-expert primitives so activations can remain on the GPU for
most of a layer. Its routed-expert group still performs a final sync/download,
so it is a pattern, not a finished answer. For DS4 the target is one synchronization
at a real dependency boundary, not one per small operation.

### Priority 3: batch-union prefill

During prefill, gather the union of routed experts across the batch/chunk and
load each unique payload once. Record:

- unique experts per layer;
- reuse factor across prompt positions;
- bytes read per prompt token;
- prefill tokens/s and TTFT separately.

This lever is independent of REAP-LOOP and should be tested before composing it
with adaptive placement.

### Priority 4: concurrent hot, warm and cold work

The desired layer schedule is:

1. launch VRAM-resident experts;
2. compute RAM-resident experts on CPU or stage them from pinned RAM in bulk;
3. load true cold misses in a bounded I/O pool;
4. join once, preserving deterministic accumulation order where required.

Which warm path wins is hardware-specific. Colibri's own measurements show an
AVX-512 CPU matching a 5090 for expert work on one host. That does **not** prove
CPU compute will win on this 3060 system. DS4 IQ2XXS also needs an efficient CPU
kernel before the comparison is meaningful. We need a measured microbenchmark,
not an architectural vote.

### Priority 5: physical adaptation behind hysteresis

REAP mass remains the placement signal. Do not rebuild the physical arena for
every small score change. Keep the logical selection responsive, but rotate
physical slots only when:

- the entrant has materially higher mass than the victim;
- or a high current router weight triggers the emergency path;
- or a request/domain boundary permits a larger bulk re-seed.

G27 already proves transactional WRAP correctness and also measures why doing
too much of it in a short decode loses throughput.

### Priority 6: prediction after transport

SPEX/PILOT should predict candidates for the **next physical need**, not decide
model semantics. Measure recall, precision, bytes wasted, confirmed-miss wait
saved and queue contention. Predicted work must be bounded and preemptible by a
confirmed miss.

## Corrected priority table

| Rank | Lever | Expected value | Main risk | Gate |
|---:|---|---|---|---|
| 1 | Direct persistent VRAM slot lookup | Removes measured D2H/lookup/D2D/sync tax on hits | Kernel/index integration | Exact A/B, `n>=3`, per-layer copy bytes |
| 2 | Device-resident layer data plane | Removes repeated activation transfers and small-op sync | Larger CUDA change | Stage timing and output regression |
| 3 | Batch-union prefill | Avoids repeated expert reads across prompt rows | Memory peak, batch numerics | Prefill A/B, TTFT, unique-expert reuse |
| 4 | Concurrent VRAM/RAM/cold execution | Hides cold service behind useful work | Deterministic join, CPU kernel quality | Timeline plus foreground wait |
| 5 | Bounded mass/LFRU physical re-pin | Tracks domain without constant transfers | Churn and slow amortization | Long decode/request A/B |
| 6 | SPEX/PILOT prefetch | Hides remaining predictable misses | Wasted bandwidth | Recall plus saved wait, not recall alone |
| spike | CPU IQ2XXS expert kernel | Could eliminate RAM-to-GPU traffic | May be slower on this CPU | Standalone expert microbenchmark |

The previous recommendation to prioritize three independent gate/up/down
transfer streams is downgraded. It can improve a real miss, but G28 shows that
we can remain slow with zero expert evictions. First remove work performed on
every resident hit.

## Measurement contract

Every performance run should record, in one machine-readable result:

- executable, source and model hashes;
- complete environment and command line;
- prompt hash, context, generated tokens, sampling and `nothink`;
- prefill tokens/s, TTFT and decode tokens/s;
- process read bytes and disk service time;
- H2D, D2D and D2H bytes plus synchronization count/time;
- VRAM/RAM/cold hits, compulsory misses and evictions;
- WRAP publications, entrants, victims and total WRAP time;
- GPU utilization and dedicated/shared memory;
- output hash and quality grade where semantics can change.

Use `n>=3` for verdicts. Keep `n=1` only for safety, exactness and mechanism
probes. A cache-fit or transport test with a cross-domain mask must never be
reported as a quality result.

## Deferred TODOs

- K8 direct-slot execution or physical K8 GGUF bake, resumed only on request.
- Mixed expert quantization and further cold-expert compression.
- Same-domain K8 quality grading with at least three valid runs.
- CPU IQ2XXS expert-kernel feasibility benchmark.
- Compare Colibri-style batch-union prefill with DS4's current prefill path.
- Add a first-class placement planner and runtime telemetry similar in spirit
  to Colibri's `resource_plan.py`, adapted to GGUF and WDDM.

## Reference landscape

The original survey remains directionally useful, with these corrections:

| Project | Transferable lesson | Caveat |
|---|---|---|
| chripell/ds4-rtx3090 | pinned host memory plus persistent expert cache can be fast | 3090, 128 GB RAM, native Linux; not our WDDM capacity |
| KTransformers / llama.cpp CPU MoE | compute-in-place can avoid PCIe traffic | depends strongly on CPU kernels and memory bandwidth |
| llama.cpp hybrid CPU/GPU MoE work | execute resident and missed rows concurrently | integration is substantial and version-specific |
| ProMoE / PreScope | prediction needs bounded, preemptible transfer and scheduling | prediction accuracy alone is not throughput |
| Colibri | disjoint VRAM/RAM tiers, batch-union, deferred cold I/O, safe live repin | different model, int4 format and CPU-first runtime |
| mixtral-offloading | fixed buffers and overlap beat per-miss allocation | smaller model geometry |

Colibri source: <https://github.com/JustVugg/colibri>. The audited implementation
is Apache-2.0, but any borrowed code still requires a license and integration
review against DS4's repository before use.
