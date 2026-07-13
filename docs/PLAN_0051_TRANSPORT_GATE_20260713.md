# 0051 transport gate and dynamic arena decision plan

**Status:** preregistered execution plan, 2026-07-13.  
**Branch:** `plan/0051-transport-gate-20260713`  
**Base commit:** `f0b9a4b021e2d832a428d2ab8d31045f4e5434e2`  
**Scope:** Linux/RunPod reference first, native Windows comparison when the
port is ready, then a measured go/no-go decision for patch 0051.

This plan supersedes the assumption that patch 0051 must immediately be a
24 GiB dynamic pinned arena. The arena remains a candidate architecture. It
is implemented only if a controlled transport gate proves that direct pinned
sources retain a material advantage after the existing asynchronous pipeline
is correctly enabled.

No static domain mask is accepted as the production policy. Static masks are
allowed only as mechanism fixtures when all compared arms use the same
selection target.

## 1. Objective

Remove the decode critical-path toll caused by expert fetches:

`pread/pageable source -> pinned staging -> H2D -> host/device sync`

The final system must preserve:

1. GEMM execution from VRAM;
2. a selection mask that may change between interactions and during a turn;
3. correct fallback for experts not present in the pinned tier;
4. exact artifact provenance and repeatable measurements;
5. no quality verdict from `n=1`, token repetition flags, or visual intuition
   alone.

## 2. Measured facts frozen before execution

### 2.1 Native bandwidth is not the DS4 throughput

- Native Windows RTX 3060 pinned H2D: 24.1-24.4 GiB/s.
- WSL2 pinned H2D reference: about 3.0 GiB/s.
- Native A5000/PCIe Gen4 pinned H2D: 23.63 GiB/s.
- The DS4 coffee run on that native Gen4 node used only 2.4-3.6 GiB/s
  effective copy bandwidth and produced 4.4-4.9 decode t/s.
- The run observed roughly 450-490 VRAM-miss copies per token and hundreds of
  thousands of CUDA synchronizations.

Therefore the physical PCIe link is not the current ceiling. Transfer
granularity, staging, admission behavior, and synchronization must be isolated
before a new large host-memory architecture is justified.

### 2.2 The chripell fork is a reference, not a patch base

The two relevant commits are:

- `3fc5b217ee41b03ea117b03511613c92f23b05ba`
- `5a854d2d4d496df4bb845087bce3fa82661f588d`

They demonstrate the useful principle that a pinned mmap source can avoid
`pread` staging, but they are not compute zero-copy and do not provide a real
overlapped pipeline. The implementation attempts to register the whole model,
has an unsafe fallback when registration fails, retains per-copy
synchronization, and has an incorrect unregister/munmap lifetime. It must not
be ported unchanged.

### 2.3 Patch 0050 proved mechanism activation, not its ceiling

Patch 0050 improves the chripell concept by adding bounded selected mmap
registration, fallback, diagnostics, and correct drain/unregister/munmap
lifetime. The decisive gate was nevertheless underpowered:

- `DS4_SELECTED_UPLOAD_EVENT` was not enabled, so covered gate/up/down copies
  were followed by `cudaStreamSynchronize`;
- a 24 GiB budget registered 3,640 expert triplets, or 91 per routed layer;
- those experts were admitted by contiguous expert-ID order rather than by
  routing mass;
- the actual pinned subset covered only about 44-57 percent of observed
  routing, despite a much wider compute mask;
- exactness after final hardening is not closed because the only controlled
  OFF/ON pair differed in its last token and there are no repeated OFF/OFF and
  ON/ON controls.

### 2.4 Existing asynchronous work must be evaluated first

The current repo already contains:

- `0032-async-pipeline-rebased.patch`: decode upload/compute overlap;
- `0047-no-whole-mmap-register.patch`: bounded staging configuration;
- `0048-prefill-overlap-s1.patch`: protected staging ring and prefill overlap.

Patch 0048 has already produced byte-identical outputs in its campaign and
reduced prefill time by about 10-21 percent. Its separate speculative prefill
readahead arm degraded WSL performance by about 46-59 percent and remains OFF.
Patch 0032 has not yet been measured on the current 0050 stack.

### 2.5 Offline routing coverage and churn

At 24 GiB the arena geometry is exactly 3,640 slots of 7,077,888 bytes. A
fair per-layer mass ranking gives 91 slots per each of 40 routed layers.

Measured on existing K0 traces:

- same-trace top-91/layer coverage: about 91-99 percent of routing mass;
- causal first-half to second-half coverage: about 85-96 percent of mass;
- global top-3,640 improves same-trace coverage by only 0.2-0.4 percentage
  points over the fair per-layer quota;
- adjacent 64-token windows retain about 67-82 percent of slots;
- a window change would replace about 684-1,214 slots, roughly 4.5-8.0 GiB.

These are `n=1` trace statistics and are not a quality verdict. They show that
0050 selected its pinned subset poorly and that a fully rotating 24 GiB arena
may have a significant refill cost.

## 3. Experimental invariants

Every measured arm must save:

- DS4 base commit and complete ordered patch list;
- SHA256 of `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`, executable, and model;
- GPU name, VRAM, driver, CUDA runtime, PCIe maximum/current state;
- complete CLI and environment after expansion;
- prompt, request, raw streamed response, extracted content, and stop reason;
- wall timestamps, TTFT, prefill duration/rate, decode duration/rate;
- cache hits/misses, DMA bytes, fallback bytes, direct-pin coverage;
- copy calls/time, synchronization calls/time, event waits, and failures;
- host available memory, process RSS, pinned allocation, VRAM use, and thermal
  state where available;
- functional L0-L3 grade and the grader version.

Secrets, API tokens, SSH private material, and ephemeral access URLs must never
be committed.

All performance comparisons use at least three measured repetitions per arm,
after an explicit warmup, in alternating or balanced order. Cold-start and
warm-state results are reported separately. A micro-smoke may reject an arm
but cannot promote it.

## 4. Frozen transport matrix

All arms use the same model, prompt, context, sampling, compute-selection mask,
VRAM cache capacity, and stopping rule. The pin plan is separate from the
compute-selection mask so that host transport can be isolated from model
quality.

### Arm A: synchronous staged baseline

- Current canonical source stack.
- Direct selected mmap pinning OFF.
- Selected-upload event OFF.
- Async pipeline OFF.
- Existing `pread -> pinned stage -> H2D -> sync` behavior.

### Arm B0: asynchronous staged transport, reactive predictor disabled

- Arm A plus the compatible portions of 0032, 0047, and 0048.
- Persistent staging cursor and write-after-read event protection are
  mandatory before deferred synchronization is enabled.
- Decode upload/compute overlap ON.
- Prefill overlap S1 ON.
- Speculative readahead S2 OFF.
- `DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER=1`, so the reactive L-to-L+1 expert
  predictor cannot change residency traffic in this arm.
- Direct selected mmap pinning OFF.

This arm isolates event/barrier/staging behavior from prediction quality.

### Arm B1: full 0032 asynchronous pipeline

- Same binary and transport settings as B0.
- Reactive L-to-L+1 expert prefetch enabled.
- Direct selected mmap pinning OFF.

The B1/B0 comparison measures the incremental value or cost of the reactive
predictor. `DS4_ASYNC_PIPELINE=0` must behave exactly like unset; the integrated
patch must not treat the mere presence of a zero-valued environment variable
as enabled.

### Arm C1: asynchronous direct pin, legacy allocation order

- The better valid B0/B1 transport base plus patch 0050 selected mmap pinning.
- Reproduce the existing contiguous expert-ID admission policy.
- This arm isolates the cost of direct pinned sources while preserving the
  known suboptimal coverage.

### Arm C2: asynchronous direct pin, mass-ranked allocation

- Same binary behavior as C1 except for the generated pin plan.
- Keep 91 expert triplets per routed layer, ranked by measured gate mass.
- Complete gate/up/down triplets move as one unit.
- Use a causal plan built from an earlier observation window for promotion
  evidence.
- A same-trace oracle plan may be run once as an explicitly labelled upper
  bound. It cannot support a production or quality claim.

The C1/C2 comparison determines how much of the 0050 result was lost to the
expert-ID allocation policy. The selected B*/C2 comparison determines whether
direct pinning still matters after correct overlap.

## 5. Test sequence

### Phase 0: provenance and build hygiene

1. Freeze the pod source and binary hashes before editing.
2. Reconstruct the ordered patch chain from source, not from filenames or
   memory.
3. Build distinct, named binaries for A and B*/C from the same DS4 base.
4. Run `git apply --check` and a clean compile for the combined 0032/0047/0048
   stack.
5. Confirm no DS4 server is active before each arm and stop only the recorded
   PID after it.

### Phase 1: event and staging safety

1. Exercise the staging ring with deferred synchronization and forced rapid
   reuse.
2. Verify the persistent cursor and write-after-read fence from patch 0048.
3. Inject event creation/record failures and require a safe synchronized
   fallback.
4. Require zero stale-slot reads, zero checksum mismatches, and zero DMA
   failures before throughput tests.

### Phase 2: exactness and repeatability

Use a short deterministic coffee request at `temp=0`, 60 generated tokens:

1. A/A/A, B0/B0/B0, B1/B1/B1, C1/C1/C1, and C2/C2/C2
   repeatability;
2. balanced cross-arm order rather than one sequential OFF/ON pair;
3. byte hashes and first differing token/byte for every mismatch;
4. no causal attribution when within-arm nondeterminism is as large as
   between-arm divergence.

### Phase 3: throughput on the full 3090 pod

Use a bounded decode request long enough to reach a stable regime without
mixing context saturation into the result. Run at least three repetitions per
arm and report median, range, and each raw run.

Primary comparisons:

- B0 versus A: value of asynchronous staging and reduced synchronization;
- B1 versus B0: incremental value of reactive L-to-L+1 prefetch;
- C1 versus the selected B*: direct pinning with the old allocation policy;
- C2 versus C1: value of mass-ranked pinned coverage;
- C2 versus the selected B*: residual value that could justify a dynamic
  arena.

The physical H2D microbenchmark is repeated once on the node but is never used
as a substitute for effective DS4 copy bandwidth.

### Phase 4: quality campaign

Performance and quality are separate campaigns. Use the canonical coffee and
cyberpunk tasks with enough context/output budget to complete the document,
preferably context 8192, maximum output 4000, and a stop condition at a valid
`</html>` rather than judging an arbitrary truncation.

- At least `n=3` per promoted arm.
- Grade L0-L3 with the committed functional grader.
- Preserve complete HTML and rendered/functional evidence.
- A coherent page truncated mid-CSS is labelled token-budget truncation, not
  model collapse.
- An arm that immediately emits malformed code may be rejected early, but an
  isolated L0 cannot establish a stochastic quality rate.

### Phase 5: 12 GiB envelope

Repeat the promoted A/B*/C2 subset with the second 3090 configured to expose
the same DS4 VRAM budget and cache pressure as the local 12 GiB setup.

This is a memory-envelope test, not a 3060 speed proxy: the 3090 retains its
larger compute throughput and memory/PCIe characteristics. Record the exact
mechanism used to cap DS4 VRAM; do not claim the physical GPU became a 3060.

### Phase 6: native Windows port

When the native port is ready, run the same semantic arms and counters on the
local RTX 3060:

1. verify source/binary/model provenance;
2. confirm native pinned and pageable H2D bandwidth with the existing probe;
3. repeat exactness before performance;
4. compare effective DS4 copy bandwidth, copy granularity, and sync time with
   Linux/RunPod and WSL artifacts;
5. keep OS/runtime conclusions separate from GPU-model conclusions.

Native Windows may remove the WSL pinning and bandwidth ceilings, but it does
not by itself remove DS4 small-copy and synchronization overhead.

## 6. Go/no-go criteria for patch 0051

Correctness gates are absolute:

- no stale-slot or partial-triplet use;
- no event/staging overwrite race;
- no unexplained exactness regression beyond measured within-arm variance;
- no quality regression in the `n>=3` L0-L3 campaign;
- fallback remains functional for all unpinned experts.

Proceed to a dynamic pinned arena only when all of the following are measured:

1. C2 reaches at least 80 percent direct-pin routing-mass coverage causally;
2. C2 improves median decode throughput over the selected B* by at least 10
   percent, or
   reduces measured copy/synchronization stall by at least 20 percent with a
   credible path to end-to-end gain;
3. the benefit repeats on both the full 3090 and the 12 GiB envelope, or the
   hardware-specific exception is measured and documented;
4. estimated arena refill cost is amortized within the measured useful turn
   length rather than inferred from bandwidth alone.

If the best B0/B1 arm closes nearly all of the gap, stop 0051 and keep the
simpler asynchronous staging architecture. Complexity is not accepted for a
negligible residual gain.

## 7. Conditional 0051 architecture

If the gate passes, implement 0051 incrementally rather than as one large
patch:

1. one permanent `cudaHostAlloc` arena allocated after mandatory context and
   staging allocations;
2. fixed-size slots containing complete expert triplets;
3. dense `(layer, expert) -> slot/generation/state` tables;
4. no dynamic `cudaHostRegister`/`cudaHostUnregister` in the inference path;
5. per-slot or per-batch CUDA events fencing H2D reads before reuse;
6. WRAP batch refill of only the target delta;
7. quiescent publication of selection mask and slot map as one snapshot;
8. normal pageable/pread fallback for selected but unpinned experts;
9. measured source-page reclamation, with no assumption that `DONTNEED`
   immediately reduces physical memory;
10. engine-scoped ownership and drain-before-free teardown.

The first policy candidate is hybrid rather than full rotation:

- a stable mass-ranked core;
- a smaller reassignable frontier;
- fair per-layer quotas to prevent starvation;
- optional global spare slots only if measured to improve coverage;
- live current-interaction routing mass as input, never a named-domain mask.

Core/frontier percentages and update cadence are not fixed by this plan. They
must be chosen from measured retain, refill bytes, and useful-turn latency.

## 8. Pod execution and cost discipline

- Full 3090 pod: build, safety, exactness, and complete A/B0/B1/C1/C2 matrix.
- 12 GiB-envelope pod: acceptance under local memory pressure.
- Do not stop a pod while a model download or irreplaceable artifact upload is
  in progress.
- Before releasing a pod, verify that all raw logs, manifests, outputs, and
  source/binary hashes are present in `runs/ds4/...` and uploaded or committed.
- Never keep a pod running merely because a launcher PID file exists; verify
  the actual process and GPU state.

## 9. Immediate next actions

1. Let the active full-3090 pod finish its R2 model transfer.
2. Audit and snapshot the pod source/binary state without modifying the active
   download.
3. Prepare A and the integrated B*/C build first; do not start 0051
   implementation.
4. Close Phase 1 staging/event safety.
5. Run the repeated exactness matrix.
6. Run full-3090 throughput A/B0/B1/C1/C2.
7. In parallel, ingest the native Windows port and map its memory/copy APIs to
   the same counters and invariants.
8. Update this document with artifact paths and a measured go/no-go verdict.
