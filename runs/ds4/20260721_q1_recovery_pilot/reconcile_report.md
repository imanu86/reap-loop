# CPU expert-forward discrepancy reconciliation

## Verdict

The two programs compute the same batch-1 expert forward. The approximately 14x
discrepancy was a timing bug in the original overlap harness: it scanned a 512 MiB
cache-eviction buffer **inside the timed loop, once per expert**. The standalone
spike also evicts 512 MiB, but does so **before** starting each expert timer. The
extra 512 MiB scan accounts for about 16.2 ms of the reported 17.5--17.9 ms per
expert.

There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
masks are `0x3`, `0xc`, `0x30`, `0xc0`, and so on. The old `0x3f` and `0xff`
masks allowed 6 and 8 OpenMP workers, respectively, but confined them to only 3
and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."

After fixing both issues, a CPU-only run over 24 distinct, cache-cold, resident
experts measures:

| CPU allocation | Total for 24 experts | Mean per expert | Effective packed-byte rate |
|---|---:|---:|---:|
| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |

For planning, use **0.9--1.0 ms/expert when reserving two cores** and
**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
that also covers the legacy SMT-paired masks is **0.7--1.7 ms/expert**, not
17.5 ms/expert.

The CPU tail itself is therefore viable enough to continue the overlap experiment:
24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
overlap run is still needed to measure H2D/DRAM contention; no GPU code was run for
this reconciliation.

## 1. The work is the same

Both sources define embedding width 4096 and hidden width 2048
(`overlap_spike.c:12-14`; `../cpugemv_spike/cpugemv_spike.c:13-15`) and validate
the same three tensors:

- gate: 2048 x 4096 = 8,388,608 weights;
- up: 2048 x 4096 = 8,388,608 weights;
- down: 4096 x 2048 = 8,388,608 weights;
- total: 25,165,824 weights.

The geometry checks are `overlap_spike.c:203-209` and
`../cpugemv_spike/cpugemv_spike.c:219-225`. Both load gate, up, and down payloads
with the same layout (`overlap_spike.c:212-233` and
`../cpugemv_spike/cpugemv_spike.c:228-258`). The actual packed size is 7,077,888
bytes/expert = 6.75 MiB: 2,162,688 gate + 2,162,688 up + 2,752,512 down
(`../cpugemv_spike/cpugemv_spike_results.json:8`). Gate/up are IQ2_XXS; down is
Q2_K, so "exact IQ2" here means the exact model quantization path, not that all
three tensors use the same IQ2 type.

The two `quant_expert_forward` bodies are functionally identical
(`overlap_spike.c:395-420`; `../cpugemv_spike/cpugemv_spike.c:431-456`):

1. quantize the 4096-float input to Q8_K;
2. compute gate and up GEMVs for all 2048 hidden rows;
3. apply `silu(gate) * up` once per hidden row;
4. quantize the 2048-float intermediate to Q8_K;
5. compute the down GEMV for all 4096 output rows.

There is no batch greater than one, no second SiLU, and no full FP32 matrix
dequantization. Weight decoding occurs on the fly inside the quantized dot product.
The input Q8 conversion is redundantly repeated for each expert even though the
input is shared, but it touches only 4096 floats and is not remotely large enough
to explain 16 ms.

The overlap harness does not reload weights in the N loop. It reads all 24 distinct
experts once during context creation (`overlap_spike.c:483-489`), before timing,
then indexes `experts[i]` in the timed loop (`overlap_spike.c:547-550`).

## 2. Exact cause of the 17.5 ms number

The standalone cold benchmark has this order:

```c
evict_cache(evict, evict_bytes); // 512 MiB scan
double t0 = now_sec();
quant_expert_forward(...);
```

See `../cpugemv_spike/cpugemv_spike.c:614-623`. Thus its reported 1.267 ms is a
genuinely cache-cold expert forward, not a hot-L3 best case. Its warm measurement
is 1.257 ms, almost identical
(`../cpugemv_spike/cpugemv_spike_results.json:15`). Reusing expert 0 changes what
is allocated, but the explicit 512 MiB sweep makes each reported cold iteration
start with expert 0 out of cache.

The original overlap source instead started the tail timer and then did this for
every expert:

```c
for (int i = 0; i < n_experts; i++) {
    if (ctx->evict) evict_cache(ctx->evict, ctx->evict_bytes);
    quant_expert_forward(&ctx->experts[i], ...);
}
```

That charged 512 MiB of unrelated reads to every 6.75 MiB expert. The corrected
path now evicts once before `t0`, then streams distinct experts without synthetic
inter-expert eviction (`overlap_spike.c:542-550`). Distinct expert weights have no
reuse to preserve, so every subsequent payload is naturally cold.

The numerical reconciliation is nearly exact:

| Mode | Original overlap result | Corrected result with same legacy mask | Difference attributable to timed eviction |
|---|---:|---:|---:|
| 8 workers, `0xff`, N=24 | 17.586 ms/expert | 1.334 ms/expert | **16.252 ms/expert** |
| 6 workers, `0x3f`, N=24 | 17.939 ms/expert | 1.727 ms/expert | **16.212 ms/expert** |

A 512 MiB read in 16.2 ms is about 33 GB/s, entirely plausible for this machine.
The old N=4 and N=24 values were constant per expert because the harness injected
one identical 512 MiB scan per expert.

## 3. Cache and bandwidth interpretation

The standalone result's 5.584 GB/s is `packed_bytes / complete_forward_time`, not
a measured hardware DRAM ceiling. It includes scalar/table IQ2 decoding, Q2_K
math, OpenMP scheduling, activation quantization, and SiLU. It therefore cannot be
used as a hard bandwidth floor. The eviction scan itself demonstrates roughly
33 GB/s of cache-line traffic on the same CPU.

With correct physical-core affinity, the complete distinct-expert forward reaches
7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
be below the proposed 1.2 ms floor while still reading every packed weight. The
observed practical floor in this implementation is around 0.7 ms on eight physical
cores, not the raw DRAM floor.

"Cold" here means cache-cold but resident in RAM. The harness loads and faults in
all packed experts during setup. This matches a REAP design that keeps cold expert
payloads resident in system memory; disk I/O or first-touch page faults would be a
separate cost.

## 4. Threading and contention

- Both matrix phases are OpenMP-parallel over output rows
  (`overlap_spike.c:398-407` and `overlap_spike.c:411-419`). Weight decoding is
  inside those loops, so it is parallel. Only the small activation-to-Q8_K
  conversions are serial (`overlap_spike.c:396` and `overlap_spike.c:409`).
- The CPU-only verifier requested teams of 1, 2, 4, 6, and 8 and observed exactly
  those effective team sizes (`overlap_cpu_bench.c:32-45`; recorded in
  `overlap_cpu_physical_results.json:8-32`). There was no fallback to one thread.
- There is no explicit `CreateThread` inside the expert loop. Each expert enters
  two OpenMP parallel regions; the runtime manages/reuses its worker pool. The
  CUDA harness's one Win32 CPU-job thread is created once per overlap result row,
  before the overlap wall timer, not once per expert (`overlap_spike.cu:309-323`).
- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
  core (`overlap_spike.cu:103-112`).
- CUDA submit/driver contention cannot explain the discrepancy: `cpu_tail_ms` in
  the SERIAL path is measured only after synchronized GPU work
  (`overlap_spike.cu:302-308`), yet it showed the same approximately 17.5 ms/expert.
  The CPU-only correction also reproduces the expected approximately 1 ms result
  with no CUDA initialization.

## 5. CPU-only confirmation

The confirmation used the same MSVC `/O2 /openmp` compilation model, a single
512 MiB eviction before each timed tail, 10 repetitions per row, and distinct
preloaded experts. Full results are in `overlap_cpu_physical_results.json`.

| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
|---:|---:|---:|
| 4 | 3.948 / 0.987 ms | 2.700 / 0.675 ms |
| 8 | 7.097 / 0.887 ms | 5.723 / 0.715 ms |
| 16 | 14.455 / 0.903 ms | 11.121 / 0.695 ms |
| 24 | 22.345 / 0.931 ms | 17.393 / 0.725 ms |

The nearly linear totals show that distinct streaming does not introduce a hidden
per-N reload or growing cache penalty.

## 6. Fixes and remaining optimization headroom

Implemented fixes:

- moved the eviction scan before the tail timer and reduced it to once per tail
  (`overlap_spike.c:542-550`);
- corrected default physical-core masks and documented the SMT topology
  (`overlap_spike.cu:91-112`, `README.md`);
- added a CUDA-free verifier with effective-team reporting
  (`overlap_cpu_bench.c`, `build_overlap_cpu_bench.bat`);
- added the eviction scope to future overlap JSON output
  (`overlap_spike.cu:228-232`).

The full CPU/CUDA harness was rebuilt successfully, but not executed.

Further optimization is incremental rather than a prerequisite for viability:

- **Parallelize dequant:** there is no full-matrix dequant stage to fix. Weight
  decoding is already parallel. Parallelizing 4096 + 2048 activation values would
  add synchronization and is unlikely to help; hoisting the shared input Q8
  conversion out of the expert loop is a safe small optimization.
- **Hoist the thread pool:** there is no per-expert Win32 spawn/join. A persistent
  OpenMP region around the full tail could remove two parallel-region barriers per
  expert and reduce jitter, but it is not the source of the discrepancy.
- **Cache-block GEMV:** distinct weights are single-use, so blocking cannot create
  weight reuse. Inputs and lookup tables are already small/hot. Expert-parallel
  lanes or a fused persistent region may improve scheduling, but the corrected
  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.

The next decision point is therefore not CPU GEMV feasibility; it is a future
corrected overlap run measuring whether the approximately 22 ms six-core tail
remains near that value while real GPU H2D and compute are active.
