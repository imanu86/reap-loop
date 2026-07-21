# CPU-GEMV Feasibility Spike: P3 of G130

## Verdict

**NO-GO for cold->CPU / hot->GPU hybrid dispatch on this machine.**

The CPU path is correct enough, but it is not fast enough per cold expert. The best measured full-machine CPU expert forward is **1.267 ms/expert at 8 threads**, while the attributed GPU cold path is **0.52-0.72 ms/expert including the ~0.02 ms GPU GEMM**. Replacing current cold GPU expert loads with serial 8-thread CPU expert forwards would reduce projected full/open throughput from **1.65 t/s** to about **1.44 t/s** at the nominal 150-cold-expert/token attribution.

An optimistic CPU-lane model using eight independent single-thread experts gives `5.103 ms / 8 = 0.638 ms/expert` asymptotically. That only barely beats the nominal/high GPU attribution when the lane queue is full, loses against the low attribution, and was not directly measured as a multi-expert concurrent workload. It is not enough margin to justify G130 implementation complexity.

## Benchmark

Artifacts:

- Native benchmark: `cpugemv_spike.c`
- Build wrapper: `build_cpugemv_spike.bat`
- Extracted ds4 IQ2 tables: `ds4_iq2_tables.inc`
- Machine-readable results: `cpugemv_spike_results.json`

Build/run:

```bat
build_cpugemv_spike.bat
cpugemv_spike.exe
```

Inputs:

- Model: `C:\ds4-models\ds4-2bit.gguf`
- Teacher weights: `C:\Users\imanu\g130i\recovery_poc\teacher_l3e0_fp32.npz`
- Real inputs: `C:\Users\imanu\g130i\trace_out\l3e0.vectors.f32le`

The harness reuses the ds4 CPU quantized math: Q8_K activation quantization, IQ2_XXS gate/up dot products, and Q2_K down dot products. On this MSVC x64 build, the ds4 IQ2/Q2 code path is the scalar fallback wrapped in OpenMP; I did not find an AVX2-specialized IQ2 GEMV in `ds4.c`.

## Tensor Geometry

Layer 3 expert 0 tensors found in the GGUF:

| Tensor | Type | Shape | Compressed bytes |
|---|---:|---:|---:|
| gate | 16, IQ2_XXS | 2048 x 4096 | 2,162,688 |
| up | 16, IQ2_XXS | 2048 x 4096 | 2,162,688 |
| down | 10, Q2_K | 4096 x 2048 | 2,752,512 |
| total | mixed | 25,165,824 weights | 7,077,888 |

## Correctness

Forward tested: `down(silu(gate(x)) * up(x))` for all 17 captured input vectors.

| Metric | Value |
|---|---:|
| Overall cosine | 0.9998191365 |
| Min per-sample cosine | 0.9996956753 |
| NMSE | 0.0003621614 |
| Max absolute error | 0.01613939 |
| Max relative error, denominator floored at 1e-6 | 6133.804 |

Correctness passes the requested `>0.999` cosine target. The large max-relative value is from near-zero teacher elements; cosine and NMSE are the useful pass/fail metrics here.

## Throughput

Timing is batch=1, one full expert forward, with a 512 MiB cache-eviction sweep before each cold timed iteration. Effective GB/s uses compressed weight bytes only: 7,077,888 bytes/expert.

| Threads | Cold mean ms/expert | Cold min ms | Warm mean ms | Effective weight GB/s |
|---:|---:|---:|---:|---:|
| 1 | 5.103 | 5.014 | 5.003 | 1.387 |
| 2 | 2.521 | 2.500 | 2.528 | 2.807 |
| 4 | 1.637 | 1.281 | 1.846 | 4.322 |
| 8 | 1.267 | 1.246 | 1.257 | 5.584 |

## GPU Comparison

Given attribution:

- `h2d_enqueue`: 33 ms/token
- `selection_d2h`: 66 ms/token
- About 150 cold expert-loads/token
- H2D + selection overhead: about 0.5-0.7 ms/expert
- GPU GEMM: about 0.02 ms/expert
- GPU cold path used for comparison: **0.52-0.72 ms/expert**

Measured CPU vs GPU per expert:

| Path | ms/expert | Result |
|---|---:|---|
| GPU cold path, low attribution | 0.52 | CPU loses |
| GPU cold path, high attribution | 0.72 | CPU loses vs 8-thread serial |
| CPU, 8-thread full-machine expert | 1.267 | Too slow |
| CPU, optimistic 8 single-thread lanes | 0.638 effective | Edge only; not robust |

Break-even:

- Serial 8-thread CPU experts: **never wins** for any cold-expert count, because `1.267 > 0.72` per expert.
- Eight single-thread CPU lanes: asymptotic `5.103 / 8 = 0.638 ms/expert`; can beat nominal/high GPU only when the queue is sufficiently full, but never beats the low 0.52 ms attribution.

## 43 x 8 Hybrid Arithmetic

Using the requested `43 layers x 8 routed = 344 routed expert slots/token`:

| Cold % | Cold experts | GPU cold ms, 0.52-0.72 | CPU 8-thread serial ms | CPU 8-lane optimistic ms |
|---:|---:|---:|---:|---:|
| 10% | 34.4 | 17.9-24.8 | 43.6 | 25.5 |
| 25% | 86.0 | 44.7-61.9 | 109.0 | 56.1 |
| 43.6% | 150.0 | 78.0-108.0 | 190.1 | 97.0 |
| 50% | 172.0 | 89.4-123.8 | 218.0 | 112.3 |
| 100% | 344.0 | 178.9-247.7 | 436.0 | 219.4 |

At the observed ~150 cold experts/token, serial CPU replacement adds about **+88 ms/token** at nominal GPU attribution. The optimistic eight-lane model saves only about **5 ms/token nominal**, which is inside likely integration overhead and ignores contention with router/CPU support work.

## Projected Full/Open t/s

Using G123 exact full/open IQ2 control at **1.65 t/s**, current token time is `1000 / 1.65 = 606.06 ms/token`.

Nominal cold GPU attribution: `150 * (0.66 + 0.02) = 102 ms/token`.

| Scenario | Token ms | Projected t/s |
|---|---:|---:|
| Current exact full/open IQ2 control | 606.1 | 1.65 |
| Replace 150 cold experts with measured 8-thread serial CPU | 694.2 | 1.44 |
| Optimistic 8 single-thread CPU lanes, no extra overhead | 601.0 | 1.66 |
| Unrealistic full overlap ceiling after removing nominal cold GPU work | 504.1 | 1.98 |

This does not move G130 toward the >6 t/s target. The useful next lever remains removing/overlapping GPU-side cold transport and D2H selection work, not computing exact cold experts on this CPU.
