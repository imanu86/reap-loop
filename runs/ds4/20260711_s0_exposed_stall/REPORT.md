# S0 — exposed expert-load-stall baseline on local 3060 (is there stall to hide?)

Gate for `docs/ASYNC_PIPELINE_PLAN.md` Stage-0: measure how much of time/token is
**exposed expert-load stall** vs compute, and the current **overlap%**, to decide
whether building the async prefetch pipeline is worth it — or whether warm-RAM
already hides the stall.

Host: RTX 3060 12 GB, WSL Ubuntu-24.04, sm_86. Bin `/root/ds4_pin/ds4`
(`ds4_cuda.cu` md5 `430716f4` = canonical v2.1 + 0024 resident-hit-fix + 0031;
**`DS4_PACE_PIN` OFF** -> clean reactive path, no prefetch, no pin — S0 = 0001 counters only).
Cross-checked against `/root/ds4` (`7d57f58d`, the CURVE/highK binary): identical t/s.
Model `/root/models/ds4-2bit.gguf` (86.7 GB on SSD; 43 MoE layers, 6 experts/layer/token).
Task: cyberpunk-wide (HTML-primed), static REAP masks, greedy temp0, ctx 2048.

Config (vaccinations honored): own lock `/tmp/ds4_s0_stall.lock` (UI:8000 never touched —
no 8000/8014 listener existed; GPU exclusive, only GPU job), `DS4_CUDA_NO_Q8_F16_CACHE=1`
(clean 2-bit path), warm-RAM lever (`DS4_CUDA_NO_DIRECT_IO=1`,`DS4_CUDA_KEEP_MODEL_PAGES=1`),
`DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1` (parse-safe, **NOT** 16 -> no cap/abort),
`DS4_SPEX_STATS=1` + `DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1`. Masks are **nested**
(K12 keeps subset of K23 keeps, verified) so one K23 warmup warms both. Warmth persists only
**within one WSL session** (WSL2 reclaims page-cache on process exit -> each fresh `ds4`
starts cold), so the whole warm sweep ran in a single session after one discard warmup.

## Method
`DS4_SPEX_STATS` prints `hit_rate`, `copy_ms`/`copy_ms_per_batch`, `sync_ms`/`sync_ms_per_batch`,
`copied MiB`. **Caveat on `copy_ms`:** it is a SUM over ~20 individually-timed async H2D
copies per batch (copy_calls/batch ~= 20) which overlap on the upload stream -> `copy_ms`
is NOT wall-clock (it exceeds the token time at cache32). So `copy_ms` is used as "copy
**work**", and the *recoverable* stall is read from the **decisive A/B** (vary residency,
watch t/s), not from the raw counter. `sync_ms` = time in `cudaStreamSynchronize(upload)`
(the blocking barrier at `:2237` the async design would replace with an event).
batches = tokens x 43.

## Regime 1 — warmth is the t/s driver (cold vs warm, same K23/cache32/hit0)
| state | gen t/s | ms/tok | note |
|---|---|---|---|
| cold (SSD, buffcache ~1 GB) | 0.65-0.91 | 1100-1540 | 1st run of a session; SSD page-faults |
| warm (RAM, buffcache ~57 GB) | **3.38** | 296 | 2nd+ run; keep-set resident in RAM |

3.7x from cold->warm at **identical** copy_ms_per_batch (~28-31 ms) => the speedup is the
SSD->RAM read, not the H2D copy. Matches CURVE (3.69) and highK (3.06-3.60) warm numbers,
and the ~0.9 cold. Model 81/86 GB > 60 GB RAM, so the *keep-set* (not the whole model) warms.

## Regime 2 — the DECISIVE A/B: residency removes the copy but NOT the time
All warm (buffcache 57 GB), cyberpunk-wide, ctx2048, greedy, -n140. **Bit-exact** within each
mask (genmd5 stable across cache sizes) — residency never changes tokens.

| run | mask | cache | **hit** | **gen t/s** | copied (GB) | copy_ms tot | sync_ms/tok | peakGPU |
|---|---|---|---|---|---|---|---|---|
| WS_K23_c32  | K23 | 32  | **0.000** | **3.38** | 274 | 188007 | 107 ms | 11672 |
| WS_K23_c256 | K23 | 256 | 0.018 | 3.23 | 270 | 32511 | 95 ms | 11998 |
| WS_K23_c516 | K23 | 516 | **0.728** | **2.94** | 98 | 12496 | 35 ms | 12005 |
| WS_K12_c256 | K12 | 256 | 0.014 | 3.00 | 271 | 37435 | 96 ms | 12014 |
| WS_K12_c516 | K12 | 516 | **0.890** | **3.09** | 59 | 12733 | 21 ms | 12014 |

Reading the K23 residency ladder (c32->c516): copy **work** collapses 15x (copy_ms
188007->12496; copied 274->98 GB), the blocking-sync falls 67% (107->35 ms/tok), yet
**gen t/s goes DOWN, not up (3.38->2.94)**. Eliminating ~93% of the expert-load copy buys
**zero** decode speedup — the residency/LRU bookkeeping slightly outweighs the copy saved.
=> At warm-RAM the reactive H2D copy + its barrier are **not a net exposed bottleneck**;
they are already hidden/absorbed behind the 43-layer compute pipeline. Net-recoverable ~= 0.

## Overlap% now = 0 (confirmed, not assumed)
`DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1` printed **nothing** in all 6 runs — the stats line
only prints when `scheduled>0 || seed_calls>0 || ready>0`. So `seed_calls=0`, nothing
scheduled, nothing seeded: the async/prefetch machinery is **completely inactive**, the
reactive path closes every copy with the blocking barrier before the MoE kernel.
**overlap% = 0 by construction, verified empirically.**

## K12-fits vs K23-wide — the "compute-bound" point is NOT faster
- K12 c516 = "fits" / compute-bound (**hit 0.89**, the task's ~0.92) -> **3.09 t/s**.
- K23 c32 = refetch-bound (**hit 0.00**) -> **3.38 t/s**.
The compute-bound<->refetch-bound **gap is ~= 0 (inverted)**: making the working set resident
does not raise t/s. The task's premise of a "1.5 (refetch) vs 3-4 t/s (compute)" gap that
async could recover is a **cold-vs-warm artifact** (cold K23 ~0.9-1.2 measured earlier vs
warm K12), NOT a compute-vs-refetch gap — at equal warmth both regimes sit at ~3.0-3.4 t/s.
(This reproduces CURVE/highK's "velocity flat vs K, driven by warmth+fit".)

## VERDICT (8 lines)
1. **overlap% now = 0** — HID/seed_calls=0 in all 6 runs; reactive blocking-barrier path, async machinery inactive.
2. **Exposed stall (naive counters, warm K23 c32 refetch):** blocking-sync = 107 ms/tok = **36%** of the 296 ms token; `copy_ms` overcounts (20 async copies/batch) so it is not wall-usable.
3. **Empirical recoverable ~= 0:** residency that removes 93% of copy-work and 67% of sync (K23 c32->c516) yields **no speedup** (3.38->2.94 t/s, worse) — the stall is already hidden at warm-RAM.
4. **compute-bound<->refetch-bound gap ~= 0:** K12 hit0.89 = 3.09 t/s <= K23 hit0.00 = 3.38 t/s -> no t/s headroom for async to recover on this 3060.
5. The "1.5 vs 3-4 t/s" opportunity the mission cited is a **cold(SSD)-vs-warm(RAM) artifact**, not compute-vs-refetch (cold->warm = 3.7x; hit0->hit0.89 warm = flat).
6. The only large stall is **cold SSD I/O** (~2 GB/tok), which 1-layer prefetch can't hide (copy >> 1-layer compute window, sec.4) and which warm-RAM (`KEEP_MODEL_PAGES`) already addresses.
7. Bit-exact across all caches (residency = loading, not selection) — the invariant holds.
8. **Do NOT build the async pipeline for this target (3060 + cyberpunk-wide): RAM-warmth already hides the expert-load stall — there is no exposed stall to recover.** Async pays off only where the working set fits VRAM so hit->1 actually raises t/s, or on hardware where SSD-cold is unavoidable *and* multi-layer predictive prefetch is available.

## Artifacts
`measure_s0.sh`, `warmsweep_s0.sh` (drivers), `warmsweep_results.log`,
`warmsweep_progress.log`, `warmsweep_mem.log`, per-run `WS_*/gen.txt`+`diag.txt`,
`CTRL_rootds4_K23_c32/` (binary control), `WARMPAIR_K23_c32_r{1,2}/` (cold->warm pair).
GPU left free (694 MiB idle); UI:8000 untouched throughout.
