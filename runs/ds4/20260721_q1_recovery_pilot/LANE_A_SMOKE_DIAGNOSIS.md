# Lane-A smoke diagnosis (2026-07-21) — NOT disk-bound; parallelism + overlap

The first lane-A cut (g132/lane-a-smoke, ALL ~70% Q1-fallback -> CPU, ~40/43 layers) measured:
0.09 t/s at ctx1536/1024tok, GPU 2%, CPU 4%, ~16-23 ms/expert. This config is a QUALITY probe,
NOT a speed design (it ADDS work to CPU instead of removing it from GPU — wrong for speed).

Profiler (cpu_lane_profile, CPU-only) corrects the diagnosis — it is NOT disk I/O:
- process_io_read_mb = 0.0 for BOTH cold and hot: zero disk reads. The 1736 faults/expert are
  SOFT page faults (page-cache), not disk.
- Cold/hot ratio: 4.05x (fault/TLB overhead, ~4ms vs ~1ms).
- TINY-BATCH kills parallelism: 8 workers on n=2 experts = 2.97 ms/expert vs 0.85 at full batch.
  The engine does 2-5 experts/layer = tiny batches = under-uses the pool. This is the big lever.
- 8-worker shared-pool speedup 6.31x (parallel capability IS present — just not used at n=2).
- Inline PrefetchVirtualMemory does NOT help (costs 3.2ms, retains 90% of cold time) -> only an
  ASYNC prefetch a token ahead (lane B) would.
- No overlap in the live engine (GPU 2% + CPU 4%): per-layer join barrier serializes.

Root cause = 4x(cold soft-fault) x ~3x(tiny-batch under-parallelism), with no GPU overlap.
Fix levers, in order: (1) resident-only gate + small per-layer cap so CPU stays fast and hidden;
(2) batch experts across layers or ensure the pool fills; (3) real GPU/CPU overlap (not join barrier);
(4) async prefetch ahead (lane B) for warming, never inline. Redesign spec in progress.
