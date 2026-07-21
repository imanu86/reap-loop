# Lane-A v2 (resident-IQ2 gate + reservation) — measured 2026-07-21 (run stopped early)

Build g132/lane-a-resident @ f250e08, DS4_G132_CPU_LANE=1, DS4_G132_CPU_LANE_MAX=3, ctx640/256tok.
Run stopped early by owner (~gen 50-100) after the trend was clear.

| Metric | v1 (all-Q1->CPU) | **v2 (resident-gate)** | target |
|---|---|---|---|
| decode t/s | 0.09 | **0.46** (5x better) | 4.86 |
| CPU / GPU util | 4% / 2% | **18% / 30%** (both active) | — |
| ms/expert | 16-23 | **~7** | 0.83 (resident-hot spike) |
| CPU routes/layer | ~6 | **5-6** | cap should be 3 |
| join_wait vs cpu_ms | equal | **still equal (~37=37)** | should be ~0 (hidden) |

## Verdict: BETTER, NOT FIXED. Two unsolved axes remain.

1. **Cap not applied**: DS4_G132_CPU_LANE_MAX=3 did not take effect — routes/layer still 5-6. Env not
   propagated or cap logic wrong. The CPU is doing more work than intended.
2. **NO OVERLAP (the real wall)**: join_wait_ms == cpu_ms (~37ms). The CPU compute is SERIALIZED
   against the per-layer join, NOT hidden under GPU work. The resident-gate fixed the cold-mmap axis
   (7ms vs 20ms) but the engine still waits for the CPU at each layer barrier. Even 3 resident experts
   at 1ms would still be serial time added, not saved. Both units at 18/30% (not saturated) = they
   take turns, don't overlap.
3. **Resident experts still ~7ms** (8x the 0.83ms spike): resident IQ2 bytes are in RAM but cold to
   CPU L2/L3 (soft-fault first-touch per token) and/or small-batch under-parallelism.

## What the redesign fixed vs what it didn't
- FIXED: admission gate (resident-only, no cold mmap deref), slot reader-reservation (no mid-read
  eviction race — reviewed APPROVE), both units now doing work.
- NOT FIXED: true CPU/GPU overlap (compute layer-N warm on CPU WHILE GPU does layer-N hot, non-blocking
  join), the per-layer cap, resident-hot cache locality.

## Next (the real lever)
The next work is NOT another gate tweak — it is async overlap: restructure so the CPU lane computes
concurrently with the GPU hot path and the join does not block on the CPU per layer. This is the
architectural piece the hybrid gate-3 spike assumed but the engine integration does not yet deliver.
Quality of the exact engine STILL unmeasured (run stopped before a gradeable long output).
