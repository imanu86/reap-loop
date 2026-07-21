# F1+F2 measured on runtime (2026-07-21, 64 tok full/open, short prompt, diagnostic n=1)

Build: g131/f2-vram-lru (F1+F2 stacked), DS4_Q1_VRAM_LRU_SLOTS=600, attribution on.

| Metric | Clean baseline | F1+F2 | Δ |
|---|---|---|---|
| Decode | 244.7 ms/tok (4.09 t/s) | **205.7 ms/tok (4.86 t/s)** | **+19% t/s** |
| existing_stream_sync_wait | 11.1 | 0.0 | F1 removed the sync ✅ |
| mixed_q1_call | 95.8 | 63.2 | F1 overlap ✅ |
| selection_d2h | 65.9 | 71.4 | up (span re-attribution under new scheduling) |
| h2d_enqueue (F2 target) | 32.5 | 33.5 | **unchanged — F2 delivered ~nothing** ❌ |

## Honest findings
1. **F1 works: +19% t/s, new full/open record 4.86 t/s** (beats WSL 3.4, approaches G73 4.99 closed WITHOUT a mask). Sync elimination confirmed.
2. **F2 (VRAM LRU cache) delivered no h2d reduction on this workload.** The `[q1-vram-lru]` telemetry did not appear — cannot yet distinguish "cache didn't engage" from "cache engaged but ~0 hits". Most likely the known ~0 cross-token Q1 expert reuse in full/open decode defeats caching.
3. **Projection miss owned: gate-3 hybrid spike projected 9.0 t/s with F1+F2; runtime measured 4.86.** The spike's GPU work was "representative" (flagged as needing confirmation) and was optimistic; F1 rendered ~half its span, F2 zero.

## Next (to settle F2 + re-baseline the projection)
- Longer run (512+ tok) with F2 telemetry gate on: confirm cache hit rate; measure cross-token Q1 reuse directly.
- If reuse is ~0: F2 is inert for the current engine (bit-exact, OFF-safe — no harm), and orthogonal to REAP-revisited (which retires Q1). Keep F1, shelve F2.
- Re-derive the F1-only path to >6 t/s from the MEASURED 4.86, not the spike's 9.0.
