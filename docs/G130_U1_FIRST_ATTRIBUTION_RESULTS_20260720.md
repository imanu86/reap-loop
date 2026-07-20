# G130 U1 — First Attribution Run Results (2026-07-20 21:32)

**DIAGNOSTIC n=1 — not quotable as a performance claim.** Purpose: P0-U1 of the G130 plan — attribute the decode cost of the G129-class full/open config with the zero-sync profiler. Both goals achieved, plus an unexpected strategic finding.

## Setup

- Build: integration branch `g130/integration-u1` (base 08a4d27 + u1-neutrality + u1-attribution-v2 + harness/test branches), Ninja Release sm_86, short-path checkout `C:\Users\imanu\g130i`.
- Config: G129 frozen full/open resident set extracted from `g130_u1_common.ps1:70-120` (Q1_0 dual arena 24.5 GiB pinned + pageable overflow, IQ2 dynamic arena 5.5 GiB, cache320 + seed320/floor4, split-fused, mass-lfru, router **open**) with **promotion OFF** and **all tracing OFF**; `DS4_G130_U1_ATTRIBUTION=1`. Model `ds4-2bit.gguf` + sidecar `ds4-q1-layers0-42-derived.gguf` (receipts verified by size).
- Machine preflight: GPU 842 MiB/0%/13 W idle, 50.8 GB RAM free, no other ds4 process.
- Canonical U1 prompt (12 tok), 64 tokens, temp 0, think=false, ctx 256, streamed.

## Results

| Metric | Value |
|---|---|
| Server ready (load+arena) | 97.3 s |
| Request wall | 46.8 s (prefill ≈ 31 s + decode 15.7 s) |
| **Decode** | **15.658 s / 64 tok = 244.7 ms/token = 4.09 t/s (full/open)** |
| Attribution closure | **91.4% attributed; residual 1.346 s = 8.6% (≈21 ms/token)** |
| Wall closure | wall 15.741 s = decode 15.658 + loop_overhead 0.082 ✓ |
| Contract validation | `g130_attrib_validator.py` on the 65 live lines: **exit 0** |
| Q1 resident arena | 13,241 hits / 0 misses; H2D 46.86 GB total (incl. prefill); selected_loads 2,560; failures 0 |
| Output | coherent start ("Prof…"); quality grading NOT performed (n=1 diagnostic) |

Per-token span breakdown (request totals / 64):

| Span | ms/token | share of decode |
|---|---|---|
| mixed_q1_call (generic envelope) | 95.8 | 39.2% |
| **selection_d2h** | **65.9** | **26.9%** |
| h2d_enqueue | 32.5 | 13.3% |
| residual (unattributed) | 21.0 | 8.6% |
| existing_stream_sync_wait | 11.1 | 4.5% |
| hot_route | 10.7 | 4.4% |
| kernel_launch_enqueue | 4.2 | 1.7% |
| route_classify | 2.9 | 1.2% |
| selected_load | 0.4 | 0.15% |
| mixed_join | 0.2 | 0.08% |
| promotion_staging | 0.0 | (off) |

Tokens are remarkably stable (e.g. t2 150.6 ms, t32 138.6 ms, t64 149.1 ms; residual ~20 ms flat).

## Findings

1. **The profiler works as designed**: 91.4% of decode attributed with per-token and request-level closure, live-validated against the contract. The instrument the plan demanded exists and measures without distorting (zero added CUDA syncs, proven at review).
2. **STRATEGIC: the "unattributed 4.5 s/token" of the G129 safety receipt did not reproduce.** Same runtime base, same full/open resident config, but with **promotion OFF + mixed-trace OFF**, decode is **244.7 ms/token (4.09 t/s)** vs the receipt's 4,878 ms/token (0.205 t/s) — a ~20× gap attributable to the safety run's promotion+trace-enabled configuration (and its environment), NOT to the Q1 transport core. The prior review's estimate that trace+promotion explain only ~150 ms/token is inconsistent with this observation and needs a dedicated A/B (promotion on/off × trace on/off) to decompose.
3. **4.09 t/s full/open (diagnostic, short-prompt) exceeds the WSL 3.4 baseline and approaches the closed-mask G73 record (4.99) without any mask.** Caveats before celebrating: 12-token prompt, ctx 256, n=1, no quality grading, promotion off. The mandate protocol (long prompt, ≥L2 quality, n≥3) remains to be run.
4. **Next levers are now visible and ranked**: selection_d2h 66 ms/token (the F1 blocking D2H — known fix sketch exists), the generic mixed_q1_call envelope 96 ms/token (needs one finer span pass to decompose), h2d_enqueue 33 ms/token (F2 Q1 VRAM reuse / F4 pinning).

## Next steps (per plan ladder)

- A/B: promotion on/off × trace on/off at this exact config (4 × n=1 diagnostics) to decompose the 20× gap.
- T3 quality probe: 512–2000 tokens, rendered L-grading, stop on L0/L1.
- Span refinement inside mixed_q1_call; then F1 fix (host-ids entry + event fencing) targeting the 66 ms selection_d2h.
- Only after L2+: the n≥3 campaign per the mandate.
