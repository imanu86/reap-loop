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


---

# ADDENDUM 2026-07-21: A/B matrix promotion x trace — mystery SOLVED

| Run | promotion | mixed-trace | request wall | decode ms/token | t/s |
|---|---|---|---|---|---|
| A | off | off | 46.8 s | 244.7 (measured) | 4.09 |
| B | ON | off | 67.5 s | ~569 (est. from wall) | ~1.76 |
| C | off | ON | 328.7 s | ~4,650 (est.) | ~0.21 |
| D | ON | ON | 361.8 s | **5,180 (measured)** | 0.193 |

D reproduces the G129 safety receipt (4,878 ms/token) and its attribution closes at **98.5%** (residual 1.5%):
route_classify **3,401 ms/token (65.7%)** = the mixed-trace fprintf rows (~344/token through the PowerShell
redirect pipe at ~10 ms/row); selection_d2h inflated to 988 ms/token (vs 66 clean); promotion_staging
**304 ms/token** (the known F5 sync preads — matches review); residual 80 ms/token.

**Conclusions:** (1) the "unattributed 4.5 s/token" was ~87% telemetry artifact + ~6% synchronous promotion;
the Q1 transport core does 4.09 t/s full/open (diagnostic, short prompt). (2) The safety-vs-perf run
distinction in the G130 plan (T2 diagnostics never quotable) is vindicated — trace-on numbers are a
different physical regime. (3) F5 (promotion off the decode thread / SSD-WRAP) is confirmed worth ~324 ms/token.
(4) B/C stderr logs were lost to a driver filename bug (single-quoted $Tag); their decode values are wall-derived.
Next per plan: T3 quality probe (512+ tokens, rendered grading), then F1 (selection_d2h 66 ms clean-path), then n>=3.

---

# ADDENDUM 2: T3 quality probes (2026-07-21 morning)

| Probe | Config | Tokens | Decode | Quality (visual, raw text inspection) | Q1 route share |
|---|---|---|---|---|---|
| T3-clean | promo OFF | 213 (self-stop) | 177.9 ms/tok = **5.62 t/s** | **L0/L1 NEGATIVE** — CSS repetition loop from ~150 tok ("font-weight: base, 400" repeated, invalid comma-joined properties) | 71.6% (39,886/55,617) |
| T3-promo | promo ON (frozen gates) | 213 (self-stop) | 388 ms/tok = 2.58 t/s | **L0/L1 NEGATIVE** — different loop (endless invented pseudo-elements *::dialog, *::input...) | 67.4% (37,486/55,626) |

**Findings:**
1. Long-run decode is even faster than the 64-tok diagnostic: 5.62 t/s clean full/open (cache warms).
2. **Quality breaks in BOTH arms** at ~150-200 tokens of long-form HTML — the Q1 fallback (~70% of routes at per-expert cosine 0.811) cannot sustain L2. The plan's "quality is unmechanized" diagnosis is now EMPIRICAL.
3. **F6 empirically confirmed**: the frozen promotion budget (64/request) moved Q1 share by only 4.2 points at 2.2x the cost. Promotion as parameterized cannot fix quality.
4. Strategic consequence: the mandate (>6 t/s AND >=L2) cannot be met by tuning THIS design. The evidence now points at the P3 architectural exit — **CPU-GEMV for cold experts serving EXACT IQ2 from host RAM** — which attacks BOTH failures at once: quality (exact weights everywhere, Q1 retired) and transport (~432x less PCIe traffic per cold route). Alternative paths (activation-aware Q1 recovery; replay-guided IQ2 split) remain but are slower to validate.

Runs: diagnostic n=1, short prompt, ctx 4096, temp 0. NEGATIVE results recorded per protocol (T3 stop rule).
