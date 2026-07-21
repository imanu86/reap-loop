# PLAN G131 — REAP-Revisited: exact-IQ2 resident base + CPU tail (2026-07-21)

Supersedes the strategic direction of `HANDOFF_CODEX_20260720_G130_FIX_PLAN_AND_TEST_PROTOCOL.md`
(whose test protocol, preflight and abort rules REMAIN in force). Context and evidence:
`docs/G130_U1_FIRST_ATTRIBUTION_RESULTS_20260720.md` (+3 addenda) and `runs/ds4/20260721_q1_recovery_pilot/`.

## Where the evidence stands (all measured 2026-07-20/21)

- Transport SOLVED: 4.09-5.62 t/s full/open clean (attribution 91.4% closed, validator exit 0).
- The 0.2 t/s mystery SOLVED: mixed-trace fprintf (65.7%) + sync promotion (304 ms/tok). Diagnostic-only.
- Quality BROKEN by Q1 fallback (~70% routes at real-activation cosine 0.55-0.60): T3 NEGATIVE both arms.
- Door 1 (train the Q1, Bonsai-style) CLOSED: STE+rank4+100ep ceiling **0.701 test cosine** (naive 0.602).
  Real learning, insufficient capacity at 1.125 bpw. Full-model QAT (pods, $10k+) is the only stronger form.
- CPU-GEMV as PRIMARY compute NO-GO (1.267 vs 0.6 ms/expert; dual-channel DDR4 wall) — but fine as a TAIL.
- Door 2 design GO: `reap_revisited_design.md` — **K=5,600 exact-IQ2 experts resident in RAM (36.9 GiB),
  CPU-GEMV for the non-resident tail, Q1 retired.** Quality exact by construction.
  Projection: 5.05-6.72 t/s current plumbing; **8.03-13.25 t/s with F1+F2**. The mandate is inside this range.

## Execution order

**GATE 1 — coverage curve on real replay** (in progress): 64-tok cyberpunk decode with mixed-trace on,
log to a DEDICATED file (the $Tag single-quote bug that clobbered logs is fixed in run_replay.ps1).
Compute cumulative gate-mass coverage vs K (mass-ranked, per-layer floor variants) at K=3000/4000/5000/5600.
PASS: fallback ≤ ~15% by count at K=5,600, no pathological high-mass misses. Firm up with 2-3 more replays
(different prompts) before the prototype.

**GATE 2 — one-layer CPU/GPU overlap spike**: prove the CPU tail (1.267 ms/expert, 8 threads) hides under
GPU compute without starving CUDA submission. Reuse cpugemv_spike harness + one real layer.

**GATE 3 — K=5,600 bootstrap**: arena loads reliably with ≥2 GiB measured RAM margin under the real server.
**SSD note (user-confirmed): the PC freezes because the 86+39 GB load saturates C: (the OS disk) — ALWAYS
start ds4_server at PriorityClass Idle during load (I/O priority follows), restore Normal after ready.**

**GATE 4 — numeric tolerance**: forced-tail end-to-end output vs all-GPU (CPU/GPU divergence measured 0.9998).

**PROTOTYPE** (only after gates): implementation seams enumerated in reap_revisited_design.md —
mass-ranked resident load list at bootstrap; dispatch branch cold→CPU queue + join; retire Q1 tiers.
Then **F1** (host-ids entry + event fencing: selection_d2h 66→~10 ms/tok) and **F2** (keyed VRAM LRU ~600 slots:
h2d 33→~15) — sketches in the G130 plan §5/P2, findings table Appendix A.

**MANDATE CAMPAIGN** (unchanged bar): >6.0 t/s mean, n≥3, quality ≥L2 rendered-graded, full/open,
per the G130 ladder T0→T4. Only after the prototype passes T3 quality.

## Standing rules (unchanged)

One writer per branch; implementer ≠ reviewer (codex cycles); diagnostics never quotable; negatives recorded;
one change per experiment; commit+push everything immediately (sessions can die).
Delegation: all work to codex agents (gpt-5.5 default, 5.6-sol for hardest), chunked ≤150k tokens each.
