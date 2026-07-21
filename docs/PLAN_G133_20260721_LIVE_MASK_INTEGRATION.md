# PLAN G133 — Integrating the owner's live-mask design (capacity-lock grounded)

Date 2026-07-21 (evening). Supersedes execution order of PLAN_G132 (design unchanged — this plan
sequences its INTEGRATION on the measured evidence of b105e6a). Branch for docs/ledger:
`plan/0051-transport-gate-20260713`. Code branches: `g133/*` on github imanu86/ds4-win.

## 0. Why this plan exists (the evidence that reframes everything)

Ledger rows `WIN-ANALYSIS-H-STATICO-REFUTED-20260721` + `WIN-ANALYSIS-CAPACITY-LOCK-20260721`:

- **No static ceiling, no compute ceiling.** Compute is 50 ms/token (17%) — the 3060 uses ~1-2%
  of peak. The 25.82 t/s transient (CLAIM-008) shows the real GPU potential.
- **The 287 ms token is: 42% CPU host orchestration (121 ms, GPU idle) + 27% exposed serial
  expert H2D copy (76 ms) + 17% compute (50 ms) + 9% launch (26 ms) + small-copy 17 ms.**
- **Residency is the decisive lever**: fit-in-VRAM configs measure 12-16 t/s on >12GB GPUs
  (3090 c1024 = 14-16, 3080Ti coding-fit = 12). The 3060 is CAPACITY-LOCKED: 12 GB cannot hold
  the working set, so it pays the serial PCIe copy every token. A 2.6x faster engine = +13%;
  residency = ~4x.
- **Therefore the owner's live mask (tiers + heat promotion/reap + CPU lanes) is the correct
  architecture for this machine**: it is the only way to (a) minimize the per-token exposed copy
  by keeping the LIVE working set resident, and (b) serve the non-resident tail WITHOUT copying
  (CPU lane A) while (c) staging ahead of demand (lane B). The complementary lever — equally
  large — is the 42% host orchestration, attacked separately (M4).

**Goal metric (owner, verbatim intent): sustained t/s on a REAL multi-turn chat with
continuation, at full exact quality, n>=3.** Not peak t/s on a short fixture. We have NEVER
measured this. The exact full/open anchor is 1.65 t/s (G123). Anything sustained >=3 t/s exact
on real chat is new territory; ~5 would be a landmark result.

## 1. Assets already in hand (nothing starts from zero)

| Asset | Where | State |
|---|---|---|
| 0033 tiered-hysteresis residency (VRAM/RAM/SSD tiers, X/X+Y knock promotion, dynamic seed, re-entry, reap) | `patches/0033-pace-tiered-hysteresis-residency.patch` @ 8723e29 + `docs/TIERED_RESIDENCY.md` | AUTHORED, bit-exact, smoke-tested on pod (671fec2: no-thrash convergence + engagement) |
| 0039 mass-pressure (pressure = demand mass, wshare = freq x weight) | livemask branches (see seam map) | designed/measured shift behavior |
| 0040 pin-by-mass producer (`g_reap_pin_mass` seam; K8-by-mass renders) | livemask branches | authored |
| Lane-A v2: RAM-resident-IQ2 admission gate + slot reader-reservation (CAS) | `g132/lane-a-resident` @ f250e08 | built, reviewed APPROVE, measured (0.46 — no overlap yet) |
| F1 host-ids + event fence | `g131/f1-selection-d2h` @ bb0591a | built, measured |
| Attribution profiler (host spans + budget closure) | in win tree (`DS4_G132_U1_ATTRIBUTION`) | working |
| SPEX predictor | prior campaign | candidate for lane B |
| CPU exact IQ2 forward 0.7-0.9 ms/expert HOT, 6.31x 8-worker scaling | measured (cpugemv spike) | kernel exists |

Seam map for porting 0033 onto the win tree (conflicts with lane-A v2 slot lifecycle):
Codex read-only pass in flight -> results appended as §7 when in.

## 2. Method (owner's rule, binding)

Hypothesis -> test -> ledger. NO tangents, NO continuous patching. Each milestone has ONE
hypothesis, a measurable gate, and writes ledger rows `WIN-G133-*`. A milestone that fails its
gate STOPS the line and we re-analyze (no "fix forward"). Claude = brain (design, criteria,
adversarial review, decisions); Codex = all implementation and heavy reading (gpt-5.5 medium
mechanical, gpt-5.6-sol for M2/M4 hard parts). One writer per branch. All hot reads from C:
NVMe; D: never on a hot path. Server loads at PriorityClass Idle, Normal after ready; staging
I/O at LOW priority (anti-freeze discipline).

## 3. Milestones

### M0 — The chat fixture + causal baseline (no engine changes; cheapest first)
**Hypothesis**: none — this creates the measuring stick everything else is judged by.
**Work**: (a) define `chat_multiturn_fixture`: 3 turns with continuation (answer -> follow-up ->
follow-up), deterministic (temp 0), ~1.5-2k ctx by turn 3, >=256 gen tokens/turn; graded for
coherence (the CSS-collapse check). (b) One run, full/open exact, attribution ON + the NVTX/
CUDA-event split from the H-STATICO verdict: separate ranges for non-MoE static kernels vs
routed-MoE transport/compute on their actual streams -> causal `static_ms/token` vs
`expert_ms/token`. (c) Record the baseline: sustained t/s per turn, miss/copy counts, VRAM/
pinned occupancy, the full span budget.
**Gate**: fixture is reproducible (n=3 within 5%); span budget closes (sum ~= token time).
**Cost**: hours, 1 GPU run. **Ledger**: `WIN-G133-M0-CHAT-BASELINE`.

### M1 — Port 0033 tiered residency to the win tree (the design's core lands)
**Hypothesis**: heat-driven residency (dynamic seed + X/X+Y knock + reap) cuts the exposed
expert H2D on the chat fixture by >=50% (76 -> <38 ms/token) at equal quality, because the live
working set (novelty 24.8%@W8, half-life 81.7%@16tok) stays resident instead of re-copied.
**Work**: apply/rebase 0033 per the §7 seam map onto `g132/lane-a-resident` base -> branch
`g133/m1-residency`. Preserve the lane-A slot reader-reservation discipline: tier demotion/reap
MUST skip reserved slots (the §7 interaction list is the review checklist). Wire attribution
counters: h2d_ms/token, warm-hit%, promotions/token, reap/token, thrash guard.
**Gate**: (1) bit-exact ON vs OFF (A/B hash, not asserted); (2) no-thrash convergence on win
(same criterion as the pod gate); (3) h2d_ms/token reduction >=50% on the chat fixture;
(4) sustained t/s >= M0 baseline (no regression tolerated).
**Cost**: the seam map decides (est. M). **Ledger**: `WIN-G133-M1-RESIDENCY`.

### M2 — TRUE async lane A (the unsolved wall — hardest, highest value)
**Hypothesis**: CPU-computing the RAM-warm tail OFF the critical path (overlapped with GPU
layer compute, non-blocking join) removes those experts' H2D at ~zero added token time — the
wall in every previous cut was join_wait == cpu_ms (CPU serialized against the per-layer join).
**Work** (branch `g133/m2-lane-a-async`, gpt-5.6-sol, adversarial review mandatory):
- At layer-N route time, split routed experts: VRAM/pinned-resident -> GPU; RAM-warm (reserved
  slots, lane-A v2 gate) -> CPU worker pool (8 workers, batch >= 4 experts to dodge the
  tiny-batch 3x penalty; if fewer, fold into GPU path — do NOT under-fill workers).
- CPU output joins at an EVENT at layer end; GPU proceeds with its own experts meanwhile.
  Deadline guard: if CPU misses the layer deadline, fail-open = enqueue the H2D copy (bounded
  tail latency, never a stall).
- Keep DS4_G132_CPU_LANE_MAX enforcement FIXED (v2 bug: cap did not apply — routes were
  5-6/layer with cap 3; root-cause before building on it).
**Gate**: (1) join_wait_ms <= 10% of cpu_ms (overlap REAL, measured); (2) per-expert output
bit-exact vs GPU path; (3) sustained t/s > M1 (each CPU-served expert saves ~0.6-1.2 ms copy);
(4) DRAM contention within gate-2 envelope (lane A degrades <=12%).
**Cost**: L (the hard one). **Ledger**: `WIN-G133-M2-LANE-A-ASYNC`.

### M3 — Lane B anti-miss staging (the owner's continuous-CPU principle)
**Hypothesis**: with residency (M1) + heat/SPEX-driven ASYNC staging SSD->RAM->pinned ahead of
demand, the miss rate (expert needed but neither resident nor CPU-servable) on the chat fixture
falls below 10% — the ledger already shows near-zero miss at cache 258 and hit 0.74 at cap1024.
**Work** (branch `g133/m3-lane-b`): background stager thread(s), LOW I/O priority, C: NVMe only,
ASYNC ahead-of-time prefetch (inline PrefetchVirtualMemory is proven useless — 90% cold time
retained); demand signal = heat window + 0039 mass-pressure (wshare); promotion budget P/token
capped; never touches the critical path (pure producer).
**Gate**: (1) miss% < 10% sustained on turn 2-3 of the chat fixture (after domain settle);
(2) zero critical-path stalls attributable to staging (attribution span); (3) no freeze, C:
queue depth bounded, PC usable (owner check).
**Cost**: M. **Ledger**: `WIN-G133-M3-LANE-B`.

### M4 — Host orchestration cut (the other 42% — parallel track, independent)
**Hypothesis**: the 121 ms/token host orchestration (per-expert host ops, LRU/bookkeeping,
launch prep) can be halved (<60 ms) by phased dispatch (Colibri STRUCTURE: build the per-layer
plan once, launch batched, no per-expert host round-trips) without touching kernels.
**Work** (branch `g133/m4-dispatch`): differential profile first (which host spans grow with
expert count?), then restructure ONLY the top span. This track is independent of M1-M3 and can
run in parallel on a separate branch/writer.
**Gate**: orchestration ms/token < 60 measured by the same attribution; bit-exact; t/s gain
composes with M1/M2 (measured together on the chat fixture).
**Cost**: M-L. **Ledger**: `WIN-G133-M4-DISPATCH`.

### M5 — The result: sustained exact chat (the number that has never existed)
**Work**: full stack (M1+M2+M3, +M4 when landed) on the chat fixture, n>=3, plus a 512-token
single-prompt endurance run. Grade output coherence (exact quality). Record VRAM/pinned/RAM
occupancy and the final span budget.
**Gate — honest tiers**: >=3.0 t/s sustained exact chat = SUCCESS (1.8x the 1.65 anchor, first
usable number ever); >=5.0 = LANDMARK (the owner's "risultato degno"); <3.0 = STOP and
re-analyze with the M0 causal split (is the residual copy, orchestration, or something new?).
**Ledger**: `WIN-G133-M5-CHAT-SUSTAINED` (+ per-turn rows).

### M6 (conditional) — IQ1 miss-cover, done RIGHT (owner's idea, parked not dropped)
ONLY IF M3 leaves miss% > 10%: train IQ1 properly (thousands of iterations, many experts,
offline CPU track — the 0.701 pilot was undertrained, ceiling UNKNOWN) as a RAM-resident
miss-cover the CPU serves while real IQ2 stages. Never a serving tier by default.

## 4. Decision points

- **After M1**: if h2d drops >=50% but t/s barely moves -> orchestration is the binding
  constraint -> M4 jumps ahead of M2.
- **After M2**: if overlap is real but t/s still flat -> the WDDM copy engine or launch path is
  serializing something new -> back to M0's causal split before any further code.
- **Failure of any gate** = stop + re-analyze (ledger row with honest verdict), never a hotfix
  chain on top of an unexplained number.

## 5. Risks (from the audit + capacity data)

1. **Residual mass**: ~57% of routed mass was outside any tested static set — if the LIVE set
   on real chat is much larger than the coherence replays suggest, miss% stays high; M3's gate
   catches it, M6 is the mitigation.
2. **Slot-lifecycle collision** between 0033 reap/demotion and lane-A reservations — the §7
   interaction list is the mandatory review item for M1 (a repeat of the v2 race, but on the
   writer side).
3. **Pinned-RAM ceiling**: pinning too much of 64 GB DDR4 starves the OS/page cache (the
   4x cold-fault penalty returns through the back door). Cap pinned; measure occupancy at M0/M5.
4. **KV growth vs slot budget on real chat**: ctx 1536 already forced seed 192 (capacity 235);
   turn-3 ctx will squeeze VRAM slots further — the M0 occupancy numbers feed the M1 budget.
5. **WDDM**: launch/copy behavior differs from the pod (Linux) where 0033 was smoke-tested —
   the port gate re-verifies no-thrash ON WINDOWS, not by analogy.

## 6. Sequencing summary

```
M0 (fixture+causal baseline)
  -> M1 (0033 residency port)        [seam map §7]
       -> M2 (async lane A)           M4 (dispatch cut, parallel track)
            -> M3 (lane B staging)
                 -> M5 (sustained exact chat, n>=3)   [M6 only if miss%>10]
```

## 7. Seam map (Codex read-only pass) — APPENDED WHEN IN

(placeholder — `D:\ds4_work\seam_map_0033.log`; summarized here + committed on arrival)
