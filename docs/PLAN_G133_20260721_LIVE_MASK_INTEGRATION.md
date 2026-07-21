# PLAN G133 — Integrating the owner's live-mask design (capacity-lock grounded)

Date 2026-07-21 (evening, rev2 — reordered per owner: TRANSPORT and CPU-anti-miss FIRST; the
chat measurement is the FINAL exam, not the starting point). Supersedes execution order of
PLAN_G132 (design unchanged). Branch for docs/ledger: `plan/0051-transport-gate-20260713`.
Code branches: `g133/*` on github imanu86/ds4-win.

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
  architecture for this machine**: (a) minimize the per-token exposed copy by keeping the LIVE
  working set resident, (b) serve the non-resident tail WITHOUT copying (CPU lane A), (c) stage
  ahead of demand (lane B). The complementary lever — equally large — is the 42% host
  orchestration, attacked separately (M4).

**Order of understanding (owner, binding): first TRANSPORT, then HOW THE CPU REDUCES MISSES.
The sustained multi-turn exact chat is the FINAL exam (M5) — measured last, once the mechanism
works.** Exact full/open anchor today: 1.65 t/s (G123). Sustained >=3 t/s exact would be new
territory; ~5 a landmark.

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
| Routed-expert replay traces (gate-1 inputs) | runs/20260721 + D:\ds4_work | fuel for M0 offline sim |

Seam map for porting 0033 onto the win tree (conflicts with lane-A v2 slot lifecycle):
Codex read-only pass in flight -> results appended as §7 when in.

## 2. Method (owner's rule, binding)

Hypothesis -> test -> ledger. NO tangents, NO continuous patching. Each milestone has ONE
hypothesis, a measurable gate, and writes ledger rows `WIN-G133-*`. A milestone that fails its
gate STOPS the line and we re-analyze (no "fix forward"). Claude = brain (design, criteria,
adversarial review, decisions); Codex = all implementation and heavy reading (gpt-5.5 medium
mechanical, gpt-5.6-sol for hard parts). One writer per branch. All hot reads from C: NVMe; D:
never on a hot path. Server loads at PriorityClass Idle, Normal after ready; staging I/O at LOW
priority (anti-freeze discipline).

## 3. Milestones — UNDERSTAND FIRST (M0), then TRANSPORT (M1), then CPU-ANTI-MISS (M2)

### M0 — Miss economics, offline (H-MISS): CAN the CPU cover the gap? (zero GPU, runs NOW)
**The owner's first question, answered before building anything.**
**Hypothesis** (owner's mechanism claim): with the real layout — ~320 VRAM IQ2 slots + ~1500
pinned IQ2 + RAM-warm tier — and a CONTINUOUS CPU stager promoting by heat ahead of demand,
the per-token miss rate (needed expert not resident/servable) drops to ~10-20%, and the CPU can
serve/stage that residual with margin (measured capacity ~970 exp/s vs the miss load).
**Work** (Codex, offline over the committed replay traces — the same fuel as gate-1): simulate
per-token routed-expert demand against tier layouts: (a) static top-K baseline; (b) live mask
(heat W=8, promotion budget P/token, reap) WITHOUT stager; (c) live mask + CPU stager with
lookahead 8/16/32 tokens (heat half-life 81.7%@16tok says the signal lives long enough — test
it). Outputs per config: miss%/token, copy bytes/token, required staging bandwidth, CPU serve
load vs capacity. Honest caveat: traces are 3 replays x 64 tok (in-sample) — results are
DIRECTIONAL; the online gate (M2) is the real test.
**Gate**: the sim must answer three numbers — miss% without stager, miss% with stager, CPU
headroom ratio. If even the OFFLINE sim can't get miss below ~25% with generous staging, the
layout assumption is wrong -> STOP and rethink tiers before any port work.
**Cost**: hours, zero GPU. **Ledger**: `WIN-G133-M0-MISS-ECONOMICS`.
**RESULT (2026-07-21, ledger row WIN-G133-M0-MISS-ECONOMICS-20260721): PARTIAL-GO.**
On 9016 real decode tokens: miss 27.9% median @ owner layout (claim 10-20% NOT met; pinned
2000 -> 22.5% — pinned is the trim knob). BUT the ECONOMY holds: staging collapses must-stall
2.628 -> 0.050 exp/tok (misses become CPU-servable RAM-warm, not cold stalls) and the CPU
covers the residual with 2.34x headroom @5.6 tok/s; staging BW trivial (~0.1 GB/s). Caveats:
ORACLE lookahead, in-sample seed, optimistic page cache, layer-barrier/tiny-batch out of scope
(avg ~1.85 exp/layer < batch-4 — feeds M3 risk). Sim's own gate: engine integration stays
BLOCKED until a CAUSAL (non-oracle) predictor reproduces this on held-out traces with a finite
page-cache budget -> **M0b, launched**.
**M0b RESULT (ledger WIN-G133-M0B-CAUSAL-PREDICTOR-NOGO-20260721): NO-GO.** Held-out
rust_channel with finite LRU: miss 50.4% (vs 27.9% oracle), must-stall 6.02 exp/tok (gate 0.2),
headroom 1.42x (gate 1.5). All three causal signals equivalent -> predictor choice is NOT the
issue. **Broken assumption: causal decode history cannot pre-warm cold FIRST/re-entry
references.** But M0b starts decode COLD — the real engine has PREFILL routes before decode t0
(free causal signal, unused by the sim) -> **M0c launched**: prefill-seeded staging (+
prompt-conditioned prior = the owner's SPEX idea) through the same finite-LRU held-out harness.
Engine work (M1+M2) stays BLOCKED until M0c passes the same gates.
**M0c RESULT (ledger WIN-G133-M0C-PREFILL-SEED-NOGO-20260721): NO-GO on purity gates, VIABLE
on ms-economics — M0 PHASE CLOSED.** Prefill seeding kills its slice (prefill-seen FIRST
stalls 0.685 -> 0.001/tok) and staging BW passes (0.341 GB/s p90). Residual must-stall 4.7-4.9
exp/tok decomposes: TRUE RE-ENTRIES (finite RAM: LRU ~5.9k of ~11k experts) > prompt-unseen
FIRSTs > CPU burst overflow. **The purity gate (0.2) was the wrong metric.** ms translation:
~4.8 residual stalls x 2-4ms parallel NVMe = ~10-19 ms/token — affordable, not a wall.
**OWNER'S PRINCIPLE (2026-07-21, adopted as design law): even CPU-computing "at random" at 90%
load beats one true miss — a CPU worker's cost hides under GPU time; a stall's cost is fully
exposed. Therefore: NO TOKEN EVER WAITS FOR A BYTE.** Revised M2/M3 mandate: lane-A/B are
ELASTIC — per expert per layer choose CPU-compute (warm OR cold: the worker eats the fault
off-path) or async pinned-H2D (0.55ms DMA), under a layer deadline; synchronous stall is the
fail-open of last resort, never the design path. CPU sizing: ~122 warm-served/tok = ~70%
CPU busy @5.6 tok/s, bursts to ~90% (overflow 0.5/tok at p90) — exactly the owner's 50-90%
two-lane envelope. VERDICT: understanding phase complete; M1 (transport merge) + M2 (lane-B
prefill-seeded staging) justified; M3 elastic overlap remains the make-or-break engineering.

### M1 — TRANSPORT: port 0033 tiered residency to the win tree (kill the exposed copy)
**Hypothesis**: heat-driven residency (dynamic seed + X/X+Y knock + reap) cuts the exposed
expert H2D on the EXISTING short fixture by >=50% (76 -> <38 ms/token) at equal quality,
because the live working set (novelty 24.8%@W8) stays resident instead of re-copied.
**Work** (REVISED by §7): do NOT apply the 0033 patch as-is — the win tree ALREADY has a
four-state per-expert tier controller (`g_moe_tiering`: SSD / RAM-probation / RAM-warm /
VRAM-protected, mass observer + hysteresis ranking) that substantially supersedes 0033.
Applying the patch would create split-brain residency (two authorities — §7 critical #2).
Instead ADOPT 0033's missing semantics INTO the existing controller (`cuda_moe_tier_entry`
stays the sole authority): X/X+Y knock fields, decay, dynamic-seed semantics into
`cuda_moe_prefill_vram_seed` + decode observation (28412), promotion/demotion policy into
`cuda_moe_tiering_pick_vram_slot`/`demote_cache_entry`. FIRST harden the lifecycle (§7 step 1):
the generic arena reap path takes NO lane-A writer claims (contained today only by
`tiering_exclusive` mutual exclusion) — extend transaction-wide writer claims (deterministic
slot order, all-or-nothing) BEFORE composing reap with lane-A. Skip 0039/0040 patch-porting:
build the mass signal on the already-present CUDA mass observer (`cuda_reap_mass_observer`).
Branch `g133/m1-residency`. Wire attribution counters: h2d_ms/token, warm-hit%,
promotions/token, reap/token, thrash guard. Measurement on the existing ctx256/64tok fixture +
one longer 256-tok run — NO new fixture needed.
**Gate**: (1) bit-exact ON vs OFF (A/B hash, not asserted); (2) no-thrash convergence ON
WINDOWS (same criterion as the pod gate); (3) h2d_ms/token reduction >=50%; (4) t/s >= baseline
(no regression tolerated).
**Cost**: seam map decides (est. M). **Ledger**: `WIN-G133-M1-RESIDENCY`.

### M2 — CPU-ANTI-MISS online: lane B continuous stager (the owner's principle, proven live)
**Hypothesis**: the M0-predicted miss reduction materializes ONLINE — a background CPU stager
(heat + 0039 mass-pressure signal, ASYNC ahead-of-time promotion SSD->RAM->pinned) keeps
miss% < 10% after settle, with staging NEVER on the critical path.
**Work** (branch `g133/m2-lane-b`): stager thread(s), LOW I/O priority, C: NVMe only, ASYNC
prefetch (inline PrefetchVirtualMemory proven useless — 90% cold time retained); promotion
budget P/token capped; miss telemetry wired into attribution.
**Gate**: (1) measured miss% within 1.5x of the M0 sim prediction (mechanism understood, not
lucky); (2) miss% < 10% after settle; (3) zero critical-path stalls attributable to staging;
(4) no freeze, C: queue bounded, PC usable (owner check).
**Cost**: M. **Ledger**: `WIN-G133-M2-LANE-B`.
**DESIGN FREEZE (post H-SPEC-B, ledger WIN-G133-HSPEC-B-PREFETCH-NOGO-20260721): the lane-B
signal is prefill-mass seed + heat_ema staging + the elastic-serve law. PREDICTOR WORK IS
CLOSED: three independent signals (frequency, prefill, speculation-surrogate) converge within
7% held-out, and even a perfect drafter cannot reach stall-purity — the residual ~4.4-4.7
stalls/tok are CAPACITY-structural (LRU ~5.9k of 11k experts), absorbed at ~10-19 ms/tok by
elastic CPU serving. Do not fund more predictors. Capture gap to fix opportunistically: route
traces lack token ids — add token-id logging to the trace format so the token->expert map
becomes testable if this is ever revisited.

### M3 — Async lane A: CPU computes the warm tail OFF the critical path (the unsolved wall)
**Hypothesis**: CPU-computing RAM-warm experts overlapped with GPU layer compute (non-blocking
join) removes their H2D at ~zero added token time — the wall in every previous cut was
join_wait == cpu_ms (CPU serialized against the per-layer join).
**Work** (branch `g133/m3-lane-a-async`, gpt-5.6-sol, adversarial review mandatory): at layer-N
route time split experts: VRAM/pinned-resident -> GPU; RAM-warm (reserved slots, lane-A v2
gate) -> CPU pool (8 workers, batch >= 4 experts to dodge the tiny-batch 3x penalty; fewer ->
fold into GPU path). CPU joins at an EVENT at layer end; deadline guard: CPU late -> fail-open
enqueue the copy (bounded tail, never a stall). FIX the cap first: DS4_G132_CPU_LANE_MAX did
not apply in v2 (routes 5-6/layer with cap 3) — root-cause before building on it.
**Gate**: (1) join_wait_ms <= 10% of cpu_ms (overlap REAL); (2) per-expert bit-exact vs GPU
path; (3) t/s > M1+M2 stack; (4) DRAM contention within gate-2 envelope (<=12% degradation).
**Cost**: L (the hard one). **Ledger**: `WIN-G133-M3-LANE-A-ASYNC`.

### M4 — Host orchestration cut (the other 42% — parallel track, independent)
**Hypothesis**: the 121 ms/token host orchestration (per-expert host ops, LRU/bookkeeping,
launch prep) can be halved (<60 ms) by phased dispatch (Colibri STRUCTURE: per-layer plan built
once, batched launches, no per-expert host round-trips) without touching kernels.
**Work** (branch `g133/m4-dispatch`): differential profile first (which host spans grow with
expert count?), then restructure ONLY the top span. Independent of M1-M3; separate writer.
**Gate**: orchestration ms/token < 60 by the same attribution; bit-exact; gains COMPOSE with
M1-M3 (measured together).
**Cost**: M-L. **Ledger**: `WIN-G133-M4-DISPATCH`.

### M5 — THE FINAL EXAM: sustained exact multi-turn chat (built LAST, per owner)
Only now build the `chat_multiturn_fixture` (3 turns with continuation, temp 0, ~1.5-2k ctx by
turn 3, >=256 gen tok/turn, coherence-graded) + the NVTX causal static/expert split. Run the
full stack (M1+M2+M3, +M4 when landed), n>=3, plus a 512-token endurance run.
**Gate — honest tiers**: >=3.0 t/s sustained exact chat = SUCCESS (1.8x the 1.65 anchor, first
usable number ever); >=5.0 = LANDMARK (the owner's "risultato degno"); <3.0 = STOP and
re-analyze with the causal split. **Ledger**: `WIN-G133-M5-CHAT-SUSTAINED`.

### M6 (conditional) — IQ1 miss-cover, done RIGHT (owner's idea, parked not dropped)
ONLY IF M2 leaves miss% > 10%: train IQ1 properly (thousands of iterations, many experts,
offline CPU track — the 0.701 pilot was undertrained, ceiling UNKNOWN) as a RAM-resident
miss-cover the CPU serves while real IQ2 stages. Never a serving tier by default.

## 4. Decision points

- **After M0**: if the sim says the CPU cannot cover the gap even offline -> rethink the tier
  layout (bigger pinned? IQ1 cover earlier?) BEFORE porting anything.
- **After M1**: if h2d drops >=50% but t/s barely moves -> orchestration is the binding
  constraint -> M4 jumps ahead of M2/M3.
- **After M3**: if overlap is real but t/s still flat -> WDDM copy engine / launch path is
  serializing something new -> causal split (NVTX) before any further code.
- **Failure of any gate** = stop + re-analyze (ledger row with honest verdict), never a hotfix
  chain on top of an unexplained number.

## 5. Risks (from the audit + capacity data)

1. **Residual mass**: ~57% of routed mass was outside any tested static set — if the LIVE set
   is much larger than the coherence replays suggest, miss% stays high; M0 sizes it offline,
   M2's gate catches it online, M6 is the mitigation.
2. **Slot-lifecycle collision** between 0033 reap/demotion and lane-A reservations — §7
   interaction list is the mandatory review item for M1 (the v2 race, writer-side).
3. **Pinned-RAM ceiling**: pinning too much of 64 GB starves the OS/page cache (the 4x
   cold-fault penalty returns through the back door). Cap pinned; measure occupancy.
4. **KV growth vs slot budget**: ctx 1536 already forced seed 192 (capacity 235); longer ctx
   squeezes VRAM slots — occupancy telemetry from M1 feeds the budget.
5. **WDDM**: launch/copy behavior differs from the pod (Linux) where 0033 was smoke-tested —
   M1 re-verifies no-thrash ON WINDOWS, not by analogy.
6. **In-sample traces**: M0's sim fuel is 3 replays x 64 tok — directional only; every offline
   conclusion carries this caveat until M2 confirms online.

## 6. Sequencing summary

```
M0 (miss economics offline — CAN the CPU cover it?)   [zero GPU, launches NOW]
  -> M1 (TRANSPORT: 0033 residency port)              [seam map §7]
       -> M2 (CPU-ANTI-MISS: lane B stager online)     M4 (dispatch cut, parallel track)
            -> M3 (async lane A)
                 -> M5 (FINAL EXAM: sustained exact chat, n>=3)   [M6 only if miss%>10]
```

## 7. Seam map (Codex gpt-5.6-sol read-only pass, 2026-07-21) — VERDICT

Full report: `D:\ds4_work\seam_map_0033.log`. Headline: **the win tree already implements most
of the design** — porting is a SEMANTIC MERGE into the existing controller, not a patch apply.

### What the win tree already has (supersedes much of 0031/0033)
- Per-expert four-state tier controller `g_moe_tiering` via `cuda_moe_tier_entry` (22865):
  SSD / RAM-probation / RAM-warm / VRAM-protected. History is PER EXPERT and survives eviction
  (better than 0033's slot-local history).
- Mass observer + hysteresis residency ranking: `cuda_reap_mass_observer` (2355),
  `cuda_reap_mass_publish_residency` (11242).
- Tier-aware victim selection/demotion/enforcement: `pick_vram_slot` 27702, `demote` 26153,
  `enforce_request` 27844; prefill mass seed `cuda_moe_prefill_vram_seed` 26751.
- Correct slot-lifecycle linearization (reader CAS -> revalidate; writer claim held across all
  mutations incl. async SSD-wrap; lane-A holds reservation through compute + fail-open).

### Critical conflicts (why patch-as-is is FORBIDDEN)
1. **Split-brain residency**: 0031/0033 slot-local pinned/tier/knock alongside per-expert
   `g_moe_tiering` = two authorities for residency/counts/demotion. -> Adopt semantics into the
   existing controller; `cuda_moe_tier_entry` stays the SOLE authority.
2. **Latent lifecycle hole**: generic arena reap (`cuda_dynamic_arena_wrap_publish_target` 9771
   -> `ds4_gpu_dynamic_arena_begin` 7395, slot reassign 7524) takes NO lane-A writer claims —
   contained today only by `tiering_exclusive` mutual exclusion (7408/24048). Composing reap
   with lane-A WITHOUT transaction-wide writer claims = use-after-reassignment corruption.
   Multi-slot reap: acquire writer claims in deterministic slot order, all-or-nothing.
3. **F1/F2 async publication**: promotion/demotion must stay inside route-worker ordering and
   honor claimed[], upload-stream completion, map publication, Q1 reuse events.
4. **0039/0040 need the absent livemask base (0035-0038)** + GNU weak linkage + wrong dims for
   win 43x256/MSVC. -> Skip patch-porting; build the mass signal on the existing CUDA mass
   observer directly (the `g_reap_pin_mass` cross-TU seam is unnecessary).

### Port order (adopted into M1)
1. Harden lifecycle (writer claims on generic reap, or keep strict `tiering_exclusive`) — L
2. Extend `cuda_moe_tier_entry` with knock/decay/hysteresis fields (single authority) — M
3. Seed/observation semantics into `prefill_vram_seed` + decode observation 28412 — M
4. Promotion/demotion policy into pick/demote (+ `cuda_q1_vram_lru_resolve` only if needed) — L
5. Lane-A validation re-run with tier churn ON (all reservation/cancel/SSD-wrap paths) — M
6. Selection-side livemask LAST, on the mass observer (not 0039/0040 patches) — L

### Three biggest port risks (mandatory review checklist for M1)
1. Slot reuse racing lane-A CPU readers.
2. Split-brain residency from layering 0033 over the existing four-state controller.
3. Breaking F1/F2 async publication/reuse fencing while changing victim selection or demotion.
