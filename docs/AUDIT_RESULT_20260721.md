# AUDIT RESULT — 2026-07-21 ds4 native-Windows port (adversarial technical review)

Auditor: Fable 5 (max rigor), with 4 read-only Codex passes (gpt-5.6-sol, high effort) over
code worktrees and logs. Everything below was verified against committed artifacts in
`runs/ds4/20260721_q1_recovery_pilot/`, docs on `plan/0051-transport-gate-20260713`, on-disk
logs in `D:\ds4_work\`, and the ds4-win worktrees `wt-f1`/`wt-f2`/`wt-lane-a`/`wt-lane-a2`.
Mandate: `docs/AUDIT_BRIEF_20260721.md`.

## Verdict table

| # | Claim | Verdict |
|---|---|---|
| 1 | F1 = 4.86 t/s (+19%), bit-exact, record | **OVERSTATED** (real but ~94% of the Δ is first-token warm-up; steady-state gain ~2%; bit-exactness asserted, not demonstrated) |
| 2 | F2 inert — cross-token Q1 reuse ~0 | **CONFIRMED outcome / UNPROVEN mechanism** (h2d unchanged is real; "reuse ~0" was unobservable because `DS4_Q1_0_PROFILE` was off) |
| 3 | Q1 unfixable at 1.125 bpw — STE ceiling 0.701 | **OVERSTATED** (0.701 is real but n=1 expert, 256 samples, train 0.902 ⇒ data-limited, not a proven representational ceiling) |
| 4 | Gate-1 adaptive beats static +45% | **CONFIRMED as computed / OVERSTATED as generalization** (weak static baseline, 3 in-sample replays × 64 tok) |
| 5 | Gate-2 DRAM: lanes coexist (A −10-12%) | **CONFIRMED** (as a synthetic CPU-only microbench; live-engine coexistence unmeasured) |
| 6 | CPU exact forward 0.7-0.9 ms/expert, 0.9998 | **CONFIRMED with caveat** (0.7-0.9 is the best bench of four; 1.0-1.3 is representative hot, ~4-5 ms cold; 0.9998 on 17 activations of ONE expert) |
| 7 | Lane-A smoke NOT disk-bound; root cause = gate picked SSD_COLD | **Mechanism CONFIRMED in code / headline numbers UNPROVEN** (0.09 t/s and 16-23 ms/expert are in no surviving log; "0 disk reads" is from a different process) |
| 8 | Lane-A v2: gate correct, slot-race fix in flight, unmeasured | **MEASURED MID-AUDIT — FAILS the acceptance bar** (0.46 t/s vs required ≥4.86, `LANE_A_V2_MEASURED.md` @ d37260d: no overlap, cap not applied; gate + race fix confirmed implemented in `f250e08`; "5x better than 0.09" rests on a lost log; quality STILL unmeasured) |

---

## Claim 1 — F1 = 4.86 t/s full/open (+19% vs 4.09), bit-exact, "record"

**Verdict: OVERSTATED.**

What the raw log actually shows (`runs/ds4/20260721_q1_recovery_pilot/f1f2_attribution.stderr.log`,
byte-identical to `D:\ds4_work\u1_f1f2.server.stderr.log`):

- 4.86 t/s is the server's own 64-token decode average: `gen=64 ... avg=4.86 t/s 13.181s` (log:105).
  It is decode-only (excludes the 31.8 s prefill), ctx=256, 15-token prompt, temp=0, n=1.
- **Token 1 is a 4.118 s outlier** (`token=1 decode_ms=4117.67`, log:40 — VRAM route-resolver
  warm-up). Mean over all 64 tokens = 205.7 ms (matches the doc); mean over tokens 2-64 =
  **143.6 ms/tok ≈ 6.96 t/s steady state** (server's own last chunk: `chunk=6.88 t/s`, log:105).
- The clean baseline (`docs/G130_U1_FIRST_ATTRIBUTION_RESULTS_20260720.md`) quotes steady tokens
  t2=150.6, t32=138.6, t64=149.1 ⇒ steady ≈ 146 ms/tok; its 64-tok total 15.658 s implies a
  **~6.4 s first token**. So of the 2.49 s total improvement, **~2.3 s (~94%) is the first-token
  warm-up difference**; steady-state decode moved ~146 → ~143.6 ms (**~2%, within n=1 noise** —
  the two steady ranges overlap almost completely: 137-153 vs 138-151).
- The F1-targeted spans confirm this: `existing_stream_sync_wait` 11.1→0.0 is real (0.0 on every
  token), but steady-state `selection_d2h` (~69-72 ms) and `h2d_enqueue` (~33 ms) are unchanged.
  The doc's table numbers (mixed_q1_call 95.8→63.2 etc.) are 64-token means dominated by token 1
  (e.g. token-1 mixed_q1_call = 3786 ms; steady is ~4 ms).
- **Comparison hygiene**: baseline measured 2026-07-20, F1+F2 2026-07-21 (cross-day n=1 vs n=1).
  "Record vs G73 4.99 closed / WSL 3.4" is not comparability-checked (measurement windows for
  those numbers not established). And this "full/open record" engine routes **80.2% of expert
  slots through the Q1 approximate path** (log:113: `q1_resident=13241` of `selected=16512`) —
  the very path whose quality is known broken; it is not an exact-engine record.
- **Bit-exactness** (Codex pass over `wt-f1` @ bb0591a, `wt-f2` @ 1fefa7d): the change is
  scheduling-only (host-resident selection IDs + `cudaEventRecord`/`StreamWaitEvent` replacing a
  `cudaStreamSynchronize`; ds4_cuda.cu:31290, 31423, 34519) — bit-exactness is *plausible by
  construction* and asserted in the commit message ("Bit-exact (scheduling only)"), but **no
  output-hash A/B or golden test is committed anywhere**. Claimed as established; it is not.

Honest restatement: F1 removes a real sync, sharply improves the first decode token after
prefill, and the engine's steady state is ~7 t/s at ctx256 short-prompt on the Q1-hybrid path.
"+19% steady throughput" and "record" are not supported.

## Claim 2 — F2 (VRAM LRU) inert, cross-token Q1 reuse ~0

**Verdict: CONFIRMED outcome / UNPROVEN mechanism.**

- Outcome: steady-state `h2d_enqueue` ≈ 31.7 (clean) vs ≈ 33.0 (F1+F2) — no reduction; F2
  delivered nothing measurable on this workload. CONFIRMED (n=1 each, cross-day).
- Mechanism: the day's docs say "`[q1-vram-lru]` telemetry did not appear — cannot distinguish".
  Code audit (wt-f2) explains why: **that telemetry only prints when `DS4_Q1_0_PROFILE` is set**
  (ds4_cuda.cu:26298, 26191), and `run_f1f2.ps1` does not set it. An active, hitting cache would
  have been equally silent. Also, `iq2_vram_cache=q1-routes-bypass` in the log is a hard-coded
  IQ2-cache status string (ds4_cuda.cu:3792) unrelated to F2's Q1 LRU — it proves nothing about
  F2. `DS4_Q1_VRAM_LRU_SLOTS` is parsed lazily (ds4_cuda.cu:26163, default 600) behind several
  preconditions (ds4_cuda.cu:31621ff).
- Therefore "cross-token Q1 reuse ~0" is an *inference*, not a measurement. It is plausible (it
  is consistent with the unchanged h2d span) but a single 512-tok re-run with `DS4_Q1_0_PROFILE=1`
  would settle it and was not done. The handoff's flat "MEASURED inert ... cross-token Q1 reuse ~0"
  upgrades an ambiguous n=1 into a fact.

## Claim 3 — Q1 quality unfixable at 1.125 bpw; STE ceiling 0.701

**Verdict: OVERSTATED as worded; the practical decision it supports is defensible.**

Evidence checked (`ste_pilot_r4_final.json`, `ste_pilot_l15e176_256samples.json`,
`ste_trainer_report.md`):

- Split honesty: OK — deterministic request/session-disjoint 70/15/15 (train 169 / val 38 /
  test 49 samples), test block held out; metrics evaluated after fp16 scale round-trip. Sound.
- The 0.701 test cosine (rank-4 LoRA, 100 epochs, best 92) vs Q1 baseline 0.602 is real.
- **But train cosine is 0.902 and val 0.684**: the model *can* represent much more than 0.70 at
  this bpw on seen data — the gap is generalization from 169 training samples, not representational
  capacity. "Ceiling" is therefore the wrong word: a 10-50× larger trace could move the test
  number materially, and this was not tested.
- **n=1 expert** (layer 15 expert 176, one trace manifest) generalized to all 11,008 experts.
  The 17-sample layer-3 smoke explicitly disclaims any quality conclusion.
- Nit: rank-4 LoRA adds 147,456 B on 3,538,944 B payload ⇒ effective ~1.17 bpw, not 1.125.
- Cross-doc drift: PLAN_G131 says baseline cosine "0.55-0.60", HANDOFF says "0.55"; the JSON test
  baseline is 0.602.

The decision actually taken — don't bet the quality recovery on training the 1.125-bpw sidecar —
is reasonable risk management given 0.70 << the 0.90 bar. But it's a judgment call supported by a
pilot, not a proven impossibility, and nothing between 1.125 and 2.06 bpw was explored (see
Strategy).

## Claim 4 — Adaptive beats static +45% (42.8% vs 29.5% @ P=8), heat 81.7%@16

**Verdict: CONFIRMED as computed / OVERSTATED as a general "adaptive beats static".**

- Numbers check out in `working_set_coherence.md` §4 (pooled P=8: 42.8 vs 29.5; heat table
  pooled mass@16 = 81.7%). Direction is consistent in all 9 replay×P cells (relative gap
  +34%/+56%/+41% per replay at P=8).
- **The static baseline is weak**: same-size top-K ranked on gate mass of the *first 16 tokens
  of the same prompt* (§Method). That is neither the deployable static (G131's global mass-ranked
  K from prior traces) nor an oracle static (full-trace ranking). The +45% quantifies "adaptive
  beats a 16-token snapshot", not "adaptive beats the best static alternative". No oracle-static
  or cross-prompt-static control was run.
- Sample: 3 replays × 64 tokens, steady state scored on tokens 16-63, all in-sample and all
  generated the same day on the same engine. The pooled p10 at P=8 is 22.4% — under prompt
  shift the warm tier misses ~78% of mass at p10.
- Domain-switch: "recovers in 15-27 tok" is recovery to 90% *of the in-domain rate*; the table's
  own "absolute 90%" column reads "not reached". PLAN_G132/HANDOFF phrasing is borderline here.
- Also note the absolute level: at P=8 the adaptive warm tier (median 153 experts) captures
  **42.8%** of gate mass — the strategic question is who serves the other 57% (see Strategy).

## Claim 5 — Gate-2 DRAM: lanes coexist, A degrades 10-12%

**Verdict: CONFIRMED (as a microbench).**

`dram_contention_report.md`: at the target 56-112 MB/token pacing, lane-A degradation is
10.0-12.4% and lane B holds ≥95% pace, matching the claim (full range across all rows is
5.4-12.4%). Caveats that keep this from being a system result: CPU-only MSVC bench, lane A is a
ring of layer-3 experts (best-case locality), no GPU/driver/server load, lane B is synthetic
unbuffered reads + memcpy. Pass criterion is self-defined but stated up front. One more gap:
the MB/token→MB/s conversion uses **1.650 t/s**; at the >6 t/s mandate rate the same
56-224 MB/token becomes 0.34-1.35 GB/s, whose top end exceeds the benched sustainable C: rate
(1.43 GB/s unthrottled single-stream, 1.11 GB/s at 2 workers) — coexistence at mandate speed is
untested. Fine as a pre-build gate; not evidence the live engine coexists.

## Claim 6 — CPU exact forward 0.7-0.9 ms/expert, 0.9998 correctness

**Verdict: CONFIRMED with a material caveat (the quoted range is the best of four benches).**

- 0.9998 overall cosine: confirmed in `cpugemv_spike_report.md` — but on **17 captured
  activations of one expert (l3e0)**; adequate for a math-path check, thin for "correctness".
- The 0.7-0.9 figure comes from the *fixed* overlap harness (`reconcile_report.md`: 0.725 ms
  @8 physical cores / 0.931 @6 cores, N=24, RAM-resident cache-cold) after finding a genuine
  17.5 ms timing artifact (512 MiB eviction scan inside the timed loop — good catch, well
  documented). The other benches: cpugemv 1.267 ms @8t (cold-swept), profiler hot 1.277 ms
  (`cpu_lane_profile_results.json` summaries.hot), DRAM bench 0.988 ms @6t alone / 1.09-1.11
  under contention. So the honest planning band is **~1.0-1.3 ms/expert hot in-engine**, 0.7-0.9
  only with all 8 physical cores dedicated and batch ≥4.
- The HOT-vs-COLD caveat is real and flagged where it matters (`lane_a_redesign.md`: "the 0.83 ms
  standalone result measured a different condition: already-hot pages"; cold/hot = 4.05×,
  cold ≈ 4-5+ ms). However PLAN_G132 and the HANDOFF quote "0.7-0.9 ms/expert" unqualified —
  the caveat is adequately on record but not consistently carried.

## Claim 7 — Lane-A smoke 0.09 t/s NOT disk-bound; root cause = gate admitted SSD_COLD experts

**Verdict: mechanism CONFIRMED in code; headline numbers UNPROVEN by surviving artifacts.**

Code (Codex pass, `wt-lane-a` @ 8f76a05 — all three anchors verified):

- Resolver returns `q1_resident` only when `tier.state == SSD_COLD && !has_2bit_ram`
  (ds4_cuda.cu:34923-34929) — i.e., exact IQ2 bytes NOT in RAM.
- Lane-A gate converts exactly those routes to `iq2_cpu_exact`, cap = compile-time 8, no
  latency/deadline guard (ds4_cuda.cu:35273-35279; CUDA_G132_CPU_LANE_MAX_ROUTES=8 at :773).
- `cuda_g132_cpu_lane_source_ptr` points the CPU directly into the main-model mmap, with the
  damning comment in-source: "the CPU consumes the existing main-model mmap directly"
  (ds4_cuda.cu:34555-34572). Root-cause chain: CONFIRMED.

Measurements, however:

- **No surviving log shows 0.09 t/s.** `smoke_lanea.stderr.log` shows `avg=0.72 t/s` (64 tok,
  ctx256), with `span_mixed_join_total = 62.37 s of 89.28 s` (70% in the CPU-lane join) and
  2560 joins/64 tok ⇒ ~40 CPU-active layers/token (supports that part). The quality run
  (`quality_lanea.stderr.log`, ctx=640 — not 1536) shows 0.46-0.52 t/s avg, degrading
  (token 112: 1753 ms, join 1207 ms), killed at ~112 tok. The "0.09 t/s at ctx1536/1024tok"
  and "16-23 ms/expert" figures evidently come from an earlier run whose log was overwritten.
  Surviving per-layer data gives 24.4 ms/joined-layer ≈ 4-5 ms/expert.
- **"0 disk reads" is a different process**: `process_io_read_mb = 0` comes from the standalone
  CPU-only profiler (`cpu_lane_profile_results.json`), which ran with a mostly-free RAM budget.
  The live engine had committed ~39 GB Q1 arena + 5.5 GB CUDA pinned, leaving ~13 GB for page
  cache against an 86 GB mmap (log: `available_after=13108346880`). A 64-tok union of touched
  experts (~2300 × 7 MB ≈ 16 GB) already exceeds that. No decode-time I/O counter was captured
  for the live run, so "NOT disk-bound" is proven for the profiler, extrapolated for the engine.
  The 4×(soft fault) × 3×(tiny batch) decomposition explains ~10 of the observed 16-23 ms/expert;
  the residual is unaccounted.
- The fix direction (resident-only gate + cap + real overlap) is right regardless — it holds
  under either fault regime.

## Claim 8 — Lane-A v2 (resident gate + cap), slot-lifetime race, fix in flight

**Verdict: the brief's snapshot was stale in both directions — the code state was AHEAD of it,
and mid-audit the parallel session landed the measurement: v2 = 0.46 t/s, which FAILS the
design's own acceptance bar (≥ 4.86 t/s, `lane_a_redesign.md` acceptance §5).**

Measured (landed during this audit, `runs/.../LANE_A_V2_MEASURED.md` @ d37260d; consistent with
my independent read of `D:\ds4_work\quality_lanea.stderr.log`, 17:03-17:09):

- v2 (f250e08, ctx640, stopped early at ~gen 100-112): **0.46-0.52 t/s avg**, degrading
  (token 112: decode 1753 ms, mixed_join 1207 ms). `join_wait_ms == cpu_ms` per layer — the CPU
  lane is **serialized at the per-layer join, not overlapped**; the cap env
  `DS4_G132_CPU_LANE_MAX=3` did not take effect (5-6 routes/layer observed); resident experts
  cost ~2.4-7 ms, not the 0.83 ms spike figure. The doc's own verdict ("BETTER, NOT FIXED; real
  lever is async overlap") is honest and matches the log.
- Caveat on "5x better than v1 (0.09)": the 0.09 log does not survive (see claim 7). Against the
  *surviving* v1 log (0.72 t/s at ctx256/64tok) v2 at 0.46 (ctx640/256tok) is not demonstrably
  better at all — configs differ in both directions. The 5x framing should not be quoted.

Code state (Codex pass over `D:\ds4_work\wt-lane-a2`, branch `g132/lane-a-resident`):

- The redesigned gate is exactly as specified and **already committed locally** at
  `f250e08` ("resident-IQ2-only admission gate + slot reader-reservation", +608/−128, committed
  17:02:58) — the brief says "NOT yet committed / fix in flight". Working tree is clean apart
  from build/review logs.
- Gate condition verified: representation ∈ {iq2_snapshot_ram, iq2_tier_ram} AND `has_2bit_ram`
  AND state ∈ {RAM_PROBATION, RAM_WARM} (ds4_cuda.cu:34795, 34804), plus **pageable==0**
  (:34845). `DS4_G132_CPU_LANE_MAX` exists, default 3, clamped (:34652). CPU pointers come from
  the validated arena slot (`slot.host_ptr`, :34875); `cuda_g132_cpu_lane_source_ptr` has zero
  occurrences — the mmap CPU path is gone.
- **The slot-lifetime race fix is implemented, not merely in flight**: per-slot
  `cpu_lane_slot_refs` (reader count + writer-claim bit, :1096), atomic acquire/release (:1125),
  writer claim requires zero readers (:1163), reservation held through worker
  completion/reduction (:35047), tier victim selection skips reserved slots and claims
  atomically (:24169), sync replacement holds an RAII writer guard (:25521/:25701), SSD-wrap
  commit re-verifies ownership (:25031/:25068). This covers the three writer sites named in the
  brief. Whether the race is *provably* closed needs the planned re-review + a stress run — the
  mechanisms exist; no proof yet.
- nvcc build artifacts postdate the source (`ds4_server.exe` 17:02:36; NVCC success in
  `codex_impl_fix2.log:41990`).
- **Exact-engine QUALITY remains unmeasured** (run stopped before a gradeable long output) —
  still the single biggest open item, now joined by two measured defects: no CPU/GPU overlap
  (join barrier) and the broken cap propagation. Whether the race is *provably* closed still
  needs a stress test; the mechanisms exist and passed static review.

---

## Strategic direction — does the live mask hold up on the measured numbers?

**What the data genuinely supports:**

- Temporal coherence is real: 75.3% of next-token gate mass sits in the last-8-token union;
  81.7% of mass re-routes within 16 tokens. A chase-able working set exists.
- CPU as a *bounded, resident-only, overlapped* tail is viable: ~1 ms/expert hot, join hidden in
  the spike, DRAM coexistence at target pacing.
- Q1 retirement is directionally sound: the fast engine's quality is broken and the 1.125-bpw
  rescue pilot came in at 0.70 vs a 0.90 bar.

**What the data does NOT yet support: the throughput story.** The mandate is >6 t/s exact
quality. Today's measured anchors are: exact full/open = **1.65 t/s** (G123); Q1-hybrid steady =
~7 t/s at ctx256 (80% approximate routes); F1+F2 left the two dominant steady-state costs
untouched (`selection_d2h` ~70 ms/tok — G131 projected 66→~10, unrealized; `h2d_enqueue`
~33 ms/tok). The adaptive warm tier at P=8 covers 42.8% of mass; **the remaining ~57% must be
served by exactly the transport paths that did not move**, or by the CPU lane — whose corrected
version measured **0.46 t/s with zero overlap** (join_wait == cpu_ms) during this audit. The two
end-to-end integrations of the CPU lane so far went 0.72→0.46 t/s against a 4.86 bar, and the
one end-to-end projection (9.0 t/s) missed by ~2× (owned in writing, to their credit). There is
currently **no measured configuration between 1.65 and >6 t/s at exact quality** — the whole gap
is carried by machinery that is either unbuilt (promoter/reaper, lane B, async overlap) or
measured failing (lane A v1/v2). Note also the unresolved architectural contradiction on record:
the cpugemv spike's committed verdict is "NO-GO for hybrid dispatch on this machine" while
PLAN_G132 makes the CPU a "PRIMARY compute lane" at 50-90% — the reconcile report only reopens a
*bounded overlapped tail*, and both integrated attempts so far confirm the overlap does not yet
exist in the engine.

**Single biggest risk:** the plan retires Q1 *before* demonstrating that exact-IQ2 residual
service fits the token budget. If the ~57% non-warm mass cannot be served exact-in-time, the
engine either stalls back toward 1.65 t/s (mandate fails on speed) or serves stale/approximate
experts (quality collapse returns — the exact failure the plan exists to fix). Compounding it:
the concurrency surface (promoter + reaper + SSD-wrap + CPU lanes + GPU routes over shared
slots) already produced one real slot-lifetime race at first review, pre-measurement; the
engineering cost of making this bit-safe is the schedule risk.

**Cheaper paths that should be priced before milestone-2 (promoter/reaper):**

1. **Async CPU/GPU overlap is now the proven critical path for lane A** (the v2 doc reaches the
   same conclusion). Before building it, run the cap-0 control: v2 with `DS4_G132_CPU_LANE_MAX=0`
   must reproduce F1 within noise, and fix the cap propagation — otherwise every lane-A number
   remains confounded. And measure the exact-engine QUALITY on the fastest available exact
   config first: if exact IQ2 does not fix the CSS collapse, the entire tier design is moot.
2. **Attack `selection_d2h` ~70 ms/tok directly.** It is 33% of the steady token on the only
   fast engine and appears in the exact engine too. G131 already sketched 66→~10; that alone is
   worth more than F1+F2 delivered combined, with no new concurrency.
3. **Price a mid-bpw cold-tail** (1.5-1.75 bpw STE, trainer already exists): if the residual
   mass can't be exact-in-time, a better-than-0.60-cosine tail at half the exact bytes may
   rescue quality at far less machinery than a fully adaptive exact tier. Only 1.125 bpw was
   tested, on one expert.
4. **A deployable-static control**: prefix-warmed static mask refreshed every N tokens, measured
   on held-out prompts. The live mask must beat *that* (not a first-16 snapshot) to justify its
   complexity.

## Repo hygiene

1. **HANDOFF_20260721_LIVE_MASK_STATE.md is stale and would misroute a fresh session**: it
   still describes lane-A round-2 as a *fail-open* fix on `wt-lane-a` (`g132/lane-a-smoke`) with
   smoke/quality *to be run* — in reality the smoke ran and failed, the diagnosis/redesign
   landed (f1dbe76, 2c5bb9a), the real tree is `wt-lane-a2` (`g132/lane-a-resident` @ f250e08,
   committed+built), and v2 has now been measured (0.46 t/s, d37260d). A fresh chat following
   the handoff's "NEXT STEPS 1-3" would rebuild and re-smoke the known-broken v1 gate; the
   on-disk harness scripts (`smoke_harness/run_*_lanea.ps1`) still point at the obsolete
   `wt-lane-a`, and the handoff's env block omits `DS4_G132_CPU_LANE_MAX` (whose propagation is
   now a known defect). The audit brief (e1026da) said "not committed" when f250e08 existed.
   The new INDEX doc (d37260d) mitigates but the handoff itself was not amended. **Fix: one
   status doc, updated last, that supersedes; broken paths in it
   (`docs/G130_FLEET_STATUS` without date, placeholder `runs/.../promoter_design.md`) checked.**
2. **Log overwriting destroyed baseline evidence twice**: `run_f1f2.ps1` deletes and reuses the
   same `u1_f1f2.server.stderr.log` path (baseline per-token data for the +19% claim is gone);
   the 0.09 t/s lane-A run's log was likewise lost to the 17:03 quality re-run. This is the
   second occurrence (B/C logs lost to the `$Tag` quoting bug on 07-20). **Fix: run-ID in every
   log filename, logs are append-only artifacts.**
3. **Headline numbers quoted without their measurement window**: "4.86 t/s (+19%) record" is a
   64-token mean including a 4.1 s warm-up token; steady state is ~7 t/s and the F1 steady gain
   is ~2%. Docs should always state window + first-token treatment (the F1F2 doc's own table is
   internally consistent but the derived one-liners in HANDOFF/PLAN_G132/commit messages are not).
4. **"Bit-exact" is used as a fact but no output-hash A/B is committed** for F1/F2 (assertions
   in commit messages only). Cheap fix: one committed golden-token-hash run per branch.
5. **F2 was declared inert without observability**: the telemetry that would have answered the
   question requires `DS4_Q1_0_PROFILE=1`, which the runner never set. One 512-tok re-run with
   profiling on closes claim 2's mechanism.
6. **Plan targets have quietly diverged**: G131 carries "8.03-13.25 t/s with F1+F2" while the
   measured F1+F2 result (4.86, and ~2% steady) is committed two commits later; PLAN_G132 still
   quotes "0.7-0.9 ms/expert" unqualified, "+19% record", and "tail hides under GPU" (both
   integrated lane-A runs show join_wait == cpu_ms — no hiding). Neither plan was amended after
   the measurements. The cpugemv NO-GO verdict coexists with G132's "CPU as PRIMARY compute lane
   50-90%" without a bridging decision record (the reconcile report only reopens a bounded
   overlapped tail). Recommended (from the coherence pass): SUPERSEDED-BY banners on G130/G131,
   and one canonical metrics table separating diagnostic vs quotable / exact vs Q1 / n / tokens /
   quality grade — today three different "current best" numbers circulate (4.86 diagnostic
   Q1-hybrid, 1.65 exact n=3, 0.46 lane-A v2).
7. **The branch moved under the audit**: d37260d (v2 measurement + INDEX) landed on
   `plan/0051-transport-gate-20260713` while this audit was in progress, from a second writer
   session. The one-writer-per-branch rule the project itself states (PLAN_G131 §Standing rules)
   was not observed for the docs branch; audits and measurements racing on the same branch is
   how stale claims (claim 8's "not committed") happen.

## Method note

Codex passes (read-only, gpt-5.6-sol high): (1) wt-f1/wt-f2 F1 mechanism + bit-exactness +
F2 gating; (2) wt-lane-a root-cause anchors + smoke log extraction; (3) wt-lane-a2 state, gate
fields, race-fix presence, build timestamps; (4) plan/handoff coherence sweep. All quantitative
log statistics (steady-state means, token-1 decomposition, route-mix percentages) recomputed
independently by the auditor from the raw logs.
