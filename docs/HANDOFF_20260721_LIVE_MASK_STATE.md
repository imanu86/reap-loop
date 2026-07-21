# HANDOFF — Live-mask state, resumable (2026-07-21 late)

Read this + `PLAN_G132_20260721_REAP_LOOP_LIVE_MASK.md` to resume with zero context. Everything
below is either committed (repo = state) or on-disk in `D:\ds4_work\` (survives chat compaction).

## Where we are: building the lane-A smoke (exact-quality full/open)

Design (owner's reap-loop, see PLAN_G132): expert mask adapts token-by-token; tier ladder
SSD->RAM->pinned->VRAM with heat-driven promotion + reap; CPU active 50-90% in two lanes
(A=compute warm experts exactly from mmap, B=anti-miss staging at low I/O prio); SPEX = lane-B
predictor. ALL hot reads from C: NVMe (model at C:\ds4-models). Quality EXACT by construction.

Both pre-build gates PASSED (committed in runs/ds4/20260721_q1_recovery_pilot/):
- Working-set coherence: adaptive promoter beats static +45% (42.8 vs 29.5% warm-hit at P=8),
  heat half-life 81.7%@16tok, domain switch recovers in ~27 tok. THE LOOP CHASES.
- DRAM contention: lanes coexist (A degrades 10-12%, B sustains pace), all reads from C:.

## Committed branches (github imanu86/ds4-win)

- `g131/f1-selection-d2h` @ bb0591a — F1: kill per-layer serialization (host-ids entry + event
  fence). MEASURED +19% -> 4.86 t/s full/open (record). Bit-exact. 2-round reviewed, built.
- `g131/f2-vram-lru` @ 1fefa7d — F2: keyed VRAM LRU for Q1 experts (F1 included). MEASURED inert
  on current engine (cross-token Q1 reuse ~0); bit-exact, OFF-safe. Keep for the VRAM tier later.
Other g130/* branches (m4/neutrality/epoch/attrib-tests/watchdog/m5/m1m3/utf8/ssdwrap-*/g73-*/
spec-restore) all landed earlier — see docs/G130_FLEET_STATUS.

## IN FLIGHT (NOT yet committed — on disk D:\ds4_work\wt-lane-a, branch g132/lane-a-smoke off F2)

**Lane A integration** (ds4_cuda.cu +875/-34): env DS4_G132_CPU_LANE=1 makes the ~70% Q1-fallback
routes compute the EXACT IQ2 expert on CPU from the mmap (queue cap 8, overflow->Q1), per-layer
join via F1's event pattern. Reuses cpugemv_spike forward (0.9998), tracer input capture, add_f32.
- Round-1 adversarial review: normal path CONFIRMED correct (CPU IQ2 math IQ2_XXS/Q2_K matches
  verified reference incl. 0.125 factor; input identity; join-exactly-once; split; concurrency;
  OFF-path byte-identical). REJECT on ONE defect: post-dispatch failures (worker/H2D/event) don't
  fail-open to Q1 and don't latch failed_open.
- Round-2 fix IN PROGRESS (codex 5.6-sol): uniform post-dispatch failure -> recompute CPU routes
  via Q1 + set failed_open + reset stop=0 on re-init. Then: re-review -> Claude builds -> commit.

## NEXT STEPS (exact, resumable)

1. Lane-A fix2 lands -> re-review (D:\ds4_work\wt-lane-a) -> `cmd /c D:\ds4_work\wt-f1\build_f1.cmd`
   pattern to build (copy build script, point at wt-lane-a) -> git add ds4_cuda.cu, commit
   `g132/lane-a-smoke`, push. (git safe.directory='*' env trick; push from the worktree may need
   `git -C ... push`; worktree push sometimes errors '$GIT_DIR too big' -> push from main clone.)
2. SMOKE (GPU run): `D:\ds4_work\smoke_harness\run_smoke_lanea.ps1` (ready, parse-clean). Assert
   exit 0, cpu_lane routes>0, output non-empty. Idle-priority-during-load (freeze mitigation).
3. QUALITY (GPU run): `run_quality_lanea.ps1` — 1024-tok cyberpunk HTML, A/B (lane on vs Q1
   baseline), renders to D:\ds4_work\lanea_output.html. GRADE VISUALLY (render the HTML), not by
   flags. If lane-A HTML >= L2 where Q1 baseline was L0 -> QUALITY PROBLEM SOLVED, commit result.
4. Only after smoke passes: milestone-2 = promoter/reaper (spec committed:
   runs/.../promoter_design.md — W=8 heat, P=8 promotion to VRAM, 3-way dispatch; critical path is
   safe exact-slot movement, reuse F1+SSD-WRAP machinery). Then lane B + SPEX.

## Measured ceilings (don't re-derive)

Q1 quality unfixable at 1.125bpw (STE ceiling 0.701, real-activation baseline 0.55). CPU-GEMV as
PRIMARY no-go (bandwidth) but as a TAIL/lane fine (0.7-0.9 ms/expert, hides under GPU). F1=4.86 t/s
real. The path to >6 is the live mask (exact quality + heat-driven tiering + CPU lanes), not micro-fixes.

## Machine/tooling notes

RTX 3060 12GB, 64GB DDR4 dual-channel, model 86GB IQ2 on C: NVMe. FREEZE cause = C: saturation on
load -> ALWAYS start ds4_server PriorityClass Idle during load, Normal after ready. codex exec needs
`< /dev/null` headless + `--skip-git-repo-check` on non-git dirs; valid models gpt-5.6-sol/gpt-5.5
(NOT gpt-5.5-codex). Windows clones need `git clone -c core.longpaths=true`. Delegation: codex does
the work, independent codex reviews, Claude verifies+builds+commits. One writer per branch.
