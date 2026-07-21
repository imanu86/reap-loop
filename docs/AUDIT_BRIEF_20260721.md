# AUDIT BRIEF — hand this to a fresh Fable 5 chat for a full technical audit

You are auditing a day of work on the ds4 native-Windows MoE inference port (DeepSeek-v4, 86GB IQ2,
11,008 routed experts, RTX 3060 12GB + 64GB DDR4). **Your job: verify the claims below against the
committed evidence and the code — find errors, overstatements, and unproven conclusions. Be adversarial.**

## Rules for the audit (owner's instructions)

- **Delegate the heavy lifting to Codex/ChatGPT** (cheap): use `codex exec --sandbox read-only
  --skip-git-repo-check -m gpt-5.6-sol -c model_reasoning_effort=high "<task>" < /dev/null`.
  You (Fable 5, max rigor) are the brain; Codex reads code/logs and reports. Keep the fleet modest
  (a handful of Codex passes, not dozens).
- Everything is committed to reap-loop branch `plan/0051-transport-gate-20260713`. Working code/logs
  are on disk under `D:\ds4_work\` and `C:\Users\imanu\g130i\`. Model at `C:\ds4-models\`.
- Write your audit to `docs/AUDIT_RESULT_20260721.md`, commit + push. Verdict per claim:
  CONFIRMED / OVERSTATED / WRONG / UNPROVEN, with the evidence you checked.

## Claims to verify (each has a committed artifact)

1. **F1 = 4.86 t/s full/open (+19% vs 4.09 clean), bit-exact.** Evidence:
   `runs/ds4/20260721_q1_recovery_pilot/F1F2_MEASURED.md` + `f1f2_attribution.stderr.log`. Branch
   `g132/lane-a-resident` base has F1 (github imanu86/ds4-win, `g131/f1-selection-d2h` @ bb0591a).
   Check: is 4.86 the DECODE-only number, short-prompt ctx256/64tok? Is "record" fair vs G73 4.99
   (closed) and G123 1.65 (open)? Is bit-exactness actually established or just claimed?
2. **F2 (VRAM LRU) inert — cross-token Q1 reuse ~0.** Same doc. Check the h2d span unchanged claim.
3. **Q1 quality unfixable at 1.125bpw — STE ceiling 0.701 test cosine.** Evidence:
   `runs/.../ste_pilot_r4_final.json`, `ste_pilot_l15e176_256samples.json`. Check the train/val/TEST
   split honesty (256 samples, 1 expert l15e176), whether 0.701 generalizes (n=1 expert!), and
   whether "unfixable" is warranted from ONE expert on 256 samples.
4. **Gate-1 working-set coherence: adaptive beats static +45% (42.8 vs 29.5% at P=8), heat 81.7%@16.**
   Evidence: `runs/.../working_set_coherence.md/json`. Check: only 3 replays x 64 tokens — is the
   +45% robust or small-sample? Is the static baseline a fair comparison?
5. **Gate-2 DRAM: lanes coexist (A degrades 10-12%).** `runs/.../dram_contention_report.md`.
6. **CPU exact forward 0.7-0.9 ms/expert, 0.9998 correctness.** `runs/.../cpugemv_spike_report.md` +
   `cpu_lane_profile_results.json`. Check the HOT-vs-COLD caveat (0.83ms is HOT cached; cold=4.05x).
7. **Lane-A smoke: 0.09 t/s NOT disk-bound — root cause admission gate picked SSD_COLD experts.**
   Evidence: `runs/.../LANE_A_SMOKE_DIAGNOSIS.md`, `lane_a_redesign.md`, `cpu_lane_profile_results.json`.
   Check: does the profiler actually show 0 disk reads? Is the "picked q1_resident=SSD_COLD" root
   cause correct against the code (ds4_cuda.cu:34923-34929, 35273-35279, 34555-34612)?
8. **Lane-A v2 (resident-IQ2 gate + cap):** branch `g132/lane-a-resident` (off `g132/lane-a-smoke`),
   NOT yet committed (on disk `D:\ds4_work\wt-lane-a2`). Admission gate REDESIGNED per lane_a_redesign.md:
   admits ONLY RAM-resident exact-IQ2 experts (has_2bit_ram + RAM_WARM/PROBATION + iq2_snapshot_ram/
   tier_ram), excludes SSD_COLD/q1_resident, reads from the validated arena slot NOT the mmap, cap
   DS4_G132_CPU_LANE_MAX=3/layer. nvcc-compiled. Adversarial review: gate CONFIRMED correct (the 20ms
   cold-fault bug IS fixed at check time), but REJECTED on a real SLOT-LIFETIME RACE — the CPU workers
   deref saved raw slot pointers while other threads (GPU route worker ~28378, tier replacement ~24092/
   ~25554, SSD-wrap commit ~24947) can evict/reassign the slot mid-read -> corrupted output. Round-2 fix
   IN PROGRESS: a reader-reservation/pin holding the slot from admission through CPU compute, with all
   arena writers skipping reserved slots. **STATUS AT HANDOFF: gate correct, slot-race fix in flight —
   NOT yet built/measured. The corrected speed (must be >= 4.86 t/s with CPU actually saving H2D) and
   the EXACT-ENGINE QUALITY are both still UNMEASURED.** This + item's quality are the two biggest open
   items. Resume: finish the reservation fix, re-review (slot race must be provably closed), build,
   run a SHORT (256-tok) quality+speed A/B with the resident gate. Machinery/rules: HANDOFF_20260721.

## Strategic direction to stress-test

The plan (G132, `docs/PLAN_G132_...`): a LIVE MASK / reap-loop — expert tiers SSD->RAM->pinned->VRAM
with heat-driven promotion + reap, CPU active 50-90% in two lanes (A=compute resident-warm experts
exactly, B=anti-miss staging), SPEX as lane-B predictor, all exact IQ2 (Q1 retired). **Question for
the audit: do the measured numbers actually support this direction, or is there a simpler/cheaper path?
What is the single biggest risk that could make it fail?** Also read G130/G131 plans + G130_FLEET_STATUS
+ HANDOFF_20260721_LIVE_MASK_STATE for coherence — do the three plans contradict each other anywhere?

## Known honest caveats already on record (confirm they are adequately flagged, not buried)

- Gate-3 spike projected 9.0 t/s from F1+F2; real F1 measured 4.86 (spike GPU work was "representative").
- Coverage 2.58% fallback @ K=5600 is IN-SAMPLE on 3 prompts (owner objected; K=5600 static was dropped
  in favor of the live mask).
- Quality of the EXACT engine (does lane A fix the CSS collapse) is NOT yet measured — the lane-A smoke
  was too slow (wrong config) to produce a gradeable long output. This is the biggest OPEN item.
