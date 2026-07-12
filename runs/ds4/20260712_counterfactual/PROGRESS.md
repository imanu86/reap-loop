# Counterfactual admission (0046) — progress log

Branch: `spex-predictive-mask-study-2026-07-12`. Working tree (WSL):
`/root/ds4-cf-work` (copy of `/root/ds4-fullstack` = 0044+0045 already applied
+ 2 pre-existing local edits, see "baseline notes" below).

## Timeline (2026-07-12, times local)

- ~17:45 implementation started. Fix per handoff: when adaptive-K grows,
  admit the excluded experts with the highest recent counterfactual weight
  (rolling mean ~3 tokens) instead of top-mass10; mass10 stays for incumbent
  protection and expulsions. Env: `DS4_PACE_ADAPTIVE_CF_ADMIT=1` (+
  `DS4_PACE_ADAPTIVE_CF_ADMIT_WINDOW`, default 3, max 8). OFF = 0045
  identical.
- Soft-mask: `DS4_REAP_MASK_SOFT_BIAS` (default off = hard -1e9). Finite
  negative value (es. -2.0) replaces the veto at all 3
  `g_reap_bias_masked` build sites. Invalid values (NaN/inf/>=0) rejected
  with fallback to hard veto.
- 18:08 first clean build (`make cuda CUDA_ARCH=sm_86`, zero warnings).
- Adversarial review (separate agent) BEFORE any GPU run. Verdict: 2 real
  issues, both fixed:
  1. GHOST CHURN: an expelled expert kept its frozen rolling-mean sum from
     before its admission stint, making it a false top re-admission
     candidate. Fix: `ds4_cf_admit_clear()` at both live resident->excluded
     transitions (adapt_k shrink victim + ordinary scan swap-out victim).
  2. Unvalidated `atof` on DS4_REAP_MASK_SOFT_BIAS (nan/inf/positive typo
     would poison logits or reward exclusion). Fix: isfinite + <0 guard.
- 18:19 rebuild clean. Patch regenerated, verified to re-apply byte-identical
  on a fresh baseline copy. Committed:
  - `cc5bf5f` 0046 patch (`patches/ds4/0046-counterfactual-admission.patch`)
  - `7655e55` harness (runner + tripwire monitor + stream guard live-text)
- 18:21 armA_run1 started (adaptive-K + CF admit ON, SPEX off, port 8073,
  flock GPU serialized).
- 18:24:39 armA_run1 KILLED at gen=17 by a concurrent agent's
  `pkill -x ds4-server` (prefill-test agent; orchestrator confirmed it was
  their coordination error). Output at kill time: 64 chars of clean prose.
  => INVALID (environmental contamination, NOT a verdict, NOT a fail-fast
  strike). See `armA_run1/INVALID.txt`. Useful mechanism facts salvaged:
  both 0046 diagnostic lines printed; livemask seeded at tok=16 K=16.
- NEW KILL DISCIPLINE (binding, from orchestrator): never `pkill`; only
  `kill $(cat <run_dir>/server.pid)`.
- Machine now exclusive (other 2 agents finished). Fail-fast count restarted.
- 18:3x armA_run1b started (true run 1 of arm A).

## Protocol (from task + user rules)

- Prompt: unchanged historical cyberpunk request
  (`runs/ds4/20260709_k23_unit_vs_weighted_cache256_html800/html_local_k23_weighted_warmup_cache256_r01/request_measured.json`),
  max_tokens 4000, ctx 6144, temp 0, streaming, stop on `</html>` /
  objective repetition / budget.
- FAIL-FAST: run 1 degenerates => arm stops (no run 2/3). Run 1 passes =>
  complete n=3.
- Arm A: 0045 adaptive-K (K16-50, thr 0.15, update2, +4/-1) + CF admit ON,
  SPEX off. Arm B: K23 fixed + soft-bias -2.0 (informational if A fails).
- Tripwires (auto-kill by PID + log): union admitted >50% pool per layer;
  admit/token churn that never decays; t/s < 1.5 sustained >50 tok.
- Grading: `scripts/functional_grade.py` frontpage L0-L3, honest; micro-smoke
  may only reject, never promote.

## Baseline notes

`/root/ds4-fullstack` (and thus `/root/ds4-cf-work`) contains 2 local edits
not in the 0044/0045 patch files: the livemask window clamp in
`ds4_pace_init` is [3..32] instead of the patch-file clamp — documented as
acceptable per task brief. `git status` in the tree shows `ds4.c` and
`ds4_cuda.cu` modified vs its internal git baseline (that diff = 0044+0045+
local clamp edits; my 0046 delta is measured against the fullstack tree
as-found, see patch header).

## Run table (updated live)

| run | esito | motivo stop | grader | K avg/p90 | admit/tok | union max | t/s |
|---|---|---|---|---|---|---|---|
| armA_run1 | INVALID (killed by concurrent agent) | external pkill at gen=17 | n/a | n/a | n/a | n/a | n/a |
| armA_run1b | IN CORSO | — | — | — | — | — | — |
