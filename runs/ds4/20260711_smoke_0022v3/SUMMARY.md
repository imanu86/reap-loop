# 0022 v3 S1-guided rewind — mechanism smoke (FIRE test)

**Date:** 2026-07-11 · **Pod:** ds4-podD-0022-smoke (RunPod 7qgalm9sasqnr7, RTX 3090, left RUNNING)
**Verdict: PASS — the v3 "time machine" fires mechanically. All three success criteria (a/b/c) met.**

## What was validated
Patch patches/ds4/0022-pace-s1-rewind.patch (content IS v3 — Subject confirmed
"pace: 0022 v3 … deep pre-erosion checkpoint ring + char-garbage detector + resume
mini-warmup"; header target ds4.c md5 d4ff85af).

## Build (gate 0 — PASS)
- cd /root/ds4 && git reset --hard a8a38a8 -> ds4.c md5 62ed2e71 (clean canonical+0027+0028 base). OK
- git apply /root/0022-v3.patch (clean; git apply --check passed first) -> ds4.c md5 d4ff85af = exact v3 target from patch header. OK
- rm -f ds4 ds4-server ... && make -B ... CUDA_ARCH=sm_86 -> exit 0, no "ld: Is a directory"; ds4 = fresh ELF. OK (build_0022v3.log)

## What was run (FIRE test — after coordinator course-correction)
Initial direct-CLI smoke on the coherent coffee prompt at DS4_PACE_REWIND_GARBAGE=0.3 did NOT fire —
CORRECT: valid HTML has ~0.2 non-structural-char fraction, below threshold; nothing to rewind
(direct_cli_smoke_GARBAGE030.log). Lowering GARBAGE on healthy text = false positives, not a real
test. So the FIRE test reproduces the pivotal-v2 arm2 collapse regime on the v3 binary:

- Harness: canonical run_w_sweep_freeze_safe.py via run_pivotal_arm.py wrapper (PACE/rewind env
  injected phase-2-only; phase-1 stays clean to build the routing trace).
- Cyberpunk prompt (199 B), W50 two-phase weighted freeze-safe -> mask K12/layer STATIC
  (sess.txt 62428 B = 9760 pruned pairs = (256-12)x40, byte-identical to the v2 pivotal mask).
- ctx_p1 2048 / ctx_p2 8192, cache 1024, greedy temp 0, --runs 1 --total 800.
- Phase-2 env = pivotal arm2 base (DS4_PACE=1 S1=1 KEEP=KEEP_MIN=KEEP_MAX=12 ROTATE=0 RELEARN=0
  BREATH_EVERY=999999 DRIFT=2.0 WEIGHTED_SELECTED=1 REWIND=1 SPEX_TRACE_TOKENS=...) PLUS the three
  v3 levers: DS4_PACE_REWIND_GARBAGE=0.80, DS4_PACE_REWIND_CKPT_DEPTH=8, DS4_PACE_REWIND_WARMUP=40.
  Full env in v3_smoke_p2env.json; exact command in run_v3smoke.sh.

Result: rc=0, generation 20.44 t/s, no crash / no NaN / no hang. (Output still ultimately
re-collapses — expected for K12-static; see caveat.)

## Success criteria — all PASS (evidence: v3_smoke_W050_r00/pace_events.jsonl + tokens.csv)

(a) char_garbage detector fires DURING erosion, PRE-lock — YES.
  {"ev":"rewind_arm","from":209,"to":85,"reason":"char_garbage",...} then
  {"ev":"rewind","from":209,"to":118,"reason":"char_garbage","regen":92}.
  The NEW detector armed AND fired at pos 209 — inside the CSS-erosion ramp (erosion ~char 401 ~=
  pos 190-215, sfoko lock ~= pos 215), i.e. pre-lock and ~51 tokens earlier than v2's S1 (v2: armed
  249 / fired 260). The char_garbage reason string is definitive that the new char-level EWMA path
  — not the old S1 slope — pulled the trigger.

(b) rewind lands DEEP (pre-erosion), thanks to CKPT_DEPTH=8 — YES.
  The char_garbage FIRE landed at to=118 — a 91-token backward jump, well below the erosion onset
  (~pos 190) and far below the lock (~pos 215). The arm's ideal target was to=85 (phase-2 start =
  deepest ring slot). Contrast v2, which jumped only ~14 tokens back (260->246) — inside-lock to
  inside-lock. The 8-deep ring gives a genuinely pre-erosion anchor.

(c) regenerated span is NOT byte-identical (mini-warmup broke the v2 no-op) — YES.
  tokens.csv shows the 209->118 pos-jump then re-emits 138 of 148 positions DIFFERENT from the
  original (same=10, diff=138). The regen produced structurally better HTML — e.g. a real
  <title>AI Cyber Shop</title> and a fresh <style type="text/css"> block — where the v2 temp-0
  retrace was byte-identical "sfoko sfoko". DS4_PACE_REWIND_WARMUP=40 broke the greedy+pinned-K
  no-op exactly as designed.

  (A second, backup fire followed after the trajectory re-eroded: s1_cusum_fire 195->140, also
  divergent — the older S1 path still works as a second line behind the new garbage detector.)

## Which v2 bug each result closes
1. ring depth 8 -> deep pre-erosion target (to=118, ideal 85) — fixes "checkpoint frozen inside the lock".
2. char_garbage EWMA -> fires at the erosion (pos 209), before S1 could — fixes "detector only saw the lock".
3. warmup 40 -> 138/148 regen tokens diverge — fixes "resume is a greedy no-op".

## Honest caveat (scope of this smoke)
This confirms the mechanism fires, lands deep, and diverges — it does NOT claim K12-static is
rescued. This single n=1 run still re-collapses into a new "1:"-repetition lock (deliverable tail),
consistent with the pivotal REPORT's "narrow mask / poisoned context wins" at K12. Whether v3
converts the collapse into an L2 completion is the RI-PIVOTAL question (K12+rewind-v3 wide, n=3,
total 4050, vs the v2 pivotal 0/7 -> historical 0/23). That larger experiment is now UNBLOCKED by
this a/b/c PASS but was NOT run here (distinct, materially larger pod spend; smoke mandate is
mechanism-fire only). L per-seed vs 0/23 pending that go-ahead.

## Files
- build_0022v3.log — build (exit 0).
- direct_cli_smoke_GARBAGE030.log — initial coffee/GARBAGE=0.3 run (correctly no-fire on healthy HTML).
- run_v3smoke.sh, v3_smoke_p2env.json, v3_smoke_manifest.json, v3_smoke.out — exact recipe/provenance.
- v3_smoke_W050_r00/ — pace_events.jsonl (a/b/c evidence), tokens.csv (divergence), p2.diag (tps),
  deliverable.html, sess.txt (K12 mask), frozen.txt, p2prompt.txt, diags.
