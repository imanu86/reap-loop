# G130 Fleet Status — 2026-07-20 evening

Self-contained handoff: any agent (Codex or Claude) can resume from this file + the repos alone.
Context: `docs/HANDOFF_CODEX_20260720_G130_FIX_PLAN_AND_TEST_PROTOCOL.md` (the plan these tasks implement).
Process used: every branch went through implementer → independent reviewer → (fix rounds) → independent test execution → commit+push. Reviewer and implementer are never the same agent. Commit messages record the cycle.

## Landed branches (pushed, reviewed, tests verified)

Repo `https://github.com/imanu86/ds4-win.git`, all branched off `feature/q1-0-resident-base` @ 08a4d27:

| Branch | SHA | What | Rounds |
|---|---|---|---|
| g130/m4-preflight-admission | 7fe9b94 | Fail-closed Q1-arena-aware RAM admission + executable wired tests | 3 |
| g130/u1-neutrality | 573ba0e | OFF-path neutrality: cleanup-sync log gated | 1 |
| g130/ssdwrap-epoch-guard | 00c3cc4 | F11: epoch guard vs live request epoch; stale = non-failing discard | 1 |
| g130/u1-attrib-tests | 272a465 | Attribution-v2 emission-contract validator + executed fixtures (3/3) | 1 |
| g130/watchdog-manifest | e1973c8 | A-1 threshold in manifest; A-7 vocabulary extended (contract markers) | 1 |
| g130/m5-receipts | c6bcba8 | VirtualUnlock classification: intentional-ws-trim-idiom / no-unlock-activity / FAIL-before-throw | 2 |
| g130/m1m3-meter | 9504826 | Decode-only headline t/s; samples==Repeats hard gate; raw %.9f recompute via shared parser | 2 |
| g130/utf8-framer-fix | c6de103 | Pre-existing suite failure: TEST bug (PS5.1 stdout encoding), framer proven correct | 1 |
| g130/ssdwrap-fault-injection | 08f26f4 | Fault-injection suite via DS4_TESTING seams. **CAVEAT: compile unverified — run cmake+ctest first** | 2 |
| g130/g73-runner-hardening | deeb483 | Progress-aware readiness (monotonic metric + absolute cap), closed-loop aborts, epoch joins, mock 5/5 | 2 |
| g130/g73-quality-gates | af3c388 | WCAG-luminance advisory gates + synthetic-stderr mock scenario, 7/7 | 1 |
| g130/u1-test-portability | 7a3664e | G130_BASE_WORKTREE env + clean skip (removes author-local path) | 1 |

Repo `https://github.com/imanu86/reap-loop.git`:

| Branch | SHA | What |
|---|---|---|
| g130/spec-restore | 1fad675 | Restores fail-open + telemetry requirements and ledger admission rule deleted by 2fd6a5f, dated notes |

## In flight (uncommitted, in scratchpad clones — if lost, relaunch from the instructions below)

1. **g130/u1-attribution-v2** (worktree exists locally; round-2 fix by gpt-5.6-sol in progress). Round-1 review PROVED: zero added CUDA sync (full API call-site inventory unchanged), closure arithmetic sound, blocking-site coverage complete, wire format matches g130/u1-attrib-tests. Round-2 blockers being fixed: (a) OFF-path must be single predicted branch on a cached global — no TLS loads when disabled, lazy TLS alloc, plus compile-out macro DS4_G130_ATTRIB_COMPILED_OUT; (b) emission must be inside the measured window (span_attrib_emit_ms in sum; request summary reconciles decode_total vs wall with loop_overhead_s); (c) every failure return restores the saved span. Acceptance: re-prove the API inventory + nvcc/MSVC compile.
2. **g130/ssdwrap-semantics** (round-3 fix done by gpt-5.6-sol, round-3 re-review in progress). History: R1 caught terminal-stale/refund/dead-counter; R2 caught commit-after-terminal via poll_internal, refund-before-charge race (charge was outside submit's mutex), first-terminal-only counter. R3 claims: poll_internal early-out on state.failed; charge moved into submit under mutex (charge_applied gates refund); per-event structural counter.

## Integration notes for whoever merges

- Branches are independent off 08a4d27; merge into `feature/q1-0-resident-base` in any order EXCEPT: `u1-attribution-v2` and `u1-neutrality` both touch the mixed-Q1 cleanup region of ds4_cuda.cu (~5520-5529) — merge `u1-neutrality` first, resolve in favor of gating BOTH logs; `ssdwrap-semantics` and `ssdwrap-epoch-guard` touch the same state machine — epoch-guard first (smaller), then semantics.
- The 2 pre-existing suite failures at 08a4d27 are fixed by `utf8-framer-fix` + `u1-test-portability`; after merging both, `python -m pytest tests/` should be fully green (G130_BASE_WORKTREE optional).
- After merging attribution-v2 + attrib-tests: the U1 run becomes launchable per the plan's §3 ladder (T0→T1), using `g130_u1_q1_profile.ps1` with the watchdog. Gate: plan §5 P0.
- SSD-WRAP promotion GPU runs remain gated on: semantics branch merged + fault-injection suite compiled and green (cmake+ctest) + the restored spec (reap-loop g130/spec-restore).

## Environment notes (for a fresh Codex chat)

- codex CLI 0.144.6 installed, ChatGPT auth. Models available: gpt-5.6-sol/terra/luna, gpt-5.5, gpt-5.4(-mini), gpt-5.3-codex-spark. `codex exec` needs `< /dev/null` when headless; sandbox blocks `.git` writes (commit outside).
- Windows git clones need `git clone -c core.longpaths=true` or files silently vanish from checkout.
- Worktree pushes can fail with `fatal: '$GIT_DIR' too big` — push the branch from the main clone instead.
