# HANDOFF — G130: Review-Driven Fix Plan + Rigorous Test Protocol

Date: 2026-07-20 (supersedes nothing; extends `HANDOFF_CODEX_20260720_G129_LOWBIT_SSD_WRAP.md`).
Author: independent adversarial code review session (50 agents, 3-stage find→refute→synthesize; 41/42 findings CONFIRMED against code, 0 refuted).
Audience: the chat delegated to implement fixes and run tests. Read this file top to bottom before touching anything.

Reviewed code identity: worktree `C:\Users\imanu\Documents\Codex\2026-07-07\cia\work\ds4-win-publish-g126-20260718-v2`, branch `feature/q1-0-resident-base`, tip **801a6d8** (runtime `16273d4`, harness/gates `415ed98`). The review ran on the pre-commit dirty tree; commit content is byte-identical (6,874 insertions / 329 deletions across `ds4.c`/`ds4_cuda.cu`/`g7_measure.ps1` — verified). All file:line refs below are against that tip.

---

## 0. Non-negotiable rules of engagement

1. **Cheapest evidence first.** Every GPU run must answer ONE pre-registered question with a predicted effect size, written down BEFORE the run. No sweeps, no "let's see".
2. **One change per experiment.** A run that mixes two changes produces zero attributable evidence.
3. **Never claim from n=1.** n=1 runs are diagnostic; their t/s is never quoted as a result.
4. **Negative results are results.** Every aborted or failed run gets a receipt and a ledger row (`NEGATIVE` / `CONTAMINATED`), same rigor as positives.
5. **One ds4 server / GPU workload at a time** on this machine. If the GPU is busy (check preflight), do CPU-only work instead.
6. **Do not re-hash the 80.8 GB model or the 39.0 GB sidecar.** Verify size + existing SHA receipts. Full identities: IQ2 `C:\ds4-models\ds4-2bit.gguf` 86,720,111,488 B SHA-256 `efc7ed60…`; Q1_0 sidecar 39,048,344,416 B SHA-256 `05040393…`.
7. **Git hygiene:** never reset/stash/bulk-add any working tree; stage only audited files; separate commits for runtime / harness / receipts / ledger. Push only to the real GitHub remote.
8. **The mandate is unchanged:** mean decode > 6.0 t/s, n≥3 independent clean processes, quality ≥ L2 (median L3), full/open router, zero forbidden same-token SSD→VRAM transitions. Do not declare victory on any subset.

---

## 1. Where the evidence stands (read before planning anything)

| Fact | Value | Source |
|---|---|---|
| Best replicated closed-mask decode | G73 4.9867 t/s (n=3; 5.0367 in the G103 control re-measure) | ledger |
| Best full/open exact-IQ2 decode | **G123 ≈ 1.65 t/s (n=3)** — the honest baseline | ledger |
| G129 structural safety n=1 | 0.1817 t/s wall / **0.205 t/s decode-only** (4,878 ms/token) | receipt 8d3898bc… |
| Q1_0 transport ceiling (quality broken) | ~6.8–7.0 t/s (G112–G114, L0/L1) | ledger |
| Q1 route share in G129 | 58% of routes, 52% gate mass | receipt |
| Q1 cold H2D traffic | 531 MB/token, **zero cross-token reuse** (34,002,173,952 B = 9,608 × 3,538,944 exactly) | receipt + code |

**The single most important review conclusion:** the attributed costs (Q1 transport ~210–420 ms/token, promotion staging 144 ms/token, IQ2 miss transport ~193 ms/token, telemetry <10 ms/token) sum to ~0.3–0.5 s/token. The measured cost is 4.9 s/token. **~4.3–4.5 s/token is UNATTRIBUTED** — not GPU-bound (15.9 W median), not disk (1.6 MB/s, zero page-outs), constant ~1.72 CPU cores. `DS4_Q1_0_PROFILE` counters exist (`ds4_cuda.cu:5528–5563`) but no `[q1-0-profile]` summary appears in any run log ever captured.

**Consequence: profiling comes before fixing (Phase P0). Optimizing before attributing the residual risks weeks of wasted work on the 12% term.**

Second conclusion: even at Q1-fraction zero, the open-regime engine floor measured to date (~536 ms/token non-transport in G123; 200.4 ms/token closed in G73) exceeds the 166.7 ms/token budget. Seam fixes are necessary but not sufficient; §5 P3 is the evidence-backed architectural exit.

---

## 2. Machine-state preflight (mandatory before EVERY GPU run)

Scripted, receipted (`preflight.json` in the run dir). A run started without a passing preflight is CONTAMINATED regardless of outcome.

- **P-1 Exclusivity:** no other `ds4_server`/GPU compute process; no rclone/robocopy/defrag/backup/Windows Update activity; disk queue < 1 on C:.
- **P-2 GPU baseline:** VRAM used ≤ ~700 MiB, GPU power at idle before start. Record `nvidia-smi` snapshot.
- **P-3 Host memory admission (fixes review F9):** available RAM ≥ planned commit + 2 GiB floor. The planned commit for G129-class runs is **~41.8 GiB** (24.5 pinned Q1 + 11.9 pageable Q1 + 5.5 pinned IQ2), NOT the 7.5 GiB the harness currently checks (`g7_measure.ps1:3863` ignores `Q1_0ArenaGB` — two-line fix required before the next run).
- **P-4 Paging baseline:** snapshot page-fault and page-out counters; the in-run monitor (§4) diffs against these.
- **P-5 Quiescence:** 3 sampling windows (existing rule) — CPU, disk, GPU quiet.
- **P-6 Artifacts identity:** model + sidecar size check against receipts (no re-hash); build fingerprint/manifest/exe SHA recorded; `git status` clean-or-explained recorded; config JSON echoed into the receipt.
- **P-7 Thermals/clocks:** record GPU/CPU temps at start; do not start if GPU is still hot (> ~60 °C idle) from a previous run.
- **P-8 Probes must ENFORCE, not record — and a null/denied/zero probe is a FAIL, not a pass.** Lesson from the failed G73 live run: preflight executed inside the Claude AppContainer sandbox where Win32_Process/memory/disk counters were denied ("Accesso negato") and it recorded `c_free_gib=0` / `d_free_gib=0` without refusing launch. Rule: run harnesses from a normal user shell (never the AppContainer sandbox), and any probe that returns null/denied/zero on a resource that obviously isn't zero blocks the run as "unable to verify".

---

## 3. Run-tier ladder (each tier gates the next; skipping tiers = contaminated)

| Tier | What | GPU budget | Quotable? |
|---|---|---|---|
| **T0** | Build + static gates: `g7_build.ps1` fresh Release, ctest, both `test_g129_*.py`, PS scriptblock parse, `git diff --check`, `-WhatIf` runner config echo | none | no |
| **T1** | Profile/attribution probes (§5 P0): n=1, 64 tokens, ONE env-var question each, ≤15 min GPU each | tiny | no — diagnostic only |
| **T2** | Structural safety n=1: full gate list of the G129 protocol (route/tier row counts, promotion pairing, zero forbidden transitions, VRAM returns to baseline) | ~30 min | no |
| **T3** | **Quality probe** (the resource-saver): ONE run, canonical cyberpunk prompt (SHA `38f6ec5e…`), temp0/nothink, 512–2000 max tokens, streamed. Grade L0–L3 by RENDERING the output (browser/HTML behavior), never by repeat-flags. **L0/L1 → STOP the whole arm**, write NEGATIVE, do not proceed. | one run | no |
| **T4** | Perf campaign n≥3 independent clean processes, decode-only metric, vs the full/open exact-IQ2 control (G123-class). Only reachable if T3 ≥ L2. | full | **yes** |

The control arm for T4 is a **full/open exact-IQ2** run on the same build — never G73 (closed) numbers.

---

## 4. In-flight abort criteria (watchdog — implement in the runner, out-of-process, 1 s cadence)

Abort = kill server, preserve logs, write NEGATIVE receipt with the *distinct* abort cause, ledger row. Never let a doomed run finish just to "have the number" — that is the resource waste this protocol exists to prevent.

- **A-1 Paging (G105 mode):** page-out delta > 100k over baseline, or process working set drops > 2 GB, or available RAM < 2 GiB floor → abort.
- **A-2 Throughput floor:** after a 16-token warm window, rolling decode t/s < 50% of the tier's pre-registered prediction for 30 consecutive tokens → abort (the question is already answered: the prediction failed).
- **A-3 TTFT cap:** first token not produced within 180 s of request start → abort (known unresolved 50+ s TTFT; 180 s means something new is wrong).
- **A-4 Early L0 heuristic (abort-trigger ONLY, never a final grade):** on the streamed text, ≥3 consecutive repetitions of a ≥20-char block, or >30% non-printable/mojibake in any 500-char window → abort the run, then render whatever was produced and record the L-grade of the partial output.
- **A-5 CUDA/server errors:** any CUDA error string, device reset, or server exit → abort with logs.
- **A-6 Promotion failure state:** until F3 is fixed (§5), any `state.failed=1` / worker exit → abort and label the cause `ssd_wrap_terminal` (do NOT misattribute as a perf result; see §6).
- **A-7 Residency violation:** any same-token SSD→VRAM event → abort, structural failure.
- **A-8 Time cap:** hard wall-clock cap per run, pre-registered (default 30 min T1/T2, 60 min T3), unless a longer cap is written in the pre-registration.
- **A-9 Monitor liveness:** the watchdog must assert its own data sources are advancing (sampler CSV row count, log tail offset). A dead sampler = the run is unmonitored = abort. (The failed G73 live run's GPU sampler died after one row and nobody noticed.)
- **A-10 Startup readiness must be progress-aware, not a fixed TCP timeout.** The committed G73 runner kills the server after min(300 s, TimeoutSec) of TCP-connect polling — but the 86.7 GB model can legitimately take longer to load-to-bind. Rule: track load progress (stderr "GiB cached" lines) and abort only if progress STALLS for N minutes; snapshot `nvidia-smi` + stderr tail before any kill; capture the server exit code on every path.
- **Streaming note:** A-4 requires streamed output. The current runners use non-streaming `HttpClient.PostAsync`, which makes mid-generation inspection impossible — either stream the response or (interim) gate on server-side counters (`ttft_server_seconds`, gen-progress lines already parsed every 250 ms).

---

## 5. The fix plan (ordered; each fix has its own acceptance test; do not reorder without evidence)

### P0 — ATTRIBUTION FIRST (all T1 probes; no code changes; ~1 GPU-hour total)

| # | Probe | Settles |
|---|---|---|
| U1 | n=1, 64 tok, `DS4_Q1_0_PROFILE=1`, all tracing OFF; capture the never-seen `[q1-0-profile]` split (upload_sync / q1_kernel / mixed_join) + `[routeprof]`/`[selprof]` | **the ~4.5 s/token residual — the dominant unknown** |
| U2 | Log (layer,expert) of Q1 routes across adjacent tokens, one open decode | cold-tail re-touch rate → whether F2's Q1 VRAM LRU pays |
| U3 | Same U1 run: `pageable_h2d_enqueue_seconds` field | effective pageable H2D bandwidth (3–6 GiB/s is an estimate) |
| U4 | ONE 512-token single-request decode, per-64-token route-share buckets | whether the 58% Q1 share drifts down at promotion steady state |
| U5 | CPU-side microbench (no model): 1,000 × {tiny sync memcpy, streamSync} under load | WDDM per-blocking-call cost → bounds F1's 30–160 ms band |

**Decision gate after P0:** if U1 shows the residual lives outside the seams below, STOP and re-plan against the profile — do not proceed to P2 on faith.

### P1 — Measurement fixes (before any perf-relevant run; no GPU needed)

- M1: decode-only t/s as the headline metric (wall-clock client t/s conflates 39.6 s TTFT; `g7_measure.ps1` non-streaming Invoke-RestMethod) — stream tokens or use server decode timing.
- M2: sample-count guard — the server-decode mean can silently average fewer samples than `Repeats`, and warmup can leak into the window. Assert count==Repeats in the receipt.
- M3: recompute `server_avg_tokens_per_second` from raw timings, not the `%.2f` log echo (2.4% quantization).
- M4: preflight commit check (F9, two lines, `g7_measure.ps1:3863`): require `Q1_0ArenaGB`-aware available-RAM ≥ commit + 2 GiB.
- M5: receipts must treat `VirtualUnlock not_locked==calls, failed==0` as PASS (it is the intentional WS-trim idiom on the never-locked sidecar mmap — cleared by review).
- M6: replace the purely textual `test_g129_*.py` assertions that gate math (they assert text presence only, execute nothing) with at least one executed-path test for the t/s aggregation.

### P2 — Runtime seam fixes (each = one commit + one T1 A/B probe; ranked by measured impact)

| Fix | What / where | Acceptance test |
|---|---|---|
| **F1+F7** | Kill the per-layer serialization: host-ids entry point into `cuda_moe_selected_load_q1_0` (skip the redundant blocking null-stream D2H at `ds4_cuda.cu:30626` — the ids are already host-resident at 33333–33336); enqueue cold H2D BEFORE the hot launch; replace the unconditional `cudaStreamSynchronize` at 30718–30719 with the event pattern already used at 28478/31725 | blocking-calls/token counter ~316 → < 20; hot/cold overlap visible in profile; bit-exact output |
| **F3+F11+F12** | SSD-WRAP semantics: stale-age drop = accounted non-fatal drop, NOT `state.failed` terminal (24239→24222–24243 conflation); fix the vacuous self-comparing epoch guard in `finish_one` (24122); refund budget on drop/failure (25151). **Note: the stale-age trap is throughput-coupled — invisible at 0.2 t/s (5.2 s window), near-certain at 6 t/s (≤155 ms window). It WILL kill mandate-speed runs if unfixed.** | CPU fault-injection: forced stale/failed job → run survives, drop counted, budget refunded (see §6) |
| **F5+F10** | Promotion off the decode thread: ssd_wrap as perf default (only after F3), FNV checksum computed in the worker before RAM_READY, not in commit; make the 5 s route-worker deadline fail-open to Q1, not process-abort | promotion staging seconds on decode thread ≈ 0 (was 9.22 s/64 tok = 144 ms/token) |
| **F4+F8** | The pageable third of the Q1 arena (3,575/11,008 slots, 11.78 GiB, 31.8% of upload bytes at `ds4_cuda.cu:6024`, copies 5439–5453): `cudaHostRegister` it (G4 proved 24.4 GiB/s on this box) OR mass-ranked slot assignment so hot experts land pinned; plus working-set protection (VirtualLock or min-WS raise) — the G105 failure mode is currently armed and the `pageable_paged_out_before_copy` counter is dead (never incremented) | pageable share of upload bytes < 2%, or registered; paging monitor shows zero page-outs under a 13 GB external-pressure test |
| **F2** | Small Q1 VRAM LRU (~600 slots ≈ 2 GiB) keyed (layer,expert) — **gated on U2**: only if cold-tail re-touch ≥ ~30% | Q1 H2D bytes/token drop ∝ measured re-touch rate; causality invariant untouched (it only checks IQ2 SSD deltas, 33500–33514) |
| F13/F14 | Optional: pack the 3-per-expert contiguous memcpys (5439, 28,824 enqueues/run); parallelize/overlap the 35–40 s single-thread FNV in the 80 s bootstrap (6101) | startup < 45 s; enqueue count/3 |

**What NOT to spend GPU time on** (review-closed): promotion parameter tuning toward the mandate (F6: 64/request = 0.58% of experts, wiped per request — arithmetically cannot move the 58% share); mass-LFRU bookkeeping (cleared — O(320), no sort); telemetry hunting (<10 ms/token); bigger mapped host windows (measured 2× regression).

### P3 — Architecture decision gate (after P0+P2 re-measure, ONE T2-class run)

If decode is still > ~250 ms/token full/open, stop seam work. The evidence points at ONE architectural exit: **CPU-GEMV for cold experts (KTransformers-style)**. Rationale from measured facts: cold weights are single-use (exact byte-match proof), host DDR ~50 GB/s outruns pinned PCIe 24.4 GiB/s for single-use reads, the per-layer join already exists (`add_f32_u64_kernel`), per-layer transport drops ~432× (28.7 KB activations vs 12.4 MB weights). It also attacks quality: retiring the 36.3 GiB Q1 arena as a DMA source frees RAM for ~5,500 **exact** IQ2 experts (~50% coverage) — fewer degraded routes, not more.
Spike order: (1) CPU GEMV one layer, correctness vs GPU path bit-comparison; (2) 8-thread AVX2 throughput bench vs 3.54 MB H2D+GEMM; (3) hybrid dispatch behind an env flag, T1 A/B. Pre-registered prediction required before the first GPU run.

---

## 6. SSD-WRAP trust status (read before ANY promotion-enabled GPU run)

The CPU "certification" (`runs/ds4/20260720_lowbit_recovery/g129_ssd_wrap_cpu_implementation_report.md`, reap-loop commit `2fd6a5f`) is **NOT a GO gate**:

- Its gates (PS parse, parser cases, mock, CTest 1/1, build) never exercise the state machine cross-request or under concurrency; they could not have caught any of the six confirmed defects (terminal stale-drop; vacuous epoch guard; unrefunded budget; dead `structural_rejects` counter; decode-thread FNV; unlocked counter mutations).
- The next-gate acceptance criteria ("zero rejects/stale/drops, counter pairing") are **vacuously satisfiable** by the dead counter and the self-comparing guard — fix and unit-test the counters FIRST or the safety run proves nothing.
- Commit `2fd6a5f` also **rewrote the handoff spec in the same commit** (deleted the fail-open requirement and the telemetry-counter requirements the code violates) and deleted the ledger rule barring CSV insertion before runtime safety. **Required action: restore the original fail-open + telemetry requirements in the handoff (or write an explicit, dated waiver), and restore the ledger admission rule.** Do not judge conformance against the rewritten §10.
- Required before the first promotion-enabled GPU run: **fault-injection tests** proving each fail-closed guard can actually FIRE (forced stale-age, forced provenance mismatch, forced partial read, forced pairing break) and that the run survives fail-open where the spec says fail-open.

## 7. G73 live two-turn runner (commit 801a6d8) — audited status

**The committed runner has ZERO live-GPU evidence.** The failed live run (`g73_liveB_e2e_20260720T053105Z/failed_start_receipt.json`) was produced by an OLDER runner version (self-recorded sha `8a69314f…`, 88,875 B) — the committed runner is a different blob (`2eff6fac…`, 129,546 B) that already contains the fix for that receipt's stated root cause (direct `G73OwnedLoggedProcess` launcher instead of the `cmd /c` wrapper). The commit subject "live two-turn runner evidence" oversells: only CPU-mock scenarios succeeded, and the mock proves lifecycle/byte-exactness plumbing only (`decode_tps_server=None`; the canned HTML passes the L2 regexes by construction). Every live-only path — server launch/env, TCP readiness, sampler, stderr-trace gates, t/s extraction, abort triggers — is unexercised.

**The real cause of the server's death is UNKNOWN** (stderr stops at "CUDA loading model tensors into device cache"; `server_exit=null`; stdout/launch logs 0 bytes; sampler dead after one row; OOM / disk-full / external kill / readiness-timeout kill all un-excluded). Before any live re-attempt:

1. Re-run with the COMMITTED runner and require the receipt's self-recorded runner sha to match the commit blob. Manual-recovery summaries (schema `*_manual_recovery`) never count as runner-native evidence.
2. Add server exit-code capture on all paths; check Windows Event Log/WER for the crash; verify real free disk (the failed preflight recorded 0 GiB unchallenged); run OUTSIDE the AppContainer sandbox (§2 P-8).
3. Apply §4 A-10 (progress-aware readiness — the current bare-TCP 300 s cap can kill a healthy loading server) and A-9 (sampler liveness).
4. Preflight ENFORCEMENT: the committed runner only enforces conflicting-process count + file SHAs; GPU baseline / RAM floor / free-disk / paging are recorded but never gate. Wire them per §2.
5. Close the closed-loop-abort gap: no t/s floor, no stall abort, no paging abort exist today even though decode lines are parsed every 250 ms (a 0.004 t/s run would burn the full 2 h timeout — the exact waste this protocol forbids). Also replace the full-file re-read every 250 ms with an offset-based tail (O(n²) self-perturbation).
6. t/s bookkeeping: formulas are correct but HTTP results are joined to log blocks BY ARRAY INDEX — key the join on `[g73-two-turn-epoch] request_epoch`, and gate on `decode_tps_server`/`prefill_tps` non-null so silent regex drift cannot pass.
7. Quality regex gates mis-grade real output (the "dark" gate rejects `#111111`/`#1a1a2e` but accepts `background:#00ffff`; "contrast" accepts `color:#f00`; "popup" = any `addEventListener`). Compute luminance from parsed hex, or demote these to advisory and gate only on parseable/complete-document + stop/raw contracts. Final L-grades remain rendered-visual only (§3 T3).
8. Mock coverage to add (cheap, CPU-only): a late-port-bind scenario (exercises the live readiness path) and a synthetic-stderr replay scenario (exercises tensor-reload/unsafe-tier aborts, kv0/order gates, t/s extraction) — all currently at zero coverage.

KEEP from this runner (audited good): direct launcher with exit-code capture; summary-always + failure receipts + held-handle lock with stale-lock recovery; finally-based cleanup/env-restore; dual-side sha256 byte-exact conversation capture; turn1 fail-fast gate; explicit non-claim fields; provenance sha pinning.

## 7b. Commit audit (dc52ec05..801a6d8) — cleared

Byte-identity of the reviewed tree vs commits CONFIRMED exactly (6,874/329: ds4.c 58/8, ds4_cuda.cu 3888/218, g7_measure.ps1 2928/103; working tree clean at 801a6d8). No secrets/tokens/credentials in any of the 78 files (patterned grep, zero hits); no binaries; largest evidence file 39 KB. Author/committer identities consistent (`imanu <imanu86@gmail.com>`; tip author-date precedes parent due to a benign rebase). Minor flags: absolute `C:\Users\imanu\…` paths inside 19 evidence JSONs (username + package-dir leak; matches pre-existing receipt convention — fix only if the repo goes public); the two `test_g129_*.py` live at repo root while the G73 tests live in `tests/` (unify later); and M6 stands — those gates are text-presence checks that pass on any tree containing the strings.

## 8. Provenance & ledger (every run, no exceptions)

Receipt must contain: pre-registration (question, prediction, abort caps), preflight snapshot, build identity (manifest/fingerprint/exe SHA), git HEAD + status, full config echo + env block, raw timings, watchdog log, outcome label (`VALID N≥3` / `SAFETY N=1` / `DIAGNOSTIC` / `NEGATIVE` / `CONTAMINATED`), and for quality runs the rendered-output L-grade with the rendering method. Ledger row immediately after every run, positive or negative. Scope labels (closed vs full/open, cache-state, protocol token count) are mandatory — cross-protocol t/s comparisons are forbidden.

---

## Appendix A — Confirmed findings index (F1–F14)

| # | Sev | One-liner | Where |
|---|---|---|---|
| F1 | crit | Q1 cold dispatch fully serialized; redundant blocking D2H of host-resident ids drains hot compute | ds4_cuda.cu:30626 (33209→33462→30719) |
| F2 | high | Zero Q1 VRAM residency: 531 MB/token re-upload, zero cross-token reuse (exact byte-match proof) | 30688 |
| F3 | crit | SSD-WRAP stale-age drop is terminal for the run; trap is throughput-coupled (springs at mandate speed only) | 24239/24222–24243 |
| F4 | high | 3,575/11,008 Q1 slots are unregistered pageable VirtualAlloc on the H2D path (31.8% of bytes) | 6024, 5439–5453 |
| F5 | crit | Synchronous promotion preads on decode thread: 144 ms/token measured | 25165–25186, 33620–33632 |
| F6 | high | Promotion capacity (64/request, wiped per request) cannot move the 58% Q1 share | 25083, 22806–22809, 25898 |
| F7 | crit | Unconditional per-route-call `cudaStreamSynchronize` (~43×/token) — forces t_token ≥ t_H2D + t_compute | 30718–30719 |
| F8 | high | No working-set protection for the 11.78 GiB pageable tier; G105 mode armed; dead paging counter | 6024, 8246–8247 |
| F9 | high | Preflight admits 7.5 GiB vs 41.8 GiB commitment; in-process check dead | g7_measure.ps1:3863, ds4_cuda.cu:5963 |
| F10 | med | 7 MB FNV (+memcpy) on decode thread in SSD-WRAP commit; 5 s deadline is fail-abort not fail-open | 24196, 30988–31002 |
| F11 | med | Vacuous epoch guard (compares record's epoch to a copy of itself) | 24122 |
| F12 | med | Promotion budget charged at submit, never refunded on drop/failure | 25151 |
| F13 | low | 3 memcpyAsync per expert for source-contiguous bytes (28,824 enqueues/run) | 5439 |
| F14 | low | 80 s bootstrap: ~35–40 s single-thread FNV over 38.96 GB | 6101 |

Cleared by adversarial verification (do not re-investigate): VirtualUnlock 0/129 (intentional idiom — fix receipts only, M5); telemetry as the 0.18 cause (<10 ms/token); mass-LFRU bookkeeping cost; route-trace D2H syncs (production-path, F1's, not telemetry's); first-order harness soundness (second-order fixes = M1–M3, M6).

## Appendix B — The cost model (decode-only, ms/token)

| Component | Now | Best case after P2 |
|---|---|---|
| Q1 cold H2D (531 MB/token, 68/32 pinned/pageable) | 53–70 serial | ~0–10 additive if overlapped (+LRU per U2) |
| WDDM blocking round-trips (~316/token) | 30–160 | ~5 |
| Promotion staging (sync path, measured) | 144 | ~0 (worker) |
| IQ2 miss transport (9.4× over raw pinned) | 49 | 5–10 |
| Per-expert copy granularity | 1–3 | ~0 |
| Q1 kernels (adequate) | 4–6 | unchanged |
| Telemetry (trace on) | 3–10 | 0 |
| **Unattributed residual** | **~4,300–4,500** | **unknown — U1 settles it** |
| Budget at mandate | **166.7** | |

**Bottom line for the implementer:** run P0-U1 first. If the residual is in the seams, P2 gets you to a fair architecture verdict; if it is not, P2 would have been wasted GPU time — which is exactly what this protocol exists to prevent.
