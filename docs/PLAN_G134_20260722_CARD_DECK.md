# PLAN G134 — THE CARD DECK: forge all, measure one-by-one, combine winners

Date 2026-07-22 (owner-directed methodology: "prima tutte insieme per creare patch e renderle
eseguibili, poi una alla volta per vedere che risultato danno, poi si comincia a combinarle").
Base for all cards: `g133/m1-residency @ dc5fe15` (hardened, measured, portable). Docs/ledger
branch: `plan/0051-transport-gate-20260713`. Ledger rows: `WIN-G134-CARD-*`.

## 0. Why this campaign closes the mandate

The measured 146 ms/token (F1 steady) decomposes into attack surfaces the deck covers:
orchestration 42% (untouched), launch 9% (untouched), exposed copy 27% (G73-OPEN, in flight),
plus multiplicative accept-amortization (spec-dec). Compose conservatively and >6 t/s EXACT on
the 3060 is arithmetic, not hope. Every card is open-source-grounded (Colibri structure,
llama.cpp CUDA graphs, DeepSeek MTP, Bonsai-style training).

## 1. THE DECK

| # | Card | Branch | Flag | Attacks | Expected | Prereq |
|---|---|---|---|---|---|---|
| C0 | **G73-OPEN** (mask+valve+rotator) | g133/g73-open | DS4_G73_OPEN | copy 27% + quality | ~5 t/s exact | in review (fix3) |
| C1 | **M4 dispatch cut** (Colibri phased structure: per-layer plan built once, no per-expert host round-trips) | g134/m4-dispatch | DS4_G134_DISPATCH | orchestration 42% (121ms) | halve -> +25-30% | none |
| C2 | **CUDA Graphs** (capture per-token launch sequence, replay ~zero-cost; llama.cpp-proven, works ON Windows/WDDM) | g134/cuda-graphs | DS4_G134_GRAPHS | launch 9% + WDDM tax | -15-20ms/tok | none |
| C3 | **Spec-dec n2k2** (prompt-lookup drafter, batched verify, candidate gating; H-SPEC-A GO) | g134/specdec | DS4_G134_SPECDEC | per-token costs x accept | +13-33% (measured accept 47.8%) | none |
| C4 | **Split-fused re-enable** (G73's original overlap trick, hardened for the transient path) | on g133/g73-open | existing env | route overlap | part of G73's 4.98 | C0 approved |
| C5 | **Lock Pages in Memory** (SeLockMemoryPrivilege) | no code — system toggle | — | arena 30->40GB+ | +30% mask coverage | owner gpedit + relogin |
| C6 | **MTP-head check** (does the GGUF retain DeepSeek's MTP layer? If yes: a free quality drafter for C3) | investigation only | — | drafter quality | unknown | none |
| C7 | **Bonsai-style IQ1 training done right** (thousands of iters, many experts — our pilot was undertrained; ceiling UNKNOWN per ledger) | offline track | — | M6 miss-cover | unknown | activation capture |

## 2. PHASE A — FORGE (all in parallel, now)

Rules (the discipline that made M1 land):
- One writer per branch; every card branches from dc5fe15; every card behind its OWN
  OFF-default env flag; flag-off = bit-identical (contract-asserted where feasible).
- Each card: implement -> adversarial review (gpt-5.6-sol) -> fix rounds until APPROVE ->
  build (Windows + keep Linux-clean: C11, no compiler extensions) -> static contract -> local
  commit; push only on APPROVE.
- Cards conflict-freely COEXIST as branches; NO merging in Phase A/B. Merge cost is paid only
  in Phase C for winners.
- C1 and C2 both touch the decode hot path — they will conflict at merge; irrelevant until C.

## 3. PHASE B — MEASURE (one at a time, same ruler)

- Ruler: the ctx256/64tok cyber_html fixture + one 256tok run; n>=3; temp 0; attribution ON.
- Control: dc5fe15 with ALL flags off (one run, the shared baseline).
- Per card: flag ON alone on ITS branch build; record decode t/s, span deltas (the span the
  card claims to attack MUST move — mechanism verification, not just headline), bit-exact
  vs control where the card claims exactness (C1/C2 yes; C3 changes decoding path — verify
  output EQUALITY at temp 0 instead).
- Ledger row per card with honest verdict; a card that doesn't move its span is INERT (the
  F2 lesson) regardless of headline noise.

## 4. PHASE C — COMBINE (winners only, greedy)

- Rank cards by measured isolated gain. Start from the best; merge next-best onto it;
  re-measure the pair (same ruler, n>=3); keep if the composition holds (no negative
  interaction beyond noise), else record the interaction and try the next.
- Target stack (expected): C0 + C1 + C2 (+C3 on top). C5 composes with C0 trivially (config).
- Every combination gets its own ledger row; the final stack gets the full quality grade
  (long-generation CSS-collapse check) + the sustained multi-turn chat measure (M5 of G133).

## 5. Current state hooks

- C0 in fix round 3 (bounded-terminal architecture); proceeds on its own track.
- C5 is a 2-minute owner action (gpedit -> Lock Pages -> relogin) + one measured load.
- C7 needs an activation-capture run design first (offline CPU track, weeks-scale, parallel).
- The 3090 pod re-enters at Phase B/C for the Linux/no-WDDM comparison once a
  memlock-unlimited template is set (console action).
