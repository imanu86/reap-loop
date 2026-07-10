# E-PHASE — structural phase-mask analysis (piecewise-static design evidence)

**Date:** 2026-07-10 · **Mode:** OFFLINE, trace-only (no GPU / WSL writes / pod)
**Script:** `scripts/analyze_phase_masks.py` (reproducible, `py_compile` clean)
**Artifacts:** `phase_coverage.csv`, `boundary_churn.csv`, `stats.json`

## Hypothesis under test (user)

Expert demand changes with the **structural phase** of the generated document
(CSS vs body-markup vs script), so the right mask is **piecewise-static** with a
**relearn at structural boundaries**. E1 (`runs/ds4/20260710_e1_top_expert_mass/`)
measured cross-phase top-1 overlap **53.6%** — but with a **blind temporal**
split (first vs last third). Here the split follows the *actual structure of the
generated text*, and we quantify how much the frozen warmup mask starves each
later phase, what a per-phase mask would recover, and what a boundary relearn
costs in delta-prefetch.

---

## 1 — Data inventory & alignment (declared per source)

| Source | Weights | Text | Alignment trace↔text | Usable |
|---|---|---|---|---|
| `20260710_pod_cache1024_warmup_replay/W130/route_W130.csv` (129 gen tok, full model) | yes | `tw_W130.txt` | **HIGH/MEDIUM** — trace logs generated tokens only (pos 218–346 = prompt offset, no prompt rows); char→token mapped proportionally (no tokenizer on disk) → sensitivity band ±5 tok reported | **primary** (head→CSS) |
| `…/W50/route_W50.csv` (49 gen tok) + `sess_W50.txt` | yes | `tw_W50.txt` | same as W130 | **warmup donor** + validation |
| `k91_coding_vram/trace_coding.tgz` (11 traces, 52–255 gen tok, full model) | yes | `gen_*.log` minus `ds4:` diag (interleaved mid-line, handled) | **MEDIUM** — same proportional mapping; text extraction verified on p00 ("Here"+"'s a Python function" rejoin) | **primary** (prose→code analog) |
| `20260710_scope_divergence_pod/r1/s1_r1.csv.gz` | — | r1 text unsaved (invalid UTF-8 tail) | schema `pos,layer,pruned_mass,total_mass`: **aggregate only, no expert ids** | **NOT usable** for masks (context only) |
| `20260710_scope_divergence_pod/ctrl/content.txt` (1200 tok coherent) | no trace | yes | — | NOT usable (no routing) |
| `20260709_trace_ab_html220_v2` / `_rev` routing.csv | yes | yes | possible | **excluded**: mask-constrained (SOTA run) — selected-6 biased by the applied mask, not free-router demand |
| k91 `knee/out/route_*.csv` | **no weight cols** | — | — | excluded (E1 precedent) |
| k91 `loop/*.tgz` long runs | yes | partial | — | excluded: mask-constrained |
| `20260710_m1a/m1b_*` (untracked, untouched) | **no routing trace** | yes | — | corroboration of collapse text only |

**Coding phases:** the mandate's import/def/body sub-split was implemented but
does not trigger — the 255-token traces end **inside** the first function/
docstring (e.g. python-csv ends mid-`Raises:`), so no complete def/body exists;
the robust structural boundary available is **prose-intro → code-fence**
(register switch, the coding analog of `<style>`/`<script>`). Declared, not hidden.

**Key data gap (honest):** **no full-model weighted trace reaches the HTML
body/JS phase** (longest = W130, 129 tok, still inside CSS; runs that do reach
`<script>` — trest_W50, m1a — have no routing trace). The JS leg of the
hypothesis is therefore tested via (i) the coding prose→code analog and (ii) the
cross-run sess_W50→CSS starvation; a JS-phase trace needs a future pod capture.

### Validation of the mask pipeline

Computed top-23-by-cumulative-mass from `route_W50.csv` vs the **real**
`sess_W50.txt` kept-set (inverted prune list, 233/layer):
**Jaccard = 1.000 (exact match, 40/40 layers)** — the offline mask builder
reproduces `build_session_mask.py` bit-for-bit.

---

## 2 — Frozen warmup mask vs per-phase local mask (def-1 routed mass, mean over 40 layers)

Frozen mask = top-K from the first 50 generated tokens (= `PACE_WARMUP`).
Local mask = top-K from the phase's own tokens (in-sample optimum = ceiling).

### K = 23 (REAP-LOOP keep)

| trace | phase | tok | cov frozen | (±5 tok sens) | cov local | **gain** | cov prev-phase |
|---|---|--:|--:|--|--:|--:|--:|
| html_W130 | head | 32 | 0.808 | 0.797–0.808 | 0.844 | +3.5pt | — (in warmup) |
| **html_W130** | **css** | 97 | **0.641** | 0.632–0.651 | **0.819** | **+17.8pt** | 0.494 |
| code_python-csv | code | 216 | 0.546 | 0.540–0.553 | 0.702 | +15.5pt | 0.513 |
| **code_js-debounce** | **code** | 99 | **0.327** | 0.325–0.328 | **0.802** | **+47.5pt** | 0.480 |
| **code_sql-window** | **code** | 118 | **0.334** | 0.330–0.334 | **0.711** | **+37.7pt** | 0.471 |
| **code_c-pointers** | **code** | 140 | **0.475** | 0.469–0.480 | **0.716** | **+24.1pt** | 0.594 |
| code_py-asyncio | code | 225 | 0.560 | 0.553–0.567 | 0.744 | +18.4pt | 0.452 |
| **code_git-rebase** | **code** | 110 | **0.359** | 0.354–0.362 | **0.793** | **+43.4pt** | 0.609 |
| code_docker-multi | code | 216 | 0.640 | 0.637–0.645 | 0.801 | +16.1pt | 0.570 |
| **code_rust-owner** | **code** | 105 | **0.482** | 0.482–0.488 | **0.701** | **+21.9pt** | 0.585 |
| code_api-paging | code | 221 | 0.526 | 0.520–0.532 | 0.693 | +16.8pt | 0.439 |

(bold = phases **fully outside** the warmup window — the clean measurements;
regex-email and test-mock have a single structural phase → no boundary.)

**Aggregates K=23:** post-warmup phases (n=11): cov frozen **0.564** vs local
**0.740** → **gain +17.6pt**. Phases *fully* out of warmup (n=5): frozen
**0.395** vs local **0.745** → **gain +34.9pt**. Previous-phase mask on next
phase: **0.521** — *worse than the frozen warmup mask* (0.564): hand-me-down
masks don't transfer either; the relearn must come from the **new phase's own
tokens**.

**K = 38** (E-CAL Kmin-cov90): same picture, higher floor: post-warmup frozen
**0.653** vs local **0.843** → **gain +19.1pt**; fully-out gain **+38.1pt**.
Raising K alone does **not** close the structural gap.

### The real frozen mask, cross-run (the production scenario)

`sess_W50.txt` — the actual mask phase 2 ran with in the warmup replay — scored
against W130's phases (same prompt family, temp 0; cross-run, MEDIUM confidence):

| phase | coverage of real sess_W50 |
|---|--:|
| head (markup, what the warmup saw) | **0.803** |
| **css (next structural phase)** | **0.498** |

A **30.5pt structural drop**: the deployed frozen mask leaves **half of the
routed mass un-covered** as soon as the document enters CSS. This is the
starvation the user hypothesised — measured on the mask actually shipped.

---

## 3 — Structure vs blind time (does structural beat E1's temporal split?)

| metric (K=23, mean) | cross-structural-boundary | blind terciles (E1-style) | within-phase split-half |
|---|--:|--:|--:|
| top-K mask Jaccard | **0.289** | 0.326 | **0.447** |
| dominant top-1 identity overlap | **0.440** | **0.544** | **0.687** |

- The blind-tercile top-1 overlap **0.544 replicates E1's 53.6%** on this pool —
  the pipeline is consistent with E1.
- Ordering is exactly what the structural hypothesis predicts:
  **within-phase (0.687) > blind-time (0.544) > cross-phase (0.440)**. The blind
  temporal split under-estimates the change because its thirds mix phases;
  segmenting at the real structural boundary exposes **more** demand shift, and
  within a phase demand is far more stable than E1's temporal number suggested.
  **Structural segmentation explains more variance than blind time.**

## 4 — Churn at the boundary (delta-prefetch cost of a relearn)

| K | experts changed / layer (mean) | total (40 layers) | worst-case delta-prefetch |
|--:|--:|--:|--:|
| 23 | **12.9 / 23 (56%)** | ~516 | **~3.4 GiB** |
| 38 | 20.1 / 38 (53%) | ~805 | ~5.3 GiB |

Range across boundaries K=23: 10.0–15.2 experts/layer (2.7–4.1 GiB). This is a
**once-per-phase** cost, the same order as a *single* rotate32 delta event under
0021 (smoke: 0.6–4.2 GiB *every 32 tokens*), and an upper bound — at cache≥512
part of the entering set is already resident.

---

## VERDICT

> **POSITIVO — the structural-phase hypothesis is supported on every axis
> measurable offline.** (1) The warmup-frozen mask **starves later structural
> phases**: −17.6pt coverage vs the local optimum on post-warmup phases (K=23),
> **−34.9pt on phases fully outside the warmup** (frozen 0.395 vs local 0.745),
> and the *real* production mask sess_W50 drops **0.803 → 0.498** crossing
> head→CSS. The gain of a per-phase mask clears the 10–15pt design bar at both
> K=23 (+17.6) and K=38 (+19.1) — raising K alone does not close the gap.
> (2) **Structure explains more than blind time**: mask stability within-phase
> 0.447 vs cross-boundary 0.289 Jaccard; dominant-expert overlap 0.687 / 0.544 /
> 0.440 (within / blind-tercile / cross) — and the 0.544 replicates E1's 53.6%.
> (3) The boundary cost is bounded and amortised: **~13/23 experts per layer
> (~3.4 GiB worst case)**, once per phase, deliverable by 0021 delta-prefetch.
> (4) Hand-me-down masks don't help (prev-phase cov 0.521 < frozen 0.564): the
> relearn must be a **mini-warmup on the new phase's tokens**.

### Key numbers
- Frozen-mask coverage on post-warmup phases: **0.564** (K23) / 0.653 (K38); fully-out-of-warmup **0.395**.
- Per-phase local optimum: **0.740** (K23) / 0.843 (K38) → **gain +17.6pt / +19.1pt** (fully-out **+34.9 / +38.1pt**).
- Real sess_W50 cross-run: head **0.803** vs CSS **0.498** (−30.5pt).
- Structure vs time: Jaccard within 0.447 / tercile 0.326 / cross 0.289; dom-overlap 0.687 / 0.544 / 0.440 (tercile replicates E1 53.6%).
- Boundary churn: **12.9/23 experts per layer** (K23) ≈ **3.4 GiB** worst-case delta-prefetch, once per phase.

### Limits (honest)
- **No full-model trace reaches HTML body/JS** — the JS leg rests on the coding
  prose→code analog (5 clean boundaries) + cross-run CSS starvation. A pod trace
  through a full document (~1200 tok, TRACE on, unmasked) would close this.
- Token↔char alignment is proportional (no tokenizer on disk): ±5-token
  sensitivity moves coverages by ≤1.5pt (see `cov_warm_sens_*`) — conclusions robust.
- n = 1 HTML two-phase trace + 9 coding boundaries; single model/quant (ds4-2bit).
- Coverage is **def-1 routed mass** (6 selected experts), as E-CAL; def-2
  full-256 is unobservable offline. E-CAL F3 stands: coverage alone doesn't
  predict survival — actuation (static-per-phase, never rotate) and provenance
  (session-learned) are part of the design below, and the whole thing still
  needs a pod A/B before shipping.

---

## DESIGN (verdetto positivo) — piecewise-static mask / phase-gated relearn — candidate patch **0025** `DS4_PACE_PHASE`

1. **Trigger** = structural markers in the *emitted* text — `</style>`, `<body>`,
   `<script>`, code fences — detected via the existing PACE ring n-gram buffer /
   light detok of the last emitted tokens; no new trace path.
2. **On trigger**: phase-gated relearn = **mini-warmup** of 30–50 tokens on the
   new phase: keep the current mask *active* (no speed cliff) while the 0020
   unbiased routed-mass accumulator restarts (fresh or strongly decayed —
   prev-phase carry-over measured *harmful*: cov 0.521 < frozen 0.564).
3. **Rebuild**: weighted relearn (rank by cumulative gate mass, same
   `build_session_mask.py` rule validated here at Jaccard 1.000), K fixed
   (23/38) or per-layer via 0024 cov-sizing computed **on the phase window**.
4. **Swap** via **0021 delta-prefetch**: measured need ~13/23 experts per layer
   (~3.4 GiB worst case, once per phase — same order as ONE rotate32 delta).
5. **Then FREEZE static** until the next marker — never rotate (E-CAL F3:
   static ≫ rotate at equal coverage).
6. **Env**: `DS4_PACE_PHASE=1`, `DS4_PACE_PHASE_MARKERS` (default html+fence set),
   `DS4_PACE_PHASE_MINIWARMUP=30`.
7. **Roadmap slot**: **S3**, prevention rung next to 0020/0022 (rewind stays the
   correction for *unexpected* drift; 0025 removes the *predictable* structural
   part). **Gate**: pod A/B frozen-W50 vs phase-gated (n≥3, ≥2000 tok, L0-L3 +
   t/s) — plus one unmasked full-document trace to close the JS data gap.

### Reproduce
```
python scripts/analyze_phase_masks.py --reap-loop-root <reap-loop>
```
