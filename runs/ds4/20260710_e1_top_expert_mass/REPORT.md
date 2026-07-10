# E1 — Top-mass precision pin: dominance & stability analysis

**Date:** 2026-07-10 · **Mode:** OFFLINE, trace-only (no GPU / WSL / pod)
**Script:** `scripts/analyze_top_expert_mass.py` (reproducible, `py_compile` clean)
**Artifacts:** `stats.json`, `per_layer_dominance.csv`, `overlap_matrix_top1.csv`

## The lever under test

User proposal: for each of the 40 routed layers, take the *top expert* and keep
it resident in VRAM at a **higher quant (Q4 ≈ 4.5 bpw)** instead of the 2-bit
base, pinned as a quasi-shared expert — a static "franken-gguf" — to buy
precision/stability for the masked system. This probe decides whether that is
worth a pod A/B (E2).

Model geometry (k91 `meta.json`): DeepSeek-V4-Flash, 43 layers, first 3 dense
(hash) → **40 routed layers** (3–42), 256 experts, **top-6** routing,
`bytes_per_expert = 7 077 888 = 6.75 MiB` at the IQ2_XXS ≈ 2.0625-bpw anchor.
Shared expert is separate and already **Q8** in this gguf (`SExpQ8`), so it is
out of scope — the pin targets the 2-bit *routed* experts.

## Data sources (weighted routing traces, full-model = true gate distribution)

Trace schema `pos,layer,n,e0..e5,w0..w5`. **The weights are NOT reliably sorted**
(verified: `trace_ab`, coding traces have w4<w5 rows), so top-1 = argmax over
`w*`, not `e0`. "Mass" per (pos,layer) = sum of the 6 routed gate weights; a
share is that expert's weight / this sum.

| Set | Source | Traces | Tokens/trace | distinct experts/layer |
|---|---|---|---|---|
| Coding (11 prompts) | `k91_coding_vram/trace_coding.tgz` (python-csv, js-debounce, sql-window, c-pointers, py-asyncio, regex-email, git-rebase, docker-multi, rust-owner, api-paging, test-mock) | 11 | 52–255 | ~140–162 |
| HTML/frontpage (phase-1) | `reap-loop/runs/ds4/20260710_pod_cache1024_warmup_replay/{W50,W130}/route_*.csv` | 2 | 49 / 129 | 74 / 109 |

All traces are unmasked full-model routing (≈140–160 distinct experts touched
per layer over the window → the router picks freely from all 256). The k91 `loop/`
long runs and knee `route_*.csv` were **not** used for dominance: the former are
mask-constrained, the latter carry **no weight columns** (frequency only) — noted
per instructions. `trace_ab` HTML routing.csv showed only 27 distinct top-1/layer
(SOTA-restricted) and was excluded from the full-model pool.

---

## Q1 — DOMINANCE: how much mass does the per-token top-1 carry?

Pooled over all 13 full-model traces (~2 780 tokens/layer):

| Metric | top-1 | top-2 cum | top-3 cum |
|---|---|---|---|
| **pooled per-token mean** | **30.5%** | 51.2% | 66.8% |
| pooled per-token median | 28.9% | — | — |
| per-layer median (mean across 40 layers) | 29.4% | 51.7% | 67.2% |
| per-layer p10–p90 | 26.1% – 31.7% | 47.5% – 54.5% | 63.7% – 69.7% |
| within-token p10–p90 (spread) | ~22% – ~44% | — | — |
| HTML pool | 33.3% | — | — |
| Coding pool | 30.3% | — | — |

**Finding:** the per-token top-1 routed expert **does** carry a meaningful,
layer-uniform share of the gate mass — **~30% mean**, above the 25–30% bar the
user set. It never drops below ~25% mean on any layer (range 24.7% L16 → 35.0%
L28; see `per_layer_dominance.csv`). Halving the quant error of *whichever
expert is top-1 for a given token* would touch ~30% of the routed output. **So
far the lever's premise looks right.** The catch is Q2.

---

## Q2 — STABILITY: is the top-1 the *same* expert across prompts/phases?

The `dom_expert` column of `per_layer_dominance.csv` shows the single
highest-mass expert differs on nearly every layer between tasks (L3: 101, L4:
233, L5: 236 …). Quantified:

### Cross-prompt top-1 identity overlap matrix (fraction of 40 layers with same #1-by-mass expert)

Off-diagonal is essentially noise (`overlap_matrix_top1.csv`, excerpt):

| | html_W50 | html_W130 | py-csv | js-deb | sql-win | c-ptr | rust-own |
|---|---|---|---|---|---|---|---|
| **html_W130** | 0.30 | 1.00 | 0.03 | 0.03 | 0.03 | 0.00 | 0.00 |
| **py-csv** | 0.05 | 0.03 | 1.00 | 0.10 | 0.15 | 0.08 | 0.10 |
| **c-ptr** | 0.03 | 0.00 | 0.08 | 0.10 | 0.05 | 1.00 | **0.38** |
| **rust-own** | 0.03 | 0.00 | 0.10 | 0.13 | 0.03 | **0.38** | 1.00 |

(The only non-trivial pair, c-pointers↔rust-owner 0.38, is two systems-programming
prompts — otherwise nothing transfers.)

| Stability axis | top-1 identity overlap | top-2 set Jaccard |
|---|---|---|
| **Cross-task (HTML vs coding, pooled)** | **2.5%** (1/40 layers) | 0.04 |
| Within-coding (mean pairwise, 11 prompts) | 7.8% | 0.10 |
| Within-HTML (W50 vs W130, small n) | 30.0% | 0.22 |
| Cross-phase (first⅓ vs last⅓, same run) | 53.6% | — |

### Coverage — what a *single static pin per layer* actually captures

Because the top-1 role rotates over ~150 experts/layer, the single highest-mass
expert per layer captures only a sliver of the layer's total routed mass:

| Coverage of the one pinned expert | mass share / layer |
|---|---|
| Pooled all tasks (a truly static gguf) | **5.7%** (median 5.1%) |
| Best-case per-task pool — HTML | 15.5% |
| Best-case per-task pool — coding | 5.9% |
| Single-run ceiling (live per-session pin) | **16.7%** |
| Fraction of tokens the pinned expert *is* the top-1 | 8.0% |
| Fraction of tokens the pinned expert is selected at all | 27.7% |

**Finding:** the 30% dominance is a property of a **moving target**, not of a
fixed expert. A static per-layer pin upgrades only **~6% of routed mass** (≤16.7%
even in the most favourable single-session case), and the identity it would pin
is **task-specific**: a coding-built gguf pins the wrong experts for HTML
(2.5% overlap). Coding is especially hostile (5.9% coverage, 7.8% prompt-to-prompt
overlap) — consistent with the prior "coding = dominio largo". The only regime
with any headroom is a *live per-session* pin (cross-phase 53.6%, ceiling 16.7%),
which is precisely the existing session-mask machinery, **not** a static gguf.

---

## Q3 — BUDGET: VRAM cost of the pin

Base 2-bit expert = **6.75 MiB**; Q4_K at 4.5 bpw = 6.75 × (4.5/2.0625) =
**14.73 MiB/expert** (×2.18; matches the user's "~15 MiB"). Cache slot =
6.75 MiB; usable expert cache in real 12 GB = **407 slots** (`meta.json`).

| Pin depth | experts (×40 layers) | Q4 resident | = cache slots | = % of 407-slot cache |
|---|---|---|---|---|
| **top-1** | 40 | 589 MiB (0.58 GiB) | 87 | **21.4%** |
| top-2 | 80 | 1178 MiB (1.15 GiB) | 175 | 42.9% |
| top-3 | 120 | 1767 MiB (1.73 GiB) | 262 | 64.3% |

Even top-1 costs **~21% of the entire usable expert cache** to upgrade **~6% of
per-layer routed mass** with a static pin. (Partial rebate: the pinned expert is
the highest-hit one, so keeping it resident removes some stream traffic — but the
gross 87-slot displacement is the honest cost.)

---

## VERDICT

> **FRANKEN-GGUF STATICO NON VIABLE.** Per-token dominance is real (**30.5%** of
> gate mass on the top-1) but it is **not carried by a stable expert**: the top-1
> role rotates over ~150 experts/layer, so a single static pin captures only
> **~5.7%** of per-layer mass (best-case per-task pool 15.5% HTML / 5.9% coding),
> and its identity does **not** transfer cross-task (**2.5%** top-1 overlap,
> Jaccard-top2 0.04). **Per-output-type variants don't rescue it** either
> (within-coding prompt-to-prompt overlap **7.8%**). The only version with margin
> is a **LIVE per-session pin** (cross-phase 53.6%, ceiling **16.7%**), i.e. the
> session-mask machinery already in the repo — not a static gguf. And the cost is
> steep regardless: top-1 pin = **21.4%** of the 407-slot cache.

**Recommendation:** do **not** spend the E2 pod A/B on a static franken-gguf —
the ceiling is a ~6% precision uplift on a fraction of layers, bought with 21% of
cache. If precision-pinning is pursued at all, the only defensible form is a
**live, per-session** top-mass pin folded into the existing session-mask build
(rank-by-cumulative-mass already computed there), whose realistic ceiling is
~17% mass on ~40 experts. Q1's real value is the confirmation that top-3 covers
**~67%** of routed mass — relevant to *how many* experts a coverage-mask must
keep, not to a fixed-expert precision pin.

### Key numbers
- Per-token top-1 mass share: **30.5%** (top-2 51%, top-3 67%) — layer-uniform.
- Cross-task top-1 identity overlap: **2.5%**; within-coding **7.8%**.
- Static single-pin coverage: **5.7%** of layer mass (live-session ceiling 16.7%).
- Cost: top-1 pin @ Q4 = 589 MiB = **21.4%** of the usable cache.

### Reproduce
```
python scripts/analyze_top_expert_mass.py \
  --reap-loop-root <reap-loop> \
  --moe-root <moe-aggressive-commit>/.claude/worktrees/elastic-bose-6ae1c7
```
