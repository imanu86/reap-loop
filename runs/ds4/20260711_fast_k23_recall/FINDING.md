# 2026-07-11 Recall: "K23 + cache256, sito parziale e veloce" — NOT FOUND as a single run

Read-only forensic pass over `runs/ds4/`, `docs/EXPERIMENTS_LEDGER.md`,
`docs/DS4_EXPERIMENT_LEDGER_20260710.md`, `docs/CLAIMS_CURRENT.md`,
`docs/INTUITION_ARCHAEOLOGY_20260711.md`, `docs/LEVER_RETROSPECTIVE_20260711.md`,
`git log --all`, and Codex transcripts under `~/.codex/sessions/2026/07/`.

## Verdict

**No single run in this repo combines K23 + cache256 + "altissime" speed +
partial/functional quality.** The two axes the user remembers ("K23+cache256"
and "velocità altissime + sito parziale funzionante") belong to **two
different, mutually exclusive experiment families**:

1. Every measured **K23 + cache256** run (local 3060 *and* pod) tops out at
   **~2.4–3.14 t/s avg**. Nothing at cache256 ever reaches "altissime" speed.
2. Every run that reaches "altissime" speed (13–17+ t/s) does so with
   **cache1024** (never cache256), and is **always on a pod GPU** (RTX
   3090/3080Ti/4070Ti), never on the local 3060.

The single artifact that best matches "parziale e in parte funzionante a
velocità altissime" is a **cache1024 pod** run, not cache256, detailed below.

## (a) All measured K23 + cache256 runs (exact numbers)

Source: `runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv` (master
ledger, `pace_cache_target`/`server_cache_experts` = 256, variant contains
`k23`), cross-checked against the underlying `summary.csv` files.

| run_id | variant | hardware | avg_tps | first50 | last_chunk | completion_tokens | doctype/popup/form/script | repeat_flag | verdict |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| html_local_breath_k96_return_k23_cache256_r01 | local_breath_k96_return_k23_cache256 | local 3060 | **3.14** | 2.39 | 3.21 | 800 | 1/1/0/0 | 1 | (loop) |
| html_local_k23_cache256_r01 | local_k23_cache256 | local 3060 | 3.06 / 2.98 (2 runs) | 2.4–2.85 | 3.01–3.04 | 800 | 1/1/0/0 | 1 | CSS-reset loop, no `</html>`/`<form>`/`<script>` |
| html_w50_rotate32_k23_cache256_html4000_r01 | w50_rotate32_k23_cache256_html4000 | local 3060 | 2.79 | 1.06 | 2.90 | 4000 | 1/1/0/0 | 0 | `diagnostic_ctx4096_edge_css_loop_no_body` |
| html_w100_rotate32_k23_cache256_html4000_ctx8192_r01 | w100_rotate32_k23_cache256_html4000_ctx8192 | local 3060 | 2.78 | 0.93 | 2.99 | 4000 | 1/1/1/1 | 1 | `script_comment_loop_no_html_close` |
| html_local_k23_rotate32_cache256_r01 | local_k23_rotate32_cache256 | local 3060 | 2.61 | 2.32 | 2.89 | 800 | 1/1/0/0 | 0 | 23 rotations, no triple-repeat loop through 800 tok |
| html_w50_rotate32_k23_cache256_r01 | w50_rotate32_k23_cache256 | local 3060 | **2.59** | 0.93 | 2.88 | 2000 | 1/1/0/0 | **0** | **`visually_renderable_token_budget_limited_w50_no_loop`** — best local "partial/functional" signal at cache256 |
| html_local_k23_weighted_warmup_cache256_r01 | local_k23_weighted_warmup_cache256 | local 3060 | 2.57 | 2.32 | 2.54 | 800 | 1/1/0/0 | 1 | selector-list loop (215× `h6`) |
| html_w100_direct_k23_cache256_r01 | w100_direct_k23_cache256 | local 3060 | 2.55 | 0.66 | 3.08 | 2000 | 1/1/0/0 | 1 | `fail_quality_loop_early` |
| html_w100_rotate32_k23_cache256_r01 | w100_rotate32_k23_cache256 | local 3060 | 2.43 | 0.78 | 2.70 | 2000 | 1/1/0/0 | 0 | `visually_renderable...needs_more_tokens` |
| html_local_breath_k0_return_k23_cache256_r01 | local_breath_k0_return_k23_cache256 | local 3060 | 2.36 | 2.32 | 3.03 | 800 | 1/1/0/0 | 1 | loop |
| s5_coffee_k23 r00/r01/r02 (`20260710_pod_static_ab_ctx8192/s5_coffee_k23`) | static K23, coffee prompt, **cache=256**, ctx3072 | **pod RTX 3090** | **1.28 / 1.32 / 1.29** | — | — | ~1200 budget | L2/L1/L2 (functional-grade scale) | r00,r02 have `</html>=1`, form+button wired; r01 loops | Even the deliberate pod K23+cache256 run is *slower* than local, not faster. |

Local reference floor (no-mask/full or other K at cache256/258, for scale):
plain cache sweep (`20260709_local_cache_sweep_k23_RESULTS.md`) shows cache128
peaking at 3.34 t/s and cache258 at 3.23 t/s under the same K23 — so 3.1–3.3
t/s is roughly the **ceiling for any K23+~256-cache local config**, not an
outlier low number.

**Direct quote confirming this was checked and failed to reproduce a faster
"old" result:** `runs/ds4/20260710_direct_k23_vs_stepdown_html800/DIRECT_PARTIAL_SUMMARY.md`:
> "Conclusion: in this cache256 reproduction, direct K23 is clearly faster,
> but it does not reproduce better quality. The old good direct result likely
> depended on different conditions such as cache size, exact prompt/run
> boundary, or the previous session-learned weighted mask."

## (b)/(c) The actual "fast + partial-functional K23" artifact — cache**1024**, not cache256

`runs/ds4/20260710_pod_cache1024_warmup_replay/README.md` + `docs/EXPERIMENTS_LEDGER.md`
("2026-07-10 Pod Cache1024 Follow-up" section):

| Phase | keep | env | Result |
|---|---|---|---|
| Direct K23 (no session mask), cache1024 | 23 | `local_k23_cache1024` | wall 94.576s, **avg 14.12 t/s**, last-chunk 24.79 t/s, 800 tok — but `repeat_flag=1`, no `</html>`/`<form>`/`<script>` (invalid) |
| Direct K23, weighted warmup, cache1024 | 23 | `local_k23_weighted_warmup_cache1024` | wall 79.577s, **avg 16.37 t/s**, last-chunk 24.71 t/s — fastest, but also `repeat_flag=1`, invalid |
| **Two-phase session-learned mask, W=50, cache1024** | 23 (built from gate-mass over the W50 wide-warmup trace) | `20260710_pod_cache1024_warmup_replay/W50` | Phase1 (wide/K0) 2.03 t/s → **Phase2 (frozen K23) 14.60 t/s**. Quality: `doctype=2, </html>=1, <form>=1, <script>=1, alert=2, repeat=0` — **"functionally complete-ish and no repeat"**, deliverable at `runs/ds4/20260710_pod_cache1024_warmup_replay/W50/deliverable_W50.html` (one restart/duplicate doctype, imperfect JS, so graded L2/L3, not clean L3). |
| Two-phase session-learned mask, W=130, cache1024 | 23 | `.../W130` | Phase2 16.24 t/s, but `</html>=0, repeat=1` — loops on `document.addEventListener("DOM"...` |

**This W50 two-phase run is the best-documented "fast AND partial/functional"
K23 artifact in the repo.** The env that distinguishes it from the failed
cache256 direct-K23 reproductions is **not a RAM/leva knob** — it is the
**two-phase session-learning recipe**: Phase 1 runs W=50 tokens wide
(K0/full router, routing+weights trace on) to *observe* which experts the
model actually uses for this specific prompt; a keep-23 mask is then built
from that trace by cumulative gate-mass (`build_session_mask.py ... 23`); the
prompt + phase-1 output are re-prefilled and generation continues **frozen**
on that session-learned mask for the remaining budget. This is different from
every cache256 run above, which all use "direct K23" (either a hand-set mask
from token 0/50, or the runtime PACE controller's own descent/tighten
schedule) — no live gate-mass-ranked session mask.

Cross-reference in `docs/CLAIMS_CURRENT.md` (row `SESSION-LEARNING riscatta il
cold-collapse`) and `docs/EXPERIMENTS_LEDGER.md` row
`HIST-W50-W130-SESSION-CACHE1024-20260707`: the *original* claim this replay
was chasing is dated **2026-07-07**, recovered from `docs/CLAIMS_CURRENT.md` /
`docs/paper/PAPER.md` / Claude session artifacts, not from a raw reap-loop
run — "old raw not found; freeze-point sensitive; new replay narrows the
claim." A sibling historical claim (`HIST-CACHE1024-STATIC-K23-20260707`)
even cites a **static** file-mask keep-23 at **17.3 t/s, hit-rate 0.986**,
also cache1024/pod, also "raw summary not found in current tree."

So: the user's memory of "K23 + very high speed + partially working site" is
real and matches this cache1024/pod, two-phase-session-mask lineage almost
exactly on the numbers (13.6–17.3 t/s peaks were quoted historically; the
2026-07-10 replay reproduced 14.60 t/s cleanly for W50). The "256" in memory
does not match any artifact — every fast K23 number in this codebase is paired
with **cache1024**, and every genuine cache256 K23 number caps at ~3.1 t/s.

## (d) Pod vs local — explicit for every number above

- **Local RTX 3060, real t/s** (all K23+cache256 in section (a) except the s5 row): **2.36–3.14 t/s**, i.e. NOT "altissime" by any measure — this is the normal/slow local range for this whole experiment matrix.
- **Pod, t/s don't count per your own rule** (RTX 3090, cache1024, section (b), and the s5_coffee_k23 cache256 pod row in (a)): 1.28–16.37 t/s. The "altissime" 14–16 t/s numbers are 100% pod/cache1024.
- No cache256 pod run was ever fast either (s5_coffee_k23, cache256, pod 3090: 1.28–1.32 t/s — slower than local cache256).

## (e) "HTML di confronto" side-by-side — not found

No file matching a genuine multi-config side-by-side comparison page was
found anywhere in `runs/ds4/`, `runs/reap/`, `tools/`, or `scripts/` (searched
for `compare`, `side_by_side`, `confronto`, and scanned all `*.html` under
`runs/`). What exists instead:

- **Tabular "confronto" artifacts** (not HTML): `docs/EXPERIMENTS_LEDGER.md`
  (running markdown log, K23+cache256 section at "2026-07-10 Direct K23 vs
  K64->K23 Reproduction" and "2026-07-10 Pod Cache1024 Follow-up"); the master
  CSV `runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv` (305
  rows, every run/variant/K/cache/tps/quality-flag in one table — this is
  almost certainly the "TABELLA con tutti i risultati" you remember).
- **Individual rendered HTML** (not side-by-side, one file per run):
  `content_measured_render.html` next to each `content_measured.txt` in the
  cache256 run folders, plus the two standalone deliverables
  `runs/ds4/20260710_pod_cache1024_warmup_replay/W50/deliverable_W50.html`
  and `.../W130/deliverable_W130.html` (the cache1024 pod pair described
  above — these are real, openable, partially-functional HTML pages).
- A separate, unrelated multi-file HTML gallery exists at
  `runs/reap/k91_coding_vram/sites/{full,k50,k91,k96}.html` — but that is the
  **K91-coding-VRAM track** (mask name "K91" = keep 9%, not keep-23; cache
  64/380, pod 3080Ti), already closed/retracted in your own memory index
  ("K91 non entra in 12GB reali E non codifica"). It is not the K23+cache256
  run you're recalling; flagged here only so it isn't confused with the
  target.

## Summary of what to trust

- If "K23 + cache256" is the hard constraint: the fastest **and** cleanest
  local match is `w50_rotate32_k23_cache256` (`20260710_w50_rotate32_k23_cache256_html2000`,
  local 3060, **2.59 t/s**, `repeat=0`, verdict
  `visually_renderable_token_budget_limited_w50_no_loop`) — moderate speed,
  genuinely partial/no-loop.
- If "altissime speed + partial/functional K23 site" is the hard constraint:
  it is `20260710_pod_cache1024_warmup_replay/W50` — **14.60 t/s**, pod RTX
  3090, **cache1024**, two-phase session-learned mask, `repeat=0`,
  `doctype=2/</html>=1/<form>=1/<script>=1/alert=2`, deliverable HTML on
  disk. This is almost certainly the run being remembered, with "cache 256"
  as the misremembered detail (it was cache**1024**).
