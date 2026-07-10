# T4 / T5 runbook — freeze-safe W-sweep & weighted-vs-unit mask A/B

Turn-key offline prep for the two decisional tests of Fase 2
(`docs/NEXT_STEPS_PLAN_20260710.md`). Everything here is *ready to launch*; none
of it has been run on GPU. Run on a pod (pure Linux) or on the local 3060 from
**inside WSL**. Do not run the phase-2 masked pass without the phase-1 freeze
(that is the whole point — see ledger note **J44**).

## What each test decides

- **T4** — *W-sweep with a safe freeze point.* The old cache1024 W-table
  (W=50/130 clean, W=80/110/150 broken) was a **lottery of the cut point**: the
  phase-2 re-prefill `[instruction] + [partial HTML]` restarted the document
  (`<!DOCTYPE html>` again) whenever W truncated the prefix *inside* a CSS
  declaration. Freezing phase-1 at a structural boundary (`}` / `;` / `>` /
  blank line) before re-prefill removes the confound. **Verdict:** does the W
  scale become **monotone** (rising L0-L3 with W) once the freeze is safe, or
  does it **flatten** — proving the old table was the freeze-point lottery?
- **T5** — *weighted OFFLINE vs unit in-engine.* The historical good recipe
  ranked experts by cumulative **gate mass** (`build_session_mask.py`,
  reconstructed from the raw router trace, two-phase). In-engine PACE relearn
  effectively ranks by **unit count**. **Verdict:** at equal W and equal
  seed-batch, does weighted-offline beat unit-in-engine on median L0-L3? That
  tells the relearn path what it should actually compute.

## Prerequisites

- `ds4` runtime binary with patch **0011 v2** (runtime `DS4_REAP_MASK_FILE`) and
  routing-weight trace (`DS4_SPEX_TRACE_ROUTING` + `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`).
  Replay binary SHA256 `8746a873…9e2d74` (RUN_META of the pod replay).
- `ds4-2bit.gguf` model.
- Python 3 on the run host (pod/WSL) for the scripts; no extra deps.
- Prompt: default is the recovered compact coffee-shop prompt
  `runs/ds4/20260710_pod_cache1024_warmup_replay/frontpage_prompt.txt` (819 B).
- Grading uses `scripts/functional_grade.py` (imported automatically). `node` on
  PATH sharpens the JS-syntax check; otherwise a heuristic is used.

## Scripts (all new, offline-tested)

| Script | Role |
| --- | --- |
| `scripts/freeze_boundary.py` | `find_safe_freeze_point(text, W, tokenizer_len_fn=None)` → nearest safe cut ≤ W. Unit-tested on the J44 pathologies (`tests/test_freeze_boundary.py`). |
| `scripts/build_session_mask_canonical.py` | Canonical builder, `--mode weighted|unit`. Emits the runtime pruned-pair `.txt` (`DS4_REAP_MASK_FILE`, format of patch 0011) **and** a keep-list `.json` sidecar. Weighted mode reproduces the pod's `sess_W50.txt` byte-for-byte. Tests: `tests/test_build_session_mask_canonical.py`. |
| `scripts/run_w_sweep_freeze_safe.py` | T4 harness: two-phase per W×run, freeze between phases, grade, per-seed CSV + median + verdict. Tests: `tests/test_w_sweep_freeze_safe.py`. |

Reused from the pod replay (`runs/ds4/20260710_pod_cache1024_warmup_replay/`):
the two-phase env/flag recipe, the gate-mass mask builder, the
`ds4: prefill: X t/s, generation: Y t/s` diag line, the
`deliverable = frozen_phase1 + phase2` assembly, and the prompt.

## T4 — run it

### On a pod (Linux)

```bash
cd reap-loop
python scripts/run_w_sweep_freeze_safe.py \
  --binary /root/ds4/ds4 \
  --model  /root/models/ds4-2bit.gguf \
  --cache 1024 --ctx-p1 2048 --ctx-p2 3072 \
  --w-values 30,50,70,90,110,130,150 --runs 3 \
  --mask-mode weighted
# output: runs/ds4/<date>_w_sweep_freeze_safe/
```

### On the local 3060 (from inside WSL, GPU free)

```bash
# smaller cache fits the 12GB card; keep n=3 for the ±50% noise floor
python scripts/run_w_sweep_freeze_safe.py \
  --binary <wsl-path-to>/ds4 --model <wsl-path-to>/ds4-2bit.gguf \
  --cache 256 --ctx-p1 2048 --ctx-p2 3072 \
  --w-values 30,50,70,90,110,130,150 --runs 3 --mask-mode weighted
```

Preview the plan without running anything (safe anywhere, no GPU/WSL touched):

```bash
python scripts/run_w_sweep_freeze_safe.py --binary ds4 --model m.gguf --dry-run
```

Knobs: `--headroom` (extra phase-1 tokens so the freeze lands *on* a boundary at
or below W, default 16); `--total` (phase1+phase2 budget, default 1000);
`--keep-k` (default 23); `--temp`/`--seed-base` (at `--temp 0` the 3 runs measure
timing variance only — greedy quality is deterministic; set `--temp>0` to vary
generations, which passes `--seed` per run); `--port` (recorded in the manifest
for server-mode parity, but the executed path is CLI-direct).

## T5 — run it

Same harness, flip the ranking mode, keep W, seeds, and everything else fixed so
it is a single-variable A/B:

```bash
# arm A — weighted OFFLINE (gate mass)
python scripts/run_w_sweep_freeze_safe.py --binary ... --model ... \
  --w-values 50,90,130 --runs 3 --mask-mode weighted \
  --outdir runs/ds4/<date>_t5_weighted

# arm B — unit in-engine proxy (selection count)
python scripts/run_w_sweep_freeze_safe.py --binary ... --model ... \
  --w-values 50,90,130 --runs 3 --mask-mode unit \
  --outdir runs/ds4/<date>_t5_unit
```

Build a single mask by hand (either arm) from an existing phase-1 trace:

```bash
python scripts/build_session_mask_canonical.py route.csv sess.txt 23 --mode weighted
#   -> sess.txt  (DS4_REAP_MASK_FILE, pruned pairs)
#   -> sess.json (keep-list sidecar, --mask-load / catalogue)
```

## What to measure

Per `(W, run)` the harness records to `summary.csv`:

- `freeze_boundary` (`}`/`;`/`>`/`blankline`/`none`) and `freeze_within_target`
  — confirms the cut was safe and ≤ W.
- `p1_gen_tps`, `p2_prefill_tps`, `p2_gen_tps` — from the diag lines.
- `l0l3` — render grade of `deliverable.html` (`frozen_phase1 + phase2`).
- `restart`, `doctype`, `html_close`, `form`, `script`, `alert_in_script`,
  `repeat`, `button_wired`, `form_wired`, `tag_mismatch`.

`summary_median.csv` gives, per W, the **median** `p2_gen_tps` and `l0l3`, and a
majority vote for `repeat`/`restart` (report the median, never a single run —
3060 noise ±50%). `VERDICT.txt` states whether median L0-L3 is monotone in W.

## Verdict criteria

- **T4 — the W-table is real** iff median `l0l3` is **non-decreasing** across
  ascending W (`VERDICT.txt: monotone_non_decreasing = True`) with a level spread
  ≥ 2, and `restart_majority = 0` at every W. **The old table was the
  freeze-point lottery** iff, once the freeze is safe, the spread collapses
  (levels flat, e.g. all L2-L3) and no W restarts — i.e. safe framing erases the
  W=80/110/150 failures. Cross-check: `freeze_boundary ∈ {} ; > blankline` and
  never `none` at any cell.
- **T5 — weighted beats unit** iff, at matched W and seed-batch, median `l0l3` in
  the `weighted` arm is **strictly higher** at ≥1 W and never lower, with
  `p2_gen_tps` within the noise band (weighted must not buy quality with a large
  speed regression — J45/J46 warned weighted warmup was *slower and degraded
  earlier* in-engine, so a genuine offline win here is the reconcile). If they
  tie, prefer `unit` (cheaper to compute in the relearn path) and record that the
  in-engine drift was harmless.

## Guardrails

- Do not edit `scripts/run_ds4_exchange_matrix.py`, the Codex `runs/` of
  2026-07-10, the master ledger, or `CLAIMS_CURRENT.md` — these scripts only add
  new files.
- `git pull --rebase` before pushing (Codex works M1 in parallel).
- Numbers from cache1024/warm-RAM pods are upper bounds; 3060 numbers are the
  headline per `DS4_RUNNER_PROTOCOL.md`. Keep routing trace **on** in T4/T5
  (phase-1 needs it) — this is a diagnostic run, not a SOTA timing row.
