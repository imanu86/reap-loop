# REAP-LOOP

> **Status: work-in-progress research. Numbers here are PRELIMINARY and several parts are unvalidated.**
> This repository is a research artifact, not a product. Read the caveats section before quoting any figure.

**REAP-LOOP is a reversible gate-bias that live-calibrates the expert working-set of a
Mixture-of-Experts (MoE) model *during the session itself*.**

Instead of pruning a MoE offline for a fixed domain, REAP-LOOP:

1. **Starts at the full model** (`K0` — no experts masked), so the very first tokens run at full quality.
2. **Watches which experts the running session actually uses** (via the router's own signals — no
   extra model, no separate predictor).
3. **Progressively narrows** the resident working-set toward an aggressive keep-level (e.g. keep-9%),
   biasing the gate away from experts the session is not using.
4. **Reverses when it must**: the bias is a *mask over the gate*, not surgery on the weights. Expert
   IDs are preserved, so the model can widen back out (a "breath") the moment the router shows drift.

Because the working-set is learned **from the live session**, REAP-LOOP adapts to **any task**. It is
**not** domain-pretrained and it does not require a fine-tune. A catalogue of pre-learned working-sets
(per output-type) is an **optional pre-warm** — a way to skip the warm-up cost on a familiar workload —
never a prerequisite.

The direction this points at is **PACE — a Perceptive Adaptive Control Engine: a self-calibrating controller inside the inference engine**:
`keep-level = f(routing-drift)`. Drift low → tighten; drift rising → widen proportionally. Dynamic MoE
sparsification framed as a *control problem*, with the router itself as the drift sensor.

---

## Built on ds4 / DwarfStar (credit in evidence)

REAP-LOOP is built **on top of [ds4 / DwarfStar](https://github.com/antirez/ds4)** by
**Salvatore Sanfilippo (antirez)** — the MoE-streaming inference engine is the foundation this work
stands on. ds4/DwarfStar is MIT-licensed; all credit for the underlying engine, the SSD/host-side
expert streaming, and the CUDA/Metal MoE paths belongs to it.

Everything in `patches/` is a set of diffs *against* ds4 — this project does not reimplement the engine,
it extends it. A verbatim copy of the upstream README and the relevant paper are kept under
`docs/references/` for attribution and context.

---

## Companion mechanisms

- **SPEX (Speculative Prefetch of EXperts)** — predicts *which experts to page in next*, hiding
  expert-load latency behind compute. SPEX predicts the **loading**, never the **gating**: the real
  router always decides, so accuracy is preserved by construction. See `docs/SPEX_spec.md`,
  `docs/SPEX_LOOP.md`, and `src/msc/spex/`.
- **WRAP — Working-set Resident Aggregate Prefetch** (bulk page-in) — a host-side bulk expert page-in path that batches the transfer
  of a working-set into resident memory in one shot, rather than fetching expert-by-expert on demand.
- Together with REAP-LOOP these are three composable levers: **warm + cache + REAP-loop (+ SPEX)**.
  Because REAP-LOOP acts in expert-**ID space** (a gate bias, not surgery), the SPEX/prefetch metadata
  stays valid across mask changes — the levers compose without re-dumping.

---

## Repository layout

| Path | What |
|---|---|
| `patches/ds4/` | Diffs against ds4/DwarfStar: SPEX prefetch stages, routing/hidden trace capture, MTP streaming, the bulk page-in path, plus a small C predictor (`ds4_spex_predict.{c,h}`). |
| `src/msc/` | Research code: `spex/` (predictor + speed sims), `dspark/` (scheduler/STS sim), the working-set estimator, residency manager, policies, router instrumentation, validators, and the metrics/report pipeline. |
| `tools/cockpit/` | A small self-contained HTML cockpit UI for the ds4 backend. |
| `scripts/` | Research scripts: warm-up tracing, working-set estimation, coverage/saturation curves, REAP saliency/bias-mask builders, benchmarks, and report generation. |
| `configs/` | Model specs (OLMoE, Granite-MoE) and policy/grid configs for the accuracy-vs-VRAM sweep. |
| `tests/` | Unit tests for the code above. |
| `docs/` | Design notes, novelty/prior-art analysis, an experiments ledger, and the SPEX/REAP specs. `docs/paper/` is intentionally empty here — the paper is added by a separate process. |

---

## Caveats (read this)

- **The numbers are PRELIMINARY.** Much of the measured speed-up is **confounded by cold-vs-warm cache
  effects** and has not been isolated with a clean A/B/A protocol. Treat every t/s and every
  accuracy-drop figure as indicative, not final.
- **Many parts are WIP and unvalidated.** PACE, the self-calibrating controller (drift-driven keep-level), is
  a research direction, not a shipped algorithm; the fixed keep-level schedules in the ledger are the
  datapoints that would *map* the response curve, not the final controller.
- **The prior-art analysis is deliberately conservative.** No single building block is claimed as new.
  What is defensible is the *reversible closed-loop composition* (mask actuator, zero-violation), a
  *paired rand/reap contrast* (1.345x, CI[1.270, 1.423], same GPU, confound-clean), and static keep-23
  as a **speed diagnostic** (11-17 t/s on a pod 3090, **not** generalized). The previously claimed
  "causal asymmetry in pruning" is **RETRACTED** — it did not replicate under multiseed (N=3) and was a
  mask-inert + n=1 artifact. For the current, canonical status of every claim, see
  [`docs/CLAIMS_CURRENT.md`](docs/CLAIMS_CURRENT.md) (single source of truth), plus
  `docs/REAP_LOOP_NOVELTY.md` and `docs/PRIOR_ART.md`.

## License

MIT — see [`LICENSE`](LICENSE). Copyright 2026 REAP-LOOP contributors.
The upstream ds4/DwarfStar engine that this work extends is MIT-licensed by its author; see
`docs/references/` for the upstream material.
