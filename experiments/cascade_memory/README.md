# Confidence-Cascade Memory — measurement harness (Step 0 + Step 1)

Phase-2 implementation of the concept in
[`../../docs/CONFIDENCE_CASCADE_MEMORY.md`](../../docs/CONFIDENCE_CASCADE_MEMORY.md),
following the binding spec
[`../../docs/CONFIDENCE_CASCADE_MEASUREMENT_SPEC.md`](../../docs/CONFIDENCE_CASCADE_MEASUREMENT_SPEC.md).

**Non-negotiable principle: measure FIRST, build the cascade AFTER, and only if the
pre-registered go/no-go gates pass.** This directory is Step 0 (scaffold) + Step 1
(synthetic generator + baselines B0/B1 + Gate #1 measurement). **There is no cascade
controller here** — it is not written until Gate #1 is GO.

## What is (and is not) in here
| File | Role |
|------|------|
| `config.py` | endpoint (`base_url`+`model`), pre-registered sensor, cost weights |
| `llm_client.py` | OpenAI-compat adapter (stdlib urllib) + offline mock backend |
| `cost.py` | deterministic cost **vector** `(prefill,decode,n_embed,n_rag,n_llm,wall)` |
| `confidence.py` | FLARE-style sensor (mean-logprob primary; schema fallback) |
| `scorer.py` | exact-match + abstention (separate) + anon LLM-judge hook |
| `logger.py` | per-item JSONL (every attempt logged → denominator = N_total) |
| `synth_gen.py` | fictitious-universe generator, distance grid {5,20,40,60,80} |
| `retrieval_adapter.py` | BM25 over conversation turns (reuses an external `BM25Index`, has fallback) |
| `arms.py` | **B0** no-memory, **B1** always-RAG (only these two in Step 1) |
| `run.py` | run an arm over a dataset → JSONL |
| `gate1.py` | AUROC(confidence,correct) + baseline cost/accuracy frontier + verdict |

## Design invariants (anti-gaming, from the spec)
- **Denominator is always `N_total_items`** — the cost of failures stays in the average.
  "Cost conditioned on success" is survivorship bias and is forbidden.
- **Cost is a deterministic vector**, token/call counts read from the API `usage`
  (reproducible); wall-clock is secondary.
- **The confidence sensor is pre-registered** (`SENSOR_PRIMARY = mean_logprob`) before
  any AUROC is inspected. Other signals are logged exploratory-only.
- **Fictitious universe** so answers can't leak from pretraining; a `--leak-control`
  twin (plant removed) must score ~chance.
- **Abstention is a separate metric**, never counted as correct nor folded into cost.

## Honest expected result (pre-registered substrate)
The concept can win on **MHA/GQA** backbones (dense KV = real cost). It is expected to
be **inert on MLA/ds4** (KV already a tiny latent → rung-0 collapses into text
re-inject = recall-reinject). The MLA negative is a deliverable, not a failure.

## Quick start
```bash
# 0) offline plumbing smoke (no network, no model): proves the cost vector logs
python synth_gen.py --out data/smoke.jsonl --n-per-cell 3 --leak-control
python run.py --dataset data/smoke.jsonl --arm all --base-url mock:oracle_decay
python gate1.py --runs "runs/*_B0_no_memory.jsonl"

# 1) real Gate-1 measurement (needs an OpenAI-compat endpoint that returns logprobs)
python synth_gen.py --out data/synth_full.jsonl --n-per-cell 300 --leak-control
export CASCADE_BASE_URL=http://localhost:8080/v1   # llama.cpp / vLLM / LM Studio
export CASCADE_MODEL=your-model-id
python run.py --dataset data/synth_full.jsonl --arm all
python gate1.py --runs "runs/*_B0_no_memory.jsonl" "runs/*_B1_always_rag.jsonl"
```
`base_url=mock:echo` (flat confidence) and `mock:oracle_decay` (synthetic
distance-decayed correctness/confidence) run fully offline; **a mock AUROC is a
plumbing fixture, never a real Gate-1 result.**

## Gate #1 (pre-registered)
- **GO** if `AUROC(confidence,correctness) ≥ 0.65` on the baseline arm → proceed to Step 2 (cascade).
- **NO-GO** if `< 0.60` → STOP, report the honest negative (sensor can't discriminate).
- `[0.60, 0.65)` → grey / inconclusive.

Reuse is read-only until the cascade is built: `an external KB project/retrieval/bm25.py`
(imported for B1), `local_runner_v2.py:confidence_low` (studied for the schema
fallback). New flag `USE_CONFIDENCE_CASCADE` (OFF) is reserved for Step 2.
