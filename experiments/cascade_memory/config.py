"""Central configuration for the Confidence-Cascade Memory harness.

Everything is model-agnostic: the LLM is reached over an OpenAI-compatible
`/v1/chat/completions` endpoint whose `base_url` + `model` are configurable via
env vars or CLI. NOTHING here is ds4-specific.

Design invariants (from docs/CONFIDENCE_CASCADE_MEASUREMENT_SPEC.md):
  * Cost is a deterministic VECTOR, not a scalar. See cost.py.
  * The denominator for any average is ALWAYS N_total_items (never N_resolved).
  * The confidence sensor is PRE-REGISTERED before AUROC is inspected (anti-gaming).
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Endpoint (OpenAI-compatible). Override via env or CLI. base_url must include
# the /v1 suffix, e.g. http://localhost:8080/v1  (llama.cpp, vLLM, LM Studio...).
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("CASCADE_BASE_URL", "http://localhost:8080/v1")
MODEL = os.environ.get("CASCADE_MODEL", "local-model")
API_KEY = os.environ.get("CASCADE_API_KEY", "")  # only needed for remote APIs
REQUEST_TIMEOUT_S = float(os.environ.get("CASCADE_TIMEOUT_S", "120"))

# Deterministic decoding for reproducibility. temperature=0 + fixed seed.
GEN_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 48,          # short-answer format → clean logprob span + exact-match
    "seed": 1234,
    "logprobs": True,          # FLARE-style sensor needs token logprobs
    "top_logprobs": 5,
}

# ---------------------------------------------------------------------------
# PRE-REGISTERED confidence sensor (frozen BEFORE looking at any AUROC).
# Primary = mean token log-probability of the generated (short) answer span.
# All other signals (min_logprob, entropy, verbalized, schema_format) are logged
# as EXPLORATORY ONLY and MUST NOT be used to decide Gate #1. Choosing the
# best-separating signal post-hoc would be p-hacking (spec §4: "nessuna metrica
# scelta post-hoc").
# ---------------------------------------------------------------------------
SENSOR_PRIMARY = "mean_logprob"

# ---------------------------------------------------------------------------
# Cost scalarization for the Pareto frontier. The PRIMARY reported cost axis is
# raw total LLM tokens (prefill+decode) — fully deterministic, from API `usage`.
# Extra-call axes (embed/rag/llm counts) are reported SEPARATELY as their own
# vector components; converting them to a token-equivalent needs assumptions, so
# the surcharges below default to 0 and the scalar == total tokens. They exist
# only so a sensitivity analysis can price calls without editing code. Any
# non-zero weight used in a headline number must be disclosed.
# ---------------------------------------------------------------------------
COST_WEIGHTS = {
    "prefill_tokens": 1.0,
    "decode_tokens": 1.0,
    "n_embed_calls": 0.0,      # token-equivalent surcharge per embed call
    "n_rag_calls": 0.0,        # token-equivalent surcharge per RAG retrieval
    "n_llm_calls": 0.0,        # token-equivalent surcharge per LLM call
}

# ---------------------------------------------------------------------------
# Reuse (read-only until the cascade is built). BM25 index from an external KB project is
# imported via sys.path in retrieval_adapter.py. e5 embed is deferred to rung-1.
# ---------------------------------------------------------------------------
EXTERNAL_RETRIEVAL_DIR = os.environ.get(
    "CASCADE_EXTERNAL_RETRIEVAL_DIR",
    r"/path/to/retrieval",
)

# Production kill-switch mirror: the cascade controller (Step 2) lives behind
# USE_CONFIDENCE_CASCADE, OFF by default. Step-0/1 code never reads it.
USE_CONFIDENCE_CASCADE = os.environ.get("USE_CONFIDENCE_CASCADE", "0") == "1"

# Output location for per-item JSONL logs.
RUNS_DIR = os.environ.get(
    "CASCADE_RUNS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
)
