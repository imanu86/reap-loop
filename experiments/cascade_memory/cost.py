"""Deterministic cost VECTOR for the confidence-cascade harness.

Spec (CONFIDENCE_CASCADE_MEASUREMENT_SPEC.md §1):
    Cost = (prefill_tokens, decode_tokens, n_embed_calls, n_rag_calls,
            n_llm_calls, wall_clock_s)
    Primary = token + call counts (from API `usage`, reproducible).
    Latency (wall_clock_s) is SECONDARY (machine-dependent), logged but not the
    headline axis.

This module only ACCUMULATES cost. Averaging over the correct denominator
(N_total_items, never N_resolved) is the analysis layer's job — enforced there.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields


@dataclass
class CostVector:
    prefill_tokens: int = 0        # = usage.prompt_tokens
    decode_tokens: int = 0         # = usage.completion_tokens
    n_embed_calls: int = 0         # embedding requests (rung-1, Step 2)
    n_rag_calls: int = 0           # retrieval requests (BM25/e5 over memory)
    n_llm_calls: int = 0           # chat/completions requests
    wall_clock_s: float = 0.0      # SECONDARY: latency, machine-dependent

    def __add__(self, other: "CostVector") -> "CostVector":
        return CostVector(**{f.name: getattr(self, f.name) + getattr(other, f.name)
                             for f in fields(self)})

    def __radd__(self, other):
        # supports sum([...])
        if other == 0:
            return self
        return self.__add__(other)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def total_tokens(self) -> int:
        """PRIMARY scalar cost axis: raw LLM tokens, fully deterministic."""
        return self.prefill_tokens + self.decode_tokens

    def scalarize(self, weights: dict) -> float:
        """Weighted token-equivalent cost. Default weights (config.COST_WEIGHTS)
        collapse this to total_tokens; non-zero call surcharges are opt-in and
        must be disclosed when used in a headline number."""
        return (
            self.prefill_tokens * weights.get("prefill_tokens", 1.0)
            + self.decode_tokens * weights.get("decode_tokens", 1.0)
            + self.n_embed_calls * weights.get("n_embed_calls", 0.0)
            + self.n_rag_calls * weights.get("n_rag_calls", 0.0)
            + self.n_llm_calls * weights.get("n_llm_calls", 0.0)
        )


def from_usage(usage: dict) -> CostVector:
    """Build a CostVector from an OpenAI-compatible `usage` object.
    Missing fields default to 0 (some servers omit them); we count this LLM call.
    Token counts are read from the server, never estimated locally — that keeps
    the primary cost axis reproducible and non-gameable."""
    usage = usage or {}
    return CostVector(
        prefill_tokens=int(usage.get("prompt_tokens", 0) or 0),
        decode_tokens=int(usage.get("completion_tokens", 0) or 0),
        n_llm_calls=1,
    )
