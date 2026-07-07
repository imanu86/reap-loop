"""Measurement arms. Step 1 implements the two mandatory baselines:

  B0  no-memory   — full conversation in the native context, NO retrieval. Isolates
                    the backbone's own attention-decay (lost-in-the-middle) as
                    distance grows. Large prefill.
  B1  always-RAG  — retrieve the top-k most relevant PAST TURNS (BM25 over history)
                    and inject only those; no full history. Small prefill + 1 RAG call.

B2 (reactive-binary), B3 (RANDOM at equal budget), B4 (cascade), B5 (cascade-rung0)
are Step 2 and are NOT implemented here — the cascade is not built until Gate #1 GO.
Every arm returns {result, cost, arm, meta}; scoring/logging happen in run.py.
"""
from __future__ import annotations

from llm_client import chat
from cost import from_usage
from retrieval_adapter import MemoryRetriever

SYSTEM = (
    "You are a memory assistant. Answer the final question using ONLY information "
    "stated in the conversation. Reply with just the value, no explanation. "
    "If the information is not present, reply exactly: I don't know."
)


def _mock_meta(item, effective_distance=None):
    return {
        "answer": item["answer"],
        "distance": item["distance"] if effective_distance is None else effective_distance,
        "item_id": item["item_id"],
    }


def run_b0(item, base_url, model, **_):
    history = item["messages"]
    msgs = [{"role": "system", "content": SYSTEM}] + history + \
           [{"role": "user", "content": item["question"]}]
    res = chat(msgs, base_url=base_url, model=model, mock_meta=_mock_meta(item))
    cost = from_usage(res.usage)
    cost.wall_clock_s = res.wall_clock_s
    return {"result": res, "cost": cost, "arm": "B0_no_memory", "meta": {}}


def run_b1(item, base_url, model, k=5, **_):
    retr = MemoryRetriever(item["messages"])
    hits = retr.search(item["question"], k=k)
    notes = "\n".join(f"- {h['content']}" for h in hits) or "(no notes found)"
    aug = ("Relevant earlier notes:\n" + notes + "\n\n" + item["question"])
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": aug}]

    # mock only: effective distance ~0 if the plant message was retrieved
    plant_idx = item.get("plant_turn_index")
    found = any(h["idx"] == plant_idx for h in hits) if plant_idx is not None else False
    eff = 2 if found else 95
    res = chat(msgs, base_url=base_url, model=model,
               mock_meta=_mock_meta(item, effective_distance=eff))

    cost = from_usage(res.usage)
    cost.n_rag_calls = 1
    cost.wall_clock_s = res.wall_clock_s
    return {"result": res, "cost": cost, "arm": "B1_always_rag",
            "meta": {"retriever": retr.backend, "n_hits": len(hits), "plant_retrieved": found}}


import cascade as _casc

ARMS = {
    # Step 1 (full-window baselines)
    "B0_no_memory": run_b0,
    "B1_always_rag": run_b1,
    # Step 2 (small-native-window regime)
    "B0_sw": _casc.run_b0_sw,
    "B2_reactive": _casc.run_b2_reactive,
    "B3_random": _casc.run_b3_random,
    "B4_cascade": _casc.run_b4_cascade,
    "B5_cascade_no_rung0": _casc.run_b5_cascade_no_rung0,
}

# Arms that need the confidence threshold theta (cascade family). run.py sweeps theta
# for these and tags each run with it; B0/B1/B0_sw ignore theta.
THETA_ARMS = {"B2_reactive", "B3_random", "B4_cascade", "B5_cascade_no_rung0"}
