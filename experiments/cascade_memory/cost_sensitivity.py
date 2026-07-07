"""Counterfactual: at what RAG PRICE does the 2-tier (B2) beat always-RAG (B1)?

Uses the ALREADY-COLLECTED E4 runs (no new pod). Models an expensive retrieval by adding
C_rag token-equivalents per retrieval call (embedding of the query + reranking k candidates
with a cross-encoder is real compute the token count ignores):
    cost(item) = total_tokens + C_rag * n_rag_calls
B1 always-RAG pays C_rag on EVERY item (1 retrieval each); B2 pays it only when the trigger
fires (~50% here), answering the in-window rest from the cheap native tier. So as C_rag
grows, B1's cost rises faster and B2 eventually wins on cost. We find the crossover C* and
report the honest accuracy gap that persists (B1 recovers everything; B2's trigger misses a
few), i.e. whether B2 becomes Pareto-competitive (cheaper for slightly less accuracy).
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for pat in ([p] if os.path.isfile(p) else glob.glob(p)):
            with open(pat, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return [r for r in rows if r.get("has_plant", True) and r.get("error") is None]


def arm_vecs(rows, arm, theta=None):
    sel = [r for r in rows if r["arm"] == arm and (theta is None or r.get("theta") == theta)]
    tok = np.array([r["total_tokens"] for r in sel], float)
    rag = np.array([r["n_rag_calls"] for r in sel], float)
    acc = np.mean([r["correct"] for r in sel])
    return tok, rag, acc


def cost_at(tok, rag, C):
    return float((tok + C * rag).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--b2-theta", type=float, default=-0.001, help="T_learned operating theta")
    a = ap.parse_args()
    rows = load(a.runs)
    b1_tok, b1_rag, b1_acc = arm_vecs(rows, "B1_always_rag")
    b2_tok, b2_rag, b2_acc = arm_vecs(rows, "B2_reactive", a.b2_theta)
    b2_recovery = b2_rag.mean()
    print(f"B1 always-RAG : acc={b1_acc:.3f}  retrieval/item={b1_rag.mean():.2f}  base_tok={b1_tok.mean():.1f}")
    print(f"B2 T_learned  : acc={b2_acc:.3f}  retrieval/item={b2_recovery:.2f}  base_tok={b2_tok.mean():.1f}  (theta={a.b2_theta})")
    print(f"accuracy gap (B1 - B2) = {b1_acc - b2_acc:+.3f} (persists at any C_rag; B1 always retrieves)\n")

    # crossover C*: cost_B1(C) == cost_B2(C)
    # b1_tok.mean()+C*b1_rag.mean() == b2_tok.mean()+C*b2_rag.mean()
    num = b1_tok.mean() - b2_tok.mean()
    den = b2_rag.mean() - b1_rag.mean()
    cstar = num / den if den != 0 else float("inf")
    print(f"  {'C_rag':>7} {'cost_B1':>9} {'cost_B2':>9} {'winner(cost)':>14}")
    for C in [0, 100, 200, 300, 500, 800, 1200, 2000]:
        c1, c2 = cost_at(b1_tok, b1_rag, C), cost_at(b2_tok, b2_rag, C)
        w = "B2 cheaper" if c2 < c1 else "B1 cheaper"
        print(f"  {C:>7} {c1:>9.1f} {c2:>9.1f} {w:>14}")
    print("\n" + "=" * 60)
    if cstar < 0:
        print("B2 is NEVER cheaper (it retrieves at least as often as B1 here).")
    else:
        print(f"CROSSOVER: C_rag* = {cstar:.0f} token-equivalents.")
        print(f"  For C_rag > {cstar:.0f}, the 2-tier costs less than always-RAG,")
        print(f"  at a persistent accuracy cost of {b1_acc - b2_acc:.3f}.")
        print(f"  Interpretation: retrieval must cost > ~{cstar:.0f} tok-equiv (query embed + rerank")
        print(f"  of the candidates) before skipping it on in-window items pays off.")


if __name__ == "__main__":
    main()
