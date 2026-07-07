"""Fase B / STS — Sequential Temperature Scaling (DSpark paper §3.2.1) sui dati reali.

Input: accept.csv del run teacher-forcing (colonne c1..cB = LOGIT pre-sigmoide della
confidence head; d/g = draft e ground-truth per posizione).

Calibrazione fedele al paper: per ogni posizione k (sinistra→destra) grid-search 1D della
temperatura T_k che minimizza l'ECE del PRODOTTO CUMULATO P_k = Π_{i<=k} σ(c_i/T_i)
contro la sopravvivenza empirica del prefisso S_k = 1{d_1..d_k tutti accettati},
tenendo fisse le temperature già calibrate. Order-preserving per costruzione.

Anti-overfit: fit su cicli pari, ECE riportata su cicli dispari (holdout) e sul totale.

Uso:
  python -m msc.dspark.sts_fit --csv runs/dspark/.../accept.csv --out runs/dspark/<dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

N_BINS = 10
T_GRID = [0.25 * 1.06 ** i for i in range(52)]  # 0.25 .. ~4.9, passo geometrico


@dataclass
class Cycle:
    domain: str
    logits: list[float]      # c_k grezzi (pre-sigmoide)
    accept: list[bool]       # d_k == g_k per posizione
    survival: list[bool]     # prefisso 1..k tutto accettato

    @property
    def prefix_len(self) -> int:
        n = 0
        for a in self.accept:
            if not a:
                break
            n += 1
        return n


def load_cycles(path: str) -> list[Cycle]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            block = sum(1 for k in row if k.startswith("c"))
            logits = [float(row[f"c{k}"]) for k in range(1, block + 1)]
            acc = [row[f"d{k}"] == row[f"g{k}"] for k in range(1, block + 1)]
            surv, ok = [], True
            for a in acc:
                ok = ok and a
                surv.append(ok)
            out.append(Cycle(row["prompt"], logits, acc, surv))
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cum_products(cy: Cycle, temps: list[float]) -> list[float]:
    p, out = 1.0, []
    for c, t in zip(cy.logits, temps):
        p *= sigmoid(c / t)
        out.append(p)
    return out


def ece(pairs: list[tuple[float, int]]) -> float:
    """Expected Calibration Error, bin uniformi su [0,1]."""
    if not pairs:
        return float("nan")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(N_BINS)]
    for p, y in pairs:
        bins[min(N_BINS - 1, int(p * N_BINS))].append((p, y))
    tot = len(pairs)
    return sum(
        abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) * len(b)
        for b in bins if b
    ) / tot


def ece_at_k(cycles: list[Cycle], temps: list[float], k: int) -> float:
    pairs = []
    for cy in cycles:
        p = cum_products(cy, temps)[k]
        pairs.append((p, 1 if cy.survival[k] else 0))
    return ece(pairs)


def fit_sts(cycles: list[Cycle], block: int) -> list[float]:
    """Grid search sequenziale sinistra->destra (Alg. del paper §3.2.1)."""
    temps = [1.0] * block
    for k in range(block):
        best_t, best_e = 1.0, float("inf")
        for t in T_GRID:
            trial = temps[:k] + [t] + temps[k + 1:]
            e = ece_at_k(cycles, trial, k)
            if e < best_e:
                best_e, best_t = e, t
        temps[k] = best_t
    return temps


def report(cycles: list[Cycle], temps_raw: list[float], temps_fit: list[float],
           label: str, block: int) -> list[str]:
    lines = [f"### {label} (n={len(cycles)} cicli)"]
    lines.append("| pos | ECE raw | ECE STS | surv. reale | surv. predetta STS |")
    lines.append("|---|---|---|---|---|")
    for k in range(block):
        e0 = ece_at_k(cycles, temps_raw, k)
        e1 = ece_at_k(cycles, temps_fit, k)
        real = sum(1 for c in cycles if c.survival[k]) / len(cycles)
        pred = sum(cum_products(c, temps_fit)[k] for c in cycles) / len(cycles)
        lines.append(f"| {k+1} | {e0:.3f} | {e1:.3f} | {real:.3f} | {pred:.3f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cycles = load_cycles(a.csv)
    block = len(cycles[0].logits)
    raw = [1.0] * block

    fit_set = [c for i, c in enumerate(cycles) if i % 2 == 0]
    holdout = [c for i, c in enumerate(cycles) if i % 2 == 1]

    temps = fit_sts(fit_set, block)

    os.makedirs(a.out, exist_ok=True)
    lines = ["# STS fit — risultati", "",
             f"csv: `{a.csv}` · cicli: {len(cycles)} (fit={len(fit_set)}, holdout={len(holdout)})",
             f"temperature fittate: {[round(t, 3) for t in temps]}", ""]
    lines += report(fit_set, raw, temps, "FIT set (pari)", block) + [""]
    lines += report(holdout, raw, temps, "HOLDOUT set (dispari) — il numero che conta", block) + [""]
    for dom in sorted({c.domain for c in cycles}):
        sub = [c for c in cycles if c.domain == dom]
        lines += report(sub, raw, temps, f"dominio: {dom} (fit globale)", block) + [""]

    with open(os.path.join(a.out, "sts_params.json"), "w") as f:
        json.dump({"temperatures": temps, "block_size": block,
                   "source_csv": a.csv, "confidence_input": "pre-sigmoid logits",
                   "formula": "P_k = prod_{i<=k} sigmoid(c_i / T_i)"}, f, indent=2)
    with open(os.path.join(a.out, "STS_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
