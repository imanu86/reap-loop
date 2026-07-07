"""Fase B / Scheduler — simulazione R=1 dello scheduler confidence-scheduled (Alg. 1
del paper ridotto a richiesta singola) con cost-model IO-aware per il regime streaming.

Idea (DESIGN §2.2): in streaming l'IO expert domina; il costo di verifica di un blocco-k
cresce come gli expert UNICI u(k), non come 6·k. u(k) fittata (power law) sui punti
misurati dalla trace reale (runs/ds4_routing_trace_smoke): u(1)=6, u(2)=10.6, u(4)=16.4,
u(8)=24.3.

Policy confrontate sui CICLI REALI (accept.csv, esiti veri per posizione):
  - nospec:     ℓ=0 (decode liscio)
  - fixed-ℓ:    ℓ costante 1..B
  - dynamic:    ℓ* = argmax_ℓ (1 + Σ_{j<=ℓ} a_j) / T(ℓ)   con a_j = survival STS-calibrata
  - oracle:     ℓ = lunghezza prefisso realmente accettato (upper bound)

Metriche: token/step, unità-IO per token committato (la moneta del 3060), speedup vs nospec.
Unità di costo: 1 = caricamento di UN expert di UN layer. T(ℓ) = t_fix + δ·ℓ·T1 + L·u(ℓ)
con L layer routed, T1 = costo verify 1 token, δ = costo relativo di un forward di draft.

Uso:
  python -m msc.dspark.sched_sim --csv accept.csv --params sts_params.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os

from .sts_fit import Cycle, load_cycles, sigmoid

TRACE_POINTS = [(1, 6.0), (2, 10.6), (4, 16.4), (8, 24.3)]  # misurati (trace smoke)
L_ROUTED = 40           # layer con router nel trace reale
T_FIX_FRAC = 0.10       # overhead fisso per step, frazione del verify 1-token
DELTA_DRAFT = 0.10      # costo di UN forward di draft, frazione del verify 1-token


def fit_unique_curve() -> tuple[float, float]:
    """u(k) = a * k^b in log-log least squares sui punti misurati."""
    xs = [math.log(k) for k, _ in TRACE_POINTS]
    ys = [math.log(u) for _, u in TRACE_POINTS]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = math.exp(my - b * mx)
    return a, b


A_U, B_U = fit_unique_curve()


def unique(k: int) -> float:
    return A_U * (k ** B_U) if k > 0 else 0.0


def step_cost(length: int) -> float:
    """Costo del passo (unità = 1 expert-load) per blocco di verifica di `length` token."""
    t1 = L_ROUTED * unique(1)                       # verify di 1 token
    fix = T_FIX_FRAC * t1
    draft = DELTA_DRAFT * t1 * length
    verify = L_ROUTED * unique(max(length, 1))
    return fix + draft + verify


def survivals(cy: Cycle, temps: list[float]) -> list[float]:
    p, out = 1.0, []
    for c, t in zip(cy.logits, temps):
        p *= sigmoid(c / t)
        out.append(p)
    return out


def run_policy(cycles: list[Cycle], choose) -> dict:
    tok, cost = 0, 0.0
    for cy in cycles:
        l = choose(cy)
        committed = 1 + min(cy.prefix_len, l)
        tok += committed
        cost += step_cost(l if l > 0 else 1)  # ℓ=0 = decode liscio di 1 token
    return {"tokens": tok, "cost": cost,
            "tok_per_step": tok / len(cycles),
            "io_per_token": cost / tok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cycles = load_cycles(a.csv)
    temps = json.load(open(a.params))["temperatures"]
    block = len(temps)

    def dyn(cy: Cycle) -> int:
        surv = survivals(cy, temps)
        best_l, best = 0, 1.0 / step_cost(1)
        for l in range(1, block + 1):
            theta = (1 + sum(surv[:l])) / step_cost(l)
            if theta > best:
                best, best_l = theta, l
        return best_l

    policies: dict[str, callable] = {"nospec": lambda cy: 0}
    for L in range(1, block + 1):
        policies[f"fixed-{L}"] = lambda cy, L=L: L
    policies["dynamic-STS"] = dyn
    policies["oracle"] = lambda cy: cy.prefix_len

    lines = ["# Scheduler R=1 — simulazione su cicli reali", "",
             f"u(k) = {A_U:.2f}·k^{B_U:.3f} (fit su trace misurata); L_routed={L_ROUTED}; "
             f"t_fix={T_FIX_FRAC:.0%} di T1; δ_draft={DELTA_DRAFT:.0%} di T1 per token draftato", ""]
    for scope, cyc in [("TUTTI", cycles)] + [
            (d, [c for c in cycles if c.domain == d])
            for d in sorted({c.domain for c in cycles})]:
        base = run_policy(cyc, policies["nospec"])
        lines.append(f"## {scope} (n={len(cyc)})")
        lines.append("| policy | tok/step | IO per token | speedup IO vs nospec |")
        lines.append("|---|---|---|---|")
        for name, fn in policies.items():
            r = run_policy(cyc, fn)
            lines.append(f"| {name} | {r['tok_per_step']:.2f} | {r['io_per_token']:.0f} "
                         f"| {base['io_per_token'] / r['io_per_token']:.2f}× |")
        lines.append("")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "SCHED_SIM.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
