#!/usr/bin/env python3
"""Offline study for SPEX-style predictive live masks.

Given a DS4 routing trace with weights (pos,layer,n,e0..,w0..), simulate equal-K
keep sets. The route trace is treated as high-K demand; this is an offline
ranking screen, not a quality verdict.

Important constraint: this script does not train on the prompt, on the evaluated
trace, or on a mask. A predictive band may only come from an external prediction
CSV produced by a pre-existing predictor such as the SPX1 hidden model. Without
that CSV, only past-mass baselines and oracle upper bounds are reported.
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

DemandMap = dict[tuple[int, int], dict[int, float]]


def parse_trace(path: Path) -> tuple[list[int], list[int], DemandMap]:
    per: DemandMap = {}
    positions: set[int] = set()
    layers: set[int] = set()
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise SystemExit(f"empty trace: {path}")
        header_l = [h.strip().lower() for h in header]
        fixed_expert_cols = [i for i, h in enumerate(header_l) if h.startswith("e") and h[1:].isdigit()]
        fixed_weight_cols = [i for i, h in enumerate(header_l) if h.startswith("w") and h[1:].isdigit()]
        for row in reader:
            if len(row) < 4:
                continue
            try:
                pos = int(row[0])
                layer = int(row[1])
                n = int(row[2])
                if fixed_expert_cols:
                    experts = [int(row[i]) for i in fixed_expert_cols[:n] if i < len(row) and row[i] != ""]
                    if fixed_weight_cols and len(fixed_weight_cols) >= len(experts):
                        weights = [float(row[i]) for i in fixed_weight_cols[: len(experts)] if i < len(row) and row[i] != ""]
                    else:
                        weights = [1.0] * len(experts)
                else:
                    experts = [int(row[3 + i]) for i in range(n)]
                    weight_start = 3 + n
                    if len(row) >= weight_start + n:
                        weights = [float(row[weight_start + i]) for i in range(n)]
                    else:
                        weights = [1.0] * n
            except Exception:
                continue
            if len(weights) < len(experts):
                weights.extend([1.0] * (len(experts) - len(weights)))
            per[(pos, layer)] = {e: w for e, w in zip(experts, weights)}
            positions.add(pos)
            layers.add(layer)
    return sorted(positions), sorted(l for l in layers if l >= 1), per


def parse_predictions(path: Path | None) -> DemandMap:
    """Parse external SPEX predictions.

    Expected format is flexible but explicit:
      pos,layer,n,p0,p1,...,s0,s1,...
    where layer is the target layer predicted by the external predictor.
    Aliases for pN are predN/spexN/expertN/eN, and aliases for sN are scoreN/wN.
    """
    if path is None:
        return {}
    pred: DemandMap = {}
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise SystemExit(f"empty prediction csv: {path}")
        h = [x.strip().lower() for x in header]
        try:
            pos_i = h.index("pos")
            layer_i = h.index("layer")
            n_i = h.index("n") if "n" in h else -1
        except ValueError as exc:
            raise SystemExit(f"{path}: prediction CSV needs pos,layer columns") from exc
        pred_cols = [i for i, name in enumerate(h) if any(name == f"{p}{j}" for p in ("p", "pred", "spex", "expert", "e") for j in range(256))]
        score_cols = [i for i, name in enumerate(h) if any(name == f"{p}{j}" for p in ("s", "score", "w") for j in range(256))]
        if not pred_cols:
            raise SystemExit(f"{path}: prediction CSV needs p0/pred0/spex0 columns")
        for row in reader:
            try:
                pos = int(row[pos_i])
                layer = int(row[layer_i])
                n = int(row[n_i]) if n_i >= 0 and row[n_i] != "" else len(pred_cols)
                experts = [int(row[i]) for i in pred_cols[:n] if i < len(row) and row[i] != ""]
                if score_cols:
                    scores = [float(row[i]) for i in score_cols[: len(experts)] if i < len(row) and row[i] != ""]
                else:
                    scores = [1.0] * len(experts)
            except Exception:
                continue
            if len(scores) < len(experts):
                scores.extend([1.0] * (len(experts) - len(scores)))
            pred[(pos, layer)] = {e: s for e, s in zip(experts, scores)}
    return pred


def topk(score: dict[int, float], k: int) -> set[int]:
    if k <= 0 or not score:
        return set()
    return set(sorted(score, key=lambda e: (-score[e], e))[:k])


def topk_excluding(score: dict[int, float], k: int, excluded: set[int]) -> set[int]:
    if k <= 0 or not score:
        return set()
    candidates = (e for e in score if e not in excluded)
    return set(sorted(candidates, key=lambda e: (-score[e], e))[:k])


def add_counter(dst: collections.Counter[int], src: dict[int, float], sign: float = 1.0) -> None:
    for e, w in src.items():
        dst[e] += sign * w
        if abs(dst[e]) <= 1e-12:
            del dst[e]


def simulate(
    positions: list[int],
    layers: list[int],
    demand: DemandMap,
    predictions: DemandMap,
    *,
    k: int,
    window: int,
    delta_horizon: int,
    alpha_delta: float,
    pred_gamma: float,
    pred_band: int,
    pred_mass_window: int,
    oracle_band: int,
    oracle_add: int,
    churn_min_new: int,
) -> dict[str, float | int]:
    total_miss = total_dem = 0
    churn_miss = churn_dem = churn_events = 0
    cold_miss = 0
    pred_slots = 0
    oracle_slots = 0
    keep_slots = evaluated_rows = 0

    for layer in layers:
        ring: collections.deque[dict[int, float]] = collections.deque()
        mass: collections.Counter[int] = collections.Counter()
        hist: collections.deque[collections.Counter[int]] = collections.deque()
        pred_ring: collections.deque[dict[int, float]] = collections.deque()
        pred_mass: collections.Counter[int] = collections.Counter()
        prev_dem: set[int] | None = None

        for pos in positions:
            dem_w = demand.get((pos, layer))
            if dem_w is None:
                continue
            dem = set(dem_w)
            cur_pred = predictions.get((pos, layer), {})

            old = hist[0] if len(hist) >= delta_horizon else collections.Counter()
            score: dict[int, float] = {}
            candidates = set(mass) | set(old) | set(pred_mass) | set(cur_pred)
            for e in candidates:
                m = mass.get(e, 0.0)
                delta = m - old.get(e, 0.0)
                score[e] = m + alpha_delta * delta + pred_gamma * (pred_mass.get(e, 0.0) + cur_pred.get(e, 0.0))

            keep: set[int] = set()
            if oracle_band > 0:
                oracle = topk(dem_w, min(oracle_band, k))
                keep |= oracle
                oracle_slots += len(oracle)
            if pred_band > 0:
                pred_score = dict(pred_mass)
                for e, s in cur_pred.items():
                    pred_score[e] = pred_score.get(e, 0.0) + s
                pred = topk_excluding(pred_score, min(pred_band, max(0, k - len(keep))), keep)
                keep |= pred
                pred_slots += len(pred)
            keep |= topk_excluding(score, k - len(keep), keep)

            if oracle_add > 0:
                additions = topk_excluding(dem_w, oracle_add, keep)
                keep |= additions
                oracle_slots += len(additions)

            keep_slots += len(keep)
            evaluated_rows += 1

            miss = len(dem - keep)
            total_miss += miss
            total_dem += len(dem)
            if miss:
                cold_miss += sum(1 for e in dem - keep if e not in mass)

            if prev_dem is not None:
                new = len(dem - prev_dem)
                if new >= churn_min_new:
                    churn_events += 1
                    churn_miss += miss
                    churn_dem += len(dem)
            prev_dem = dem

            hist.append(mass.copy())
            if len(hist) > delta_horizon:
                hist.popleft()
            ring.append(dem_w)
            add_counter(mass, dem_w, +1.0)
            if len(ring) > window:
                add_counter(mass, ring.popleft(), -1.0)

            if pred_mass_window > 0:
                pred_ring.append(cur_pred)
                add_counter(pred_mass, cur_pred, +1.0)
                if len(pred_ring) > pred_mass_window:
                    add_counter(pred_mass, pred_ring.popleft(), -1.0)

    def pct(num: float, den: float) -> float:
        return 100.0 * num / den if den else 0.0

    return {
        "miss_pct": pct(total_miss, total_dem),
        "churn_miss_pct": pct(churn_miss, churn_dem),
        "cold_miss_share_pct": pct(cold_miss, total_miss),
        "demand": total_dem,
        "churn_events": churn_events,
        "pred_slots": pred_slots,
        "oracle_slots": oracle_slots,
        "avg_keep_size": keep_slots / evaluated_rows if evaluated_rows else 0.0,
    }


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def parse_csv_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--predictions", type=Path, help="External pre-trained SPEX prediction CSV; never trained here")
    ap.add_argument("--ks", default="23,33,40,50,64")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--delta-horizon", type=int, default=4)
    ap.add_argument("--churn-min-new", type=int, default=3)
    ap.add_argument("--alphas", default="0")
    ap.add_argument("--pred-gammas", default="0,1,2,4")
    ap.add_argument("--pred-bands", default="0,2,4,6,8")
    ap.add_argument("--pred-mass-window", type=int, default=4)
    ap.add_argument("--oracle-bands", default="0,4")
    ap.add_argument(
        "--oracle-adds",
        default="0",
        help="Duplicate-free perfect-foresight experts added on top of K",
    )
    args = ap.parse_args()

    positions, layers, demand = parse_trace(args.trace)
    predictions = parse_predictions(args.predictions)
    if not predictions:
        pred_gammas = [0.0]
        pred_bands = [0]
    else:
        pred_gammas = parse_csv_floats(args.pred_gammas)
        pred_bands = parse_csv_ints(args.pred_bands)

    print(f"trace={args.trace}")
    print(f"predictions={args.predictions if args.predictions else 'none'}")
    print(
        f"tokens={len(positions)} layers={len(layers)} window={args.window} "
        f"delta_horizon={args.delta_horizon} pred_mass_window={args.pred_mass_window}"
    )
    print(
        "K,alpha_delta,pred_gamma,pred_band,oracle_band,oracle_add,miss_pct,"
        "churn_miss_pct,cold_miss_share_pct,churn_events,pred_slots,oracle_slots,avg_keep_size"
    )
    for k in parse_csv_ints(args.ks):
        for a in parse_csv_floats(args.alphas):
            for gamma in pred_gammas:
                for pb in pred_bands:
                    for ob in parse_csv_ints(args.oracle_bands):
                        for oa in parse_csv_ints(args.oracle_adds):
                            if (pb and ob) or (ob and oa):
                                continue
                            r = simulate(
                                positions,
                                layers,
                                demand,
                                predictions,
                                k=k,
                                window=args.window,
                                delta_horizon=args.delta_horizon,
                                alpha_delta=a,
                                pred_gamma=gamma,
                                pred_band=pb,
                                pred_mass_window=args.pred_mass_window,
                                oracle_band=ob,
                                oracle_add=oa,
                                churn_min_new=args.churn_min_new,
                            )
                            print(
                                f"{k},{a:g},{gamma:g},{pb},{ob},{oa},{r['miss_pct']:.3f},"
                                f"{r['churn_miss_pct']:.3f},{r['cold_miss_share_pct']:.3f},"
                                f"{r['churn_events']},{r['pred_slots']},{r['oracle_slots']},"
                                f"{r['avg_keep_size']:.3f}"
                            )


if __name__ == "__main__":
    main()
