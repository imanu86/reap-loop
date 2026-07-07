"""coverage_curve.py — la FORMA della concentrazione da una traccia gia raccolta (CPU, no GPU).

Il punto singolo a theta=0.95 puo ingannare: conta il ginocchio. E mischiare i token di PREFILL
(template + prompt) con quelli di DECODE (il modello che genera) puo gonfiare la diffusivita: il
"working set del task" e meglio misurato sui token di decode.

Distinzione prefill/decode senza metadati extra: ogni forward corrisponde a una coppia (layer, step).
Un forward di PREFILL processa >1 token (tanti token_pos per quello step); un forward di DECODE ne
processa 1 (un solo record per quello step). Classifichiamo i gruppi (layer, step) per dimensione.

Uso:
    python scripts/coverage_curve.py --trace runs/smoke_granite/trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict

_THETAS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def _read(trace_path):
    """Ritorna lista di record minimali (layer, step, topk_ids) dalla traccia jsonl."""
    recs = []
    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append((int(r["layer"]), int(r["step"]), [int(x) for x in r["topk_ids"]]))
    return recs


def _split_prefill_decode(recs):
    """Classifica ogni (layer, step) come prefill (>1 token) o decode (==1 token)."""
    by_group = defaultdict(list)  # (layer, step) -> list di record
    for layer, step, ids in recs:
        by_group[(layer, step)].append((layer, step, ids))
    prefill, decode = [], []
    for group in by_group.values():
        (decode if len(group) == 1 else prefill).extend(group)
    return prefill, decode


def _per_layer_hist(recs):
    """layer -> Counter(expert_id -> conteggio attivazioni)."""
    hist = defaultdict(Counter)
    for layer, _step, ids in recs:
        hist[layer].update(ids)
    return hist


def _committed_fraction(hist, n_total, theta):
    """Quota media (sui layer) di expert necessari a coprire `theta` dell'uso."""
    fracs, ws_sizes = [], []
    for _layer, counter in hist.items():
        total = sum(counter.values())
        if total == 0:
            continue
        counts = sorted(counter.values(), reverse=True)
        cum, k = 0, 0
        for c in counts:
            cum += c
            k += 1
            if cum / total >= theta:
                break
        fracs.append(k / n_total)
        ws_sizes.append(k)
    frac = statistics.mean(fracs) if fracs else 0.0
    ws_med = int(statistics.median(ws_sizes)) if ws_sizes else 0
    return frac, ws_med


def _print_table(name, hist, n_total):
    print(f"\n=== {name} ===")
    print(f"{'theta':>6} | {'commit_frac':>11} | {'ws_mediano':>10} | taglio VRAM_expert")
    print("-" * 58)
    for theta in _THETAS:
        frac, ws_med = _committed_fraction(hist, n_total, theta)
        cut = 1.0 - frac
        bar = "#" * int(round(cut * 30))
        print(f"{theta:>6.2f} | {frac:>11.3f} | {ws_med:>6}/{n_total} | {cut*100:5.1f}%  {bar}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="runs/smoke_granite/trace.jsonl")
    args = p.parse_args()

    recs = _read(args.trace)
    n_total = max((max(ids) for _, _, ids in recs if ids), default=0) + 1
    prefill, decode = _split_prefill_decode(recs)
    n_pf_fwd = len({(l, s) for l, s, _ in prefill})
    n_dc_fwd = len({(l, s) for l, s, _ in decode})
    print(f"[coverage] traccia: {args.trace}  (n_total expert osservati={n_total})")
    print(f"[coverage] forward prefill={n_pf_fwd} decode={n_dc_fwd} "
          f"| record prefill={len(prefill)} decode={len(decode)}")

    _print_table("TUTTI i token", _per_layer_hist(recs), n_total)
    _print_table("solo PREFILL (template+prompt)", _per_layer_hist(prefill), n_total)
    _print_table("solo DECODE (il modello genera = il task)", _per_layer_hist(decode), n_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
