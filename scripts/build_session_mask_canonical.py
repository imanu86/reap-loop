"""Canonical session-mask builder (T5 A/B: weighted OFFLINE vs unit in-engine).

Consolidates the historical two-phase builder recovered from the pod replay
(``moe .../scripts/build_session_mask.py``, run in
``runs/ds4/20260710_pod_cache1024_warmup_replay/``) into one script with an
explicit ranking-mode switch so the T5 comparison is a single-variable A/B:

  * ``--mode weighted`` — rank experts per layer by cumulative GATE MASS
    (sum of the router weights w0..w5 the expert received over the observed
    tokens). This is the historical *good* offline recipe.
  * ``--mode unit``     — rank by unit COUNT (how many times the expert was
    selected), the tie-break the in-engine PACE relearn effectively uses.

Both modes keep the top-K experts per layer and prune the rest.

Input:  routing-trace CSV from the runner / ``DS4_SPEX_TRACE_ROUTING`` with
        ``DS4_SPEX_TRACE_ROUTING_WEIGHTS=1``. Columns:
        ``pos,layer,n,e0..e5,w0..w5``.

Outputs (same basename):
  * ``<out>``        runtime mask, lines ``"<layer> <expert>"`` = PRUNED
                     (bias -1e9). This is the exact format patch 0011 parses via
                     ``fscanf("%u %u")`` and the engine loads from
                     ``DS4_REAP_MASK_FILE``.
  * ``<out>.json``   keep-list sidecar matching the schema of
                     ``reap_mask_session_d5_k23.json`` (``n_expert``, ``keep_n``,
                     ``keep`` = {layer: [kept ids]}), for ``--mask-load`` /
                     inspection / catalogue reuse.

CLI (positional args are back-compatible with the original builder)::

    python build_session_mask_canonical.py route.csv sess.txt 23 [--mode weighted]
        [--n-expert 256] [--tag TAG] [--note NOTE] [--no-json]

Pure helpers (``read_route_trace``, ``rank_keep``, ``pruned_pairs``,
``keep_json``) are import-safe and unit-tested.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict

N_EXPERT_DEFAULT = 256
MAX_TOPK = 6  # e0..e5 / w0..w5


def read_route_trace(path):
    """Read a routing-trace CSV into per-layer mass, count, and seen-sets.

    Returns ``(mass, count, seen, have_weights)`` where ``mass[layer][expert]``
    is the cumulative gate mass, ``count[layer][expert]`` the selection count,
    ``seen[layer]`` the set of distinct experts, and ``have_weights`` whether any
    row carried the w0..w5 columns.
    """
    mass = defaultdict(lambda: defaultdict(float))
    count = defaultdict(lambda: defaultdict(int))
    seen = defaultdict(set)
    have_weights = False
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd, None)  # header
        for r in rd:
            if len(r) < 9:
                continue
            try:
                layer, n = int(r[1]), int(r[2])
            except ValueError:
                continue
            for s in range(min(n, MAX_TOPK)):
                try:
                    e = int(r[3 + s])
                except (ValueError, IndexError):
                    continue
                if e < 0:
                    continue
                count[layer][e] += 1
                seen[layer].add(e)
                if len(r) >= 3 + MAX_TOPK + s + 1:  # w-column present
                    try:
                        mass[layer][e] += float(r[3 + MAX_TOPK + s])
                        have_weights = True
                    except (ValueError, IndexError):
                        pass
    return mass, count, seen, have_weights


def rank_keep(seen, score, K):
    """Top-K experts per layer by ``score`` (desc), tie-broken by expert id.

    ``score[layer][expert]`` is the ranking signal (mass or count). A layer that
    saw fewer than K distinct experts keeps all it saw (wider mask there).
    Returns ``{layer: [kept expert ids, sorted ascending]}``.
    """
    keep = {}
    for layer in sorted(seen):
        experts = seen[layer]
        ranked = sorted(experts, key=lambda e: (-score[layer][e], e))
        keep[layer] = sorted(ranked[:K])
    return keep


def pruned_pairs(keep, n_expert=N_EXPERT_DEFAULT):
    """Yield ``(layer, expert)`` for every expert NOT kept (= pruned, bias -1e9)."""
    for layer in sorted(keep):
        kept = set(keep[layer])
        for e in range(n_expert):
            if e not in kept:
                yield layer, e


def keep_json(keep, *, n_expert, keep_n, mode, tag, note):
    """Build the keep-list JSON sidecar dict (schema of the k91 reference mask)."""
    method = "session_mass_rank" if mode == "weighted" else "session_unit_rank"
    return {
        "tag": tag,
        "method": method,
        "note": note,
        "n_expert": n_expert,
        "keep_n": keep_n,
        "keep": {str(layer): keep[layer] for layer in sorted(keep)},
    }


def build(route_csv, K, mode, n_expert=N_EXPERT_DEFAULT):
    """Read the trace and compute the keep map for the requested mode.

    Returns ``(keep, have_weights)``. Raises if the trace is empty, or if
    ``weighted`` mode is requested but the trace carried no router weights.
    """
    mass, count, seen, have_weights = read_route_trace(route_csv)
    if not seen:
        raise SystemExit(f"ERRORE: nessun routing letto da {route_csv} (trace vuoto?)")
    if mode == "weighted":
        if not have_weights:
            raise SystemExit(
                f"ERRORE: --mode weighted richiede le colonne peso (w0..w5) ma "
                f"{route_csv} non le ha (rilancia fase-1 con "
                f"DS4_SPEX_TRACE_ROUTING_WEIGHTS=1, oppure usa --mode unit).")
        score = mass
    elif mode == "unit":
        score = count
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"modo sconosciuto: {mode}")
    keep = rank_keep(seen, score, K)
    return keep, seen, have_weights


def _coverage_line(keep, seen, K, mode, have_weights):
    covered = [len(v) for v in keep.values()]
    avg = sum(covered) / len(covered) if covered else 0
    full = sum(1 for c in covered if c >= K)
    distinct = [len(v) for v in seen.values()]
    ranked_by = "massa-gate" if (mode == "weighted" and have_weights) else "unit-count"
    return (f"session-mask keep-{K} (rank per {ranked_by}, mode={mode}): "
            f"{len(keep)} layer, keep medio {avg:.1f}/{K} ({full}/{len(keep)} pieni). "
            f"esperti distinti visti/layer: min {min(distinct)} max {max(distinct)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Canonical session-mask builder (weighted|unit).")
    ap.add_argument("route_csv", help="routing-trace CSV (pos,layer,n,e0..e5,w0..w5)")
    ap.add_argument("out_txt", help="output runtime mask (DS4_REAP_MASK_FILE, pruned pairs)")
    ap.add_argument("K", type=int, help="experts to KEEP per layer (e.g. 23)")
    ap.add_argument("--mode", choices=("weighted", "unit"), default="weighted",
                    help="ranking signal: gate mass (weighted) or selection count (unit)")
    ap.add_argument("--n-expert", type=int, default=N_EXPERT_DEFAULT)
    ap.add_argument("--tag", default=None, help="tag written into the JSON sidecar")
    ap.add_argument("--note", default="", help="free-text note for the JSON sidecar")
    ap.add_argument("--no-json", action="store_true", help="skip the keep-list JSON sidecar")
    args = ap.parse_args(argv)

    keep, seen, have_weights = build(args.route_csv, args.K, args.mode, args.n_expert)

    with open(args.out_txt, "w", newline="\n") as f:
        for layer, e in pruned_pairs(keep, args.n_expert):
            f.write(f"{layer} {e}\n")

    if not args.no_json:
        json_path = (args.out_txt[:-4] if args.out_txt.endswith(".txt")
                     else args.out_txt) + ".json"
        tag = args.tag or f"session_{args.mode}_k{args.K}"
        obj = keep_json(keep, n_expert=args.n_expert, keep_n=args.K, mode=args.mode,
                        tag=tag, note=args.note)
        with open(json_path, "w", newline="\n") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")

    print(_coverage_line(keep, seen, args.K, args.mode, have_weights))
    return 0


if __name__ == "__main__":
    sys.exit(main())
