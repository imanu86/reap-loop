"""Unit tests for scripts/build_session_mask_canonical.py (T5 weighted-vs-unit).

Hermetic: builds a tiny synthetic routing trace in memory, no CSV file, no
server/WSL. Proves (a) weighted ranks by gate mass while unit ranks by count so
the two masks can diverge, (b) the pruned-pair / keep invariants, and (c) the
JSON sidecar schema.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_session_mask_canonical.py"
SPEC = importlib.util.spec_from_file_location("build_session_mask_canonical", SCRIPT)
assert SPEC and SPEC.loader
bsm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bsm
SPEC.loader.exec_module(bsm)


def _write_trace(tmp_path, rows, *, header=True, weights=True):
    """Write a route CSV. Each row: (layer, [ (expert, weight), ... ])."""
    lines = []
    if header:
        lines.append("pos,layer,n,e0,e1,e2,e3,e4,e5,w0,w1,w2,w3,w4,w5")
    pos = 0
    for layer, picks in rows:
        es = [str(e) for e, _ in picks]
        ws = [f"{w:.6f}" for _, w in picks]
        es += ["-1"] * (6 - len(es))
        ws += ["0.0"] * (6 - len(ws))
        if not weights:
            ws = []
        cols = [str(pos), str(layer), str(len(picks))] + es + ws
        lines.append(",".join(cols))
        pos += 1
    p = tmp_path / "route.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_weighted_vs_unit_diverge(tmp_path):
    # Layer 3: expert 7 is picked ONCE but with huge mass; experts 1,2 picked
    # twice each with small mass. Unit-count ranks 1,2 above 7; mass ranks 7 top.
    rows = [
        (3, [(1, 0.10), (2, 0.10)]),
        (3, [(1, 0.10), (2, 0.10)]),
        (3, [(7, 5.00), (9, 0.01)]),
    ]
    p = _write_trace(tmp_path, rows)
    keep_w, _, hw = bsm.build(str(p), K=1, mode="weighted", n_expert=16)
    keep_u, _, _ = bsm.build(str(p), K=1, mode="unit", n_expert=16)
    assert hw is True
    assert keep_w[3] == [7]            # gate mass wins -> expert 7
    assert keep_u[3] == [1]            # unit count + id tie-break -> expert 1
    assert keep_w != keep_u


def test_pruned_pairs_are_complement_of_keep(tmp_path):
    rows = [(5, [(0, 0.9), (3, 0.5), (8, 0.2)])]
    p = _write_trace(tmp_path, rows)
    keep, _, _ = bsm.build(str(p), K=2, mode="weighted", n_expert=16)
    assert keep[5] == [0, 3]  # top-2 by mass, sorted
    pruned = {(l, e) for l, e in bsm.pruned_pairs(keep, n_expert=16)}
    assert len(pruned) == 16 - 2                 # everything except the 2 kept
    assert (5, 0) not in pruned and (5, 3) not in pruned
    assert (5, 8) in pruned


def test_under_seen_layer_keeps_all_seen(tmp_path):
    # Only 2 distinct experts seen but K=5 -> keep both, prune the other 14.
    rows = [(9, [(4, 0.7), (11, 0.3)])]
    p = _write_trace(tmp_path, rows)
    keep, _, _ = bsm.build(str(p), K=5, mode="weighted", n_expert=16)
    assert keep[9] == [4, 11]
    pruned = list(bsm.pruned_pairs(keep, n_expert=16))
    assert len(pruned) == 14


def test_weighted_requires_weights(tmp_path):
    p = _write_trace(tmp_path, [(3, [(1, 0.0)])], weights=False)
    try:
        bsm.build(str(p), K=1, mode="weighted", n_expert=16)
    except SystemExit as exc:
        assert "weighted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for weighted without weights")
    # unit mode still works without weights
    keep, _, hw = bsm.build(str(p), K=1, mode="unit", n_expert=16)
    assert hw is False and keep[3] == [1]


def test_keep_json_schema(tmp_path):
    rows = [(3, [(1, 0.9), (2, 0.1)]), (4, [(5, 0.8), (6, 0.2)])]
    p = _write_trace(tmp_path, rows)
    keep, _, _ = bsm.build(str(p), K=1, mode="weighted", n_expert=16)
    obj = bsm.keep_json(keep, n_expert=16, keep_n=1, mode="weighted",
                        tag="t", note="n")
    assert obj["method"] == "session_mass_rank"
    assert obj["n_expert"] == 16 and obj["keep_n"] == 1
    assert obj["keep"] == {"3": [1], "4": [5]}


def test_empty_trace_raises(tmp_path):
    p = tmp_path / "route.csv"
    p.write_text("pos,layer,n,e0,e1,e2,e3,e4,e5,w0,w1,w2,w3,w4,w5\n", encoding="utf-8")
    try:
        bsm.build(str(p), K=1, mode="weighted", n_expert=16)
    except SystemExit:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for empty trace")
