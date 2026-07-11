"""Collapse/onset token estimator — post-hoc addendum to run_w_sweep_freeze_safe.py.

The T4 harness (scripts/run_w_sweep_freeze_safe.py) records per-run L0-L3 grade,
restart/repeat FLAGS (0/1) and chars, but no *position* of the collapse signature.
For the low-K hazard fit (docs/DECISION_MODEL.md) we need an estimated GENERATED
TOKEN INDEX at which degeneration onsets, per run.

Method (pure post-hoc text scan over deliverable.html, no re-execution):
  - doc_restart: char offset of the SECOND ``<!doctype html`` occurrence (the
    first is the legitimate phase-1 opening tag; a second means the model
    re-opened a fresh document = the restart signature functional_grade.py
    already flags via grade_frontpage's ``restart`` bool).
  - repeat_loop: char offset where the harness's own repeat regex
    ``(.{24,160})\\1\\1`` first matches (a 24-160 char block repeated 3x) —
    same pattern as run_w_sweep_freeze_safe.output_checks's ``repeat`` flag.
  - onset = the EARLIER of the two signatures found (a run can show both; the
    first one to occur is the actual onset). None if neither fires (clean run).

Token estimate uses the SAME len/4 heuristic as freeze_boundary._default_token_len
(this repo's own established convention for offline token-count estimates when no
tokenizer is available inline) — NOT an exact tokenizer count. Reported as
``onset_tok_est`` and clearly labeled as an estimate everywhere it is surfaced.

Usage: python onset_probe.py <run_group_dir>  (e.g. .../k12_coffee)
Writes <run_group_dir>/onset.csv, one row per (w, run) matching summary.csv.
"""
from __future__ import annotations

import csv
import pathlib
import re
import sys

_DOCTYPE_RE = re.compile(r"<!doctype html", re.IGNORECASE)
_REPEAT_RE = re.compile(r"(.{24,160})\1\1", re.S)


def chars_to_tok(n_chars: int) -> int:
    """len/4 estimate — same heuristic as scripts/freeze_boundary.py's
    _default_token_len, kept identical for cross-artifact consistency."""
    return max(0, round(n_chars / 4))


def find_collapse_onset(text: str):
    """Return (onset_char, onset_kind) or (None, None) if no signature found."""
    doctype_hits = [m.start() for m in _DOCTYPE_RE.finditer(text or "")]
    doc_restart_char = doctype_hits[1] if len(doctype_hits) > 1 else None
    m = _REPEAT_RE.search(text or "")
    repeat_char = m.start() if m else None
    cands = [(c, k) for c, k in
             ((doc_restart_char, "doc_restart"), (repeat_char, "repeat_loop"))
             if c is not None]
    if not cands:
        return None, None
    return min(cands, key=lambda x: x[0])


def probe_group(group_dir: pathlib.Path):
    summary = group_dir / "summary.csv"
    rows_out = []
    if not summary.exists():
        print(f"skip (no summary.csv): {group_dir}", file=sys.stderr)
        return rows_out
    with open(summary, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w, run_index = int(row["w"]), int(row["run_index"])
            rdir = group_dir / f"W{w:03d}" / f"r{run_index:02d}"
            deliverable = rdir / "deliverable.html"
            text = deliverable.read_text(encoding="utf-8", errors="ignore") if deliverable.exists() else ""
            onset_char, onset_kind = find_collapse_onset(text)
            onset_tok_est = chars_to_tok(onset_char) if onset_char is not None else None
            rows_out.append({
                "w": w, "run_index": run_index, "seed": row.get("seed"),
                "l0l3": row.get("l0l3"), "restart": row.get("restart"),
                "repeat": row.get("repeat"), "chars": row.get("chars"),
                "onset_kind": onset_kind or "none",
                "onset_char": onset_char if onset_char is not None else "",
                "onset_tok_est": onset_tok_est if onset_tok_est is not None else "",
            })
    out_csv = group_dir / "onset.csv"
    fields = ["w", "run_index", "seed", "l0l3", "restart", "repeat", "chars",
              "onset_kind", "onset_char", "onset_tok_est"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows_out:
            wr.writerow(r)
    print(f"wrote {out_csv} ({len(rows_out)} rows)")
    return rows_out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        probe_group(pathlib.Path(arg))
