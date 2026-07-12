#!/usr/bin/env python3
import json
import re
import statistics
import sys
from pathlib import Path


def load_jsonl(path: Path, event: str):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ev") == event:
            rows.append(row)
    return rows


def summarize(run: Path):
    adaptive = load_jsonl(run / "livemask.jsonl", "adaptive_k")
    spex = load_jsonl(run / "spex_mask.jsonl", "spex_add")
    stderr = (run / "server.stderr.log").read_text(
        encoding="utf-8", errors="replace"
    )

    response_path = run / "response.json"
    content = ""
    if response_path.exists() and response_path.stat().st_size:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        content = payload["choices"][0]["message"]["content"]

    speeds = re.findall(r"gen=\d+ decoding .*? avg=([0-9.]+) t/s", stderr)
    hits = re.findall(r"expert tiering observe summary .*?hit_rate=([0-9.]+)", stderr)
    finish = re.findall(r"gen=\d+ finish=\S+ ([0-9.]+)s", stderr)

    ks = [int(row["new_k"]) for row in adaptive]
    changes = sum(row.get("new_k") != row.get("old_k") for row in adaptive)
    entrants = sum(int(row.get("entrants", 0)) for row in spex)
    changed = sum(bool(row.get("changed")) for row in spex)
    failed_wrap = sum(not bool(row.get("wrap_ready", 1)) for row in spex)
    wrap_ms = [float(row.get("wrap_ms", 0.0)) for row in spex if row.get("entrants", 0)]

    malformed_doctype = bool(re.search(r"<!DOCTYPE\s+html\s*\n\s*<html", content, re.I))
    return {
        "run": run.name,
        "t_s": float(speeds[-1]) if speeds else None,
        "total_s": float(finish[-1]) if finish else None,
        "hit_rate": float(hits[-1]) if hits else None,
        "k_avg": statistics.fmean(ks) if ks else None,
        "k_min": min(ks) if ks else None,
        "k_max": max(ks) if ks else None,
        "k_changes": changes,
        "spex_entrants": entrants,
        "spex_changed": changed,
        "wrap_fail": failed_wrap,
        "wrap_ms_mean": statistics.fmean(wrap_ms) if wrap_ms else 0.0,
        "has_doctype": "<!DOCTYPE" in content,
        "malformed_doctype": malformed_doctype,
        "prefix": content.replace("\n", " ")[:100],
    }


for arg in sys.argv[1:]:
    print(json.dumps(summarize(Path(arg)), ensure_ascii=True, sort_keys=True))
