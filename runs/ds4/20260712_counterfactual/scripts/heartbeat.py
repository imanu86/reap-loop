#!/usr/bin/env python3
"""Print one status line for a counterfactual-protocol run dir. Best-effort:
never raises, always prints something so a shell heartbeat loop never dies."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
chars = 0
p = out / "stream_live.txt"
if p.exists():
    try:
        chars = p.stat().st_size
    except Exception:
        pass

summary = out / "tripwire_summary.json"
extra = "no-summary-yet"
if summary.exists():
    try:
        d = json.loads(summary.read_text())
        extra = (
            f"k_avg={d.get('k_avg')} k_p90={d.get('k_p90')} "
            f"union_pct={d.get('union_max_pct')} tps={d.get('tps_recent')} "
            f"admits={d.get('admit_events_total')} "
            f"bt={d.get('breakthrough_events_total')} "
            f"bt_frac={d.get('breakthrough_recent_frac')}"
        )
    except Exception as exc:  # noqa: BLE001
        extra = f"summary_read_error:{exc}"

print(f"HEARTBEAT chars={chars} {extra}")
