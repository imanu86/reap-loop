#!/usr/bin/env python3
"""Summarize all arm_*/run* dirs under runs/ds4/20260712_virtual_bake into a
markdown table: mask, quality grade, close_html, chars, t/s, RAM peak."""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return None


def tps_from_events(path):
    """Approximate decode t/s from stream_events.jsonl: last event's
    content_chars / elapsed / ~4 chars-per-token heuristic isn't reliable;
    prefer usage.completion_tokens / elapsed_s from response.json when present."""
    return None


def summarize(run_dir):
    name = os.path.basename(run_dir)
    meta = read(os.path.join(run_dir, "RUN_META.txt")) or ""
    mask = re.search(r"^mask=(.*)$", meta, re.M)
    mask = mask.group(1) if mask else "?"
    stop = (read(os.path.join(run_dir, "STOP_REASON.txt")) or "").strip()
    grade_json = read(os.path.join(run_dir, "grade.json"))
    level = "?"
    if grade_json:
        try:
            level = "L" + str(json.loads(grade_json)["level"])
        except Exception:
            level = "parse_error"
    content_stats = read(os.path.join(run_dir, "content_stats.json"))
    chars = usage = elapsed = None
    if content_stats:
        try:
            d = json.loads(content_stats.strip().splitlines()[-1])
            chars = d.get("chars")
            usage = d.get("usage") or {}
            elapsed = d.get("elapsed_s")
        except Exception:
            pass
    tps = None
    if usage and elapsed:
        ct = usage.get("completion_tokens")
        if ct and elapsed:
            tps = round(ct / elapsed, 2)
    ram_log = read(os.path.join(run_dir, "ram_log.txt")) or ""
    avail = [int(m) for m in re.findall(r"MemAvailable_MB=(\d+)", ram_log)]
    ram_min_avail_gb = round(min(avail) / 1024, 1) if avail else None
    invalid = os.path.exists(os.path.join(run_dir, "INVALID.txt"))
    return {
        "run": name,
        "mask": mask,
        "stop_reason": stop,
        "grade": level,
        "chars": chars,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "elapsed_s": elapsed,
        "t/s": tps,
        "ram_min_available_gb": ram_min_avail_gb,
        "INVALID": invalid,
    }


def main():
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(ROOT, "arm_*_run*"))):
        if os.path.isdir(run_dir):
            rows.append(summarize(run_dir))
    cols = ["run", "mask", "grade", "chars", "completion_tokens", "t/s",
            "stop_reason", "ram_min_available_gb", "INVALID"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
