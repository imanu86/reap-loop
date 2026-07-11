#!/usr/bin/env python3
"""Grade the pulled clock-breath batch: L0-L3, collapse token, structural, tps.
Usage: grade_batch.py <cb_runs_dir> <scripts_dir>"""
import csv, json, re, statistics, sys, pathlib, importlib.util

CB = pathlib.Path(sys.argv[1])
SCR = pathlib.Path(sys.argv[2])

def load(name):
    spec = importlib.util.spec_from_file_location(name, SCR / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
fg = load("functional_grade")

REPEAT = re.compile(r"(.{24,160})\1\1", re.S)
COUNTING = re.compile(r"(?:\d+,\s*){12,}")  # counting runs evade block-repeat AND n-gram
TPS = re.compile(r"generation: ([0-9.]+) t/s")

def collapse_char(text):
    m = REPEAT.search(text or "")
    c = COUNTING.search(text or "")
    cands = [x.start() for x in (m, c) if x]
    return min(cands) if cands else None

def grade_one(d):
    out = (d / "gen.out").read_text(encoding="utf-8", errors="ignore") if (d/"gen.out").exists() else ""
    err = (d / "gen.err").read_text(encoding="utf-8", errors="ignore") if (d/"gen.err").exists() else ""
    status = (d / "status.txt").read_text(encoding="utf-8", errors="ignore") if (d/"status.txt").exists() else ""
    try:
        level, det = fg.grade_frontpage(out)
    except Exception as e:
        level, det = None, {"err": str(e)}
    low = out.lower()
    cc = collapse_char(out)
    tps_all = [float(x) for x in TPS.findall(err)]
    breaths = 0
    pace = d / "pace.jsonl"
    if pace.exists():
        breaths = pace.read_text(errors="ignore").count('"breath(clock)"')
    return {
        "run": d.name, "l0l3": level, "chars": len(out),
        "collapse_char": cc,
        "collapse_tok_est": round(cc/3.3) if cc is not None else None,
        "html_close": low.count("</html>"),
        "doctype": low.count("<!doctype"),
        "form": low.count("<form"), "script": low.count("<script"),
        "alert": low.count("alert(")+low.count("confirm("),
        "repeat": 1 if cc is not None else 0,
        "restart": int(bool(det.get("restart"))) if isinstance(det, dict) else None,
        "button_wired": int(bool(det.get("button_wired"))) if isinstance(det, dict) else None,
        "form_wired": int(bool(det.get("form_wired"))) if isinstance(det, dict) else None,
        "gen_tps": (tps_all[-1] if tps_all else None),
        "breaths": breaths,
        "status": status.strip(),
    }

def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None

arms = sorted([p for p in CB.iterdir() if p.is_dir() and p.name[0] in "A"])
all_rows = []
summary = []
for arm in arms:
    runs = sorted([r for r in arm.iterdir() if r.is_dir()])
    rows = [grade_one(r) for r in runs]
    for r in rows: r["arm"] = arm.name
    all_rows += rows
    levels = [r["l0l3"] for r in rows]
    summary.append({
        "arm": arm.name, "n": len(rows),
        "l0l3_per_run": levels,
        "l0l3_median": med(levels),
        "collapse_tok_per_run": [r["collapse_tok_est"] for r in rows],
        "html_close_per_run": [r["html_close"] for r in rows],
        "repeat_per_run": [r["repeat"] for r in rows],
        "gen_tps_median": med([r["gen_tps"] for r in rows]),
        "breaths_per_run": [r["breaths"] for r in rows],
        "chars_per_run": [r["chars"] for r in rows],
    })

fields = ["arm","run","l0l3","chars","collapse_char","collapse_tok_est","html_close",
          "doctype","form","script","alert","repeat","restart","button_wired",
          "form_wired","gen_tps","breaths","status"]
with open(CB/"GRADED.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in all_rows: w.writerow({k:r.get(k) for k in fields})
(CB/"SUMMARY.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
