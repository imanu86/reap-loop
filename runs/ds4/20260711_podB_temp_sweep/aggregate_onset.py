#!/usr/bin/env python3
"""Post-process podB temp-sweep outputs: merge summary.csv per group + add a
diagnostic token-onset estimate for the loop/repeat signature (word-count up
to the first 24-160 char pattern repeated 3x, same regex as the harness'
own `repeat` flag). Token-onset is an ESTIMATE (whitespace-token count over
the full deliverable text), consistent in spirit with the M1a/armA "loop
onset (est)" figures already in the repo, but not a literal generated-token
index (this harness has no per-token stream_events for the two-phase path).

Usage: python aggregate_onset.py <group_label>:<outdir> [<group_label>:<outdir> ...]
Prints a combined CSV to stdout and a markdown table to stderr... actually
writes combined.csv and combined.md next to the first outdir's parent.
"""
import csv
import json
import re
import sys
import statistics
from pathlib import Path

_REPEAT_RE = re.compile(r"(.{24,160})\1\1", re.S)
_TOKEN_RE = re.compile(r"\S+")


def onset_est(text):
    m = _REPEAT_RE.search(text or "")
    if not m:
        return None, None
    prefix = text[:m.start()]
    tok_count = len(_TOKEN_RE.findall(prefix))
    sample = m.group(1)[:80].replace("\n", "\\n")
    return tok_count, sample


def main(argv):
    groups = []
    for spec in argv:
        label, outdir = spec.split(":", 1)
        groups.append((label, Path(outdir)))

    rows = []
    for label, outdir in groups:
        manifest_path = outdir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        summary_path = outdir / "summary.csv"
        if not summary_path.exists():
            print(f"WARN: no summary.csv in {outdir}", file=sys.stderr)
            continue
        with open(summary_path, newline="") as f:
            for r in csv.DictReader(f):
                w = int(r["w"]); run = int(r["run_index"])
                deliverable = outdir / f"W{w:03d}" / f"r{run:02d}" / "deliverable.html"
                text = deliverable.read_text(encoding="utf-8", errors="ignore") if deliverable.exists() else ""
                onset, sample = onset_est(text)
                rows.append({
                    "group": label,
                    "temp": manifest.get("temp"),
                    "top_p": manifest.get("top_p"),
                    "seed": r["seed"],
                    "w": w,
                    "run_index": run,
                    "l0l3": r["l0l3"],
                    "repeat": r["repeat"],
                    "restart": r["restart"],
                    "html_close": r["html_close"],
                    "chars": r["chars"],
                    "onset_tok_est": onset,
                    "onset_sample": sample or "",
                    "p2_gen_tps": r["p2_gen_tps"],
                })

    out_root = groups[0][1].parent
    combined_csv = out_root / "combined_temp_sweep.csv"
    fields = ["group", "temp", "top_p", "seed", "w", "run_index", "l0l3", "repeat",
              "restart", "html_close", "chars", "onset_tok_est", "onset_sample", "p2_gen_tps"]
    with open(combined_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    # markdown table + per-group loop-rate/median L
    md_lines = ["| group | temp | seed | L | loop | onset(tok,est) | close</html> | chars |",
                "|---|---|---:|---|---|---:|---|---:|"]
    for r in rows:
        onset_s = str(r["onset_tok_est"]) if r["onset_tok_est"] is not None else "-"
        md_lines.append(
            f"| {r['group']} | {r['temp']} | {r['seed']} | L{r['l0l3']} | "
            f"{'SI' if r['repeat'] == '1' else 'no'} | {onset_s} | "
            f"{'si' if r['html_close'] not in ('0', 0) else 'no'} | {r['chars']} |"
        )
    md_lines.append("")
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    md_lines.append("### Per-group summary")
    md_lines.append("| group | temp | n | median L | loop-rate | restart-rate |")
    md_lines.append("|---|---|---:|---:|---:|---:|")
    for g, rs in by_group.items():
        levels = [int(x["l0l3"]) for x in rs if x["l0l3"] not in (None, "")]
        med = statistics.median(levels) if levels else None
        loops = sum(1 for x in rs if x["repeat"] == "1")
        restarts = sum(1 for x in rs if x["restart"] == "1")
        md_lines.append(f"| {g} | {rs[0]['temp']} | {len(rs)} | L{med} | {loops}/{len(rs)} | {restarts}/{len(rs)} |")

    combined_md = out_root / "combined_temp_sweep.md"
    combined_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {combined_csv}")
    print(f"wrote {combined_md}")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main(sys.argv[1:])
