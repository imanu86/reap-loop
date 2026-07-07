#!/usr/bin/env python3
"""Analizza i log del test acceptance MTP (pod 3090).

Input: directory con i log del pod (out/). Output: summary.md + calib_pairs.csv.
Fonti (ds4@80ebbc3):
  - probe:   "ds4: mtp probe token=T draft=D hit=H/N"        (ds4.c:27117)
  - conf:    "ds4: mtp conf drafted=D committed=C ... margin=M ..." (ds4.c:27486)
  - miss1:   "ds4: mtp spec miss first draft=D"              (ds4.c:27246)
  - timing:  "ds4: mtp timing micro drafted=.. verify=.. ms" (ds4.c:27538+)
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")

RE_PROBE = re.compile(r"mtp probe token=\S+ draft=\S+ hit=(\d+)/(\d+)")
RE_CONF = re.compile(
    r"mtp conf drafted=(\d+) committed=(\d+) .*?margin=([0-9.eE+-]+)")
RE_MISS1 = re.compile(r"mtp spec miss first")
RE_TIMING = re.compile(
    r"mtp timing (micro|margin-skip|seq) drafted=(\d+) committed=(\d+).*?"
    r"draft=([0-9.]+) ms.*?verify=([0-9.]+) ms.*?total=([0-9.]+) ms")


def parse_log(path):
    txt = path.read_text(errors="replace")
    probe = None
    for m in RE_PROBE.finditer(txt):
        probe = (int(m.group(1)), int(m.group(2)))
    confs = [(int(m.group(1)), int(m.group(2)), float(m.group(3)))
             for m in RE_CONF.finditer(txt)]
    miss1 = len(RE_MISS1.findall(txt))
    timings = [(m.group(1), int(m.group(2)), int(m.group(3)),
                float(m.group(4)), float(m.group(5)), float(m.group(6)))
               for m in RE_TIMING.finditer(txt)]
    return probe, confs, miss1, timings


def main():
    runtimes = {}
    rt = OUT / "runtimes.csv"
    if rt.exists():
        for row in csv.DictReader(rt.open()):
            runtimes[row["run"]] = (int(row["rc"]), float(row["wall_s"]))

    lines = ["# Risultati acceptance MTP — pod 3090 (ds4@80ebbc3 stock)", ""]
    calib_rows = []
    groups = defaultdict(list)
    for log in sorted(OUT.glob("*.log")):
        name = log.stem
        if name.startswith(("warmup", "setup")):
            continue
        groups[re.sub(r"_r\d+$", "", name)].append(log)

    for gname in sorted(groups):
        lines.append(f"## {gname}")
        for log in groups[gname]:
            probe, confs, miss1, timings = parse_log(log)
            rc, wall = runtimes.get(log.stem, (None, None))
            parts = [f"- `{log.name}` rc={rc} wall={wall}s"]
            if probe:
                h, n = probe
                parts.append(f"  probe: **{h}/{n} = {h/n:.3f}** acceptance MTP-1")
            if confs:
                cycles = len(confs)
                drafted = sum(c[0] for c in confs)
                committed = sum(c[1] for c in confs)
                # cicli che raggiungono il verifier: draft[0] gia' accettato
                # committed>=1 sempre; acceptance del token j (2..max):
                depth_acc = {}
                for j in range(2, max(c[0] for c in confs) + 1):
                    elig = [c for c in confs if c[0] >= j and c[1] >= j - 1]
                    acc = [c for c in elig if c[1] >= j]
                    if elig:
                        depth_acc[j] = (len(acc), len(elig))
                parts.append(
                    f"  cicli-verifier={cycles} miss-first={miss1} "
                    f"drafted={drafted} committed={committed} "
                    f"tok/ciclo-verifier={committed/cycles:.2f}")
                for j, (a, e) in depth_acc.items():
                    parts.append(f"    P(accept pos{j} | pos{j-1} ok) = {a}/{e} = {a/e:.3f}")
                for d, c, m in confs:
                    calib_rows.append((log.stem, d, c, m))
            if timings:
                med_total = statistics.median(t[5] for t in timings)
                med_verify = statistics.median(t[4] for t in timings)
                med_draft = statistics.median(t[3] for t in timings)
                parts.append(
                    f"  timing mediano/ciclo: draft={med_draft:.1f}ms "
                    f"verify={med_verify:.1f}ms total={med_total:.1f}ms "
                    f"(POD-ONLY, non trasferisce al 3060)")
            lines.extend(parts)
        lines.append("")

    with (OUT.parent / "calib_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "drafted", "committed", "margin_last_draft"])
        w.writerows(calib_rows)
    lines.append(f"calib_pairs.csv: {len(calib_rows)} cicli (margine ultimo draft vs committed)")

    (OUT.parent / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
