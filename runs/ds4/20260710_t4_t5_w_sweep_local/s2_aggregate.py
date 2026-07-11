"""Aggregatore S2 (T4+T5): fonde i summary.csv per-gruppo in summary_all.csv e
stampa le tabelle del REPORT — scala W per-seed e per freeze_class, mediane e
monotonia T4, confronto T5 weighted-vs-unit.

Note di lettura:
  * freeze_class: raw = nessun boundary sicuro trovato (freeze_boundary='none',
    taglio grezzo — patologia fence/prosa); clean = taglio su boundary
    strutturale. La scala W va letta per classe (mandato coordinator).
  * doc_restart := doctype >= 2. Il flag 'restart' del grader e' valorizzato
    solo sui percorsi L2/L3 (grade_frontpage assegna det['restart'] tardi),
    quindi sottoconta su righe L0/L1; il conteggio doctype e' il ground truth.

Uso: python s2_aggregate.py [run_dir]   (default: la dir di questo script)
"""
import csv
import pathlib
import statistics
import sys

BASE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parent


def read_rows(pattern):
    rows = []
    for f in sorted(BASE.glob(pattern)):
        with open(f, newline="") as fh:
            first = fh.readline()
            if first.startswith("SKIPPED-MARKER"):
                continue
            fh.seek(0)
            for r in csv.DictReader(fh):
                r["_group"] = f.parent.name
                r["freeze_class"] = ("raw" if r.get("freeze_boundary") == "none"
                                     else "clean")
                r["doc_restart"] = 1 if int(r.get("doctype") or 0) >= 2 else 0
                rows.append(r)
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    t4 = read_rows("t4_W*/summary.csv")
    t5 = read_rows("t5_*/summary.csv")

    all_rows = t4 + t5
    if all_rows:
        fields = ["_group"] + [k for k in all_rows[0] if k != "_group"]
        with open(BASE / "summary_all.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    print("== T4: W x seed (mask weighted) ==")
    print("W    run  boundary   class  l0l3  doc_restart  repeat  p2gen  chars")
    by_w = {}
    for r in t4:
        w = int(r["w"])
        by_w.setdefault(w, []).append(r)
        print(f"{w:<4} {r['run_index']:<4} {r['freeze_boundary']:<10} "
              f"{r['freeze_class']:<6} {r['l0l3']:<5} {r['doc_restart']:<12} "
              f"{r['repeat']:<7} {r['p2_gen_tps'] or '-':<6} {r['chars']}")

    print("\n== T4 mediane per W ==")
    med = {}
    for w in sorted(by_w):
        lv = [fnum(r["l0l3"]) for r in by_w[w] if fnum(r["l0l3"]) is not None]
        med[w] = statistics.median(lv) if lv else None
        dr = sum(r["doc_restart"] for r in by_w[w])
        rp = sum(int(r["repeat"]) for r in by_w[w])
        print(f"W={w:<4} mediana L={med[w]}  doc_restart={dr}/{len(by_w[w])}  "
              f"loop(repeat)={rp}/{len(by_w[w])}")
    lv = [v for _, v in sorted(med.items()) if v is not None]
    mono = all(b >= a for a, b in zip(lv, lv[1:])) if len(lv) > 1 else None
    spread = (max(lv) - min(lv)) if lv else None
    print(f"monotonia non-decrescente={mono}  spread={spread}")

    print("\n== T5 weighted vs unit (W=50, ABAB) ==")
    arms = {}
    for r in t5:
        arm = "weighted" if "weighted" in r["_group"] else "unit"
        arms.setdefault(arm, []).append(r)
        print(f"{r['_group']:<16} l0l3={r['l0l3']} doc_restart={r['doc_restart']} "
              f"repeat={r['repeat']} p2gen={r['p2_gen_tps'] or '-'} chars={r['chars']}")
    for arm in sorted(arms):
        lv = [fnum(r["l0l3"]) for r in arms[arm] if fnum(r["l0l3"]) is not None]
        tps = [fnum(r["p2_gen_tps"]) for r in arms[arm] if fnum(r["p2_gen_tps"])]
        print(f"{arm}: mediana L={statistics.median(lv) if lv else None} "
              f"mediana p2gen={statistics.median(tps) if tps else None} n={len(arms[arm])}")


if __name__ == "__main__":
    main()
