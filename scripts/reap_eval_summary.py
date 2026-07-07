"""Track REAP-ds4 — sintesi eval bias-mask (docs/REAP_DS4_eval_plan.md §3).

Input: results_raw.csv dal pod (righe: cfg,chunk,rc=0,tokens=.. scored=.. nll=..
avg_nll=.. ppl=..). Output: summary json con ppl aggregata per corpus/config,
rapporti vs full (aggregati e appaiati per-chunk) e verdetto contro i criteri
PRE-REGISTRATI del piano: reap_dom <= 1.10x full; rand_dom >= 1.5x full e > reap.

Uso: python scripts/reap_eval_summary.py --raw runs/reap/2026-07-05_eval_biasmask/eval/results_raw.csv
"""
import argparse
import json
import math
import os
import re

ROW = re.compile(
    r"^(?P<cfg>\w+),(?P<chunk>[\w]+),rc=(?P<rc>\d+),tokens=(?P<tokens>\d+) "
    r"scored=(?P<scored>\d+) nll=(?P<nll>[\d.]+) avg_nll=(?P<avg>[\d.]+) ppl=(?P<ppl>[\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = {}
    for line in open(a.raw):
        m = ROW.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        assert d["rc"] == "0", f"run fallita: {line}"
        rows[(d["cfg"], d["chunk"])] = {
            "scored": int(d["scored"]), "nll": float(d["nll"]),
            "avg_nll": float(d["avg"]), "ppl": float(d["ppl"])}

    cfgs = sorted({c for c, _ in rows})
    chunks = sorted({k for _, k in rows})
    doms = [k for k in chunks if k.startswith("dom")]
    gens = [k for k in chunks if k.startswith("gen")]
    res = {"raw": {f"{c}/{k}": rows[(c, k)] for c in cfgs for k in chunks
                   if (c, k) in rows},
           "aggregate": {}, "ratios_vs_full": {}, "paired_per_chunk": {}}

    def has_all(c, ks):
        return ks and all((c, k) in rows for k in ks)

    for c in cfgs:
        for corpus, ks in (("dom", doms), ("gen", gens)):
            if not has_all(c, ks):
                continue
            tot_nll = sum(rows[(c, k)]["nll"] for k in ks)
            tot_sc = sum(rows[(c, k)]["scored"] for k in ks)
            res["aggregate"][f"{c}/{corpus}"] = {
                "scored": tot_sc, "avg_nll": round(tot_nll / tot_sc, 6),
                "ppl": round(math.exp(tot_nll / tot_sc), 4)}

    for c in cfgs:
        if c == "full":
            continue
        for corpus, ks in (("dom", doms), ("gen", gens)):
            if not (has_all(c, ks) and has_all("full", ks)):
                continue
            r = res["aggregate"][f"{c}/{corpus}"]["ppl"] / \
                res["aggregate"][f"full/{corpus}"]["ppl"]
            res["ratios_vs_full"][f"{c}/{corpus}"] = round(r, 4)
            paired = [round(math.exp(rows[(c, k)]["avg_nll"] -
                                     rows[("full", k)]["avg_nll"]), 4)
                      for k in ks]
            res["paired_per_chunk"][f"{c}/{corpus}"] = paired
            res["paired_per_chunk"][f"{c}/{corpus}_geomean"] = round(
                math.exp(sum(math.log(x) for x in paired) / len(paired)), 4)

    reap_dom = res["ratios_vs_full"].get("reap/dom")
    rand_dom = res["ratios_vs_full"].get("rand/dom")
    verdict = []
    if reap_dom is not None:
        verdict.append(f"reap_dom {reap_dom}x {'<=' if reap_dom <= 1.10 else '>'} 1.10 "
                       f"-> {'PASS' if reap_dom <= 1.10 else 'FAIL'}")
    if rand_dom is not None and reap_dom is not None:
        ok = rand_dom >= 1.5 and rand_dom > reap_dom
        verdict.append(f"rand_dom {rand_dom}x (>=1.5 e >reap) -> "
                       f"{'PASS' if ok else 'FAIL'}")
    res["verdict_preregistrato"] = verdict

    out = a.out or os.path.join(os.path.dirname(a.raw), "..", "eval_summary.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"aggregate": res["aggregate"],
                      "ratios_vs_full": res["ratios_vs_full"],
                      "verdict": verdict}, indent=1))
    print("scritto", out)


if __name__ == "__main__":
    main()
