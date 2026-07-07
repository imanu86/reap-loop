"""Track REAP-ds4 — sintesi eval v2 paper-grade (mandato SPEX-main 2026-07-05).

Input: results_raw.csv merged dai pod (config: full, reap_k50, reap_k25, reap_k70,
rand50_s0/s1/s2; chunk: dom_chunk0..9, gen_chunk0..9).
Output: eval_summary_v2.json con
  - ppl aggregata per config/corpus e rapporto vs full
  - rapporti appaiati per-chunk (stesso testo) + geomean
  - bootstrap CI95 sul geomean ratio (resampling dei chunk, B=10000, seed fisso)
  - random multi-seed: ratio per seed + media/min/max cross-seed
  - tabella dose-response K25/K50/K70
  - verdetti vs criteri pre-registrati (K50)

Uso: python scripts/reap_eval_summary_v2.py --raw runs/reap/2026-07-05_eval_biasmask_v2/results_raw.csv
"""
import argparse
import json
import math
import os
import random
import re

ROW = re.compile(
    r"^(?P<cfg>[\w]+),(?P<chunk>[\w]+),rc=(?P<rc>\d+),tokens=(?P<tokens>\d+) "
    r"scored=(?P<scored>\d+) nll=(?P<nll>[\d.]+) avg_nll=(?P<avg>[\d.]+) ppl=(?P<ppl>[\d.]+)")
B = 10000
BOOT_SEED = 12345


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def bootstrap_ci(paired, b=B, seed=BOOT_SEED):
    rng = random.Random(seed)
    n = len(paired)
    stats = []
    for _ in range(b):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        stats.append(geomean(sample))
    stats.sort()
    return stats[int(0.025 * b)], stats[int(0.975 * b)]


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
    corpora = {"dom": [k for k in chunks if k.startswith("dom")],
               "gen": [k for k in chunks if k.startswith("gen")]}
    res = {"n_chunks": {c: len(k) for c, k in corpora.items()},
           "aggregate": {}, "ratios_vs_full": {}, "paired": {},
           "random_multiseed": {}, "dose_response": {}, "verdicts": []}

    def has_all(c, ks):
        return ks and all((c, k) in rows for k in ks)

    for c in cfgs:
        for corpus, ks in corpora.items():
            if not has_all(c, ks):
                continue
            tot_nll = sum(rows[(c, k)]["nll"] for k in ks)
            tot_sc = sum(rows[(c, k)]["scored"] for k in ks)
            res["aggregate"][f"{c}/{corpus}"] = {
                "scored": tot_sc, "ppl": round(math.exp(tot_nll / tot_sc), 4)}

    for c in cfgs:
        if c == "full":
            continue
        for corpus, ks in corpora.items():
            if not (has_all(c, ks) and has_all("full", ks)):
                continue
            paired = [math.exp(rows[(c, k)]["avg_nll"] - rows[("full", k)]["avg_nll"])
                      for k in ks]
            gm = geomean(paired)
            lo, hi = bootstrap_ci(paired)
            res["ratios_vs_full"][f"{c}/{corpus}"] = round(
                res["aggregate"][f"{c}/{corpus}"]["ppl"] /
                res["aggregate"][f"full/{corpus}"]["ppl"], 4)
            res["paired"][f"{c}/{corpus}"] = {
                "per_chunk": [round(x, 4) for x in paired],
                "geomean": round(gm, 4),
                "ci95": [round(lo, 4), round(hi, 4)]}

    # random multi-seed
    for corpus in corpora:
        seeds = {}
        for c in cfgs:
            if c.startswith("rand50_s") and f"{c}/{corpus}" in res["paired"]:
                seeds[c] = res["paired"][f"{c}/{corpus}"]["geomean"]
        if seeds:
            vals = list(seeds.values())
            res["random_multiseed"][corpus] = {
                "per_seed": seeds,
                "mean": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4), "max": round(max(vals), 4)}

    # dose-response
    for corpus in corpora:
        row = {}
        for kname, cfg in (("K25", "reap_k25"), ("K50", "reap_k50"),
                           ("K67", "reap_k67"), ("K70", "reap_k70")):
            p = res["paired"].get(f"{cfg}/{corpus}")
            if p:
                row[kname] = {"geomean": p["geomean"], "ci95": p["ci95"]}
        if row:
            res["dose_response"][corpus] = row

    # verdetti K50 (criteri pre-registrati, piano §3)
    rk = res["paired"].get("reap_k50/dom", {}).get("geomean")
    rnd = res["random_multiseed"].get("dom", {}).get("mean")
    if rk is not None:
        res["verdicts"].append(
            f"reap_k50 dom {rk}x {'<=' if rk <= 1.10 else '>'} 1.10 -> "
            f"{'PASS' if rk <= 1.10 else 'FAIL'}")
    if rk is not None and rnd is not None:
        ok = rnd > rk
        res["verdicts"].append(
            f"random dom mean {rnd}x (3 seed, min {res['random_multiseed']['dom']['min']}) "
            f"{'>' if ok else '<='} reap {rk}x -> {'PASS selezione-conta' if ok else 'FAIL'}")
    rg = res["paired"].get("reap_k50/gen", {}).get("geomean")
    if rg is not None:
        res["verdicts"].append(f"reap_k50 GEN {rg}x (trade-off F3 su ds4; Qwen era ~1.68x = 9.36/5.56)")

    out = a.out or os.path.join(os.path.dirname(a.raw), "eval_summary_v2.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in
                      ("aggregate", "ratios_vs_full", "random_multiseed",
                       "dose_response", "verdicts")}, indent=1))
    print("scritto", out)


if __name__ == "__main__":
    main()
