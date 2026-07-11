import csv, gzip, collections, json, sys, os
TRACE = sys.argv[1]
OUT   = sys.argv[2]
N_EXPERT = 256
KS = [8, 9, 12, 16, 23]
mass = collections.defaultdict(lambda: collections.defaultdict(float))  # layer -> expert -> summed gate-weight
rows = 0
with gzip.open(TRACE, 'rt') as f:
    r = csv.reader(f); header = next(r)
    for row in r:
        rows += 1
        L = int(row[1]); n = int(row[2])
        es = [int(row[3 + i]) for i in range(n)]
        ws = [float(row[9 + i]) for i in range(n)]
        for e, w in zip(es, ws):
            mass[L][e] += w
layers = sorted(mass)
print(f"rows={rows} layers={len(layers)} range={layers[0]}..{layers[-1]}")
distinct = {L: len(mass[L]) for L in layers}
print("distinct experts/layer min/med/max:",
      min(distinct.values()),
      sorted(distinct.values())[len(distinct)//2],
      max(distinct.values()))
os.makedirs(OUT, exist_ok=True)
for K in KS:
    mask_lines = []
    keep = {}
    for L in layers:
        ranked = sorted(mass[L].keys(), key=lambda e: (-mass[L][e], e))
        kept = ranked[:K]
        keep[L] = sorted(kept)
        keptset = set(kept)
        for e in range(N_EXPERT):
            if e not in keptset:
                mask_lines.append(f"{L} {e}")
    txt = os.path.join(OUT, f"sessK{K}_coffee.txt")
    with open(txt, 'w') as fo:
        fo.write("\n".join(mask_lines) + "\n")
    js = os.path.join(OUT, f"sessK{K}_coffee.json")
    json.dump({"tag": f"coffee_k0_mass_k{K}", "method": "k0_coffee_mass_rank",
               "source": os.path.basename(TRACE), "n_expert": N_EXPERT,
               "keep_n": K, "keep": {str(L): keep[L] for L in layers}},
              open(js, 'w'), indent=0)
    print(f"K={K}: mask_lines={len(mask_lines)} (expect {(N_EXPERT-K)*len(layers)}) keep/layer={K} ws={K*len(layers)}")
