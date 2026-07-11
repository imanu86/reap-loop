#!/usr/bin/env python3
"""
Expert temporal locality & predictability — OFFLINE (no GPU).

Question: we never need all ~1000 keep-experts at once. Per token a MoE decode
step touches exactly 6 experts x 40 MoE layers = 240 distinct (layer,expert)
"slots". 240 x 6.75 MiB ~= 1.6 GB -> fits the ~2 GB free on the 3060. The
"989 don't fit" wall is the UNION over time, not the instantaneous need.

So the right lever is: keep the ACTIVE set resident + async-prefetch the next
token's experts behind compute. This script MEASURES whether that is viable:
how LOCAL (token-to-token overlap) and how PREDICTABLE (cheap resident-set
policies at a fixed budget) the active set is, vs the static-pin control.

Input CSV format (real post-mask routing): pos,layer,n,e0..e5,w0..w5
A globally-unique expert = (layer, expert_id). Per-token footprint = 240 slots.
"""
import csv, gzip, io, os, sys
from collections import defaultdict, Counter, OrderedDict

EXPERT_MIB = 6.75          # per-expert VRAM footprint (given)
PICKS_PER_LAYER = 6        # top-k router
FREE_VRAM_GB = 2.0         # assumed free on 3060 for the expert cache

def load(path):
    """Return list of tokens, each = {layer: [experts]} ordered by pos asc."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        r = csv.reader(fh)
        header = next(r)
        by_pos = defaultdict(dict)
        for row in r:
            if not row or row[0] == "pos":
                continue
            pos = int(row[0]); layer = int(row[1]); n = int(row[2])
            experts = [int(x) for x in row[3:3+n]]
            by_pos[pos][layer] = experts
    tokens = [by_pos[p] for p in sorted(by_pos)]
    return tokens

def analyze(path, name):
    tok = load(path)
    T = len(tok)
    layers = sorted({l for t in tok for l in t})
    L = len(layers)
    out = {"name": name, "T": T, "L": L}

    # ---- 1. INSTANTANEOUS FOOTPRINT ----------------------------------------
    per_tok_distinct = []
    union = set()
    for t in tok:
        s = set()
        for l, es in t.items():
            for e in es:
                s.add((l, e)); union.add((l, e))
        per_tok_distinct.append(len(s))
    inst = sum(per_tok_distinct) / T
    out["inst_distinct"] = inst
    out["inst_min"] = min(per_tok_distinct); out["inst_max"] = max(per_tok_distinct)
    out["footprint_mib"] = inst * EXPERT_MIB
    out["footprint_2x_mib"] = 2 * inst * EXPERT_MIB
    out["union_pairs"] = len(union)
    out["union_mib"] = len(union) * EXPERT_MIB
    out["fits_1x"] = out["footprint_mib"] <= FREE_VRAM_GB * 1024
    out["fits_2x"] = out["footprint_2x_mib"] <= FREE_VRAM_GB * 1024

    # ---- 2. TEMPORAL LOCALITY (per-layer, consecutive tokens) --------------
    overlap_hist = Counter()      # overlap 0..6 of the 6 picks, token t vs t-1
    stay_sum = 0; pairs = 0
    for i in range(1, T):
        for l in layers:
            a = set(tok[i-1].get(l, [])); b = set(tok[i].get(l, []))
            if not a or not b:
                continue
            ov = len(a & b)
            overlap_hist[ov] += 1
            stay_sum += ov; pairs += 1
    out["mean_stay"] = stay_sum / pairs          # of 6, how many persist
    out["mean_new"]  = PICKS_PER_LAYER - out["mean_stay"]
    out["overlap_hist"] = {k: overlap_hist.get(k, 0) / pairs for k in range(7)}

    # window turnover: fraction of a token's 6 picks NOT covered by the union
    # of the previous W tokens (per layer) -> intrinsic miss rate, unbounded set
    out["win_miss"] = {}
    for W in (1, 4, 16, 64):
        miss = 0; tot = 0
        for i in range(W, T):
            for l in layers:
                b = set(tok[i].get(l, []))
                if not b:
                    continue
                hist = set()
                for j in range(i-W, i):
                    hist |= set(tok[j].get(l, []))
                miss += len(b - hist); tot += len(b)
        out["win_miss"][W] = miss / tot
    return out, tok, layers

def predictors(tok, layers, budgets=(6, 12), hot_window=32):
    """Causal resident-set policies. Report mean hit = fraction of the 240
    actual picks already resident BEFORE the token is processed.
    Budget b is PER LAYER (b=6 -> 240 total, b=12 -> 480 total)."""
    T = len(tok)
    res = {}

    # static control: whole-trace top-b per layer (best-case static pin = 0031)
    freq = {l: Counter() for l in layers}
    for t in tok:
        for l in layers:
            for e in t.get(l, []):
                freq[l][e] += 1
    for b in budgets:
        static_res = {l: set([e for e, _ in freq[l].most_common(b)]) for l in layers}
        hit = 0; tot = 0
        for t in tok:
            for l in layers:
                picks = t.get(l, [])
                if not picks:
                    continue
                hit += len(set(picks) & static_res[l]); tot += len(picks)
        res[("static", b)] = hit / tot

    # prev-token (b=6 effective): resident = previous token's 6 picks
    hit = 0; tot = 0
    for i in range(1, T):
        for l in layers:
            picks = tok[i].get(l, []); prev = set(tok[i-1].get(l, []))
            if not picks:
                continue
            hit += len(set(picks) & prev); tot += len(picks)
    res[("prev", 6)] = hit / tot

    # LRU-b and hot-window-b (causal)
    for b in budgets:
        # LRU: per layer, OrderedDict of expert->recency (move-to-end on use)
        lru = {l: OrderedDict() for l in layers}
        hit_lru = 0; tot_lru = 0
        # hot: trailing window of picks per layer
        from collections import deque
        win = {l: deque() for l in layers}       # each entry = list of picks
        cnt = {l: Counter() for l in layers}
        hit_hot = 0; tot_hot = 0
        for i in range(T):
            for l in layers:
                picks = tok[i].get(l, [])
                if not picks:
                    continue
                # --- predict using state BEFORE this token ---
                # LRU resident = last b keys
                resident_lru = list(lru[l].keys())[-b:]
                # hot resident = top-b in trailing window
                resident_hot = [e for e, _ in cnt[l].most_common(b)]
                if i > 0:  # skip first token (cold start, nothing to predict from)
                    hit_lru += len(set(picks) & set(resident_lru)); tot_lru += len(picks)
                    hit_hot += len(set(picks) & set(resident_hot)); tot_hot += len(picks)
                # --- update state with this token ---
                for e in picks:
                    if e in lru[l]:
                        lru[l].move_to_end(e)
                    else:
                        lru[l][e] = 1
                win[l].append(picks)
                for e in picks:
                    cnt[l][e] += 1
                if len(win[l]) > hot_window:
                    old = win[l].popleft()
                    for e in old:
                        cnt[l][e] -= 1
                        if cnt[l][e] <= 0:
                            del cnt[l][e]
        res[("lru", b)] = hit_lru / tot_lru
        res[("hot", b)] = hit_hot / tot_hot
    return res

def lru_sweep(tok, layers, bs=(6, 9, 12, 18, 24, 32)):
    """LRU hit-rate vs per-layer cache budget b (total = b*L). Answers
    'what cache size -> what hit' and 'what fits ~2GB (b~6) vs ~3.2GB (b~12)'."""
    from collections import OrderedDict
    T = len(tok); out = {}
    for b in bs:
        lru = {l: OrderedDict() for l in layers}
        hit = 0; tot = 0
        for i in range(T):
            for l in layers:
                picks = tok[i].get(l, [])
                if not picks:
                    continue
                resident = set(list(lru[l].keys())[-b:])
                if i > 0:
                    hit += len(set(picks) & resident); tot += len(picks)
                for e in picks:
                    if e in lru[l]:
                        lru[l].move_to_end(e)
                    else:
                        lru[l][e] = 1
        out[b] = hit / tot
    return out

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(root, "..", "..", ".."))
    files = [
        # (path, name)  -- PRIMARY first: long masked-real cyber trace
        ("runs/ds4/20260711_domain_calibration/route_maskedCyber_K48_cyberpunk.csv.gz", "K48_cyber_LONG(2854tok)"),
        ("runs/ds4/20260711_masked_route_traces/route_masked_K48_cyberpunk.csv", "K48_cyber"),
        ("runs/ds4/20260711_masked_route_traces/route_masked_K23_cyberpunk.csv", "K23_cyber"),
        ("runs/ds4/20260711_masked_route_traces/route_masked_K12_cyberpunk.csv", "K12_cyber"),
        ("runs/ds4/20260711_masked_route_traces/route_masked_K23_coffee.csv", "K23_coffee(narrow)"),
        ("runs/ds4/20260711_masked_route_traces/route_masked_K12_coffee.csv", "K12_coffee(narrow)"),
    ]
    results = []
    for rel, name in files:
        p = os.path.join(repo, rel)
        if not os.path.exists(p):
            print(f"# MISSING {rel}", file=sys.stderr); continue
        info, tok, layers = analyze(p, name)
        info["pred"] = predictors(tok, layers)
        info["sweep"] = lru_sweep(tok, layers)
        results.append(info)
        print(f"== {name}  T={info['T']} L={info['L']} ==")
        print(f"  inst_distinct/tok = {info['inst_distinct']:.1f} (min {info['inst_min']} max {info['inst_max']})  "
              f"footprint {info['footprint_mib']:.0f} MiB (2x {info['footprint_2x_mib']:.0f})  "
              f"fits2GB 1x={info['fits_1x']} 2x={info['fits_2x']}")
        print(f"  union_pairs = {info['union_pairs']}  ({info['union_mib']:.0f} MiB)")
        print(f"  mean_stay/6 = {info['mean_stay']:.3f}  mean_new/layer = {info['mean_new']:.3f}  "
              f"new_experts/token = {info['mean_new']*info['L']:.1f}")
        oh = info['overlap_hist']
        print("  overlap hist P(k of 6 persist): " + " ".join(f"{k}:{oh[k]:.2f}" for k in range(7)))
        print("  window intrinsic miss (unbounded): " + " ".join(f"W{W}:{info['win_miss'][W]:.3f}" for W in (1,4,16,64)))
        pr = info['pred']
        print(f"  HIT@240 (b=6/layer):  static={pr[('static',6)]:.3f}  prev={pr[('prev',6)]:.3f}  "
              f"lru={pr[('lru',6)]:.3f}  hot={pr[('hot',6)]:.3f}")
        print(f"  HIT@480 (b=12/layer): static={pr[('static',12)]:.3f}  "
              f"lru={pr[('lru',12)]:.3f}  hot={pr[('hot',12)]:.3f}")
        best6 = max(pr[('prev',6)], pr[('lru',6)], pr[('hot',6)])
        print(f"  best cheap @240 = {best6:.3f} vs static {pr[('static',6)]:.3f}  "
              f"-> misses/token = {(1-best6)*240:.0f} experts = {(1-best6)*240*EXPERT_MIB:.0f} MiB")
        sw = info["sweep"]
        print("  LRU hit vs cache: " + " ".join(
            f"b{b}({b*info['L']}exp,{b*info['L']*EXPERT_MIB/1024:.1f}GB):{sw[b]:.3f}" for b in sorted(sw)))
        print()

    # ---- speedup model (rough) ---------------------------------------------
    print("== SPEEDUP MODEL (rough, stated assumptions) ==")
    BW = 12288.0  # MiB/s effective PCIe host->device (3060 gen3x16~6, gen4~13-25; assume 12 GB/s)
    for X in (0.13, 0.50, 0.80, 0.90):
        miss = (1 - X) * 240
        mib = miss * EXPERT_MIB
        tms = mib / BW * 1000
        print(f"  hit={X:.2f}: miss={miss:.0f} exp = {mib:.0f} MiB/tok, transfer={tms:.1f} ms/tok")

if __name__ == "__main__":
    main()
