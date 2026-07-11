#!/usr/bin/env python3
"""Phase-segmented expert usage — does the K23 capacity wall dissolve into
shifting per-phase hot-cores?

Falsification target: pin_analysis.py (aabaa97) measured expert usage over the
WHOLE trace and found ~quasi-uniform usage inside the keep (Gini ~0.4-0.5, needs
66-71% of the keep to cover 90% of a layer's hits) -> deduced a CAPACITY WALL for
K23 (working-set 989 > 394 cache slots).

Hypothesis here: that flatness is an AGGREGATION ARTIFACT across phases. If per-PHASE
usage is concentrated but the concentrated hot-core SHIFTS phase-to-phase, the
whole-trace average looks flat while every INSTANT fits in VRAM. We segment each
trace (sliding window = phase proxy; explicit HTML labels where token text exists)
and measure, PER SEGMENT, PER LAYER (rotation is per-layer):
  1. CONCENTRATION  : Gini + k90 (experts to cover 90% inside the segment)
  2. HOT-CORE SHIFT : Jaccard of top-M keep experts between CONSECUTIVE segments
  3. INSTANT WS     : sum over layers of per-layer k90 (90%-hit residency need)
                      -> its distribution vs the 394-slot VRAM budget, and vs the
                         whole-trace union (the 989/920 wall).
  4. ENTROPY GATE   : (cyberpunk only, has per-token conf) do entropy spikes line
                      up with the low-Jaccard (hot-core-shift) boundaries?

Offline, no GPU. Same full-model->keep-filter proxy caveat as pin_analysis.py:
traces are FULL model filtered to the keep set (no masked route.csv exists).
"""
import csv, json, gzip, collections, statistics, os, sys

ROOT = 'runs/ds4'
MASKS = f'{ROOT}/20260711_local_clean_lowK/masks'
VRAM_SLOTS = 394           # boot-probe cache slots (the budget)
N_TRACED_LAYERS = 40       # MoE layers observed in the traces (3..42)

def load_keep(tag):
    d = json.load(open(f'{MASKS}/sess{tag}.json'))['keep']
    return {int(k): set(v) for k, v in d.items()}

K23 = load_keep('K23')
K12 = load_keep('K12')
KEEP_LAYERS = sorted(K23)

# ---- trace groups ------------------------------------------------------------
COFFEE = [  # WIDE-healthy HTML (the exact traces pin_analysis used), full model
    f'{ROOT}/20260711_podA_narrow_traces/a_coffee_full/route.csv',   # 299 tok
    f'{ROOT}/20260710_pod_cache1024_warmup_replay/W130/route_W130.csv',  # 129
]
PYTHON = [f'{ROOT}/20260711_podA_narrow_traces/c2_python_long_full/route.csv']  # 235
JSON   = [f'{ROOT}/20260711_podA_narrow_traces/b2_json_long_full/route.csv']    # 206
CYBER_ROUTE = f'{ROOT}/20260711_instrumented_collapse/conf_run_16a21de6/route_p2.csv.gz'
CYBER_CONF  = f'{ROOT}/20260711_instrumented_collapse/conf_run_16a21de6/conf.csv'
CYBER_TOK   = f'{ROOT}/20260711_instrumented_collapse/conf_run_16a21de6/tokens.csv'

def _open(p):
    return gzip.open(p, 'rt') if p.endswith('.gz') else open(p)

def load_route(path, keep=None):
    """Return dict[pos] -> dict[layer] -> list of experts (optionally keep-filtered)."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    with _open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            pos = int(row['pos']); L = int(row['layer']); n = int(row['n'])
            if keep is not None and L not in keep:
                continue
            ks = keep[L] if keep is not None else None
            for i in range(n):
                e = int(row[f'e{i}'])
                if ks is None or e in ks:
                    out[pos][L].append(e)
    return out

# ---- metrics -----------------------------------------------------------------
def k90(counter):
    tot = sum(counter.values())
    if tot == 0: return 0
    c = 0; k = 0
    for v in sorted(counter.values(), reverse=True):
        c += v; k += 1
        if c >= 0.90 * tot: break
    return k

def gini(counter, keep_n):
    vals = sorted(list(counter.values()) + [0] * (keep_n - len(counter)))
    n = len(vals); s = sum(vals)
    if s == 0: return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * cum) / (n * s) - (n + 1) / n

def counters_for(posset, route, layers):
    """Per-layer Counter of expert activations over a set of positions."""
    per = {L: collections.Counter() for L in layers}
    for p in posset:
        lp = route.get(p, {})
        for L in layers:
            for e in lp.get(L, []):
                per[L][e] += 1
    return per

def segment(route, window):
    """Non-overlapping windows of `window` decode tokens, in position order."""
    positions = sorted(route)
    segs = []
    for i in range(0, len(positions), window):
        segs.append(positions[i:i + window])
    return segs

# ---- core analysis -----------------------------------------------------------
def analyze(name, route, keep, keep_n, window, M=6):
    layers = KEEP_LAYERS
    segs = segment(route, window)
    # per-segment per-layer counters
    seg_counts = [counters_for(s, route, layers) for s in segs]

    # (1) concentration per segment (median over layers of k90 and gini)
    per_seg_k90_med = []
    per_seg_gini_med = []
    per_seg_instant_ws = []      # sum over layers of k90 (90%-hit residency need)
    per_seg_instant_used = []    # sum over layers of distinct experts used (100%)
    for sc in seg_counts:
        k90s = [k90(sc[L]) for L in layers]
        ginis = [gini(sc[L], keep_n) for L in layers]
        per_seg_k90_med.append(statistics.median(k90s))
        per_seg_gini_med.append(statistics.median(ginis))
        per_seg_instant_ws.append(sum(k90s))
        per_seg_instant_used.append(sum(len(sc[L]) for L in layers))

    # (2) hot-core shift: consecutive-segment Jaccard of top-M per layer, averaged
    shift_jac = []
    for a, b in zip(seg_counts, seg_counts[1:]):
        jl = []
        for L in layers:
            sa = set(x for x, _ in a[L].most_common(M))
            sb = set(x for x, _ in b[L].most_common(M))
            if sa or sb:
                jl.append(len(sa & sb) / len(sa | sb))
        if jl:
            shift_jac.append(statistics.mean(jl))

    # (3) whole-trace UNION (recovers the pin_analysis whole-trace number)
    whole = counters_for(sorted(route), route, layers)
    union_k90 = sum(k90(whole[L]) for L in layers)
    union_used = sum(len(whole[L]) for L in layers)
    whole_k90_med = statistics.median([k90(whole[L]) for L in layers])

    r = dict(
        name=name, keep_n=keep_n, window=window, n_seg=len(segs),
        seg_k90_med=statistics.median(per_seg_k90_med),
        seg_gini_med=statistics.median(per_seg_gini_med),
        instant_ws_med=statistics.median(per_seg_instant_ws),
        instant_ws_max=max(per_seg_instant_ws),
        instant_ws_p90=sorted(per_seg_instant_ws)[int(0.9*(len(per_seg_instant_ws)-1))],
        instant_used_med=statistics.median(per_seg_instant_used),
        union_k90=union_k90, union_used=union_used, whole_k90_med=whole_k90_med,
        shift_jac_med=(statistics.median(shift_jac) if shift_jac else float('nan')),
        shift_jac_min=(min(shift_jac) if shift_jac else float('nan')),
        fits=all(w <= VRAM_SLOTS for w in per_seg_instant_ws),
        fits_med=(statistics.median(per_seg_instant_ws) <= VRAM_SLOTS),
    )
    return r

def fmt(r):
    return (
f"""[{r['name']}]  keep_n={r['keep_n']}  window={r['window']}tok  n_seg={r['n_seg']}
  CONCENTRATION  per-seg k90/layer median = {r['seg_k90_med']:.1f} of {r['keep_n']}   (whole-trace k90/layer = {r['whole_k90_med']:.1f})   per-seg Gini median = {r['seg_gini_med']:.2f}
  HOT-CORE SHIFT consecutive-seg Jaccard(top6) median = {r['shift_jac_med']:.2f}  min = {r['shift_jac_min']:.2f}   (1.0=frozen core, 0.0=full rotation)
  INSTANT WS     sum per-layer k90: median = {r['instant_ws_med']:.0f}  p90 = {r['instant_ws_p90']:.0f}  max = {r['instant_ws_max']:.0f}   vs budget {VRAM_SLOTS}
  UNION (wall)   whole-trace sum per-layer k90 = {r['union_k90']}   distinct-used = {r['union_used']}
  VERDICT        instant-median fits 394? {'YES' if r['fits_med'] else 'NO'}   every-seg fits? {'YES' if r['fits'] else 'NO'}"""
    )

# ---- entropy alignment (cyberpunk) -------------------------------------------
def entropy_alignment(window=20, M=6):
    # entropy per position
    ent = {}
    with open(CYBER_CONF) as f:
        for row in csv.DictReader(f):
            ent[int(row['pos'])] = float(row['entropy'])
    route = load_route(CYBER_ROUTE, keep=None)  # full route, all experts
    positions = sorted(route)
    # restrict to the pre-lock span (healthy + collapse onset), avoid the long
    # terminal repetition lock which is a degenerate single "phase".
    positions = [p for p in positions if p <= 260]
    segs = [positions[i:i+window] for i in range(0, len(positions), window)]
    seg_counts = [counters_for(s, route, KEEP_LAYERS) for s in segs]
    rows = []
    for i in range(1, len(segs)):
        a, b = seg_counts[i-1], seg_counts[i]
        jl = []
        for L in KEEP_LAYERS:
            sa = set(x for x, _ in a[L].most_common(M))
            sb = set(x for x, _ in b[L].most_common(M))
            if sa or sb: jl.append(len(sa & sb) / len(sa | sb))
        shift = 1 - statistics.mean(jl)          # 1 = full hot-core rotation
        seg_ent = statistics.mean(ent.get(p, 0.0) for p in segs[i])
        rows.append((segs[i][0], segs[i][-1], shift, seg_ent))
    # correlation shift vs entropy
    xs = [r[2] for r in rows]; ys = [r[3] for r in rows]
    corr = _pearson(xs, ys)
    return rows, corr

def _pearson(x, y):
    n = len(x)
    if n < 2: return float('nan')
    mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    vx = sum((a-mx)**2 for a in x); vy = sum((b-my)**2 for b in y)
    return cov / ((vx*vy) ** 0.5) if vx > 0 and vy > 0 else float('nan')

# ---- run ---------------------------------------------------------------------
if __name__ == '__main__':
    print(f"VRAM budget = {VRAM_SLOTS} slots ; traced MoE layers = {N_TRACED_LAYERS} (3..42)")
    print(f"whole-trace union working-set = keep_n*43 canonical (K23=989, K12=516)\n")

    print("### WIDE-HEALTHY (coffee HTML, the pin_analysis traces) — window sweep")
    # a_coffee is the long healthy wide trace; analyze it standalone per window
    a_coffee = COFFEE[0]
    for W in (30, 50, 75):
        r23 = analyze('coffee/K23', load_route(a_coffee, K23), K23, 23, W)
        print(fmt(r23)); print()
    for W in (30, 50):
        r12 = analyze('coffee/K12', load_route(a_coffee, K12), K12, 12, W)
        print(fmt(r12)); print()

    print("### WIDE-COLLAPSE (cyberpunk K12, pre-lock span) — keep-filtered")
    cyber = load_route(CYBER_ROUTE, K12)
    cyber = {p: v for p, v in cyber.items() if p <= 260}   # pre-lock
    for W in (30, 50):
        r = analyze('cyber/K12', cyber, K12, 12, W)
        print(fmt(r)); print()

    print("### NARROW single-phase baseline (python / json) — keep-filtered (caveat: coffee-mask coverage ~0.18)")
    for nm, paths in (('python/K23', PYTHON), ('json/K23', JSON)):
        r = analyze(nm, load_route(paths[0], K23), K23, 23, 50)
        print(fmt(r)); print()

    print("### ENTROPY-GATE alignment (cyberpunk, full route, window=20)")
    rows, corr = entropy_alignment(window=20)
    print(f"per-boundary (start..end : hot-core-shift[1-Jac] : mean-entropy):")
    for a, b, sh, en in rows:
        star = '  <== SHIFT+SPIKE' if (sh > 0.6 and en > 0.5) else ''
        print(f"  {a:>4}..{b:<4}  shift={sh:.2f}  ent={en:.2f}{star}")
    print(f"\nPearson(hot-core-shift, entropy) = {corr:.2f}  (positive => entropy rises where the hot-core rotates)")
