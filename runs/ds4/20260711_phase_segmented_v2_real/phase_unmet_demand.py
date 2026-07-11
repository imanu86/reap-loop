#!/usr/bin/env python3
"""Phase-segmentation v2 on the REAL unmasked K0 full-model trace + per-phase
UNMET-DEMAND map = the targeting spec for a phase-adaptive admission controller.

Removes the proxy caveat of v1 (20260711_phase_segmented_usage): v1 segmented
FULL-model traces of OTHER prompts (coffee/python/json) FILTERED TO KEEP first,
so it structurally never saw demand outside the keep. Here we use the REAL
full-router trace (K0, no mask, all 256 experts eligible, native top-6, weighted)
of the SAME cyberpunk prompt that collapses under mask, and read the router's TRUE
per-phase demand -> exactly which experts the coffee-tuned masks prune away.

DATA  : runs/ds4/20260711_k0_fullmodel_baseline/route_k0_cyberpunk.csv.gz
        159,960 rows = 3999 decode tok x 40 MoE layers[3..42] x top-6 (weighted).
        route_k0_coffee.csv.gz = in-domain control (same prompt family the mask
        was built on).
MASKS : runs/ds4/20260711_local_clean_lowK/masks/sess{K12,K23,K38}.json
        method=session_mass_rank on the COFFEE W50 phase-2 trace (frontpage HTML).

PHASES: real HTML regions of gen_k0_cyberpunk.txt. No per-token text file exists,
so boundaries are token positions from char-offset of each structural marker
(<!DOCTYPE / <style> / <body> / <script>) mapped proportionally onto the decode
token axis (15012 chars over 3999 tok, pos 65..4063). Cross-check: char-prop
body_start=3051 vs task hint ~3200 (CSS tokenizes denser); +/-150 tok robustness
sweep emitted; conclusions are stable across it.

Metrics are WEIGHTED: each (token,layer) contributes total mass 1, split over its
top-6 experts by normalized gate weight. "demand" = summed weighted mass.

Offline, no GPU. Reproduce: python runs/ds4/20260711_phase_segmented_v2_real/phase_unmet_demand.py
"""
import csv, gzip, json, collections, statistics

RID   = 'runs/ds4/20260711_k0_fullmodel_baseline'
K0CY  = f'{RID}/route_k0_cyberpunk.csv.gz'
K0CO  = f'{RID}/route_k0_coffee.csv.gz'
MASKS = 'runs/ds4/20260711_local_clean_lowK/masks'
VRAM_SLOTS = 394
N_EXPERT   = 256
W = 50                      # sub-phase window for instantaneous residency

def load_keep(tag):
    d = json.load(open(f'{MASKS}/sess{tag}.json'))['keep']
    return {int(k): set(int(x) for x in v) for k, v in d.items()}
K12 = load_keep('K12'); K23 = load_keep('K23'); K38 = load_keep('K38')
LAYERS = sorted(K23)

# ---- phase boundaries (decode token positions) -------------------------------
POS0, POSN, NTOK, NCHAR = 65, 4063, 3999, 15012
def c2p(c): return POS0 + round(c / NCHAR * NTOK)
CSS_START  = c2p(674)      # <style>   ~245
BODY_START = c2p(11211)    # <body>    ~3051 (hint ~3200)
JS_START   = c2p(14697)    # <script>  ~3980
def phases(body_start=BODY_START):
    return [('head', POS0, CSS_START), ('css', CSS_START, body_start),
            ('body', body_start, JS_START), ('js', JS_START, POSN + 1)]

# ---- load trace as per-position per-layer weighted expert dict ---------------
def load_trace(path):
    """pos -> layer -> {expert: weighted mass (per-token-layer sums to 1)}."""
    T = collections.defaultdict(lambda: {L: collections.defaultdict(float) for L in LAYERS})
    with gzip.open(path, 'rt') as f:
        r = csv.reader(f); next(r)
        for row in r:
            pos = int(row[0]); L = int(row[1]); n = int(row[2])
            es = [int(row[3 + i]) for i in range(n)]
            ws = [float(row[9 + i]) for i in range(n)]
            sw = sum(ws) or 1.0
            d = T[pos][L]
            for e, w in zip(es, ws):
                d[e] += w / sw
    return T

def agg(T, positions):
    """aggregate weighted demand over a set of positions -> layer -> {expert: mass}."""
    per = {L: collections.defaultdict(float) for L in LAYERS}
    for p in positions:
        lp = T.get(p)
        if lp is None: continue
        for L in LAYERS:
            for e, m in lp[L].items():
                per[L][e] += m
    return per

# ---- metrics -----------------------------------------------------------------
def k90(dist):
    tot = sum(dist.values())
    if tot == 0: return 0
    c = k = 0
    for v in sorted(dist.values(), reverse=True):
        c += v; k += 1
        if c >= 0.90 * tot: break
    return k

def gini(dist, n=N_EXPERT):
    vals = sorted(list(dist.values()) + [0.0] * (n - len(dist)))
    s = sum(vals)
    if s == 0: return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * cum) / (n * s) - (n + 1) / n

def coverage(dist, keepL):
    tot = sum(dist.values())
    return (sum(m for e, m in dist.items() if e in keepL) / tot) if tot else 1.0

def admit_to(dist, keepL, target):
    """min experts OUTSIDE keep to bring coverage to `target`."""
    tot = sum(dist.values())
    if tot == 0: return 0
    cov = sum(m for e, m in dist.items() if e in keepL)
    if cov >= target * tot: return 0
    add = 0
    for m in sorted((m for e, m in dist.items() if e not in keepL), reverse=True):
        cov += m; add += 1
        if cov >= target * tot: break
    return add

def k_to(dist, target):
    """min experts (any, mass desc) to reach `target` coverage of dist."""
    tot = sum(dist.values())
    if tot == 0: return 0
    c = k = 0
    for v in sorted(dist.values(), reverse=True):
        c += v; k += 1
        if c >= target * tot: break
    return k

def topM(dist, M=6):
    return set(e for e, _ in collections.Counter(dist).most_common(M))

def windows(positions, w=W):
    ps = sorted(positions)
    return [ps[i:i + w] for i in range(0, len(ps), w)]

# ---- whole-trace coverage (for in-domain target) -----------------------------
def whole_coverage(T, keep):
    per = agg(T, list(T.keys()))
    tot = sum(sum(per[L].values()) for L in LAYERS)
    cov = sum(sum(m for e, m in per[L].items() if e in keep[L]) for L in LAYERS)
    return cov / tot

# ---- main per-prompt analysis ------------------------------------------------
def analyze(T, label, PH, targets):
    order = [nm for nm, _, _ in PH]
    pos_by = {nm: [p for p in T if a <= p < b] for nm, a, b in PH}
    demand = {nm: agg(T, pos_by[nm]) for nm in order}

    # whole-trace (aggregate) concentration -- the artifact v1 measured
    whole = agg(T, list(T.keys()))
    res = dict(label=label, order=order, ntok={nm: len(pos_by[nm]) for nm in order},
               whole_k90=statistics.median(k90(whole[L]) for L in LAYERS),
               whole_gini=statistics.median(gini(whole[L]) for L in LAYERS),
               phase={})

    for nm in order:
        d = demand[nm]
        k90s = [k90(d[L]) for L in LAYERS]
        rec = dict(k90_med=statistics.median(k90s), gini_med=statistics.median(gini(d[L]) for L in LAYERS),
                   instant_ws_cum=sum(k90s))
        # windowed (instantaneous) residency + admit over W-token sub-windows
        wins = windows(pos_by[nm])
        wws, wadm90, wadmT = [], [], {t: [] for t in targets}
        wj = []  # within-phase consecutive-window Jaccard (top-6)
        prev = None
        for win in wins:
            wd = agg(T, win)
            wws.append(sum(k90(wd[L]) for L in LAYERS))
            wadm90.append(sum(admit_to(wd[L], K23[L], 0.90) for L in LAYERS))
            for t, cov in targets.items():
                wadmT[t].append(sum(admit_to(wd[L], K23[L], cov) for L in LAYERS))
            cur = {L: topM(wd[L]) for L in LAYERS}
            if prev is not None:
                jl = [len(prev[L] & cur[L]) / len(prev[L] | cur[L]) for L in LAYERS if prev[L] or cur[L]]
                wj.append(statistics.mean(jl))
            prev = cur
        def med(x): return statistics.median(x) if x else 0
        def p90(x): return sorted(x)[int(0.9 * (len(x) - 1))] if x else 0
        # windowed residency to reach only the IN-DOMAIN service level (target)
        idm_cov = list(targets.values())[0]
        widm = [sum(k_to(agg(T, win)[L], idm_cov) for L in LAYERS) for win in wins]
        rec['win_idm_ws_med'] = med(widm); rec['win_idm_ws_max'] = max(widm) if widm else 0
        rec['win_ws_med'] = med(wws); rec['win_ws_p90'] = p90(wws); rec['win_ws_max'] = max(wws) if wws else 0
        rec['win_adm90_med'] = med(wadm90); rec['win_adm90_max'] = max(wadm90) if wadm90 else 0
        rec['within_jac'] = statistics.mean(wj) if wj else float('nan')
        # per-keep unmet + phase-level admit
        for tag, keep in (('K23', K23), ('K12', K12)):
            covs = [coverage(d[L], keep[L]) for L in LAYERS]
            adm90 = [admit_to(d[L], keep[L], 0.90) for L in LAYERS]
            rec[tag] = dict(unmet=100 * (1 - statistics.mean(covs)),
                            adm90_med=statistics.median(adm90), adm90_sum=sum(adm90))
            for t, cov in targets.items():
                admT = [admit_to(d[L], keep[L], cov) for L in LAYERS]
                rec[tag][f'admT_{t}_med'] = statistics.median(admT)
                rec[tag][f'admT_{t}_sum'] = sum(admT)
                rec[tag][f'admT_{t}_max'] = max(admT)
        # windowed target-admit (K23) using in-domain target
        rec['win_admT_med'] = med(wadmT[list(targets)[0]])
        rec['win_admT_max'] = max(wadmT[list(targets)[0]]) if wadmT[list(targets)[0]] else 0
        res['phase'][nm] = rec

    # between-phase hot-core shift (full router top-6), consecutive phases
    res['between'] = []
    for a, b in zip(order, order[1:]):
        jl = [len(topM(demand[a][L]) & topM(demand[b][L])) / len(topM(demand[a][L]) | topM(demand[b][L]))
              for L in LAYERS]
        res['between'].append((a, b, statistics.mean(jl)))
    return res

def report(o, tname):
    P = []
    P.append(f"===== {o['label']} =====")
    P.append(f"phase tok: " + "  ".join(f"{nm}={o['ntok'][nm]}" for nm in o['order']))
    P.append(f"WHOLE-TRACE aggregate: k90/layer med={o['whole_k90']:.0f}  Gini med={o['whole_gini']:.2f}   "
             f"(v1 proxy measured THIS, but keep-filtered -> saw only 9/layer)")
    P.append("")
    P.append("[A] PER-PHASE concentration (full router, /256) + hot-core stability")
    P.append(f"  {'phase':5} {'k90/L med':>9} {'Gini':>5} | {'within-phase Jac':>16} (window={W}: 1=frozen core inside phase)")
    for nm in o['order']:
        r = o['phase'][nm]
        P.append(f"  {nm:5} {r['k90_med']:9.0f} {r['gini_med']:5.2f} | {r['within_jac']:16.2f}")
    P.append("  BETWEEN-phase hot-core shift (Jaccard top-6, consecutive phases):")
    for a, b, j in o['between']:
        P.append(f"      {a:>4} -> {b:<4}  Jaccard={j:.2f}  ({1-j:.2f} rotates)")
    P.append("")
    P.append("[B] UNMET DEMAND per phase = % of phase demand on experts the mask PRUNES")
    P.append(f"  {'phase':5} {'K23 unmet%':>10} {'K12 unmet%':>10} | admit/L to reach targets (K23):")
    tks = list(o['phase'][o['order'][0]].keys())
    P.append(f"  {'':5} {'':10} {'':10} | {'to-90%(med/sum)':>16} {'to-indomain(med/sum/max)':>26}")
    for nm in o['order']:
        r = o['phase'][nm]
        k23 = r['K23']
        idk = [k for k in k23 if k.startswith('admT_') and k.endswith('_med')][0]
        t = idk[len('admT_'):-len('_med')]
        P.append(f"  {nm:5} {k23['unmet']:10.1f} {r['K12']['unmet']:10.1f} | "
                 f"{k23['adm90_med']:6.0f}/{k23['adm90_sum']:<8d} "
                 f"{k23[f'admT_{t}_med']:6.0f}/{k23[f'admT_{t}_sum']:<5d}/{k23[f'admT_{t}_max']:<4d}")
    P.append(f"  (in-domain target coverage = coffee whole-trace K23 coverage, the displacement the mask was built to absorb)")
    P.append("")
    P.append("[B] INSTANTANEOUS residency vs 394 budget (window=%d, full router, sum_L k):" % W)
    P.append(f"  {'phase':5} | to-90%(med/p90/max) | to-INDOMAIN(med/max) | fits394 @90%? @indom?")
    for nm in o['order']:
        r = o['phase'][nm]
        f90 = 'YES' if r['win_ws_med'] <= VRAM_SLOTS else 'NO'
        fid = 'YES' if r['win_idm_ws_med'] <= VRAM_SLOTS else 'NO'
        P.append(f"  {nm:5} | {r['win_ws_med']:6.0f}/{r['win_ws_p90']:4.0f}/{r['win_ws_max']:<4.0f} | "
                 f"{r['win_idm_ws_med']:6.0f}/{r['win_idm_ws_max']:<4.0f} | "
                 f"@90%={f90:3}  @indom={fid}")
    P.append("")
    return "\n".join(P)

if __name__ == '__main__':
    print("PHASE-SEGMENTATION v2 on REAL unmasked K0 trace + per-phase unmet-demand map\n")
    print(f"phase bounds(tok): head[{POS0}..{CSS_START}) css[{CSS_START}..{BODY_START}) "
          f"body[{BODY_START}..{JS_START}) js[{JS_START}..{POSN}]   window={W}\n")

    Tco = load_trace(K0CO)
    cov_co_k23 = whole_coverage(Tco, K23); cov_co_k12 = whole_coverage(Tco, K12)
    print(f"COFFEE in-domain control (mask's own session): whole-trace K23 coverage={100*cov_co_k23:.1f}% "
          f"(unmet {100*(1-cov_co_k23):.1f}%)  K12 coverage={100*cov_co_k12:.1f}% (unmet {100*(1-cov_co_k12):.1f}%)")
    # coffee windowed residency reference (in-domain, does IT fit 394?)
    co_wins = windows(list(Tco.keys()))
    co_w90 = [sum(k90(agg(Tco, w)[L]) for L in LAYERS) for w in co_wins]
    co_widm = [sum(k_to(agg(Tco, w)[L], cov_co_k23) for L in LAYERS) for w in co_wins]
    print(f"  COFFEE window={W} residency: to-90% med={statistics.median(co_w90):.0f}  "
          f"to-indomain(58%) med={statistics.median(co_widm):.0f}  vs 394 budget")
    print("  -> in-domain displacement the mask absorbs while staying usable; used as the admit target.\n")

    Tcy = load_trace(K0CY)
    targets = {'idm': cov_co_k23}   # admit-to-in-domain(coffee) coverage
    o = analyze(Tcy, 'CYBERPUNK (collapse-prompt, full router)', phases(), targets)
    print(report(o, 'cyberpunk'))

    # ---- emit the concrete targeting map: phase -> per-layer experts to ADMIT
    #      (experts OUTSIDE K23 keep, mass-ranked, to lift phase coverage to the
    #       in-domain service level). This is the admission-controller spec.
    with open('runs/ds4/20260711_phase_segmented_v2_real/targeting_map_K23.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['phase', 'layer', 'n_admit', 'admit_experts(mass_desc)'])
        for nm, a, b in phases():
            dem = agg(Tcy, [p for p in Tcy if a <= p < b])
            for L in LAYERS:
                d = dem[L]; tot = sum(d.values())
                cov = sum(m for e, m in d.items() if e in K23[L])
                admits = []
                if tot and cov < cov_co_k23 * tot:
                    for e, m in sorted(((e, m) for e, m in d.items() if e not in K23[L]),
                                       key=lambda x: -x[1]):
                        admits.append(e); cov += m
                        if cov >= cov_co_k23 * tot: break
                wr.writerow([nm, L, len(admits), ' '.join(map(str, admits))])
    print("wrote targeting_map_K23.csv (phase,layer,n_admit,admit_experts)\n")

    print("### ROBUSTNESS: body_start sweep (char-prop 3051 vs hint ~3200)")
    for bs in (BODY_START - 150, BODY_START, BODY_START + 150):
        ob = analyze(Tcy, f'bs={bs}', phases(bs), targets)
        css = ob['phase']['css']; body = ob['phase']['body']
        print(f"  body_start={bs}: CSS K23-unmet={css['K23']['unmet']:.1f}% win_ws_med={css['win_ws_med']:.0f} | "
              f"BODY K23-unmet={body['K23']['unmet']:.1f}% win_ws_med={body['win_ws_med']:.0f}")
