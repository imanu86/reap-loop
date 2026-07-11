#!/usr/bin/env python3
"""Pin-subset viability: usage distribution of KEEP experts per layer.
Offline analysis of full-model route traces filtered to the K12/K23 keep masks.
"""
import csv, json, collections, math, os

ROOT='runs/ds4'
MASKS='runs/ds4/20260711_local_clean_lowK/masks'
K23=json.load(open(f'{MASKS}/sessK23.json'))['keep']
K12=json.load(open(f'{MASKS}/sessK12.json'))['keep']
K23={int(k):list(v) for k,v in K23.items()}
K12={int(k):list(v) for k,v in K12.items()}
KEEP_LAYERS=sorted(K23)              # 3..42 (40 layers)
N_MOE_LAYERS=43                      # boot-probe (working-set uses this)
VRAM_SLOTS=394                       # boot-probe cache slots

COFFEE=[  # coffee-family (the session the mask was built from)
 'runs/ds4/20260710_pod_cache1024_warmup_replay/W130/route_W130.csv',
 'runs/ds4/20260710_pod_cache1024_warmup_replay/W50/route_W50.csv',
 'runs/ds4/20260711_podA_narrow_traces/a_coffee_full/route.csv',
]
PYTHON=['runs/ds4/20260711_podA_narrow_traces/c_python_full/route.csv',
        'runs/ds4/20260711_podA_narrow_traces/c2_python_long_full/route.csv']
JSON=['runs/ds4/20260711_podA_narrow_traces/b_json_full/route.csv',
      'runs/ds4/20260711_podA_narrow_traces/b2_json_long_full/route.csv']

def load_counts(paths, keep, weighted=False):
    """Return dict[layer] -> Counter(expert -> count/weight), restricted to keep."""
    per=collections.defaultdict(collections.Counter)
    for p in paths:
        with open(p) as f:
            r=csv.DictReader(f)
            for row in r:
                L=int(row['layer'])
                if L not in keep: continue
                ks=set(keep[L]); n=int(row['n'])
                for i in range(n):
                    e=int(row[f'e{i}'])
                    if e in ks:
                        w=float(row[f'w{i}']) if weighted else 1.0
                        per[L][e]+=w
    return per

def cov90(counter):
    """min experts to reach 90% of mass; returns (k90, total, top6share)."""
    tot=sum(counter.values())
    if tot==0: return 0,0,0.0
    vals=sorted(counter.values(), reverse=True)
    c=0; k=0
    for v in vals:
        c+=v; k+=1
        if c>=0.90*tot: break
    top6=sum(vals[:6])/tot
    return k, tot, top6

def gini(counter, keep_n):
    """Gini over the keep_n slots (padding missing keep experts with 0)."""
    vals=sorted(list(counter.values())+[0]*(keep_n-len(counter)))
    n=len(vals); s=sum(vals)
    if s==0 or n==0: return 0.0
    cum=sum((i+1)*v for i,v in enumerate(vals))
    return (2*cum)/(n*s) - (n+1)/n

def analyze(name, paths, keep, keep_n):
    per=load_counts(paths, keep)
    rows=[]
    for L in KEEP_LAYERS:
        k90,tot,top6=cov90(per[L])
        used=len(per[L])
        rows.append((L,k90,used,keep_n,tot,top6,gini(per[L],keep_n)))
    k90s=[r[1] for r in rows]
    useds=[r[2] for r in rows]
    top6s=[r[5] for r in rows]
    ginis=[r[6] for r in rows]
    print(f"\n=== {name} (keep_n={keep_n}) ===")
    print(f"per-layer k90 (experts to cover 90% of keep-activations): "
          f"min={min(k90s)} median={sorted(k90s)[len(k90s)//2]} max={max(k90s)} mean={sum(k90s)/len(k90s):.1f}")
    print(f"per-layer #keep-experts actually used: "
          f"min={min(useds)} median={sorted(useds)[len(useds)//2]} max={max(useds)} (of keep_n={keep_n})")
    print(f"per-layer top-6 share: min={min(top6s):.2f} median={sorted(top6s)[len(top6s)//2]:.2f} max={max(top6s):.2f}")
    print(f"per-layer Gini(keep slots): median={sorted(ginis)[len(ginis)//2]:.3f}")
    return per, rows

def budget_alloc(per, keep_n, label):
    """Global greedy: pool all (layer,expert) keep-activations, pin hottest until VRAM_SLOTS."""
    pool=[]
    for L in KEEP_LAYERS:
        for e,c in per[L].items():
            pool.append((c,L,e))
    pool.sort(reverse=True)
    total=sum(c for c,_,_ in pool)
    ws=keep_n*N_MOE_LAYERS  # working set = K * 43
    # cumulative hit at VRAM_SLOTS
    cum=0; hit_at_budget=None; slots_for_90=None
    for i,(c,L,e) in enumerate(pool,1):
        cum+=c
        if i==VRAM_SLOTS:
            hit_at_budget=cum/total
        if slots_for_90 is None and cum>=0.90*total:
            slots_for_90=i
    if hit_at_budget is None:  # fewer distinct pairs than budget
        hit_at_budget=1.0
    print(f"\n--- BUDGET ({label}, keep_n={keep_n}) ---")
    print(f"working-set (K*43) = {ws} slots ; distinct hot (layer,expert) pairs observed = {len(pool)}")
    print(f"VRAM budget = {VRAM_SLOTS} slots -> hit-rate at budget = {hit_at_budget:.3f}")
    print(f"slots needed for hit>=90% = {slots_for_90}  ({'FITS' if slots_for_90 and slots_for_90<=VRAM_SLOTS else 'DOES NOT FIT'} in {VRAM_SLOTS})")
    # also per-layer proportional cov90 total
    perlayer90=sum(cov90(per[L])[0] for L in KEEP_LAYERS)
    print(f"sum of per-layer k90 = {perlayer90} slots (independent per-layer pin to hit 90%/layer)")
    return hit_at_budget, slots_for_90, len(pool), perlayer90

def stability(paths, keep, keep_n, name):
    """Split each trace in half by position; compare per-layer hot-topM sets (Jaccard)."""
    # aggregate positions
    import statistics
    first=collections.defaultdict(collections.Counter)
    second=collections.defaultdict(collections.Counter)
    for p in paths:
        poss=[]
        with open(p) as f:
            r=list(csv.DictReader(f))
        positions=sorted(set(int(x['pos']) for x in r))
        mid=positions[len(positions)//2]
        for row in r:
            L=int(row['layer'])
            if L not in keep: continue
            ks=set(keep[L]); n=int(row['n']); pos=int(row['pos'])
            tgt=first if pos<mid else second
            for i in range(n):
                e=int(row[f'e{i}'])
                if e in ks: tgt[L][e]+=1
    jac=[]
    M=6
    for L in KEEP_LAYERS:
        a=set(x for x,_ in first[L].most_common(M))
        b=set(x for x,_ in second[L].most_common(M))
        if a or b:
            jac.append(len(a&b)/len(a|b))
    print(f"\n=== STABILITY {name} (top-{M} hot set, first vs second half) ===")
    print(f"per-layer Jaccard: min={min(jac):.2f} median={statistics.median(jac):.2f} mean={sum(jac)/len(jac):.2f}")
    return jac

def cross_domain(a_per, b_per, name, M=8):
    """Jaccard of top-M hot keep experts, domain A vs domain B, per layer."""
    import statistics
    jac=[]
    for L in KEEP_LAYERS:
        a=set(x for x,_ in a_per[L].most_common(M))
        b=set(x for x,_ in b_per[L].most_common(M))
        if a or b: jac.append(len(a&b)/len(a|b))
    print(f"\n=== CROSS-DOMAIN DRIFT {name} (top-{M}) === per-layer Jaccard: "
          f"min={min(jac):.2f} median={statistics.median(jac):.2f} mean={sum(jac)/len(jac):.2f}")
    return jac

print("LAYERS:",KEEP_LAYERS[0],"..",KEEP_LAYERS[-1],f"({len(KEEP_LAYERS)} masked layers; boot-probe N_MOE={N_MOE_LAYERS})")
p23,_=analyze("COFFEE / K23", COFFEE, K23, 23)
p12,_=analyze("COFFEE / K12", COFFEE, K12, 12)
budget_alloc(p23,23,"COFFEE K23")
budget_alloc(p12,12,"COFFEE K12")
stability(COFFEE,K23,23,"COFFEE K23")
# cross-domain
py23=load_counts(PYTHON,K23); js23=load_counts(JSON,K23)
cross_domain(p23,py23,"coffee-vs-python K23")
cross_domain(p23,js23,"coffee-vs-json K23")
