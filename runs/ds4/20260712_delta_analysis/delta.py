#!/usr/bin/env python3
# Il delta di massa e' un segnale predittivo? Test offline su trace_K91 (cyberpunk, ground-truth demand).
# Confronta il keep-set della mask ordinato per MASSA vs MASSA+alpha*DELTA.
# Metrica = MISS: esperti che il modello VUOLE (demand K91) ma la mask blocca. Meno miss = meglio.
# Focus sui token di CAMBIO-FASE (churn alto) = dove nasce la degradazione.
import csv, sys, collections

TRACE = r"C:\Users\imanu\source\repos\reap-loop\runs\ds4\20260711_highK_sweetspot\traces\trace_K91\route.csv"
K = 23          # larghezza keep (il nostro target)
W = 10          # finestra massa (token)
D = 3           # orizzonte delta (token)
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]   # peso del delta (0 = solo massa = baseline)

# 1) parse: per (pos, layer) -> {expert: weight}. Header: pos,layer,n,e0..e5,w0..w5
per = collections.defaultdict(dict)         # (pos,layer) -> {e:w}
demand = collections.defaultdict(dict)      # stessa cosa (la domanda vera del K91)
layers = set(); positions = set()
with open(TRACE, encoding="utf-8") as f:
    r = csv.reader(f); next(r, None)
    for row in r:
        if len(row) < 15: continue
        try:
            pos = int(row[0]); layer = int(row[1]); n = int(row[2])
            es = [int(row[3+i]) for i in range(n)]
            ws = [float(row[9+i]) for i in range(n)]   # 6 col expert (3..8) + 6 col weight (9..14)
        except Exception:
            continue
        d = {e: w for e, w in zip(es, ws)}
        per[(pos, layer)] = d; demand[(pos, layer)] = d
        layers.add(layer); positions.add(pos)

layers = sorted(l for l in layers if l >= 3)   # solo layer MoE
positions = sorted(positions)
print(f"trace: {len(positions)} token, {len(layers)} layer MoE, K={K} W={W} D={D}")

# 2) per ogni layer: mass windowed + delta, poi simula il keep-set e conta i miss
def run(alpha):
    total_miss = 0; total_demand = 0
    churn_miss = 0; churn_demand = 0; churn_tokens = 0
    for L in layers:
        ring = collections.deque()          # ultimi W token: lista di dict {e:w}
        mass = collections.Counter()        # massa corrente per expert
        hist_mass = collections.deque()     # snapshot di mass D-passi fa (per delta)
        prev_keep = None
        prev_demand_set = None
        for pos in positions:
            d = per.get((pos, L))
            if d is None:
                continue
            # --- keep-set per QUESTO token = deciso dal rating FINO a t-1 (mass/delta correnti) ---
            # delta = mass ora - mass D-token fa
            old = hist_mass[0] if len(hist_mass) >= D else collections.Counter()
            if prev_keep is not None:
                score = {}
                cand = set(mass) | set(old)
                for e in cand:
                    m = mass.get(e, 0.0)
                    dl = m - old.get(e, 0.0)
                    score[e] = m + alpha * dl
                keep = set(sorted(score, key=lambda e: -score[e])[:K])
            else:
                keep = set(sorted(mass, key=lambda e: -mass.get(e, 0.0))[:K])
            # --- domanda vera a questo token ---
            dem = set(d.keys())
            miss = len(dem - keep)
            total_miss += miss; total_demand += len(dem)
            # churn = quanto la domanda e' cambiata dal token prima (cambio di fase)
            if prev_demand_set is not None:
                ch = len(dem - prev_demand_set)
                if ch >= max(2, len(dem)//2):   # >=meta' della domanda e' nuova => transizione
                    churn_tokens += 1
                    churn_miss += miss; churn_demand += len(dem)
            prev_demand_set = dem
            # --- ora aggiorna il rating CON questo token (per il prossimo) ---
            hist_mass.append(mass.copy())
            if len(hist_mass) > D: hist_mass.popleft()
            ring.append(d)
            for e, w in d.items(): mass[e] += w
            if len(ring) > W:
                oldd = ring.popleft()
                for e, w in oldd.items():
                    mass[e] -= w
                    if mass[e] <= 1e-9: del mass[e]
            prev_keep = keep
    mr = 100.0*total_miss/max(1,total_demand)
    cmr = 100.0*churn_miss/max(1,churn_demand)
    return mr, cmr, churn_tokens

print(f"\n{'alpha':>6} | {'miss% TUTTI':>12} | {'miss% CAMBIO-FASE':>18}")
print("-"*44)
base_all = base_ch = None
for a in ALPHAS:
    mr, cmr, ct = run(a)
    if base_all is None: base_all, base_ch = mr, cmr
    da = mr-base_all; dc = cmr-base_ch
    print(f"{a:>6.1f} | {mr:>11.2f}% | {cmr:>17.2f}%   (vs base: {da:+.2f} / {dc:+.2f})")
print(f"\n(token di cambio-fase individuati: {ct})")
print("Se miss% CAMBIO-FASE scende con alpha>0 => il DELTA e' un segnale predittivo utile.")
