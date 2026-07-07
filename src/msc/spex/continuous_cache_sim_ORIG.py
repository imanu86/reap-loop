"""LOOP CONTINUO con VERIFICATORE+TEMPERATURA + CONTROLLO RANDOM.
Policy a confronto (miss-rate per layer, piu' basso = meglio):
  RANDOM   = cache con EVICTION CASUALE, nessun riuso ne' predizione -> pavimento-caso (~1 - C/E).
  REACTIVE = LRU (tieni i caldi appena usati) -> cattura il riuso temporale.
  MARKOV   = prefetch predittivo cieco (naive).
  ADAPTIVE = verificatore: misura l'errore di predizione (confidenza EMA) e modula n_pred = round(C*conf).
Decomposizione: REACTIVE-RANDOM = riuso temporale; ADAPTIVE-RANDOM = movimento totale (intelligenza+temperatura).
Uso: python continuous_cache_sim.py traces_q30_general.npz
"""
import sys, random, numpy as np
from collections import OrderedDict

def build_markov(E, split, L, nexp, k):
    Cm = [np.zeros((nexp, nexp)) for _ in range(L)]
    for l in range(L - 1):
        for c1 in range(k):
            for c2 in range(k):
                np.add.at(Cm[l], (E[:split, l, c1], E[:split, l + 1, c2]), 1.0)
    return Cm

def simulate(E, Cm, C, policy, split, n_tok, L, nexp, k, alpha=0.1, seed=0):
    rng = random.Random(seed)
    caches = [OrderedDict() for _ in range(L)]
    last_pred = [set() for _ in range(L)]
    conf = np.full(L, 0.5)
    miss = 0; total = 0
    def evict(cache):
        while len(cache) > C:
            if policy == 'random':
                cache.pop(rng.choice(list(cache.keys())))
            else:
                cache.popitem(last=False)   # LRU
    for t in range(split, n_tok):
        for l in range(L):
            needed = E[t, l]; cache = caches[l]
            if policy == 'adaptive' and last_pred[l]:
                conf[l] = (1 - alpha) * conf[l] + alpha * (sum(1 for e in needed if e in last_pred[l]) / k)
            for e in needed:
                total += 1
                if e in cache:
                    if policy != 'random': cache.move_to_end(e)
                else:
                    miss += 1; cache[e] = None
            evict(cache)
            if l < L - 1 and policy in ('markov', 'adaptive'):
                n_pred = C if policy == 'markov' else int(round(C * conf[l + 1]))
                if n_pred > 0:
                    pred = np.argpartition(-Cm[l][needed].sum(0), min(n_pred, nexp) - 1)[:n_pred]
                    nc = caches[l + 1]
                    for e in pred.tolist():
                        if e in nc: nc.move_to_end(e)
                        else: nc[e] = None
                    evict(nc)
                    last_pred[l + 1] = set(pred.tolist())
                else:
                    last_pred[l + 1] = set()
    return miss / total, float(conf.mean())

def run(path, C_list=(8, 12, 16, 24, 32, 48), n_tok=4000):
    d = np.load(path); E = d['experts'].astype(int); T, L, k = E.shape; nexp = int(d['n_experts'])
    n_tok = min(n_tok, T); E = E[:n_tok]; split = n_tok // 2
    Cm = build_markov(E, split, L, nexp, k)
    print(f"\n{path}: L={L} E={nexp} k={k} tok_eval={n_tok-split}")
    print(f"  {'C/lay':>5} {'%':>4} | {'RANDOM':>7} {'REACTIVE':>8} {'ADAPTIVE':>8} | {'React-Rnd':>9} {'Adapt-Rnd':>9} | {'conf':>4}")
    for C in C_list:
        rnd, _ = simulate(E, Cm, C, 'random', split, n_tok, L, nexp, k)
        r, _ = simulate(E, Cm, C, 'reactive', split, n_tok, L, nexp, k)
        a, cf = simulate(E, Cm, C, 'adaptive', split, n_tok, L, nexp, k)
        print(f"  {C:>5} {100*C/nexp:>3.0f}% | {rnd:>7.3f} {r:>8.3f} {a:>8.3f} | {rnd-r:>+9.3f} {rnd-a:>+9.3f} | {cf:>4.2f}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        run(p)
