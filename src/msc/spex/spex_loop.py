"""SPEX — loop di prefetch predittivo degli expert, FEDELE a DeepSeek DSpark (vettorizzato, markov + hidden).

Corregge continuous_cache_sim.py su tutti i gap verificati sul paper (STS vera, confidence head Eq.7,
ammissione Alg.1, split per-doc, reset per-doc, ablation senza-STS, multi-seed). Vedi docs/SPEX_LOOP.md.

Predittore selezionabile:
  --predictor markov : score = conteggi di transizione L->L+1 (dalle expert-ID della trace).
  --predictor hidden : score = logit della probe hidden per-token (dai file hidden_scores_*.npz, campo 'scores'
                       [T, L-1, E]); e' il +hidden (recall 0.93-0.99) innestato nel loop.
Un file hidden_scores_*.npz contiene SIA experts (per markov) SIA scores (per hidden) sugli STESSI token
-> confronto markov-vs-hidden equo con --predictor {markov,hidden} sullo stesso file.

Orizzonte 1 layer -> la STS calibra la temperatura PER-LAYER sulla confidence PER-CANDIDATO (adattamento
del prodotto-cumulato token-sequenziale di DSpark). Router mai bypassato -> accuratezza intatta.

Uso: python spex_loop.py models/spex/traces_q235_general.npz --caps 2,4,6,8,16 --seeds 0
     python spex_loop.py models/spex/hidden_scores_q30_gen.npz --predictor hidden --caps 2,4,6,8,16
"""
import sys, argparse, json, numpy as np
from collections import OrderedDict

try:
    from msc.spex.c_export import write_markov_spex
except ModuleNotFoundError:  # direct `python spex_loop.py ...` execution
    from c_export import write_markov_spex

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

# ----------------------------------------------------------------------------- dati + split per doc
def load(path):
    d = np.load(path)
    E = d['experts'].astype(np.int64)
    S = d['scores'].astype(np.float32) if 'scores' in d.files else None   # [T, L-1, E] per hidden
    doclens = d['doclens'].astype(np.int64) if 'doclens' in d.files else np.array([E.shape[0]])
    nexp = int(d['n_experts']); topk = E.shape[2]; L = E.shape[1]
    bounds = np.concatenate([[0], np.cumsum(doclens)])
    return E, S, bounds, nexp, topk, L

def split_docs(ndoc, seed, frac=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed); idx = rng.permutation(ndoc)
    a = int(round(frac[0]*ndoc)); b = a + int(round(frac[1]*ndoc))
    return idx[:a], idx[a:b], idx[b:]

def doc_tokens(E, S, bounds, ids, max_tok=None):
    """Ritorna lista di (D, Sd): D=[n,L,k] expert-ID, Sd=[n,L-1,E] scores hidden o None."""
    out, tot = [], 0
    for i in ids:
        D = E[bounds[i]:bounds[i+1]]
        Sd = S[bounds[i]:bounds[i+1]] if S is not None else None
        out.append((D, Sd)); tot += D.shape[0]
        if max_tok and tot >= max_tok: break
    return out

# ----------------------------------------------------------------------------- Markov (vettoriale)
def build_markov(docs, L, nexp, k):
    Ds = [D for D, _ in docs]
    if not Ds: return [np.zeros((nexp, nexp)) for _ in range(L)]
    A = np.concatenate(Ds, 0)
    C = [np.zeros((nexp, nexp)) for _ in range(L)]
    for l in range(L-1):
        cur, nxt = A[:, l, :], A[:, l+1, :]
        for c1 in range(k):
            for c2 in range(k):
                C[l] += np.bincount(cur[:, c1]*nexp + nxt[:, c2], minlength=nexp*nexp).reshape(nexp, nexp)
    return C

def lscore(D, Sd, C, l, pred):
    """[n,nexp] score per predire il layer l+1: markov (conteggi) o hidden (logit probe)."""
    if pred == 'hidden':
        return Sd[:, l, :]
    return C[l][D[:, l, :]].sum(1)

def hit_mat(nxt_ids, nexp):
    H = np.zeros((nxt_ids.shape[0], nexp), bool); np.put_along_axis(H, nxt_ids, True, 1); return H

# ----------------------------------------------------------------------------- confidence head + STS
FIT_CAP = 40000     # max punti (candidato,hit) per layer usati nel fit/STS (sottocampionati)

def _subsample(x, y, cap, seed=0):
    if len(x) <= cap: return x, y
    idx = np.random.default_rng(seed).choice(len(x), cap, replace=False)
    return x[idx], y[idx]

def fit_confidence(docs, C, L, nexp, k, topN, pred):
    a = np.ones(L); b = np.zeros(L)
    for l in range(L-1):
        F, Y = [], []
        for D, Sd in docs:
            sc = lscore(D, Sd, C, l, pred); H = hit_mat(D[:, l+1, :], nexp)
            idx = np.argpartition(-sc, min(topN, nexp)-1, axis=1)[:, :topN]
            F.append(np.take_along_axis(sc, idx, 1).ravel())          # feature grezza (logit-like)
            Y.append(np.take_along_axis(H, idx, 1).astype(float).ravel())
        if not F: continue
        x = np.concatenate(F).astype(np.float64); y = np.concatenate(Y)
        x = np.log1p(x) if pred == 'markov' else x                    # markov: log1p; hidden: gia' logit
        x, y = _subsample(x, y, FIT_CAP, seed=l)
        mu, sd = x.mean(), x.std() + 1e-6; xn = (x - mu) / sd
        A, B = 1.0, 0.0
        for _ in range(120):
            g = sigmoid(A*xn+B) - y; A -= 0.5*np.mean(g*xn); B -= 0.5*np.mean(g)
        a[l], b[l] = A/sd, B - A*mu/sd     # riporta ai coeff su x
    return a, b

def _feat(sc, idx, pred):
    v = np.take_along_axis(sc, idx, 1)
    return np.log1p(v) if pred == 'markov' else v

def collect_cand(docs, C, L, nexp, k, topN, a, b, pred):
    Z = [np.array([]) for _ in range(L)]; Y = [np.array([]) for _ in range(L)]
    for l in range(L-1):
        zs, ys = [], []
        for D, Sd in docs:
            sc = lscore(D, Sd, C, l, pred); H = hit_mat(D[:, l+1, :], nexp)
            idx = np.argpartition(-sc, min(topN, nexp)-1, axis=1)[:, :topN]
            zs.append((a[l]*_feat(sc, idx, pred)+b[l]).ravel())
            ys.append(np.take_along_axis(H, idx, 1).astype(float).ravel())
        if zs: Z[l], Y[l] = _subsample(np.concatenate(zs), np.concatenate(ys), FIT_CAP, seed=100+l)
    return Z, Y

def ece(probs, labels, M=15):
    probs = np.clip(probs, 0, 1); N = len(probs)
    if N == 0: return 0.0
    e = 0.0; edges = np.linspace(0, 1, M+1)
    for m in range(M):
        sel = (probs > edges[m]) & (probs <= edges[m+1]) if m > 0 else (probs >= 0) & (probs <= edges[1])
        if sel.sum(): e += (sel.sum()/N)*abs(labels[sel].mean() - probs[sel].mean())
    return e

def fit_sts(Z, Y, L, grid=None, M=15):
    if grid is None: grid = np.round(np.arange(0.4, 15.01, 0.2), 3)
    T = np.ones(L)
    for l in range(L-1):
        if len(Z[l]) == 0: continue
        best = (1e9, 1.0)
        for t in grid:
            e = ece(sigmoid(Z[l]/t), Y[l], M)
            if e < best[0]: best = (e, t)
        T[l] = best[1]
    return T

# ----------------------------------------------------------------------------- precompute predizioni per-doc
def precompute(docs, C, L, nexp, k, topN, a, b, T, pred):
    """Ritorna per-doc (needed[n][L] liste python, cand[l][t] liste, praw/psts[l][t] liste, n)."""
    out = []
    for D, Sd in docs:
        cand = [None]*L; praw = [None]*L; psts = [None]*L
        for l in range(L-1):
            sc = lscore(D, Sd, C, l, pred)
            idx = np.argpartition(-sc, min(topN, nexp)-1, axis=1)[:, :topN]
            sc_c = np.take_along_axis(sc, idx, 1); order = np.argsort(-sc_c, axis=1)
            f = np.take_along_axis(sc_c, order, 1); f = np.log1p(f) if pred == 'markov' else f
            z = a[l]*f + b[l]
            cand[l] = np.take_along_axis(idx, order, 1).tolist()          # liste python
            praw[l] = sigmoid(z).tolist(); psts[l] = sigmoid(z / (T[l+1] if T[l+1] > 0 else 1.0)).tolist()
        out.append((D.tolist(), cand, praw, psts, D.shape[0]))
    return out

# ----------------------------------------------------------------------------- il LOOP (una policy), tutto in liste python
def simulate(preds, cap, policy, L, k, tau=0.5, seed=0, alpha=0.1):
    rng = np.random.default_rng(seed); miss = 0; tot = 0
    predp = ('markov_naive', 'adaptive_crude', 'adaptive_raw', 'adaptive_dspark')
    crude = policy == 'adaptive_crude'; dspark = policy == 'adaptive_dspark'
    for needed_doc, cand, praw, psts, n in preds:
        caches = [OrderedDict() for _ in range(L)]; last_pred = [set() for _ in range(L)]; conf = [0.5]*L
        for t in range(n):
            nd = needed_doc[t]
            for l in range(L):
                needed = nd[l]; cache = caches[l]
                if crude and last_pred[l]:
                    lp = last_pred[l]; hf = sum(1 for e in needed if e in lp); conf[l] = 0.9*conf[l] + alpha*(hf/k)
                for e in needed:
                    tot += 1
                    if e in cache:
                        if policy != 'random': cache.move_to_end(e)
                    else:
                        miss += 1; cache[e] = None
                if len(cache) > cap: _evict(cache, cap, policy, rng)
                if l < L-1 and policy in predp:
                    cl = cand[l][t]
                    if policy == 'markov_naive': adm = cl[:cap]
                    elif crude: adm = cl[:int(round(cap*conf[l+1]))]
                    else:
                        pp = psts[l][t] if dspark else praw[l][t]
                        adm = [cl[j] for j in range(len(cl)) if pp[j] > tau][:cap]
                    nxt = caches[l+1]
                    for e in adm:
                        if e in nxt: nxt.move_to_end(e)
                        else: nxt[e] = None
                    last_pred[l+1] = set(adm)
                    if len(nxt) > cap: _evict(nxt, cap, policy, rng)
    return miss/max(tot, 1)

def _evict(cache, cap, policy, rng):
    while len(cache) > cap:
        if policy == 'random': cache.pop(list(cache.keys())[rng.integers(len(cache))])
        else: cache.popitem(last=False)

# ----------------------------------------------------------------------------- runner
POLICIES = ['random', 'reactive', 'markov_naive', 'adaptive_crude', 'adaptive_raw', 'adaptive_dspark']
HDR = {'random': 'RANDOM', 'reactive': 'REACTIVE', 'markov_naive': 'PRED-naive',
       'adaptive_crude': 'ADPT-crude', 'adaptive_raw': 'ADPT-raw', 'adaptive_dspark': 'ADPT-DSpark'}

def run(path, caps, seeds, topN, tau, max_test_tok, pred, M=15):
    E, S, bounds, nexp, topk, L = load(path); ndoc = len(bounds)-1
    if pred == 'hidden' and S is None:
        print(f"!! {path}: nessun campo 'scores' -> --predictor hidden non disponibile. Salto."); return None
    name = path.replace('\\', '/').split('/')[-1]
    print(f"\n=== {name} [{pred}]: docs={ndoc} T={E.shape[0]} L={L} E={nexp} k={topk} | caps={caps} seeds={seeds} topN={topN} tau={tau} ===", flush=True)
    rows = {p: {c: [] for c in caps} for p in POLICIES}; eb, ea = [], []
    for seed in seeds:
        tr, ca, te = split_docs(ndoc, seed)
        trd = doc_tokens(E, S, bounds, tr); cad = doc_tokens(E, S, bounds, ca); ted = doc_tokens(E, S, bounds, te, max_test_tok)
        C = build_markov(trd, L, nexp, topk) if pred == 'markov' else None
        a, b = fit_confidence(trd, C, L, nexp, topk, topN, pred)
        Zc, Yc = collect_cand(cad, C, L, nexp, topk, topN, a, b, pred); T = fit_sts(Zc, Yc, L, M=M)
        Zt, Yt = collect_cand(ted, C, L, nexp, topk, topN, a, b, pred)
        zt = np.concatenate([Zt[l] for l in range(L-1) if len(Zt[l])]); yt = np.concatenate([Yt[l] for l in range(L-1) if len(Yt[l])])
        Trep = np.concatenate([np.full(len(Zt[l]), T[l+1]) for l in range(L-1) if len(Zt[l])])
        eb.append(ece(sigmoid(zt), yt, M)); ea.append(ece(sigmoid(zt/Trep), yt, M))
        preds = precompute(ted, C, L, nexp, topk, topN, a, b, T, pred)
        ntok = sum(D.shape[0] for D, _ in ted)
        for c in caps:
            for p in POLICIES: rows[p][c].append(simulate(preds, c, p, L, topk, tau=tau, seed=seed))
        print(f"  seed {seed}: test_tok={ntok} ECE {eb[-1]:.3f}->{ea[-1]:.3f}(STS) T[mid]={T[L//2]:.1f}", flush=True)
    def ms(p, c): v = np.array(rows[p][c]); return v.mean(), v.std()
    print(f"\n  {'cap':>4} {'%E':>4} | " + " ".join(f"{HDR[p]:>13}" for p in POLICIES), flush=True)
    for c in caps:
        print(f"  {c:>4} {100*c/nexp:>3.0f}% | " + " ".join(f"{ms(p,c)[0]:.3f}±{ms(p,c)[1]:.3f}".rjust(13) for p in POLICIES), flush=True)
    print(f"\n  ECE (per-cand) TEST: raw {np.mean(eb):.3f} -> STS {np.mean(ea):.3f}", flush=True)
    return {'name': name, 'predictor': pred, 'nexp': nexp, 'L': L, 'caps': caps, 'seeds': list(seeds),
            'miss': {p: {c: [float(x) for x in rows[p][c]] for c in caps} for p in POLICIES},
            'ece_raw': float(np.mean(eb)), 'ece_sts': float(np.mean(ea))}

def export_c(path, out_path, seed, topN, pred, M=15):
    if pred != 'markov':
        raise ValueError("--export-c currently supports only --predictor markov")
    E, S, bounds, nexp, topk, L = load(path); ndoc = len(bounds)-1
    tr, ca, _ = split_docs(ndoc, seed)
    trd = doc_tokens(E, S, bounds, tr)
    cad = doc_tokens(E, S, bounds, ca)
    C = build_markov(trd, L, nexp, topk)
    a, b = fit_confidence(trd, C, L, nexp, topk, topN, pred)
    Zc, Yc = collect_cand(cad, C, L, nexp, topk, topN, a, b, pred)
    T = fit_sts(Zc, Yc, L, M=M)
    write_markov_spex(out_path, a=a, b=b, T=T, C=C, n_layer=L, n_expert=nexp, topN=topN)
    print(f"exported {out_path} [markov]: docs={ndoc} train={len(tr)} cal={len(ca)} L={L} E={nexp} topN={topN} seed={seed}", flush=True)
    return {'path': out_path, 'predictor': pred, 'nexp': nexp, 'L': L, 'topN': topN, 'seed': seed}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--caps', default='2,4,6,8,16'); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--topN', type=int, default=32); ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--max_test_tok', type=int, default=6000)
    ap.add_argument('--predictor', default='markov', choices=['markov', 'hidden'])
    ap.add_argument('--export-c', default=None, help="Export markov SPEX params to a ds4 .spex file")
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    caps = [int(x) for x in a.caps.split(',')]; seeds = [int(x) for x in a.seeds.split(',')]
    if a.export_c:
        if len(a.paths) != 1:
            raise SystemExit("--export-c expects exactly one input .npz")
        export_c(a.paths[0], a.export_c, seeds[0], a.topN, a.predictor)
        raise SystemExit(0)
    res = [run(p, caps, seeds, a.topN, a.tau, a.max_test_tok, a.predictor) for p in a.paths]
    if a.out: json.dump([r for r in res if r], open(a.out, 'w'), indent=1); print("saved", a.out)
