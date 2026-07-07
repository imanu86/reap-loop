"""Track REAP-ds4 — validazione OFFLINE (gratis, CPU-only) dell'approssimazione "g-only".

DOMANDA: la trace ds4 (patch 0006) dara' expert-ID + gate-weight g, ma NON ||f_j||
(norma dell'output dell'expert). La saliency calcolabile da ds4 e' quindi:
    S'_j = media condizionale di g_j        (g-only)
mentre la REAP vera (Eq.9, arxiv 2510.13999, reap_saliency.py) e':
    S_j  = media condizionale di g_j*||f_j||
Quanto costa l'approssimazione A LIVELLO DI RANKING (bottom-K = expert da potare)?

METODO (tutto su file gia' su disco, 30B dominio, stessi 152 prompt raft):
  - verita': models/reap_saliency_base.json -> 'reap'[l][e] = Eq.9 vera (48x128)
  - proxy 1 "freq": conteggi dalle selezioni REALI del router
        models/spex/hidden_scores_q30_dom.npz['experts'] (T,48,8) int16
  - proxy 2 "g-only": gate-weight ricostruito dai probe-scores dello stesso npz
        ['scores'] (T,47,128) fp16, layer 1..47 (probe lineare Fate-style;
        fedelta' top-k vs router vero: recall@32 = 0.9978 dom, ledger E8).
        softmax(128) -> top-8 -> renorm top-8 (come Qwen3 norm_topk_prob) -> g
        S'[l][e] = somma g / conteggi (media condizionale, zero se mai attivo)
  - metriche per layer: overlap bottom-K (K=32/64/90 su 128 ~ 25/50/70%) tra
    ranking proxy e ranking Eq.9 + Spearman su tutti i 128 valori + RETENTION:
    quota di saliency-vera trattenuta potando bottom-K del proxy, normalizzata
    a quella del prune ottimo (bottom-K vero). retention_ratio=1.0 -> il proxy
    perde zero saliency extra anche se gli ID scelti differiscono (i disaccordi
    stanno nella zona piatta di confine).

CAVEAT dichiarati: (a) 'scores' = probe hidden, non logits router (fedele al 99.8%
sui top-k ma approssimazione); (b) layer 0 non ha scores (probe parte da L1);
(c) T npz=60080 vs T reap~130k (subset di token degli stessi prompt).

Output: runs/reap/gonly_vs_eq9_30b.json (+ stampa tabella riassuntiva).
Uso: python scripts/reap_gonly_vs_eq9_30b.py
"""
import json
import os
import numpy as np

REAP_JSON = "models/reap_saliency_base.json"
NPZ = "models/spex/hidden_scores_q30_dom.npz"
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "reap", "gonly_vs_eq9_30b.json")
KS = (32, 64, 90)  # ~25% / 50% / 70% di 128
CHUNK = 4096


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def bottomk_overlap(proxy_vals, true_vals, k):
    """Overlap tra i K expert piu' freddi secondo proxy e secondo verita'."""
    bp = set(np.argsort(proxy_vals, kind="stable")[:k].tolist())
    bt = set(np.argsort(true_vals, kind="stable")[:k].tolist())
    return len(bp & bt) / k


def retention_ratio(proxy_vals, true_vals, k):
    """Saliency-vera trattenuta dai superstiti del prune-proxy, relativa al
    prune ottimo. 1.0 = nessuna perdita extra rispetto a potare col ranking vero."""
    tot = float(true_vals.sum())
    if tot <= 0:
        return 1.0
    keep_proxy = np.argsort(proxy_vals, kind="stable")[k:]
    keep_true = np.argsort(true_vals, kind="stable")[k:]
    ret_proxy = float(true_vals[keep_proxy].sum()) / tot
    ret_true = float(true_vals[keep_true].sum()) / tot
    return ret_proxy / ret_true if ret_true > 0 else 1.0


def main():
    d = json.load(open(REAP_JSON))
    E, NL = int(d["E"]), int(d["n_layers"])
    true_sal = np.zeros((NL, E))
    for l in range(NL):
        for e_str, v in d["reap"][str(l)].items():
            true_sal[l, int(e_str)] = v

    z = np.load(NPZ)
    experts = z["experts"]            # (T, 48, 8) selezioni reali router
    T = experts.shape[0]
    topk = int(z["topk"])

    # --- proxy 1: frequenza dalle selezioni reali ---
    freq = np.zeros((NL, E))
    for l in range(NL):
        freq[l] = np.bincount(experts[:, l, :].reshape(-1).astype(np.int64),
                              minlength=E)

    # --- proxy 2: g-only conditional mean dai probe scores (layer 1..47) ---
    scores = z["scores"]              # (T, 47, 128) fp16, layer j = probe del layer j+1
    gsum = np.zeros((NL, E))
    gcnt = np.zeros((NL, E))
    for s0 in range(0, T, CHUNK):
        sc = scores[s0:s0 + CHUNK].astype(np.float32)          # (c, 47, E)
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc); p /= p.sum(-1, keepdims=True)          # softmax full-E
        idx = np.argpartition(p, -topk, axis=-1)[..., -topk:]  # (c, 47, topk)
        gw = np.take_along_axis(p, idx, axis=-1)
        gw /= gw.sum(-1, keepdims=True)                        # renorm top-k (norm_topk_prob)
        c, L47 = gw.shape[0], gw.shape[1]
        for lj in range(L47):
            l = lj + 1                                         # scores[:, lj] predice layer lj+1
            flat_idx = idx[:, lj, :].reshape(-1).astype(np.int64)
            flat_g = gw[:, lj, :].reshape(-1).astype(np.float64)
            gsum[l] += np.bincount(flat_idx, weights=flat_g, minlength=E)
            gcnt[l] += np.bincount(flat_idx, minlength=E)
    gonly = np.divide(gsum, gcnt, out=np.zeros_like(gsum), where=gcnt > 0)

    layers_gonly = list(range(1, NL))   # layer 0 senza probe
    res = {"source_truth": REAP_JSON, "source_npz": NPZ, "E": E, "n_layers": NL,
           "T_npz": int(T), "topk": topk, "ks": list(KS),
           "note": "gonly usa probe-scores (recall@32 .9978 dom, ledger E8), layer 1..47",
           "per_layer": {}, "summary": {}}
    agg = {p: {f"overlap@{k}": [] for k in KS}
              | {f"retention@{k}": [] for k in KS}
              | {"spearman": []}
           for p in ("freq", "gonly")}
    for l in range(NL):
        row = {}
        for pname, vals, ok in (("freq", freq[l], True),
                                ("gonly", gonly[l], l in layers_gonly)):
            if not ok:
                continue
            ent = {f"overlap@{k}": round(bottomk_overlap(vals, true_sal[l], k), 4)
                   for k in KS}
            for k in KS:
                ent[f"retention@{k}"] = round(retention_ratio(vals, true_sal[l], k), 4)
            ent["spearman"] = round(spearman(vals, true_sal[l]), 4)
            row[pname] = ent
            for k in KS:
                agg[pname][f"overlap@{k}"].append(ent[f"overlap@{k}"])
                agg[pname][f"retention@{k}"].append(ent[f"retention@{k}"])
            agg[pname]["spearman"].append(ent["spearman"])
        res["per_layer"][str(l)] = row
    for pname, m in agg.items():
        res["summary"][pname] = {kk: {"mean": round(float(np.mean(v)), 4),
                                      "std": round(float(np.std(v)), 4),
                                      "min": round(float(np.min(v)), 4)}
                                 for kk, v in m.items()}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"scritto {os.path.normpath(OUT)}")
    print(f"T={T} topk={topk} E={E} NL={NL}")
    for pname in ("freq", "gonly"):
        s = res["summary"][pname]
        line = " ".join(f"{kk}={v['mean']:.3f}±{v['std']:.3f}(min {v['min']:.3f})"
                        for kk, v in s.items())
        print(f"{pname:6s} {line}")


if __name__ == "__main__":
    main()
