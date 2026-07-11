#!/usr/bin/env python3
"""Substitution-similarity feasibility (survey MOE_ECOSYSTEM_SURVEY_20260711 #2 /
candidate-adoption #2, retrial justified by runs/ds4/20260711_substitution_archaeology/).

OFFLINE ONLY. Reads existing weighted routing traces (no GPU / WSL / pod).

Question: on a mask-miss (router wants an expert pruned by the session keep-K
mask), does SUBSTITUTING the miss with the most-similar KEPT expert make sense?
Only if the "want" (out-of-mask, high demand) experts are *functionally close*
to a kept expert -- not orthogonal to the whole kept set.

Method
------
1. CO-ACTIVATION GRAPH (proxy for functional similarity). Pool every routed
   token from every full-model (unmasked) trace on disk; for each token's
   top-6 selection, every unordered pair of co-selected experts is one
   co-activation event. sim(layer,i,j) = co-occurrences(i,j) /
   sqrt(sel_count(i) * sel_count(j))  -- an overlap-coefficient / cosine-like
   score in [0,1], computed PER LAYER (experts are layer-local).

2. WANT SET. Per source trace, split generated tokens into a WARMUP window
   (first <=50 tokens, mirrors the real DS4_PACE warmup that freezes the
   session mask) and a REST window (everything after). Build keep-23 from
   WARMUP mass only (same rank_keep() the production mask builder uses) --
   this is the mask a live session would actually freeze. Build a REST-only
   top-23-by-mass ranking too. want[layer] = (REST top-23) - (WARMUP keep-23):
   experts the router demonstrably wanted in the continuation that the frozen
   mask does not have. This *is* the drifted-demand / miss set, not a
   theoretical one.

3. SUBSTITUTABILITY. For every want expert w, best_kept_sim = max similarity
   (from the pooled co-activation graph) to any expert in that source's
   WARMUP keep-23. Compare the distribution of best_kept_sim against two
   null/reference distributions per layer: KEPT-KEPT (pairwise similarity
   among experts already inside one canonical pooled keep-23 -- what
   "genuinely related" looks like) and RANDOM (similarity of random expert
   pairs -- what "no relation" looks like).

Data sources (full-model, weighted, e0..e5/w0..w5) -- see README pointers:
  - runs/ds4/20260710_pod_cache1024_warmup_replay/{W50,W130}/route_*.csv (html)
  - <moe-root>/runs/reap/k91_coding_vram/trace_coding.tgz (11 coding prompts)
  - runs/ds4/20260711_podA_narrow_traces/{a_coffee_full,b_json_full,
    c_python_full,b2_json_long_full,c2_python_long_full}/route.csv (narrow)

Checked and EXCLUDED: runs/ds4/20260710_scope_divergence_pod/r1/s1_r1.csv.gz --
schema is (pos,layer,pruned_mass,total_mass), no per-expert identity, so it
cannot feed a co-activation graph (verified by inspecting the header/manifest).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_session_mask_canonical import rank_keep  # noqa: E402

K_KEEP = 23
N_EXPERT = 256
WARMUP_TOKENS = 50
MIN_REST_TOKENS = 20
RNG_SEED = 20260711


def load_rows(path):
    """Return list of (pos:int, layer:int, pairs:[(expert,weight),...])."""
    rows = []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        eidx = [header.index(f"e{i}") for i in range(6)]
        widx = [header.index(f"w{i}") for i in range(6)]
        lidx = header.index("layer")
        pidx = header.index("pos")
        for r in rd:
            if not r:
                continue
            pos = int(r[pidx])
            layer = int(r[lidx])
            pairs = []
            for ei, wi in zip(eidx, widx):
                e = int(r[ei])
                if e < 0:
                    continue
                pairs.append((e, float(r[wi])))
            if pairs:
                rows.append((pos, layer, pairs))
    return rows


def gather_sources(reap_loop_root, moe_root):
    RL = reap_loop_root
    sources = []
    sources.append(("html_W50", os.path.join(
        RL, "runs/ds4/20260710_pod_cache1024_warmup_replay/W50/route_W50.csv")))
    sources.append(("html_W130", os.path.join(
        RL, "runs/ds4/20260710_pod_cache1024_warmup_replay/W130/route_W130.csv")))

    narrow_dir = os.path.join(RL, "runs/ds4/20260711_podA_narrow_traces")
    for cell in ("a_coffee_full", "b_json_full", "c_python_full",
                 "b2_json_long_full", "c2_python_long_full"):
        p = os.path.join(narrow_dir, cell, "route.csv")
        if os.path.exists(p):
            sources.append((f"narrow_{cell}", p))

    tgz = os.path.join(moe_root, "runs/reap/k91_coding_vram/trace_coding.tgz")
    if os.path.exists(tgz):
        tmp = tempfile.mkdtemp(prefix="subsim_coding_")
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp)
        coding_dir = os.path.join(tmp, "trace_coding")
        for name in sorted(os.listdir(coding_dir)):
            if name.startswith("trace_") and name.endswith(".csv"):
                tag = name.replace("trace_", "").replace(".csv", "")
                sources.append((f"code_{tag}", os.path.join(coding_dir, name)))
    return sources


def mass_and_seen(rows):
    mass = defaultdict(lambda: defaultdict(float))
    seen = defaultdict(set)
    for _pos, layer, pairs in rows:
        for e, w in pairs:
            mass[layer][e] += w
            seen[layer].add(e)
    return mass, seen


def topk_by_mass(seen, mass, k):
    keep = {}
    for layer in seen:
        experts = seen[layer]
        ranked = sorted(experts, key=lambda e: (-mass[layer][e], e))
        keep[layer] = set(ranked[:k])
    return keep


def build_co_activation(all_rows):
    """co[layer][(i,j)] (i<j) -> co-occurrence count; sel[layer][e] -> selection count."""
    co = defaultdict(lambda: defaultdict(float))
    sel = defaultdict(lambda: defaultdict(float))
    for _pos, layer, pairs in all_rows:
        experts = [e for e, _ in pairs]
        for e in experts:
            sel[layer][e] += 1
        n = len(experts)
        for a in range(n):
            for b in range(a + 1, n):
                i, j = experts[a], experts[b]
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                co[layer][(i, j)] += 1
    return co, sel


def sim(layer, i, j, co, sel):
    if i == j:
        return 1.0
    key = (i, j) if i < j else (j, i)
    c = co[layer].get(key, 0.0)
    if c == 0.0:
        return 0.0
    denom = math.sqrt(sel[layer].get(i, 0.0) * sel[layer].get(j, 0.0))
    return c / denom if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_reap = os.path.dirname(here)
    ap.add_argument("--reap-loop-root", default=default_reap)
    ap.add_argument("--moe-root",
                     default=r"C:/Users/imanu/source/repos/moe-aggressive-commit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RL = args.reap_loop_root
    out_dir = args.out or os.path.join(RL, "runs/ds4/20260711_substitution_similarity")
    os.makedirs(out_dir, exist_ok=True)

    sources = gather_sources(RL, args.moe_root)
    print(f"sources found: {len(sources)}")

    loaded = {}
    for label, path in sources:
        loaded[label] = load_rows(path)
        print(f"  {label:28s} tokens={len(set(p for p, _l, _pr in loaded[label])):5d} rows={len(loaded[label])}")

    all_rows = [row for rows in loaded.values() for row in rows]
    co, sel = build_co_activation(all_rows)

    # ---- canonical pooled keep-23 (kept-kept baseline) ---------------------
    pooled_mass, pooled_seen = mass_and_seen(all_rows)
    keep23_pooled = topk_by_mass(pooled_seen, pooled_mass, K_KEEP)

    # ---- per-source warmup/rest split + want extraction -------------------
    want_records = []  # (source, layer, want_expert, best_kept_sim, best_kept_id, rest_mass_share)
    per_source_meta = {}
    for label, rows in loaded.items():
        positions = sorted(set(p for p, _l, _pr in rows))
        n_tok = len(positions)
        warmup_n = min(WARMUP_TOKENS, max(1, n_tok // 2))
        warmup_cut = positions[warmup_n - 1] if warmup_n <= n_tok else positions[-1]
        warmup_rows = [r for r in rows if r[0] <= warmup_cut]
        rest_rows = [r for r in rows if r[0] > warmup_cut]
        rest_n_tok = n_tok - warmup_n
        per_source_meta[label] = dict(n_tok=n_tok, warmup_n=warmup_n, rest_n_tok=rest_n_tok,
                                       usable=rest_n_tok >= MIN_REST_TOKENS)
        if rest_n_tok < MIN_REST_TOKENS:
            continue  # too short to trust a rest-mass ranking

        w_mass, w_seen = mass_and_seen(warmup_rows)
        keep23_warmup = topk_by_mass(w_seen, w_mass, K_KEEP)

        r_mass, r_seen = mass_and_seen(rest_rows)
        rest_top23 = topk_by_mass(r_seen, r_mass, K_KEEP)

        for layer in rest_top23:
            kept = keep23_warmup.get(layer, set())
            if not kept:
                continue
            total_rest_mass = sum(r_mass[layer].values())
            if total_rest_mass <= 0:
                continue
            want = rest_top23[layer] - kept
            for w in want:
                best_sim = -1.0
                best_k = None
                for k in kept:
                    s = sim(layer, w, k, co, sel)
                    if s > best_sim:
                        best_sim, best_k = s, k
                want_records.append(dict(
                    source=label, layer=layer, want=w, best_kept=best_k,
                    best_kept_sim=best_sim,
                    rest_mass_share=r_mass[layer][w] / total_rest_mass,
                ))

    # ---- baselines ----------------------------------------------------------
    # want_best_kept_sim is a MAX over 23 candidates (order statistic) -- a
    # naive single-random-pair null understates the null and is not a fair
    # comparison. The fair null is: if a "want" were actually an UNRELATED
    # random expert (not selected for any similarity reason), what would its
    # best-of-23-kept similarity look like by chance? -> random_max_sims.
    # kept_kept_sims (single pairwise, no max) is kept only as a secondary,
    # order-mismatched reference (labelled as such, not used for the verdict).
    rng = np.random.default_rng(RNG_SEED)
    kept_kept_sims = []
    for layer, kept in keep23_pooled.items():
        kk = sorted(kept)
        for a in range(len(kk)):
            for b in range(a + 1, len(kk)):
                kept_kept_sims.append(sim(layer, kk[a], kk[b], co, sel))

    random_sims = []
    random_max_sims = []
    layers = sorted(pooled_seen.keys())
    for layer in layers:
        pool = sorted(pooled_seen[layer])
        kept = sorted(keep23_pooled.get(layer, set()))
        if len(pool) < 2 or not kept:
            continue
        non_kept_pool = [e for e in pool if e not in keep23_pooled.get(layer, set())]
        if not non_kept_pool:
            non_kept_pool = pool
        n_sample = min(200, len(pool) * (len(pool) - 1) // 2)
        for _ in range(n_sample):
            i, j = rng.choice(pool, size=2, replace=False)
            random_sims.append(sim(layer, int(i), int(j), co, sel))
        n_max_sample = min(300, len(non_kept_pool))
        draw = rng.choice(non_kept_pool, size=n_max_sample, replace=False)
        for e in draw:
            best = max((sim(layer, int(e), k, co, sel) for k in kept), default=0.0)
            random_max_sims.append(best)

    want_sims = [r["best_kept_sim"] for r in want_records]

    def agg(vals):
        if not vals:
            return dict(n=0)
        a = np.array(vals, dtype=float)
        return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                    p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                    p90=float(np.percentile(a, 90)), frac_zero=float((a == 0).mean()))

    want_agg = agg(want_sims)
    kk_agg = agg(kept_kept_sims)
    rnd_agg = agg(random_sims)
    rnd_max_agg = agg(random_max_sims)

    def auc_effect_size(a, b):
        """P(x_a > x_b) for x_a~a, x_b~b (common-language effect size / Mann-Whitney
        AUC), computed via rank-sum (O(n log n), no scipy dependency). 0.5 = no
        separation, 1.0 = a always greater, 0.0 = a always smaller. Ties count 0.5."""
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        na, nb = len(a), len(b)
        combined = np.concatenate([a, b])
        order = np.argsort(combined, kind="mergesort")
        ranks = np.empty(len(combined))
        ranks[order] = np.arange(1, len(combined) + 1)
        # average ranks for ties
        sorted_vals = combined[order]
        i = 0
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                avg_rank = ranks[order[i:j + 1]].mean()
                ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        rank_sum_a = ranks[:na].sum()
        u_a = rank_sum_a - na * (na + 1) / 2
        return float(u_a / (na * nb))

    # ---- verdict -------------------------------------------------------------
    # Substitution is justified if want->best-kept similarity is separated
    # from the order-matched random-max null (best-of-23 similarity of an
    # UNRELATED expert to the kept set -- same max-over-23 selection procedure
    # as the real wants, so the AUC below isolates the "is-w-actually-similar"
    # signal from the pure order-statistic inflation). AUC = P(want_sim >
    # random_max_sim), common-language effect size, no scipy needed.
    if want_agg["n"] == 0:
        verdict = "NO DATA: nessun evento want fuori-mask osservato (mask e rest coincidono ovunque)."
        substitution_justified = None
        auc = float("nan")
    else:
        auc = auc_effect_size(np.array(want_sims), np.array(random_max_sims))
        vs_kk = (want_agg["median"] / kk_agg["median"]) if kk_agg.get("median", 0) > 0 else float("inf")
        pct_above_rndmax_median = float(np.mean(np.array(want_sims) > rnd_max_agg["median"]))
        if auc >= 0.60:
            substitution_justified = True
            verdict = (
                f"SUBSTITUTION-RUNTIME GIUSTIFICATO (segnale debole-moderato): AUC = {auc:.3f} "
                f"(P(want_sim > random_max_sim), null order-matched best-of-23) -- separazione "
                f"reale ma non forte dal caso scorrelato (AUC 0.5 = nessuna separazione). "
                f"Mediana want->best-kept {want_agg['median']:.3f} vs random-max {rnd_max_agg['median']:.3f} "
                f"({pct_above_rndmax_median*100:.0f}% dei want superano la mediana random-max). "
                f"{vs_kk:.1f}x la mediana kept-kept pairwise (riferimento non order-matched). "
                f"Solo {want_agg['frac_zero']*100:.1f}% dei want hanno similarita' ESATTAMENTE zero."
            )
        else:
            substitution_justified = False
            verdict = (
                f"SUBSTITUTION-RUNTIME NON GIUSTIFICATO: AUC = {auc:.3f} "
                f"(P(want_sim > random_max_sim), null order-matched best-of-23) -- i want sono "
                f"quasi indistinguibili dal caso scorrelato (AUC 0.5 = nessuna separazione). "
                f"Mediana want->best-kept {want_agg['median']:.3f} vs random-max {rnd_max_agg['median']:.3f} "
                f"({pct_above_rndmax_median*100:.0f}% dei want superano la mediana random-max, "
                f"atteso ~50% sotto H0). Il segnale grezzo (want > kept-kept pairwise, {vs_kk:.1f}x) "
                f"e' un artefatto del max-over-23, non prossimita' funzionale reale. "
                f"Sostituire approssimerebbe con rumore, non con un vicino funzionale."
            )

    stats = dict(
        auc_vs_random_max=auc,
        sources=per_source_meta,
        n_sources=len(sources),
        want=want_agg, kept_kept=kk_agg, random=rnd_agg, random_max=rnd_max_agg,
        substitution_justified=substitution_justified,
        verdict=verdict,
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    with open(os.path.join(out_dir, "want_records.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "layer", "want", "best_kept", "best_kept_sim", "rest_mass_share"])
        for r in sorted(want_records, key=lambda r: (-r["best_kept_sim"])):
            w.writerow([r["source"], r["layer"], r["want"], r["best_kept"],
                        f"{r['best_kept_sim']:.4f}", f"{r['rest_mass_share']:.4f}"])

    print("\n=== substitution-similarity ===")
    print(f"want (out-of-mask high-demand) events: {want_agg}")
    print(f"kept-kept baseline (single-pair, order-mismatched reference): {kk_agg}")
    print(f"random baseline (single-pair): {rnd_agg}")
    print(f"random-max baseline (order-matched, best-of-23 -- the fair null): {rnd_max_agg}")
    print("\n=== VERDICT ===")
    print(verdict)
    print(f"\nwrote {out_dir}/stats.json, want_records.csv")
    return stats


if __name__ == "__main__":
    main()
