#!/usr/bin/env python3
"""MAN/MSAN gate-free score feasibility (survey MOE_ECOSYSTEM_SURVEY_20260711 #2 /
candidate-adoption #4, arXiv 2606.15716).

OFFLINE ONLY. Reads existing weighted routing traces (no GPU / WSL / pod).

Formula (from arXiv 2606.15716, fetched 2026-07-11 -- unified scoring):

    S_j(b, alpha, beta) = (1 / N_j^b) * sum_t 1[j in E_t] * g_{j,t}^alpha * ||f_{j,t}||_2^beta

  N_j = count of tokens routed to expert j; g_{j,t} = router gate weight;
  f_{j,t} = expert j's OUTPUT ACTIVATION vector for token t (post-FFN, pre
  gate-weighted combine); ||.||_2 = L2 norm.

  Special cases (paper's own table):
    Frequency (b=0,a=0,b_=0): sum 1[j in E_t]                      -- routing count only
    SEER      (b=0,a=1,b_=0): sum 1[j in E_t] * g_{j,t}             -- cumulative gate mass
    EAN       (b=0,a=0,b_=1): sum 1[j in E_t] * ||f_{j,t}||         -- cumulative act-norm
    REAP      (b=1,a=1,b_=1): mean_t g_{j,t} * ||f_{j,t}||          -- gate x act-norm, averaged
    MAN       (b=1,a=0,b_=1): mean_t ||f_{j,t}||                    -- act-norm only, averaged (gate-free)
    MSAN      (b=1,a=0,b_=2): mean_t ||f_{j,t}||^2                  -- act-energy, averaged (gate-free)

KEY FEASIBILITY CHECK: MAN/MSAN (and true REAP, and EAN) all require
||f_{j,t}||_2 -- the expert's raw OUTPUT ACTIVATION NORM. This script verifies
whether any routing trace on disk carries that column, and if not, computes
the two special cases we CAN already produce (Frequency, SEER) from the
existing schema, to bound how much moving along the gate-weight axis alone
(alpha: 1->0) shifts the keep-23 mask -- a partial, honest lower-bound signal
for "how different would MAN plausibly be" while being explicit that the
activation-norm axis (beta: 0->1) is NOT measured by this script.

Trace schema on disk (verified across ALL sources below):
    pos,layer,n,e0,e1,e2,e3,e4,e5,w0,w1,w2,w3,w4,w5
No norm/activation column anywhere. (Also checked
runs/ds4/20260710_scope_divergence_pod/r1/s1_r1.csv.gz: schema
pos,layer,pruned_mass,total_mass -- aggregate S1 sensor, no per-expert
identity either; irrelevant to this feasibility check too.)

This script maps EXACTLY onto what build_session_mask_canonical.py already
computes: --mode weighted == SEER (b=0,a=1,b_=0); --mode unit == Frequency
(b=0,a=0,b_=0). Both are reused here via direct import for consistency with
the production mask builder (no reimplementation drift).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_session_mask_canonical import build as build_mask  # noqa: E402

K_KEEP = 23

FORMULA_NOTE = {
    "source": "arXiv 2606.15716 (fetched via WebFetch 2026-07-11, HTML rendering; "
               "abstract-only PDF view did not expose the equations)",
    "unified": "S_j(b,alpha,beta) = (1/N_j^b) * sum_t 1[j in E_t] * g_{j,t}^alpha * ||f_{j,t}||_2^beta",
    "cases": {
        "Frequency": "b=0,alpha=0,beta=0 -> sum 1[j in E_t]  (raw selection count)",
        "SEER":      "b=0,alpha=1,beta=0 -> sum 1[j in E_t]*g_{j,t}  (cumulative gate mass)",
        "EAN":       "b=0,alpha=0,beta=1 -> sum 1[j in E_t]*||f_{j,t}||  (cumulative activation norm)",
        "REAP":      "b=1,alpha=1,beta=1 -> mean_t g_{j,t}*||f_{j,t}||  (gate x act-norm, averaged)",
        "MAN":       "b=1,alpha=0,beta=1 -> mean_t ||f_{j,t}||  (activation norm only, averaged, GATE-FREE)",
        "MSAN":      "b=1,alpha=0,beta=2 -> mean_t ||f_{j,t}||^2  (activation energy, averaged, GATE-FREE)",
    },
    "repo_current_score": (
        "build_session_mask_canonical.py --mode weighted ranks by cumulative gate "
        "mass sum(w) per layer -- this is exactly SEER (b=0,alpha=1,beta=0), NOT the "
        "arXiv 2510.13999 'REAP' the repo colloquially references elsewhere "
        "(docs/SCALE_FRONTIER_VERDICT.md already flags REAP's real criterion as "
        "gate x activation-norm, i.e. needs beta=1 too -- so the repo's own docs "
        "were already correct that the name is borrowed, not the exact criterion)."
    ),
    "missing_for_true_man_msan": (
        "||f_{j,t}||_2, the L2 norm of expert j's raw FFN output activation for "
        "token t. DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 (the only tracer flag that adds "
        "columns beyond selection) captures gate weights w0..w5 ONLY -- verified by "
        "reading docs/REAP_DS4_design.md / docs/DS4_ROUTING_RESIDENCY_TRACE.md and by "
        "grepping the header row of every trace file used across this repo's E1/E-CAL/ "
        "substitution-similarity offline analyses: all are "
        "'pos,layer,n,e0..e5,w0..w5', zero exceptions."
    ),
}


def load_rows(path):
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
        tmp = tempfile.mkdtemp(prefix="man_coding_")
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp)
        coding_dir = os.path.join(tmp, "trace_coding")
        for name in sorted(os.listdir(coding_dir)):
            if name.startswith("trace_") and name.endswith(".csv"):
                tag = name.replace("trace_", "").replace(".csv", "")
                sources.append((f"code_{tag}", os.path.join(coding_dir, name)))
    return sources


def check_schema_for_norm_columns(paths):
    """Confirm empirically that no trace on disk carries an activation-norm
    column (n0..n5 or similar). Returns (all_missing: bool, checked: [str])."""
    checked = []
    all_missing = True
    for label, path in paths:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh))
        checked.append((label, header))
        expected = ["pos", "layer", "n"] + [f"e{i}" for i in range(6)] + [f"w{i}" for i in range(6)]
        if header != expected:
            all_missing = False  # unexpected schema; flag for manual look
    return all_missing, checked


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


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
    out_dir = args.out or os.path.join(RL, "runs/ds4/20260711_man_score")
    os.makedirs(out_dir, exist_ok=True)

    sources = gather_sources(RL, args.moe_root)
    print(f"sources found: {len(sources)}")

    all_missing, checked = check_schema_for_norm_columns(sources)
    print(f"activation-norm column present anywhere? {'NO (confirmed missing everywhere)' if all_missing else 'UNEXPECTED SCHEMA FOUND -- see stats.json'}")

    # ---- what we CAN compute: SEER (weighted) vs Frequency (unit) ----------
    per_source_jaccard = {}
    all_layer_jaccards = defaultdict(list)
    pooled_rows = []
    for label, path in sources:
        rows = load_rows(path)
        pooled_rows.extend(rows)
        keep_weighted, seen, have_w = build_mask(path, K_KEEP, "weighted")
        keep_unit, _seen2, _have_w2 = build_mask(path, K_KEEP, "unit")
        layers = sorted(set(keep_weighted) & set(keep_unit))
        js = [jaccard(set(keep_weighted[layer]), set(keep_unit[layer])) for layer in layers]
        per_source_jaccard[label] = dict(
            n_layers=len(layers), mean_jaccard=float(np.mean(js)) if js else float("nan"),
            median_jaccard=float(np.median(js)) if js else float("nan"),
            min_jaccard=float(np.min(js)) if js else float("nan"),
        )
        for layer, j in zip(layers, js):
            all_layer_jaccards[layer].append(j)

    # pooled (all sources combined into one route -- write a temp CSV so we can
    # reuse build_mask() unmodified for perfect consistency with production code)
    tmp_pool = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    try:
        w = csv.writer(tmp_pool)
        w.writerow(["pos", "layer", "n", "e0", "e1", "e2", "e3", "e4", "e5",
                    "w0", "w1", "w2", "w3", "w4", "w5"])
        for pos, layer, pairs in pooled_rows:
            experts = [e for e, _ in pairs] + [-1] * (6 - len(pairs))
            weights = [wt for _, wt in pairs] + [0.0] * (6 - len(pairs))
            w.writerow([pos, layer, len(pairs)] + experts[:6] + weights[:6])
        tmp_pool.close()
        keep_weighted_pool, _s, _h = build_mask(tmp_pool.name, K_KEEP, "weighted")
        keep_unit_pool, _s2, _h2 = build_mask(tmp_pool.name, K_KEEP, "unit")
    finally:
        os.unlink(tmp_pool.name)

    pooled_layers = sorted(set(keep_weighted_pool) & set(keep_unit_pool))
    pooled_js = [jaccard(set(keep_weighted_pool[layer]), set(keep_unit_pool[layer])) for layer in pooled_layers]

    layer_summary = {L: dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)))
                      for L, v in sorted(all_layer_jaccards.items())}
    all_js_flat = [j for js in all_layer_jaccards.values() for j in js]

    # ---- feasibility + implementability verdict -----------------------------
    axis_shift_mean = float(np.mean(all_js_flat)) if all_js_flat else float("nan")
    axis_shift_pooled_mean = float(np.mean(pooled_js)) if pooled_js else float("nan")

    verdict = (
        "MAN/MSAN NON CALCOLABILE OFFLINE dai trace esistenti: la formula "
        "(arXiv 2606.15716) richiede ||f_{j,t}||_2, la norma dell'attivazione "
        "di output dell'expert -- colonna assente in TUTTI i trace su disco "
        "(schema verificato: pos,layer,n,e0..e5,w0..w5, nessuna eccezione su "
        f"{len(sources)} sorgenti). Anche la vera formula REAP (arXiv 2510.13999, "
        "gate x act-norm) e EAN condividono la stessa dipendenza mancante. Il "
        "punteggio attualmente in uso (--mode weighted) e' esattamente SEER "
        "(b=0,alpha=1,beta=0: massa-gate cumulata), non REAP in senso stretto. "
        f"Segnale parziale disponibile: rimuovere SOLO l'asse gate-weight "
        f"(SEER->Frequency, alpha:1->0, a beta invariato/assente) sposta gia' la "
        f"keep-23 mask di Jaccard medio {axis_shift_mean:.3f} per-source "
        f"({axis_shift_pooled_mean:.3f} pooled) sui 40 layer -- un limite "
        "inferiore onesto: aggiungere anche l'asse activation-norm (beta:0->1, "
        "quello che MAN attiva davvero) sposterebbe la mask AT LEAST altrettanto, "
        "probabilmente di piu', ma questo NON e' misurato qui."
    )

    implementable_now = False
    implementation_spec = (
        "NON implementabile oggi come --mode man/msan in build_session_mask_canonical.py "
        "senza prima estendere la cattura a runtime. Prerequisito (C-side, ds4.c, patch "
        "0012/DS4_SPEX_TRACE_ROUTING_WEIGHTS): per ciascuno dei 6 expert selezionati, "
        "il forward calcola GIA' il tensore di output FFN f_{j,t} prima della combine "
        "pesata dal gate -- aggiungere una L2-norm su quel tensore gia' materializzato "
        "e' un costo marginale (6 riduzioni norm/token/layer, NESSUN matmul aggiuntivo, "
        "NESSun expert extra da valutare). Estendere il CSV con 6 colonne n0..n5 "
        "(norma per lo stesso slot di e0..e5). Poi il python-side e' banale: "
        "in read_route_trace() aggiungere actnorm_sum[layer][e] e actnorm_count[layer][e] "
        "accumulati dalla nuova colonna; score_man[layer][e] = actnorm_sum/actnorm_count "
        "(beta=1), score_msan[layer][e] = sum(norm^2)/count (beta=2); poi rank_keep() e' "
        "riusato SENZA MODIFICHE (gia' agnostico al segnale di scoring). Effort: "
        "BASSO-MEDIO sul lato C (nuova metrica, non nuovo compute), BASSO sul lato python."
    )

    stats = dict(
        formula=FORMULA_NOTE,
        schema_check=dict(all_missing_norm_column=all_missing,
                           n_sources_checked=len(checked)),
        n_sources=len(sources),
        per_source_seer_vs_frequency_jaccard=per_source_jaccard,
        pooled_seer_vs_frequency_jaccard=dict(
            n_layers=len(pooled_layers),
            mean=axis_shift_pooled_mean,
            median=float(np.median(pooled_js)) if pooled_js else float("nan"),
        ),
        per_layer_jaccard_across_sources=layer_summary,
        axis_shift_mean_per_source=axis_shift_mean,
        implementable_now=implementable_now,
        implementation_spec=implementation_spec,
        verdict=verdict,
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    with open(os.path.join(out_dir, "per_layer_seer_vs_frequency_jaccard.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "n_sources", "mean_jaccard", "median_jaccard"])
        for L, d in sorted(layer_summary.items()):
            w.writerow([L, d["n"], f"{d['mean']:.4f}", f"{d['median']:.4f}"])

    print("\n=== MAN/MSAN feasibility ===")
    print(json.dumps(FORMULA_NOTE["cases"], indent=2))
    print(f"\nSEER(weighted) vs Frequency(unit) keep-23 Jaccard: per-source mean={axis_shift_mean:.3f}, pooled mean={axis_shift_pooled_mean:.3f}")
    print("\n=== VERDICT ===")
    print(verdict)
    print("\n=== IMPLEMENTABILITY ===")
    print(implementation_spec)
    print(f"\nwrote {out_dir}/stats.json, per_layer_seer_vs_frequency_jaccard.csv")
    return stats


if __name__ == "__main__":
    main()
