#!/usr/bin/env python3
"""E1 -- top-expert mass dominance / stability analysis (precision-pin lever).

OFFLINE ONLY. Reads routing-weight traces already on disk (no GPU / WSL / pod).

Question behind the probe (user lever "top-mass precision pin"):
  Pin the single most-important routed expert per layer into VRAM at a HIGHER
  quant (Q4 ~4.5 bpw) instead of the 2-bit base, as a quasi-shared expert, to
  buy stability/precision for the masked system. Is that worth a pod A/B?

We answer three questions from the weighted traces:

  Q1 DOMINANCE   how much of the per-token routed gate mass does the top-1
                 expert carry? (per-layer mean/median/p10/p90; top-2/top-3 cum)
  Q2 STABILITY   is the per-layer *dominant* expert the SAME expert across
                 tasks (html vs coding), across prompts within a task, and
                 across generation phases (first third vs last third)?
  Q3 BUDGET      VRAM cost of pinning top-1/top-2/top-3 per layer at Q4 vs
                 cache slots (6.75 MiB / 2-bit expert) sacrificed.

Trace schema (CSV): pos,layer,n,e0..e5,w0..w5
  e0..e5 = the 6 routed experts selected for (pos,layer); w0..w5 = their gate
  weights. NOTE: weights are NOT reliably sorted -> top-1 is argmax over w*,
  not e0. Denominator for "mass" is the sum of the 6 routed weights (shared
  expert is separate and already Q8 in this gguf, so it is out of scope).

Model geometry (k91 meta.json): DeepSeek-V4-Flash, 43 layers, first 3 dense
(hash) so 40 routed layers (3..42), 256 experts, top-6, bytes_per_expert =
7077888 = 6.75 MiB at ~2.0625 bpw (IQ2_XXS anchor).
"""
import argparse
import csv
import json
import math
import os
import statistics
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

# ---- constants (from k91 meta.json geometria + user anchors) ---------------
BYTES_PER_EXPERT = 7077888          # measured on-disk, IQ2_XXS-ish base
MIB = 1024 * 1024
BASE_MIB = BYTES_PER_EXPERT / MIB   # 6.75 MiB
BASE_BPW = 2.0625                   # IQ2_XXS anchor (user)
Q4_BPW = 4.5                        # Q4_K anchor (user)
N_ROUTED_LAYERS = 40
USABLE_CACHE_EXPERTS_12GB = 407     # meta.json risultati.usable_expert_cache_experts_12GB


def load_trace(path):
    """Return dict layer -> list of rows, each row = (top1_expert, sorted_weights_desc,
    total_mass, list_of_(expert,weight)). Robust to unsorted weights."""
    by_layer = defaultdict(list)
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        # locate columns
        eidx = [header.index(f"e{i}") for i in range(6)]
        widx = [header.index(f"w{i}") for i in range(6)]
        lidx = header.index("layer")
        for row in rd:
            if not row:
                continue
            layer = int(row[lidx])
            experts = [int(row[i]) for i in eidx]
            weights = [float(row[i]) for i in widx]
            pairs = list(zip(experts, weights))
            total = sum(weights)
            if total <= 0:
                continue
            pairs_sorted = sorted(pairs, key=lambda p: p[1], reverse=True)
            top1_expert = pairs_sorted[0][0]
            w_desc = [p[1] for p in pairs_sorted]
            by_layer[layer].append((top1_expert, w_desc, total, pairs))
    return by_layer


def dominance_stats(by_layer):
    """Per-layer dominance shares. Returns dict layer -> stats and pooled arrays."""
    per_layer = {}
    pooled_top1 = []
    for layer, rows in sorted(by_layer.items()):
        t1 = np.array([r[1][0] / r[2] for r in rows])          # top-1 share
        t2 = np.array([(r[1][0] + r[1][1]) / r[2] for r in rows])
        t3 = np.array([(r[1][0] + r[1][1] + r[1][2]) / r[2] for r in rows])
        pooled_top1.extend(t1.tolist())
        per_layer[layer] = dict(
            n=len(rows),
            top1_mean=float(t1.mean()), top1_med=float(np.median(t1)),
            top1_p10=float(np.percentile(t1, 10)), top1_p90=float(np.percentile(t1, 90)),
            top2_mean=float(t2.mean()), top3_mean=float(t3.mean()),
        )
    return per_layer, np.array(pooled_top1)


def dominant_experts(by_layer):
    """Per layer: rank experts by total accumulated gate mass over all tokens.
    Returns layer -> dict(order=[experts by mass desc], mass={expert:mass},
    total_mass, tokens, cover1 = mass share of #1 expert, top1_role = fraction of
    tokens whose argmax == #1 expert, sel_freq1 = fraction of tokens where #1
    expert appears in the selected 6)."""
    out = {}
    for layer, rows in sorted(by_layer.items()):
        mass = defaultdict(float)
        sel_count = defaultdict(int)
        top1_count = defaultdict(int)
        tokens = len(rows)
        total_mass = 0.0
        for top1_expert, w_desc, total, pairs in rows:
            top1_count[top1_expert] += 1
            for e, w in pairs:
                mass[e] += w
                sel_count[e] += 1
                total_mass += w
        order = sorted(mass, key=lambda e: mass[e], reverse=True)
        dom = order[0]
        out[layer] = dict(
            order=order,
            mass=dict(mass),
            total_mass=total_mass,
            tokens=tokens,
            dom=dom,
            cover1=mass[dom] / total_mass,
            cover_top2=(mass[order[0]] + mass[order[1]]) / total_mass if len(order) > 1 else mass[dom] / total_mass,
            cover_top3=sum(mass[order[i]] for i in range(min(3, len(order)))) / total_mass,
            top1_role=top1_count[dom] / tokens,
            sel_freq1=sel_count[dom] / tokens,
        )
    return out


def overlap_top1(dom_a, dom_b):
    """Fraction of shared layers whose #1-by-mass dominant expert is identical."""
    layers = sorted(set(dom_a) & set(dom_b))
    if not layers:
        return float("nan"), 0
    same = sum(1 for L in layers if dom_a[L]["dom"] == dom_b[L]["dom"])
    return same / len(layers), len(layers)


def overlap_topk_set(dom_a, dom_b, k=2):
    """Mean Jaccard of the top-k dominant expert *sets* per layer."""
    layers = sorted(set(dom_a) & set(dom_b))
    if not layers:
        return float("nan")
    js = []
    for L in layers:
        sa = set(dom_a[L]["order"][:k])
        sb = set(dom_b[L]["order"][:k])
        js.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(js))


def phase_split(path, frac=1 / 3):
    """Load a trace and return (dom_first, dom_last) using first/last tercile of positions."""
    by_layer_first = defaultdict(list)
    by_layer_last = defaultdict(list)
    # need positions -> re-load raw
    rows_by_pos_layer = []
    positions = set()
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        eidx = [header.index(f"e{i}") for i in range(6)]
        widx = [header.index(f"w{i}") for i in range(6)]
        lidx = header.index("layer")
        pidx = header.index("pos")
        raw = []
        for row in rd:
            if not row:
                continue
            pos = int(row[pidx]); layer = int(row[lidx])
            experts = [int(row[i]) for i in eidx]
            weights = [float(row[i]) for i in widx]
            positions.add(pos)
            raw.append((pos, layer, experts, weights))
    pos_sorted = sorted(positions)
    n = len(pos_sorted)
    lo_cut = pos_sorted[int(n * frac)]
    hi_cut = pos_sorted[int(n * (1 - frac))]
    for pos, layer, experts, weights in raw:
        total = sum(weights)
        if total <= 0:
            continue
        pairs = list(zip(experts, weights))
        pairs_sorted = sorted(pairs, key=lambda p: p[1], reverse=True)
        rec = (pairs_sorted[0][0], [p[1] for p in pairs_sorted], total, pairs)
        if pos < lo_cut:
            by_layer_first[layer].append(rec)
        elif pos >= hi_cut:
            by_layer_last[layer].append(rec)
    return dominant_experts(by_layer_first), dominant_experts(by_layer_last)


def agg(vals):
    vals = [v for v in vals if v == v]
    return dict(mean=float(np.mean(vals)), med=float(np.median(vals)),
                p10=float(np.percentile(vals, 10)), p90=float(np.percentile(vals, 90)))


def fmt_pct(x):
    return f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_reap = os.path.dirname(here)  # reap-loop repo root
    ap.add_argument("--reap-loop-root", default=default_reap)
    ap.add_argument("--moe-root",
                    default=r"C:/Users/imanu/source/repos/moe-aggressive-commit/.claude/worktrees/elastic-bose-6ae1c7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RL = args.reap_loop_root
    K91 = os.path.join(args.moe_root, "runs/reap/k91_coding_vram")
    out_dir = args.out or os.path.join(RL, "runs/ds4/20260710_e1_top_expert_mass")
    os.makedirs(out_dir, exist_ok=True)

    # ---- assemble source manifest ----------------------------------------
    # Full-model (unmasked, true gate distribution) weighted traces.
    sources = []  # (label, task, loader-callable)
    # HTML phase-1 full-model (warmup replay)
    sources.append(("html_W50", "html",
                    os.path.join(RL, "runs/ds4/20260710_pod_cache1024_warmup_replay/W50/route_W50.csv")))
    sources.append(("html_W130", "html",
                    os.path.join(RL, "runs/ds4/20260710_pod_cache1024_warmup_replay/W130/route_W130.csv")))

    # Coding full-model traces from trace_coding.tgz (11 distinct coding prompts)
    tgz = os.path.join(K91, "trace_coding.tgz")
    tmp = tempfile.mkdtemp(prefix="e1_coding_")
    with tarfile.open(tgz) as tf:
        tf.extractall(tmp)
    coding_dir = os.path.join(tmp, "trace_coding")
    for name in sorted(os.listdir(coding_dir)):
        if name.startswith("trace_") and name.endswith(".csv"):
            tag = name.replace("trace_", "").replace(".csv", "")
            sources.append((f"code_{tag}", "coding", os.path.join(coding_dir, name)))

    # ---- load + basic mask detection -------------------------------------
    loaded = {}     # label -> by_layer
    meta = {}       # label -> dict
    for label, task, path in sources:
        if not os.path.exists(path):
            continue
        bl = load_trace(path)
        # distinct experts per layer (union across 6 slots) -> mask detector
        distinct = []
        for L, rows in bl.items():
            s = set()
            for _, _, _, pairs in rows:
                for e, _ in pairs:
                    s.add(e)
            distinct.append(len(s))
        med_distinct = statistics.median(distinct)
        tokens = statistics.median([len(v) for v in bl.values()])
        loaded[label] = bl
        meta[label] = dict(task=task, path=path, tokens=int(tokens),
                           med_distinct_experts=med_distinct,
                           masked=med_distinct < 30)

    html_labels = [l for l, m in meta.items() if m["task"] == "html" and not m["masked"]]
    code_labels = [l for l, m in meta.items() if m["task"] == "coding" and not m["masked"]]

    # ---- Q1 DOMINANCE -----------------------------------------------------
    # pooled per task
    def pooled_by_layer(labels):
        merged = defaultdict(list)
        for l in labels:
            for L, rows in loaded[l].items():
                merged[L].extend(rows)
        return merged

    html_pool = pooled_by_layer(html_labels)
    code_pool = pooled_by_layer(code_labels)
    all_pool = pooled_by_layer(html_labels + code_labels)

    html_perlayer, html_t1 = dominance_stats(html_pool)
    code_perlayer, code_t1 = dominance_stats(code_pool)
    all_perlayer, all_t1 = dominance_stats(all_pool)

    # distribution across the 40 layers of the per-layer medians / means
    layer_medians = [all_perlayer[L]["top1_med"] for L in all_perlayer]
    layer_means = [all_perlayer[L]["top1_mean"] for L in all_perlayer]
    layer_top2 = [all_perlayer[L]["top2_mean"] for L in all_perlayer]
    layer_top3 = [all_perlayer[L]["top3_mean"] for L in all_perlayer]

    # ---- Q2 STABILITY -----------------------------------------------------
    dom = {l: dominant_experts(loaded[l]) for l in loaded}

    # cross-task: pooled html vs pooled coding
    dom_html_pool = dominant_experts(html_pool)
    dom_code_pool = dominant_experts(code_pool)
    xtask_top1, xtask_layers = overlap_top1(dom_html_pool, dom_code_pool)
    xtask_j2 = overlap_topk_set(dom_html_pool, dom_code_pool, 2)
    xtask_j3 = overlap_topk_set(dom_html_pool, dom_code_pool, 3)

    # cross-prompt within coding: mean pairwise top-1 agreement
    def mean_pairwise(labels):
        vals1, valsj2 = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                o1, _ = overlap_top1(dom[labels[i]], dom[labels[j]])
                j2 = overlap_topk_set(dom[labels[i]], dom[labels[j]], 2)
                vals1.append(o1); valsj2.append(j2)
        return (float(np.mean(vals1)) if vals1 else float("nan"),
                float(np.mean(valsj2)) if valsj2 else float("nan"))

    code_pw1, code_pwj2 = mean_pairwise(code_labels)
    html_pw1, html_pwj2 = mean_pairwise(html_labels)

    # cross-phase: first third vs last third (use long coding traces + html_W130)
    phase_rows = []
    phase_targets = [l for l in code_labels if meta[l]["tokens"] >= 150] + \
                    [l for l in html_labels if meta[l]["tokens"] >= 120]
    phase_vals = []
    for l in phase_targets:
        path = meta[l]["path"]
        d_first, d_last = phase_split(path)
        o1, nL = overlap_top1(d_first, d_last)
        j2 = overlap_topk_set(d_first, d_last, 2)
        phase_vals.append(o1)
        phase_rows.append((l, o1, j2))
    phase_mean = float(np.mean(phase_vals)) if phase_vals else float("nan")

    # coverage: single pinned dominant expert's share of layer mass (pooled all)
    dom_all = dominant_experts(all_pool)
    cover1_layers = [dom_all[L]["cover1"] for L in dom_all]
    cover2_layers = [dom_all[L]["cover_top2"] for L in dom_all]
    cover3_layers = [dom_all[L]["cover_top3"] for L in dom_all]
    top1role_layers = [dom_all[L]["top1_role"] for L in dom_all]
    selfreq_layers = [dom_all[L]["sel_freq1"] for L in dom_all]

    # coverage ceilings: per-task pooled and per-single-run (best case for a
    # LIVE per-session pin, no cross-task/prompt dilution).
    cover1_html_pool = float(np.mean([dom_html_pool[L]["cover1"] for L in dom_html_pool]))
    cover1_code_pool = float(np.mean([dom_code_pool[L]["cover1"] for L in dom_code_pool]))
    within_run_cover1 = []
    for l in html_labels + code_labels:
        d = dom[l]
        within_run_cover1.append(float(np.mean([d[L]["cover1"] for L in d])))
    within_run_cover1_mean = float(np.mean(within_run_cover1))

    # ---- Q3 BUDGET --------------------------------------------------------
    q4_mib = BASE_MIB * (Q4_BPW / BASE_BPW)
    budget = {}
    for k in (1, 2, 3):
        gross = k * N_ROUTED_LAYERS * q4_mib
        slots = gross / BASE_MIB
        budget[k] = dict(experts=k * N_ROUTED_LAYERS, gross_mib=gross,
                         gross_gib=gross / 1024, slots=slots,
                         pct_of_cache=slots / USABLE_CACHE_EXPERTS_12GB)

    # ---- VERDICT ----------------------------------------------------------
    dom_pooled_mean = float(all_t1.mean())
    dom_layer_med = float(np.median(layer_medians))
    cover1_mean = float(np.mean(cover1_layers))
    best_pool_cover = max(cover1_html_pool, cover1_code_pool)
    if dom_layer_med < 0.20 and dom_pooled_mean < 0.22:
        verdict = ("LA DOMINANZA E' TROPPO BASSA per aspettarsi effetto: il top-1 "
                   "routed pesa ~%s della massa, sotto la soglia 25-30%%." % fmt_pct(dom_pooled_mean))
    elif best_pool_cover < 0.12 or xtask_top1 < 0.15:
        # Per-token dominance is real, but it is NOT carried by a stable expert:
        # a single static pin captures too little mass and does not transfer.
        verdict = (
            "FRANKEN-GGUF STATICO NON VIABLE: la dominanza per-token e' reale (%s del gate mass sul "
            "top-1) MA non e' concentrata su un expert stabile. Il top-1 come identita' ruota su "
            "~150 esperti/layer: un singolo pin statico cattura solo %s della massa/layer (best-case "
            "per-task pool %s, single-run %s), e l'identita' non trasferisce cross-task (top-1 overlap "
            "%s, Jaccard-top2 %.2f). Nemmeno le varianti per output-type salvano (within-coding "
            "prompt-a-prompt overlap %s). L'unica versione con margine e' un pin LIVE per-sessione "
            "(tetto ~%s della massa), cioe' la macchina session-mask gia' esistente, NON un gguf "
            "statico. Costo comunque alto: top-1 x40 = %s del cache 407-slot." % (
                fmt_pct(dom_pooled_mean), fmt_pct(cover1_mean), fmt_pct(best_pool_cover),
                fmt_pct(within_run_cover1_mean), fmt_pct(xtask_top1), xtask_j2,
                fmt_pct(code_pw1), fmt_pct(within_run_cover1_mean),
                fmt_pct(budget[1]["pct_of_cache"])))
    elif xtask_top1 >= 0.55 and code_pw1 >= 0.55 and phase_mean >= 0.55:
        verdict = ("FRANKEN-GGUF STATICO VIABLE: dominanza %s e identita' del top-expert "
                   "stabile cross-task/prompt/fase." % fmt_pct(dom_pooled_mean))
    else:
        verdict = ("SERVONO VARIANTI PER OUTPUT-TYPE (o l'effetto e' diluito): dominanza per-token "
                   "decente (%s) ma l'identita' del top-expert per layer non e' abbastanza stabile "
                   "(cross-task top-1 overlap %s) per un singolo gguf statico." %
                   (fmt_pct(dom_pooled_mean), fmt_pct(xtask_top1)))

    # ---- write stats.json -------------------------------------------------
    stats = dict(
        sources=meta,
        html_labels=html_labels, code_labels=code_labels,
        q1=dict(
            pooled_top1_mean=dom_pooled_mean,
            pooled_top1_median=float(np.median(all_t1)),
            layer_median_of_medians=dom_layer_med,
            layer_top1_mean=agg(layer_means),
            layer_top1_median=agg(layer_medians),
            layer_top2_cum_mean=agg(layer_top2),
            layer_top3_cum_mean=agg(layer_top3),
            html_pooled_top1_mean=float(html_t1.mean()),
            code_pooled_top1_mean=float(code_t1.mean()),
        ),
        q2=dict(
            xtask_top1_overlap=xtask_top1, xtask_layers=xtask_layers,
            xtask_jaccard_top2=xtask_j2, xtask_jaccard_top3=xtask_j3,
            code_pairwise_top1=code_pw1, code_pairwise_jaccard2=code_pwj2,
            html_pairwise_top1=html_pw1, html_pairwise_jaccard2=html_pwj2,
            phase_top1_overlap=phase_mean,
            cover1_mean=cover1_mean, cover1_agg=agg(cover1_layers),
            cover2_agg=agg(cover2_layers), cover3_agg=agg(cover3_layers),
            top1_role_agg=agg(top1role_layers), sel_freq1_agg=agg(selfreq_layers),
            cover1_html_pool=cover1_html_pool, cover1_code_pool=cover1_code_pool,
            within_run_cover1_mean=within_run_cover1_mean,
        ),
        q3=dict(base_mib=BASE_MIB, base_bpw=BASE_BPW, q4_bpw=Q4_BPW, q4_mib=q4_mib,
                usable_cache_experts=USABLE_CACHE_EXPERTS_12GB, budget=budget),
        verdict=verdict,
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    # ---- write per-layer dominance table CSV (all-pool) -------------------
    with open(os.path.join(out_dir, "per_layer_dominance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "n_tok", "top1_mean", "top1_med", "top1_p10", "top1_p90",
                    "top2_cum_mean", "top3_cum_mean", "dom_expert", "cover1_massshare",
                    "top1_role_frac", "sel_freq_frac"])
        for L in sorted(all_perlayer):
            s = all_perlayer[L]; d = dom_all[L]
            w.writerow([L, s["n"], f"{s['top1_mean']:.4f}", f"{s['top1_med']:.4f}",
                        f"{s['top1_p10']:.4f}", f"{s['top1_p90']:.4f}",
                        f"{s['top2_mean']:.4f}", f"{s['top3_mean']:.4f}",
                        d["dom"], f"{d['cover1']:.4f}", f"{d['top1_role']:.4f}",
                        f"{d['sel_freq1']:.4f}"])

    # ---- pairwise top-1 identity overlap matrix (cross-prompt / cross-task) --
    order_labels = html_labels + code_labels
    with open(os.path.join(out_dir, "overlap_matrix_top1.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace"] + order_labels)
        for a in order_labels:
            rowvals = [a]
            for b in order_labels:
                o1, _ = overlap_top1(dom[a], dom[b])
                rowvals.append(f"{o1:.3f}")
            w.writerow(rowvals)

    # ---- console summary --------------------------------------------------
    print("=== E1 top-expert mass dominance / stability ===")
    print(f"sources: {len(html_labels)} html full-model, {len(code_labels)} coding full-model")
    for l in sorted(meta):
        m = meta[l]
        print(f"  {l:28s} task={m['task']:7s} tok={m['tokens']:5d} distinctE/layer~{m['med_distinct_experts']:.0f} masked={m['masked']}")
    print("\n-- Q1 dominance (routed top-1 share of the 6-expert gate mass) --")
    print(f"  pooled per-token top-1 mean = {fmt_pct(dom_pooled_mean)}  median = {fmt_pct(float(np.median(all_t1)))}")
    print(f"  per-layer top-1 median: {agg(layer_medians)}")
    print(f"  per-layer top-2 cum mean: {agg(layer_top2)}")
    print(f"  per-layer top-3 cum mean: {agg(layer_top3)}")
    print(f"  html pooled top-1 mean = {fmt_pct(float(html_t1.mean()))}   coding pooled top-1 mean = {fmt_pct(float(code_t1.mean()))}")
    print("\n-- Q2 stability (identity of #1-by-mass expert per layer) --")
    print(f"  cross-task html vs coding top-1 overlap = {fmt_pct(xtask_top1)} over {xtask_layers} layers; Jaccard top2={xtask_j2:.3f} top3={xtask_j3:.3f}")
    print(f"  within-coding pairwise top-1 overlap = {fmt_pct(code_pw1)} (Jaccard2 {code_pwj2:.3f})")
    print(f"  within-html   pairwise top-1 overlap = {fmt_pct(html_pw1)} (Jaccard2 {html_pwj2:.3f})")
    print(f"  cross-phase (first vs last third) top-1 overlap = {fmt_pct(phase_mean)}")
    print(f"  coverage: single pinned expert mass share/layer  = {fmt_pct(cover1_mean)} (median {fmt_pct(agg(cover1_layers)['med'])})")
    print(f"    coverage ceilings: html-pool {fmt_pct(cover1_html_pool)}, coding-pool {fmt_pct(cover1_code_pool)}, single-run best {fmt_pct(within_run_cover1_mean)}")
    print(f"  top-1 role of pinned expert (frac tokens it IS top1) = {fmt_pct(agg(top1role_layers)['mean'])}")
    print(f"  pinned expert selection frequency (frac tokens selected) = {fmt_pct(agg(selfreq_layers)['mean'])}")
    print("\n-- Q3 budget --")
    print(f"  base 2-bit expert = {BASE_MIB:.2f} MiB; Q4 expert = {q4_mib:.2f} MiB (x{Q4_BPW/BASE_BPW:.2f})")
    for k in (1, 2, 3):
        b = budget[k]
        print(f"  top-{k}: {b['experts']} experts -> {b['gross_mib']:.0f} MiB ({b['gross_gib']:.2f} GiB) = {b['slots']:.0f} cache slots = {fmt_pct(b['pct_of_cache'])} of 407-slot cache")
    print("\n=== VERDICT ===")
    print(verdict)
    print(f"\nwrote {out_dir}/stats.json, per_layer_dominance.csv")
    return stats


if __name__ == "__main__":
    main()
