#!/usr/bin/env python3
"""E-PHASE -- structural phase-mask analysis (piecewise-static design evidence).

OFFLINE ONLY. Reads routing-weight traces already on disk (no GPU / WSL / pod).

User hypothesis under test:
  "Expert demand changes with the STRUCTURAL PHASE of the generated document
  (CSS vs body-markup vs script), so the right mask is piecewise-static with a
  relearn at structural boundaries."  E1 measured cross-phase top-1 overlap
  53.6% with a BLIND TEMPORAL split (first vs last third). Here we segment by
  the *actual structure of the generated text* and ask:

  (a) how much does the frozen warmup mask (first 50 gen tokens = PACE_WARMUP)
      starve each later structural phase, vs the local per-phase optimum?
  (b) how well does the previous phase's local mask cover the next phase?
  (c) how many experts per layer change between adjacent phase masks
      (= delta-prefetch cost of a relearn at the boundary)?
  (d) does structure explain more than blind time? (within-phase split-half
      mask stability vs cross-phase stability; dominant-expert overlap vs the
      E1 temporal 53.6% reference)

Data reality (declared, see REPORT):
  - Full-model weighted traces exist ONLY for: html warmup replay W50 (49 tok,
    head-markup) / W130 (129 tok, head-markup + CSS) and 11 coding prompts
    (52-255 tok, prose-intro + code). NO full-model trace reaches the
    HTML body/JS phase -> the JS leg of the hypothesis is tested via the
    coding analog (prose -> code register switch) and flagged as a data gap.
  - scope_divergence_pod r1 logs only aggregate S1 (pos,layer,pruned,total):
    no expert ids -> unusable for mask building. ctrl has text but no trace.
  - trace_ab_* routing.csv are mask-constrained (SOTA runs): the selected-6
    set is biased by the applied mask -> excluded from free-router masks.
  - knee route_*.csv have no weight columns; loop/*.tgz are mask-constrained.

Trace schema: pos,layer,n,e0..e5,w0..w5 (weights NOT sorted; unbiased router
probs of the 6 selected experts -- def-1 "routed mass" denominator, as E-CAL).

Alignment token<->text: traces log generated tokens only (pos starts at
prompt_len); text = tw_W*.txt (html) or gen_*.log minus 'ds4: ' diagnostics
(coding). Char offset -> token index mapped PROPORTIONALLY (no tokenizer on
disk): boundary error is a few tokens, so every boundary metric is reported
with a +/-5-token sensitivity band. Confidence: HIGH that the window order and
extent are right, MEDIUM on the exact boundary token.
"""
import argparse
import csv
import json
import os
import re
import statistics
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

MIB_PER_EXPERT = 6.75          # 2-bit routed expert footprint (k91 meta.json)
N_ROUTED_LAYERS = 40           # layers 3..42
PACE_WARMUP = 50               # tokens seen before the mask freezes/engages
E1_TEMPORAL_TOP1_OVERLAP = 0.536  # E1 blind first-vs-last-third reference
MIN_PHASE_TOK = 12
K_LIST = (23, 38)              # REAP-LOOP keep + E-CAL Kmin-cov90
SENS = 5                       # +/- tokens boundary sensitivity


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_trace(path):
    """Return (positions_sorted, rows) with rows[t] = list of (layer, [(e,w)x6])
    indexed by token order t (0-based over generated tokens)."""
    by_pos = defaultdict(list)
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        eidx = [header.index(f"e{i}") for i in range(6)]
        widx = [header.index(f"w{i}") for i in range(6)]
        lidx = header.index("layer")
        pidx = header.index("pos")
        for row in rd:
            if not row:
                continue
            pos = int(row[pidx]); layer = int(row[lidx])
            pairs = [(int(row[eidx[i]]), float(row[widx[i]])) for i in range(6)]
            by_pos[pos].append((layer, pairs))
    positions = sorted(by_pos)
    rows = [by_pos[p] for p in positions]
    return positions, rows


def extract_gen_text_from_log(path):
    """Strip interleaved 'ds4: ...' diagnostics from a gen_*.log.
    Diag lines carry their own newline, so a partial line ('Hereds4: ...')
    joins directly with the next text line."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    out = []
    for line in raw.split("\n"):
        if "ds4: " in line:
            pre = line.split("ds4: ", 1)[0]
            if pre:
                out.append(pre)          # no newline: diag stole it
        else:
            out.append(line + "\n")
    return "".join(out).rstrip("\n")


def load_sess_mask(path):
    """sess_W*.txt = 'layer expert' lines. Detect prune-list vs keep-list by
    per-layer count; return dict layer -> kept expert set."""
    per_layer = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            L, e = int(parts[0]), int(parts[1])
            per_layer[L].add(e)
    counts = [len(s) for s in per_layer.values()]
    med = statistics.median(counts) if counts else 0
    if med > 128:   # prune list -> kept = complement
        return {L: set(range(256)) - s for L, s in per_layer.items()}
    return dict(per_layer)


# --------------------------------------------------------------------------
# mask math (def-1 routed mass, per layer, mean over layers)
# --------------------------------------------------------------------------
def layer_mass(rows, t0, t1):
    """dict layer -> dict expert -> accumulated mass over token window [t0,t1)."""
    acc = defaultdict(lambda: defaultdict(float))
    for t in range(max(0, t0), min(len(rows), t1)):
        for layer, pairs in rows[t]:
            for e, w in pairs:
                acc[layer][e] += w
    return acc


def topk_mask(mass, k):
    return {L: set(sorted(m, key=lambda e: m[e], reverse=True)[:k])
            for L, m in mass.items()}


def coverage(rows, t0, t1, mask):
    """mean over layers of (mass on mask experts / total routed mass) in window."""
    cov_num = defaultdict(float); cov_den = defaultdict(float)
    for t in range(max(0, t0), min(len(rows), t1)):
        for layer, pairs in rows[t]:
            kept = mask.get(layer, set())
            for e, w in pairs:
                cov_den[layer] += w
                if e in kept:
                    cov_num[layer] += w
    vals = [cov_num[L] / cov_den[L] for L in cov_den if cov_den[L] > 0]
    return float(np.mean(vals)) if vals else float("nan")


def jaccard_masks(a, b):
    layers = sorted(set(a) & set(b))
    if not layers:
        return float("nan")
    return float(np.mean([len(a[L] & b[L]) / len(a[L] | b[L]) for L in layers]))


def churn_masks(a, b, k):
    """per-layer #experts in b not in a (entering set at the boundary)."""
    layers = sorted(set(a) & set(b))
    per_layer = [len(b[L] - a[L]) for L in layers]
    total = int(np.sum(per_layer)) if per_layer else 0
    return (float(np.mean(per_layer)) if per_layer else float("nan"),
            total, total * MIB_PER_EXPERT)


def dominant_top1_overlap(mass_a, mass_b):
    """fraction of shared layers whose #1-by-mass expert is identical (E1 metric)."""
    layers = sorted(set(mass_a) & set(mass_b))
    if not layers:
        return float("nan")
    same = 0
    for L in layers:
        da = max(mass_a[L], key=lambda e: mass_a[L][e])
        db = max(mass_b[L], key=lambda e: mass_b[L][e])
        same += (da == db)
    return same / len(layers)


# --------------------------------------------------------------------------
# structural segmentation
# --------------------------------------------------------------------------
def segments_html(text):
    """head (pre-<style>), css (inside style), markup (post-</style>), js
    (inside <script>) -- whatever the text actually reaches."""
    segs = []
    i_style = text.find("<style")
    if i_style < 0:
        return [("head", 0, len(text))]
    segs.append(("head", 0, i_style))
    i_end = text.find("</style>", i_style)
    if i_end < 0:
        segs.append(("css", i_style, len(text)))
        return segs
    segs.append(("css", i_style, i_end))
    i_script = text.find("<script", i_end)
    if i_script < 0:
        segs.append(("markup", i_end, len(text)))
        return segs
    segs.append(("markup", i_end, i_script))
    segs.append(("js", i_script, len(text)))
    return segs


def segments_code(text, tag):
    """prose (before first ```), code (inside the fence). For python prompts,
    sub-split code into imports / def+docstring / body when markers exist."""
    i = text.find("```")
    if i < 0:
        return [("prose", 0, len(text))]
    j = text.find("\n", i)
    code_start = (j + 1) if j >= 0 else i + 3
    segs = [("prose", 0, i)]
    code_end = len(text)
    tail = text.rfind("```")
    if tail > i:
        code_end = tail
    if tag in ("python-csv", "py-asyncio"):
        m = re.search(r"^def |^async def ", text[code_start:code_end], re.M)
        if m:
            d = code_start + m.start()
            doc = re.search(r'"""', text[d:code_end])
            e2 = None
            if doc:
                doc2 = re.search(r'"""', text[d + doc.end():code_end])
                if doc2:
                    e2 = d + doc.end() + doc2.end()
            if e2:
                segs += [("imports", code_start, d), ("def_doc", d, e2),
                         ("body", e2, code_end)]
                return segs
    segs.append(("code", code_start, code_end))
    return segs


def char_to_tok(c, total_chars, ntok):
    if total_chars <= 0:
        return 0
    return max(0, min(ntok, int(round(ntok * c / total_chars))))


# --------------------------------------------------------------------------
# per-trace analysis
# --------------------------------------------------------------------------
def analyze_trace(label, rows, text, segs_char, out_rows, boundary_rows, k_list):
    ntok = len(rows)
    C = len(text)
    segs = []
    for name, c0, c1 in segs_char:
        t0, t1 = char_to_tok(c0, C, ntok), char_to_tok(c1, C, ntok)
        if t1 - t0 >= MIN_PHASE_TOK:
            segs.append((name, t0, t1))
    if not segs:
        return None

    warm_end = min(PACE_WARMUP, ntok)
    warm_mass = layer_mass(rows, 0, warm_end)
    res = dict(label=label, ntok=ntok, phases=[], boundaries=[])

    for k in k_list:
        warm_mask = topk_mask(warm_mass, k)
        prev_mask = None
        prev_mass = None
        prev_name = None
        for name, t0, t1 in segs:
            mass = layer_mass(rows, t0, t1)
            local = topk_mask(mass, k)
            cov_local = coverage(rows, t0, t1, local)
            cov_warm = coverage(rows, t0, t1, warm_mask)
            # boundary sensitivity on the frozen-mask coverage
            lo = coverage(rows, max(0, t0 - SENS), t1, warm_mask)
            hi = coverage(rows, min(t1 - MIN_PHASE_TOK, t0 + SENS), t1, warm_mask)
            cov_prev = coverage(rows, t0, t1, prev_mask) if prev_mask else float("nan")
            in_warm = t1 <= warm_end
            overlap_warm = t0 < warm_end
            rec = dict(trace=label, K=k, phase=name, t0=t0, t1=t1, ntok=t1 - t0,
                       cov_local=cov_local, cov_warmup_frozen=cov_warm,
                       cov_warm_sens_lo=min(lo, hi), cov_warm_sens_hi=max(lo, hi),
                       gain_local_vs_frozen=cov_local - cov_warm,
                       cov_prev_phase=cov_prev,
                       in_warmup=in_warm, overlaps_warmup=overlap_warm)
            res["phases"].append(rec)
            out_rows.append(rec)
            if prev_mask is not None:
                jac = jaccard_masks(prev_mask, local)
                ch_mean, ch_tot, ch_mib = churn_masks(prev_mask, local, k)
                dom = dominant_top1_overlap(prev_mass, mass)
                brec = dict(trace=label, K=k, boundary=f"{prev_name}->{name}",
                            jaccard=jac, churn_per_layer=ch_mean,
                            churn_total=ch_tot, delta_prefetch_mib=ch_mib,
                            dom_top1_overlap=dom)
                res["boundaries"].append(brec)
                boundary_rows.append(brec)
            prev_mask, prev_mass, prev_name = local, mass, name

        # temporal controls on the LARGEST phase: split-half stability
        big = max(segs, key=lambda s: s[2] - s[1])
        name, t0, t1 = big
        if t1 - t0 >= 2 * MIN_PHASE_TOK:
            mid = (t0 + t1) // 2
            m1 = topk_mask(layer_mass(rows, t0, mid), k)
            m2mass = layer_mass(rows, mid, t1)
            m2 = topk_mask(m2mass, k)
            res[f"withinphase_jaccard_K{k}"] = jaccard_masks(m1, m2)
            res[f"withinphase_cov_K{k}"] = coverage(rows, mid, t1, m1)
            res[f"withinphase_covlocal_K{k}"] = coverage(rows, mid, t1, m2)
            res[f"withinphase_dom_K{k}"] = dominant_top1_overlap(
                layer_mass(rows, t0, mid), m2mass)
            res[f"withinphase_name_K{k}"] = name
        # E1 replication: blind first vs last third
        a = topk_mask(layer_mass(rows, 0, ntok // 3), k)
        b = topk_mask(layer_mass(rows, ntok - ntok // 3, ntok), k)
        res[f"tercile_jaccard_K{k}"] = jaccard_masks(a, b)
        res[f"tercile_dom_K{k}"] = dominant_top1_overlap(
            layer_mass(rows, 0, ntok // 3),
            layer_mass(rows, ntok - ntok // 3, ntok))
    return res


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--reap-loop-root", default=os.path.dirname(here))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    RL = args.reap_loop_root
    out_dir = args.out or os.path.join(
        RL, "runs/ds4/20260710_ephase_structural_masks")
    os.makedirs(out_dir, exist_ok=True)

    replay = os.path.join(RL, "runs/ds4/20260710_pod_cache1024_warmup_replay")
    tgz = os.path.join(RL, "runs/reap/k91_coding_vram/trace_coding.tgz")

    out_rows, boundary_rows, traces = [], [], {}

    # --- html W130 (head -> css) + W50 (warmup donor) ----------------------
    posW130, rowsW130 = load_trace(os.path.join(replay, "W130/route_W130.csv"))
    with open(os.path.join(replay, "W130/tw_W130.txt"), encoding="utf-8") as fh:
        textW130 = fh.read()
    traces["html_W130"] = analyze_trace(
        "html_W130", rowsW130, textW130, segments_html(textW130),
        out_rows, boundary_rows, K_LIST)

    posW50, rowsW50 = load_trace(os.path.join(replay, "W50/route_W50.csv"))
    with open(os.path.join(replay, "W50/tw_W50.txt"), encoding="utf-8") as fh:
        textW50 = fh.read()

    # real frozen mask from disk: sess_W50 kept-set (the mask phase 2 ran with)
    sess_w50 = load_sess_mask(os.path.join(replay, "W50/sess_W50.txt"))
    computed_w50 = topk_mask(layer_mass(rowsW50, 0, len(rowsW50)), 23)
    sess_validation = jaccard_masks(sess_w50, computed_w50)

    # cross-run: sess_W50 (frozen, head-markup provenance) on W130's phases
    cross_run = {}
    for name, c0, c1 in segments_html(textW130):
        t0 = char_to_tok(c0, len(textW130), len(rowsW130))
        t1 = char_to_tok(c1, len(textW130), len(rowsW130))
        if t1 - t0 >= MIN_PHASE_TOK:
            cross_run[name] = coverage(rowsW130, t0, t1, sess_w50)

    # --- coding traces ------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="ephase_coding_")
    with tarfile.open(tgz) as tf:
        tf.extractall(tmp)
    cdir = os.path.join(tmp, "trace_coding")
    for name in sorted(os.listdir(cdir)):
        m = re.match(r"trace_(p\d+_c\d+_(.+))\.csv", name)
        if not m:
            continue
        full, tag = m.group(1), m.group(2)
        gen = os.path.join(cdir, f"gen_{full}.log")
        if not os.path.exists(gen):
            continue
        _, rows = load_trace(os.path.join(cdir, name))
        text = extract_gen_text_from_log(gen)
        traces[f"code_{tag}"] = analyze_trace(
            f"code_{tag}", rows, text, segments_code(text, tag),
            out_rows, boundary_rows, K_LIST)

    traces = {k: v for k, v in traces.items() if v}

    # --- aggregates ---------------------------------------------------------
    def agg_phase(pred, key, k):
        vals = [p[key] for t in traces.values() for p in t["phases"]
                if p["K"] == k and pred(p) and p[key] == p[key]]
        return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

    stats = dict(
        sources={l: dict(ntok=t["ntok"],
                         phases=[(p["phase"], p["ntok"]) for p in t["phases"]
                                 if p["K"] == K_LIST[0]])
                 for l, t in traces.items()},
        sess_w50_validation_jaccard_vs_computed_top23=sess_validation,
        cross_run_sess_w50_on_W130=cross_run,
        e1_temporal_reference_top1_overlap=E1_TEMPORAL_TOP1_OVERLAP,
    )
    post = lambda p: not p["in_warmup"]          # phases beyond the warmup window
    outw = lambda p: not p["overlaps_warmup"]    # phases fully after warmup
    for k in K_LIST:
        g_all, n_all = agg_phase(post, "gain_local_vs_frozen", k)
        g_out, n_out = agg_phase(outw, "gain_local_vs_frozen", k)
        cw, _ = agg_phase(post, "cov_warmup_frozen", k)
        cl, _ = agg_phase(post, "cov_local", k)
        cp, _ = agg_phase(post, "cov_prev_phase", k)
        bj = [b["jaccard"] for t in traces.values() for b in t["boundaries"]
              if b["K"] == k]
        bch = [b["churn_per_layer"] for t in traces.values()
               for b in t["boundaries"] if b["K"] == k]
        bmib = [b["delta_prefetch_mib"] for t in traces.values()
                for b in t["boundaries"] if b["K"] == k]
        bdom = [b["dom_top1_overlap"] for t in traces.values()
                for b in t["boundaries"] if b["K"] == k]
        wj = [t[f"withinphase_jaccard_K{k}"] for t in traces.values()
              if f"withinphase_jaccard_K{k}" in t]
        wc = [t[f"withinphase_cov_K{k}"] for t in traces.values()
              if f"withinphase_cov_K{k}" in t]
        wcl = [t[f"withinphase_covlocal_K{k}"] for t in traces.values()
               if f"withinphase_covlocal_K{k}" in t]
        wd = [t[f"withinphase_dom_K{k}"] for t in traces.values()
              if f"withinphase_dom_K{k}" in t]
        tj = [t[f"tercile_jaccard_K{k}"] for t in traces.values()]
        td = [t[f"tercile_dom_K{k}"] for t in traces.values()]
        stats[f"K{k}"] = dict(
            mean_gain_local_vs_frozen_postwarmup=g_all, n_phases=n_all,
            mean_gain_fully_out_of_warmup=g_out, n_out=n_out,
            mean_cov_warmup_frozen_postwarmup=cw,
            mean_cov_local_postwarmup=cl,
            mean_cov_prev_phase=cp,
            crossphase_mask_jaccard=float(np.nanmean(bj)) if bj else None,
            crossphase_churn_per_layer=float(np.nanmean(bch)) if bch else None,
            crossphase_delta_prefetch_mib=float(np.nanmean(bmib)) if bmib else None,
            crossphase_dom_top1_overlap=float(np.nanmean(bdom)) if bdom else None,
            withinphase_mask_jaccard=float(np.nanmean(wj)) if wj else None,
            withinphase_splithalf_cov=float(np.nanmean(wc)) if wc else None,
            withinphase_splithalf_covlocal=float(np.nanmean(wcl)) if wcl else None,
            withinphase_dom_top1_overlap=float(np.nanmean(wd)) if wd else None,
            tercile_mask_jaccard=float(np.nanmean(tj)) if tj else None,
            tercile_dom_top1_overlap=float(np.nanmean(td)) if td else None,
        )

    # --- verdict -------------------------------------------------------------
    k23 = stats["K23"]
    gain = k23["mean_gain_local_vs_frozen_postwarmup"]
    struct_signal = (k23["withinphase_mask_jaccard"] or 0) - \
                    (k23["crossphase_mask_jaccard"] or 0)
    if gain >= 0.15 and struct_signal >= 0.10:
        verdict = ("POSITIVO: la mask per-fase guadagna %.1f punti di copertura "
                   "vs la warmup-frozen (K=23) e la struttura e' reale "
                   "(stabilita' within-phase %.2f vs cross-phase %.2f Jaccard). "
                   "Il design piecewise-static con relearn ai confini "
                   "strutturali e' giustificato." % (
                       100 * gain, k23["withinphase_mask_jaccard"],
                       k23["crossphase_mask_jaccard"]))
    elif gain >= 0.10:
        verdict = ("MARGINALE: guadagno mask-per-fase %.1f punti (soglia 10-15); "
                   "segnale strutturale within-vs-cross = %.2f. Design "
                   "piecewise-static plausibile ma da confermare con trace "
                   "full-model che raggiungano body/JS." % (100 * gain,
                                                            struct_signal))
    else:
        verdict = ("NEGATIVO: guadagno mask-per-fase %.1f punti (<10): la mask "
                   "warmup-frozen copre gia' quasi quanto l'ottimo locale; il "
                   "relearn ai confini strutturali non paga sui dati "
                   "disponibili." % (100 * gain))
    stats["verdict"] = verdict

    # --- write ---------------------------------------------------------------
    with open(os.path.join(out_dir, "phase_coverage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    with open(os.path.join(out_dir, "boundary_churn.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(boundary_rows[0].keys()))
        w.writeheader()
        for r in boundary_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    # --- console ---------------------------------------------------------------
    print("=== E-PHASE structural phase-mask analysis ===")
    print(f"sess_W50 kept-set vs computed top-23 Jaccard = {sess_validation:.3f}")
    print(f"cross-run sess_W50 coverage on W130 phases: " +
          ", ".join(f"{k}={v:.3f}" for k, v in cross_run.items()))
    for k in K_LIST:
        s = stats[f"K{k}"]
        print(f"\n-- K={k} --")
        print(f"  post-warmup phases: cov frozen={s['mean_cov_warmup_frozen_postwarmup']:.3f} "
              f"local={s['mean_cov_local_postwarmup']:.3f} "
              f"gain={s['mean_gain_local_vs_frozen_postwarmup']*100:.1f}pt "
              f"(n={s['n_phases']}) | fully-out gain="
              f"{s['mean_gain_fully_out_of_warmup']*100:.1f}pt (n={s['n_out']})")
        print(f"  prev-phase mask cov on next = {s['mean_cov_prev_phase']:.3f}")
        print(f"  mask Jaccard: cross-phase={s['crossphase_mask_jaccard']:.3f} "
              f"within-phase={s['withinphase_mask_jaccard']:.3f} "
              f"tercile(blind)={s['tercile_mask_jaccard']:.3f}")
        print(f"  dom top-1 overlap: cross-phase={s['crossphase_dom_top1_overlap']:.3f} "
              f"within-phase={s['withinphase_dom_top1_overlap']:.3f} "
              f"tercile={s['tercile_dom_top1_overlap']:.3f} "
              f"(E1 temporal ref {E1_TEMPORAL_TOP1_OVERLAP})")
        print(f"  boundary churn: {s['crossphase_churn_per_layer']:.1f} experts/layer, "
              f"delta-prefetch ~{s['crossphase_delta_prefetch_mib']:.0f} MiB")
    print("\n=== VERDICT ===")
    print(verdict)
    print(f"\nwrote {out_dir}/phase_coverage.csv, boundary_churn.csv, stats.json")


if __name__ == "__main__":
    main()
