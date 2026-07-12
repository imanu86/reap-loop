#!/usr/bin/env python3
"""
Bake-virtuale coverage analysis (ds4, CPU-only, offline).

Domanda: se costruisco un keep-set top-N%/layer PER MASSA da una traccia
(train), quanto della domanda (massa di routing) di un'ALTRA traccia (test,
prompt mai visto) cade FUORI dal keep? Confronta col bound near-lossless
(~7.5-10% massa persa, dal coldtail 50%).

Dati: route.csv pesati, formato pos,layer,n,e0..e5,w0..w5. Layer MoE = 3..42
(40 layer). Pool esperti/layer = 256 (id 0..255). n sempre = 6 (top-6 router).

Registro tracce (con CAVEAT su cyberpunk, vedi sotto):
- CYBER_K91  : 20260711_highK_sweetspot/traces/trace_K91/route.csv
               ATTENZIONE: e' byte-identica a
               20260711_masked_route_traces/route_masked_K91_cyberpunk.csv
               (md5 ee4c74ef...). NON e' domanda naturale/full: e' gia'
               POST-MASK sotto una mask K91 CALIBRATA SU COFFEE (W50 phase2,
               vedi 20260711_highK_sweetspot/REPORT.md), con keep EFFETTIVO
               ~74.6 esperti/layer (36/40 layer tappati sotto K=91). Qualsiasi
               esperto che il modello cyberpunk avrebbe voluto ma che la mask
               coffee-W50 non conteneva e' STRUTTURALMENTE INVISIBILE in
               questa traccia. Trattare ogni cella che coinvolge "cyber" come
               LOWER BOUND ottimistico sul vero miss%, non come misura vera.
               Non esiste in questo dataset una traccia route.csv FULL/K0 per
               cyberpunk (k0_cyberpunk/ ha solo diag/gen/mem, niente routing).
- COFFEE_A  : 20260711_podA_narrow_traces/a_coffee_full/route.csv
              FULL (no mask), 299 tok, pod 3090, prompt "coffee compatto".
- COFFEE_K0W: 20260712_oracle50/route_k0w.csv
              FULL (no mask, DBGCAP=requested confermato dal log), prompt
              DIVERSO ("Bean & Brew" HTML5) sulla 3060 locale. TRONCATA a
              meta' riga finale (processo terminato): 80 posizioni valide
              (pos 64..143) su un target di generazione piu' lungo. Usata
              come test held-out same-domain-different-prompt per coffee.
- JSON_SHORT: 20260711_podA_narrow_traces/b_json_full/route.csv   (37 tok)
- JSON_LONG : 20260711_podA_narrow_traces/b2_json_long_full/route.csv (206 tok)
- PY_SHORT  : 20260711_podA_narrow_traces/c_python_full/route.csv (117 tok)
- PY_LONG   : 20260711_podA_narrow_traces/c2_python_long_full/route.csv (235 tok)

Tutte le _full/K0W sono greedy, temp=0, no mask REAP (confermato via
diag/stderr log). JSON e Python hanno DUE prompt distinti (short/long) per
lo stesso dominio -> permettono un vero held-out same-domain-different-prompt
anche li', non solo per coffee.
"""
import csv
import itertools
import math
import os
from collections import defaultdict

BASE = r"C:\Users\imanu\source\repos\reap-loop\runs\ds4"
OUT = os.path.join(BASE, "20260712_bake_coverage")

TRACES = {
    "CYBER_K91":   os.path.join(BASE, "20260711_highK_sweetspot", "traces", "trace_K91", "route.csv"),
    "COFFEE_A":    os.path.join(BASE, "20260711_podA_narrow_traces", "a_coffee_full", "route.csv"),
    "COFFEE_K0W":  os.path.join(BASE, "20260712_oracle50", "route_k0w.csv"),
    "JSON_SHORT":  os.path.join(BASE, "20260711_podA_narrow_traces", "b_json_full", "route.csv"),
    "JSON_LONG":   os.path.join(BASE, "20260711_podA_narrow_traces", "b2_json_long_full", "route.csv"),
    "PY_SHORT":    os.path.join(BASE, "20260711_podA_narrow_traces", "c_python_full", "route.csv"),
    "PY_LONG":     os.path.join(BASE, "20260711_podA_narrow_traces", "c2_python_long_full", "route.csv"),
}

DOMAIN_OF = {
    "CYBER_K91": "cyber", "COFFEE_A": "coffee", "COFFEE_K0W": "coffee",
    "JSON_SHORT": "json", "JSON_LONG": "json", "PY_SHORT": "python", "PY_LONG": "python",
}

LAYER_LO, LAYER_HI = 3, 42
POOL = 256
KEEP_PCTS = [50, 55, 60, 65]
KEEP_N = {50: 128, 55: 140, 60: 154, 65: 166}   # esatti come da mandato
MIB_PER_EXPERT = 6.75
FIXED_GIB = 13.3
WSL_RAM_GIB = 62.0
NEAR_LOSSLESS_BOUND_PCT = 8.0   # banda 7.5-10%, usiamo 8 come soglia centrale
NEAR_LOSSLESS_BAND = (7.5, 10.0)


def load_trace(path):
    """Ritorna dict[layer] -> list of (expert_id:int, weight:float), e n_tokens."""
    by_layer = defaultdict(list)
    positions_seen = defaultdict(set)
    n_rows_ok = 0
    n_rows_bad = 0
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[:3] == ["pos", "layer", "n"], f"header inatteso in {path}: {header}"
        for row in reader:
            if len(row) != 15:
                n_rows_bad += 1
                continue
            try:
                pos = int(row[0]); layer = int(row[1]); n = int(row[2])
                experts = [int(x) for x in row[3:9]]
                weights = [float(x) for x in row[9:15]]
            except ValueError:
                n_rows_bad += 1
                continue
            if layer < LAYER_LO or layer > LAYER_HI:
                continue
            if n != 6:
                n_rows_bad += 1
                continue
            for e, w in zip(experts, weights):
                by_layer[layer].append((e, w))
            positions_seen[layer].add(pos)
            n_rows_ok += 1
    n_tokens = max((len(s) for s in positions_seen.values()), default=0)
    n_tokens_min = min((len(s) for s in positions_seen.values()), default=0)
    return by_layer, n_tokens, n_tokens_min, n_rows_ok, n_rows_bad


print("Caricamento tracce...")
DATA = {}
for name, path in TRACES.items():
    by_layer, ntok, ntok_min, ok, bad = load_trace(path)
    DATA[name] = by_layer
    print(f"  {name:12s} tok(max/min-layer)={ntok}/{ntok_min}  righe_ok={ok} righe_scartate={bad}  layers={len(by_layer)}  src={path}")


def mass_per_expert(by_layer):
    """dict[layer] -> dict[expert] -> massa totale (somma pesi)."""
    out = {}
    for layer, pairs in by_layer.items():
        d = defaultdict(float)
        for e, w in pairs:
            d[e] += w
        out[layer] = d
    return out


def total_mass_per_layer(mass_dict):
    return {layer: sum(d.values()) for layer, d in mass_dict.items()}


MASS = {name: mass_per_expert(by_layer) for name, by_layer in DATA.items()}


def keep_set_single(train_name, keep_n):
    """Top-keep_n esperti per massa, per layer, da UNA traccia."""
    mass = MASS[train_name]
    keep = {}
    for layer, d in mass.items():
        ranked = sorted(d.items(), key=lambda kv: -kv[1])
        keep[layer] = set(e for e, _ in ranked[:keep_n])
    return keep


def keep_set_union_normalized(train_names, keep_n):
    """Mask multi-prompt: normalizza la massa per-layer a 1 in OGNI traccia
    (cosi' un prompt lungo non pesa piu' di uno corto), poi MEDIA le frazioni
    across traces, poi prende il top-keep_n per punteggio medio."""
    layers = set()
    for name in train_names:
        layers |= set(MASS[name].keys())
    keep = {}
    for layer in layers:
        combined = defaultdict(float)
        n_traces_with_layer = 0
        for name in train_names:
            d = MASS[name].get(layer)
            if not d:
                continue
            tot = sum(d.values())
            if tot <= 0:
                continue
            n_traces_with_layer += 1
            for e, m in d.items():
                combined[e] += m / tot
        if n_traces_with_layer == 0:
            continue
        for e in combined:
            combined[e] /= n_traces_with_layer
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])
        keep[layer] = set(e for e, _ in ranked[:keep_n])
    return keep


def coverage(keep, test_name):
    """Ritorna (miss_mass_pct, miss_selection_pct, test_total_mass, test_n_selections)."""
    by_layer = DATA[test_name]
    miss_mass = 0.0
    total_mass = 0.0
    miss_sel = 0
    total_sel = 0
    for layer, pairs in by_layer.items():
        k = keep.get(layer, set())
        for e, w in pairs:
            total_mass += w
            total_sel += 1
            if e not in k:
                miss_mass += w
                miss_sel += 1
    miss_mass_pct = 100.0 * miss_mass / total_mass if total_mass > 0 else float("nan")
    miss_sel_pct = 100.0 * miss_sel / total_sel if total_sel > 0 else float("nan")
    return miss_mass_pct, miss_sel_pct, total_mass, total_sel


# ---------------------------------------------------------------------------
# 1+2+3: matrice train->test, pesata e non pesata, per keep 50/55/60/65
# ---------------------------------------------------------------------------
PAIRS = [
    # richieste esplicitamente dal mandato
    ("CYBER_K91", "COFFEE_A"),
    ("COFFEE_A", "CYBER_K91"),
    ("CYBER_K91", "JSON_LONG"),
    ("CYBER_K91", "PY_LONG"),
    ("JSON_LONG", "PY_LONG"),
    # riempimento matrice per onesta' (tutte le coppie cross-dominio principali)
    ("COFFEE_A", "JSON_LONG"), ("COFFEE_A", "PY_LONG"),
    ("JSON_LONG", "COFFEE_A"), ("JSON_LONG", "CYBER_K91"),
    ("PY_LONG", "COFFEE_A"), ("PY_LONG", "CYBER_K91"), ("PY_LONG", "JSON_LONG"),
    ("CYBER_K91", "COFFEE_K0W"), ("COFFEE_K0W", "CYBER_K91"),
    # same-domain, different-prompt (vero held-out "prompt mai visto" pulito, no cross-dominio)
    ("COFFEE_A", "COFFEE_K0W"), ("COFFEE_K0W", "COFFEE_A"),
    ("JSON_SHORT", "JSON_LONG"), ("JSON_LONG", "JSON_SHORT"),
    ("PY_SHORT", "PY_LONG"), ("PY_LONG", "PY_SHORT"),
]

MULTI_PAIRS = [
    (("CYBER_K91", "COFFEE_A"), "PY_LONG"),          # richiesta esplicita: (cyber+coffee)->python
    (("CYBER_K91", "COFFEE_A"), "JSON_LONG"),
    (("COFFEE_A", "JSON_LONG", "PY_LONG"), "CYBER_K91"),  # tutto-il-resto -> cyber
    (("CYBER_K91", "JSON_LONG", "PY_LONG"), "COFFEE_A"),
    # leave-one-domain-out SENZA cyber (le 3 tracce full/K0 pulite, no caveat) —
    # la versione onesta della "unione familiare -> dominio mai visto"
    (("COFFEE_A", "JSON_LONG"), "PY_LONG"),
    (("COFFEE_A", "PY_LONG"), "JSON_LONG"),
    (("JSON_LONG", "PY_LONG"), "COFFEE_A"),
]

rows = []
for train, test in PAIRS:
    for pct in KEEP_PCTS:
        keep = keep_set_single(train, KEEP_N[pct])
        miss_mass_pct, miss_sel_pct, tot_mass, tot_sel = coverage(keep, test)
        rows.append({
            "train": train, "test": test, "keep_pct": pct, "keep_n_per_layer": KEEP_N[pct],
            "miss_mass_pct": round(miss_mass_pct, 3), "miss_selection_pct": round(miss_sel_pct, 3),
            "test_total_mass": round(tot_mass, 2), "test_n_selections": tot_sel,
            "under_bound_8pct": miss_mass_pct <= NEAR_LOSSLESS_BOUND_PCT,
        })

for trains, test in MULTI_PAIRS:
    label = "+".join(trains)
    for pct in KEEP_PCTS:
        keep = keep_set_union_normalized(trains, KEEP_N[pct])
        miss_mass_pct, miss_sel_pct, tot_mass, tot_sel = coverage(keep, test)
        rows.append({
            "train": label, "test": test, "keep_pct": pct, "keep_n_per_layer": KEEP_N[pct],
            "miss_mass_pct": round(miss_mass_pct, 3), "miss_selection_pct": round(miss_sel_pct, 3),
            "test_total_mass": round(tot_mass, 2), "test_n_selections": tot_sel,
            "under_bound_8pct": miss_mass_pct <= NEAR_LOSSLESS_BOUND_PCT,
        })

import csv as csvmod
matrix_path = os.path.join(OUT, "matrix_train_test_miss.csv")
with open(matrix_path, "w", newline="") as f:
    w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Scritto {matrix_path} ({len(rows)} righe)")

# ---------------------------------------------------------------------------
# Confronto single-prompt vs multi-prompt (mask mediata) sulla STESSA test
# ---------------------------------------------------------------------------
compare_rows = []
comparisons = [
    ("CYBER_K91", ("CYBER_K91", "COFFEE_A"), "PY_LONG"),
    ("COFFEE_A", ("CYBER_K91", "COFFEE_A"), "PY_LONG"),
    ("CYBER_K91", ("CYBER_K91", "COFFEE_A"), "JSON_LONG"),
    ("COFFEE_A", ("CYBER_K91", "COFFEE_A"), "JSON_LONG"),
]
for single, multi, test in comparisons:
    for pct in KEEP_PCTS:
        k_single = keep_set_single(single, KEEP_N[pct])
        m1, s1, _, _ = coverage(k_single, test)
        k_multi = keep_set_union_normalized(multi, KEEP_N[pct])
        m2, s2, _, _ = coverage(k_multi, test)
        compare_rows.append({
            "test": test, "keep_pct": pct,
            "single_train": single, "single_miss_mass_pct": round(m1, 3),
            "multi_train": "+".join(multi), "multi_miss_mass_pct": round(m2, 3),
            "delta_mass_pct_points": round(m2 - m1, 3),
        })
compare_path = os.path.join(OUT, "single_vs_multi_prompt.csv")
with open(compare_path, "w", newline="") as f:
    w = csvmod.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
    w.writeheader()
    w.writerows(compare_rows)
print(f"Scritto {compare_path}")

# ---------------------------------------------------------------------------
# 4: taglia finestra (GiB) per keep uniforme 50/55/60/65 (per-layer fisso)
# ---------------------------------------------------------------------------
window_rows = []
n_layers = LAYER_HI - LAYER_LO + 1
for pct in KEEP_PCTS:
    n = KEEP_N[pct]
    total_slots = n * n_layers
    gib = (total_slots * MIB_PER_EXPERT) / 1024.0 + FIXED_GIB
    window_rows.append({
        "keep_pct": pct, "keep_n_per_layer": n, "n_layers": n_layers,
        "total_kept_expert_slots": total_slots, "gib": round(gib, 2),
        "margin_vs_62gib": round(WSL_RAM_GIB - gib, 2),
        "fits_62gib": gib <= WSL_RAM_GIB,
    })
window_path = os.path.join(OUT, "window_size_uniform.csv")
with open(window_path, "w", newline="") as f:
    w = csvmod.DictWriter(f, fieldnames=list(window_rows[0].keys()))
    w.writeheader()
    w.writerows(window_rows)
print(f"Scritto {window_path}")

# ---------------------------------------------------------------------------
# 5: unione familiare — keep-set SELF per dominio (coffee, cyber, json, python)
# unito a 50/55/60/65%, quanto e' largo (per-layer, poi totale + GiB)?
# ---------------------------------------------------------------------------
DOMAIN_TRAIN = {
    "coffee": "COFFEE_A",
    "cyber": "CYBER_K91",   # caveat: post-mask, vedi sopra
    "json": "JSON_LONG",
    "python": "PY_LONG",
}
union_rows = []
union_detail_rows = []
for pct in KEEP_PCTS:
    n = KEEP_N[pct]
    per_layer_union_size = {}
    for layer in range(LAYER_LO, LAYER_HI + 1):
        u = set()
        for dom, trainname in DOMAIN_TRAIN.items():
            k = keep_set_single(trainname, n)
            u |= k.get(layer, set())
        per_layer_union_size[layer] = len(u)
    total_slots = sum(per_layer_union_size.values())
    gib = (total_slots * MIB_PER_EXPERT) / 1024.0 + FIXED_GIB
    avg_per_layer = total_slots / n_layers
    union_rows.append({
        "keep_pct": pct, "keep_n_per_layer_per_domain": n,
        "avg_union_experts_per_layer": round(avg_per_layer, 1),
        "min_union_per_layer": min(per_layer_union_size.values()),
        "max_union_per_layer": max(per_layer_union_size.values()),
        "total_kept_expert_slots": total_slots, "gib": round(gib, 2),
        "margin_vs_62gib": round(WSL_RAM_GIB - gib, 2),
        "fits_62gib": gib <= WSL_RAM_GIB,
    })
    for layer, sz in per_layer_union_size.items():
        union_detail_rows.append({"keep_pct": pct, "layer": layer, "union_experts": sz})

union_path = os.path.join(OUT, "union_family_size.csv")
with open(union_path, "w", newline="") as f:
    w = csvmod.DictWriter(f, fieldnames=list(union_rows[0].keys()))
    w.writeheader()
    w.writerows(union_rows)
print(f"Scritto {union_path}")

union_detail_path = os.path.join(OUT, "union_family_per_layer.csv")
with open(union_detail_path, "w", newline="") as f:
    w = csvmod.DictWriter(f, fieldnames=list(union_detail_rows[0].keys()))
    w.writeheader()
    w.writerows(union_detail_rows)
print(f"Scritto {union_detail_path}")

# ---------------------------------------------------------------------------
# Sanity extra: quanti esperti DISTINTI usa ciascuna traccia per layer (per
# capire se le tracce corte sono sotto-campionate rispetto a keep_n)
# ---------------------------------------------------------------------------
sanity_rows = []
for name, by_layer in DATA.items():
    for layer, pairs in by_layer.items():
        distinct = len(set(e for e, _ in pairs))
        n_tok = len(pairs) // 6
        sanity_rows.append({"trace": name, "layer": layer, "distinct_experts": distinct, "n_selections": len(pairs)})
sanity_path = os.path.join(OUT, "distinct_experts_per_layer.csv")
with open(sanity_path, "w", newline="") as f:
    w = csvmod.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
    w.writeheader()
    w.writerows(sanity_rows)
print(f"Scritto {sanity_path}")

print("\nDONE.")
