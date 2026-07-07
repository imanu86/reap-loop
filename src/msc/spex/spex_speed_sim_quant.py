#!/usr/bin/env python3
"""
spex_speed_sim_quant.py — Simulatore di velocita' hardware-aware per DeepSeek-V4-Flash
su RTX 3060 (12 GiB), ESTESO con QUANTIZZAZIONE PER-TIER.

Copia estesa di spex_speed_sim.py. Novita' rispetto al sim fp4 originale:

  1) PRECISIONE EXPERT parametrica per TIER (hot vs cold, bpw configurabile).
     - byte/expert = f(bpw). I RESIDENTI (hit in cache VRAM) sono i piu' caldi -> hot_bytes.
     - I MISS sono la CODA FREDDA (expert raramente toccati) -> fetch usa cold_bytes.
       Questo e' il punto chiave: il fetch pesca dalla coda fredda, quindi paga la
       quantizzazione piu' aggressiva del cold-tier (che e' anche il piu' economico
       da trasferire su PCIe/SSD).

  2) PRECISIONE STATICO parametrica (FP16 vs fp8).
     - Dal METODO ds4+byte: sul 3060 il blocco STATICO NON entra in VRAM12
       ne' in FP16 (33.95 GB) ne' in fp8 (16.98 GB). Cio' che eccede VRAM_CAP
       si legge da RAM via PCIe A OGNI TOKEN (costo fisso aggiuntivo, non
       overlappabile perche' e' sul percorso critico dell'attention/shared).
       -> t_static = (parte_in_VRAM)/VRAM_BW + (parte_in_RAM)/PCIe_BW.
       Questa e' la divergenza Mac-vs-3060 che il METODO chiede di codificare:
       Mac 96GB tiene tutto lo statico in FP16 in unified memory; il 3060 no.

  3) 3 CONFIG principali (ds4 release + estensioni temperatura-aware):
     - C_fp4     : release ufficiale ipotizzata. hot=cold=4.0 bpw, static fp8.
     - C_dwarf   : DwarfStar-4 uniforme. hot=cold=2.3 bpw, static FP16.
     - C_tempmix : estensione temp-mix. hot=2.3 bpw, cold=1.58 bpw, static fp8.

Nessun pod/GPU: pure numpy/stdlib. Costanti hardware ereditate dal sim originale.
"""

import json
import os
import itertools

# ============================================================================
# COSTANTI HARDWARE (identiche al sim originale) — RTX 3060 12GB + DDR4 + NVMe
# ============================================================================
VRAM_BW_GBs      = 320.0
RAM_BW_GBs       = 36.0
PCIE_H2D_BW_GBs  = 14.0
SSD_BW_GBs       = 5.0
COMPUTE_TFLOP    = 12.0
VRAM_CAP_GB      = 11.0     # capacita' utile VRAM (12 GiB - overhead driver/ctx)
RAM_CAP_GB       = 27.0

RAM_TIER_BW_GBs  = min(RAM_BW_GBs, PCIE_H2D_BW_GBs)   # 14 (RAM->VRAM strozzato da PCIe)
SSD_TIER_BW_GBs  = min(SSD_BW_GBs, PCIE_H2D_BW_GBs)   # 5

# ============================================================================
# COSTANTI MODELLO V4-Flash
# ============================================================================
GiB = float(2**30)
MB  = 1_000_000.0          # i byte/expert dal METODO sono in MB (base-10, MB)
GB  = 1_000_000_000.0

N_LAYERS          = 43
N_ROUTED_ACTIVE   = 6          # experts routati attivi per layer per token
EXPERT_PARAMS     = 25_165_824 # 25.17 M params/expert (3*2048*4096)

# ---- byte/expert per bpw (dal METODO ds4+byte, MB base-10) ----
#   fp4 (4.0 bpw)    = 12.58 MB
#   dwarf (2.3 bpw)  = 7.24  MB
#   1.58 bpw         = 4.97  MB
#   1.0 bpw          = 3.15  MB
# formula generale: bytes = params * bpw / 8
def expert_bytes_from_bpw(bpw):
    """byte di UN expert (w1+w2+w3 SwiGLU) alla precisione bpw."""
    return EXPERT_PARAMS * bpw / 8.0

# check di coerenza col METODO (MB)
_EXP_BYTES_REF = {4.0: 12.58e6, 2.3: 7.24e6, 1.58: 4.97e6, 1.0: 3.15e6}

# ---- STATICO (dal METODO ds4+byte, stima NAIVE/upper per budget VRAM) ----
# static FP16 = 33.95 GB ; static fp8 = 16.98 GB (GB base-10)
STATIC_FP16_GB = 33.95
STATIC_FP8_GB  = 16.98
# quota di statico che serve leggere PER TOKEN dal percorso critico.
# Lo statico completo (attn+shared+router+embed) e' letto ogni forward: e' il
# termine t_static del sim originale. Nel sim fp4 originale STATIC_BYTES_PER_TOK
# valeva ~4.724 GiB perche' assumeva statico gia' in VRAM in fp4/fp8 misto e
# NON contava embed/lmhead full. Qui modelliamo il costo REALE del blocco statico
# residente: byte statici letti/token = footprint statico / (fattore riuso).
# Assunzione: l'intero blocco statico e' toccato una volta per forward-token
# (attention+shared+router densi), quindi byte_static_letti = footprint_statico.
# Questo e' il caso onesto/pessimistico per il 3060.

# ============================================================================
# COSTANTI SPEX / DSpark (identiche al sim originale)
# ============================================================================
UNION_COEF = 0.6

def union_A(A):
    return 1.0 + (A - 1.0) * UNION_COEF

# FLOPs/token (memory-bound => poco sensibile)
ACTIVE_PARAMS_PER_TOK = (N_ROUTED_ACTIVE * N_LAYERS * EXPERT_PARAMS)
FLOPS_PER_TOKEN = 2.0 * ACTIVE_PARAMS_PER_TOK

# ============================================================================
# FOOTPRINT UNION-DOMINIO (dal brief HOT/COLD) per il verdetto union_fits_44
# ============================================================================
# geometria V4: 256 exp/layer x 43 layer. union-dominio = 170.4/256 exp/layer,
# hot=104.1/layer, cold=66.3/layer (bracket 99%, hot-set 90%).
V4_HOT_EXP_PER_LAYER  = 104.1
V4_COLD_EXP_PER_LAYER = 66.3
BUDGET_44GB = 44.0   # VRAM12 + RAM32

def union_footprint_gb(hot_bytes, cold_bytes):
    """GB (base-10) per tenere l'intera union-dominio residente:
    hot experts a hot_bytes, cold experts a cold_bytes, su 43 layer."""
    per_layer = V4_HOT_EXP_PER_LAYER * hot_bytes + V4_COLD_EXP_PER_LAYER * cold_bytes
    return per_layer * N_LAYERS / GB


# ============================================================================
# GRIGLIA SCENARI (come da consegna)
# ============================================================================
GRID_MISS      = [0.05, 0.10, 0.15, 0.20]
GRID_RAM_FRAC  = [0.6, 0.9]
GRID_A         = [2, 2.5]
GRID_SPEX      = [0, 1]

# ============================================================================
# CONFIG di quantizzazione
# ============================================================================
CONFIGS = {
    "C_fp4": {
        "desc": "release ufficiale ipotizzata: hot=cold=fp4 (4.0 bpw), static fp8",
        "hot_bpw": 4.0, "cold_bpw": 4.0, "static": "fp8",
    },
    "C_dwarf": {
        "desc": "DwarfStar-4 uniforme: hot=cold=2.3 bpw, static FP16",
        "hot_bpw": 2.3, "cold_bpw": 2.3, "static": "fp16",
    },
    "C_tempmix": {
        "desc": "temp-mix (estensione): hot=2.3 bpw, cold=1.58 bpw, static fp8",
        "hot_bpw": 2.3, "cold_bpw": 1.58, "static": "fp8",
    },
}

def static_footprint_gb(static_prec):
    return STATIC_FP16_GB if static_prec == "fp16" else STATIC_FP8_GB


def tok_per_s(cfg, miss, ram_frac, accept_len, spex_on):
    """
    Ritorna (tok_per_s, dettaglio) per una cella, con quantizzazione per-tier.

    EXPERT:
      - hit (residenti in VRAM)   -> hot_bytes @ VRAM_BW
      - miss (coda fredda, fetch) -> cold_bytes, quota ram_frac da RAM-tier (PCIe14),
                                     resto da SSD-tier (5)
    STATICO:
      - footprint statico (fp16/fp8). Parte in VRAM (fino a VRAM_CAP) @ VRAM_BW,
        eccesso letto da RAM via PCIe @ ogni token (costo fisso non overlappabile).
    SPEX: overlappa il fetch sotto il compute coperto (t_static+t_hit+t_compute).
    DSpark: union(A) gonfia i byte-expert del verify, compute ~A, costo/token = t_verify/A.
    """
    A = float(accept_len)
    u = union_A(A)

    hot_bytes  = expert_bytes_from_bpw(cfg["hot_bpw"])
    cold_bytes = expert_bytes_from_bpw(cfg["cold_bpw"])

    # --- byte expert attivi/token (6 routed * 43 layer), gonfiati da union(A) ---
    n_active = N_ROUTED_ACTIVE * N_LAYERS
    # split hit/miss sul NUMERO di expert, poi ogni tier col suo peso in byte
    n_active_u = n_active * u
    n_hit  = n_active_u * (1.0 - miss)     # residenti caldi
    n_miss = n_active_u * miss             # coda fredda da fetchare

    hit_bytes_tot  = n_hit  * hot_bytes    # residenti letti da VRAM (hot quant)
    miss_bytes_tot = n_miss * cold_bytes   # miss = coda fredda (cold quant)

    miss_ram_bytes = miss_bytes_tot * ram_frac
    miss_ssd_bytes = miss_bytes_tot * (1.0 - ram_frac)

    # --- STATICO: split VRAM-resident vs RAM-overflow (letto via PCIe/token) ---
    static_gb   = static_footprint_gb(cfg["static"])
    static_byt  = static_gb * GB
    vram_cap_by = VRAM_CAP_GB * GB
    static_in_vram_by  = min(static_byt, vram_cap_by)
    static_in_ram_by   = max(0.0, static_byt - vram_cap_by)

    t_static_vram = static_in_vram_by / (VRAM_BW_GBs   * 1e9)
    t_static_ram  = static_in_ram_by  / (PCIE_H2D_BW_GBs * 1e9)   # overflow via PCIe
    t_static      = t_static_vram + t_static_ram                 # sempre sul path critico
    # variante "expert-only": statico ipoteticamente TUTTO in VRAM (isola il
    # segnale della quantizzazione EXPERT dal penalty di precisione statica).
    t_static_novflw = static_in_vram_by / (VRAM_BW_GBs * 1e9)

    # --- tempi expert ---
    t_hit       = hit_bytes_tot  / (VRAM_BW_GBs     * 1e9)
    t_fetch_ram = miss_ram_bytes / (RAM_TIER_BW_GBs * 1e9)
    t_fetch_ssd = miss_ssd_bytes / (SSD_TIER_BW_GBs * 1e9)
    t_fetch     = t_fetch_ram + t_fetch_ssd

    # --- compute (scala ~A) ---
    t_compute = (FLOPS_PER_TOKEN * A) / (COMPUTE_TFLOP * 1e12)

    # compute coperto = cio' che gira mentre si prefetcha
    t_compute_cover = t_static + t_hit + t_compute

    if spex_on:
        t_verify = max(t_compute_cover, t_fetch + t_compute)
    else:
        t_verify = t_static + t_hit + t_fetch + t_compute

    t_per_token = t_verify / A
    tps = 1.0 / t_per_token if t_per_token > 0 else float('inf')

    # tok/s "expert-only" (statico fittiziamente in VRAM): stessa formula ma
    # t_static -> t_static_novflw. Mostra cosa rende la sola quant EXPERT.
    cover_no = t_static_novflw + t_hit + t_compute
    if spex_on:
        t_verify_no = max(cover_no, t_fetch + t_compute)
    else:
        t_verify_no = t_static_novflw + t_hit + t_fetch + t_compute
    tps_expert_only = A / t_verify_no if t_verify_no > 0 else float('inf')

    detail = {
        "config": None,  # riempito dal chiamante
        "miss": miss, "ram_frac": ram_frac, "accept_len": A,
        "spex_on": int(spex_on), "union_A": round(u, 4),
        "hot_bytes_MB": round(hot_bytes / MB, 3),
        "cold_bytes_MB": round(cold_bytes / MB, 3),
        "static_gb": static_gb,
        "static_in_ram_gb": round(static_in_ram_by / GB, 3),
        "t_static_ms": t_static * 1e3,
        "t_static_ram_ms": t_static_ram * 1e3,
        "t_hit_ms": t_hit * 1e3,
        "t_fetch_ms": t_fetch * 1e3,
        "t_compute_ms": t_compute * 1e3,
        "t_compute_cover_ms": t_compute_cover * 1e3,
        "t_verify_ms": t_verify * 1e3,
        "t_per_token_ms": t_per_token * 1e3,
        "tok_per_s": tps,
        "tok_per_s_expert_only": tps_expert_only,
        "bound": "fetch" if (spex_on and t_fetch + t_compute > t_compute_cover)
                 else ("fetch" if not spex_on and t_fetch > 0 else "vram+static+compute"),
    }
    return tps, detail


def sweep_config(name, cfg):
    rows = []
    for miss, ram_frac, A, spex in itertools.product(
            GRID_MISS, GRID_RAM_FRAC, GRID_A, GRID_SPEX):
        tps, d = tok_per_s(cfg, miss, ram_frac, A, spex)
        d["config"] = name
        rows.append(d)
    return rows


def main():
    # baseline dal sim fp4 originale (dalla consegna e verificato dal JSON)
    ORIG_DOM_TPS  = 45.71   # miss.10, cache6, ram.9, A2.5, spex_on
    ORIG_BEST_TPS = 74.44   # best_overall griglia originale
    ORIG_HONEST_LO, ORIG_HONEST_HI = 30.0, 45.0

    all_rows = {}
    key_results = {}

    print("=" * 100)
    print("DeepSeek-V4-Flash — SPEX speed sim QUANT (RTX 3060, per-tier quantization)")
    print("=" * 100)
    print(f"Baseline sim fp4 originale: dom~{ORIG_DOM_TPS} best~{ORIG_BEST_TPS} onesto {ORIG_HONEST_LO}-{ORIG_HONEST_HI} tok/s")
    print(f"Griglia per config: miss{GRID_MISS} x ram{GRID_RAM_FRAC} x A{GRID_A} x spex{GRID_SPEX} "
          f"= {len(GRID_MISS)*len(GRID_RAM_FRAC)*len(GRID_A)*len(GRID_SPEX)} celle")
    print("-" * 100)

    for name, cfg in CONFIGS.items():
        rows = sweep_config(name, cfg)
        all_rows[name] = rows

        hot_b  = expert_bytes_from_bpw(cfg["hot_bpw"])
        cold_b = expert_bytes_from_bpw(cfg["cold_bpw"])
        u_gb   = union_footprint_gb(hot_b, cold_b)
        fits44 = u_gb <= BUDGET_44GB

        # punto DOMINIO: miss=.10, ram=.9, A=2.5, spex_on
        dom_tps, dom_d = tok_per_s(cfg, 0.10, 0.9, 2.5, 1)
        dom_eo = dom_d["tok_per_s_expert_only"]
        # best della config (tutta la sua griglia)
        best = max(rows, key=lambda r: r["tok_per_s"])
        best_eo = max(rows, key=lambda r: r["tok_per_s_expert_only"])
        # onesto: bracket realistico miss .10-.20, ram .6-.9, spex_on, media
        honest = [r for r in rows if 0.10 <= r["miss"] <= 0.20 and r["spex_on"] == 1]
        honest_lo = min(honest, key=lambda r: r["tok_per_s"])["tok_per_s"]
        honest_hi = max(honest, key=lambda r: r["tok_per_s"])["tok_per_s"]

        key_results[name] = {
            "desc": cfg["desc"],
            "hot_bpw": cfg["hot_bpw"], "cold_bpw": cfg["cold_bpw"],
            "static_prec": cfg["static"],
            "hot_bytes_MB": round(hot_b / MB, 3),
            "cold_bytes_MB": round(cold_b / MB, 3),
            "union_footprint_gb": round(u_gb, 2),
            "union_fits_44": bool(fits44),
            "gb_left_44": round(BUDGET_44GB - u_gb, 2),
            "toks_dominio": round(dom_tps, 2),
            "toks_best": round(best["tok_per_s"], 2),
            "toks_dominio_expert_only": round(dom_eo, 2),
            "toks_best_expert_only": round(best_eo["tok_per_s_expert_only"], 2),
            "best_cell": {k: best[k] for k in ("miss", "ram_frac", "accept_len", "spex_on")},
            "honest_range": [round(honest_lo, 2), round(honest_hi, 2)],
            "static_in_ram_gb": round(dom_d["static_in_ram_gb"], 2),
            "delta_vs_fp4_orig": {
                "dom_abs": round(dom_tps - ORIG_DOM_TPS, 2),
                "dom_x": round(dom_tps / ORIG_DOM_TPS, 3),
                "best_abs": round(best["tok_per_s"] - ORIG_BEST_TPS, 2),
                "best_x": round(best["tok_per_s"] / ORIG_BEST_TPS, 3),
            },
        }

        print(f"\n### {name}: {cfg['desc']}")
        print(f"    hot={hot_b/MB:.2f}MB cold={cold_b/MB:.2f}MB static={cfg['static']} "
              f"({static_footprint_gb(cfg['static'])}GB, overflow RAM={dom_d['static_in_ram_gb']:.1f}GB/tok)")
        print(f"    union-dominio footprint = {u_gb:.2f} GB -> entra in 44GB? {'SI' if fits44 else 'NO'} "
              f"({BUDGET_44GB-u_gb:+.2f} GB)")
        print(f"    DOMINIO (miss.10 ram.9 A2.5 spex) = {dom_tps:6.2f} tok/s   "
              f"(vs fp4 dom {ORIG_DOM_TPS} -> {dom_tps-ORIG_DOM_TPS:+.2f}, x{dom_tps/ORIG_DOM_TPS:.2f})")
        print(f"    BEST config               = {best['tok_per_s']:6.2f} tok/s   "
              f"(vs fp4 best {ORIG_BEST_TPS} -> {best['tok_per_s']-ORIG_BEST_TPS:+.2f}, x{best['tok_per_s']/ORIG_BEST_TPS:.2f})")
        print(f"    onesto (miss.10-.20 spex) = {honest_lo:.2f} - {honest_hi:.2f} tok/s")

    # ------------------------------------------------------------------
    # DELTA esplicito: quanto guadagna il 2-bit, e quanto IN PIU' il temp-mix
    # ------------------------------------------------------------------
    dom_fp4     = key_results["C_fp4"]["toks_dominio"]
    dom_dwarf   = key_results["C_dwarf"]["toks_dominio"]
    dom_tempmix = key_results["C_tempmix"]["toks_dominio"]

    # expert-only (statico ipoteticamente in VRAM): isola la quantizzazione EXPERT
    eo_fp4     = key_results["C_fp4"]["toks_dominio_expert_only"]
    eo_dwarf   = key_results["C_dwarf"]["toks_dominio_expert_only"]
    eo_tempmix = key_results["C_tempmix"]["toks_dominio_expert_only"]

    gain_2bit_vs_fp4      = dom_dwarf - dom_fp4          # cosa aggiunge il 2-bit uniforme
    gain_tempmix_vs_dwarf = dom_tempmix - dom_dwarf      # cosa aggiunge in piu' il temp-mix
    gain_tempmix_vs_fp4   = dom_tempmix - dom_fp4        # totale temp-mix vs fp4

    # gli stessi delta ma sul percorso EXPERT puro (senza penalty statico)
    eo_gain_2bit_vs_fp4      = eo_dwarf - eo_fp4
    eo_gain_tempmix_vs_dwarf = eo_tempmix - eo_dwarf
    eo_gain_tempmix_vs_fp4   = eo_tempmix - eo_fp4

    summary = {
        "baseline_fp4_orig": {"dom": ORIG_DOM_TPS, "best": ORIG_BEST_TPS,
                              "honest": [ORIG_HONEST_LO, ORIG_HONEST_HI]},
        "dom_2bit_gain_vs_fp4_release": {
            "abs": round(gain_2bit_vs_fp4, 2),
            "x": round(dom_dwarf / dom_fp4, 3),
            "note": f"C_dwarf {dom_dwarf} vs C_fp4 {dom_fp4} tok/s @dominio",
        },
        "dom_tempmix_extra_vs_2bit": {
            "abs": round(gain_tempmix_vs_dwarf, 2),
            "x": round(dom_tempmix / dom_dwarf, 3),
            "note": f"C_tempmix {dom_tempmix} vs C_dwarf {dom_dwarf} tok/s @dominio",
        },
        "dom_tempmix_total_vs_fp4": {
            "abs": round(gain_tempmix_vs_fp4, 2),
            "x": round(dom_tempmix / dom_fp4, 3),
        },
        "EXPERT_ONLY_dom": {
            "note": "statico ipoteticamente in VRAM: isola la sola quant EXPERT dal penalty statico (che sul 3060 domina il tempo reale)",
            "fp4": eo_fp4, "dwarf2b": eo_dwarf, "tempmix": eo_tempmix,
            "2bit_gain_vs_fp4": {"abs": round(eo_gain_2bit_vs_fp4, 2),
                                 "x": round(eo_dwarf / eo_fp4, 3)},
            "tempmix_extra_vs_2bit": {"abs": round(eo_gain_tempmix_vs_dwarf, 2),
                                      "x": round(eo_tempmix / eo_dwarf, 3)},
            "tempmix_total_vs_fp4": {"abs": round(eo_gain_tempmix_vs_fp4, 2),
                                     "x": round(eo_tempmix / eo_fp4, 3)},
        },
        "union_fits_44": {k: key_results[k]["union_fits_44"] for k in CONFIGS},
        "verdict": (
            "Sul 3060 REALE il tempo/token e' dominato dal blocco STATICO che sfora "
            "VRAM12 e si legge da RAM via PCIe ogni forward: fp16-static (C_dwarf) e' "
            "PENALIZZANTE (22.9GB overflow) nonostante gli expert a 2-bit. Le config con "
            "static fp8 (C_fp4, C_tempmix) tengono l'overflow a 6GB e vincono. Sul percorso "
            "EXPERT puro invece il 2-bit e il temp-mix danno il guadagno atteso. Nessuna "
            "config fa entrare la union-dominio nei 44GB con la stima NAIVE dello statico: "
            "serve lo statico MLA-realistic (~24GB) o quantizzato sotto fp8 per chiudere."
        ),
    }

    print("\n" + "=" * 100)
    print("DELTA (punto DOMINIO miss.10 ram.9 A2.5 spex) — TOTALE (statico reale sul 3060):")
    print(f"  C_fp4 (release, static fp8)   = {dom_fp4:6.2f} tok/s   union44? {key_results['C_fp4']['union_fits_44']}")
    print(f"  C_dwarf (2-bit, static fp16)  = {dom_dwarf:6.2f} tok/s   union44? {key_results['C_dwarf']['union_fits_44']}   "
          f"-> {gain_2bit_vs_fp4:+.2f} vs fp4 (fp16-static PENALIZZA: 22.9GB overflow/tok)")
    print(f"  C_tempmix (2.3/1.58, fp8)     = {dom_tempmix:6.2f} tok/s   union44? {key_results['C_tempmix']['union_fits_44']}   "
          f"-> {gain_tempmix_vs_dwarf:+.2f} vs dwarf, {gain_tempmix_vs_fp4:+.2f} vs fp4")
    print("\nDELTA EXPERT-ONLY (statico ipotet. in VRAM: isola la quant EXPERT):")
    print(f"  C_fp4     = {eo_fp4:6.2f} tok/s")
    print(f"  C_dwarf   = {eo_dwarf:6.2f} tok/s   -> il 2-bit AGGIUNGE {eo_gain_2bit_vs_fp4:+.2f} (x{eo_dwarf/eo_fp4:.2f}) vs fp4")
    print(f"  C_tempmix = {eo_tempmix:6.2f} tok/s   -> il temp-mix AGGIUNGE {eo_gain_tempmix_vs_dwarf:+.2f} (x{eo_tempmix/eo_dwarf:.2f}) IN PIU' vs 2-bit; "
          f"totale {eo_gain_tempmix_vs_fp4:+.2f} (x{eo_tempmix/eo_fp4:.2f}) vs fp4")
    print("=" * 100)

    out = {
        "meta": {
            "model": "DeepSeek-V4-Flash",
            "gpu": "RTX 3060 12GB",
            "note": "sim memory-bound 3-tier + SPEX + DSpark, ESTESO con quantizzazione per-tier (hot/cold) e statico (fp16/fp8)",
            "extends": "spex_speed_sim.py / v4_speed_sim.json",
            "key_divergence": "statico NON entra in VRAM12 su 3060 (fp16 33.95GB, fp8 16.98GB): overflow letto da RAM via PCIe ogni token. Su Mac 96GB starebbe in FP16 in unified memory.",
        },
        "constants": {
            "VRAM_BW_GBs": VRAM_BW_GBs, "RAM_BW_GBs": RAM_BW_GBs,
            "PCIE_H2D_BW_GBs": PCIE_H2D_BW_GBs, "SSD_BW_GBs": SSD_BW_GBs,
            "RAM_TIER_BW_GBs": RAM_TIER_BW_GBs, "SSD_TIER_BW_GBs": SSD_TIER_BW_GBs,
            "COMPUTE_TFLOP": COMPUTE_TFLOP, "VRAM_CAP_GB": VRAM_CAP_GB, "RAM_CAP_GB": RAM_CAP_GB,
            "N_LAYERS": N_LAYERS, "N_ROUTED_ACTIVE": N_ROUTED_ACTIVE,
            "EXPERT_PARAMS": EXPERT_PARAMS,
            "STATIC_FP16_GB": STATIC_FP16_GB, "STATIC_FP8_GB": STATIC_FP8_GB,
            "FLOPS_PER_TOKEN": FLOPS_PER_TOKEN, "UNION_COEF": UNION_COEF,
            "V4_HOT_EXP_PER_LAYER": V4_HOT_EXP_PER_LAYER,
            "V4_COLD_EXP_PER_LAYER": V4_COLD_EXP_PER_LAYER,
            "BUDGET_44GB": BUDGET_44GB,
        },
        "grid_axes": {
            "miss": GRID_MISS, "ram_frac": GRID_RAM_FRAC,
            "accept_len": GRID_A, "spex": GRID_SPEX,
            "cells_per_config": len(GRID_MISS)*len(GRID_RAM_FRAC)*len(GRID_A)*len(GRID_SPEX),
        },
        "configs": {k: {**CONFIGS[k],
                        "hot_bytes_MB": key_results[k]["hot_bytes_MB"],
                        "cold_bytes_MB": key_results[k]["cold_bytes_MB"],
                        "union_footprint_gb": key_results[k]["union_footprint_gb"],
                        "union_fits_44": key_results[k]["union_fits_44"]}
                    for k in CONFIGS},
        "key_results": {**key_results, "_summary_delta": summary},
        "full_grid": all_rows,
    }

    os.makedirs(r"models\spex", exist_ok=True)
    out_path = r"models\spex\v4_speed_sim_quant.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON salvato in: {out_path}")
    return out


if __name__ == "__main__":
    main()
