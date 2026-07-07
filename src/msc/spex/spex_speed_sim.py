#!/usr/bin/env python3
"""
spex_speed_sim.py — Simulatore di velocita' hardware-aware per DeepSeek-V4-Flash su RTX 3060 (12 GiB).

Modello ANALITICO memory-bound a 3 livelli (VRAM-hit / RAM-miss / SSD-miss) con:
  - SPEX  : prefetch predittivo del LOADING expert -> overlap fetch<->compute
  - DSpark: speculative decoding (accept-length A), union(A) gonfia i byte-expert ma ammortizza su A token

Nessun pod/GPU: pure numpy/stdlib. Tutte le costanti derivano dalle due analisi verificate
(PARAM: decomposizione parametri/byte; HW/FORMULA: costanti hardware + formula tempo/token).
"""

import json
import os
import itertools
import numpy as np

# ============================================================================
# COSTANTI HARDWARE (blocco HW/FORMULA) — RTX 3060 12GB + DDR4 + NVMe
# ============================================================================
VRAM_BW_GBs      = 320.0   # 360 nominale derated ~0.89 (efficienza kernel reale)
RAM_BW_GBs       = 36.0    # DDR4 dual-ch, 40 teorico derated
PCIE_H2D_BW_GBs  = 14.0    # PCIe4 x16 realistico
SSD_BW_GBs       = 5.0     # NVMe Lexar EQ790 seq read
COMPUTE_TFLOP    = 12.0    # fp16 effettivo utile (memory-bound => poco sensibile)
VRAM_CAP_GB      = 11.0
RAM_CAP_GB       = 27.0

# colli di bottiglia di percorso (byte attraversa due colli in serie => min del path)
RAM_TIER_BW_GBs  = min(RAM_BW_GBs, PCIE_H2D_BW_GBs)   # RAM->VRAM limitato da PCIe = 14
SSD_TIER_BW_GBs  = min(SSD_BW_GBs, PCIE_H2D_BW_GBs)   # SSD->VRAM limitato da SSD  = 5

# ============================================================================
# COSTANTI MODELLO V4-Flash (blocco PARAM, ipotesi A: expert-hidden=4096)
# ============================================================================
GiB = float(2**30)

N_LAYERS          = 43
N_ROUTED_ACTIVE   = 6          # experts routati attivi per layer per token
BYTES_PER_PARAM   = 2          # fp16 di default per la formula tempo (override-abile)

# byte di UN expert (w1+w2+w3 SwiGLU) in fp4 = 12 MiB (dal PARAM)
EXPERT_BYTES_FP4        = 12_582_912          # 12.0 MiB
EXPERT_PARAMS           = 25_165_824          # 25.17 M

# --- byte ATTIVI per token (dal PARAM) ---
# dinamico SPEX (6 routed * 43 layer * 12 MiB fp4)
DYNAMIC_ACTIVE_BYTES = 3_246_391_296          # 3.023 GiB  (expert routati letti/token)
# statico residente in VRAM (shared fp4 + attn fp8 + embed/lmhead fp8)
STATIC_ACTIVE_BYTES  = 5_072_000_000          # ~4.724 GiB

# I byte-expert per la formula tempo usano il costo fp4 reale del path SPEX.
# (bytes_per_param resta parametrico per lo statico se si volesse variare quant;
#  qui statico e' gia' fissato in byte dai conti PARAM in fp8/fp4 mescolati.)
EXPERT_BYTES_PER_TOK = float(DYNAMIC_ACTIVE_BYTES)   # 258 experts * 12 MiB
STATIC_BYTES_PER_TOK = float(STATIC_ACTIVE_BYTES)

# FLOPs/token stimati (memory-bound => valore poco sensibile).
# 2 * active_params (routed 258*25.17M + shared + attn), MAC=2 FLOP.
ACTIVE_PARAMS_PER_TOK = (N_ROUTED_ACTIVE * N_LAYERS * EXPERT_PARAMS)  # ~6.49B routed
FLOPS_PER_TOKEN = 2.0 * ACTIVE_PARAMS_PER_TOK

# DSpark union coefficient
UNION_COEF = 0.6

# ============================================================================
# GRIGLIA SCENARI
# ============================================================================
GRID_MISS      = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
GRID_CACHE_GB  = [2, 4, 6]
GRID_RAM_FRAC  = [0.3, 0.6, 0.9]
GRID_A         = [1, 2, 2.5, 3]
GRID_SPEX      = [0, 1]

# bracket miss di riferimento (dominio-ristretto V4)
MISS_OPTIMISTIC  = (0.05, 0.10)
MISS_REALISTIC   = (0.10, 0.20)
MISS_PESSIMISTIC = 0.50


def union_A(A):
    """union(A) = 1 + (A-1)*UNION_COEF — allargamento set expert nel verify batchato."""
    return 1.0 + (A - 1.0) * UNION_COEF


def tok_per_s(miss, cache_gb, ram_frac, accept_len, spex_on,
              bytes_per_param=BYTES_PER_PARAM):
    """
    Ritorna (tok_per_s, dettaglio_dict) per una cella della griglia.

    Modello a 3 livelli memory-bound:
      - statico: sempre letto da VRAM
      - expert HIT: gia' in cache VRAM  -> VRAM_BW
      - expert MISS: quota ram_frac da RAM-tier (PCIe 14), resto da SSD-tier (5)
    SPEX overlappa il fetch sotto il compute coperto (t_static+t_hit+t_compute).
    DSpark: union(A) gonfia i byte-expert del verify, compute scala ~A, statico 1x,
            costo/token = t_verify / A.

    Nota: cache_gb non entra direttamente nella formula tempo ma e' loggato per
    correlarlo ai bracket miss (6GB->miss basso, 2GB->miss alto).
    """
    A = float(accept_len)
    u = union_A(A)

    # --- byte per il forward di verify (A token speculati) ---
    static_bytes = STATIC_BYTES_PER_TOK                      # 1x (batchato)
    expert_bytes = EXPERT_BYTES_PER_TOK * u                  # union(A) sul termine expert

    # --- tempi base (s) ---
    t_static  = static_bytes / (VRAM_BW_GBs * 1e9)
    # compute scala ~A (A token nel forward di verify)
    t_compute = (FLOPS_PER_TOKEN * A) / (COMPUTE_TFLOP * 1e12)

    # --- suddivisione expert per tier ---
    hit_bytes  = expert_bytes * (1.0 - miss)
    miss_bytes = expert_bytes * miss
    miss_ram_bytes = miss_bytes * ram_frac
    miss_ssd_bytes = miss_bytes * (1.0 - ram_frac)

    t_hit       = hit_bytes      / (VRAM_BW_GBs     * 1e9)
    t_fetch_ram = miss_ram_bytes / (RAM_TIER_BW_GBs * 1e9)
    t_fetch_ssd = miss_ssd_bytes / (SSD_TIER_BW_GBs * 1e9)
    t_fetch     = t_fetch_ram + t_fetch_ssd

    # compute coperto = cio' che gira mentre si potrebbe prefetchare
    t_compute_cover = t_static + t_hit + t_compute

    if spex_on:
        # fetch overlappato: stall residuo solo se fetch > compute coperto
        t_verify = max(t_compute_cover, t_fetch + t_compute)
    else:
        # reactive: fetch seriale, GPU stalla per l'intero trasferimento
        t_verify = t_static + t_hit + t_fetch + t_compute

    # tempo per token ACCETTATO
    t_per_token = t_verify / A
    tps = 1.0 / t_per_token if t_per_token > 0 else float('inf')

    detail = {
        "miss": miss, "cache_gb": cache_gb, "ram_frac": ram_frac,
        "accept_len": A, "spex_on": int(spex_on), "union_A": round(u, 4),
        "t_static_ms": t_static * 1e3,
        "t_hit_ms": t_hit * 1e3,
        "t_fetch_ms": t_fetch * 1e3,
        "t_fetch_ram_ms": t_fetch_ram * 1e3,
        "t_fetch_ssd_ms": t_fetch_ssd * 1e3,
        "t_compute_ms": t_compute * 1e3,
        "t_compute_cover_ms": t_compute_cover * 1e3,
        "t_verify_ms": t_verify * 1e3,
        "t_per_token_ms": t_per_token * 1e3,
        "tok_per_s": tps,
    }
    return tps, detail


def sweep():
    rows = []
    for miss, cache_gb, ram_frac, A, spex in itertools.product(
            GRID_MISS, GRID_CACHE_GB, GRID_RAM_FRAC, GRID_A, GRID_SPEX):
        tps, detail = tok_per_s(miss, cache_gb, ram_frac, A, spex)
        rows.append(detail)
    return rows


def miss_threshold_for_target(target_tps, cache_gb, ram_frac, A, spex_on,
                              lo=0.0, hi=1.0, iters=60):
    """Bisezione sul miss-rate: piu' alto il miss, piu' basso il tok/s.
    Ritorna il miss MASSIMO che tiene tps >= target (None se irraggiungibile
    anche a miss=0, o sempre raggiunto anche a miss=1)."""
    tps_lo, _ = tok_per_s(lo, cache_gb, ram_frac, A, spex_on)   # miss=0 -> tps max
    tps_hi, _ = tok_per_s(hi, cache_gb, ram_frac, A, spex_on)   # miss=1 -> tps min
    if tps_lo < target_tps:
        return None            # irraggiungibile anche a miss perfetto
    if tps_hi >= target_tps:
        return 1.0             # sempre raggiunto
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tps_mid, _ = tok_per_s(mid, cache_gb, ram_frac, A, spex_on)
        if tps_mid >= target_tps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    rows = sweep()

    # (a) MIGLIOR tok/s realistico (miss nel bracket realistico 0.10-0.20)
    realistic = [r for r in rows if MISS_REALISTIC[0] <= r["miss"] <= MISS_REALISTIC[1]]
    best_realistic = max(realistic, key=lambda r: r["tok_per_s"])
    best_overall   = max(rows, key=lambda r: r["tok_per_s"])

    # (b) SOGLIA di miss per >=10 tok/s a cache=4, ram=0.6, A=2.5, spex_on
    miss_thr = miss_threshold_for_target(10.0, cache_gb=4, ram_frac=0.6,
                                         A=2.5, spex_on=1)

    # (c) contributo separato SPEX e DSpark su un punto di riferimento realistico
    #     ref: miss=0.15, cache=4, ram=0.6
    ref_miss, ref_cache, ref_ram = 0.15, 4, 0.6
    spex_off_A25, _ = tok_per_s(ref_miss, ref_cache, ref_ram, 2.5, spex_on=0)
    spex_on_A25,  _ = tok_per_s(ref_miss, ref_cache, ref_ram, 2.5, spex_on=1)
    spex_gain_x = spex_on_A25 / spex_off_A25

    dspark_A1_spex,   _ = tok_per_s(ref_miss, ref_cache, ref_ram, 1.0, spex_on=1)
    dspark_A25_spex,  _ = tok_per_s(ref_miss, ref_cache, ref_ram, 2.5, spex_on=1)
    dspark_gain_x = dspark_A25_spex / dspark_A1_spex

    # combinato: reactive A=1 (baseline nudo) vs SPEX A=2.5 (full stack)
    baseline_naked, _ = tok_per_s(ref_miss, ref_cache, ref_ram, 1.0, spex_on=0)
    full_stack,     _ = tok_per_s(ref_miss, ref_cache, ref_ram, 2.5, spex_on=1)
    combined_gain_x = full_stack / baseline_naked

    # (d) dominio-ristretto (miss ~0.08-0.10): uso miss=0.10, cache=6, ram=0.9, A=2.5, spex_on
    dom_tps, dom_detail = tok_per_s(0.10, cache_gb=6, ram_frac=0.9,
                                    accept_len=2.5, spex_on=1)
    # anche il punto "back-of-envelope brain-dump" miss<8% -> miss=0.08
    boe_tps, boe_detail = tok_per_s(0.08, cache_gb=6, ram_frac=0.9,
                                    accept_len=2.5, spex_on=1)

    # ------------------------------------------------------------------
    # OUTPUT a video
    # ------------------------------------------------------------------
    def fmt(r):
        return (f"miss={r['miss']:.2f} cache={r['cache_gb']}GB ram={r['ram_frac']:.1f} "
                f"A={r['accept_len']:.1f} spex={r['spex_on']} "
                f"-> {r['tok_per_s']:6.2f} tok/s  "
                f"(fetch={r['t_fetch_ms']:.2f}ms cover={r['t_compute_cover_ms']:.2f}ms "
                f"tok={r['t_per_token_ms']:.2f}ms)")

    print("=" * 92)
    print("DeepSeek-V4-Flash — SPEX speed sim (RTX 3060, memory-bound 3-tier)")
    print("=" * 92)
    print(f"Costanti: VRAM_BW={VRAM_BW_GBs} RAM_tier={RAM_TIER_BW_GBs} SSD_tier={SSD_TIER_BW_GBs} GB/s "
          f"| static={STATIC_BYTES_PER_TOK/GiB:.3f}GiB expert/tok={EXPERT_BYTES_PER_TOK/GiB:.3f}GiB")
    print(f"Griglia: {len(rows)} celle "
          f"({len(GRID_MISS)}x{len(GRID_CACHE_GB)}x{len(GRID_RAM_FRAC)}x{len(GRID_A)}x{len(GRID_SPEX)})")
    print("-" * 92)

    print("\n[a] MIGLIOR tok/s (bracket realistico miss 0.10-0.20):")
    print("    " + fmt(best_realistic))
    print("    MIGLIOR tok/s ASSOLUTO (tutta la griglia):")
    print("    " + fmt(best_overall))

    print("\n[b] SOGLIA miss per >=10 tok/s @ cache=4, ram=0.6, A=2.5, spex_on:")
    if miss_thr is None:
        print("    IRRAGGIUNGIBILE anche a miss=0")
    elif miss_thr >= 1.0:
        print("    SEMPRE >=10 tok/s (anche a miss=1.0)")
    else:
        print(f"    miss_max = {miss_thr:.4f}  (={miss_thr*100:.2f}%): servono miss <= {miss_thr*100:.1f}%")

    print("\n[c] Contributo SPEX e DSpark (ref: miss=0.15, cache=4, ram=0.6):")
    print(f"    SPEX  off vs on @A=2.5 : {spex_off_A25:6.2f} -> {spex_on_A25:6.2f} tok/s  (x{spex_gain_x:.2f})")
    print(f"    DSpark A=1 vs 2.5 @spex: {dspark_A1_spex:6.2f} -> {dspark_A25_spex:6.2f} tok/s  (x{dspark_gain_x:.2f})")
    print(f"    Combinato reactive-A1 -> spex-A2.5: {baseline_naked:6.2f} -> {full_stack:6.2f} tok/s  (x{combined_gain_x:.2f})")

    print("\n[d] Dominio-ristretto (miss~0.08-0.10, cache=6, ram=0.9, A=2.5, spex_on):")
    print(f"    miss=0.10 -> {dom_tps:6.2f} tok/s")
    print(f"    miss=0.08 (back-of-envelope brain-dump <8%) -> {boe_tps:6.2f} tok/s")
    print("    Verdetto vs brain-dump (miss<8% => target): " +
          ("CONFERMATO, 10 tok/s raggiunto" if boe_tps >= 10 else "il BOE <8% NON basta da solo"))

    # tabella compatta: per ogni miss, best spex_on con A=2.5 ram=0.9 cache=6
    print("\n[tabella] best-config per miss (cache=6, ram=0.9, A=2.5, spex_on):")
    print("    miss   tok/s   t_fetch(ms)  t_cover(ms)  bound")
    for m in GRID_MISS:
        tps, d = tok_per_s(m, 6, 0.9, 2.5, spex_on=1)
        bound = "fetch" if d["t_fetch_ms"] > d["t_compute_cover_ms"] else "vram+compute"
        print(f"    {m:.2f}  {tps:6.2f}   {d['t_fetch_ms']:8.3f}   {d['t_compute_cover_ms']:8.3f}   {bound}")
    print("=" * 92)

    # ------------------------------------------------------------------
    # SALVATAGGIO JSON
    # ------------------------------------------------------------------
    out = {
        "meta": {
            "model": "DeepSeek-V4-Flash",
            "gpu": "RTX 3060 12GB",
            "note": "modello analitico memory-bound 3-tier (VRAM/RAM/SSD) + SPEX overlap + DSpark specdec",
            "hypothesis": "expert_hidden_4096 (PARAM ipotesi A, ~284B)",
        },
        "constants": {
            "VRAM_BW_GBs": VRAM_BW_GBs, "RAM_BW_GBs": RAM_BW_GBs,
            "PCIE_H2D_BW_GBs": PCIE_H2D_BW_GBs, "SSD_BW_GBs": SSD_BW_GBs,
            "RAM_TIER_BW_GBs": RAM_TIER_BW_GBs, "SSD_TIER_BW_GBs": SSD_TIER_BW_GBs,
            "COMPUTE_TFLOP": COMPUTE_TFLOP, "VRAM_CAP_GB": VRAM_CAP_GB, "RAM_CAP_GB": RAM_CAP_GB,
            "N_LAYERS": N_LAYERS, "N_ROUTED_ACTIVE": N_ROUTED_ACTIVE,
            "EXPERT_BYTES_FP4": EXPERT_BYTES_FP4,
            "STATIC_ACTIVE_BYTES": STATIC_ACTIVE_BYTES,
            "DYNAMIC_ACTIVE_BYTES": DYNAMIC_ACTIVE_BYTES,
            "FLOPS_PER_TOKEN": FLOPS_PER_TOKEN,
            "UNION_COEF": UNION_COEF,
        },
        "grid_axes": {
            "miss": GRID_MISS, "cache_gb": GRID_CACHE_GB, "ram_frac": GRID_RAM_FRAC,
            "accept_len": GRID_A, "spex": GRID_SPEX, "total_cells": len(rows),
        },
        "full_grid": rows,
        "key_points": {
            "best_realistic": best_realistic,
            "best_overall": best_overall,
            "miss_threshold_10toks": {
                "config": {"cache_gb": 4, "ram_frac": 0.6, "A": 2.5, "spex_on": 1},
                "miss_max": miss_thr,
                "miss_max_pct": (miss_thr * 100 if miss_thr is not None else None),
            },
            "spex_contribution": {
                "ref": {"miss": ref_miss, "cache_gb": ref_cache, "ram_frac": ref_ram, "A": 2.5},
                "spex_off_tps": spex_off_A25, "spex_on_tps": spex_on_A25,
                "spex_gain_x": spex_gain_x,
            },
            "dspark_contribution": {
                "ref": {"miss": ref_miss, "cache_gb": ref_cache, "ram_frac": ref_ram, "spex_on": 1},
                "A1_tps": dspark_A1_spex, "A25_tps": dspark_A25_spex,
                "dspark_gain_x": dspark_gain_x,
            },
            "combined_full_stack": {
                "baseline_reactive_A1_tps": baseline_naked,
                "full_spex_A25_tps": full_stack,
                "combined_gain_x": combined_gain_x,
            },
            "domain_restricted": {
                "miss_0.10": dom_detail,
                "miss_0.08_brain_dump_BOE": boe_detail,
                "boe_reaches_10toks": bool(boe_tps >= 10),
            },
        },
    }

    os.makedirs(r"models\spex", exist_ok=True)
    out_path = r"models\spex\v4_speed_sim.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON salvato in: {out_path}  ({len(rows)} celle in full_grid)")

    return out


if __name__ == "__main__":
    main()
