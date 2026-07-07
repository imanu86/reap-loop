"""Track REAP-ds4 — saliency g-only dalla trace ds4 (patch 0006) -> mask json.

Input: directory con trace_p*.csv prodotte da DS4_SPEX_TRACE_ROUTING=... +
DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 (colonne pos,layer,n,e0..e5,w0..w5, un file
per prompt = un "documento"). Opzionale: hotlist_p*.txt del profiler upstream
(DS4_EXPERT_HOTLIST) come cross-check indipendente della stessa run.

Saliency (design doc §4): S[l][e] = media condizionale del gate-weight
(somma w / n. selezioni) sui token in cui e e' selezionato; mai selezionato -> 0.
NON e' massa ne' frequenza (anomalia F-ctrl, ledger C8/G7): normalizza per il
numero di attivazioni. Approssimazione dichiarata dell'Eq.9 (manca ||f||),
validata in runs/reap/gonly_vs_eq9_30b.json.

Output: reap_mask_ds4_domain.json (formato design doc §5) con keep-list per
layer non-hash, old2new, controllo RANDOM a pari K (seed loggato), stima GiB.

Uso: python scripts/reap_saliency_ds4.py --tracedir runs/reap/<data>_trace_dominio \
        --keep-frac 0.5 [--seed 0] [--out runs/reap/reap_mask_ds4_domain.json]
"""
import argparse
import csv
import glob
import json
import os
import random

N_LAYER = 43
N_EXPERT = 256
HASH_LAYERS = [0, 1, 2]
# geometria misurata dal file reale (runs/reap/gguf_flash_expert_geometry.txt)
BYTES_PER_EXPERT = 7_077_888          # gate 2,162,688 + up 2,162,688 + down 2,752,512
BYTES_GATE_INP_ROW = 8_192            # F16 4096
BYTES_BIAS_ENTRY = 4                  # F32
FILE_BYTES_FULL = 86_720_111_488


def load_traces(tracedir):
    cnt = [[0] * N_EXPERT for _ in range(N_LAYER)]
    wsum = [[0.0] * N_EXPERT for _ in range(N_LAYER)]
    files = sorted(glob.glob(os.path.join(tracedir, "trace_p*.csv")))
    if not files:
        raise SystemExit(f"nessuna trace_p*.csv in {tracedir}")
    tok_per_file = {}
    n_nan = 0
    for path in files:
        pos_seen = set()
        with open(path) as f:
            rd = csv.reader(f)
            header = next(rd)
            if "w0" not in header:
                raise SystemExit(f"{path}: senza colonne pesi (header {header}) — "
                                 "trace girata senza DS4_SPEX_TRACE_ROUTING_WEIGHTS=1?")
            iw0 = header.index("w0")
            for row in rd:
                pos, layer, n = int(row[0]), int(row[1]), int(row[2])
                pos_seen.add(pos)
                for s in range(n):
                    e = int(row[3 + s])
                    if e < 0:
                        continue
                    w = float(row[iw0 + s])
                    if w != w:               # nan
                        n_nan += 1
                        continue
                    cnt[layer][e] += 1
                    wsum[layer][e] += w
        tok_per_file[os.path.basename(path)] = len(pos_seen)
    return cnt, wsum, tok_per_file, n_nan


def load_hotlists(tracedir):
    """Cross-check profiler upstream: somma hits/weight su tutti i hotlist_p*.txt."""
    files = sorted(glob.glob(os.path.join(tracedir, "hotlist_p*.txt")))
    if not files:
        return None
    hits = [[0] * N_EXPERT for _ in range(N_LAYER)]
    wsum = [[0.0] * N_EXPERT for _ in range(N_LAYER)]
    for path in files:
        for line in open(path):
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            l, e, h, w = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            hits[l][e] += h
            wsum[l][e] += w
    return hits, wsum


def rank_ascending(vals, cnts):
    """Ranking ascendente per saliency; tie-break: meno selezioni prima, poi id."""
    return sorted(range(N_EXPERT), key=lambda e: (vals[e], cnts[e], e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracedir", required=True)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ds4_flash_domain")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cnt, wsum, tok_per_file, n_nan = load_traces(a.tracedir)
    sal = [[(wsum[l][e] / cnt[l][e]) if cnt[l][e] else 0.0
            for e in range(N_EXPERT)] for l in range(N_LAYER)]

    keep_n = round(N_EXPERT * a.keep_frac)
    drop_n = N_EXPERT - keep_n
    rng = random.Random(a.seed)
    keep, old2new, rnd_keep = {}, {}, {}
    cnt0_per_layer, freq_gonly_overlap = {}, {}
    prunable = [l for l in range(N_LAYER) if l not in HASH_LAYERS
                and any(cnt[l])]
    for l in prunable:
        order = rank_ascending(sal[l], cnt[l])
        kept = sorted(order[drop_n:])
        keep[str(l)] = kept
        old2new[str(l)] = {str(e): i for i, e in enumerate(kept)}
        rnd = sorted(rng.sample(range(N_EXPERT), keep_n))
        rnd_keep[str(l)] = rnd
        cnt0_per_layer[str(l)] = sum(1 for e in range(N_EXPERT) if cnt[l][e] == 0)
        freq_order = sorted(range(N_EXPERT), key=lambda e: (cnt[l][e], e))
        freq_gonly_overlap[str(l)] = round(
            len(set(freq_order[:drop_n]) & set(order[:drop_n])) / drop_n, 4)

    skipped = [l for l in range(N_LAYER)
               if l not in HASH_LAYERS and l not in prunable]
    dropped_total = drop_n * len(prunable)
    est_bytes = FILE_BYTES_FULL - dropped_total * (
        BYTES_PER_EXPERT + BYTES_GATE_INP_ROW + BYTES_BIAS_ENTRY)

    hot = load_hotlists(a.tracedir)
    hot_check = None
    if hot:
        hits, hwsum = hot
        n_cmp, max_rel = 0, 0.0
        for l in prunable:
            for e in range(N_EXPERT):
                if cnt[l][e] and hits[l][e]:
                    n_cmp += 1
                    rel = abs(hwsum[l][e] / hits[l][e] - sal[l][e]) / max(sal[l][e], 1e-9)
                    max_rel = max(max_rel, rel)
        hot_check = {"n_confrontati": n_cmp, "max_rel_diff_condmean": round(max_rel, 6)}

    out = a.out or os.path.join(a.tracedir, "..", "reap_mask_ds4_domain.json")
    res = {
        "tag": f"{a.tag}_K{round((1 - a.keep_frac) * 100)}",
        "model": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
        "n_layer": N_LAYER, "n_expert": N_EXPERT, "hash_layers": HASH_LAYERS,
        "method": "gonly_conditional_mean",
        "source_tracedir": a.tracedir.replace("\\", "/"),
        "decode_tokens_per_file": tok_per_file,
        "decode_tokens_total": sum(tok_per_file.values()),
        "nan_weight_rows": n_nan,
        "keep_frac": a.keep_frac, "keep_n": keep_n,
        "layers_pruned": prunable, "layers_skipped_no_data": skipped,
        "cnt0_per_layer": cnt0_per_layer,
        "freq_vs_gonly_bottomK_overlap": freq_gonly_overlap,
        "hotlist_crosscheck": hot_check,
        "keep": keep, "old2new": old2new,
        "random_control": {"seed": a.seed, "keep": rnd_keep},
        "est_file_bytes": est_bytes,
        "est_file_gib": round(est_bytes / 2**30, 2),
        "saliency": {str(l): {str(e): round(sal[l][e], 8)
                              for e in range(N_EXPERT)} for l in prunable},
        "counts": {str(l): {str(e): cnt[l][e] for e in range(N_EXPERT)}
                   for l in prunable},
    }
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"mask scritta: {out}")
    print(f"decode tokens: {res['decode_tokens_total']} su {len(tok_per_file)} file; "
          f"nan rows: {n_nan}")
    print(f"layer potati: {len(prunable)} (skip no-data: {skipped}); "
          f"keep {keep_n}/{N_EXPERT}")
    print(f"stima file: {res['est_file_gib']} GiB")
    if hot_check:
        print(f"hotlist cross-check: {hot_check}")


if __name__ == "__main__":
    main()
