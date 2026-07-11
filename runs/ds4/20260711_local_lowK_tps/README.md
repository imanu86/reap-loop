# Probe tps low-K locale (3060) — esperimento #3 del decision model (fa35dd6)

Data: 2026-07-11 02:33-03:20. Termine di velocita' a K basso sul 3060, mai misurato prima.

Setup: fase-2 two-phase W50 (trace+frozen riusati da t4_W050/W050/r00 del batch
S2, provenienza due-fasi intatta), mask weighted K12 e K16 (build_session_mask_canonical,
sidecar .json), 400 tok, n=1 per K, temp 0, cache 256, ctx 4096, CLI-direct
(stesso path del batch S2; il campo porta-8014 dei runner_manifest storici e'
metadata: il percorso eseguito e' CLI).

## Numeri (fase-2, diag ds4)

| K | prefill t/s | generation t/s |
|---|---|---|
| 12 | 0.38 | **1.20** |
| 16 | 0.27 | **1.12** |
| 23 (batch S2, rif.) | 0.31-0.72 | 1.05-1.85 (mediana ~1.6) |

## Lettura

- **CONSERVATIVI per costruzione**: co-resident col server UI (porta 8000,
  ~2.9GB VRAM) e resident-hit~0 (processo CLI freddo, cache 256 slot); non
  sono tempi speed-clean.
- **Nessun guadagno da K piu' basso qui**: K12/K16 restano nella banda di K23
  -> su questo path il decode NON e' K-limited (domina il recall H2D/page-cache
  per-token), coerente con E-LAT (t_b ~0.95 ms/expert).
- Qualita' a occhio (NON gradata fine, budget 400 tok = incompleto per
  costruzione): entrambi doc-restart (attrattore re-prefill come nel batch),
  HTML coerente senza loop dentro il budget.

## Artefatti

- sessK12.txt/.json, sessK16.txt/.json (mask + keep-list)
- K12/gen.txt+diag.txt, K16/gen.txt+diag.txt, probe.log, lowk_probe.sh
