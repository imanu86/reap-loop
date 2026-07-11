# Masked route traces (REAL post-mask) — 2026-07-11

Route.csv **VERE sotto mask frozen** (non proxy full-model). Emesse con
`DS4_REAP_MASK_FILE=sessK{K}.txt` + `DS4_SPEX_TRACE_ROUTING=<path>` +
`DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`, cyberpunk (wide) e coffee (narrow), greedy, cache32.

Formato: `pos,layer,n,e0..e5,w0..w5` (esperti SELEZIONATI post-mask + pesi).
~11960 righe/file = ~299 token generazione x 40 layer MoE.
Verifica: 0.000% dei pick su esperti potati (mask enforcement reale).

- `route_masked_K12_cyberpunk.csv`  (K12, wide)
- `route_masked_K16_cyberpunk.csv`  (K16, wide)
- `route_masked_K23_cyberpunk.csv`  (K23, wide)
- `route_masked_K38_cyberpunk.csv`  (K38, wide)
- `route_masked_K48_cyberpunk.csv`  (K48, wide)   — aggiunta 2026-07-11 (high-K sweep)
- `route_masked_K64_cyberpunk.csv`  (K64, wide)   — aggiunta 2026-07-11 (high-K sweep)
- `route_masked_K91_cyberpunk.csv`  (K91 eff 74.6, wide) — aggiunta 2026-07-11 (high-K sweep)
- `route_masked_K12_coffee.csv`     (K12, narrow)
- `route_masked_K23_coffee.csv`     (K23, narrow)

Sweep cyberpunk completo per K = 12/16/23/38/48/64/91 (0.000% pick su esperti potati su tutte;
distinti-usati/layer ≈ keep). Sorgente + verdetto in `../20260711_highK_sweetspot/REPORT.md`.

**PRONTE per re-run phase-segmentation SENZA caveat-proxy.** Colmano il gap-G2 di
`runs/ds4/20260711_phase_segmented_usage/phase_segmented_usage.py` e `pin_analysis.py`
("no masked route.csv exists — full-model filtered to keep"). Caricare con `keep=None`
(gia' post-mask). Dettagli + curva velocita' in
`../20260711_local_clean_lowK/CURVE_COMPLETE.md`.
