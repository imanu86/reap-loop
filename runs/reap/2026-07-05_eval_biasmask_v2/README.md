# Eval bias-mask v2 (paper-grade) — full / reap_k50 / reap_k67 / random

Run del 2026-07-05, mandato paper-grade SPEX-main. Esegue `docs/REAP_DS4_eval_plan.md`
in Stage A bias-mask con rigore statistico: **≥10 chunk appaiati/config + bootstrap CI95**,
tutti su **UNA macchina (H200)** → pairing pulito, niente jitter cross-macchina.

## Risultato — reap_k50 statisticamente lossless, K67 è l'operating-point che entra oggi

`eval_summary_v2.json` (ppl aggregata, 10 chunk dom + 10 gen, 6000 token scored/config):

### Dominio (dove gira il modello)
| config | ppl | vs full (geomean) | CI95 |
|--------|----:|------------------:|------|
| **full**     | 3.852 | 1.000 | — |
| **reap_k50** | 3.891 | **1.010×** | **[0.996, 1.025]** |
| **reap_k67** | 4.143 | **1.076×** | [1.046, 1.110] |
| **rand_s0**  | 5.346 | **1.388×** | — |

### Generale (trade-off F3)
| config | vs full | CI95 |
|--------|--------:|------|
| reap_k50 | **1.403×** | [1.296, 1.537] |
| reap_k67 | 1.892× | — |

## Verdetti (criteri pre-registrati, piano §3)

1. **reap_k50 dom 1.010× — PASS, e statisticamente LOSSLESS**: il CI95 `[0.996, 1.025]`
   **attraversa 1.0** → il potato-50% è indistinguibile dal full sul dominio. Riproduce il v1
   (1.013×, 4 chunk, macchina diversa) con 10 chunk e CI stretto.
2. **random dom 1.388× > reap 1.010× — PASS selezione-conta**: il random degrada ~5× più di
   reap. A pari K la selezione REAP vale, il random no. (Controllo v2 su 1 seed pulito H200 +
   cross-check 3090 rand_s0/s1/s2 nei file `pods/`.)
3. **reap_k67 dom 1.076× [1.046, 1.110]**: l'operating-point che **entra nei 32GB di oggi**
   (working-set expert ~25 GiB, statico in VRAM) costa solo +7.6% sul dominio. CI ben sotto 1.15.
4. **F3 su DS4: reap_k50 gen 1.403×** (potare per il dominio costa +40% sul generale) — più mite
   del Qwen-30B (~1.68× = 9.36/5.56, ledger A23/A24). Coerente con le ipotesi granularità-fine /
   shared-expert / floor-2bit (vedi README v1 §PARTIAL).

V0 mechanism-check riverificato su H200: **11280 selezioni, 0 violazioni** (`h200/biasmask.log`).

## Dose-response footprint × qualità (la manopola)

| K | keep | file | expert in RAM | RAM libera su 32GB | dom ppl vs full |
|---|------|-----:|--------------:|-------------------:|----------------:|
| K50 | 50% | 47.0 GiB | 38.8 GiB | −9.8 (serve 64GB) | **1.010×** (lossless) |
| K67 | 33% | 35.4 GiB | 27.3 GiB | **+1.7 (entra oggi)** | **1.076×** |
| K70 | 30% | 33.5 GiB | 25.3 GiB | +3.7 | ~1.11× (P4, 3090) |

Stessa bias-mask, cambia solo K → si passa da "K67 gira oggi sul 3060+32GB" a "K50 lossless
coi 64GB in arrivo" **senza rifare nulla**, solo cambiando il file mask. Materiale per il
REAP-loop dinamico (design §7.bis).

## Metodo / note

- **1 macchina (H200 141GB VRAM)**: modello 80GB interamente in VRAM (`--ssd-streaming-cache-experts 80GB`)
  → 43s/chunk (~10× il 3090), ppl identica (full dom_chunk0 3.641 = 3090). Scelta per pairing pulito
  + velocità (verdetto in ~55min invece di ~5h sul 3090 lento).
- **corpus dom DISGIUNTO** dagli item della trace/mask (`corpus/corpus_manifest.json`) — no overfit.
- Cap `-n 600/chunk`; ppl teacher-forced (`--perplexity-file`). Chunk singoli in `results_raw_h200.csv`.
- t/s: vedi `tps_hog/`, `tps_dial/` (+ README v1). Il "fits hot 32GB" empirico NON è misurabile su
  pod RunPod (mlock+cgroup+hog tutti bloccati) → conferma su workstation reale.

## Artefatti
`h200/` (results_raw, biasmask.log, run log), `results_raw_h200.csv`, `eval_summary_v2.json`,
`corpus/` (gen+manifest committati, dom gitignored), `pods/` (cross-check 3090 P1/P2/P3 + P4 K70),
`tps_hog/`, `tps_dial/`. Masks: `runs/reap/reap_mask_ds4_domain{,_k25,_k60,_k64,_k67,_k70}.json`.
