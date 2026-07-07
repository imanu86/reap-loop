# PRIOR ART & CITAZIONI — studio MoE (REAP-pruning di dominio + SPEX-offload predittivo)

**Data ricerca:** 2026-07-03. Fonte: workflow `cite-prior-art` (6 aree, ricerca web, arXiv id verificati). Confidence: ALTA su esistenza/attribuzione paper; MEDIA sulle affermazioni di non-copertura (spazio preprint 2025-26 affollato).

## Verdetto per finding

| Finding | Verdetto | Prior art più vicino |
|---|---|---|
| **F1** esperti prunabili ~50% near-lossless su dominio, informato≫random | **GIÀ NOTO** (nucleo) | EASY-EP (2504.06792, informato 45.22% vs random 3.26%), PreMoE (2505.17639), REAP (2510.13999), NAEE (2402.14800) |
| **F2** FT col router sbloccato DISPERDE; è il DOMINIO a concentrare | **PARZIALE — candidato novelty #1** | ESFT (2407.01906), Demons-in-Detail (2501.11873), Guo NeurIPS (2505.22323); contro-caso ReMoE (2412.14711) |
| **F3** pruning di dominio devasta il generale (2-3.5× ppl) | **GIÀ NOTO** | EASY-EP Table 4, PreMoE, Less-is-MoE (2606.05538) |
| **F4** predicibilità cross-layer; statico collassa cross-distribution, condizionale tiene | **PARZIALE — novelty #2 (la robustezza)** | Fate (2502.12224, ~99% hit), Pre-gated MoE (2308.12066), SiDA (2310.18859), ExpertFlow (2410.17954) |
| **F5** offload predittivo, 235B full-quality su ~12GB | **MECCANISMI NOTI, regime estremo** | ProMoE (2410.22134), MoE-Infinity (2401.14361), Mixtral-offloading (2312.17238), PowerInfer-2 (2406.06282), HOBBIT (2411.01433) |
| **F6** mixed-precision cold experts (hot int4 / cold int2) | **GIÀ NOTO** | MC-MoE (2410.06270, ILP freq+routing), MxMoE (2505.05799), MoQE (2310.02410), MoPEQ (2509.02512) |

## Cosa NON è nostro (onesto)
- F1 nucleo, F3 intero, F6 nucleo, F5 meccanismi, F4 predicibilità-base: tutti pre-empted. **Rischio novelty maggiore: EASY-EP + PreMoE (2025) su F1+F3.**

## Cosa POTREBBE essere nostro
1. **F2 come dissociazione causale controllata** — isolare "sbloccare+FT il router disperde / il DOMINIO concentra" come variabile unica (FT-vs-noFT × dominio-vs-generale, 30B→235B). Non trovato pubblicato isolato. *Va difeso da: ReMoE (segno opposto) + confondenti geometria (Xi Wang 2604.09780) / standing-committee (Yan Wang 2601.03425).*
2. **F4 come contrasto di ROBUSTEZZA** — condizionale tiene / statico collassa cross-distribution, con misura random-vs-Markov a budget fisso (+0.13@25%). Non testato nei paper verificati.
3. **La COMBINAZIONE end-to-end** (probabilmente il vero contributo): pruning-saliency di dominio → hot/cold da N_eff di DOMINIO → resident-set dell'offload predittivo → precisione per-esperto sulla stessa distribuzione → prefetch cross-layer. Nessuno co-progetta i pezzi attorno alla concentrazione di dominio.
4. **Le misure con controlli espliciti**: random-vs-informato a budget matched; N_eff ~50/128; rapporto ppl dominio-vs-generale.
5. **Il regime**: 235B full-quality su ~12GB (più estremo dei prior: Mixtral-47B su 16-24GB).

## Must-cite (28 paper, arXiv id verificati) — vedi task output completo
EASY-EP 2504.06792 · PreMoE 2505.17639 · REAP 2510.13999 · NAEE 2402.14800 · ESFT 2407.01906 · Demons-in-Detail 2501.11873 · Guo 2505.22323 · ST-MoE 2202.08906 · ReMoE 2412.14711 · Xi-Wang 2604.09780 · Yan-Wang 2601.03425 · Mixtral 2401.04088 · Fate 2502.12224 · Pre-gated 2308.12066 · SiDA 2310.18859 · ExpertFlow 2410.17954 · Mixtral-offloading 2312.17238 · MoE-Infinity 2401.14361 · ProMoE 2410.22134 · PowerInfer-2 2406.06282 · HOBBIT 2411.01433 · MC-MoE 2410.06270 · MxMoE 2505.05799 · MoQE 2310.02410 · MoPEQ 2509.02512 · DeepSeekMoE 2401.06066 · How-to-Score-Experts 2606.15716 · Less-is-MoE 2606.05538

## Prossimo passo ricerca
Secondo giro mirato: (a) escludere che qualcuno abbia già unito "bit-per-esperto guidati dal dominio" + "offload predittivo cross-layer"; (b) trovare (se esiste) il paper che isola FT-router-vs-dominio come F2.
