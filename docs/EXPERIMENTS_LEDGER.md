# EXPERIMENTS LEDGER — MoE Aggressive-Commit (REAP-pruning + FT + SPEX)

**Scopo:** registro anti-ripetizione. Ogni test/esperimento delle ultime 3 chat (pruning K-sweep, fine-tuning, dominio-vs-generale, mixed-precision, 235B, SPEX gate+loop, prior-art) censito una sola volta, con numeri esatti, file dati e script che lo rigenera. **Prima di lanciare un pod o rifare una misura, cercala qui.**

**Data:** 2026-07-03 (studio REAP+SPEX chiuso).

**Dove stanno i dati:**
- **Risultati eval (JSON piccoli):** `*.json` (eval_*, stepf_*, mixedprec_*, subexpert_*, reap_saliency_*, hidden_predict_*)
- **Trace SPEX (npz):** `spex/*.npz` (traces_q30_*, traces_q235_*, traces_olmoe_*, hidden_scores_*)
- **Script pod (canonici):** `scripts_pod/*.py` (dump_traces, hidden_predict, spex_dump_hidden, markov_gate, markov_gate_cross, spex_speed_sim, reap_spex_combo, mixed_precision_eval, reap_saliency, step_f, prune_validate, build_general_en/it, manual_merge, debug_gate)
- **Loop SPEX:** `src\msc\spex\spex_loop.py` (DSpark-fedele, gira in locale) + `src\msc\spex\continuous_cache_sim_ORIG.py` (versione euristica ORIG, baseline)
- **Narrativa/consolidamento:** `docs\HANDOFF_5_paper_study.md`, `docs\CONSOLIDATION.md`, `docs\PRIOR_ART.md`, `docs\SPEX_spec.md`, `docs\SPEX_LOOP.md`
- **Script sorgente storici (scratchpad esecuzione):** sessione `d9c25753-...` e `df18069a-...` (stessi script poi consolidati in `scripts_pod/`)

**Note di lettura obbligatorie:**
- **validity = 0.0 in TUTTE le eval field** → campo mal-definito (audience non nel testo). La metrica di riferimento è `overall_excl_validity` (6 campi), non `overall` grezzo (depresso dallo zero).
- **Due scale di perplexity diverse, non confrontabili in assoluto:** (a) serie `prune_validate --metric perplexity` (dom ~4-6, gen ~5.7) e (b) serie `mixed_precision_eval` (dom ~17.7, gen ~6.0, subset/setup diverso). Confronta solo delta interni a ciascuna serie.
- **"saliency/sal" ≠ REAP-saliency.** I tag `eval_sal25/50/70` usano la MASSA-di-routing (frequency proxy da stepf). La vera REAP-saliency (Eq.9, media condizionale g·‖f‖) è nei tag `*_reap` e `eval_sal_dom/gen_*` (maskfile `reap_saliency_base.json`). Quindi il primo sweep è "frequency-vs-random", non "saliency-vs-frequency".
- **235B (GPTQ-int4) vs 30B (bf16):** quantizzazione diversa → NON confrontare numeri assoluti cross-modello. È un confondente da dichiarare.

---

## 1. TABELLA MASTER

Deduplicata: duplicati esatti collassati in una riga (fonti citate). Categorie: **A** Pruning, **B** Mixed-precision, **C** 235B, **D** Fine-tuning, **E** SPEX gate, **F** SPEX loop, **G** Prior-art/novelty.

### A) PRUNING (30B, K-sweep; dom-vs-gen perplexity; per-neurone)

| # | Cat | Esperimento | Setup | Risultato (numeri esatti) | Verdetto | File dati | Script | Stato |
|---|-----|-------------|-------|---------------------------|----------|-----------|--------|-------|
| A1 | Pruning | Baseline field K=0 | Qwen3-30B-A3B, N=152, metric=field, thinking OFF | overall **0.5602**, parse_fail 0. technique .638 variance .717 validity 0 field_a .776 field_b .434 field_c .691 field_d .664 | Baseline field. (eval_base == eval_full, stesso numero) | `eval_base.json`, `eval_full.json` | prune_validate.py --k 0 | done |
| A2 | Pruning | Freq(sal) K=25 | 30B, N=152, mask=massa (stepf_base) | overall **0.5771**, pf 1. technique .656 field_b .397 | K25 freq ≈ base (nessun degrado) | `eval_sal25.json` | prune_validate.py --k 25 | done |
| A3 | Pruning | Freq(sal) K=50 | 30B, N=152, mask=massa | overall **0.5733**, pf 0. technique .711 | K50 freq tiene (≥ base) | `eval_sal50.json` | prune_validate.py --k 50 | done |
| A4 | Pruning | Freq(sal) K=70 | 30B, N=152, mask=massa | overall **0.5466**, pf **14** | K70 freq degrada lieve, pf sale | `eval_sal70.json` | prune_validate.py --k 70 | done |
| A5 | Pruning | RANDOM K=25 (ctrl) | 30B, N=152, --random seed 0 | overall **0.5780**, pf 0 | K25 random ≈ freq → a K basso selezione non conta | `eval_rnd25.json` | prune_validate.py --k 25 --random | done |
| A6 | Pruning | RANDOM K=50 (ctrl) | 30B, N=152, --random | overall **0.3733**, pf **90** | **CROLLO.** Random K50 collassa vs freq 0.573 → selezione conta | `eval_rnd50.json` | prune_validate.py --k 50 --random | done |
| A7 | Pruning | RANDOM K=70 (ctrl) | 30B, N=152, --random | overall **0.0**, pf **152** (morte totale) | Random K70 = modello distrutto. Ordine di pruning decisivo | `eval_rnd70.json` | prune_validate.py --k 70 --random | done |
| A8 | Pruning | Mass K=50 (tag prune50) | 30B, k=50, N=152 | overall **0.5714**, pf 0 | Mass-pruning K50 ≥ base. Near-lossless 50% dominio | `eval_prune50.json` | prune_validate.py --k 50 | done |
| A9 | Pruning | REAP K=25 | 30B, maskfile reap_saliency_base, mass_asc | overall **0.5780**, excl_validity 0.6743, pf 0 | REAP K25 sopra base | `eval_prune25_reap.json` | prune_validate.py --k 25 --maskfile reap_saliency_base.json | done |
| A10 | Pruning | REAP K=50 | 30B, maskfile reap_saliency_base | overall **0.5883**, excl_validity 0.6864, pf 0 | **MIGLIOR K50 field**, sopra base. REAP = strada migliore | `eval_prune50_reap.json` | prune_validate.py --k 50 --maskfile reap | done |
| A11 | Pruning | REAP K=50 ordinamento EAN (ablation) | 30B, maskkey experts_by_ean_asc | overall **0.5545**, excl_validity 0.6469 | EAN < mass_asc (0.5883). mass_asc è la maskkey migliore | `eval_prune50_ean.json` | prune_validate.py --k 50 --maskkey experts_by_ean_asc | done |
| A12 | Pruning | REAP K=70 | 30B, maskfile reap_saliency_base | overall **0.5695**, excl_validity 0.6645, pf 0 | REAP K70 ancora sopra base, pf 0. REAP scala meglio | `eval_prune70_reap.json` | prune_validate.py --k 70 --maskfile reap | done |
| A13 | Pruning | PPL dominio K0 | 30B, the held-out domain eval set, n=152, tok=141323 | ppl **5.5589** (nll 1.7154) | Riferimento PPL dominio | `eval_domppl_k0.json` | prune_validate.py --metric perplexity --k 0 | done |
| A14 | Pruning | PPL dominio K25 (mass) | 30B, the held-out domain eval set | ppl **5.8987** (+6%) | Dominio robusto al pruning | `eval_domppl_k25.json` == `eval_mass_dom_k25.json` | prune_validate.py --metric perplexity --k 25 | done |
| A15 | Pruning | PPL dominio K50 (mass) | 30B, the held-out domain eval set | ppl **6.2548** (+13%) | Dominio robusto | `eval_domppl_k50.json` == `eval_mass_dom_k50.json` | prune_validate.py --metric perplexity --k 50 | done |
| A16 | Pruning | PPL dominio K70 (mass) | 30B, the held-out domain eval set | ppl **6.1023** (<K50!) | Dominio quasi insensibile ai cold. K70<K50 | `eval_domppl_k70.json` == `eval_mass_dom_k70.json` | prune_validate.py --metric perplexity --k 70 | done |
| A17 | Pruning | PPL dominio K50 RANDOM (ctrl) | 30B, the held-out domain eval set, --random | ppl **13.9676** (2.2x mass) | Saliency-mass cruciale sul dominio | `eval_domppl_k50_rand.json` | prune_validate.py --metric perplexity --k 50 --random | done |
| A18 | Pruning | PPL generale K0 | 30B, general_it, n=200, tok=152625 | ppl **5.7239** | Riferimento PPL generale | `eval_genppl_k0.json` | prune_validate.py --metric perplexity --prompts general_it | done |
| A19 | Pruning | PPL generale K25 (mass) | 30B, general_it | ppl **6.6429** (+16%) | Generale più sensibile del dominio | `eval_genppl_k25.json` == `eval_mass_gen_k25.json` | idem --k 25 | done |
| A20 | Pruning | PPL generale K50 (mass) | 30B, general_it | ppl **12.226** (2.14x) | Pruning dom-guidato danneggia molto il generale | `eval_genppl_k50.json` == `eval_mass_gen_k50.json` | idem --k 50 | done |
| A21 | Pruning | PPL generale K70 (mass) | 30B, general_it | ppl **20.387** (3.56x) | Divergenza dom-vs-gen massima a K alti | `eval_genppl_k70.json` == `eval_mass_gen_k70.json` | idem --k 70 | done |
| A22 | Pruning | PPL generale K50 RANDOM (ctrl) | 30B, general_it, --random | ppl **57.752** (4.7x mass) | Random devasta il generale | `eval_genppl_k50_rand.json` | idem --k 50 --random | done |
| A23 | Pruning | REAP-saliency PPL dominio K25/50/70 | 30B, the held-out domain eval set, maskfile reap_saliency_base | K25 **5.5999** / K50 **5.5004** / K70 **5.8592** | REAP dom K50 **sotto** base 5.559. Tenuta straordinaria | `eval_sal_dom_k25/50/70.json` | prune_validate.py --metric perplexity --maskfile reap | done |
| A24 | Pruning | REAP-saliency PPL generale K25/50/70 | 30B, general_it, maskfile reap | K25 **6.6004** / K50 **9.3556** / K70 **18.317** | REAP gen K50 9.36 < mass 12.23. REAP degrada meno il generale | `eval_sal_gen_k25/50/70.json` | idem general_it | done |
| A25 | Pruning | Sub-expert per-neurone BASE | 30B, I=768, slots=5842, soglie .8/.9/.95 | committed .8→.666 .9→.808 .95→.891; n_eff **679.5/768** (0.885) | Neuroni diffusi → sub-expert poco promettente | `subexpert_base.json` | subexpert.py (NON in repo scripts_pod; monkeypatch Qwen3MoeExperts) | done |
| A26 | Pruning | Sub-expert per-neurone FT | qwen3-domain FT, I=768, slots=5915 | committed .95→.894; n_eff **683.3/768** (0.890) | FT NON compatta i neuroni (683 vs 679). Sub-expert inutile | `subexpert_ft.json` | subexpert.py | done |

### B) MIXED-PRECISION (cold experts int4/int2/dropped)

| # | Cat | Esperimento | Setup | Risultato | Verdetto | File dati | Script | Stato |
|---|-----|-------------|-------|-----------|----------|-----------|--------|-------|
| B1 | Mixed-prec | K=50 full/int4/int2/dropped | 30B, cold bottom-50% massa, group=128, ppl dom/gen | full 17.717/6.017; int4 **17.718/6.017** (lossless); int2 17.641/**8.524**; dropped 18.745/**13.998** | int4 GRATIS; int2 ok solo dom; MAI droppare | `mixedprec_k50.json` | mixed_precision_eval.py --k 50 | done |
| B2 | Mixed-prec | K=70 full/int4/int2/dropped | 30B, cold bottom-70% | full 17.717/6.017; int4 **17.77/6.047** (~lossless); int2 20.279/**11.824**; dropped **27.234/32.607** | int4 quasi gratis anche K70; dropped devasta | `mixedprec_k70.json` | mixed_precision_eval.py --k 70 | done |

### C) 235B (Qwen3-235B-A22B-GPTQ-Int4)

| # | Cat | Esperimento | Setup | Risultato | Verdetto | File dati | Script | Stato |
|---|-----|-------------|-------|-----------|----------|-----------|--------|-------|
| C1 | 235B | STEP F routing (stepf_base235 / v2) | 235B, E=128, topk=8, 94 layer | n_eff **49.918/128**; cf .8→.290 .9→.417 .95→.524 | Routing ≈ 30B (49.92 vs 49.91). Scala a profondità costante | `stepf_base235.json`, `stepf_base235v2.json` (aggregati identici, differiscono per-layer 2 byte) | step_f.py --gptqmodel | done |
| C2 | 235B | DEBUG gate hook | 235B, gate=Qwen3MoeTopKRouter | GATE OUT = tuple len 3: logits[10,128] fp16, weights[10,8] fp16, indices[10,8] int64. num_experts 128, per_tok 8, 94 layer | Hook VALIDO → anomalia cold50>rand50 è FINDING reale, non bug | `DEBUG_gate.txt` | debug_gate.py --gptqmodel | done |
| C3 | 235B | PPL dominio K0 (full) | 235B, the held-out domain eval set, tok=141323 | ppl **4.3391** (nll 1.4677) | 235B > 30B sul dominio (4.34 vs 5.56) | `eval_full235.json` == `eval_p235_full.json` | prune_validate.py --gptqmodel --metric perplexity --k 0 | done |
| C4 | 235B | PPL generale K0 (genfull) | 235B, general_it, tok=152625 | ppl **4.4572** | 235B > 30B anche sul generale (4.46 vs 5.72) | `eval_p235_genfull.json` | idem --prompts general_it | done |
| C5 | 235B | PPL dominio K25 cold (mass) | 235B, maskfile stepf_base235(v2) | ppl **4.2403** (< full!) | Cold K25 gratis sul dominio (meglio del full) | `eval_p235_cold25.json` == `eval_prune235_25.json` | prune_validate.py --gptqmodel --k 25 | done |
| C6 | 235B | PPL dominio K50 cold (mass) | 235B | ppl **18.735** (4.32x) | A K50 il 235B degrada MOLTO più del 30B. Riprodotto identico v1/v2 | `eval_p235_cold50.json` == `eval_prune235_50.json` | prune_validate.py --gptqmodel --k 50 | done |
| C7 | 235B | PPL dominio K70 cold (mass) | 235B | ppl **29.580** (6.82x) | Degrado severo ma monotono da K50 | `eval_prune235_70.json` | prune_validate.py --gptqmodel --k 70 | done |
| C8 | 235B | PPL dominio K50 RANDOM (ctrl) — **ANOMALIA** | 235B, --random | ppl **9.205** (2.12x) | **PARADOSSO: random (9.2) < cold-mass (18.7).** La massa fallisce a scala → motiva REAP. Confound: int4-only | `eval_p235_rand50.json` == `eval_prune235_50_rand.json` | prune_validate.py --gptqmodel --k 50 --random | done |
| C9 | 235B | PPL generale K50 cold (gen50) | 235B, general_it | ppl **97.572** (21.9x base 4.457) | Pruning dom-guidato distrugge il generale a scala (21.9x vs 30B 2.14x) | `eval_p235_gen50.json` | idem general_it --k 50 | done |
| C10 | 235B | Baseline generativo full235 (field) | 235B, metric=field, torch.compile | **MAI COMPLETATO.** Compile-storm (32 inductor worker, 60+ min, 0 output). eval_full235 field perso col teardown | FALLITO/appeso. Usare perplexity forward-only (fatto in C3), non generate | — (perso) | prune_validate.py --gptqmodel --metric field (NO: usare perplexity) | fallito |

### D) FINE-TUNING (STEP E qualità, STEP F routing, cross-mask retention)

| # | Cat | Esperimento | Setup | Risultato | Verdetto | File dati | Script | Stato |
|---|-----|-------------|-------|-----------|----------|-----------|--------|-------|
| D1 | FT | FT baseline QLoRA r=32 + router full-train | 30B, aux OFF, 3 ep, lr 1e-4, bs 16, cutoff 4096, seed 42, the domain train set 1299 | train_loss **0.418**; eval token-acc **94.5%**; adapter 4.00GB | FT riuscito (baseline studio). In-distribution (KB curata) | `qwen3-domain-lora/` | run_ft.sh / fix_and_train.sh | done |
| D2 | FT | Merge manuale version-independent | LoRA fuso→per-expert bmm, SCALE=2.0, I=768, router full replace | ‖d‖/‖W‖ = 0.0087/0.0093 (layer0 e0). 16 shard bf16 ~57GB, integro | Merged FT (step D) per step_f/prune_validate | `qwen3-domain-merged/` | manual_merge.py (--check dry-run) | done |
| D3 | FT | STEP E qualità per-campo base-vs-FT | 30B vs FT, greedy, thinking OFF, N=152 (excl validity) | media 6 campi **65.3%→79.4% (+14pt)**. field_b .434→.717 (+28); stake .664→.875 (+21); technique .638→.711; variance .717→.822; field_a .776→.836; field_c .691→.803 | FT vince su TUTTI i 6 campi. In-distribution | `eval_base.json`, `eval_ft.json` | eval_quality.py / prune_validate.py --k 0 | done |
| D4 | FT | STEP F concentrazione routing FT-vs-base | 30B base vs FT, hook mlp.gate, 152 prompt, ~130k tok | n_eff base **49.91** → FT **57.12** (nota: CONSOLIDATION dice 56.9). cf@.95 .506→.572 | **IPOTESI "FT concentra" FALSIFICATA:** FT DISPERDE. È il dominio a concentrare. Prune sul BASE, non sul FT | `stepf_base.json`, `stepf_ft.json` | step_f.py --tag base/ft | done |
| D5 | FT | STEP F copertura-massa (estrazione) | da stepf_base/ft.json, E=128 | BASE: 95% massa→tieni 65, droppi 63 (49%). FT: 95%→tieni 73, droppi 55 (43%) | FT ~6pt MENO prunabile del base (proxy copertura, non qualità) | idem D4 | inline python | done |
| D6 | FT | FT+crossmask (mask BASE) K25 | qwen3-domain-merged, maskfile stepf_base | overall **0.6776**, excl 0.7906, pf 0 | Applicare mask BASE al FT funziona (≈ FT full) | `eval_ftK25_crossmask.json` | prune_validate.py --model merged --k 25 --maskfile stepf_base | done |
| D7 | FT | FT+ftmask (mask FT) K25 | merged, maskfile stepf_ft | overall **0.6748**, excl 0.7873, pf 0 | ftmask NON aiuta vs crossmask a K25 | `eval_ftK25_ftmask.json` | prune_validate.py --maskfile stepf_ft | done |
| D8 | FT | FT+RANDOM K25 (ctrl) | merged, --random | overall **0.6382**, excl 0.7445, pf 0 | A K25 il FT assorbe il random | `eval_ftK25_rand.json` | prune_validate.py --k 25 --random | done |
| D9 | FT | FT+crossmask K40 | merged, stepf_base | overall **0.6711**, pf 0 | Crossmask K40 ≈ K25 | `eval_ftK40_crossmask.json` | idem --k 40 | done |
| D10 | FT | FT+ftmask K40 | merged, stepf_ft | overall **0.6526**, pf 6 | ftmask < crossmask a K40 | `eval_ftK40_ftmask.json` | idem | done |
| D11 | FT | FT+RANDOM K40 (ctrl) | merged, --random | overall **0.3838**, pf 18 | Gap mask vs random si apre (0.67 vs 0.38) | `eval_ftK40_rand.json` | idem | done |
| D12 | FT | FT+crossmask K50 | merged, stepf_base | overall **0.6571**, excl 0.7667, pf 7 | Crossmask degrada dolcemente | `eval_ftK50_crossmask.json` | idem --k 50 | done |
| D13 | FT | FT+ftmask K50 | merged, stepf_ft | overall **0.6632**, excl 0.7737, pf 1 | **Prima volta ftmask > crossmask** a K50 | `eval_ftK50_ftmask.json` | idem | done |
| D14 | FT | FT+RANDOM K50 (ctrl) | merged, --random | overall **0.3563**, pf 71 | Random inutilizzabile da K50 anche col FT | `eval_ftK50_rand.json` | idem | done |
| D15 | FT | FT+crossmask K60 | merged, stepf_base | overall **0.6564**, pf 4 | Crossmask robustissima fino K60 | `eval_ftK60_crossmask.json` | idem --k 60 | done |
| D16 | FT | FT+ftmask K60 | merged, stepf_ft | overall **0.6286**, pf 2 | Vantaggio ftmask di K50 non regge a K60 | `eval_ftK60_ftmask.json` | idem | done |
| D17 | FT | FT+crossmask K70 | merged, stepf_base | overall **0.6438**, excl 0.7511, pf 2 | **Ancora sopra base 30B (0.5601).** Pruning aggressivo praticabile col FT | `eval_ftK70_crossmask.json` | idem --k 70 | done |
| D18 | FT | FT+ftmask K70 | merged, stepf_ft | overall **0.5949**, excl 0.6941, pf 0 | ftmask < crossmask a K70. Mask BASE più robusta a K alti | `eval_ftK70_ftmask.json` | idem | done |
| D19 | FT | FT+RANDOM K70 (ctrl) | merged, --random | overall **0.0**, pf 152 | Morte totale come rnd70. FT non salva dal random a K70 | `eval_ftK70_rand.json` | idem | done |
| D20 | FT | FT retention K10 | merged, ftmask/crossmask/rand | **NON RACCOLTO** (pod terminato prima / file assenti) | Tabella retention FT incompleta a K10 | atteso `eval_ftK10_*.json` (non confermato) | prune_validate.py --k 10 | todo |

### E) SPEX GATE (predizione next-expert L→L+1: Markov vs hidden vs static/random)

| # | Cat | Esperimento | Setup | Risultato | Verdetto | File dati | Script | Stato |
|---|-----|-------------|-------|-----------|----------|-----------|--------|-------|
| E1 | SPEX-gate | Markov gate OLMoE general (2/3-pred) | OLMoE-1B-7B, 64 exp, 16 layer, wiki-en 300, T=99152 | @25% (16/64): RANDOM .250 STATIC .426 MARKOV **.756** (conc +.176, xlayer +.329). @50%: .500/.698/.914 | Su OLMoE dominato dal CROSS-LAYER. **Non transferibile** (spazio 64 gonfia recall) — solo validazione pipeline | `spex/traces_olmoe_general.npz` | dump_traces.py + markov_gate.py | done |
| E2 | SPEX-gate | Markov gate OLMoE domain | OLMoE, domain eval set, T=75095 | @25%: RND .250 STAT .647 MK **.774** (conc +.398, xlayer +.127). @50%: .500/.864/.935 | Su dominio domina la CONCENTRAZIONE. Non transferibile | `spex/traces_olmoe_domain.npz` | idem | done |
| E3 | SPEX-gate | Gate 30B general (128 exp, vettorizzato) | Qwen3-30B, T=95841, 48 layer, split 50/50 in-dist | @25% (32/128): RANDOM .250 STATIC .693 MARKOV **.795** (conc **+.443**, xlayer M-S **+.102**) | **Ribaltato vs OLMoE:** domina concentrazione, cross-layer solo +.10. Indebolisce Markov-only in-dist | `spex/traces_q30_general.npz` | dump_traces.py + markov_gate.py (vettorizzato) | done |
| E4 | SPEX-gate | Gate 30B domain (128 exp) | Qwen3-30B, T=74612, split 50/50 | @25%: RND .250 STAT .744 MK **.875** (conc +.494, xlayer +.132) | Dominio più predicibile (routing concentrato). Cache statica fa quasi tutto in-dist | `spex/traces_q30_domain.npz` | idem | done |
| E5 | SPEX-gate | Gate CROSS-DISTRIBUTION 30B | fit su un corpus, eval sull'altro (no leakage), 4 combo | @25% in-dist: STATIC ~.75. **cross: STATIC crolla ~.44** (-40%). MARKOV mantiene M-S ~+.13 ovunque | **STATIC non trasferisce** cross-distribution → argomento pro-SPEX sotto workload shift | traces_q30_general/domain.npz | markov_gate_cross.py FIT.npz EVAL.npz | done |
| E6 | SPEX-gate | Gate 235B general (128 exp) | Qwen3-235B, T=55258, 94 layer | @25% (32/128): RANDOM .250 STATIC .716 MARKOV **.798** (xlayer M-S **+.082**) | Cross-layer aggiunge ancora meno del 30B. Coerente cross-scala. **Solo general (domain trace persa)** | `spex/traces_q235_general.npz` | dump_traces.py --gptqmodel + markov_gate.py | done |
| E7 | SPEX-gate | REAP+SPEX two-tier combo 30B | 30B, B=32, 16 REAP-static + 16 markov-dyn, in-dist+cross | dom→dom: POP .746 REAP .502 MARKOV .877 HYBRID .821. HYBRID naive PEGGIORA vs Markov puro quasi ovunque | (1) ranking REAP = pessima cache residente. (2) HYBRID naive backfira → tenere REAP/SPEX separati | traces_q30_*.npz + reap_saliency_base.json | reap_spex_combo.py FIT EVAL reap.json | done |
| E8 | SPEX-gate/hidden | hidden vs Markov recall @8/16/32 DOMAIN | Qwen3-30B, T=38400, E=128, k=8, 48 layer, domain eval set | hidden **.9316/.9906/.9978**; markov .5142/.7210/.8902 | hidden stracca markov. @25% 99.8% dominio. **NB: ~0.98 = prior-art Fate (2502.12224)** | `hidden_predict_q30_dom.json` | hidden_predict.py --tag q30_dom | done |
| E9 | SPEX-gate/hidden | hidden vs Markov recall @8/16/32 GENERAL | Qwen3-30B, T=37536, general_it | hidden **.8292/.9560/.9861**; markov .4842/.6888/.8647 | hidden alto anche fuori dominio (@25% 98.6%). In-dist only; cross-dist non misurato | `hidden_predict_q30_gen.json` | hidden_predict.py --tag q30_gen | done |
| E10 | SPEX-gate | reap_saliency_base (ranking REAP Eq.9) | 30B, E=128, 48 layer, reap_eq9_conditional_mean | n_expert_cnt0 = **302/6144** attivazione ~0. Chiavi: experts_by_mass_asc, experts_by_ean_asc, reap | 302 expert coldi → margine pruning. Input di tutte le mask REAP | `reap_saliency_base.json` (per-layer omessi nel bundle) | reap_saliency.py --tag base | partial |

### F) SPEX LOOP (predizione+purging continuo: reactive/markov/adaptive/random miss-rate + temperatura)

| # | Cat | Esperimento | Setup | Risultato | Verdetto | File dati | Script | Stato |
|---|-----|-------------|-------|-----------|----------|-----------|--------|-------|
| F1 | SPEX-loop | Loop naive: reactive-LRU vs markov-prefetch | traces q30/q235 general, C=8..48, cache LRU, markov C-prefetch | 30B gen C8: REACT .498 vs MARKOV **.661** (-33%). 235B gen C8: .480/.698 (-45%). Markov perde quasi ovunque | **NEGATIVE RESULT (parte della novelty):** markov naive < reactive. Routing concentrato → LRU cattura il reuse | traces_q30_general/q235_general.npz | continuous_cache_sim_ORIG.py (v1) | done |
| F2 | SPEX-loop | Loop adaptive (verifier+temperatura EMA) | +policy adaptive: conf EMA α=0.1, n_pred=round(C*conf) | @25% (C32) 30B gen: ADAPTIVE **.117** vs REACT .154 (-24%). conf auto .10→.88. 235B gen adaptive vince ogni taglia | ADAPTIVE mai peggio del reactive; a conf alta -25% miss. Auto-calibrazione | traces_q30/q235 | continuous_cache_sim_ORIG.py (v2) | done |
| F3 | SPEX-loop | Loop + RANDOM control + decomposizione | +policy random (eviction casuale seed 0) | @25% (C32): 30B gen RND .209/REACT .154/ADAPT **.117** (adapt vs rnd **-44%**); 235B gen .197/.142/.123 (-38%); 30B dom .268/.203/**.132** (**-51%**) | Temperatura sposta -38/-51% miss vs random a pari VRAM. Metà = cache caldi, metà = predizione | traces q30 gen/dom, q235 gen | continuous_cache_sim_ORIG.py (v3) | done |
| F4 | SPEX-loop | Loop DSpark-fedele (spex_loop.py) | split per-doc 60/20/20, Markov ExE pieno, conf head logistica, **STS vera**, ammissione Alg.1, eviction pred-aware, ≥3 seed | **SCRITTO, NON GIRATO end-to-end.** Loop attuale = markov-only + STS-on-hit-labels in locale | STS + confidence head + Alg.1 corretti vs ORIG. +hidden si innesta come feature | opz json --out | src\msc\spex\spex_loop.py | todo (implementato, non eseguito con numeri) |

### G) PRIOR-ART / NOVELTY (F1-F6, F-ctrl, ispirazioni)

| # | Cat | Finding | Numeri chiave | Verdetto novelty | Prior art | Fonte |
|---|-----|---------|---------------|------------------|-----------|-------|
| G1 | Prior-art | **F1** ~50% expert prunabili near-lossless dominio, informato>>random | 30B dom K50 saliency 5.50 vs full 5.56; EASY-EP informato 45.22% vs random 3.26% | **GIÀ-NOTO** (replica) | EASY-EP 2504.06792, PreMoE 2505.17639, REAP 2510.13999, NAEE 2402.14800 | CONSOLIDATION.md, PRIOR_ART.md |
| G2 | Prior-art | **F2** FT-router sbloccato DISPERDE; è il dominio a concentrare; FT meno prunabile | n_eff 49.9→56.9/57.1; retention FT K25 .675/K50 .663/K70 .595; crossmask≥ftmask | **NOVELTY-CANDIDATO #1** | contro-caso ReMoE 2412.14711 (segno opposto); ESFT 2407.01906, Guo 2505.22323, ST-MoE 2202.08906, Mixtral 2401.04088 | idem |
| G3 | Prior-art | **F3** pruning dominio devasta il generale (cresce con la scala) | 30B dom K50 1.13x/gen 2.14x/K70 3.56x; 235B dom 4.32x/gen 21.9x | **GIÀ-NOTO** (quantifica) | EASY-EP Table4, PreMoE, Less-is-MoE 2606.05538 | idem |
| G4 | Prior-art | **F4** predizione cross-layer + loop adattivo confidence-scheduled | static recall @25% hidden .986/markov .865/rnd .25; loop miss @25% rnd .209/react .154/adapt .117 | **NOVELTY-CANDIDATO #2** (transfer+loop; robustezza) | Fate 2502.12224 (~99% hit), Pre-gated 2308.12066, SiDA 2310.18859, ExpertFlow 2410.17954, Mixtral-offloading 2312.17238 | idem |
| G5 | Prior-art | **F5** offload predittivo 235B full-quality su ~12GB | working set ~22B + hot-set VRAM; RAM 120→33-60GB, sweet spot ~48GB | MECCANISMI NOTI, regime estremo | ProMoE 2410.22134, MoE-Infinity 2401.14361, PowerInfer-2 2406.06282, HOBBIT 2411.01433 | idem |
| G6 | Prior-art | **F6** mixed-precision cold experts (hot int4/cold int2) | int4 lossless; int2 gen 1.42x/1.97x vs dropped 2.33x/5.42x | **GIÀ-NOTO** (conferma) | MC-MoE 2410.06270, MxMoE 2505.05799, MoQE 2310.02410, MoPEQ 2509.02512 | idem |
| G7 | Prior-art | **F-ctrl** MASSA < RANDOM a scala 235B | 235B K50 massa 18.7 (4.32x) > random 9.2 (2.12x) | **NOVELTY-CANDIDATO** (esposto dal random control; confound int4-only/235B-only) | How-to-Score-Experts 2606.15716, DeepSeekMoE 2401.06066, REAP | idem |
| G8 | Prior-art | **DwarfStar** (antirez/ds4) — ispirazione, leva non tirata | cache reattiva SSD, imatrix statica, 2-bit uniforme, MTP solo per token | Ispirazione diretta. Leva non tirata = prefetch predittivo expert in decode | github.com/antirez/ds4 | CONSOLIDATION.md §3 |
| G9 | Prior-art | **DSpark** (=DeepSpec?) — macchina di calibrazione | confidence head σ(w·[h;MarkovEmb]) Eq.7; STS (min ECE su prodotto cumulato); scheduler hw-aware Alg.1. Fig.6: ECE 3-8%→~1%, ROC-AUC .81-.90 | Ispirazione diretta. Transfer da TOKEN a EXPERT. **DSpark==DeepSpec da confermare** | github.com/deepseek-ai/DeepSpec | CONSOLIDATION.md §3, DSpark_clean.txt |

---

## 2. NUMERI CHIAVE (tabelle riprodotte per esteso)

### 2.1 Pruning field-accuracy 30B (N=152, overall) — la tabella-cardine

| K | Freq(sal) | RANDOM | REAP | Mass(prune) |
|---|-----------|--------|------|-------------|
| 0 (full) | 0.5602 (pf 0) | — | — | — |
| 25 | 0.5771 (pf 1) | 0.5780 (pf 0) | 0.5780 (excl 0.674) | — |
| 50 | 0.5733 (pf 0) | **0.3733 (pf 90)** | **0.5883 (excl 0.686)** | 0.5714 |
| 70 | 0.5466 (pf 14) | **0.0000 (pf 152)** | 0.5695 (excl 0.664) | — |
| 50-EAN | — | — | 0.5545 (excl 0.647) | — |

Divergenza monotona pulita: a K50/K70 random crolla/muore, freq tiene, REAP è il migliore. **La selezione conta.**

### 2.2 Perplexity dominio-vs-generale 30B (K0/25/50/70, mass vs saliency vs random)

| Config | dom K0 | dom K25 | dom K50 | dom K70 | gen K0 | gen K25 | gen K50 | gen K70 |
|--------|--------|---------|---------|---------|--------|---------|---------|---------|
| **mass** | 5.559 | 5.899 | 6.255 | 6.102 | 5.724 | 6.643 | 12.226 | 20.387 |
| **REAP-saliency** | — | 5.600 | **5.500** | 5.859 | — | 6.600 | **9.356** | 18.317 |
| **RANDOM K50** | — | — | 13.968 | — | — | — | 57.752 | — |

REAP dom K50 (5.500) **sotto** il full base (5.559). REAP gen K50 (9.356) < mass (12.226). Random devasta entrambi.

### 2.3 Perplexity 235B (GPTQ-int4, dominio + generale)

| Config | dom full | dom K25 | dom K50 | dom K70 | dom K50-rand | gen full | gen K50 |
|--------|----------|---------|---------|---------|--------------|----------|---------|
| ppl | 4.339 | 4.240 | **18.735** | 29.580 | **9.205** | 4.457 | 97.572 |
| (xbase) | 1.00 | 0.98 | 4.32x | 6.82x | 2.12x | 1.00 | 21.9x |

**Anomalia F-ctrl:** dom K50 massa (18.735) **peggio** del random (9.205). dom-vs-gen a scala: 4.32x vs 21.9x.

### 2.4 STEP E qualità per-campo base-vs-FT (30B, N=152, excl validity)

| Campo | base | FT | Δ |
|-------|------|----|----|
| technique | 63.8% | 71.1% | +7.3 |
| variance | 71.7% | 82.2% | +10.5 |
| field_a | 77.6% | 83.6% | +6.0 |
| field_b | 43.4% | 71.7% | **+28.3** |
| field_c | 69.1% | 80.3% | +11.2 |
| field_d | 66.4% | 87.5% | **+21.1** |
| **media 6** | **65.3%** | **79.4%** | **+14.0** |
| validity | 0.0% | 0.0% | (escludere) |

### 2.5 Mixed-precision cold experts (30B, ppl dom/gen)

| Config | K50 dom | K50 gen | K70 dom | K70 gen |
|--------|---------|---------|---------|---------|
| full | 17.717 | 6.017 | 17.717 | 6.017 |
| cold int4 | 17.718 | 6.017 | 17.77 | 6.047 |
| cold int2 | 17.641 | 8.524 | 20.279 | 11.824 |
| cold dropped | 18.745 | 13.998 | 27.234 | 32.607 |

int4 = gratis. int2 = ok dom, degrada gen. dropped = devasta. **Comprimere >> droppare.**

### 2.6 SPEX loop miss-rate (@25% residency = C32) — random/reactive/adaptive

| Corpus | RANDOM | REACTIVE-LRU | ADAPTIVE | Δ adapt vs rnd |
|--------|--------|--------------|----------|----------------|
| 30B general | 0.209 | 0.154 | **0.117** | -44% |
| 235B general | 0.197 | 0.142 | 0.123 | -38% |
| 30B domain | 0.268 | 0.203 | **0.132** | **-51%** |

Negative result markov-naive (C8, 30B gen): REACTIVE 0.498 vs MARKOV 0.661 (-33%).

### 2.7 Gate hidden vs markov recall @budget (30B, next-expert)

| Budget | DOM hidden | DOM markov | GEN hidden | GEN markov |
|--------|-----------|-----------|-----------|-----------|
| 8 (6%) | 0.9316 | 0.5142 | 0.8292 | 0.4842 |
| 16 (12%) | 0.9906 | 0.7210 | 0.9560 | 0.6888 |
| 32 (25%) | 0.9978 | 0.8902 | 0.9861 | 0.8647 |

hidden @25% ≈ 98.6-99.8% (prior-art Fate ~99%). markov ≈ 86-89%.

### 2.8 Routing concentration (STEP F, n_eff/128)

| Modello | n_eff | cf@0.8 | cf@0.9 | cf@0.95 |
|---------|-------|--------|--------|---------|
| 30B base | 49.91 | 0.286 | 0.406 | 0.506 |
| 30B FT | 57.12 | 0.326 | 0.461 | 0.572 |
| 235B base | 49.92 | 0.290 | 0.417 | 0.524 |

30B ≈ 235B (routing scala a profondità costante). FT DISPERDE (n_eff sale).

---

## 3. GIÀ FATTO — NON RIFARE (lista secca)

**Pruning 30B (field):** baseline K0; freq(sal) K25/50/70; random K25/50/70; mass K50; REAP K25/50/70; REAP-EAN K50 (ablation maskkey). — `eval_full/base/sal*/rnd*/prune50/prune*_reap/prune50_ean.json`

**Pruning 30B (perplexity):** dom mass K0/25/50/70 + rand K50; gen mass K0/25/50/70 + rand K50; REAP-saliency dom+gen K25/50/70. — `eval_domppl_*/genppl_*/mass_*/sal_dom_*/sal_gen_*.json`

**Sub-expert per-neurone:** base + FT (n_eff/I ~0.885/0.890). — `subexpert_base/ft.json`

**Mixed-precision:** K50 e K70 full/int4/int2/dropped. — `mixedprec_k50/k70.json`

**235B:** STEP F routing (base235 + v2); DEBUG gate; PPL dom full/K25/K50/K70/rand50; PPL gen full/K50. — `stepf_base235*/DEBUG_gate.txt/eval_p235_*/eval_prune235_*.json` (full235 generativo field NON rifattibile: compile-storm)

**Fine-tuning:** FT QLoRA baseline (loss 0.418, tok-acc 94.5%); merge manuale; STEP E per-campo (+14pt); STEP F dispersione (49.9→57.1); retention K25/40/50/60/70 × ftmask/crossmask/random. — `qwen3-domain-lora/merged/`, `eval_base/ft.json`, `stepf_base/ft.json`, `eval_ftK*.json`

**SPEX gate:** Markov OLMoE gen/dom; gate 30B gen/dom (128 exp); cross-distribution 30B; gate 235B gen; REAP+SPEX two-tier combo; hidden vs markov recall @8/16/32 dom+gen; reap_saliency ranking. — `traces_*.npz`, `hidden_predict_q30_*.json`, `reap_saliency_base.json`

**SPEX loop (euristico ORIG):** 3 iterazioni (naive → +temperatura/EMA → +random control) su q30 gen/dom + q235 gen. — `continuous_cache_sim_ORIG.py`

**Prior-art:** mappa F1-F6 + F-ctrl, 28 paper arXiv verificati, DwarfStar/DSpark. — `docs/PRIOR_ART.md`, `docs/CONSOLIDATION.md`, `references.bib`

**Infra/ricetta ambiente (bloccata):** immagine `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` + `pip --break-system-packages transformers accelerate gptqmodel safetensors` → torch 2.8.0+cu128, transformers 5.12.1, gptqmodel 7.1.0, Qwen3MoeExperts present. 235B carica via `--gptqmodel`.

---

## 4. NON ANCORA FATTO (TODO reali)

1. **+hidden nel loop (end-to-end).** `spex_loop.py` gira markov-only + STS-on-hit-labels. I dump `hidden_predict_q30_*` sono SCALARI AGGREGATI, non per-token → non innestabili. **Serve pod re-dump per-token** (top-N IDs+scores, ~0.5GB/trace) via `spex_dump_hidden.py`. Il claim "~10x meno fetch con hidden" è STIMATO, mai girato in-loop.
2. **Trace 235B domain (`traces_q235_domain.npz`) MANCANTE** — B200 terminato. Il loop/gate 235B gira solo su general. Da re-provisionare per dom-vs-gen a scala nel loop.
3. **Loop DSpark-fedele con numeri.** `spex_loop.py` è scritto (STS vera + confidence head + Alg.1 + split per-doc + multi-seed) ma **non eseguito con output numerici**. Da girare e produrre tabella miss-rate + ECE.
4. **STS raffinata + reliability plot / ECE.** Riprodurre Fig.6 DSpark (ECE raw 3-8% → ~1% dopo STS, ROC-AUC preservato). Oggi il loop ORIG usa EMA crudo, non STS calibrata.
5. **Multi-seed / document-level su q30.** Loop ORIG = split per-token (leakage intra-doc) + single seed. Rifare con ≥3 seed e split per-documento (già previsto in spex_loop.py, da eseguire).
6. ~~npz senza doclens~~ **CORRETTO (2026-07-03): le trace HANNO `doclens`** (verificato: chiavi `['experts','doclens','n_experts','topk','n_layers']` in tutte le `traces_*.npz`). L'errore era ereditato dalla sintesi V2 (che deduceva dal sim ORIG che non li LEGGE). `spex_loop.py` li usa → **reset per-doc e split per-documento SONO possibili in locale.** Nessun re-dump necessario per questo.
7. **235B massa<random in bf16 (F-ctrl confound).** L'anomalia cold50>rand50 è **int4-only + 235B-only**. Su 30B bf16 NON succede. Replicare su 235B **bf16** per togliere il confound di precisione prima di rivendicare.
8. **FT retention K10 e K60-completo.** K10 mai raccolto (pod terminato); K60 raccolto ma rand non riportato in tutti i peek.
9. **spex_speed_sim su 3060 (tok/s reali).** Solo stime analitiche (~1-7 tok/s bandwidth-bound, expert int4 ~9.5-10MB). Nessun tok/s misurato su hardware. Tenere miss-rate come risultato primario hardware-independent.
10. **REAP-then-SPEX sequenziale.** Comporre correttamente = dumpare trace SUL modello prunato-64, non tier-mixing naive (che backfira, vedi E7). Solo proposto.
11. **Cross-distribution robustezza per hidden.** hidden misurato solo in-distribution. Il vero delta-SPEX (static-collassa / condizionale-tiene) è misurato per Markov, non per hidden.
12. **Bootstrap CI sui delta.** N=152 piccolo, greedy deterministica. Nessun intervallo di confidenza sui delta di accuracy/ppl.
13. **DSpark==DeepSpec** — identificazione inferita dall'assistant, mai confermata dall'utente. Da verificare prima di citare nel paper.

---

---

## H) BATCH 2 (2026-07-03) — SPEX loop girato + accuracy-drop + sub-expert Qwen&DeepSeek

Codice loop: `src/msc/spex/spex_loop.py` (DSpark-fedele, vettorizzato). Risultati: `spex/loops\*.json`.
Trace hidden per-token: `spex/hidden_scores_q30_{dom,gen,code,reasoning,english,mmlu}.npz` (dump A100).
Trace 235B per-workload: `spex/traces_q235_{domain,general,code,reasoning,english,mmlu}.npz` (dump B200).

| # | Cat | Esperimento | Setup | Risultato (miss-rate ADPT-DSpark) | Verdetto | Stato |
|---|-----|-------------|-------|-----------------------------------|----------|-------|
| H1 | SPEX-loop | Loop DSpark-fedele 30B markov, 6 workload | spex_loop.py, split per-doc, STS vera, 2 seed | miss@C8(6%): dom .524 gen .467 code .555 reasoning .570 english .528 mmlu .501 | Markov ≈50% miss a 6GB | done |
| H2 | SPEX-loop | Loop 30B **HIDDEN**, 6 workload | idem --predictor hidden (probe da hidden_scores) | miss@C8(6%): **dom .098** gen .175 code .207 reasoning .195 english .168 mmlu .168 | **+hidden DIMEZZA/TERZA il miss.** ECE 0.04->0.008 (STS) | done |
| H3 | SPEX-loop | Loop **235B** markov, 6 workload | traces_q235_*, L=94 | miss@C8(7.1GB): gen .464 dom .518 mmlu .510 english .534 code .567 reasoning .584 | Scala-invariante (≈30B). +hidden estrapolato 10-24% | done |
| H4 | Accuracy-drop | **double-loop-markov**: ppl droppando i non-predetti | 30B, FULL ppl 14.79, calib=general eval=domain, drop+renorm nel gate | ×full @C8: **oracle 1.00** / markov **59** / static 168 / random 3260. @C48: 1.00/2.1/7.8/52 | **DEVI FETCHARE, non droppare (con markov).** Oracolo lossless=meccanismo sano; markov manca gli high-impact. static<markov, dynamic aiuta ma solo oracle regge | done |
| H5 | Accuracy-drop | **hidden-drop** (ultima porta) | 30B, metodo hidden = probe lineare per-layer h_{L-1}·W (Fate-style) nel forward, ppl vs C, FULL 14.79 | ×full: C8 **8.7×** (vs markov 59×), C12 **2.09×**, C16 1.78×, C32 1.33×, C48 1.32× (oracle 1.00) | **Il drop APRE parzialmente col +hidden:** da ≥12% VRAM ~1.3-1.8× ppl (degradato ma usabile) dove markov faceva 16-32×; ma **NON lossless** (plateau 1.3× anche a 38%). Sotto 6% crolla (8.7×). Sweet-spot = **IBRIDO** drop-basso-impatto + fetch-alto-impatto | done |
| H8 | SPEX-loop | Loop **DeepSeek-V2-Lite** markov, domain+general | traces_ds2lite_* (caricamento NATIVO transformers 5.x, 26 layer MoE, 64 routed+2 shared, 6 attivi), spex_loop.py | miss PRED-naive C16(25%): **dom .210 gen .251** (reactive .465/.507; adaptive .405/.435) | **Loop gira su arch DeepSeek.** Markov AFFIDABILE (64-exp più markov-predicibili dei 128 Qwen) -> pred-naive stravince; adaptive τ=0.5 **troppo cauto** su predittori forti (limite noto = τ dinamico). File `spex/traces_ds2lite_{domain,general}.npz` | done |
| H6 | Sub-expert | Ridondanza cross-expert **Qwen 30B** (pesi merged) | gate_proj 128 expert, L3/24/44 | coseno off .004, eff_rank 127.9/128, x-expert neuron NN .16 frac>0.8 **0%** | **NO ridondanza** (expert ortogonali) | done |
| H7 | Sub-expert | Ridondanza **DeepSeek-V2-Lite** (64 routed+2 shared) | safetensors, L6/13, arch DeepSeekMoE | coseno off .002, eff_rank 64/64, x-expert NN .17 frac>0.8 **0%**; routed->shared NN .13 frac>0.8 **0%** | **NO ridondanza anche su DeepSeek.** Shared ortogonali ai routed. **Verdetto generalizza a 2 famiglie MoE** | done |

**Numeri chiave batch 2:**
- **Loop hidden 30B (miss@C, ADPT-DSpark):** dom C4 .511/C6 .277/C8 .098/C16 .064; gen C8 .175; code C8 .207 (il più duro). Markov 2-5× peggio.
- **Accuracy-drop 30B (×full):** oracle lossless C≥8; markov 195×(C4)/59×(C8)/16×(C16)/2.1×(C48); random floor 3331×(C4)→52×(C48).
- **Ridondanza:** Qwen expert-cos 0.004 / DeepSeek 0.002, entrambi rango pieno, 0% neuroni condivisi.

**Aperti dopo batch 2:** (1) hidden-drop (H5, in corso); (2) ridondanza FUNZIONALE activation-space (non pesi); (3) tok/s reali su 3060 (spex_speed_sim); (4) 235B+hidden misurato (non estrapolato) = altro dump B200. Doc: `SPEX_LOOP.md`, `MODELS_qwen235_vs_deepseekv2.md`.

---

## I) BATCH 3 (2026-07-03) — V4-Flash redundancy (3ª famiglia, CHIUSA) + speed-sim V4 tok/s

Pod (A100-80GB) **TERMINATO** a fine batch. Chiude la 3ª famiglia sub-expert e produce la prima stima analitica tok/s su HW utente (RTX 3060 12GB + 32GB RAM + NVMe ~5GB/s).

| # | Cat | Esperimento | Setup | Risultato (numeri esatti) | Verdetto | File dati | Script | Stato |
|---|-----|-------------|-------|---------------------------|----------|-----------|--------|-------|
| I1 | Sub-expert | Ridondanza **DeepSeek-V4-Flash** L14 | 256 routed + 1 shared, fp4, block-FP8 dequant, w1=gate | expert-cos **0.0278**, eff_rank **246.2/256**, x-expert NN 0.137 (frac>0.8 = **0%**), routed→shared 0.079 (frac>0.8 = **0%**) | **NO ridondanza** cross-expert weight-space | log pod (probe redundancy_probe_v4.py) | redundancy_probe_v4.py | done |
| I2 | Sub-expert | Ridondanza **DeepSeek-V4-Flash** L28 | idem, layer 2/3 lista MoE | expert-cos **0.0297**, eff_rank **245.2/256**, x-expert NN 0.135 (frac>0.8 = **0%**), routed→shared 0.079 (frac>0.8 = **0%**) | **NO ridondanza** (riprodotto su 2° layer) | log pod | redundancy_probe_v4.py | done |
| I3 | Sub-expert | **VERDETTO cross-famiglia** (3 famiglie) | Qwen 128/0-shared + DS2-Lite 64/2-shared + V4-Flash 256/1-shared fp4 | tutte: expert ortogonali, rango ~pieno, NN frac>0.8 = 0%, routed→shared 0% | **Sub-expert weight-space CHIUSA e NEGATIVA su 3 famiglie.** Resta solo activation-space | H6/H7 + I1/I2 | 3 probe redundancy_* | done |
| I4 | SPEX-speed | **Speed-sim V4-Flash tok/s** (modello memory-bound 3 livelli) | 284B/13B fp4, statico 4.72 GiB residente VRAM@320GB/s, expert-dyn 3.05 GiB/token, bande VRAM 320 / PCIe-tier 14 / SSD 5 / RAM 36 GB/s; grid 432 celle | best **74.44** tok/s (tetto teorico); realistic-bracket **47.29**; dominio-ristretto miss0.10 **45.71**, BOE miss0.08 **56.44**; SPEX ×1.28, DSpark ×1.31, combined ×1.83; soglia 10 tok/s a **miss ≤ 33%** | 10 tok/s RAGGIUNGIBILE in tutto il bracket realistico. **74 = tetto, NON stima operativa.** Range difendibile onesto **30-50 tok/s** | `spex/v4_speed_sim.json` | `src\msc\spex\spex_speed_sim.py` | done |
| I5 | SPEX-speed | **VERIFICA indipendente** speed-sim | ricalcolo a mano ogni numero-chiave + audit fisico 5 punti | aritmetica OK (JSON riproduce il codice esatto); verdict = **MINOR_ISSUES** | 3 assunzioni ottimistiche: (a) overlap-VRAM pieno gonfia SPEX ×1.28 (se solo-compute → ×1.02, best ~44.7); (b) confusione drafted-vs-accepted DSpark (accept perfetto; con accept2.5/3 best ~62); (c) `47.29` vale a ram_frac=**0.9** non 0.6 (punto prompt a ram0.6 = **31.85** tok/s). Best onesto **44-49**, dom onesto **30-40** | idem I4 | idem | done |
| I6 | Coverage | **Union expert per prompt ristretto** — "il dominio restringe il modello?" | expert_union_coverage.py su trace q30/q235/ds2lite/olmoe; copertura 99% per-doc (~380 tok = 1 prompt ristretto) e per-corpus | per-doc 99%-cov: q30-dom **81.8/128** (touched 100/128), q235-dom 81.7, DS2-Lite-dom **58.8/64** (touched 64/64), olmoe-dom 53.6/64; gen<dom (wiki più concentrato del dominio); mmlu 64tok→64.7 (ctx corto tocca meno). → V4: dom-union **~88 GB** (Qwen) / **~127 GB** (DS2) | **Un prompt ristretto NON restringe staticamente il modello:** attiva ~55-65% expert/layer → union **> 44 GB, non entra**. Il guadagno-dominio è **TEMPORALE** (miss .098, cache calda) non statico. Static-shrink reale **solo con fine-tune/REAP** (collassa il router). Famiglie DeepSeek fine-grained = coda ancora più grassa | trace `*.npz` | `expert_union_coverage.py` | done |
| I7 | Footprint | **Union footprint 44GB per SCHEMA di precisione** — quale quantizzazione fa entrare la union-dominio | union-dominio V4 = hot 104.1 + cold 66.3 exp/layer × 43 layer; byte/expert = 25.17M·bpw/8; budget 44GB (VRAM12+RAM32) | fp4(4.0/4.0) **92.20 GB** NO; dwarf(2.3/2.3) **53.01 GB** NO; tempmix(2.3/1.58) **46.56 GB** NO (−2.56 mancano); cold-1.0(2.3/1.0) **41.35 GB** SI; all-1.0 23.05 GB SI. Statico: fp16 33.95 / fp8 16.98 / mla-fp16 23.85 GB — **nessuno entra nei 12GB VRAM del 3060** | **Nessuna delle 3 config date scende sotto 44GB.** Serve **cold ≤1.0 bpw** (41.35) o statico MLA-realistic (~24GB). Il temp-mix arriva vicino (46.6) ma non chiude. Lo statico è il vero muro sul 3060: sfora VRAM12 in ogni precisione → overflow RAM/PCIe ogni token | `spex/v4_speed_sim_quant.json` (key_results.*.union_footprint_gb) | `src\msc\spex\spex_speed_sim_quant.py` | done |
| I8 | SPEX-speed | **Speed-sim QUANT** tok/s: fp4 vs dwarf-2bit vs temp-mix (statico per-tier) | spex_speed_sim_quant.py, 3 config × 32 celle = 96; miss{.05-.20}×ram{.6,.9}×A{2,2.5}×spex{0,1}; expert hot/cold parametrico + statico fp16/fp8 con overflow RAM via PCIe | **TOTALE 3060 reale @dominio**(miss.10 ram.9 A2.5 spex): C_fp4 **5.19** / C_dwarf **1.48** / C_tempmix **5.27** tok/s — statico domina, fp16-static di dwarf sfora 22.9GB/tok e crolla. **EXPERT-ONLY** (statico fittizio in VRAM, isola la quant expert; C_fp4 riproduce ESATTO il 45.71 del sim orig = sanity): fp4 **45.71** / dwarf **53.13** / tempmix **53.13** → **2-bit = +16% (×1.16)** sul percorso expert; al dominio il temp-mix non aggiunge (fetch nascosto sotto compute), ma vince nelle celle fetch-bound (miss.20 ram.6: +24%, fetch 50.4 vs 73.4 ms) | **Static fp8 è OBBLIGATORIO** sul 3060: sotto fp8 la velocità è cappata ~5 tok/s a prescindere dagli expert. Il 2-bit sugli expert dà +16% sul path puro ma è irrilevante se lo statico fp16 stream-a da RAM. Temp-mix = margine footprint (−6.5GB) + vittoria nelle fasi fetch-bound. Fetch scala col bpw (0.575× hot, 0.395× cold-1.58), non "dimezzato" | `spex/v4_speed_sim_quant.json` | `src\msc\spex\spex_speed_sim_quant.py` | done |
| I9 | Geometria (CORREZIONE) | **Conteggio param reale V4-Flash da HF API** — corregge 284B→158B e expert 25.2M→12.6M | HF API `safetensors.parameters` per dtype | I8 141.73B (expert) + F8_E4M3 6.02B (attn) + BF16 1.42B (embed) + F8_E8M0 8.86B (scale-ue8m0) + F32 0.04B = **totale 158.07B**, disco **159.6 GB**. expert reale **12.88M** (141.73B/11008) = conferma probe 3×2048×2048 (**hidden-expert 2048, non 4096**); **STATICO attn+embed ~7.4B ≈ 7-9 GB** (NON 24-34); scale E8M0 = streaming-expert non statico | **CORREGGE I4/I7/I8** (usavano 25.17M/exp + statico 24-34GB gonfiati): footprint union ~**½** → **dwarf-2bit dom ~26 GB ENTRA nei 44GB**, fp4 ~46GB, tempmix ~23GB; statico ~7-9GB **ENTRA in VRAM12 a fp8** → regime realistico **~45-53 tok/s** (il "5 tok/s" di I8 usava statico 24GB). **V4-Flash = 158B/13B, non 284B** | HF API deepseek-ai/DeepSeek-V4-Flash | curl | done |
| I10 | REAP-union | **REAP/FT su dominio: quanto riduce l'union** (offline, gratis, coi router-scores) | reap_union_sim.py su hidden_scores_q30_dom/gen; prune per saliency (massa-gate softmax) + reroute ai superstiti; reroute_rate = LIMITE-SUP del danno (healing lo recupera) | dom: **keep50% reroute 8.2% mass 99.3% → union V4-2bit 40→20 GB**; keep37.5% reroute 16.9% mass 97.7% (14.9GB); **keep25% reroute 30% mass 93.5% → 10 GB ENTRA in VRAM12**; keep12.5% reroute 49.6% mass 82.9% (5GB). gen tollera meglio (keep50 reroute 4.4%) | **REAP-50 dimezza il footprint quasi-lossless (99.3% massa, 8% reroute pre-healing) → domain-union ~20 GB; REAP-75 (keep25%) lo fa entrare tutto in VRAM12 (10GB) ma reroute 30% = serve healing/FT.** FT/REAP = leva reale per il fit-VRAM (coerente con "REAP-50 sweet-spot" dello studio pruning). Proxy routing-level; qualità vera = the rubric-scored eval set post-healing | `hidden_scores_q30_*.npz` | `reap_union_sim.py` | done |
| I11 | Qualità-2bit | **Eval reale IQ2_XXS 2-bit su ds4** (il 2-bit degrada?) | ds4 buildato CUDA sm_80 su pod A100-80GB (build OK, inferenza OK, "Paris" corretto); ds4-server OpenAI-API; 15 prompt from the rubric-scored eval set (5/cat) temp0; grade unit-test/exact-match/mcq | **CODING 4/5, MATH 5/5, MMLU-collegemath 4/5 = 13/15 (87%)**; **entrambi i fail = TRUNC** (cap-token mid-thinking, non risposte errate) → sui completati **13/13**. avg 1.04 t/s (streaming A100, NON il 3060) | **2-bit NON distrugge la capacità**: coding passa gli unit-test, math exact-match, MMLU-hard ok; i fail sono artefatti di budget-token (V4-Flash è reasoning, serve più budget). Build ds4 CUDA **validata end-to-end**. Segnale forte ma N=15, nessun riferimento 4-bit ancora | pod effimero (terminato) | `run_eval_ds4.py` + `domain_rubric_eval.jsonl` (140 item, rubric-scored eval set) | done |

**Numeri chiave batch 3:**
- **V4-Flash redundancy:** L14 cos 0.0278 / eff_rank 246.2/256; L28 cos 0.0297 / eff_rank 245.2/256. NN cross-expert e routed→shared entrambi frac>0.8 = **0%**. Coerente con Qwen (0.004) e DS2-Lite (0.002): 3 famiglie, stessa conclusione.
- **Speed-sim V4 (tok/s):** tetto 74.44; bracket realistico 47.29 (ma dipende da ram_frac=0.9); dominio-ristretto 45.71 (miss0.10) / 56.44 (BOE miss0.08); punto miss0.10 ram0.6 A2.5 = 31.85. Guadagni: SPEX ×1.28, DSpark ×1.31, stack ×1.83. Soglia 10 tok/s: miss ≤ ~33% (tutto il bracket; solo markov 0.50 scende a 9.5).
- **Config V4 verificata:** expert_dtype=fp4, hidden 4096, moe_int 2048, 256 routed + 1 shared, 6 attivi, 43 layer, attention CSA+HCA compressa, MTP draft head nativo (DSpark). Working-set dinamico SPEX = **solo expert routed ~3 GB/token fp4** (non 6.5 GB).
- **Caveat verifica (I5):** il 74.44 è un tetto teorico (overlap-VRAM pieno + accept perfetto). Onesto: best 44-49, dom 30-40. La leva dominante è **ram_frac alto** (miss serviti da RAM 36GB/s, non SSD 5GB/s); sopra miss~0.10 il fetch SSD/PCIe domina → fetch-bound. Note fisiche minori: FLOP contati solo su routed (esclude shared+attn, ma memory-bound → trascurabile); KV-cache non modellata (ok a ctx corto); statico 4.72 GiB entra nei 12GB VRAM ma cache=6 GiB lascia ~0.3 GiB per KV → al limite.

**Coverage/union (I6) — conclusione:** restringere il prompt dà un guadagno **temporale** (cache calda, miss .098) ma **NON** uno static-shrink (union > 44 GB anche per 1 prompt). Quindi il nord star si regge su **stream + prefetch SPEX + DSpark**, non su domain-pruning statico. L'unico static-shrink reale = **fine-tune/REAP sul dominio** (riscrive il router, collassa l'union).

**Quantizzazione (I7+I8) — conclusione:** la quant per-tier NON cambia il nord star, lo **rende raggiungibile con un vincolo esplicito**: (a) il **fit-44GB** richiede **cold ≤1.0 bpw** (union 41.35 GB); temp-mix 2.3/1.58 arriva a 46.56 (−2.56 mancano). (b) Sul **3060 il collo è lo STATICO, non gli expert**: fp16-static (33.95 GB) sfora VRAM12 e stream-a 22.9 GB/tok da RAM → **static fp8 è obbligatorio** (cappa ~5 tok/s se non lo fai). (c) Il **2-bit sugli expert vale ×1.16** sul percorso expert puro (45.71→53.13 expert-only), ma è **irrilevante finché lo statico sfora**. Il temp-mix aggiunge solo margine di footprint + vittoria nelle celle fetch-bound (+24% a miss.20/ram.6). **Caveat: il costo-qualità del 2-bit sugli HOT non è misurato da noi** (H4/H5 coprono solo il cold/drop) — serve eval reale IQ2 vs fp4.

**⚠ CORREZIONE I9 (HF API):** i footprint di I7/I8 usano expert **25.17M** (hidden 4096) ma il reale è **12.6M** (hidden 2048) e lo statico reale è **~7-9 GB** (non 24-34). Quindi: footprint union ~**dimezzati** → **dwarf-2bit dom ~26 GB → ENTRA nei 44 GB**; statico ~7-9 GB → **ENTRA in VRAM12 a fp8** → il regime realistico è **~45-53 tok/s, NON ~5** (il 5 nasceva dallo statico gonfiato che sforava la VRAM). Il vincolo resta reale ma morbido: tieni lo **statico a fp8 in VRAM** e gli expert **2-bit** in streaming. Il nord star (10 tok/s) ha così **ampio margine**. Da validare su run reale ds4.

**Aperti dopo batch 3:** (1) **V4 traces REALI** (dump routing V4-Flash via ds4/arch custom) per confermare la miss-rate assunta dal sim (il sim usa miss come parametro, non misurato su V4); (2) poi **run REALE ds4/llama.cpp** per tok/s veri (chiude la matematica del nord star); (3) **misurare quanto un fine-tune/REAP sul dominio ridurrebbe l'union** (I6) → quanto vale davvero fine-tunare per il fit statico; (4) **run reale ds4 a 2-bit + eval qualità 2-bit(IQ2_XXS hot / 1.58 cold) vs fp4** — il guadagno tok/s I8 è simulato e il costo-qualità del 2-bit HOT è NON misurato; (5) ibrido drop/fetch; (6) τ dinamico; (7) ridondanza funzionale activation-space (unica porta rimasta all'idea sub-expert). Sub-expert weight-space **CHIUSA su 3 famiglie**. Doc: `SPEX_LOOP.md`, `HANDOFF_6.md`.

---

*Note di deduplica: duplicati esatti collassati — (A14-A16 == eval_mass_dom_*), (A19-A21 == eval_mass_gen_*), (C3 == eval_full235==eval_p235_full), (C5 == p235_cold25==prune235_25), (C6 == p235_cold50==prune235_50), (C8 == p235_rand50==prune235_50_rand). stepf_base235 e stepf_base235v2 hanno aggregati identici (differiscono solo per-layer). `subexpert.py` non è in `scripts_pod/` (monkeypatch, mai portato nel repo).*

---

## J) BATCH 4 (2026-07-08) — RTX 3060 local: PACE prefill-learned mask, not dynamic compression

Local machine: RTX 3060 12GB, WSL memory raised to 62GB, DS4 V4-Flash 2-bit GGUF
via `/root/ds4/ds4-server --cuda --ssd-streaming --ssd-streaming-cache-experts 258`.

| # | Cat | Esperimento | Setup | Risultato | Verdetto | Stato |
|---|-----|-------------|-------|-----------|----------|-------|
| J1 | PACE/REAP | Baseline practical run, warmup late | `DS4_PACE_WARMUP=50`, `DS4_PACE_WRAP=0`, `DS4_PACE_CACHE_FLUSH=1`; prompt 105, stream client later disconnected | prompt 64.370s; first 50 decode tokens 0.31 t/s; later chunks 2.46-3.17 t/s; final error `client stream write failed` | Mask learned too late; cache flush and no wrap make the first decode window awful. Not a usable config. | done |
| J2 | PACE/REAP | Faster decode warmup + WRAP | `warmup=8`, `keep=23`, `wrap=1`, `cache_flush=0`, cache floor 258 slots; prompt 23, max 32 | prompt 52.106s; PACE applied at tok=9; fattorino touched 6.07 GiB in 1569 ms; gen 32 at 1.75 t/s | Decode improves after PACE applies, but prefill is still the dominant wall-clock bottleneck. | done |
| J3 | PACE/REAP | Prefill-learned dynamic mask | Commit `/root/ds4` `c8dd670 pace-dynamic-prefill-mask`; collect selected experts during prompt, reset stale mask before prefill, apply mask at `tok=0`, wait WRAP before decode | prompt 19 -> prompt done 26.284s; `prefill_apply tok=0`; fattorino 6.07 GiB in 445 ms + wait 454 ms; gen 24 at 2.83 t/s | Positive: dynamic prompt-derived mask is applied before decode and removes the late-warmup penalty. Still not a full prefill fix. | done |
| J4 | SPEX | Local SPEX predictor file check | `DS4_SPEX_MARKOV_FILE=/mnt/c/Users/imanu/source/repos/moe-aggressive-commit/runs/spex/spex_model/ds4flash_d2_nextlayer.spex` | File exists and is `SPX1` hidden predictor (`predictor=2`, shape `43 x 4096 x 256`, fp16 `W_nl`). At the time of the local 3060 runs, DS4 only expected Markov `SPEX` (`predictor=0`) and disabled it. | SPEX was not active in the 3060 local runs; any speedup above is PACE/REAP/WRAP, not SPEX. Runtime support must be tracked separately from artifact existence. | done |
| J5 | Quant/compression | Dynamic expert compression | Desired idea: when an expert is outside the active REAP/PACE working set, store/eject it in a lower-bpw representation and restore/stream it when needed | Not implemented. No alternate low-bpw expert tensor store, no runtime recompression/ejection policy, no quality/speed numbers. | Open. Do not describe current PACE as "dynamic compression"; it is dynamic pruning/masking plus page-cache prefetch. | open |
| J6 | Quality regression | Aggressive keep-23 on coding/HTML | User prompt asked for HTML landing page; local launcher used `keep=23..32`, `relearn=1`, n-gram breath; output showed malformed HTML and repeated `S_INIT`/`NOT` fragments | Log: prompt 95 took 114.994s; decode 2.7-3.5 t/s; breath at tok=125, relearn at tok=205, returned to keep=27 while ngram=0.932 | Negative: keep-23 is too aggressive for code/HTML quality. Local safe-coding launcher moved to `keep=64`, `min=64`, `max=96`, `breath_keep=96`, `relearn=0`, `drift=0.25`, `prefill_apply=0`, `warmup=50`. | done |
| J7 | PACE/REAP | Gradual prebreath ramp before hard breath | Commit `/root/ds4` `c44b0e8 pace: add gradual prebreath ramp`; env: `DS4_PACE_PREBREATH=1`, `PREBREATH_DRIFT=0.18`, `PREBREATH_EVERY=64`, `PREBREATH_KEEP_MAX=96`, `KEEP_STEP=4`; hard breath still available but capped at `breath_keep=96` | Build OK; local server restarted with env confirmed. No completed A/B throughput-quality run yet after the change. | Hypothesis: keep breath as recovery, but start small K ramps before the n-gram drift cliff. Needs pod/local A/B, especially micro-steps 1/2/4, because prior coarse steps likely paid cache churn. | open |
| J8 | SPEX | Recognize hidden `SPX1` artifact in DS4 | Commit `/root/ds4` `bec221c spex: recognize hidden SPX1 predictor`; validates `version=1`, `predictor=2`, `reserved=0`, `L/D/E=43/4096/256`, finite ridge and expected file size; adds `DS4_SPEX_HIDDEN_FILE` alias | Build OK. DS4 can now distinguish hidden SPEX from broken Markov SPEX and report that runtime hidden prefetch is disabled. | Diagnostic/loader milestone only: hidden SPEX still does not score `ffn_norm` or seed prefetch. Need GPU-side hidden scoring/topK to avoid expensive per-token readback. | open |
| J9 | PACE/REAP | Prebreath micro-step test harness | Commit `reap-loop` `e20dcf6 chore(pace): add prebreath microstep harness`; script `scripts/run_pace_prebreath_microstep.sh`; matrix `prebreath_off`, `step4_every64`, `step2_every64`, `step1_every64`, optional `step1_every32`; prompts short HTML and medium code-review | Harness syntax OK and writes logs, outputs, event JSONL and `summary.csv`. RunPod deploy/resume did not start because available 3090/4090/A6000/A40/L40S attempts returned `SUPPLY_CONSTRAINT`; old pod `6ulk5ctgd6w7ir` was `EXITED` and not resumable as on-demand. | Test is prepared but not executed. Next run: on first available pod, `RUNS=1 N=160 INCLUDE_EVERY32=0 bash scripts/run_pace_prebreath_microstep.sh`; add `INCLUDE_EVERY32=1` only if the first sweep is cheap. | open |
| J10 | Tiering | CUDA expert tiering observe-only | Commit `/root/ds4` `94e9a7d cuda: add expert tiering observe mode`; env `DS4_EXPERT_TIERING=observe`, optional JSONL `DS4_EXPERT_TIERING_LOG`, optional periodic stderr `DS4_EXPERT_TIERING_SUMMARY_EVERY`; launcher writes `/root/ds4_tiering_observe.jsonl`; analyzer `scripts/analyze_tiering_observe.py` | Build OK with `make ds4-server CUDA_ARCH=sm_86`. Smoke prompt `Rispondi solo OK`, max_tokens=2: response OK; JSONL created. Diagnostic run: 473 events, 12 resident-cache events at cap 222, 461 selected-direct events, direct loads 2766, hits 0, misses 72, evictions 0. | Positive instrumentation milestone, not compression. It shows the next bottleneck to attack: most selected loads in the smoke are still direct selected staging, so cold-format compression will not help enough until the direct/resident path distinction is handled. | done |
| J11 | Launcher/TTFT | Local 3060 no-think launcher sanity after observe | UI launcher changed locally: `DS4_SPEX_HIDDEN_PREFETCH=0`, trace routing off by default, `DS4_PACE_PREFILL_APPLY=1`, `DS4_PACE_PREFILL_WAIT_WRAP=0`, cache slots 258. Test prompt: no-think `Rispondi solo OK`, max_completion_tokens=1, repeated twice after restart. | Hidden readback failure reproduced first: a 77-token prompt kept running after client abort, no first token for >124s, and finished only on shutdown at 258.252s with 87 tokens. With hidden-readback off and prefill wait on: 21.9s cold-ish, 9.66s warm. With wait off: 10.07s first run, 8.81s second run. Log: `prefill_apply tok=0 keep=64`, fattorino async 16.89 GiB in 1581 ms, prompt done 7.458s then 6.690s, decode 2.197s then 1.695s. Observe JSONL still shows selected_direct:resident 86:86, hit_rate 0, misses 516, direct_loads 3912, cap 258. VRAM was nearly full, about 11.8/12.3 GiB used. | Useful config fix, not a full performance fix. Hidden readback must stay off until GPU-side SPEX exists. `prefill_wait_wrap=0` improves visible TTFT. The remaining bottleneck is prefill selected-direct plus a resident cache that does not hit across these requests; increasing cache slots is not available on 12GB VRAM. Next work: selected-direct/resident reuse, or hidden-SPEX GPU scoring/prefetch; dynamic cold compression remains separate. | done |
| J12 | PACE/quality | K0 first-token A/B: prefill mask versus warmup mask | Same no-think HTML prompt, 73 prompt tokens, 64 completion tokens, temp 0. A: `DS4_PACE_PREFILL_APPLY=1`, `PREFILL_WAIT_WRAP=0` means mask K64 before first generated token. B: `DS4_PACE_PREFILL_APPLY=0` means K0 for warmup; log confirms `PACE learned tok=50 keep=0`, then `PACE descent tok=51 keep=64`. | A: prompt done 54.595s; first 50 generated at 1.33 t/s; last 14 at 2.90 t/s; finish 97.007s; output started valid but more bare. B: prompt done 33.392s; first 50 generated at 0.63 t/s; K64 applied at token 51; last 14 at 1.83 t/s; finish 120.060s; output start looked slightly healthier (`lang=it`, viewport, title, comment). Both observe runs were almost all selected-direct: 2794/2795 events, hit-rate 0, direct_loads 20639. | Quality hypothesis supported but costly: B preserves K0 for the first 50 tokens and starts cleaner, but costs about +23s total on this short HTML sample. For coding/HTML, prefer B until a better split exists: learn/prefetch from prompt without applying mask before token 50. Needed runtime feature: separate prompt-derived prefetch from prompt-derived mask. | done |
| J13 | PACE/quality-speed | K0 warmup then K23: direct cut versus step-down | Same no-think HTML prompt, 73 prompt tokens, 160 completion tokens, temp 0, `DS4_PACE_PREFILL_APPLY=0`, `PREFILL_WAIT_WRAP=0`, hidden SPEX off, observe tiering on. Direct cold-ish: `KEEP=23,MIN=23`. Stepped: `KEEP=64,MIN=23,STEP=8,STABLE=16,ANNEAL_WARM=50,TIGHTEN_LO=1.0,HIT_HI=0.0`. Direct warm rerun: `KEEP=23,MIN=23`. | Direct cold-ish: prompt 61.868s; first 50 K0 at 0.97 t/s; K23 at tok 51; next chunks 3.03/3.35/3.24 t/s; server finish 147.982s, decode 86.112s; output invented broken external URLs. Stepped warm: prompt 15.703s; first 50 K0 at 1.59 t/s; K64 at tok 51 then `tighten` at tok 67/83/99/115/131/147 to 56/48/40/32/24/23; finish 79.713s, decode 64.010s. Direct warm: prompt 10.460s; first 50 K0 at 2.79 t/s; K23 at tok 51; finish 61.004s, decode 50.543s. | On this 3060 short HTML test, step-down does **not** pay: warmed direct K23 is about 19s faster than warmed stepped and has fewer prefetch churn points. This supports the user's prior local finding. Quality is still fragile at K23 for code/HTML, so the safe conclusion is not "K23 is quality-good"; it is "if we are going to cut to K23, gradual downward steps are slower than the direct cut here." Keep K0 for the first 50 tokens; use direct cut for speed tests; solve quality with better predictive/prefetch or dynamic tiering, not with coarse K steps. | done |
| J14 | PACE/quality-speed | Micro-breath after first cut: +1 expert ramps | Same no-think HTML prompt, 73 prompt tokens, 240 completion tokens, temp 0, `PREFILL_APPLY=0`, hidden SPEX off, observe tiering on. Baseline direct: `KEEP=23,PREBREATH=0`. Micro16: `KEEP=23,KEEP_STEP=1,PREBREATH=1,PREBREATH_DRIFT=0.03,PREBREATH_EVERY=16,PREBREATH_KEEP_MAX=64`. Micro8: same but `EVERY=8`. | Baseline direct 240: prompt 9.824s; K0 first 50 at 2.79 t/s; K23 at tok 51; hard `breath(ngram)` at tok 200 to K96; finish 84.816s, decode 74.991s; output had valid colors early but degraded syntax (`max-width 1200`). Micro16: prompt 10.473s; K23 at tok 51; prebreath K24 at tok 52, then K25/K26/.../K35 by tok 237; no hard breath; finish 84.742s, decode 74.269s; output avoided some syntax loss but invented CSS tokens (`#ciseRed/#ciseWhite`). Micro8: K24 at tok 52, then to K45 by tok 233; no hard breath; finish 85.722s, decode 75.728s; output worse lexically (`#cciano`, `#purple`, `70-bar`). All observe runs still selected-direct dominated with hit_rate 0. | Micro16 is the only promising variant: essentially no throughput penalty and it delays/avoids the hard breath. But it did **not** fix code quality on this sample, and micro8 was visibly too aggressive. Do not make prebreath default yet. If revisited, test micro16 on longer generations and with a higher trigger (`PREBREATH_DRIFT` around 0.05-0.08) so it starts when drift actually rises, not immediately after descent. | done |
| J15 | SPEX-hidden | Offline SPX1 hidden predictor eval on DS4 local trace | Added `scripts/analyze_spex_hidden_trace.py`. Collected a tiny local trace with `DS4_SPEX_TRACE_HIDDEN=/root/spex_hidden_trace.dsh`, `DS4_SPEX_TRACE_ROUTING=/root/spex_hidden_trace_routes.csv`, `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`, max_completion_tokens=8, hidden prefetch still off. Evaluated `ds4flash_d2_nextlayer.spex` (`SPX1`, L43 D4096 E256) by scoring hidden at layer L against selected experts at layer L+1. | Trace run took 39.606s for 8 tokens because hidden trace uses synchronous readback. Analyzer: 312 comparable rows. top6 recall 0.5155, hit_any 0.9872, weighted_recall 0.5893. top8 recall 0.5748, weighted 0.6424. top12 recall 0.6368, weighted 0.7021. top16 recall 0.6779, weighted 0.7376. top23 recall 0.7260, hit_any 1.0000, weighted 0.7776. | Positive for predictor quality, negative for CPU readback runtime. Hidden SPX1 is worth pursuing, but only with GPU-side scoring/topK or an async path; the readback path is too expensive for normal use. Keep `DS4_SPEX_HIDDEN_PREFETCH=0` in launcher. Next implementation target: upload SPX1 W to GPU, compute scores from `ffn_norm`, topK cap, prefetch next-layer experts without host synchronization. | done |
| J16 | Tiering | Observe IDs + LRU simulation hook | Commit `/root/ds4` `4de3131 tiering: optionally log selected expert ids`; env `DS4_EXPERT_TIERING_LOG_IDS=1` appends `selected` and `compact_ids` to each observe JSON row. Updated `scripts/analyze_tiering_observe.py` with `--simulate-cap` LRU simulation over `(layer,expert)` keys. | Build OK (`make ds4-server CUDA_ARCH=sm_86`). Smoke `Rispondi solo OK`, max 1: JSON valid with `selected_len=6`, `compact_len=6`. Analyzer on smoke: 516 rows, selected_direct 504 / resident 12, unique_selected=1508, unique_compact=1508. LRU sim: cap64 hit_rate 0.0000, cap128 0.0000, cap258 0.2917. | Instrumentation milestone for dynamic tiering/compression. The old observe log could not simulate policies because it lacked expert IDs. Now we can estimate cache/compression policy hit rates from real traces without changing runtime behavior. Default launcher keeps `DS4_EXPERT_TIERING_LOG_IDS=0` because ID logging is heavier. | done |
| J17 | Tiering | HTML160 ID trace: cache-cap pressure estimate | Same no-think HTML prompt as J13, 73 prompt tokens, 160 completion tokens, `KEEP=23`, `PREFILL_APPLY=0`, hidden SPEX off. Temporarily enabled `DS4_EXPERT_TIERING_LOG_IDS=1`; saved trace as `/root/ds4_tiering_observe_html160_ids.jsonl` (2.4 MB). | Runtime 69.934s client. Analyzer: 6923 events, selected_direct 6922 / resident 1, selected_req=60114, compact_req=45413, unique_selected=5653, unique_compact=5653. LRU sim over compact IDs: cap64 0.0000, cap128 0.0000, cap258 0.3396, cap512 0.5927, cap1024 0.7438. Worst layers by direct loads: L1 1158, L0 1145, L2 1148, L19 1073, L4 1069. | This quantifies the 3060 bottleneck: current effective cap ~258 is far below the useful compact working set. Dynamic compression/tiering should be framed as increasing **effective resident capacity** (toward 512-1024 expert slots equivalent) rather than just shaving bytes off rarely touched cold data. Cap 258 still leaves about two thirds of compact requests as misses in this trace. | done |
| J18 | SPEX-hidden | Optional GPU upload of SPX1 hidden weights | Commit `/root/ds4` `e85e256 spex: add optional hidden GPU weight upload`; env `DS4_SPEX_HIDDEN_GPU_LOAD=1` allocates a `ds4_gpu_tensor` and uploads the hidden SPX1 weight blob. `DS4_SPEX_HIDDEN_PREFETCH` remains independent and can stay off. | Build OK. Smoke with `DS4_SPEX_HIDDEN_GPU_LOAD=1`, `DS4_SPEX_HIDDEN_PREFETCH=0`, prompt `Rispondi solo OK`, max 1: log printed `SPEX hidden GPU weights uploaded 86.00 MiB`; SPX1 loaded; output `OK`; finish 6.268s. Launcher restored to `DS4_SPEX_HIDDEN_GPU_LOAD=0` after test. | First GPU-side SPEX milestone. This does not score or prefetch yet, but proves the hidden predictor weights can live on GPU without enabling the slow readback path. Next step: expose a score/topK function using `g->ffn_norm`, `ds4_gpu_matmul_f16_tensor`, and `ds4_gpu_indexer_topk_tensor`, then feed bounded IDs into next-layer prefetch. | done |
| J19 | SPEX-hidden | Tensor-weight F16 matmul API for SPX1 scoring | Commit `/root/ds4` `269056a cuda: add f16 tensor-weight matmul API`; adds `ds4_gpu_matmul_f16_weight_tensor(out, weights, in_dim, out_dim, x, n_tok)` so F16 weights already resident in a `ds4_gpu_tensor` can be multiplied against F32 activations. | Build OK. Smoke `Rispondi solo OK`, max 1 returned `OK`. API is not wired into SPEX yet. | Removes the blocker found after J18: the existing `ds4_gpu_matmul_f16_tensor` only accepted weights by GGUF `model_map + offset`, not arbitrary SPX1 tensors. Next step is now narrower: allocate a 256-float score tensor, view the correct SPX1 layer slice, call tensor-weight matmul with `g->ffn_norm`, then run `ds4_gpu_indexer_topk_tensor`. | done |
| J20 | SPEX-hidden | GPU-loaded SPX1 overhead A/B before scoring | Same no-think HTML160 prompt as J13/J17, `KEEP=23`, `PREFILL_APPLY=0`, hidden prefetch off, observe tiering on. Temporary launcher `DS4_SPEX_HIDDEN_GPU_LOAD=1` loads the 86 MiB SPX1 tensor but still does not score/topK/prefetch. Compared against the warm direct K23 baseline. | GPU-load run: prompt 10.585s, first 50 decode at 1.75 t/s, finish 73.508s server, decode 62.923s, 79.817s client. Warm direct K23 baseline without GPU load: prompt 10.460s, first 50 at 2.79 t/s, finish 61.004s server, decode 50.543s. Log confirms `SPEX hidden GPU weights uploaded 86.00 MiB`. Launcher restored to `DS4_SPEX_HIDDEN_GPU_LOAD=0`. | Negative but useful: do not enable GPU-loaded SPX1 by default until scoring/topK uses it. The upload foothold is technically valid, but idle SPX1 weights add VRAM/cache pressure and cost about +12.5s decode on this local HTML160 run. | done |
| J21 | SPEX-hidden | GPU score/topK plumbing for SPX1 layout | Commit `/root/ds4` `dfceee3 spex: add hidden GPU scoring path`; adds `ds4_gpu_spex_hidden_score_tensor`, a CUDA kernel for the real SPX1 layout `[hidden][expert]`, graph scratch tensors for 256 scores/topK, and env `DS4_SPEX_HIDDEN_GPU_SCORE=1`. `GPU_SCORE=1` implies GPU weight upload but does **not** enable prefetch. | Build OK (`make ds4-server CUDA_ARCH=sm_86`). Smoke with `DS4_SPEX_HIDDEN_GPU_SCORE=1`, prefetch off, prompt `Rispondi solo OK`, max 1: output `OK`; log printed `SPEX hidden GPU weights uploaded 86.00 MiB` and `SPEX hidden GPU scoring active cap=6 (prefetch still off)`; finish 14.791s on that cold-ish smoke. Launcher restored to `DS4_SPEX_HIDDEN_GPU_SCORE=0`. | Important plumbing milestone: the previous tensor-weight matmul API was not sufficient for SPX1 because the artifact is stored input-major (`hidden,expert`), not output-major (`expert,hidden`). DS4 now has a layout-correct GPU score/topK path. Remaining work: compare predicted IDs to router IDs without expensive per-token readback, then feed bounded predicted IDs into next-layer prefetch/residency. | done |
| J22 | Tiering/compression | LRU capacity-cost reporting | Updated `scripts/analyze_tiering_observe.py` with `--slot-mib` and `--capacity-scale` so `--simulate-cap` reports native and compressed memory cost per effective capacity. Added `tests/test_tiering_observe_analyzer.py`. | On the J17 HTML160 ID trace with slot 6.75 MiB: cap258 hit 0.3396 costs 1741.5 MiB native / 870.8 MiB at x0.5; cap512 hit 0.5927 costs 3456.0 MiB native / 1728.0 MiB at x0.5; cap1024 hit 0.7438 costs 6912.0 MiB native / 3456.0 MiB at x0.5. Test: `pytest tests/test_tiering_observe_analyzer.py` passed. | This turns the dynamic compression target into a sizing tool: the local trace says "behave like cap512+" and the analyzer now shows the memory price of getting there under candidate cold/warm compression scales. It is not yet a runtime compression implementation. | done |
| J23 | SPEX-hidden | HTML160 overhead with score/topK GPU active | Same no-think HTML160 prompt as J13/J20, `KEEP=23`, `PREFILL_APPLY=0`, hidden prefetch off. Temporary launcher `DS4_SPEX_HIDDEN_GPU_SCORE=1` so DS4 uploads SPX1 and runs GPU score/topK every decode layer, but still does not prefetch from it. | Client 69.523s. Server: prompt 11.942s; first 50 decode at 2.70 t/s; 160 decode 51.617s avg 3.10 t/s; finish 63.559s. Logs confirmed `SPEX hidden GPU scoring active cap=6 (prefetch still off)`. Baseline warm direct K23 was finish 61.004s / decode 50.543s; J20 GPU-load-without-score run was finish 73.508s / decode 62.923s. Launcher restored to `DS4_SPEX_HIDDEN_GPU_SCORE=0`. | Positive: layout-correct score/topK itself is cheap on this run, roughly +2.6s server versus the best warm baseline and much better than the noisy/pressure-heavy J20 upload-only run. This supports wiring a consumer/prefetch path next; the kernel cost is not the blocker. | done |
| J24 | SPEX-hidden | Consumer design after GPU topK | Source read of `/root/ds4` CUDA streaming cache: `ds4_gpu_stream_expert_cache_seed_experts_async` takes host `int32_t *expert_ids`, sorts priorities on CPU, computes model offsets, and drives CPU/SSD-backed copies. | Conclusion documented in `docs/SPEX_INTEGRATION_PLAN.md`: a literal zero-host consumer is not compatible with the current loader because SSD prefetch needs CPU-visible IDs. The right next design is async D2H of only the compact topK IDs (6-23 ints) into pinned host buffers, event-polled by the next-layer prefetch path; if not ready, skip and rely on miss fallback. | Important correction: do not reintroduce the old hidden-readback mistake. Reading 4096-float `ffn_norm` and scoring on CPU is bad; asynchronously handing off a tiny topK result is the likely safe bridge from GPU scoring to CPU-driven SSD prefetch. | done |
| J25 | Tiering/compression | LRU target-hit reporting | Extended `scripts/analyze_tiering_observe.py` with `--target-hit-rate`, reporting the first simulated cap that reaches a desired hit rate. Test updated in `tests/test_tiering_observe_analyzer.py`. | On J17 HTML160 ID trace with caps 258/512/1024/2048, slot 6.75 MiB, scales 0.5/0.33: target 0.60 first met by cap1024 (hit 0.7438, 6.75 GiB native / 3.38 GiB at x0.5); target 0.75 first met by cap2048 (hit 0.8153, 13.5 GiB native / 6.75 GiB at x0.5). Test passed. | This makes the dynamic-compression target less hand-wavy: cap512 nearly reaches 60% but misses; if the target is 75%+ on this trace, we need either effective cap around 2048 or a smarter predictor/policy than plain LRU over compact IDs. | done |
| J26 | Verification | Default launcher sanity + public test suite | After all SPEX GPU score experiments, launcher restored to `DS4_SPEX_HIDDEN_PREFETCH=0`, `DS4_SPEX_HIDDEN_GPU_LOAD=0`, `DS4_SPEX_HIDDEN_GPU_SCORE=0`. Ran the same HTML160 prompt once more and ran the full `reap-loop` test suite. | Default HTML160 after restore: client 65.392s; server prompt 10.324s, decode 49.170s, finish 59.494s, avg 3.25 t/s; log showed SPX1 loaded but no GPU weight upload and no GPU scoring line. `python -m pytest -q`: 167 passed, 1 skipped in 16.95s. | Good final state: local UI/server is not left with experimental SPEX flags enabled, and the public repo tests pass after the analyzer/doc changes. | done |
| J27 | Verification | Default 240-token sanity after SPEX patches | Same default launcher as J26, same HTML prompt, max_completion_tokens 240, hidden prefetch/load/score off. | Client 89.719s; server prompt 9.452s, 240-token decode 72.552s avg 3.31 t/s, finish 82.004s. No `SPEX hidden GPU scoring active` line. Earlier direct baseline J14 was prompt 9.824s, decode 74.991s, finish 84.816s. | Longer default path is not regressed by the SPEX scoring patches. This also gives a fresh baseline for the next async-topK handoff experiment. | done |
| J28 | Verification | Default 768-token long sanity | Same default launcher as J26/J27, expanded HTML prompt, max_completion_tokens 768, hidden prefetch/load/score off. The Codex tool call was interrupted, but the DS4 server completed the request and wrote full runtime metrics to the log. | Server prompt 27.214s for 87 prompt tokens. First 50 decode tokens were slow at 0.78 t/s due to cold-ish loading; after K23 descent chunks stabilized around 2.9-3.3 t/s. Hard `PACE breath(ngram)` at tok 290 raised keep to 96 and touched 25.34 GiB in 2679 ms; `breath_end` at tok 370 returned to keep 31. Final: 768 tokens, decode 300.698s avg 2.55 t/s, finish 327.912s. | Important long-run baseline: direct K23 remains fast after warmup, but quality sensor still triggers a hard breath around 290 tokens on longer HTML. This supports the earlier micro-breath finding: a gentle prebreath may be useful as a stability valve, but the next test should compare it on long generations, not only 240 tokens. | done |
| J29 | Tiering/compression | Legacy routing traces feed hot/cold policy replay | Extended `scripts/analyze_tiering_observe.py` to accept tiering JSONL, routing CSV, and routing `.tgz`; added metadata-only tier simulation over hot LRU, warm grace, cold recall, and frozen recall. Added CSV test coverage. | Tests: `pytest tests/test_tiering_observe_analyzer.py -q` -> 4 passed. Reused old traces instead of launching pods: K91 coding trace LRU cap258/512/1024 hit `0.4003/0.5167/0.6611`; product trace `0.4065/0.5250/0.6676`; domain trace `0.3927/0.5075/0.6500`. With `warm_grace=64`, served hot+warm at cap1024 was about `0.68-0.69`; target hit-rate 0.75 remained unmet at cap1024. | This answers the "do we already have sessions?" question: yes for hot/cold shape and policy replay. New pod/local observe runs are only needed for resident-cache vs selected-direct timing/path validation, not for basic hot/cold ranking. | done |
| J30 | Tiering/compression | Local 3060 runtime observe-ID smoke | Same tiny no-think prompt, `DS4_EXPERT_TIERING=observe`, temporary `DS4_EXPERT_TIERING_LOG_IDS=1`, `PACE_PREFILL_APPLY=0`, hidden SPEX GPU flags off. Server restored afterward to default `LOG_IDS=0`. Also ran full public tests. | `pytest -q` -> 170 passed, 1 skipped. Runtime IDs run: prompt 27, gen 2, wall 15.263s cold-ish and 12.181s repeat; output `OK.`. Observe rows per run: 129 events = 86 resident + 43 selected_direct, resident hit-rate 0.2248, hits 116, misses 400, direct_loads 2351, evictions 142, direct 15.87 GiB, compact 19.35 GiB. ID trace: selected_req 7482, compact_req 2867, unique `(layer,expert)` 2438. LRU over compact IDs was very low on this micro-run: cap258 0.0405, cap512 0.0586, cap1024 0.0907. | Confirms runtime path bottleneck even when IDs are captured: the tiny K0/prefill-heavy prompt explodes the working set and is not useful for hot/cold policy quality, but it is useful as a path smoke. Use historical long routing traces for policy shape; use observe-ID runtime only for selected_direct/resident timing and path validation. | done |
| J31 | Tiering/compression | Local 3060 HTML160 runtime observe-ID pair | Current default launcher (`PACE_PREFILL_APPLY=0`, hidden SPEX GPU flags off, `DS4_EXPERT_TIERING=observe`) with only `DS4_EXPERT_TIERING_LOG_IDS=1` temporarily enabled. Prompt: cyberpunk HTML/CSS/JS landing page, temp 0, prompt 84, completion 160. Server restored afterward to default `LOG_IDS=0`. | Run1 cold-ish wall 294.992s; run2 warm wall 133.259s. Each run produced 6923 observe rows: `selected_direct=6922`, `resident=1`, resident hit-rate 0.0, direct_loads 45622, compact_req 45628. Unique compact IDs: 5511 then 5520. LRU over compact IDs run1/run2: cap258 `0.3413/0.3450`, cap512 `0.6099/0.6139`, cap1024 `0.7609/0.7622`, cap2048 `0.8146/0.8138`. At slot 6.75 MiB, cap1024 is 6.75 GiB native or about 2.23 GiB at x0.33. | Warm filesystem/page-cache roughly halves wall time, but the runtime path shape does not change: almost everything still flows through `selected_direct`. For dynamic compression, the useful target is not "compress rare cold experts" in isolation; it is to make the selected/direct path behave like an effective cap512-cap1024 working set, ideally with compressed cold/warm backing and fewer direct SSD pulls. | done |
| J32 | Tiering/compression | Prompt-preloaded all-compressed policy simulation | Extended `scripts/analyze_tiering_observe.py` with prompt-derived preloading: `--tier-prefill-rows auto` uses the first layer cycle as prompt/router signal, ranks `selected` experts, preloads up to cap, then replays decode rows. Reused J31 run2 trace; no new DS4 generation. | Prompt-preload improved decode replay materially. With `warm_grace=64`, `cold_scale=0.33`, cap1024 reached hot-hit `0.8489`, served `0.8586`, runtime promotions `5836/41280 = 0.1414`, total promotions including preload 6860. With immediate demotion to compressed (`warm_grace=0`, all cold in RAM at x0.33), cap1024 still reached hot-hit `0.8489`, promotions `6237/41280 = 0.1511`, scaled footprint about 13.13 GiB. Cap512 was cheaper but weaker: hot-hit `0.6811`, promotion-rate `0.3189`, footprint about 10.54 GiB. | This is the first positive shape for the user's idea: "all experts compressed, prompt router promotes likely hotset, decode dynamically promotes misses." On-demand-only is too stall-prone, but prompt-preload cuts runtime promotions enough to justify a runtime prototype. Still not a speed/quality claim: DS4 must wire a real cold sidecar/promote path, and cap1024 native hot capacity may exceed 12GB VRAM unless it lives as host-native staging or replaces existing selected-direct mechanics. | done |
| J33 | Tiering/compression | DS4 runtime prompt-preload promotion observe policy | Commit `/root/ds4` `e3167cc tiering: add prompt-preload promotion observe policy`; env `DS4_EXPERT_TIER_POLICY=observe_promote`, `DS4_EXPERT_TIER_PROMOTE_CAP`, optional `DS4_EXPERT_TIER_PROMOTE_VERBOSE`. This is metadata-only: no sidecar, no lossy compression, no changed expert tensors. | Build OK: `make ds4-server CUDA_ARCH=sm_86`. Smoke with prompt `Rispondi solo OK.`, max 2, cap1024, `LOG_IDS=1`: output `OK.`, JSONL rows 129 with phases `prompt:43`, `decode:86`; first decode row finalized the prompt hotset with `tier_policy_preloaded=1024`; decode totals in smoke were hot_hits 342, promotions 174, evictions 174. | Runtime plumbing milestone. DS4 can now observe the exact policy shape online instead of only offline in `reap-loop`. This validates prompt->preload->decode state transitions, but it is not yet a speed win. Next step: exact native sidecar/promote path, then compressed cold RAM backing. | done |
| J34 | Tiering/compression | DS4 lossless cold RAM sidecar path | Commit `/root/ds4` `859d3db tiering: add lossless cold RAM sidecar path`; env `DS4_EXPERT_COLD_RAM_LOSSLESS=1`, optional `DS4_EXPERT_COLD_RAM_PREFILL=1`, `DS4_EXPERT_COLD_RAM_VERIFY=1`. The sidecar stores exact native quantized expert bytes in RAM blobs and copies H->D from there before falling back to GGUF/mmap. Default is off; prefill use is off unless explicitly enabled. | Build OK: `make ds4-server CUDA_ARCH=sm_86`. Decode-only smoke did not materialize because the 2-token run hit resident cache. Prefill-enabled correctness smoke (`prefill=1`, `verify=1`, prompt 15, gen 1) returned `OK`, zero failures, materialized 1836 entries, 2220 reuses, 4056 copies, 12.393 GiB RAM blobs. It was intentionally slow: prompt 158.896s, finish 166.687s. Server was restored afterward to default with cold sidecar off. | Correctness/plumbing milestone, not a performance recipe. Lossless RAM proves exact byte addressing/checksum/fallback and frees us to replace the blob payload with a smaller cold format. It also proves why native lossless cannot be the final answer: even a tiny prefill can duplicate ~12 GiB and stall badly. Next target is a smaller cold RAM payload plus prompt-preloaded promotion, not SSD miss recovery. | done |
| J35 | Tiering/compression | Real expert compressibility + CQ1 cold-format lab | Added `scripts/gguf_inspect_ds4.py` GGUF payload sampling and `scripts/ds4_cold_codec_lab.py` CQ1 error/RAM estimator. Tests: `pytest tests/test_gguf_inspect_ds4.py tests/test_ds4_cold_codec_lab.py -q` -> 5 passed. | Real `/root/models/ds4-2bit.gguf`: routed experts are `IQ2_XXS` gate/up + `Q2_K` down, 6.75 MiB/expert, ~72.56 GiB for all 11008 routed experts. Generic lossless compression is a negative: zlib1 avg ratio 0.9931, zlib6 0.9923, lzma0 0.9966, bz2 expands, entropy ~7.95 bits/byte. `ds4-staticQ4.gguf` has the same routed expert formats, so it is not a smaller expert source. CQ1 lab on 768 random blocks: dot nMAE about 0.029 avg across `IQ2_XXS/Q2_K`; estimates `cq1g256` 34.27 GiB all experts, `cq1g64` 40.31 GiB, `cq1g32` 48.38 GiB. Dynamic estimates: `1024` hot native + cold `all:cq1g32` ~= 50.62 GiB; `2048` hot native + cold `all:cq1g32` ~= 52.88 GiB. | Lossless cold RAM is a dead end for memory savings. The first runtime candidate should be hot native + cold CQ1, probably `cq1g32` on the 3060 to leave RAM margin. This is still a quality-risk prototype: weight nRMSE is high (~0.58), so it needs opt-in subset tests and graded outputs before any claim. | done |

Key local finding: on 3060 the useful implemented path is **dynamic working-set
selection**, not static domain masking and not dynamic quantization. Static domain
masking remains rejected for the product path because it risks bad loops and stale
behavior. The current PACE controller is a live K/residency controller; it does
not yet compress cold experts. The next real research step is a separate dynamic
compression tier: hot/kept experts stay in the fast representation, evicted/cold
experts use a more compressed representation, and REAP/PACE decides K and
residency live.

Control hypothesis after J7: do not remove breath. Split it into (a) gradual
prebreath ramps that start before the n-gram cliff, and (b) a capped hard breath
as recovery. Test micro-steps before adopting this in the launcher as a default.

Operational note after J9: no new RunPod was left running. The harness exists so
the next available pod can be used immediately without rediscovering the matrix.

Operational note after J10: SPEX hidden is still not operational in the strong
sense. DS4 can recognize/load the SPX1 artifact and had an experimental
readback seed path. As of J21, GPU-side hidden score/topK exists, but the missing
useful step remains consuming that device-side topK for bounded next-layer
prefetch/residency without host synchronization. Keep this separate from tiering.

Operational note after J11: the UI launcher should not enable hidden-readback
SPEX by default. It creates a CPU/GPU sync loop during prefill/early decode and
can make TTFT unusable. The local practical launcher default is now
`DS4_SPEX_HIDDEN_PREFETCH=0`, `DS4_PACE_PREFILL_APPLY=1`,
`DS4_PACE_PREFILL_WAIT_WRAP=0`.

Operational note after J12: for quality-sensitive coding/HTML, the launcher can
leave `DS4_PACE_PREFILL_APPLY=0` so the first generated tokens are K0. This is
slower but closer to the intended REAP-loop shape: learn during prompt/early
decode, cut only after the warmup window. The better future behavior is
prefill-derived async WRAP without applying the mask before token 50.

Operational note after J13: the step-down controller path is useful as a test
knob, but it should not become the default speed path on the 3060. In the
measured short HTML case, once caches were warm, a direct K0->K23 cut beat
K64->56->48->40->32->24->23 by about 19 seconds server-side. The launcher was
left in the speed-test shape `DS4_PACE_PREFILL_APPLY=0`, `KEEP=23`,
`KEEP_MIN=23`, with test-only forced-tighten knobs removed.

Operational note after J14: one-expert micro-breath should be treated as a
stability valve, not a quality fix. `+1 every 16` can prevent a hard K96 breath
with almost no wall-clock cost on a 240-token local HTML run, but a too-low
trigger starts it immediately after descent and does not repair malformed code.
`+1 every 8` ramps too quickly and worsens lexical weirdness. Launcher default
was restored to direct K23 with prebreath off after the test.

Operational note after J15: the hidden SPEX artifact is not the weak link.
The weak link is runtime plumbing. CPU readback makes even an 8-token trace
slow, but offline scoring shows useful signal. Do not spend more time on Markov
fallbacks for DS4 local; the next meaningful SPEX task is GPU-side hidden
scoring plus bounded next-layer prefetch.

Operational note after J16: ID-bearing observe traces are the bridge from
diagnosis to policy. Before writing any runtime compression/ejection logic,
collect a representative `DS4_EXPERT_TIERING_LOG_IDS=1` trace and run
`analyze_tiering_observe.py --simulate-cap ...`; otherwise we are guessing
which resident/cold capacities actually hit.

Operational note after J17: the first representative ID trace points to a
concrete target: make the 3060 behave more like cap 512+ without actually
allocating uncompressed 512+ resident experts. That is the practical definition
of "dynamic compression" for this path.

Operational note after J18: SPEX hidden now has a safe GPU-loading foothold.
Do not enable `DS4_SPEX_HIDDEN_GPU_LOAD=1` by default on the 3060 yet; it costs
about 86 MiB VRAM and has no runtime benefit until score/topK uses it.

Operational note after J19: the GPU scoring path should not use the GGUF-offset
matmul API. Use the new tensor-weight API so SPX1 can remain outside the model
map and still feed the existing F16 kernels.

Operational note after J20: `DS4_SPEX_HIDDEN_GPU_LOAD=1` is a development flag,
not a runtime default. On the 3060 it should stay off until the score/topK path
turns those resident SPX1 weights into fewer exposed expert misses.

Operational note after J21: SPEX hidden is no longer blocked at "can we score on
GPU?". It is now blocked at "can we consume the GPU topK without adding a
readback/sync and use it safely for next-layer residency?". Keep
`DS4_SPEX_HIDDEN_GPU_SCORE=0` by default until the prefetch consumer is wired.

Operational note after J23: score/topK GPU appears cheap enough to keep pursuing.
The next experiment should not be another scoring microbenchmark; it should wire
the predicted topK into residency/prefetch and measure whether exposed expert
misses actually fall.

Operational note after J24: because the current expert loader is CPU/SSD driven,
the next code step should be an async pinned-host topK handoff with event polling,
not a blocking `ds4_gpu_tensor_read` and not a pretend device-only prefetch.
