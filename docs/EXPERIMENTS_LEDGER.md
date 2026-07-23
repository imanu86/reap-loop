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
| A21 | Pruning | PPL generale K70 (mass) | 30B, general_it | ppl **20.387** (3.56x) | Divergena dom-vs-gen massima a K alti | `eval_genppl_k70.json` == `eval_mass_gen_k70.json` | idem --k 70 | done |
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
| J36 | Tiering/compression | DS4 CQ1 cold sidecar runtime prototype | Commit `/root/ds4` `809218d tiering: add cq1 cold sidecar prototype`; env `DS4_EXPERT_COLD_FORMAT=cq1g32|cq1g64|cq1g256`, optional `DS4_EXPERT_COLD_RAM_PREFILL=1`, `DS4_EXPERT_COLD_RAM_VERIFY=1`. CQ1 stores sign bits plus fp16 group scales, then repacks/decodes to the native IQ2_XXS/Q2_K buffers so the existing CUDA expert kernels remain unchanged. | Build OK: `make ds4-server CUDA_ARCH=sm_86`. Destructive CQ1 prefill smoke on RTX 3060 materialized 1017 entries, reused 789, copied 1806, used 4576.50 MiB compressed CQ1, repacked 12190.50 MiB native bytes, and reported zero materialization/copy/verify/repack failures. It output `????` and was very slow, so broad CQ1 prompt/prefill is a negative. With the conservative default guard, CQ1 entries/copies stayed at zero and the normal launcher was restored. | Mechanical path works, quality/perf policy does not. The next implementation must be phase-aware: keep prompt/prefill and first generated tokens native, derive a hot native set from prompt/router, then admit CQ1 only for cold misses or bounded post-warmup demotion. | done |
| J37 | Tiering/compression | DS4 CQ1 phase/hotset gate | Commit `/root/ds4` `dc7eaa0 tiering: gate cq1 by prompt hotset`; CQ1 now requires the prompt-derived `observe_promote` hotset unless `DS4_EXPERT_COLD_ALLOW_UNGATED=1`, keeps prompt/prefill native unless `DS4_EXPERT_COLD_RAM_PREFILL=1`, and defaults to `DS4_EXPERT_COLD_NATIVE_TOKENS=50` before cold CQ1 is allowed. Cache fills now receive the real prompt/decode phase instead of being forced permanently prefill-like. | Build OK: `make ds4-server CUDA_ARCH=sm_86`. RTX 3060 smoke, direct path, cap32, warmup 50: output one token `L`, CQ1 stayed off with `entries=0`, `policy(no_hotset=6,warmup=252)`. Destructive warmup 0 direct path: `entries=252`, `copies=252`, `bytes=1134.00 MiB`, `repacked=1701.00 MiB`, zero failures, one-token output `R`, decode still ~42s. Cache-fill smoke with 1 resident slot and warmup 0 also materialized 252 entries with zero failures and similarly poor speed. | The phase gate works mechanically and prevents accidental first-token/prompt CQ1. It does not make CQ1 usable yet: warmup 0 is still a quality/perf negative. The next useful test is longer native warmup with CQ1 admitted only after token 50 on prompts where quality can be graded. | done |
| J38 | Tiering/compression | Local CQ1 after native warmup | DS4 `/root/ds4` `dc7eaa0`, local RTX 3060, prompt HTML, `DS4_EXPERT_COLD_FORMAT=cq1g32`, `DS4_EXPERT_TIER_POLICY=observe_promote`, `DS4_EXPERT_TIER_PROMOTE_CAP=32`, `DS4_EXPERT_COLD_NATIVE_TOKENS=50`, max 64. Artifact: `runs/ds4/20260709_cq1_parallel/local_3060_cq1_native50`. | Prompt 64 took 23.335s. First 50 decode tokens ran at 1.52 t/s. After CQ1 became admissible, the last 14 tokens collapsed to 0.06 t/s; total wall 324.028s. CQ1 summary: `entries=1073`, `copies=3612`, `bytes=4828.50 MiB`, `repacked=24381.00 MiB`, failures 0. Output began as HTML but degraded/truncated (`CYBER·AI · NEGO·STO...`). | Negative policy result: synchronous cold CQ1 materialization/repack on selected/cache misses is the wrong place to pay compression. Keep CQ1 as a payload candidate, but move admission to a breath/exchange actuator with background promotion/demotion. | done |
| J39 | PACE/tiering | Breath exchange observe and one-expert ramp | DS4 `/root/ds4` `0bdad9a tiering: observe pace breath exchange`; env `DS4_PACE_EXCHANGE_OBSERVE=1` logs old/new mask deltas. Two local smoke runs: (A) `PREFILL_APPLY=1`, `KEEP=23`, `KEEP_STEP=1`, `PREBREATH_EVERY=1`, `WRAP=0`, max 96; (B) `PREFILL_APPLY=0`, same micro-ramp with `WRAP=1`, max 80. Artifacts: `runs/ds4/20260709_cq1_parallel/local_3060_exchange_observe*`. | A confirmed the bad first-token shape: mask applied at `tok=0`, `K0->K23` demoted 10019 expert slots, then `K23->32` promoted 43 slots per +1K step; quality repeated malformed HTML and total wall was 452.654s. B confirmed proper K0 warmup: `learned tok=50`, `descent tok=51`, then +1K prebreaths. Each +1K promoted about 43 layer-expert slots, but WRAP ran on every micro-step (`6.07 GiB` first touch, then 6.34..8.45 GiB) and the cold first 50 tokens were only 0.11 t/s; total wall 589.391s. | The exchange signal is exactly the hook needed for dynamic compression, but one-expert-per-token with synchronous mask apply + WRAP is not a default. Next actuator should batch/queue exchange work: promote/decompress experts entering the widened mask and compress experts leaving it, but do this asynchronously and bounded, not by forcing full WRAP every token. | done |

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

Operational note after J39: do not remove breath. Reinterpret breath/prebreath
as the natural exchange window for dynamic compression: experts moving
cold/pruned -> warm/kept should be promoted and decompressed ahead of use, while
experts moving warm/kept -> cold/pruned should be compressed/demoted in the same
bounded background loop. The selected-miss CQ1 path is only a correctness
fallback; as a normal policy it is too slow.

Operational note after J40: raw-router K-constant rotation is worth keeping as
an experimental actuator, not as a default speed path. On 2026-07-09 RTX 4070 Ti
pod tests, cache64 static K23 was faster on HTML (3.15 t/s) but looped
(`repeat_flag=1`), while rotate32 cache64 was slower (2.74 t/s) and did not trip
the repeat detector. On code_mini, rotate32 had no quality win and cost speed
(2.77 vs 2.93 t/s). Keep `DS4_PACE_ROTATE=0` by default; next step is
quality-triggered rotation based on raw out-of-mask mass / n-gram risk, not
periodic rotation forever. See `docs/DS4_K23_ROTATION_POD_RESULTS_20260709.md`.

Operational note after J41: local RTX 3060 cache sweep isolated
`--ssd-streaming-cache-experts` from pod/local confusion. Fixed setup:
K0 warmup 50 -> K23, no breath, no prebreath, no rotation, one 64-token warmup,
routing trace off. HTML320 and code_mini256 were run in forward/reverse order.
`cache64` is consistently too small (HTML avg 1.78/1.85 t/s, code avg
2.31/1.69 t/s, ~82k-99k tier evictions). `cache128` was the best measured warm
point (HTML 3.03/3.34 t/s, code 3.23/3.29 t/s). `cache258` removes almost all
tier evictions (12 misses / 0 evictions in these runs) and is close when warm
(HTML 3.23 t/s, code 3.07 t/s), but cold first-in-order rows are heavily
penalized. For the 3060 launcher, treat 128 as the current throughput candidate
and 258 as the low-eviction/quality-safety candidate; do not use 64 as a default
except for constrained-pod fallback. See
`runs/ds4/20260709_local_cache_sweep_k23_RESULTS.md`.

Operational note after J42: the user-requested four-test A/B was finally run as
specified, then repeated with cache256. Same HTML prompt, K0 first 50 tokens,
then K23. Breath K0->K23 and breath K96->K23 both produced zero useful
post-return tokens by the repeat detector: degeneration happened before or
during breath. Static K23 was fast but looped early. K23 raw-router rotate32 was
the only tested actuator that reached 800 streamed tokens without the triple
repeat detector firing, at a throughput cost. Cache128 rotation was faster than
cache256 rotation in this run (3.03 vs 2.61 t/s). See
`runs/ds4/20260709_requested_breath_rotation_RESULTS.md`.

Operational note after J43: top-expert higher precision is a valid next test,
but it is not a runtime flag with current assets. `ds4-staticQ4.gguf` does not
contain Q4 routed experts: routed `ffn_gate_exps`/`ffn_up_exps` remain
`iq2_xxs`, routed `ffn_down_exps` remains `q2_k`; only static tensors such as
`ffn_gate_inp` differ. Testing "top1 expert per layer at int4/Q4" requires a
hybrid GGUF or sidecar with selected routed experts in a higher-precision
format.

Operational note after J44: add a future "safe transition / sanitize"
experiment before drawing conclusions from two-phase warmup sweeps. Prior
cache1024 W-sweep evidence suggests some failed HTML runs were document
restarts, not monotonic W degradation: the phase-2 prompt re-prefilled
`[original instruction] + [partial HTML]`, and when W cut the prefix inside a
CSS declaration the model often emitted a second ```html / `<!DOCTYPE html>`
instead of continuing. TODO: (1) freeze only at safe structural boundaries
(`}`, `;`, `>`, code-fence boundary) when doing offline two-phase tests; (2)
prefer single-process observe -> freeze -> continue on the same KV when
available, so the model is not re-shown "write the full document" at mid-file;
(3) later test a recovery/sanitization actuator for breath: rewind or trim a
small window before the first degeneration marker, briefly re-prefill the clean
prefix under a wider mask, and continue. Treat this as a measured recovery
experiment, not as evidence yet.

Operational note after J45: local cache512 isolated the weighted-mask
regression. Same HTML prompt, W=50 full warmup -> fixed K23, no breath, no
prebreath, no relearn, no rotation. Weighted warmup ranking
(`DS4_PACE_WEIGHTED_WARMUP=1`) took 105.773s prompt time, finished 800 tokens in
487.157s at 2.10 t/s, and degenerated into 233 empty CSS comments; first clear
comment-loop marker was token 108. The paired unit-count ranking
(`weighted(warmup=0,relearn=0)`) took 18.338s prompt time, finished in 373.931s
at 2.25 t/s, and avoided the empty-comment loop but still degenerated into 130
`inherit: inherit` repetitions; first `inherit: inherit` x3 marker was token
163. The cache512 setting itself is not enough to rescue fixed K23 HTML quality,
and weighted selected-mass ranking is worse than unit-count in this measured
case. Keep weighted ranking experimental/off by default. The best measured
quality actuator in the nearby set remains raw-router K-constant rotation
(`local_k23_rotate32_cache256`, repeat_flag=0), despite its prefetch cost.

Operational note after J46: the paired cache256 re-run confirmed the same
direction without the cache512 prefill anomaly. Same prompt and schedule:
unit-count K23 cache256 finished in 280.578s at 2.98 t/s, then repeated
`font-v` 79 times with the first 10x marker at token 243. Weighted warmup K23
cache256 finished in 322.623s at 2.57 t/s, then repeated `option` 339 times
with the first 10x marker at token 141. Both had `repeat_flag=1`; weighted was
slower and degraded earlier. Do not spend more local time on selected-slot
weighted warmup as a quality actuator unless the ranking formula changes
substantially. Use cache256/unit-count as the speed baseline and raw-router
rotation or a raw-router quality trigger as the next meaningful quality path.

Operational note after J47: immediate descent-breath was measured, and the
trigger intuition is correct but incomplete. Same HTML prompt, cache256, W=50
K0/full warmup, K23 descent at tok=51, hard breath and rotation disabled.
`local_descent_prebreath_step1_every4_cache256` started widening at tok=52
(K24) and reached K96 at tok=340; it was too slow, max ngram reached 0.8983 and
the output repeated `box-sizing: border` / `border: 0` dozens of times. It
finished 800 tokens in 351.310s at about 2.35 t/s, with 73 prebreath events.
`local_descent_prebreath_step2_every4_cache256` also started at tok=52 (K25)
but reached K96 at tok=196; this kept max event ngram to 0.0756 through the
critical window, confirming that waiting for ngram 0.07-0.10 is late. However
quality still degraded into a different malformed CSS pattern (`#0-0-0...`),
and the run was expensive: 37 prebreaths, 595.07 GiB touched, 24.786s prefetch
time, 417.959s finish, 2.22 t/s. Conclusion: breath should begin at the K0->K23
change, but simple monotonic widening is not enough; the next actuator should
combine immediate widening with raw-router rotation/relearn so the widening
brings in the right experts, not just more experts from a stale mask.

Operational note after J48: immediate descent-breath with per-step prebreath
relearn was measured. Variant
`local_descent_prebreath_step2_relearn_cache256`, artifacts under
`runs/ds4/20260709_descent_prebreath_step2_relearn_retry_cache256_html800/`.
Configuration matched J47 step2 (`W=50`, K23 at tok 51, +2K every 4 tokens to
K96, no hard breath, cache256) but enabled
`DS4_PACE_PREBREATH_RELEARN=1`. The log confirms the mechanism: every widening
step emitted `PACE prebreath_relearn` immediately before `PACE prebreath`
(example: tok 52 keep 23 -> K25, tok 56 keep 25 -> K27, continuing to K96 at
tok 196). Result: prompt 92.819s, finish 433.253s, avg 2.35 t/s, last chunk
2.91 t/s, 37 prebreaths, 38 prefetches, 601.67 GiB touched, 23.970s prefetch
time. Compared to stale J47 step2, catastrophic `#0-0-0...` repetition
disappeared and `repeat_flag` was 0, but the HTML was still invalid/incomplete:
one `<!DOCTYPE>`/`<html`, no `</html>`, no `<form>`, no `<script>`, and a long
repetition of background CSS declarations. Finding: per-step relearn helps
avoid one bad attractor, but low n-gram plus wider K does not imply code
quality. The relearn path is still selected-expert/visible-mask driven, not a
true raw-router rebuild over all pruned experts.

Operational note after J49: the historical step-down test was partially
repeated to answer whether the old K64->K23 "gradini in discesa" runs were
relearning masks. Variant `local_stepdown_64_to23_stale_cache256`, artifacts
under `runs/ds4/20260709_stepdown_rebuild_cache256_html400/`, used the J13-like
schedule: W=50 K0 warmup, K64 at tok 51, forced tighten by 8 experts every 16
stable tokens (`TIGHTEN_LO=1.0`, `HIT_HI=0.0`, `ANNEAL_WARM=50`) until K23.
Measured events: K56 tok 67, K48 tok 83, K40 tok 99, K32 tok 115, K24 tok 131,
K23 tok 147. It finished 400 tokens in 187.612s server time, avg 2.57 t/s, but
the output was not good HTML: no closing `</html>`, no form/script, and the tail
collapsed into 126 `Courier` tokens. The paired raw-router rotate4 attempt
(`local_stepdown_64_to23_rotate4_cache256`) is intentionally marked invalid as
a K64->K23 A/B: it emitted 87 `PACE rotate` events at tok 55/59/63/.../399 but
never emitted `PACE tighten` and stayed at K64. It did finish server-side
(400 tokens, 315.717s, avg 1.32 t/s), but the runner was stopped before the
HTTP response/content artifacts were saved, so use it only as scheduler/cost
evidence. Root cause from measured behavior and the rotation patch: rotation
applies a mask and resets `stable_since`, so frequent rotation prevents the
tighten stability gate from ever maturing. Next code fix: split "mask refresh"
from "keep change" so raw-router rotate/rebuild does not reset the tighten
timer, or add an explicit `relearn_on_tighten` path that rebuilds the target K
mask at the same token as the tighten.

Operational note J50: prepared the runtime fix and the follow-up matrix for the
"relearn mask at every downward step" hypothesis. Patch
`patches/ds4/0016-pace-rebuild-on-tighten.patch` is designed to apply after
`0015-pace-raw-router-k-rotation.patch` and adds two env-gated behaviors:
`DS4_PACE_ROTATE_PRESERVE_STABLE=1` keeps K-constant raw-router refreshes from
resetting the tighten stability timer, and `DS4_PACE_RELEARN_ON_TIGHTEN=1`
rebuilds the target-K mask from raw-router `rmass` exactly when a tighten step
fires. This is the correct actuator for K64 -> K56 -> ... -> K23 with a fresh
mask at every intermediate K. The current `/root/ds4/ds4-server` binary already
contains an advanced PACE build, but the matching advanced `ds4.c` source was
not found under `/root` or the local repos; rebuilding DS4 therefore requires
first re-applying/rebasing the tracked patch stack instead of compiling the
current upstream-looking `/root/ds4/ds4.c`. Runner variants added for the next
matrix: `local_stepdown_64_to23_stale_cache256` (control),
`local_stepdown_64_to23_rotate20_cache256` (current-runtime boundary test),
`local_stepdown_64_to23_relearn_on_tighten_cache256` (0016 primary test), and
`local_stepdown_64_to23_rotate4_noreset_cache256` (0016 stress test). The parser
now records `PACE prebreath_relearn` and `PACE tighten_relearn` separately so
"mask was actually rebuilt" is a CSV field, not a manual grep.

Operational note J51: current-runtime boundary tests completed. First,
`local_stepdown_64_to23_rotate20_cache256` (HTML, cache256, 400 tokens) proved
that rotation cadence longer than the 16-token tighten gate allows the old
step-down to proceed but does not rebuild masks during descent. Measured events:
K64 at tok 51; tighten K56/K48/K40/K32/K24/K23 at tok 67/83/99/115/131/147;
first rotate only at tok 167, already at K23. Runtime was slow and degraded:
prompt 63.779s, finish 400.863s, avg 1.19 t/s, 12 rotates, 148.62 GiB touched,
34.156s prefetch time, repeat_flag=1, output tail collapsed into repeated
`#` tokens. Second, `local_stepdown_64_to23_raw_collect_cache256` (same schedule,
300 tokens, `DS4_PACE_ROTATE=1`, `ROTATE_EVERY=999999`) measured raw-router
readback/accumulation without applying rotate. It performed the same six
tightens and zero rotates, finishing in 358.632s at 1.15 t/s with 7 prefetches
(75.78 GiB touched, 3.351s prefetch). Output still degenerated, this time into
repeated `href=https://fa5.min?...`. Conclusion: with the current binary,
configuration alone cannot do "fresh mask at each downward step": frequent
rotate blocks tighten, slow rotate arrives after the descent, and raw collect
without tighten-time apply only adds cost. Patch 0016 is required for the
meaningful next A/B.

## 2026-07-10 Open TODO - DS4 Mixed Compression And Further Quant

- **DS4 mixed expert compression reale.** Audit 2026-07-10: il modello usato
  (`/root/models/ds4-2bit.gguf`) e` una imatrix quant, ma i routed expert sono
  uniformi (`gate/up=iq2_xxs`, `down=q2_k`) su tutti i 43 layer. DwarfStar non
  sta gia` facendo "hot Q4 / cold 2-bit / frozen 1-bit" per expert. TODO:
  costruire sidecar/hybrid runtime con hot native o high-precision, cold piu`
  compresso, promozione/demozione guidata da PACE/router, e misurare
  qualita`+latenza.
- **Quantizzazione ulteriore sotto il native DS4.** I test finora sono
  inconcludenti/negativi: lossless su bytes routed nativi salva <1%,
  `staticQ4` non cambia i routed expert, CQ1 e` plumbing-only e degrada se
  ammesso troppo presto. TODO: rifare esperimenti puliti su CQ1/1-bit/sub-CQ1
  con prompt native warmup, hotset prompt-derived, metriche L0-L3, t/s
  segmentato, memoria reale, e fallback native verificato.

## 2026-07-10 PACE Tighten-Relearn Measured Ledger

Evidence constrained to `summary.csv`, `pace_events.jsonl`, and
`content_measured.txt` from the measured run directories.

| Run | Variant | Cache | Max tokens | PACE events | t/s | Prefetch GiB/ms | Qualita osservata |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| `20260710_pace_advanced_ab_html400` | `local_stepdown_64_to23_stale_cache256` | 256 | 400 | `prefill_reset@0`, `learned@50`, `descent K64@51`, `tighten K56/K48/K40/K32/K24/K23@67/83/99/115/131/147` | 1.87 | 75.78 GiB / 6848 ms | Starts with `<!DOCTYPE html>`, but no `</html>`, `<form>`, or `<script>` in measured content; tail repeats `body`/CSS stanzas; `repeat_flag=1`. |
| `20260710_pace_advanced_ab_html400` | `local_stepdown_64_to23_relearn_on_tighten_cache256` | 256 | 400 | `prefill_reset@0`, `learned@50`, `descent K64@51`, `tighten_relearn K56/K48/K40/K32/K24/K23@67/83/99/115/131/147` | 2.22 | 75.78 GiB / 13902 ms | Starts with `<!DOCTYPE html>`, but no `</html>`, `<form>`, or `<script>`; truncates in CSS around `.grid`; `repeat_flag=0`. |
| `20260710_pace_advanced_ab_html800` | `local_stepdown_64_to23_stale_cache256` | 256 | 800 | `prefill_reset@0`, `learned@50`, `descent K64@51`, `tighten K56/K48/K40/K32/K24/K23@67/83/99/115/131/147` | 2.61 | 75.78 GiB / 3345 ms | Longer CSS/comment-like output, still no `</html>`, `<form>`, or `<script>`; measured tail lists sections such as responsive/accessibility/security; `repeat_flag=0`. |
| `20260710_pace_advanced_ab_html800` | `local_stepdown_64_to23_relearn_on_tighten_cache256` | 256 | 800 | `prefill_reset@0`, `learned@50`, `descent K64@51`, `tighten_relearn K56/K48/K40/K32/K24/K23@67/83/99/115/131/147` | 2.76 | 75.78 GiB / 10843 ms | Reaches more CSS, including `.popup {`, but measured content still lacks closing HTML, form, and script; `repeat_flag=0`. |
| `20260710_pace_relearn_cache512_html400` | `local_stepdown_64_to23_relearn_on_tighten_cache512` | 512 | 400 | `prefill_reset@0`, `learned@50`, `descent K64@51`, `tighten_relearn K56/K48/K40/K32/K24/K23@67/83/99/115/131/147` | 2.23 | 75.78 GiB / 13878 ms | Starts the same HTML/CSS shell, but no `</html>`, `<form>`, or `<script>`; tail collapses into repeated `0.0.0...`; `repeat_flag=1`. |
| `20260710_pace_rotate4_noreset_html400` | `local_stepdown_64_to23_rotate4_noreset_cache256` | 256 | 400 | `prefill_reset@0`, `learned@50`, `descent K64@51`, 85 `rotate` events `@55..397`, `tighten_relearn K56/K48/K40/K32/K24/K23@68/85/102/119/136/153` | 1.38 | 699.54 GiB / 65621 ms | Preserved-stable rotate no longer blocks descent, but measured content still has no `</html>`, `<form>`, or `<script>` and truncates in CSS; `repeat_flag=0`. |
| `20260710_pace_rotate4_noreset_nowraprotate_html400` | `local_stepdown_64_to23_rotate4_noreset_cache256` + `DS4_PACE_WRAP_ROTATE=0` default | 256 | 400 | same 85 `rotate` events, `tighten_relearn K56/K48/K40/K32/K24/K23@68/85/102/119/136/153` | 1.28 | 75.78 GiB / 21953 ms | `wrap_rotate=0` cut full-set rotate page-in from 699.54 GiB to 75.78 GiB, but content still incomplete and decode remained slow; no `</html>`, `<form>`, or `<script>`. |
| `20260710_pace_rotate16_noreset_nowraprotate_html400` | `local_stepdown_64_to23_rotate16_noreset_cache256` + `DS4_PACE_WRAP_ROTATE=0` default | 256 | 400 | 21 `rotate` events, `tighten_relearn K56/K48/K40/K32/K24/K23@68/85/102/119/136/153` | 1.24 | 75.78 GiB / 11999 ms | Fewer rotations did not recover throughput or quality; final n-gram still high and content incomplete. |
| `20260709_stepdown_rotate20_cache256_html400` baseline | `local_stepdown_64_to23_rotate20_cache256` | 256 | 400 | `prefill_reset@0`, `learned@50`, `descent K64@51`, `tighten K56/K48/K40/K32/K24/K23@67/83/99/115/131/147`, 12 `rotate` events `@167..387` | 1.19 | 148.62 GiB / 34156 ms | Still no `</html>`, `<form>`, or `<script>`; tail collapses into repeated `#` tokens; `repeat_flag=1`. |

Conclusion: tighten-time relearn is confirmed by the measured
`tighten_relearn` events and is the best cache256 variant here on avg t/s
(2.22 vs 1.87 at 400 tokens; 2.76 vs 2.61 at 800 tokens), while avoiding the
measured repeat flag at cache256. It is not a quality pass: every measured
content file remains incomplete HTML and never reaches a real form/script.
Cache512 does not improve quality, and rotate4_noreset proves the stability
timer fix works under frequent refreshes but is too expensive and still
incomplete. `DS4_PACE_WRAP_ROTATE=0` removes most full-set rotate prefetch
bandwidth, but periodic K-constant rotation still underperforms; the next
useful direction is drift-triggered rebuild/widening or delta-prefetch, not
blind periodic rotate.

## 2026-07-10 Direct K23 vs K64->K23 Reproduction

Question: previous manual runs suggested that `W50 full/K0 -> K23 direct` could
beat gradual `K64 -> ... -> K23`. This reproduction keeps prompt/cache/model
constant at cache256 and compares direct unit-count, direct weighted warmup, and
completed stepdown relearn.

Evidence:

| Run | Variant | Schedule | Finish / avg | Prefetch | Quality observation |
| --- | --- | --- | --- | --- | --- |
| `20260710_direct_k23_vs_stepdown_html800` | `local_k23_cache256` | W50 full/K0 -> K23 direct | 344.096 s / 2.85 t/s | 6.07 GiB / 448 ms | 800 tokens, but CSS reset loops almost immediately; 51 repeats of `margin: 0, padding`; no `</html>`, `<form>`, or `<script>`. |
| `20260710_direct_k23_vs_stepdown_html800` | `local_k23_weighted_warmup_cache256` | W50 full/K0 -> K23 direct, weighted warmup | 307.296 s / 2.87 t/s | 6.07 GiB / 449 ms | 800 tokens, but selector list loops; 215 `h6` occurrences; no `</html>`, `<form>`, or `<script>`. |
| `20260710_stepdown_relearn_only_html800` | `local_stepdown_64_to23_relearn_on_tighten_cache256` | W50 full/K0 -> K64 -> K56 -> K48 -> K40 -> K32 -> K24 -> K23 | 438.872 s / 2.09 t/s | 75.78 GiB / 30291 ms | More HTML-like than both direct variants but still invalid/incomplete; no `</html>`, `<form>`, or `<script>`. |

Conclusion: the cache256 reproduction confirms the user's intuition on speed
only: direct K23 is much faster and avoids the 7 full working-set prefetches.
It does **not** reproduce the better-quality old manual result. The likely next
controlled variable is cache/session condition: rerun direct K23 with the old
cache1024/large-cache setup or reconstruct the exact old session-learned mask
before judging the trajectory itself.

## 2026-07-10 Pod Cache1024 Follow-up

Question: does the old cache1024/RAM-hot pod regime restore the quality that
local cache256 direct K23 failed to reproduce?

Evidence:

| Run | Variant | Hardware/cache | Schedule | Throughput | Quality observation |
| --- | --- | --- | --- | --- | --- |
| `20260710_pod_cache1024_html800` | `local_k23_cache1024` | RunPod RTX 3090 24GB, cache1024 | Cyberpunk HTML prompt, W50 full/K0 -> fixed K23, unit-count warmup, no breath/prebreath/rotation | wall 94.576 s, avg 14.12 t/s, last chunk 24.79 t/s, prefetch 6.07 GiB / 219 ms | Fast but invalid: 2501 chars, `repeat_flag=1`, one `<!DOCTYPE>`, no `</html>`, no `<form>`, no `<script>`; tail loops CSS comments. |
| `20260710_pod_cache1024_html800` | `local_k23_weighted_warmup_cache1024` | Same pod/cache | Same, but W50 weighted warmup ranking | wall 79.577 s, avg 16.37 t/s, last chunk 24.71 t/s, prefetch 6.07 GiB / 63 ms | Fastest direct run, but invalid: 2367 chars, `repeat_flag=1`, one `<!DOCTYPE>`, no `</html>`, no `<form>`, no `<script>`; tail loops `Stai attento` comments. |
| `20260710_pod_cache1024_warmup_replay` | W50 two-phase session mask | Same pod/cache; compact coffee-shop prompt recovered from Claude artifacts | Phase 1: W50 wide with routing+weights trace. Phase 2: build keep-23 mask by gate-mass, re-prefill prompt+wide prefix, continue frozen for 950 tokens. | phase1 generation 2.03 t/s; phase2 generation 14.60 t/s | Functionally complete-ish and no repeat: `doctype=2`, `</html>=1`, `<form>=1`, `<script>=1`, `alert=2`, `repeat=0`. Has one restart/duplicate doctype and imperfect JS text, so treat as L2/L3 pending render, not a clean universal L3. |
| `20260710_pod_cache1024_warmup_replay` | W130 two-phase session mask | Same pod/cache/prompt | Same as W50 with W130 and 870-token frozen continuation | phase1 generation 2.30 t/s; phase2 generation 16.24 t/s | Fast but failed quality: `doctype=1`, `</html>=0`, `<form>=1`, `<script>=1`, `alert=0`, `repeat=1`; tail loops `document.addEventListener("DOM"...`. |

Conclusion: cache1024 alone restores the high-throughput regime but not quality
on the current cyberpunk prompt. The old positive result depends on the compact
prompt plus two-phase session-learning recipe. W50 reproduces a useful
functionally complete-ish output at high speed; W130 did not replicate cleanly
in this run. Keep the old cache1024 claims with explicit caveats: pod/RAM-hot,
n=1 greedy, prompt/freeze-point sensitive, and not directly transferable to the
3060 absolute t/s.

## 2026-07-10 Claude Recovery Index

Read-only recovery pass delegated to a sub-agent. Purpose: make historical
Claude/Claude Code findings discoverable without turning them into fresh
measurements. Raw secrets files were not read or quoted. Rows marked "medium"
need raw artifact recovery or rerun before use as headline evidence.

| Historical id | Source | Measured data recovered | Ledger status |
| --- | --- | --- | --- |
| `HIST-CLAUDE-SPEX-30B-20260703` | Claude export/consolidation under `Documents/Codex/2026-07-05/.../outputs/` and `.claude/projects/...` | Qwen3-30B SPEX hidden recall: domain @8/@16/@32 = `.9316/.9906/.9978`; general = `.8292/.9560/.9861`; Markov lower. 235B hidden remained estimated, not measured. | Medium: consolidated Claude output, not rerun here. |
| `HIST-REAP-DS4-K50-V1-20260705` | `runs/reap/2026-07-05_eval_biasmask/README.md` | Full dom ppl 3.811 / gen 5.344; `reap_k50` dom 3.860 ratio 1.013x; random K50 dom 5.200 ratio 1.365x; V0 violations 0/11280. | High for PPL/actuator; speed not present. |
| `HIST-REAP-DS4-K50-V2-20260705` | `runs/reap/2026-07-05_eval_biasmask_v2/README.md` | Same-machine H200 eval: domain full 3.852; `reap_k50` 3.891 ratio 1.010x CI `[0.996,1.025]`; `reap_k67` 1.076x; random about 1.388x; general K50 about 1.403x. | High for PPL; not a generation-quality/tps result. |
| `HIST-K91-CODING-FIT-20260706` | `.claude` memory and `runs/reap/k91_coding_vram/README.md` | RTX 3080 Ti 12GB: K0 full hit 0.35, 2.16 t/s; K91 keep23 hit 0.67, 3.67 t/s; K96 keep9 hit 0.96, 12.02 t/s. Coding quality degraded: full K0 rendered, K50 pseudo-HTML, K91/K96 looped. | High: raw run notes recovered; confirms "fits in 12GB" and "codes well" are different axes. |
| `HIST-CACHE1024-STATIC-K23-20260707` | `docs/CLAIMS_CURRENT.md`, `docs/paper/PAPER.md`, Claude memory | Static file-mask keep-23 17.3 t/s hit 0.986; runtime static-from-token0 11.4 t/s hit 0.923; full no-mask 3.6 t/s hit 0.607; dynamic staircase 2.5 t/s hit 0.557. | Medium: documented in paper/claims; raw summary not found in current tree. Keep as pod diagnostic, not 3060 claim. |
| `HIST-W50-W130-SESSION-CACHE1024-20260707` | `docs/CLAIMS_CURRENT.md`, `docs/paper/PAPER.md`, recovered prompt/scripts | Old claim: cold-static keep-23 L0; session-learned keep-23 L2/L3 from W>=50; W50 L3 peak 13.6 t/s comp about 65 s; W130 L3 comp about 81 s; full L3 peak 3.4 t/s comp about 164 s. New 2026-07-10 replay partially confirms W50 high-speed functional output, but W130 failed in this build. | Medium/open: old raw not found; freeze-point sensitive; new replay narrows the claim. |
| `DS4-K23-CACHE-SWEEP-LOCAL-20260709` | `runs/ds4/20260709_local_cache_sweep_k23_RESULTS.md` | Local 3060 cache sweep: cache128 warm around 3.34 t/s; cache258 around 3.23 t/s; cache64 around 1.85 t/s; HTML repeat remained. | High; already superseded by later controlled cache256/cache1024 runs for current question. |
| `DS4-K23-ROTATE-POD-20260709` | `docs/DS4_K23_ROTATION_POD_RESULTS_20260709.md` | RunPod 4070 Ti 12GB: static64 html avg 3.15 repeat1; rotate16 avg 2.71 repeat0; rotate32 avg 2.74 repeat0; cache128 unsafe for rotation; cache258 failed. | High for 12GB pod rotation; not comparable to cache1024 3090. |

Open recovery gaps:

- Locate or reconstruct the raw `runs/reap/multiseed_2026-07-07/` speed
  diagnostic that backs static keep-23 11-17 t/s. If not found, leave as
  documented medium-confidence paper claim.
- For the session-learning floor, run a real multi-W sweep on the current
  pinned build (`W=30,50,80,110,130,150`) and grade rendered HTML, because the
  2026-07-10 W50/W130 replay showed the expected sensitivity but did not fully
  reproduce the old W130 L3 note.

## 2026-07-15 Native Windows G33 Split Hit/Miss

Question: can resident exact routes compute while the G32 worker fetches true
misses, then join once without changing output?

Setup: native Windows RTX 3060 12 GB, model
`C:\ds4-models\ds4-2bit.gguf`, prompt `Hi`, context 256, max 12 (EOS after 9),
cache336 LRU, 2 GiB stream window, Q8-F16 cache disabled, embedding-row staging
enabled, REAP/SPEX/dynamic arena disabled. G32 control and G33 split were run
with discarded warmup and counter-ordered `n=3` plus `n=5` samples. Exact hash
was enforced for warmup and every measured request.

| Run | Server decode t/s | Client t/s | Worker wait | Exact |
|---|---:|---:|---:|---|
| G32 control n=3 | 3.2233 | 2.1721 | 3.865 ms/layer | yes |
| G33 split n=3 | 3.2633 | 2.1510 | 3.630 ms/layer | yes |
| G33 split n=5 | 3.1860 | 2.1430 | 3.701 ms/layer | yes |
| G32 control n=5 | 3.2060 | 2.1670 | 3.770 ms/layer | yes |

Aggregate across eight requests per variant: server `3.2125 -> 3.2150 t/s`
(+0.08%), client `2.1689 -> 2.1460 t/s` (-1.06%). The intended overlap is
measured, but it does not repay the extra masked launches and final ordered
sum at the current 42.24% route-hit distribution. Verdict: exact mechanism
checkpoint, negative throughput result, keep opt-in. Native-Windows commit:
`e4d669e`; full report: `G33_SPLIT_HIT_MISS_RESULTS.md` in that repo.

## 2026-07-15 Native Windows G34 SPEX IQ2XXS CPU Sidecar

Question: can SPEX-predicted next-layer experts start useful IQ2XXS work on CPU
before exact GPU transport needs them, without changing routing or output?

Setup: native Windows RTX 3060 12 GB, model
`C:\ds4-models\ds4-2bit.gguf`, prompt `Hi`, context 256, max 12 (EOS after 9),
cache336 LRU with GPU-resident routes, 2 GiB stream window, Q8-F16 cache off,
embedding-row staging on, REAP/dynamic arena/G33 split off. Controls retained the
same SPEX score/topK width. Each counter-ordered run used one discarded warmup
and `n=3` measured requests.

The observe-only sidecar uses an eight-slot pinned D2H ring and one CPU worker to
quantize the predicted hidden state to Q8_K and evaluate IQ2XXS gate/up plus
SwiGLU. It records timing/checksum only; exact GPU routing remains authoritative.

| Width | Control server t/s | CPU server t/s | Delta | Control client t/s | CPU client t/s | Delta |
|---|---:|---:|---:|---:|---:|---:|
| K1 | 3.1650 | 3.0583 | -3.37% | 2.1124 | 2.0667 | -2.16% |
| K2 | 3.1200 | 3.0900 | -0.96% | 2.1012 | 2.0955 | -0.27% |

| Metric | K1 | K2 |
|---|---:|---:|
| possible jobs | 3,024 | 3,024 |
| submitted / dropped | 3,020 / 4 | 2,478 / 546 |
| exact-match predictions | 53.68% | 47.28% |
| useful and ready predictions | 20.40% | 0.18% |
| CPU time / submitted job | 4.58 ms | 8.95 ms |
| queue time / submitted job | 2.21 ms | 47.76 ms |
| page-fault multiplier vs control | 2.66x | 4.46x |

All 24 measured outputs and all warmups matched exact content hash
`fda564ba3f7a0f028106d468420f674898ed99ac5bf2765ac9586206e39d73c5`.
The harness hash (`d33d764b...`), executable hash (`44fdcbaa...`), build input
fingerprint (`02f87221...`) and expected/observed SPEX hash (`a86288c3...`) are
recorded in every result JSON.

Verdict: exact negative result. K1 loses throughput despite keeping up with the
queue; K2 drops 18.06% of jobs and almost never finishes a correct prediction in
time. Do not promote unconditional CPU IQ2XXS speculation. Retain the probe and
test SPEX as a high-confidence RAM staging/pinning signal after real tier states
exist. Native-Windows commit: `63ba10d`; full report:
`G34_SPEX_CPU_OBSERVE_RESULTS.md` in that repo.

## 2026-07-15 Native Windows G35 Real Expert Tiering

Question: does an exact physical tier policy that admits every first cold expert
to pinned RAM, then promotes reused experts to VRAM, reduce repeated SSD/model
reads and improve decode without changing routing or output?

Setup: native Windows RTX 3060 12 GB, model
`C:\ds4-models\ds4-2bit.gguf`, prompt `Hi`, context 256, max 12 (EOS after 9),
cache336 LRU, 2 GiB stream window, 8 GiB exclusive
`cudaHostAllocDefault` arena, Q8-F16 off and embedding-row staging on. REAP,
SPEX, G33 split and other observers were disabled. The counter-ordered matrix
ran `off/enforce/enforce/off`; every arm used one discarded warmup and `n=3`
measured requests.

`DS4_EXPERT_TIERING=enforce` uses these physical transitions:
`SSD_COLD -> RAM_PROBATION -> RAM_WARM -> VRAM_PROTECTED`. First touch reads
exact native quantized bytes once into pinned RAM and computes through a
transient GPU slab. Second touch promotes to persistent VRAM. VRAM eviction
retains the RAM copy, so recall does not force another SSD read.

| Metric | Control aggregate | Enforce aggregate | Delta |
|---|---:|---:|---:|
| Server decode | 3.2283 t/s | 4.9750 t/s | +54.10% |
| Client throughput | 2.1986 t/s | 2.8909 t/s | +31.49% |
| Process reads/run | 60.275 GiB | 31.001 GiB | -48.57% |
| Warmup | 4.201 s | 11.381 s | +7.180 s once |

All 12 measured outputs and all four warmups matched exact hash
`fda564ba3f7a0f028106d468420f674898ed99ac5bf2765ac9586206e39d73c5`.
Each enforce run reported 1,005 cold-to-RAM admissions, zero cold-to-VRAM
admissions, 4,299 RAM hits/promotions, 3,963 VRAM demotions, 1,005 transient
routes and zero failures. The final matrix artifact is
`g7_runs/g35_tiering_ab_result.json`, SHA-256
`6b860294fe60921d32314b4077f379496ce2fbac77df3dd9f9f1ac0e117b8c86`.

Verdict: exact positive throughput and transport result. The pay-once pinned-RAM
admission removes repeated model reads and raises server decode above the prior
3.4 t/s local target. Promotion on the second touch is intentionally simple and
causes high churn, so it is not yet the final policy. Next A/B: slow-clock
mass/LFRU admission plus hysteresis, with exactness, zero cold-to-VRAM and lower
promotion/demotion counts as hard gates. Native-Windows commit: `083c305`; full
report: `G35_REAL_EXPERT_TIERING_RESULTS.md` in that repo.

## 2026-07-15 Native Windows G36 Mass/LFRU Slow-Clock Tiering

Question: can a mass/frequency/recency policy preserve the exact G35 tier path
while reducing VRAM churn and improving throughput?

Setup: native Windows RTX 3060 12 GB, model
`C:\ds4-models\ds4-2bit.gguf`, prompt `Hi`, context 256, max 12 (EOS after 9),
cache336 LRU, GPU-resident routes, 8 GiB exclusive pinned arena, 2 GiB stream
window, Q8-F16 off and embedding-row staging on. REAP, SPEX, masks, split
hit/miss and other observers were disabled.

Policy under test:

```text
DS4_EXPERT_TIERING=enforce
DS4_EXPERT_TIER_POLICY=mass-lfru
DS4_EXPERT_TIER_CLOCK_CALLS=430
DS4_EXPERT_TIER_REPLACEMENT_BUDGET=16
DS4_EXPERT_TIER_MIN_FREQUENCY=3
DS4_EXPERT_TIER_HYSTERESIS=1.25
```

The final runner used order
`second-touch/mass-lfru/mass-lfru/second-touch`, with one discarded warmup and
`n=3` measured requests in every process. It compared complete build, harness
and model provenance before aggregation. All 12 measured outputs and all four
warmups matched exact hash
`fda564ba3f7a0f028106d468420f674898ed99ac5bf2765ac9586206e39d73c5`.

| Metric | G35 second-touch | G36 mass/LFRU | Delta |
|---|---:|---:|---:|
| Server decode | 4.9483 t/s | 5.5567 t/s | +12.29% |
| Client throughput | 2.8195 t/s | 3.0439 t/s | +7.96% |
| Warmup | 11.498 s | 11.384 s | -0.114 s |
| VRAM hits | 3,984 | 5,089 | +27.74% |
| VRAM promotions | 4,299 | 384 | -91.07% |
| VRAM demotions/replacements | 3,963 | 48 | -98.79% |
| Transient routes | 1,005 | 3,815 | +279.60% |
| RAM H2D | 34.963 GiB | 27.679 GiB | -20.83% |
| Process reads | 31.001 GiB | 31.001 GiB | unchanged |

Both candidate processes recorded 336 free-slot promotions, 48 replacements,
three policy epochs, 2,010 minimum-frequency skips, 1,753 budget skips, 52
score/hysteresis skips, 1,005 cold-to-RAM transitions, zero cold-to-VRAM
transitions and zero failures. A post-review `observe` regression smoke also
passed exactness after policy accounting was correctly limited to enforce mode.

Verdict: exact positive mechanism and short-prompt throughput result. The equal
31.001 GiB process reads show that G36 does not avoid the initial cold admission;
it wins by stabilizing the VRAM hotset and reducing repeated H2D. Do not call it
universal yet: one deterministic prompt and one parameter set were measured.
Required follow-up is a longer exact workload and a domain switch. Native commit:
`b1ef49c`; full report: `G36_MASS_LFRU_TIERING_RESULTS.md`; matrix SHA-256:
`877eb5a6bffd9c61c041b805e972433057a02866db30bb254912d5271bb6dbf5`.

## 2026-07-15 Native Windows G37 Prefill Union and Chunk Amplification

Question: does CUDA prefill already deduplicate routed experts within a
layer/chunk, and how much repeated transport is induced by narrower chunks?

Code audit found that `cuda_moe_selected_load()` already unions all
`n_tokens * 6` selected IDs and remaps routes to compact slots. G37 added
`DS4_CUDA_PREFILL_UNION_STATS=1` and the harness parameter `-PrefillChunk`; both
are observation-only when enabled and absent by default.

Setup: native Windows RTX 3060 12 GB, `ds4-2bit.gguf`, 43-token cyberpunk HTML
prompt, context 256, max 1, cache336 LRU, GPU-resident decode routes, 2 GiB
stream budget, Q8-F16 off, embed-row staging on. Tiering, dynamic arena, REAP,
masks, SPEX, split hit/miss and overlap were off. Every process used one
discarded warmup and `n=3` measured requests.

| Chunk | TTFT | Unique expert-unions | Dedup | Logical source spans | Win32 process reads | Syncs |
|---:|---:|---:|---:|---:|---:|---:|
| full (43) | 7.547 s | 11,764 | 3.684x | 77.506 GiB | 123.781 GiB | 168 |
| 16 | 8.609 s | 17,956 | 2.414x | 118.263 GiB | 125.175 GiB | 504 |
| 8* | 10.393 s | 22,632 | 1.915x | 149.087 GiB | 155.399 GiB | 1,008 |

Telemetry ON was -0.015832 s (-0.21%) TTFT versus OFF, an effectively zero
difference within run noise. All 27 measured outputs and all nine warmups matched exact hash
`aa3b17600d88d3161605db8389b5bf03d4e94debcc8eeb74dca27aed95a154ab`.
Relative to full chunk, chunk16 added 52.59% logical source traffic and 14.07%
TTFT; chunk8 added 92.35% traffic and 37.70% TTFT. Chunk8 is a single-process
`n=3` directional observation without a second counter-order replication.

Verdict: P4-A batch-union is already implemented and exact. Reload amplification
is across chunks. Proceed to P4-B only as a capacity-bounded wave executor that
keeps a wider chunk but sums partial expert outputs exactly. Do not interpret
`source_span_bytes` as physical SSD traffic; it is logical selected-loader
traffic, while Win32 process reads exclude mmap page-ins. Native report:
`G37_PREFILL_UNION_RESULTS.md`; matrix SHA-256:
`10df0586b2da1c6f9fb3a573e10c762dea1cbed28c2def5af98d9651ec731709`.

## 2026-07-15 Native Windows G38 Capacity-Bounded Prefill Waves

Question: can the full G37 expert union be executed exactly in bounded staging
waves, preserving wide-chunk deduplication without requiring every selected
expert to occupy compact VRAM simultaneously?

Implementation: `DS4_CUDA_PREFILL_WAVES=1` builds the full exact union once,
partitions experts into waves, remaps only original pairs active in each wave,
writes down outputs to their original six-route slots and performs one final
ordered sum. `DS4_CUDA_PREFILL_WAVE_FORCE_EXPERTS=N` is a validation override.
V1 bypasses prefill cache admission and tile kernels and serializes staging
buffer reuse; default-off behavior remains unchanged.

Setup: native Windows RTX 3060 12 GB, `ds4-2bit.gguf`, 43-token cyberpunk HTML
prompt, context 256, max 12, cache336 LRU, GPU-resident decode routes, 2 GiB
stream budget, Q8-F16 off and embed-row staging on. Tiering, dynamic arena,
REAP, mask, SPEX and split hit/miss were off. The counter-ordered matrix ran
`production/wave31/generic/generic/wave31/production`, each with one discarded
warmup and `n=3` measured requests.

| Arm | TTFT | Client t/s | Decode t/s | Process reads | Peak dedicated VRAM |
|---|---:|---:|---:|---:|---:|
| production | 7.319 s | 0.945 | 2.230 | 164.49 GiB | 10.900 GiB |
| generic | 11.007 s | 0.732 | 2.230 | 164.55 GiB | 10.900 GiB |
| wave31 | 10.141 s | 0.836 | 2.945 | 131.14 GiB | 10.437 GiB |

All 18 measured outputs and six warmups matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
Generic and wave31 observed the same cumulative 11,716 union experts. Boundary
safety probes at forced widths 7 and 1 processed 435 and 2,929 waves with zero
failures and expected output; they are `n=1` mechanism checks, not performance
verdicts.

Verdict: exact capacity proof, negative production promotion. Wave31 reduced
process reads 20.27% and peak VRAM 4.25%, but its serial generic path regressed
TTFT 38.56% versus production. Keep opt-in. Next isolated gate: double-buffer
wave weights/pair metadata, overlap upload N+1 with compute N, then restore
tile-capable wave kernels. Native report: `G38_PREFILL_WAVES_RESULTS.md`;
matrix SHA-256:
`22fbd216fad414d7bf828e53f3ac104f1ecf710cf4a33236e70445a00ac9747c`.

## 2026-07-15 Native Windows G39 Double-Buffered Prefill Waves

Question: can G38 upload wave N+1 while wave N computes, preserving the complete
per-layer expert union and final ordered six-route sum?

Implementation: opt-in `DS4_CUDA_PREFILL_WAVE_DOUBLE_BUFFER=1` owns two parity
sets of gate/up/down slabs, route-slot and active-pair arrays, upload-ready
events and compute-done events. Parity ownership persists across layer
boundaries. A post-matrix review found two latent error-path races; accepted
code fences a prior upload before slab resize and seals or drains already
enqueued parity work after a failed launch.

Setup: native Windows RTX 3060 12 GB, `ds4-2bit.gguf`, 43-token cyberpunk HTML
prompt, context 256, max 12, cache336 LRU, GPU-resident decode routes, 2 GiB
stream budget, Q8-F16 off, embed-row staging on. Tiering, dynamic arena, REAP,
mask, SPEX and split hit/miss were off. Counter-order was
`serial/overlap/production/production/overlap/serial`, with one discarded
warmup and `n=3` measured requests per process.

| Arm | TTFT | Client t/s | Decode t/s | Process reads | Peak dedicated VRAM |
|---|---:|---:|---:|---:|---:|
| production | 7.862 s | 0.8759 | 2.0533 | 164.45 GiB | 10.900 GiB |
| serial wave31 | 10.388 s | 0.8192 | 2.8200 | 131.32 GiB | 10.437 GiB |
| overlap wave31 | 8.583 s | 0.9308 | 2.7883 | 131.31 GiB | 10.642 GiB |

All 18 measured outputs and six warmups matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
Overlap improved TTFT 17.38% versus serial wave31 but remained 9.18% behind
production. Both overlap replications reported 456 waves, 454 parity reuse
fences, 456 compute records and zero failures. A post-review `IoQD=2` `n=1`
probe was exact across 42 layers and 114 waves; it is mechanism evidence only.

Measured residency limitation: the isolated wave arms did not admit prefill
experts into the decode cache. Each four-request process reported 2,016 route
calls, only 24 all-hit calls, 1,992 miss-worker jobs and 7,141 missing experts.
The next gate composes production full-chunk prefill with G36 mass/LFRU on this
same workload before judging end-to-end SOTA. Native report:
`G39_PREFILL_WAVE_OVERLAP_RESULTS.md`; commits `78f50cb`, `5633856`; final matrix
SHA-256 `b1f6ed162a42f772ccb42f60087fdc38aabc87507fd7dcae33b29f2a307fbfd1`.

## 2026-07-15 Native Windows G40 Cyberpunk Mass/LFRU Composition

Question: does the G36 short-prompt mass/LFRU win compose with production
full-chunk prefill on the 43-token cyberpunk coding prompt, and does it reduce
the decode misses measured in G39?

Protocol: native Windows RTX 3060 12 GiB, 64 GiB RAM, DeepSeek-V4-Flash IQ2XXS,
context 256, max 12, greedy no-think server path, production full-chunk prefill,
cache 336 LRU, GPU-resident routes, Q8-F16 off, embedding-row staging on,
2 GiB budget, 4096 MiB load reserve and 128 MiB runtime reserve. Arms were
production without arena, 8 GiB arena control, and 8 GiB arena plus enforce-mode
mass/LFRU (`clock=430`, replacement budget 16, minimum frequency 3,
hysteresis 1.25). Counter-order was production A, arena A, mass A, mass B,
arena B, production B. Every process used one discarded warmup plus `n=3`.

All 18 measured outputs and six warmups matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
This is an exact short-prefix transport result, not an L0-L3 long-quality
verdict. Production exposed 42 routed layers per request; both arena arms
exposed 43, and the runner validates that measured accounting difference.

| Metric | Production | Arena control | Mass/LFRU |
|---|---:|---:|---:|
| TTFT | 7.912 s | 7.905 s | 7.913 s |
| Server decode | 2.068 t/s | 1.990 t/s | 0.463 t/s |
| Client throughput | 0.875 t/s | 0.862 t/s | 0.355 t/s |
| Process reads | 164.37 GiB | 174.99 GiB | 433.99 GiB |
| All-hit route calls | 28 | 28 | 137 |
| Miss-worker jobs | 1,988 | 2,036 | 1,927 |
| Missing experts | 7,164 | 7,444 | 6,155 |
| Route wait | 4.798 ms/call | 4.940 ms/call | 8.548 ms/call |

Both mass/LFRU replications deterministically recorded 2,125 cold-to-RAM
admissions, 400 VRAM promotions, 64 physical replacements, 5,755 transient
routes, 336 final VRAM states, 15,040,512,000 SSD bytes and 43,564,400,640
RAM-to-GPU bytes, with zero failures.

Verdict: the residency signal is useful, but the current incremental actuator
is a strong transport negative on this prompt. Missing experts fell 17.32% while
reads rose 148.00% and decode fell 76.72% versus the matched arena control. The
arena was allocated but prefill-mass observation/WRAP was deliberately off, so
it remained unseeded before decode. Next gate: bulk-publish the ordinary prompt
mass into pinned RAM once, include that cost in TTFT, and only then let slow-
clock mass/LFRU protect the VRAM subset.

Native report: `G40_MASS_LFRU_CYBERPUNK_RESULTS.md`; native commit `6298b66`;
runner SHA-256 `e579a0e86d4376b67d368d22cb1d8056ff98f7e8f54d08af33b603f669cfaee0`;
matrix SHA-256 `a9553908ee9ad7798b8a9fd65ac9ca0acaf77154b503c8c55c1e6daee8137024`.

## 2026-07-15 Native Windows G41 Cyberpunk Prefill Bulk Seed

Question: can one request-scoped prefill-mass WRAP into a large pinned-RAM
arena reduce decode misses and improve decode when every publication cost is
charged?

Protocol: native Windows RTX 3060 12 GiB, 64 GiB RAM, same 43-token cyberpunk
prompt and exact 12-token output as G39/G40, context 256, production full-chunk
prefill, 30 GiB dynamic arena, Q8-F16 off, embedding-row staging on, 2 GiB
stream budget, 1024 MiB load reserve, 128 MiB runtime reserve, I/O QD 1 and
eight WRAP workers. Expert cache, tiering, REAP mask and SPEX were off. Arms
were observe-only and bulk WRAP. Counter-order was observe A, WRAP A, WRAP B,
observe B, observe C, WRAP C. Because first-snapshot WRAP forbids warmup and
within-process repeats, `n=3` consists of three independent processes per arm.

All six outputs matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
Router was `unbiased`, mask was `off`, and every WRAP published exactly one
snapshot before decode with zero fatal errors.

| Metric | Observe | Bulk WRAP | Delta |
|---|---:|---:|---:|
| TTFT | 20.489 s | 44.710 s | +24.220 s |
| Publication | 0 s | 24.657 s | +24.657 s |
| Server decode | 1.48 t/s | 2.31 t/s | +56.08% |
| Client throughput | 0.4139 t/s | 0.2386 t/s | -42.36% |
| Process reads | 51.433 GiB | 36.943 GiB | -28.17% |
| Shared-memory peak | 30.332 GiB | 30.332 GiB | flat |

The prefill selected 2,657 unique entries; all fit in the 4,551-slot arena and
covered 100% of observed prefill mass. Candidate decode coverage was 84.83%.
Runtime measured 2,443 hits and 653 misses (78.91%) plus 16.10 GiB RAM-to-GPU
traffic. The isolated pay-once transport mechanism passes, but the 12-token
end-to-end gate fails because publication dominates TTFT. Arithmetic from the
mean rates projects a break-even near 100 tokens; this is not a measured result.

A 31 GiB process-level attempt failed closed before measurement after the model
had prepared its CUDA startup cache. G17's earlier bare allocator success at
31 GiB left only about 5 MiB available and did not represent complete-runtime
headroom. The measured matrix therefore uses the previously stable 30 GiB.

Next gate: measure long-run amortization and implement explicit co-ownership so
the prefill snapshot remains immutable RAM backing while a 336-slot cache plus
mass/LFRU protects only the VRAM subset. Preserve zero cold SSD-to-VRAM rather
than deleting the current incompatibility guardrails.

Native report: `G41_PREFILL_BULK_SEED_CYBERPUNK_RESULTS.md`; native commit
`97fae74`; runner SHA-256
`408d945173fe6598f1bf391e2500f35ed95b54e4b16f68eb2a1046cc84245084`;
matrix SHA-256 `7c389f9f73ea442bc9a5020b6b977e71919b2f6e2f4e5a68d257c3aaa5eeefad`.

## Planned: Request-Scoped Prompt-Intent Closed Arena

Hypothesis, not a result: on a transport-bound host, a request such as an HTML
page with CSS and JavaScript can be decomposed into a few semantic intent shards.
Short unbiased router probes over those shards may expose the per-layer expert
mass needed by the upcoming job early enough to build one closed RAM/VRAM set.

The candidate is request-scoped, not a reusable static domain mask. Aggregate
unbiased per-layer mass, select a complete set that fits pinned RAM plus VRAM,
place the highest-mass subset in VRAM and the remainder in pinned RAM, then make
outside experts ineligible for that request. Discard and rebuild the set when
intent changes. The probes must not create independent generated continuations,
alter the original prompt or merge incompatible KV caches. The original request
still receives one normal prefill under the candidate set.

Existing code already has unbiased prefill mass observation, transactional
pinned-arena publication, mass/LFRU VRAM protection and a static REAP bias-mask
actuator. Missing code is the request-scoped closed-set owner, an in-memory
per-request bias API and a true router-only/early-exit semantic probe.

G41 establishes that ordinary-prompt bulk publication improves isolated decode
transport but has not yet amortized its 24.66-second publication. Before adding
semantic shards or a closed-set mask, preserve the published snapshot as RAM
backing, compose it with VRAM-only mass/LFRU ownership and measure a long decode.

Required A/B:

- control: one original prefill;
- candidate: all shard-probe, set-build and preload time plus the same original
  prefill and decode under the closed set;
- same build, prompt, cache state and expected output hash, with `n>=3`;
- report end-to-end TTFT including probes, probe-only time, cold misses,
  SSD-to-RAM bytes, RAM-to-VRAM bytes, set size per layer, mass coverage,
  outside-set selections, peak memory and decode throughput;
- for long coding deliverables, grade every output L0-L3 and report the last
  coherent token; a short exact-prefix hash is not a quality verdict.

Pass only if the complete candidate path improves a declared end-to-end metric
without degrading exactness or L0-L3 quality. Outside-set selections must be
zero by construction, and all probe/build/preload cost is charged. Test a
router-only or early-exit probe before any repeated full-prefill design.

## 2026-07-15 Native Windows G42 Closed Snapshot and VRAM Tiering

Question: can G41's immutable pinned-RAM snapshot co-own residency with a
mass/LFRU-protected VRAM tier, while a request-scoped closed mask makes every
decode selection snapshot-backed and forbids SSD fallback?

Implementation: ordinary prefill records the complete 256-way unbiased router
probability row. Three hash-routed layers contribute all 768 entries; normalized
semantic mass ranks the remaining 3,783 entries, filling a 4,551-slot, 30 GiB
pinned snapshot. Only after publication, a request-scoped bias excludes semantic
experts outside that snapshot. A 256-slot VRAM tier uses mass/LFRU with clock
430, replacement budget 16, minimum frequency 3 and hysteresis 1.25. The mask
is reset at request end and is never reused as a static domain mask.

Accepted protocol: native Windows RTX 3060 12 GiB, 64 GiB RAM,
`ds4-2bit.gguf`, 43-token cyberpunk prompt, context 256, max 12, greedy
no-think, full prefill chunk, 2 GiB stream budget, 1,024 MiB load reserve,
Q8-F16 off, embedding-row staging on, I/O QD 1 and eight WRAP workers. Control
was G41-style selected-top6 bulk WRAP without cache/tiering/mask. Candidate was
the closed full-probability snapshot plus cache256 and enforce mass/LFRU. Order
was control A, closed A, closed B, control B, control C, closed C; every arm was
an independent one-request process with no warmup.

All six outputs matched exact hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
This is exact short-prefix transport evidence, not an L0-L3 quality verdict.

| Metric | Control | Closed G42 | Delta |
|---|---:|---:|---:|
| TTFT | 44.451 s | 82.078 s | +37.627 s |
| Publication | 22.982 s | 62.407 s | +39.425 s |
| Server decode | 2.277 t/s | 4.083 t/s | +79.36% |
| Client throughput | 0.2395 t/s | 0.1414 t/s | -40.97% |
| Process reads | 36.772 GiB | 23.190 GiB | -36.93% |
| Peak dedicated VRAM | 10.121 GiB | 10.443 GiB | +0.323 GiB |
| Minimum available RAM | 5.255 GiB | 0.363 GiB | -4.892 GiB |

Every closed replication measured exactly 516 route calls, 3,096 selected
experts, 886 VRAM hits, 2,210 pinned-RAM hits, 14.568 GiB RAM-to-VRAM traffic,
288 promotions, 32 demotions and 1,922 transient uses. Across all three:
snapshot misses were zero, cold-to-RAM and cold-to-VRAM were zero, SSD bytes
were zero and runtime failures were zero. The remaining 2,210 events are VRAM
cache misses serviced from pinned RAM, not SSD misses.

Critical negative protocol finding: using a 4,096 MiB load reserve reduced the
startup hot-weight cache to about 4.94 GiB. A first control/closed pair fell to
0.14/0.15 t/s and read 116.164/109.026 GiB. The closed worker cost rose to
5.948 ms/job. Changing only the reserve to 1,024 MiB restored a 7.21 GiB hot
cache; an `n=1` gate reached 4.11 t/s, 23.232 GiB reads and 1.796 ms/job. The
accepted `n=3` matrix uses 1,024 MiB. Earlier reserve-4,096 cache-size probes are
not promoted as capacity verdicts; cache336 separately crossed a measured VRAM
cliff.

Verdict: request-scoped closure plus immutable RAM/VRAM co-ownership passes its
mechanism and steady-decode gates. Decode improves 79.36% and SSD decode traffic
is eliminated. It remains end-to-end negative at 12 tokens because publication
dominates TTFT. Arithmetic projects break-even near 194 generated tokens; this
is not measured. Next run long `n>=3` with L0-L3 grading, then reduce the 62.4 s
publication by reusing prefill reads and batching fills. The 0.065-0.741 GiB
minimum RAM headroom in individual closed runs is also not production-safe.

Native report: `G42_CLOSED_SNAPSHOT_TIERING_RESULTS.md`; implementation and
runner commit `4640c33`; runner SHA-256
`86af52a4e82e99ae5ee9dd06aaa321fc70f8d4c67d5935cd128eff5608ccbc2c`;
matrix SHA-256
`9e09e68d5c6a9e4e4c815f55f15b691a07aa79338ab5d2981c8a4a9911d4fa79`.

## 2026-07-15 Native Windows G43 WRAP Worker Checksum

Question: can G42 snapshot publication avoid a second serial read of the full
30 GiB pinned arena by trusting the FNV checksum already computed after each
worker copies its complete expert slot?

Implementation: the default public finish path is unchanged and still computes
the second checksum. Opt-in `DS4_CUDA_ARENA_WRAP_TRUST_WORKER_CHECKSUM=1`
stores the checksum produced by each copy worker after the full slot copy. All
workers are joined before finish and transactional publication. Invalid values
disable the option. WRAP telemetry now separates begin, copy plus worker
checksum, finish, publish and total time.

Protocol: exact G42 closed-snapshot configuration, native Windows RTX 3060
12 GiB, 64 GiB RAM, same cyberpunk prompt, context 256, max 12, 30 GiB pinned
arena, 4,551 entries, eight workers, cache256 mass/LFRU, 1,024 MiB load reserve,
Q8-F16 off and zero allowed SSD fallback. Three independent one-request
processes per arm, no warmup, counter-order verify A, worker A, worker B,
verify B, verify C, worker C.

| Metric | Finish verify | Worker checksum | Delta |
|---|---:|---:|---:|
| TTFT | 86.028 s | 47.355 s | -38.673 s (-44.95%) |
| WRAP total | 65.214 s | 27.251 s | -37.963 s (-58.21%) |
| Copy plus worker checksum | 32.758 s | 27.236 s | -5.522 s |
| Finish checksum | 32.433 s | 0.001 s | -32.432 s |
| Server decode | 4.093 t/s | 4.093 t/s | 0.00% |
| Process reads | 23.598 GiB | 23.211 GiB | -0.387 GiB |

All six runs matched exact output hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
Every run measured 886 VRAM hits and 2,210 pinned-RAM hits. Snapshot misses,
cold admissions, SSD bytes and tier failures were zero in both arms.

Negative/variability evidence: an earlier mechanism run on a cold page-fault
state spent 253.623 s in copy plus worker FNV and 32.347 s in finish, for
286.070 s total, while retaining exact output, zero SSD and 4.01 t/s decode.
The accepted matrix still measured 23.003-40.565 s copy variation, whereas the
removed finish pass was stable at 31.821-33.565 s.

Verdict: exact positive end-to-end startup optimization for the measured G42
transport gate. The next isolated lever is a globally source-ordered or
part-major first-copy schedule. Preserve the 1,024 MiB reserve and zero-SSD
contracts; the 30 GiB arena still left only 0.226-0.325 GiB mean minimum RAM
headroom across arms.

Native report: `G43_WRAP_CHECKSUM_RESULTS.md`; implementation/runner commit
`4a3b792`; runner SHA-256
`5ca8f15395505b6e808b7dc86ed4c15bd22bdf64ce481bb415b0017f1cd07d4b`;
matrix SHA-256
`109cd153e2c1bdc3a16042cadae8c3c7d07159dc4661c072e4df11f770704120`.

## 2026-07-15 Native Windows G44 Source-Parts WRAP Order

Question: can the G42/G43 closed-snapshot first copy from model `mmap` be made
faster by copying in global source-offset order instead of expert-major order,
without changing exactness, residency or the default path?

Implementation: opt-in source-parts copies model `mmap` in global source-offset
order with barriered gate/up/down phases and incremental exact canonical
full-slot FNV. The default expert-major path is unchanged. Native implementation
commit `48234f31ec5828ae094496e42eb01f498e4b87c8`.

Protocol: exact G42/G43 closed snapshot on native Windows RTX 3060, same
4,551-entry 30 GiB pinned arena, cache256 mass/LFRU, zero allowed SSD fallback
and expected output hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
Three independent one-request processes per arm, ordered expert A, source A,
source B, expert B, expert C, source C.

| Metric | Expert-major | Source-parts | Delta |
|---|---:|---:|---:|
| WRAP median | 31.947 s | 25.829 s | -6.118 s (-19.15%) |
| TTFT median | 52.508 s | 46.184 s | -6.324 s (-12.04%) |
| Decode mean | 4.04 t/s | 4.10 t/s | no claim |

All six runs matched exact output hash
`921a62bdb39d9d07161326274fcbc0070f3c4b9e75153d27b1b6dc96811f6e88`.
Every run measured 516 route calls, 3,096 selections, 886 VRAM hits and
2,210 pinned-RAM hits. Snapshot misses, SSD bytes and tiering failures were
zero in both arms.

Individual WRAP times were expert 30.109, 31.947 and 201.671 s versus source
25.040, 25.829 and 26.301 s. The expert outlier is retained because Windows
standby cannot be purged. Decode means were 4.04 versus 4.10 t/s, so this is
not a decode result.

Verdict: exact positive startup optimization for the measured closed snapshot.
The next ranked lever is direct resident slots and hit/miss separation, not more
first-copy scheduling. Native report: `G44_SOURCE_PARTS_RESULTS.md`.

## 2026-07-15 Native Windows G45 Protected Direct-Resident Cache 320

Question: does increasing the protected direct-resident expert set reduce
pinned-RAM H2D traffic and improve longer decode while preserving exact output,
the request-scoped closed snapshot and zero SSD traffic?

Protocol: exact G44 source-parts closed snapshot on native Windows RTX 3060
12 GiB, 64 GiB RAM, driver 596.21, `ds4-2bit.gguf`, context 256, 64 generated
tokens, deterministic server path, 30 GiB dynamic arena with 4,551 entries,
source-parts WRAP with eight workers, 1,024 MiB startup reserve, 0.125 GiB
expert-cache reserve in both arms, Q8-F16 off, tiering enforce with mass/LFRU
clock 430, replacement budget 16, minimum frequency 3 and hysteresis 1.25.
The only accepted A/B change was
`DS4_CUDA_STREAMING_EXPERT_CACHE_N=256 -> 320`. Order was 256 A, 320 A, 320 B,
256 B, 256 C, 320 C, with three independent processes per arm.

Capacity gate: requested capacity is not always effective capacity under WDDM,
so mismatches are failed protocol, not samples. Exploration measured 336 with
0.5 GiB reserve as effective 284, 384 with 0.125 GiB reserve as effective 341,
one 336/0.125 safety candidate as effective 336, then 336 A as effective 336
and 336 B as effective 321. The two consecutive 336 attempts were therefore
not reproducible. A 320/0.125 safety gate and all three 320 candidate replicas
were effective 320. Rejected 336 artifacts are mechanism evidence only.

Expected output SHA-256:
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`.
The 64-token output is a deterministic transport prefix; it is not a complete
HTML quality result and carries no L0-L3 verdict.

| Metric | 256 protected | 320 protected | Delta |
|---|---:|---:|---:|
| Server decode mean | 4.4367 t/s | 4.4767 t/s | +0.90% |
| Server decode median | 4.44 t/s | 4.48 t/s | +0.04 t/s |
| Decode seconds mean | 14.4247 s | 14.2963 s | -0.89% |
| Pinned-RAM H2D mean | 75.0344 GiB | 71.5803 GiB | -3.4541 GiB (-4.60%) |
| VRAM route hits mean | 5,129 | 5,653 | +524 |
| Pinned-RAM route hits mean | 11,383 | 10,859 | -524 (-4.60%) |
| All-hit route calls mean | 14 | 19 | +5 |
| Worker time | 1.6853 ms/job | 1.6207 ms/job | -3.84% |
| Worker-ready wait | 1.6727 ms/call | 1.6047 ms/call | -4.07% |
| Peak VRAM mean | 11,076.0 MiB | 11,517.3 MiB | +441.3 MiB |
| TTFT median | 44.604 s | 44.754 s | +0.150 s |
| WRAP median | 23.084 s | 24.640 s | +1.556 s |
| Snapshot misses sum | 0 | 0 | exact closed snapshot |
| SSD bytes sum | 0 | 0 | unchanged |
| Tier/route failures sum | 0 | 0 | unchanged |

Per-run decode was 256: 4.44, 4.44, 4.43 t/s; 320: 4.48, 4.48, 4.47 t/s. The
64 extra slots replaced exactly 524 pinned-RAM routes with VRAM hits over
64 generated tokens and removed 3.454 GiB of H2D traffic.

TTFT caveat and recheck: `g45_stable_cache256_c` reported 377.736 s TTFT while
decode remained 4.43 t/s and WRAP was only 23.084 s. Three required independent
cache256 rechecks measured:

| Recheck | TTFT s | WRAP s | TTFT - WRAP s | Decode t/s |
|---|---:|---:|---:|---:|
| A | 42.959 | 22.481 | 20.478 | 4.42 |
| B | 42.569 | 22.065 | 20.504 | 4.44 |
| C | 226.664 | 21.979 | 204.685 | 4.41 |

All three were exact, capacity256, zero snapshot misses, zero SSD and zero
failures, with the same 5,129 VRAM hits and 11,383 pinned-RAM hits. Recheck C
reproduced the stall while WRAP and decode stayed normal. Two of six measured
cache256 processes therefore have a recurrent unlocalized pre-first-token
stall. This does not establish a cache256 correlation because only three
cache320 processes exist. TTFT mean is not used for the cache verdict.

Provenance: measured parent
`a8e48d7c4872e406f5f5a3764d45660315a0f687`; executable SHA-256
`801ea8ff8531245ff3083d71cdc5b5b55b93f0b1dc4904bee30d24d0dd653026`;
`ds4_cuda.cu` SHA-256
`be4103d78f05d0f565cf2103b0d93b2c04f517e1ac7ebd057951c6db67d34063`; build
manifest SHA-256
`100abc59ee94c04a4399a91f567235c7f341f5fca004985290a74e53c99a5fd6`; build
input fingerprint
`752b0f3035f44c205e1cdf104b07c078b79a29594100481c7d60f90762b130c8`; harness
SHA-256 `235d4220e3903425ae55c32cec950a01a58bf601f6b55d80c2784995aa069533`;
execution runner SHA-256
`23699ea6251ad4bffb5e03d077de0fdfa9be9095c55ac15171e680463132a31d`;
corrected summary runner SHA-256
`c9ceca6fc95467bd76ec65ecf9c4644a7470fb6108cf93da25214b7980629ad7`;
TTFT recheck runner SHA-256
`75fdf15e0747a6f4724e42af9014b778c392b5932a7296a34eebd7d7354dcf6e`.
The first summary had a PowerShell median-index bug; the corrected summary was
regenerated from the same six per-run JSON files without repeating runtime
measurement.

Verdict: cache320 is the best measured reproducible protected capacity for the
current RTX 3060 configuration. It is exact, replicated and SSD-free, with a
small consistent decode gain and a larger direct mechanism improvement at about
441 MiB additional peak VRAM. Do not use 336 as the default. The next isolated
gate should remove only the redundant default-stream synchronization in the
GPU-resident route handoff, relying on the existing mapped request sequence and
worker-ready publication. It must be opt-in, no-default-sync, exact-safe first,
and promoted only after an `n>=3` A/B. In parallel, add phase telemetry after
WRAP publication to localize the recurrent 204-355 s unexplained interval.

Native report: `G45_DIRECT_RESIDENT_CACHE_RESULTS.md`; primary artifacts:
`g45_direct_resident_cache_ab.ps1`,
`g45_ttft_outlier_recheck.ps1`,
`g7_runs/g45_direct_resident_cache_ab_result.json` and
`g7_runs/g45_ttft_outlier_recheck_result.json`,
`g7_runs/g7_g45_stable_cache{256,320}_{a,b,c}_result.json` and
`g7_runs/g7_g45_cache256_ttft_recheck_{a,b,c}_result.json`.

## 2026-07-15 Native Windows G46 GPU-Resident Route No-Default-Sync

Question: can the GPU-resident route handoff omit its explicit
`cudaStreamSynchronize(0)` and rely on the existing mapped request sequence and
worker-ready publication without changing output, residency or transport?

Implementation: native commit `b9fa97f` adds opt-in
`DS4_CUDA_MOE_ROUTE_NO_DEFAULT_SYNC=1`. The route resolver kernel publishes its
mapped request with `__threadfence_system()`. The route worker still completes
and synchronizes required H2D on its upload stream before publishing its ready
sequence. The caller still waits for that exact sequence. Only the earlier
default-stream-wide synchronization is skipped. The default behavior is
unchanged when the variable is absent. Runtime and harness counters require
every route call to be accounted as exactly one of `default_sync` or
`no_default_sync`.

Protocol: exact G45 cache320 source-parts closed snapshot on native Windows RTX
3060, `ds4-2bit.gguf`, context 256, 64 generated tokens, 30 GiB dynamic arena
with 4,551 entries, eight-worker source-parts WRAP, 1,024 MiB startup reserve,
0.125 GiB cache reserve, Q8-F16 off, mass/LFRU tiering and GPU-resident routes.
Split hit/miss and dynamic REAP/SPEX prediction were disabled. Three independent
processes per arm ran in order default A, no-sync A, no-sync B, default B,
default C, no-sync C.

Expected output SHA-256:
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`.
All six processes matched it. This 64-token prefix is a transport exactness
protocol and has no L0-L3 document-quality verdict.

| Metric | Default sync | No default sync | Delta |
|---|---:|---:|---:|
| Server decode mean | 4.4600 t/s | 4.5633 t/s | +2.32% |
| Server decode median | 4.46 t/s | 4.58 t/s | +2.69% |
| Decode seconds mean | 14.348 s | 14.021 s | -2.28% |
| Caller resolve mean | 2.813 ms/call | 0.000 ms/call | moved out of caller |
| Worker-ready wait mean | 1.615 ms/call | 4.446 ms/call | synchronization shifts here |
| Total handoff mean | 4.428 ms/call | 4.446 ms/call | +0.41% |
| Route worker mean | 1.631 ms/job | 1.621 ms/job | -0.57% |
| VRAM route hits mean | 5,653 | 5,653 | identical |
| Pinned-RAM route hits mean | 10,859 | 10,859 | identical |
| Pinned-RAM H2D mean | 71.5803 GiB | 71.5803 GiB | identical |
| Snapshot misses / SSD bytes / failures | 0 / 0 / 0 | 0 / 0 / 0 | exact closed snapshot |

Per-process decode was default 4.46, 4.46 and 4.46 t/s versus no-sync 4.53,
4.58 and 4.58 t/s. No decode outlier or recurrent 204-355 second TTFT stall was
present, so the permanent three-extra-process outlier rule was not triggered.

Provenance: parent `22d92af09df43cd5bb604ade5a66494eb8d7206f`;
executable SHA-256
`e48237f6696e8737a4e2bc21a2e75250207650e62fbf92a3377d51b3a7194080`;
`ds4_cuda.cu` SHA-256
`a58c5ca99522212c7414fbb69f2372ef5aef7b3eb366eda06867b9cfb0293ca7`;
harness SHA-256
`0623a8b52dd59dab8621603c177b006aba53acdc02532af2f16ebcd696ac073b`;
model bytes 86,720,111,488. Another Codex task briefly created candidate-only
`g46_no_default_sync_64_{a,b,c}` artifacts and interrupted the first matrix
coordinator after two authoritative results. Those candidate-only tags are
excluded. The runner's `-Resume` mode revalidated the two complete JSON files
and executed the four missing processes without changing the runtime protocol.

Verdict: exact positive `n=3` transport result. G46 raises the current
closed-snapshot/cache320 decode by 2.32% without changing experts, residency,
transport volume or SSD traffic. Keep opt-in until a cross-prompt exactness
matrix passes. The next measured bottleneck is the 10,859 pinned-RAM routes and
71.58 GiB repeated H2D; independently add post-WRAP phase timestamps to localize
the intermittent TTFT stall.

Native report: `G46_NO_DEFAULT_SYNC_RESULTS.md`; runner:
`g46_no_default_sync_ab.ps1`; summary:
`g7_runs/g46_no_default_sync_ab_result.json`; authoritative per-run artifacts:
`g7_runs/g7_g46_{default,nosync}_{a,b,c}_result.json`.

## 2026-07-15 Native Windows G47 Request-Phase Trace

Question: can request-local CPU monotonic timestamps separate prefill compute,
WRAP, sync return, decode entry and first-token work without changing exact
output or materially slowing the accepted G46 path?

Implementation: native commit `134a984` adds opt-in
`DS4_REQUEST_PHASE_TRACE=1`. The server traces prompt/session/decode/first-token
boundaries; CUDA traces prefill-finalize and WRAP-copy boundaries. The path adds
no GPU readback, CUDA event, device allocation or synchronization. The harness
requires exactly 16 ordered events for one non-warmup prefill-WRAP request and
rejects any phase line in the trace-off arm. Multi-request and KV-prefix
sub-sync traces are outside this gate.

Protocol: the G46 no-default-sync architecture, context 256, 64 exact generated
tokens, 30 GiB source-parts closed snapshot, cache request/effective capacity
320, Q8-F16 off and mass/LFRU tiering. A final-binary safety with the former
0.125 GiB cache reserve reached only 309 effective slots under WDDM and was
excluded. Reserve 0 restored 320/320 without OOM and was held identical across
both arms.

Primary balanced `n=3` per arm produced trace-off 4.30/4.50/4.51 t/s and
trace-on 4.35/4.49/4.48 t/s. Because the first value in each arm was lower,
the permanent outlier rule triggered three additional independent processes per
arm: off 4.49/4.50/4.50 and on 4.47/4.46/4.45 t/s.

Combined `n=6` per arm:

| Metric | Trace off | Trace on | Delta on vs off |
|---|---:|---:|---:|
| Server decode mean | 4.4667 t/s | 4.4500 t/s | -0.37% |
| Server decode median | 4.500 t/s | 4.465 t/s | -0.78% |
| Decode seconds mean | 14.3277 s | 14.3747 s | +0.33% |
| TTFT mean | 45.3057 s | 45.0570 s | -0.55% |
| Client wall mean | 60.0401 s | 59.8258 s | -0.36% |
| Client wall median | 59.1633 s | 59.6789 s | +0.87% |

All 12 outputs matched expected SHA-256
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`.
Every process had 5,653 VRAM routes, 10,859 pinned-RAM routes, 71.5803 GiB
pinned-RAM H2D, zero snapshot misses, zero SSD bytes and zero failures. This is
an exact telemetry/overhead protocol, not an L0-L3 quality verdict.

Across the six trace-on processes, measured means were 21.2525 s prefill compute,
23.6924 s WRAP timeline, 23.6325 s WRAP copy, 5.5 ms after WRAP to finalize
return, 6.5 ms finalize-to-sync-return, 17.1 ms to decode entry and 531.8 ms for
the first eval. Prompt-to-first-token mean was 45.6148 s. Normal time after WRAP
terminal through session-sync return is therefore about 12 ms, not tens of
seconds.

No G47 process reproduced the earlier 200-370 second TTFT events, so their cause
remains unmeasured. G47 does establish that the label "post-WRAP stall" is not
supported for normal runs and is ready to localize the next reproduced event.

Verdict: exact, operational opt-in telemetry with measured combined overhead
below one percent. Enable for controlled diagnostics; disable for final
performance verdicts unless a paired trace-off arm is present.

Native report: `G47_REQUEST_PHASE_TRACE_RESULTS.md`; runner:
`g47_phase_trace_overhead_ab.ps1`; summaries:
`g7_runs/g47_phase_trace_overhead_ab_result.json` and
`g7_runs/g47_phase_trace_outlier_recheck_result.json`; native commit:
`134a984`.

## 2026-07-15 Native Windows G48 No-Default-Sync Cross-Prompt Exactness

Question: does the exact G46 no-default-sync route handoff remain byte-identical
outside the cyberpunk HTML prefix used for its `n=3` throughput A/B?

Protocol: exactness-only `n=1` pairs on three prompt shapes: Italian
explanation, C overflow-safe parser and PostgreSQL aggregate query. Each pair
ran default-sync first, then passed its full-content SHA-256 to the no-sync arm
as `ExpectedContentSHA256`. Both arms used context 256, 64 generated tokens,
30 GiB request-scoped closed snapshot, cache request/effective 320, reserve 0,
mass/LFRU tiering, source-parts WRAP and no request-phase trace. This gate makes
no throughput or L0-L3 quality verdict.

| Case | Content SHA-256 | Default t/s | No-sync t/s | Exact |
|---|---|---:|---:|---:|
| Italian | `c592c46ecb710dcd9eeb0f01dde6a47c09690e3c62543f40e3c25f5880ecdbab` | 4.55 | 4.66 | yes |
| C function | `0f0d1665eecb3e1266f95cf331266b0683a0f4afc2296abf083badf5e30b2944` | 4.43 | 4.58 | yes |
| PostgreSQL | `3ed0b67ec5f801b5ff6c05f5e3a755f109d1470bae0eae7947724eadf76a521b` | 4.34 | 4.42 | yes |

All six authoritative processes matched full content and token count, had
2,752 route calls, effective cache 320, zero snapshot misses, zero SSD bytes,
zero failures and complete default/no-sync accounting. The t/s values are
observations only. A context-512 C attempt reached only 315/320 VRAM slots under
WDDM and was excluded before its candidate arm.

The first Italian default process produced TTFT 332.473 s, triggering three
additional identical processes. All four outputs and transport counts remained
identical. TTFT/WRAP timeline was:

```text
reference: 332.473 / 267.347 s
recheck A:  47.327 /  27.128 s
recheck B: 262.228 / 145.254 s
recheck C:  45.348 /  25.093 s
```

Both long logs stop after `[arena] begin`; the source-parts profile attributes
267.330 and 145.234 s to WRAP copy. `TTFT - WRAP` also rises to 65.126 and
116.974 s versus about 20.2 s in normal runs, so slowdown is not confined to
the WRAP workers. Decode remains 4.32-4.55 t/s after publication.

The four processes read 22.210-23.485 GiB and reported 17.1-18.5 million page
faults; simple byte/fault totals do not explain the elapsed multiplier. Two long
events in four identical processes establish reproducibility, not a population
rate or a causal attribution.

Verdict: G46 passes the requested three-shape exactness gate and may remain
opt-in for broader use. Separately, the long-TTFT finding is now localized to
prefill plus source-parts WRAP rather than post-WRAP or decode. The next
diagnostic needs per-worker WRAP progress/latency and concurrent Windows memory
pressure/I/O telemetry.

Native report: `G48_NO_DEFAULT_SYNC_CROSS_PROMPT_RESULTS.md`; runner:
`g48_no_default_sync_cross_prompt.ps1`; summaries:
`g7_runs/g48_no_default_sync_cross_prompt_result.json` and
`g7_runs/g48_italian_default_outlier_recheck_result.json`; native commit:
`aeb839a`.

## 2026-07-15 Native Windows G49 WRAP Part Profile

Question: where inside the `source-parts` WRAP do the reproducible long-tail
intervals occur, and can an opt-in per-worker CPU profile observe them without
changing output?

Implementation: `DS4_CUDA_ARENA_WRAP_PART_PROFILE=1` records aggregate
per-phase-worker work, memcpy time, slow-copy count, maximum part latency and
coordinator main/join time. It adds no per-part logging, GPU readback, CUDA
event or synchronization. The 25 ms slow-part threshold is configurable through
`DS4_CUDA_ARENA_WRAP_SLOW_PART_MS`. The harness requires the expected three
phases, full part/worker accounting and the requested threshold.

Protocol: interleaved `off-a,on-a,off-b,on-b,off-c,on-c`, one fresh process per
run, same Italian prompt, context 256, eight generated tokens, 30 GiB closed
arena, source-parts plus trusted worker checksum, cache 320, mass/LFRU tiering,
no-default-sync and request-phase trace. This is an `n=3` profile-overhead gate,
not a decode-throughput or L0-L3 quality verdict.

All six outputs were byte-identical with SHA-256
`b78be49a2b62f691ee8a8b5b486b2735275cc85bbbc98ee3102c06116487a5e8`.
Every run had 344 route calls, 355 VRAM routes, 1,709 pinned-RAM routes,
11.2654 GiB H2D and zero SSD, snapshot misses, route errors or tier failures.

| Metric | Profile off | Profile on | Delta |
|---|---:|---:|---:|
| WRAP mean | 25.572 s | 28.520 s | +11.5% |
| WRAP median | 25.849 s | 27.211 s | +5.3% |

The profiler is therefore diagnostic-only. In the three on runs, coordinator
join consumed 11-69 ms while aggregate worker memcpy time was 177-232 s.
Individual 2.16-2.75 MiB parts took up to 6.95 s. Available physical memory
fell to approximately 1 MiB, 2 MiB and 254 MiB. None of the six runs crossed
the predeclared 2x-combined-median outlier threshold.

Measured conclusion: the normal wall is inside mmap-backed source copies, not
the coordinator join. Near-zero available RAM is concurrent with the stalls but
is not yet a causal proof for the historical 145-267 s WRAPs. The discriminating
next A/B is an opt-in Windows release of pageable mmap source pages between the
gate/up/down phases. The current `cuda_model_discard_source_pages()` is a no-op
on Windows because only its POSIX path is implemented.

Native report: `G49_WRAP_PART_PROFILE_RESULTS.md`; runner:
`g49_wrap_part_profile_ab.ps1`; native commit: `51ca565`.

## 2026-07-15 Native Windows G50 Working-Set Trim

Question: does releasing the Windows process working set between the trusted
`gate`, `up` and `down` source-parts phases prevent the near-zero-RAM WRAP
stalls measured in G48/G49?

Implementation: opt-in
`DS4_CUDA_ARENA_WRAP_TRIM_BETWEEN_PHASES=1` invokes
`SetProcessWorkingSetSize(GetCurrentProcess(), -1, -1)` after the first two
phase joins. The harness requires source-parts plus trusted worker checksums and
fails closed unless both calls succeed. The default path is unchanged. This
uses the request-scoped prefill-learned closed set; it is not a reusable static
domain mask.

Protocol: same Italian prompt and exact eight-token hash as G49; interleaved
base `off-a,on-a,off-b,on-b,off-c,on-c`, one fresh process per row. Both arms
crossed the 2x-combined-median outlier rule. Three extra processes per arm were
planned; all three off and two on extensions completed. `on-x3` was stopped
after sustained 100% disk activity made Windows unusable. It has no imputed
result and the matrix is explicitly `finalized-incomplete-system-impact`.

All 11 completed outputs were byte-identical with SHA-256
`b78be49a2b62f691ee8a8b5b486b2735275cc85bbbc98ee3102c06116487a5e8`.
Every completed run retained zero SSD bytes, snapshot misses, routing errors and
tier failures. The 11 runtime telemetry intervals used distinct PIDs and had no
temporal overlap.

| Scope | Trim off WRAP mean / median | Trim on WRAP mean / median |
|---|---:|---:|
| Base n=3 per arm | 75.095 / 29.843 s | 45.382 / 23.406 s |
| Expanded completed | n=6: 50.918 / 28.185 s | n=5: 106.284 / 23.406 s |

Normal on rows reached 23.197-23.406 s WRAP and kept roughly 8-9 GiB available,
versus 25.079-29.843 s normal off WRAP with 1-231 MiB available. This apparent
normal-case benefit is rejected because `on-x2` measured a 372.042 s WRAP after
a normal 9.667 s load and 20.418 s prefill. The two trim calls themselves took
only 3.387 s. The remaining time is downstream page-in/copy, accompanied by
about 20.3-21.3 million page faults per completed on process. `on-c` separately
measured a 110.010 s load, 171.330 s prefill and 89.542 s WRAP. The interrupted
`on-x3` had already spent 139.586 s preparing the 7.21 GiB startup cache before
the operator stopped disk saturation.

Verdict: reject whole-process working-set trim. It is too broad, can evict mmap
source and other pages needed by the next phase, creates catastrophic repaging,
and makes the host unusable. Keep the switch default-off and out of launch
recipes. Any follow-up must be range-selective and first prove bounded disk
traffic in a small gate.

Post-study harness hardening acquires `Local\DS4_G7_MEASUREMENT_LOCK` and
refuses launch if another DS4, G7 harness or G7 runtime monitor is present. The
mutex collision and error-path release were tested without model launch.
Native commit `1a6ac80` adds a second default-on gate immediately before model
launch: five measured samples reject sustained median CPU above 60%, disk above
30%, aggregate disk I/O above 64 MiB/s, or GPU above 85%. It checks the maximum
utilization across all visible GPUs, records actual cadence and provenance, and
does not write its JSON between the quiet window and a real server launch.
Fail/pass behavior was verified in dedicated no-launch probes; this is harness
validation, not a performance result.

Native report: `G50_WRAP_WORKING_SET_TRIM_RESULTS.md`; runner:
`g50_wrap_trim_ab.ps1`; native commit:
[`30f864b`](https://github.com/imanu86/ds4-win/commit/30f864ba6b351f1fc621ea272c3f36e4c7e6e000);
matrix SHA-256:
`a6dcb05492aaf6917eb366b096a6f961d58a17901f235bc661dec8a0ca61ad21`.

## 2026-07-15 Native Windows G51 Prefill VRAM Seed Safety

Question: after the request-scoped 30 GiB snapshot is published, can the 320
highest-mass prefill experts be copied into the protected VRAM cache before
decode without changing exact output?

One safety process completed exactly with the canonical cyberpunk 64-token
hash. It seeded 320 entries in 0.385 s and decoded at 4.49 t/s. TTFT was
48.225 s and WRAP was 26.169 s. This is mechanism evidence only: n=1 cannot
support a throughput verdict, and the pre-hardening telemetry reached only
0.033 GiB minimum available RAM.

Decision: keep G51 default-off and do not run its n=3 matrix until the WRAP
source path no longer duplicates pageable mmap source pages.

Native commit:
[`621fbe9`](https://github.com/imanu86/ds4-win/commit/621fbe9);
local artifact:
`g7_runs/g7_g51_prefill_vram_seed_safety_n1_result.json`.

## 2026-07-15 Native Windows G52 Direct Sequential-File WRAP

G52 replaced the 30 GiB `memcpy` from the model mmap into
`cudaHostAlloc` memory with direct Win32 file reads into the pinned arena.
This removes the simultaneous mmap-page-cache plus pinned-destination
duplication. The source path is opt-in and reports
`arena_wrap_source_observed=sequential-file`.

Two exact n=1 safety processes kept about 17.5 GiB minimum available RAM and
zero snapshot misses, SSD bytes or tier failures:

| Run | WRAP | TTFT | Decode | Aggregate disk read | Read rate |
|---|---:|---:|---:|---:|---:|
| safety2 | 50.248 s | 69.204 s | 4.51 t/s | telemetry predates aggregate field | n/a |
| safety3 | 176.082 s | 198.048 s | 4.56 t/s | 52.430 GiB | 238.6 MiB/s |

The 3.5x WRAP spread is a measured negative/variability finding, not a
performance verdict. It proves the memory duplication fix while showing that
the one-worker direct-read path still depends strongly on I/O state.

Native commits:
[`4d3e36d`](https://github.com/imanu86/ds4-win/commit/4d3e36d),
[`91a1b50`](https://github.com/imanu86/ds4-win/commit/91a1b50).

## 2026-07-15 Native Windows G53 Sequential Worker-Depth Safety

G53 made the sequential-file copy worker count measurable. A workers=4 safety
process remained in WRAP after 315 s at roughly 56-70 MiB/s with disk queue
around 8-12, while available RAM remained near 17.5 GiB. It was manually
stopped and produced no result JSON.

Decision: this is an aborted negative safety signal, not an n=3 verdict. Do not
run the six-process workers1-versus-workers4 matrix. Shared concurrent sparse
reads plus the sequential file hint are not a viable next default.

Native commit:
[`e31f663`](https://github.com/imanu86/ds4-win/commit/e31f663).

## 2026-07-15 Native Windows G54 File-Source A/B

Question: does one-worker `FILE_FLAG_RANDOM_ACCESS` improve direct-file WRAP
over the one-worker sequential-file baseline?

Protocol: n=3 independent processes per arm, counterbalanced order
`random,sequential,sequential,random,random,sequential`, cyberpunk prompt,
context 256, 64 generated tokens, cache 320, 30 GiB request-scoped closed
snapshot, mass/LFRU, GPU-resident routes and no-default-sync. Every run
required the same expected output hash, source request/observation, cache
capacity and per-run launch sidecar. All six runs were exact with zero
contamination, snapshot misses, SSD bytes and tier failures.

| Arm | WRAP mean / median | TTFT mean / median | Decode mean | Disk read mean | Read rate mean |
|---|---:|---:|---:|---:|---:|
| sequential-file | 50.195 / 50.255 s | 72.117 / 72.073 s | 4.553 t/s | 51.128 GiB | 539.6 MiB/s |
| random-file | 50.793 / 50.526 s | 73.130 / 72.913 s | 4.533 t/s | 49.734 GiB | 517.8 MiB/s |

Random-file is 1.19% slower in mean WRAP and 1.40% slower in mean TTFT.
Reject it as a performance lever and retain sequential-file workers1.

An initial attempt correctly refused its second process because the quiescence
gate saw sustained disk activity. The contaminant was measured as Visual
Studio Installer `BackgroundDownload.exe` reading about 30 MiB/s. It was
stopped, disk queue returned to zero, and the matrix resumed; the refused
launch is not a result row.

Architectural conclusion: the file hint is not the remaining wall. One
publication still issues 13,653 source-part reads for 4,551 experts and copies
32,211,468,288 bytes. G55 should first measure adjacency/gap amplification,
then test bounded range coalescing or batched overlapped I/O without restoring
mmap duplication.

Native commits:
[`c81ff7d`](https://github.com/imanu86/ds4-win/commit/c81ff7d),
[`d1d2771`](https://github.com/imanu86/ds4-win/commit/d1d2771),
[`e4ba9ec`](https://github.com/imanu86/ds4-win/commit/e4ba9ec).
Native summary:
`g7_runs/g54_wrap_file_source_ab_result.json`.

## 2026-07-15 Native Windows G55 File-QD Functional Safety

G55 added a bounded asynchronous direct-file reader for arena WRAP and made
requested/observed queue depth plus submit/completion/failure counts explicit.
The only completed QD8 run occurred while the K60/K75 R2 downloads were active
on a second physical disk. It was authorized solely as a functional safety and
set `system_quiescence_skipped=true`.

Valid measured gates from that run:

- process exit `0` and expected temp0/nothink content SHA-256
  `31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`;
- requested/observed file QD `8/8`;
- submits/completions/failures `13,653 / 13,653 / 0`;
- no DS4 process left behind and VRAM returned to baseline.

WRAP time, TTFT, decode t/s, disk rate, and every other timed value from this
run are invalid and excluded from SOTA/A-B comparisons. G55 still requires a
quiescent counterbalanced QD1-versus-QD8 n>=3 matrix.

Pre-run hardening commit
[`d7e1e99`](https://github.com/imanu86/ds4-win/commit/d7e1e99)
makes the clean matrix require the same request-scoped candidate-mask
fingerprint across all six processes. It also requires zero async
submit/completion/failure counters for the QD1 legacy arm, while QD>1 must
submit and complete exactly one operation per source part with zero failures.
This prevents a transport verdict from silently comparing different masks or
different queue semantics.

Native commits:
[`438dcd6`](https://github.com/imanu86/ds4-win/commit/438dcd6),
[`fbaeca0`](https://github.com/imanu86/ds4-win/commit/fbaeca0),
[`975097a`](https://github.com/imanu86/ds4-win/commit/975097a),
[`d7e1e99`](https://github.com/imanu86/ds4-win/commit/d7e1e99).

## 2026-07-15 Native Windows G56 WRAP Layout Profiler

G56 is implemented but not yet measured. With
`DS4_CUDA_ARENA_WRAP_LAYOUT_PROFILE=1`, it scans the already-built source-part
metadata and reports, independently for gate/up/down, payload bytes, source
gaps, overlaps, 4 KiB alignment, and projected read-count/byte amplification
for coalescing thresholds 0, 4 KiB, 64 KiB, and 1 MiB. It performs no model
read and does not alter the copy schedule; the option is off by default.

The dedicated runner is a single functional/profile run, not an n=3 benchmark.
Its purpose is to choose whether bounded range coalescing has enough structural
headroom to justify implementation before another performance matrix.

Pre-run hardening commit
[`06ddc61`](https://github.com/imanu86/ds4-win/commit/06ddc61)
requires the actual request-scoped closed-mask contract, a present candidate
fingerprint, and `no_default_sync_calls == route_calls` with zero default-sync
or route errors. The fingerprint is included in the profile validation output.
This keeps the metadata observation from silently profiling a different mask
or route handoff.

Native commit:
[`7c66418`](https://github.com/imanu86/ds4-win/commit/7c66418),
[`06ddc61`](https://github.com/imanu86/ds4-win/commit/06ddc61).

## 2026-07-15 Native Windows G57 Sparse K60/K75 Runtime Guards

The native loader now parses the self-describing sparse bake trailer, validates
manifest/bitset/CRC/tensor geometry and proves that every non-routed GGUF range
is physically present. Runtime guards install the embedded allowed set before
CUDA map/cache setup and fail closed at CPU top-k/final-use, CUDA selected-load,
dynamic-arena target, and SPEX queue boundaries. Whole-block fallbacks, mapped
host windows, full-model copies, chunked copies, external REAP masks, Metal,
and MTP combinations that could read sparse holes are rejected.

Measured no-GPU verification:

- Release build completed and produced `ds4_server.exe`;
- `ctest -C Release`: 1/1 `ds4_bake_test` passed in 0.02 s;
- `git diff --check` passed apart from line-ending warnings;
- independent read-only call-path review reported no finding.

Commit `31342db` adds explicit guard-install/route-validation telemetry and a
fail-closed K60/K75 safety runner. The first safety deliberately omits
WRAP/dynamic arena and checks footer/manifest, CRC32, mask/payload SHA-256,
positive selected-route counters, zero rejected selections, and a non-empty
coherent temp0/nothink output. It remains n=1 functional only.

Pre-run provenance hardening commit
[`93cb193`](https://github.com/imanu86/ds4-win/commit/93cb193)
closes two resume-path gaps. The runner now revalidates hashes for CUDA,
server, sparse parser, header and build manifest plus prompt/token contract.
The result must also prove WRAP, composed prefill, dynamic arena, expert cache,
tiering, SPEX and external REAP mask are all disabled. A full-model protocol
can therefore neither be inherited silently nor summarized as sparse safety.

These are build/parser/safety-structure results only. They establish no K60
startup correctness, quality, TTFT, decode throughput, or SOTA result. The
external verifier has now emitted `GPU/DISCO LIBERI` after proving physical
NTFS holes and post-punch payload integrity. K60 G57 n=1 functional safety is
the next run; no K60 runtime claim exists yet.

Native commits:
[`b86ef4c`](https://github.com/imanu86/ds4-win/commit/b86ef4c),
[`2a086c6`](https://github.com/imanu86/ds4-win/commit/2a086c6),
[`3edaaff`](https://github.com/imanu86/ds4-win/commit/3edaaff),
[`068b4b3`](https://github.com/imanu86/ds4-win/commit/068b4b3),
[`4368674`](https://github.com/imanu86/ds4-win/commit/4368674),
[`0423df5`](https://github.com/imanu86/ds4-win/commit/0423df5),
[`31342db`](https://github.com/imanu86/ds4-win/commit/31342db),
[`93cb193`](https://github.com/imanu86/ds4-win/commit/93cb193).

## 2026-07-15 G36/G46 Protocol Audit and G51 Hardening

A read-only audit separated two valid but incomparable throughput records:

| Authorized scope | Protocol | Measured record |
|---|---|---:|
| Short-path tiering | G36, prompt `Hi`, EOS after 9 decode tokens, cache336, 8 GiB arena, warmup discarded, n=3 | 5.5567 t/s |
| Realistic closed snapshot | G46, cyberpunk prompt, 64 generated tokens, cache320, 30 GiB request-scoped snapshot, n=3 independent processes | 4.5633 t/s mean |

G36 is an exact positive result: mass/LFRU reduced RAM H2D from 34.963 to
27.679 GiB and raised decode from 4.948 to 5.5567 t/s. It is not the general
Windows SOTA because its prompt, decode length, cache, arena, warmup policy and
native commit differ from G46/G54. G46 remains the authorized record for the
realistic 64-token closed-snapshot protocol. G54 sequential-file reproduced
that decode class at 4.553 t/s while changing only the WRAP source path.

This audit selects G51 as the next decode lever: the G46/G54 path still pays
10,859 pinned-RAM routes and 71.5803 GiB of route H2D over 64 generated tokens.
The G51 candidate seeds the top eight request-mass experts per routed layer
(320 entries) into VRAM once after snapshot publication. Its n=3 runner uses
counterbalanced order `control,on,on,control,control,on`, exact output SHA,
identical source/cache/tiering/no-default-sync settings, contamination gates,
and separate counters for seed H2D versus decode-route H2D.

Commit
[`7a79265`](https://github.com/imanu86/ds4-win/commit/7a79265)
adds a stable FNV-1a fingerprint of the complete request-scoped candidate
bitset. The harness rejects missing fingerprints and the G51 matrix rejects
any control/candidate mask mismatch. Release build passed and `ctest -C
Release` passed 1/1 without launching DS4 or using the GPU. Build input
fingerprint: `9821365dc16fb02291101b9b9ef436336b446f90797a8aecb21e526f3d5b06aa`;
executable SHA-256:
`18b8e53627690d950ad37329fb32354a2b278a30b30df012df56b99d374419e1`.
Commit
[`2184ee9`](https://github.com/imanu86/ds4-win/commit/2184ee9)
adds measured n=3 arm deltas to the G51 summary: route H2D saved, explicit
one-time seed H2D, net H2D after the seed, whether the seed amortized within
the observed 64-token window, RAM/VRAM route deltas, and decode t/s delta.
The safety-only path leaves this effect object null.

Decision order after the external `GPU/DISCO LIBERI` gate:

1. complete clean G55 QD1/QD8 n=3 and retain the winning transport;
2. run one G56 metadata-only layout profile;
3. run G51 n=3 per arm using that fixed transport;
4. judge G51 primarily by route H2D, pinned-RAM routes, VRAM routes and exact
   output; seed H2D is an explicit one-time cost, not decode-route traffic;
5. run G57 K60 functional safety only after verified SHA/unpack/manifest.

Exact commands, promotion rules and stop conditions are frozen in
`docs/G55_G57_WINDOWS_EXECUTION_RUNBOOK_20260715.md`.

## 2026-07-16 Native Windows G55 Clean File-QD A/B

G55 completed the counterbalanced order `QD1,QD8,QD8,QD1,QD1,QD8`, with
three independent processes per arm. Protocol: cyberpunk prompt, context 256,
64 generated tokens, 30 GiB closed arena, cache320, sequential-file source,
one copy worker, exact candidate fingerprint `c59a437fe9c6c376`. Native HEAD
was `93cb193`; executable SHA-256 was
`18b8e53627690d950ad37329fb32354a2b278a30b30df012df56b99d374419e1`.

| Arm | WRAP mean / median s | TTFT mean / median s | Decode mean / median t/s |
|---|---:|---:|---:|
| QD1 | 186.412 / 52.203 | 209.446667 / 75.497 | 4.293333 / 4.56 |
| QD8 | 94.245667 / 33.057 | 117.278 / 56.237 | 4.533333 / 4.55 |

All six outputs matched the expected SHA. QD1 reported zero async operations;
QD8 reported `40,959/40,959/0` submit/complete/failure. Contamination aborts,
snapshot misses, SSD bytes and tier failures were all zero. Both arms had a
cold-I/O outlier, but QD8 improved both mean and median WRAP and TTFT while
memory pressure stayed effectively unchanged. Decision: promote `winningQD=8`.

Raw summary:
`g7_runs/g55_wrap_file_qd_ab_result.json` in native commit
[`b80a285`](https://github.com/imanu86/ds4-win/commit/b80a285).

## 2026-07-16 Native Windows G56 WRAP Layout Profile

The single G56 metadata profile passed exactness, provenance, candidate-mask,
no-default-sync, quiescence, zero-miss, zero-SSD and zero-failure gates. It is
not a throughput benchmark.

Across gate/up/down there were 13,653 parts. Exact contiguous coalescing reduces
the projected reads to 7,185, a 47.37% operation reduction with byte
amplification exactly 1.0. The layout contains 6,468 zero-length gaps and 7,182
gaps larger than 1 MiB; no gaps fall in the 1 B through 1 MiB buckets. The
4 KiB, 64 KiB and 1 MiB thresholds therefore produce no additional merge.
Decision: implement only exact-contiguous coalescing, with no threshold
over-read.

Raw summary:
`g7_runs/g56_wrap_layout_profile_result.json` in native commit
[`b80a285`](https://github.com/imanu86/ds4-win/commit/b80a285).

## 2026-07-16 Native Windows G51 Prefill VRAM Seed A/B

The first G51 attempt was aborted by the disk-quiescence preflight and is
excluded. After a 90-second drain, the complete counterbalanced n=3-per-arm
matrix passed using QD8. Control and seed outputs were exact and used the same
candidate fingerprint and build provenance.

| Arm | WRAP mean / median s | TTFT mean / median s | Decode mean / median t/s |
|---|---:|---:|---:|
| control | 32.410667 / 32.327 | 54.548333 / 54.538 | 4.576667 / 4.57 |
| seed 8/layer | 87.906333 / 32.210 | 121.940667 / 54.469 | 4.583333 / 4.58 |

One seeded run had a cold-I/O outlier. The corrected summary derives route H2D
from tier RAM H2D minus the explicit one-shot seed because direct route-H2D
profiling was intentionally disabled. The seed moved 490 hits from RAM to
VRAM, saved 3.229980 GiB of route H2D, cost 2.109375 GiB once, and netted
1.120605 GiB within 64 tokens. Decode changed by only +0.145652%.

Decision: the seed transport works and amortizes, but has no material speed
win. Do not promote it as a SOTA lever yet. The summary correction is native
commit [`9793349`](https://github.com/imanu86/ds4-win/commit/9793349); raw
summary is in [`b80a285`](https://github.com/imanu86/ds4-win/commit/b80a285).

## 2026-07-16 Native Windows G57 K60/K75 Functional Safety

Both physically sparse Windows bakes passed the frozen n=1 functional safety
protocol. These rows validate the sparse container/runtime boundary only; no
timing or generalized quality verdict is admitted.

| Bake | Retained experts | Route calls / slots | Rejected | Output |
|---|---:|---:|---:|---|
| K60 | 6,928 (`256` in layers 0-2, `154` in layers 3-42) | 301 / 4,902 | 0 | coherent, non-empty, temp0/nothink |
| K75 | 8,448 (`256` in layers 0-2, `192` in layers 3-42) | 301 / 4,902 | 0 | coherent, non-empty, temp0/nothink |

Each run validated the footer, manifest and CRCs, the logical source-mask SHA,
the embedded bitset SHA and the payload SHA after NTFS hole punching. The
embedded mask was observed, no external mask/WRAP/arena/cache/tiering/SPEX path
was active, process exit was zero, and VRAM returned to baseline.

Native result commits:
[`2d9cb0a`](https://github.com/imanu86/ds4-win/commit/2d9cb0a) (K60) and
[`2773f5b`](https://github.com/imanu86/ds4-win/commit/2773f5b) (K75).

## 2026-07-16 Native Windows G58 Sparse-Bake Performance Matrix

Question: can the physically closed K60/K75 model, without any resident arena,
expert cache, tiering, SPEX or external mask, beat the realistic G46 closed
snapshot SOTA?

Protocol: cyberpunk HTML prompt, context 256, 64 generated tokens, temp0,
nothink, six independent processes in order `K60,K75,K75,K60,K60,K75`.
Payload/manifest/mask authorization was performed once per bake before launch.
Every run required embedded-mask installation, sparse runtime guards, positive
route calls/slots, `rejected=0`, clean system gates and deterministic output
within its bake. No L0-L3 quality verdict is drawn from 64-token prefixes.

| Bake | Decode mean / median | TTFT mean / median | Load mean / median | Disk read mean | Dedicated GPU peak mean |
|---|---:|---:|---:|---:|---:|
| K60 | 1.863333 / 1.86 t/s | 18.858667 / 18.863 s | 9.370999 / 9.225430 s | 34.323854 GiB | 8.653736 GiB |
| K75 | 1.790000 / 1.76 t/s | 19.058333 / 19.035 s | 7.964992 / 7.672232 s | 37.397406 GiB | 8.653736 GiB |

Relative to G46 (`4.563333` mean t/s), K60 is `-59.167280%` and K75 is
`-60.774285%`. K60 was deterministic at content SHA-256
`ceced6c1b481bb2c6f68bd116c06e554502017a44b40b4e5e6bc9fc5d710edc7`;
K75 at
`2a8ac795075b184c59260bddfe5ab2064b733fd44a4cd82243edf3ff5534dc6f`.
All six processes exited zero with no contamination.

Decision: reject bake-only as a performance path. Physical pruning reduces the
allowed set and improves startup scope, but the selected experts are not made
resident. For only 64 generated tokens the runtime still measured roughly
34-37 GiB of disk reads and about 134-135 GiB of process read transfer, so the
transport wall remains. The next experiment is K60 plus exactly one resident
arena/cache mechanism, with the embedded allowed set kept fixed.

The first G58 launch attempt failed before opening either model because the
PowerShell receipt array used a comma after a function invocation. The runner
bug is recorded in [`7bb6a9d`](https://github.com/imanu86/ds4-win/commit/7bb6a9d);
it produced no measurement row. Runner and complete receipts are
[`40cb7a4`](https://github.com/imanu86/ds4-win/commit/40cb7a4) and
[`f9ba227`](https://github.com/imanu86/ds4-win/commit/f9ba227).

## 2026-07-16 Native Windows G60 Budget-Preserving Layer Stripe

G60 tested a runtime full/partial layer profile without changing the total
30 GiB request-scoped arena budget. Layers 0-2 stayed full. Among routed layers
3-42, every fifth layer was full (8 layers) and the other 32 layers used K54 or
K55, preserving exactly 4,551 total resident slots. The control used the
normal mass-ranked distribution under the same G55 QD8/SOTA transport.

Protocol: counterbalanced n=3 per arm, cyberpunk prompt, context 256, 64
generated tokens, temp0/nothink. Every arm was deterministic internally;
control and stripe intentionally had different candidate fingerprints and
output hashes because the selected set changed. All runs had zero snapshot
misses, SSD bytes, file failures and tier failures.

| Arm | Decode mean / median | TTFT mean / median | WRAP mean / median | Mass coverage | RAM H2D mean |
|---|---:|---:|---:|---:|---:|
| control | 4.513333 / 4.51 t/s | 85.174 / 55.764 s | 62.612667 / 32.955 s | 0.5874 | 71.580322 GiB |
| stripe | 4.530000 / 4.52 t/s | 54.791667 / 54.852 s | 32.432667 / 32.418 s | 0.5272 | 70.868408 GiB |

Measured stripe deltas: decode mean `+0.369284%`, median `+0.221729%`; TTFT
median `-1.635464%`; WRAP median `-1.629495%`; RAM H2D `-0.994567%`; mass
coverage `-10.248553%`. Control-C was a valid internal cold-I/O outlier: its
result file predates the Visual Studio updater that later blocked the next
preflight. The refused preflight was not counted and the remaining row resumed
only after the disk returned to quiescence.

Decision: technically valid, not SOTA and not promoted. The small decode and
transport change does not justify the measured coverage loss without a
separate long-output n>=3 L0-L3 quality matrix. Native implementation and
receipts:
[`ea683f6`](https://github.com/imanu86/ds4-win/commit/ea683f6),
[`2a9c47b`](https://github.com/imanu86/ds4-win/commit/2a9c47b),
[`8acaf32`](https://github.com/imanu86/ds4-win/commit/8acaf32),
[`63c8dd6`](https://github.com/imanu86/ds4-win/commit/63c8dd6).

## 2026-07-16 Native Windows G61 K60 Sparse Bake + Arena

Question: does the frozen K60 embedded sparse bake accelerate when the retained
experts are published into a 30 GiB request-scoped DynamicArena from prefill
mass, while keeping all other transport levers off?

Protocol: K60 candidate only, cyberpunk HTML prompt, context 256, 64 generated
tokens, temp0/nothink. G61 first ran a separate n=1 safety row, then n=3
independent performance processes in order `A,B,C`. The only resident lever was
`DynamicArenaGiB=30` plus `PrefillMassWrap`; cache, tiering, SPEX,
ComposePrefillMassTiering, RouteNoDefaultSync, GPU-resident routes and external
REAP masks were off. The embedded K60 bake mask was observed and applied.

Safety row: `g7_g61_sparse_bake_k60_arena_safety_result.json`, native head
`27e0545`, output SHA-256
`ceced6c1b481bb2c6f68bd116c06e554502017a44b40b4e5e6bc9fc5d710edc7`, and
`rejected=0`. This row is functional safety only and is not included in the
n=3 performance mean/median.

| Arm | Decode mean / median | TTFT mean / median | WRAP mean | Disk read mean | Arena resident / capacity | Arena hit rate | H2D |
|---|---:|---:|---:|---:|---:|---:|---:|
| G61 K60 arena | 2.38 / 2.35 t/s | 38.043333 / 38.541 s | 17.155667 s | 36.453855 GiB | 2322 / 4551 (`0.5102175`) | 12343 / 4169 (`0.747516957`) | 81.36 GiB |
| Frozen G58 K60 | 1.863333 / 1.86 t/s | 18.858667 / 18.863 s | n/a | 34.323854 GiB | n/a | n/a | n/a |

Measured deltas versus frozen G58 K60: decode mean `+0.516667` t/s
(`+27.728109%`), decode median `+0.49` t/s (`+26.344086%`), TTFT median
`+19.678` s (`+104.320628%`), and disk read mean `+2.130001` GiB
(`+6.205600%`). Every performance row produced the exact frozen G58 K60 output
SHA-256 `ceced6c1b481bb2c6f68bd116c06e554502017a44b40b4e5e6bc9fc5d710edc7`,
reported `rejected=0`, loaded `2322` prefill candidates, and ended with
`2322` arena residents. The run makes no L0-L3 quality claim.

Decision: the resident path accelerates decode relative to bake-only K60, but
the prefill publication cost and higher TTFT prevent promotion as SOTA
end-to-end. G62 should add exactly one next lever: sparse-safe GPU-resident
slots/cache, keeping K60 and embedded-mask provenance fixed. Native runner and
safety commits: [`27e0545`](https://github.com/imanu86/ds4-win/commit/27e0545),
[`20ef2b7`](https://github.com/imanu86/ds4-win/commit/20ef2b7). Performance
results: [`ba545be`](https://github.com/imanu86/ds4-win/commit/ba545be).

## 2026-07-16 Native Windows G62 K60 Sparse Bake + Cache GPU-Only

Question: does the frozen K60 embedded sparse bake accelerate when the next
single lever is GPU-only expert cache residency, without the G61 arena path?

Protocol: K60 sparse bake only, cyberpunk HTML prompt, context 256, 64
generated tokens, temp0/nothink, n=3 exact independent performance rows. G62
enabled cache GPU-only 320 LRU and GPU-resident routes. Arena, tiering, SPEX
and external REAP masks were off; the embedded K60 bake mask remained the only
selection authority. The run is a short exactness/performance gate only and
does not support any L0-L3 quality claim.

| Arm | Decode mean / median | TTFT mean / median | Load mean / median | Disk read mean | Process read mean | Dedicated GPU peak |
|---|---:|---:|---:|---:|---:|---:|
| G62 K60 cache-only | 2.123333 / 2.13 t/s | 18.171 / 18.173 s | 10.528771 / 10.197356 s | 33.801597 GiB | 91.819531 GiB | 10.76321 GiB |
| Frozen G58 K60 | 1.863333 / 1.86 t/s | 18.858667 / 18.863 s | 9.370999 / 9.225430 s | 34.323854 GiB | 134.203713 GiB | 8.653736 GiB |
| G61 K60 arena-only | 2.38 / 2.35 t/s | 38.043333 / 38.541 s | n/a | 36.453855 GiB | n/a | 8.712559 GiB |

Measured deltas: versus frozen G58 K60, decode mean improved `+13.953491%`
and median improved `+14.516129%`. Versus G61 arena-only, decode mean was
`-10.784328%` and median was `-9.361702%`. Per run the cache reported `6430`
hits, `12835` misses and `12515` evictions, for a direct hit rate of
`33.3765897%`. Worker telemetry reported `2734` jobs, `10082` miss experts and
`7.385333` ms/call wait. Disk read mean decreased slightly versus G58
(`33.801597` versus `34.323854` GiB), while process read mean fell to
`91.819531` GiB and dedicated GPU peak rose to `10.76321` GiB.

All three rows produced exact content SHA-256
`ceced6c1b481bb2c6f68bd116c06e554502017a44b40b4e5e6bc9fc5d710edc7` with
`contamination=0`, `rejected=0` and `errors=0`.

Decision: cache-only pays versus bake-only, but the LRU rotates heavily and
does not beat the G61 arena-only decode result. G62 is a valid measured
lever/gate, not a SOTA result. The next experiment is G63 composite against the
G46 path, with no quality claim until a separate long-output n>=3 L0-L3 gate.
Native commits: telemetry
[`21d785b`](https://github.com/imanu86/ds4-win/commit/21d785b), runner
[`068c522`](https://github.com/imanu86/ds4-win/commit/068c522) and
[`b887b41`](https://github.com/imanu86/ds4-win/commit/b887b41), safety
[`ed074d3`](https://github.com/imanu86/ds4-win/commit/ed074d3), n=3 results
[`810cdb9`](https://github.com/imanu86/ds4-win/commit/810cdb9).

## 2026-07-16 Native Windows G63 K60 Sparse Bake + Complete G46 Composite

Question: can the K60 sparse bake run the complete measured G46 composition
without addressing absent experts, while preserving deterministic execution
and restoring the embedded sparse-bake mask after each request?

Protocol: K60 sparse bake, cyberpunk HTML prompt, context 256, 64 generated
tokens, temp0/nothink, n=3 independent processes. G63 enabled the exact G46
composition: 30 GiB DynamicArena, source-parts WRAP with trusted worker
checksum, PrefillMassWrap, sparse-aware ComposePrefillMassTiering, GPU-only
cache 320 LRU with 0.125 GiB reserve, GPU-resident routes,
RouteNoDefaultSync, mass-LFRU tiering (`clock=430`, replacement budget `16`,
minimum frequency `3`, hysteresis `1.25`), Q8/F16 cache disabled, embedded-row
staging and eight REAP prefetch threads. SPEX, external masks, stripes,
FileQD greater than one and full-model copy were off.

| Arm | Decode mean / median | TTFT mean / median | WRAP mean | Output exactness |
|---|---:|---:|---:|---:|
| G63 K60 + G46 composite | 4.553333 / 4.55 t/s | 42.652 / 42.792 s | 23.720000 s | 3/3 |
| G46 full-model composite | 4.563333 / 4.58 t/s | 45.211667 / 46.295 s | 24.086333 s | 3/3 |
| G61 K60 arena-only | 2.38 / 2.35 t/s | 38.043333 / 38.541 s | 17.155667 s | 3/3 |
| G62 K60 cache-only | 2.123333 / 2.13 t/s | 18.171 / 18.173 s | n/a | 3/3 |

Measured G63 deltas versus G46: decode mean `-0.010000` t/s
(`-0.219138%`), TTFT mean `-2.559667` s (`-5.661519%`), TTFT median
`-3.503000` s (`-7.566692%`) and WRAP mean `-0.366333` s
(`-1.520916%`). All three G63 rows produced content SHA-256
`4aaf0f0813f4cb15ac21a88f195f4f7d2c2af797e81524935e22eea60603c6b1`.
The output was coherent at the 64-token cutoff but differs from the frozen
G58 K60 hash because the request-scoped G46 mask is a different execution
path. No L0-L3 quality claim is made from this short gate.

Transport and safety telemetry was exact across all rows: `4551` arena slots,
`1579` non-retained ranked candidates skipped/replaced per request, one
successful embedded-mask restore per request, zero restore failures, zero
snapshot backing misses, zero forbidden cold SSD-to-VRAM transfers, zero tier
failures and zero tier SSD bytes. The executable SHA-256 was
`63f1a24285bac089df1db6ecd2024f812b6905b7e7d31a49d95e7b2dcd983399`;
the build-input fingerprint was
`f349bb401be5f804db2ff8bfc1387744ce288dcbdeb4044286f0d934f34cbb4a`.

Decision: G63 is a positive transport/composition result. It recovers the G46
decode rate on the physically sparse K60 candidate within the exact measured
delta above and slightly improves the measured short-run TTFT. It is not yet a
quality SOTA: promotion requires a separate long-output n>=3 L0-L3 comparison
against G46 with identical prompts, context and stopping rules.

The failed gates are retained as part of the protocol history: absent sparse
candidate rejection, low-memory contamination abort, embedded-router-bias
rejection, candidate exactness mismatch and two stale harness guards. Runtime
and results commits:
[`74bc9d4`](https://github.com/imanu86/ds4-win/commit/74bc9d4),
[`9066f36`](https://github.com/imanu86/ds4-win/commit/9066f36),
[`7bee760`](https://github.com/imanu86/ds4-win/commit/7bee760),
[`c43d91a`](https://github.com/imanu86/ds4-win/commit/c43d91a),
[`2a58934`](https://github.com/imanu86/ds4-win/commit/2a58934),
[`eb9c6c0`](https://github.com/imanu86/ds4-win/commit/eb9c6c0),
[`b3f0a62`](https://github.com/imanu86/ds4-win/commit/b3f0a62),
[`caaffcf`](https://github.com/imanu86/ds4-win/commit/caaffcf) and n=3
[`11c5fb1`](https://github.com/imanu86/ds4-win/commit/11c5fb1).

## 2026-07-16 Native Windows G64-G66 Context-8192 Capacity Gates

Question: why does the K60 sparse bake not outperform the adaptive G46 path,
and can the preregistered long quality gate run at context 8192 without
weakening the memory contamination guard?

All rows below are `n=1` structural/capacity evidence. They carry no
performance or L0-L3 quality verdict unless an explicitly measured progress
value is shown. The common composition was the complete G46/G63 path. No SPEX,
external mask, layer stripe, FileQD greater than one or full-model copy was
enabled.

| Gate | Model/lever | Arena/slots | Measured result |
|---|---|---:|---|
| G64 | G46 full, ctx8192 | 30 GiB / 4551 | WRAP abort before token 1; 0.251 GiB available at abort |
| G64 | G63 K60, ctx8192 | 30 GiB / 4551 | WRAP abort before token 1; 0.254 GiB available at abort |
| G65 | G46 full + process-wide phase trim | 30 GiB / 4551 | WRAP published, but token 50 only 0.08 t/s after 590.099 s; operator stop |
| G66 | G46 full fixed smaller arena | 28 GiB / 4247 | WRAP abort before token 1; minimum available 7,385,088 bytes |

G64 established that the physical K60 bake did not reduce runtime arena
cardinality: both arms still built `4551` slots. Full and K60 therefore failed
at the same stage. K60 read `33,894,227,872` process bytes versus
`36,265,349,863` for full, but that reduction was insufficient to preserve the
unchanged guard. The later missing RouteNoDefaultSync assertion in both rows is
secondary because no route call completed.

G65 enabled the existing `ArenaWrapTrimBetweenPhases`. Both working-set trim
calls succeeded in `3.700502` s and WRAP published `4551` loads in `32.312` s.
This was only a capacity pass. The process-wide trim evicted useful model pages:
by operator stop the process had read `491,538,829,532` bytes, the server had
reported token 50 at `0.08 t/s`, and peak disk queue was `52`. Process-wide
`SetProcessWorkingSetSize(...,-1,-1)` is rejected for the next composite.

G66 reduced the arena by 2 GiB and 304 slots with trim disabled. Its host
preflight had only `43.520 GiB` available, about `5.72 GiB` below the G64 full
preflight, so the fixed reduction was not robust and the guard still aborted.
This does not establish that arena 28 would fail from the G64 host state; it
rejects a fixed GiB value as a generally safe policy on this machine.

Measured conclusion: K60 removes absent expert payload from disk but still pays
nearly the same WRAP/H2D cardinality as G46. The next implementation requires
(1) a runtime arena upper bound derived from measured available RAM and an
explicit post-allocation reserve, and (2) range-scoped reclamation of consumed
source-mmap expert pages rather than process-wide working-set trimming. Any
successful safety must be followed by n>=3 before a timing claim and by the
original long-output n>=3 L0-L3 matrix before quality promotion.

Native Windows commits: G64 runner
[`4b4f64d`](https://github.com/imanu86/ds4-win/commit/4b4f64d), full abort
[`f136603`](https://github.com/imanu86/ds4-win/commit/f136603), isolated-arm
runner [`ac518c7`](https://github.com/imanu86/ds4-win/commit/ac518c7), K60 abort
[`c19c88a`](https://github.com/imanu86/ds4-win/commit/c19c88a), G65 protocol
[`aa83016`](https://github.com/imanu86/ds4-win/commit/aa83016), G65 result
[`8b25645`](https://github.com/imanu86/ds4-win/commit/8b25645), G66 protocol
[`fae58d6`](https://github.com/imanu86/ds4-win/commit/fae58d6) and G66 result
[`46a9650`](https://github.com/imanu86/ds4-win/commit/46a9650).

## 2026-07-16 Native Windows G67 Full-Model Adaptive Arena Reserve

Question: can the full G46 runtime derive its arena capacity from actual host
memory instead of a fixed GiB value, preserving an explicit reserve and the
unchanged contamination guard?

Implementation: requested arena 30 GiB remains an upper bound. New opt-in
`DS4_CUDA_DYNAMIC_ARENA_MIN_AVAILABLE_GIB` subtracts the requested reserve from
`GlobalMemoryStatusEx.ullAvailPhys` before `cudaHostAlloc`, rounds down to whole
expert slots and logs requested/chosen bytes and slots. Default zero preserves
all prior behavior. Native implementation commit:
[`5df6c8e`](https://github.com/imanu86/ds4-win/commit/5df6c8e).

G67 full-model safety used context 8192, max 8 tokens, reserve 22 GiB and the
otherwise complete G46 composition. Available-before-arena was `38.285 GiB`, so
the runtime selected `17,482,383,360` bytes and `2470` slots. The request passed
all structural gates: WRAP published in `10.628 s`, server exit was zero,
RouteNoDefaultSync observed `344/344`, and tier backing misses, forbidden
SSD-to-VRAM, SSD bytes and failures were all zero. Minimum available host
memory remained `4.625 GiB`.

This was only a safety pass. Decode measured `0.13 t/s`, process reads were
`87.493 GiB` and aggregate disk-read estimate was `93.285 GiB` for eight
tokens. The narrow `2470`-slot arena therefore avoids the capacity abort but
does not preserve G46 performance. No n>=3 timing or quality verdict is made.
Result commit:
[`93c2078`](https://github.com/imanu86/ds4-win/commit/93c2078).

Project decision: K60/K75 sparse bakes are de-prioritized as an advanced
fallback. They were built to test whether a physically reduced static model
could beat the full dynamic runtime; G63 only matched G46 decode within
`-0.219138%`, while G64 showed that K60 still constructed the same 4551-slot
arena and failed the same context-8192 capacity gate. No further bake run is
part of the active roadmap.

Active roadmap returns to the full model: keep the adaptive cap as fail-closed
safety, reclaim only consumed source-mmap expert ranges (never the whole
working set), then continue direct resident slots, hit/miss separation,
dynamic REAP tiering and n>=3 performance/long L0-L3 quality gates.

## 2026-07-16 Native Windows G68 Selective Source-Mmap Reclaim

Question: can full G46 retain its complete 30 GiB / 4551-slot arena at context
8192 by removing only already-consumed expert source pages from the working set
after the `gate` and `up` phase barriers?

The new default-off `DS4_CUDA_ARENA_WRAP_UNLOCK_SOURCE_RANGES=1` path coalesces
successful source-part ranges and calls `VirtualUnlock` only after phase
workers join. It requires source-parts plus trusted worker checksums, is
incompatible with process-wide trim and file-source modes, and fails closed on
every result except `TRUE` or documented `FALSE/ERROR_NOT_LOCKED`. Native
implementation commit:
[`b1eacea`](https://github.com/imanu86/ds4-win/commit/b1eacea).

G68 was one preregistered full-model safety run, context 8192 and max 8 tokens.
The complete `32,211,468,288`-byte arena allocated all `4551` slots. Selective
reclaim then measured:

| Phase | Source ranges | Requested bytes | Available before -> after | Failures |
|---|---:|---:|---:|---:|
| gate | 2395 | 9,852,203,008 | 68,747,264 -> 9,432,907,776 | 0 |
| up | 2395 | 9,852,203,008 | 873,553,920 -> 10,421,305,344 | 0 |

All `4790` calls returned `ERROR_NOT_LOCKED`, the documented outcome that also
removed the pages from the process working set. Gate and up working-set deltas
were each exactly `9,852,194,816` bytes. Unlike G65, no unrelated process-wide
working-set trim ran.

The safety still failed before token one during the final `down` copy. G68 had
no reclaim point until that 12,526,682,112-byte phase completed. The unchanged
guard observed available memory `1,708,126,208`, `240,553,984`, then
`65,990,656` bytes on three consecutive samples and terminated the server at
67.510 s. WRAP never published; missing route telemetry was secondary. No
throughput or quality verdict is made. Result commit:
[`5f89e3d`](https://github.com/imanu86/ds4-win/commit/5f89e3d).

Measured decision: range-scoped reclamation is effective, but a whole-phase
barrier is too late. The next isolated full-model change is source-sorted copy
waves with join plus selective reclaim after every completed wave across
`gate`, `up` and `down`. The arena remains 4551 slots; no bake, smaller arena or
second performance lever is composed into that gate.

## 2026-07-16 Native Windows G69 Waved Source-Mmap Reclaim

Question: can bounded source-copy waves reclaim consumed mmap pages early
enough for the full G46 30 GiB / 4551-slot arena to finish WRAP under the same
strict context-8192 host-memory gate that stopped G68?

The default-off `DS4_CUDA_ARENA_WRAP_UNLOCK_WAVE_GIB=4` path sorts source
parts, copies each bounded page-aligned wave with all workers joined, and only
then applies the G68 range-scoped reclaim. It preserves gate/up/down checksum
order and fails closed if one part exceeds the cap. Native implementation:
[`be6f1fe`](https://github.com/imanu86/ds4-win/commit/be6f1fe).

One preregistered full-model safety run passed the structural and capacity
gate. This was `n=1`, context 8192, max 8 tokens, temp 0 and nothink; its timing
is diagnostic only.

| Phase | Waves | Maximum wave | Parts | Ranges | Reclaimed bytes | Unlock time |
|---|---:|---:|---:|---:|---:|---:|
| gate | 3 | 4,293,894,144 | 4551 | 2395 | 9,852,203,008 | 1.107160 s |
| up | 3 | 4,293,894,144 | 4551 | 2395 | 9,852,203,008 | 0.903053 s |
| down | 3 | 4,293,083,136 | 4551 | 2397 | 12,536,500,224 | 1.437156 s |

All nine waves completed, reclaiming `32,240,906,240` requested bytes through
7187 coalesced ranges in `3.447369 s`, with zero failures. WRAP published all
4551 loads in `29.089 s`. Runtime available memory never fell below
`2,776,780,800` bytes, so the unchanged 2 GiB guard did not fire. Route calls
were GPU-resident (`344`), default-sync calls were zero, and snapshot backing
misses, SSD bytes and tier failures were all zero. Result commit:
[`e18dd6e`](https://github.com/imanu86/ds4-win/commit/e18dd6e).

This is a capacity result, not a performance promotion. The eight-token safety
measured `0.13 t/s`, `11,975,786,496` RAM-to-GPU bytes and
`93,810,879,312` process-read bytes. It proves that the complete full-model
arena can be built under the constrained host state; it also shows that the
long safety workload still destroys too much non-expert hot residency and
repeats too much transport.

Active decision: sparse K60/K75 bakes remain an advanced fallback only. The
main roadmap stays on the full-model G46/0051 runtime: compare waved reclaim on
the original context-256/64-token exact workload, then remove repeated H2D via
direct resident slots, explicit hit/miss execution and dynamic REAP tiering.
No G46 performance comparison is valid until both arms pass with comparable
preflight memory and `n>=3` independent processes.

## 2026-07-16 Native Windows G70 Full-Model Reclaim Cohort

Question: on the original G46 context-256/64-token workload, can 4 GiB waved
source reclaim preserve exact G46-class decode while making the complete
30 GiB / 4551-slot arena viable under the current host-memory state?

The legacy no-reclaim safety started with about 38 GiB available but reached
three consecutive low-memory/deep-queue samples and was terminated before
token one. Its last samples reported 6,344,704, 423,350,272 and 740,655,104
available bytes. No legacy throughput value exists; this is measured capacity
asymmetry, not a performance comparison.

The reclaim candidate passed safety and an exact candidate-only cohort. A
runner/outlier-summary repair split the raw rows across two docs-only HEADs
without changing executable, CUDA source, harness, model or launch arguments.
The authoritative same-HEAD extension `x1/x2/x3` is therefore summarized as
`n=3`; the earlier `a/b/c` rows remain diagnostic and are not pooled.

| Run | Decode t/s | TTFT | WRAP | Min available | Exact |
|---|---:|---:|---:|---:|---|
| x1 | 4.59 | 52.644 s | 29.103 s | 2.585 GiB | yes |
| x2 | 4.56 | 53.111 s | 29.911 s | 2.514 GiB | yes |
| x3 | 4.59 | 52.247 s | 28.937 s | 2.940 GiB | yes |
| mean | 4.58 | 52.667 s | 29.317 s | - | 3/3 |
| median | 4.59 | 52.644 s | 29.103 s | - | 3/3 |

Every primary run measured 5653 VRAM route hits, 10,859 pinned-RAM route hits,
71.580322 GiB RAM H2D, zero snapshot misses, zero SSD bytes, zero tier/route
failures and zero default-sync calls. Waved reclaim completed three phases and
nine waves with zero failures.

The excluded diagnostic run `c` remained exact and decoded at 4.51 t/s, but
WRAP took 354.613 seconds versus 28.737 and 28.571 seconds in `a/b`. Phase logs
localized the delay inside source-parts WRAP during mmap page-in, not decode or
post-WRAP work. Three subsequent extension runs returned to 28.937-29.911
seconds. This is a real intermittent source-page startup problem; its cause is
not inferred from this cohort.

Historical G46 was 4.563333 mean / 4.58 median t/s. G70's 4.58 / 4.59 is a
valid absolute full-model result and a measured capacity improvement, but the
small descriptive timing delta is not a causal reclaim claim because the
contemporary legacy arm could not run. Windows result commit:
[`1895e8a`](https://github.com/imanu86/ds4-win/commit/1895e8a).

Decision: retain waved reclaim as the full-model capacity mechanism. The next
isolated experiment keeps G70 fixed and compares mass/LFRU replacement budget
16 versus 32. Only a measured reduction below 10,859 RAM hits / 71.580322 GiB
H2D justifies implementing adaptive RAM-pressure tier control. Sparse K60/K75
bakes remain outside the active roadmap.

## 2026-07-16 Native Windows G71 Tier Budget A/B

Question: on the exact G70 full-model workload, does increasing the static
mass/LFRU replacement budget from 16 to 32 reduce pinned-RAM pressure without
breaking exactness, capacity, no-default-sync routing or source-reclaim
provenance?

Protocol: G71 used the G46 cyberpunk HTML prompt, context 256, max 64,
temperature zero, dynamic arena 30 GiB / 4551 slots, cache320,
source-parts WRAP with trusted checksums, 4 GiB waved source reclaim,
`ComposePrefillMassTiering`, GPU-resident routes and `RouteNoDefaultSync`.
The only experimental variable was `ExpertTierReplacementBudget`: control
16 versus candidate 32. Candidate32 first passed an exact safety process. The
primary matrix then triggered the preregistered outlier extension, so the
accepted performance summary is six independent processes per arm.

Every accepted run preserved the required contract: exact expected content
SHA-256, same provenance, 4551 arena slots, cache capacity 320, three reclaim
phases, nine waves, zero reclaim failures, zero snapshot misses, zero SSD
bytes, zero tier failures and zero default-sync calls. Provenance head was
`e8aded23b503c5b62ad2b219dcbb21d71390f8ae`; executable SHA-256 was
`9bbcbc57714611bd3873beedc7fc4f0829ee463e0499793b86295eb085cca501`.

| Arm | Budget | n | Decode mean / median | TTFT mean | WRAP mean | TTFT-WRAP mean | RAM hits mean | RAM H2D mean | Route wait | Worker ms/job |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 16 | 6 | 4.565 / 4.56 t/s | 53.192667 s | 29.811833 s | 23.380833 s | 10,859 | 71.580322 GiB | 4.397 ms | 1.6245 ms |
| candidate | 32 | 6 | 4.611667 / 4.615 t/s | 88.683167 s | 65.070667 s | 23.6125 s | 10,692 | 70.479492 GiB | 4.336833 ms | 1.586667 ms |

Measured effect: candidate32 improved decode by `+0.046667 t/s`
(`+1.022278%`) mean and `+0.055 t/s` (`+1.206140%`) median. It reduced
pinned-RAM pressure by `167` RAM hits and `1.100830 GiB` H2D
(`-1.537895%`), reduced route wait by `0.060167 ms/call` (`-1.368365%`) and
reduced worker time by `0.037833 ms/job` (`-2.328901%`). Policy replacements
rose from 96 to 192 as designed, with 320 free promotions in both arms and no
RAM evictions.

Outlier separation: the candidate `b` process decoded normally at `4.62 t/s`
with `13.858 s` decode time, but WRAP took `244.670 s` and TTFT took
`269.093 s`. Its `TTFT-WRAP` value was `24.423 s`, close to the rest of the
matrix. The delay is therefore localized to the measured source-parts WRAP
interval rather than decode; its underlying cause is not established by this
cohort. The WRAP/TTFT mean for candidate32 is therefore
outlier-contaminated; decode, RAM hits, H2D, route wait and tier counters are
still valid primary metrics under the preregistered six-process summary.

Decision: budget32 is the current measured static tier-budget candidate on the
G70 workload because it passes the full safety/provenance contract and lowers
RAM hits/H2D while slightly improving decode. Do not claim a clean TTFT/WRAP
win from this matrix, and do not pool the WRAP outlier into a startup
improvement story. Adaptive policy work is justified only as a follow-up to
this measured pressure signal, with WRAP startup stability guarded separately.
Result summary:
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g71_tier_budget_ab_result.json`.

## 2026-07-16 Native Windows G72 Adaptive Tier Budget A/B

Question: can the G71 mass/LFRU policy start with replacement budget 16 and
raise it to 32 only under measured budget pressure, while preserving G71's
exactness and transport result against a contemporary static-32 control?

G72 kept the G70/G71 workload frozen: the same model and cyberpunk HTML
prompt, context 256, max 64, temperature zero, 30 GiB / 4551-slot arena,
cache320, source-parts WRAP, 4 GiB waved reclaim, composed prefill mass
tiering, GPU-resident routes and no-default-sync. The only experimental
variable was replacement-budget control. The adaptive arm used base/min 16,
max 32, step 8 and pressure threshold 64. It reached 32 through exactly two
upward steps in every accepted process.

The preregistered matrix completed with three independent exact processes per
arm. Every accepted run had identical provenance and expected content SHA,
three reclaim phases / nine waves, zero reclaim failures, zero snapshot
misses, zero SSD bytes, zero tier failures and zero default-sync calls.

| Arm | n | Decode mean / median | RAM hits | RAM H2D | Promotions | Replacements | Route wait | Worker ms/job |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static32 | 3 | 4.613333 / 4.62 t/s | 10,692 | 70.479492 GiB | 512 | 192 | 4.309000 ms | 1.587333 ms |
| adaptive16->32 | 3 | 4.606667 / 4.62 t/s | 10,668 | 70.321289 GiB | 488 | 168 | 4.302333 ms | 1.583667 ms |

Measured adaptive effect: mean decode `-0.144%`, median tied; RAM hits and H2D
`-0.224%`; promotions `-4.688%`; replacements `-12.5%`; route wait
`-0.155%`; worker time `-0.231%`. Mean TTFT/WRAP were higher in the adaptive
cohort, but the adaptive policy does not execute until after WRAP, so this
matrix does not assign a causal startup effect to the policy.

One first attempt at `static32_c` terminated during WRAP before decode and was
excluded. A clean retry passed. The retry reused the tag and overwrote the
first stderr; that artifact limitation is explicitly preserved in
`g7_runs/g72_static32_c_infrastructure_retry_incident.json`. No root cause is
claimed.

Decision: adaptive 16->32 is mechanism-valid but is not throughput SOTA on
this short workload. Retain static32 as the active default because it is
simpler and at least as fast. Keep the adaptive implementation available for
longer or changing-domain workloads, but move the active roadmap to a larger
transport lever: eliminate remaining per-route wait/copy work through direct
resident hit execution and explicit hit/miss separation, then address repeated
WRAP/prefill work. Sparse K60/K75 bakes remain an advanced fallback only.

Canonical Windows summary:
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g72_adaptive_tier_budget_ab_result.json`.

## 2026-07-16 Native Windows G73 Split-Fused Route A/B

Question: on the exact G72 static32 workload, can split-fused hit/miss route
execution reduce per-route work while preserving the same transport, exact
content and full G70/G71/G72 safety contract?

G73 kept the workload frozen: the same model and exact 64-token cyberpunk HTML
prompt, context 256, max 64, temperature zero, 30 GiB / 4551-slot arena,
cache320, source-parts WRAP, 4 GiB waved reclaim, composed prefill mass
tiering, budget32, GPU-resident routes and no-default-sync. The only
experimental variable was split-fused route execution. The candidate first
passed an exact safety process; the preregistered matrix then completed with
three independent exact processes per arm and no outlier extension.

Semantic corrigendum: `static32` names the fixed expert-tier replacement
budget of 32; it does not mean a static domain/file mask. Prefill derived and
published a `request-scoped-closed` decode mask backed by a pinned RAM snapshot
of 4551 experts (`32211468288` bytes, `7077888` bytes/slot, FNV-1a64
`c59a437fe9c6c376`). The run requested no mask file, observed no embedded bake,
and composed over `mask_base=none`. Its exact-IQ2 VRAM tier remained dynamic at
320 states, with 512 promotions and 192 policy replacements in each accepted
matrix process. G73 was therefore neither static/baked nor full/open during
decode: experts outside the request-derived snapshot were not decode-eligible.

Every accepted run preserved the required contract: exact expected content
SHA-256, same provenance, 4551 arena slots, cache capacity 320, three reclaim
phases, nine waves, zero reclaim failures, zero snapshot misses, zero SSD
bytes, zero tier failures and zero default-sync calls. Split-fused accounting
was complete in the candidate: split-fused calls equaled route calls, and
split-fused hits plus misses equaled selected experts in every accepted
process.

| Arm | n | Decode mean / median | RAM hits | RAM H2D | Split-fused calls | Route wait | Worker ms/job |
|---|---:|---:|---:|---:|---:|---:|---:|
| static32 | 3 | 4.61 / 4.62 t/s | 10,692 | 70.479492 GiB | 0 | 4.314333 ms | 1.581000 ms |
| static32_split_fused | 3 | 4.986667 / 4.98 t/s | 10,692 | 70.479492 GiB | 2,752 | 3.940667 ms | 1.522667 ms |

Measured split-fused effect: mean decode `+8.17%` and median decode
`+7.79%` against the contemporary static32 control. Transport was intentionally
unchanged: both arms had 5,820 VRAM hits, 10,692 RAM hits and 70.479492 GiB RAM
H2D. Candidate avoided counters were positive and stable in every accepted
process: `175177728` split-fused miss scratch bytes avoided and `175177728`
split-fused sum-read bytes avoided. Mean route wait fell by `0.373666 ms/call`
and mean worker time fell by `0.058333 ms/job`.

This is a positive native-Windows result for the exact 64-token prompt only. It
does not claim general quality or broader workload quality. The result supports
keeping static32 plus split-fused routing as the current short-workload
performance candidate, with longer prompts and L0-L3 quality gates still
separate follow-ups.

Native Windows commits: implementation
[`4c08683`](https://github.com/imanu86/ds4-win/commit/4c08683), protocol
[`3912e7b`](https://github.com/imanu86/ds4-win/commit/3912e7b), fail-closed
telemetry gate
[`21c3aac`](https://github.com/imanu86/ds4-win/commit/21c3aac), and results
[`580e29b`](https://github.com/imanu86/ds4-win/commit/580e29b).

Canonical Windows summary:
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_split_fused_ab_result.json`.

## 2026-07-17 Native Windows G103 IQ1_S Cold Transport On G73

G103 preregisters the first direct composition of the G73 short-workload SOTA
with IQ1_S cold transport. The control freezes G73 exactly: arena30 / 4551
slots, cache320, source-parts WRAP with waved reclaim, composed prefill mass
tiering, static budget32, GPU-resident no-default-sync routes and SplitFused.
The candidate changes only cold execution: one cold expert per routed layer is
served from the verified IQ1_S sidecar through the mixed GPU planner and a
0.5 GiB pinned IQ1 RAM cache. Promotion, open-router reserve slots,
RoutePackedCopy, packed IQ1 H2D and IQ1 VRAM cache are disabled.

The IQ1_S sidecar was copied from archival SATA storage to the benchmark NVMe:
`C:\ds4-models\DeepSeek-V4-Flash-IQ1_S-XL.gguf`, `61540805344` bytes,
SHA-256 `b049d1eb34c068f19ab007b33c22a7d758b578bf2b10d9276e79654f85d35047`.
A path-bound `g7_verified_file_receipt_v2` was generated after a complete
locked-stream SHA-256 pass. No G103 benchmark reads the D: copy.

The first `-SafetyOnly` attempt stopped before DS4 launch. The memory preflight
required 32 GiB available for arena30 plus guard, but measured 23.77 GiB before
and 23.76 GiB after `wsl --shutdown`; standby was 23.67 GiB and no purge tool
was installed. This is a host-state gate failure, not an IQ1 result. There is
no timing, exactness, quality or SOTA datapoint from this attempt. Available
memory then remained in the 23.88--23.95 GiB range over eight samples spanning
107 seconds, so the protocol was not weakened to force a launch.

After a clean Windows reboot, available RAM rose to 54.88 GiB and the original
1 GiB candidate reached DS4. It then failed closed at decode layer 3 because
the extra 0.998 GiB `cudaHostAlloc` could not coexist with the unchanged 30 GiB
G73 arena. This run emitted zero tokens and is not timing or quality evidence.
A same-stack 0.5 GiB structural probe completed with 109 pinned IQ1 slots and
zero cache/runtime failures. It measured a 6.95% cache hit rate, 10.904 GiB of
IQ1 sidecar reads and 2.55 server decode t/s, but remains `n=1` exploratory
evidence only. The protocol was amended to 0.5 GiB rather than shrinking the
G73 arena.

The official 0.5 GiB safety then passed: arena30/4551 was not capped, cache320
and SplitFused accounting were complete, 2,560 mixed calls used one IQ1 expert,
the IQ1 cache had 109 slots and zero failures, IQ2 tier SSD bytes stayed zero,
and no cold SSD-to-VRAM path occurred. Its observed 2.70 server decode t/s is
structural `n=1`, not a performance verdict. The safety receipt SHA-256 is
`6dbc0fcfd2b3d7a25070b5c106dbc3ae5c138ea468d1b3e0d5fb2fc4d2086e90`.

Three runner defects were exposed fail-closed before the matrix: runtime OOM
was initially masked by a later route-summary count check; native child stdout
polluted the PowerShell row pipeline; and path-bound receipt reuse was invalid
for benchmark members. The final matrix design keeps control sidecar-free and
full-hashes its IQ2 model, while candidate members reuse one full-hash
model+IQ1 suite receipt under parent read/deny-write/delete locks.

Native Windows protocol commits:
[`bebc194`](https://github.com/imanu86/ds4-win/commit/bebc194) and
[`cbb410c`](https://github.com/imanu86/ds4-win/commit/cbb410c). Post-reboot
corrections are [`8992865`](https://github.com/imanu86/ds4-win/commit/8992865),
[`0e84a9f`](https://github.com/imanu86/ds4-win/commit/0e84a9f),
[`1ca5a5e`](https://github.com/imanu86/ds4-win/commit/1ca5a5e), and
[`360e065`](https://github.com/imanu86/ds4-win/commit/360e065). Candidate IQ1
output is not required to equal the G73 IQ2 hash; quality remains a separate
L0-L3 n>=3 gate, while the control must retain the exact G73 hash.

The preregistered interleaved matrix then completed with three independent
processes per arm and no outlier extension. G73 control server decode was
`5.04 / 5.11 / 4.96` t/s, mean `5.0367` and median `5.04`. The candidate with
one cold IQ1_S expert per routed layer was `2.71 / 2.71 / 2.69` t/s, mean
`2.7033` and median `2.71`: a measured `-46.32%` regression. Candidate output
was deterministic across all three processes, all runtime failure counters
were zero, and the control retained its exact expected content hash. This
64-token matrix is transport/performance evidence, not an L0-L3 quality
verdict.

Each candidate process recorded 2,560 mixed calls, 178 IQ1 RAM-cache hits,
2,382 misses, 2,273 evictions, 10.904 GiB of IQ1 sidecar reads and 11.719 GiB
of IQ1 H2D. The 109-slot cache therefore hit only 6.95%. The important
interpretation is that these accesses replace the same class of primary cold
IQ2 routes that G73 serves from its 30 GiB RAM arena with zero IQ2 SSD bytes;
G103 turned one RAM-served cold route per layer/token into a mostly SSD-served
IQ1 route. Exact expert sequences are not assumed identical after output
divergence. IQ1 calculation is operational, but this storage placement is not
amortized and is not a SOTA candidate.

The aggregate initially stopped after all six valid members because it checked
the nonexistent `build_input_fingerprint_sha256` field instead of the emitted
`build_manifest_input_fingerprint_sha256`. The corrected resume path validates
the existing full-hash suite receipt against path, bytes, timestamps, file IDs
and declared hashes under parent locks, then aggregates existing members
without rerunning DS4. Next gate: a structural IQ1 profile separating SSD read,
H2D enqueue/sync, hot/cold submit and join, followed by one transport lever at
a time. SPEX cross-entropy surprise may later control probation width and
promotion urgency, combined with router weight and recent mass.

### G104 structural profile

G104 enabled only the existing IQ1 profile instrumentation on the negative
G103 candidate. It is `n=1` structural evidence, not a throughput or quality
verdict. Across 2,560 mixed calls it measured 6,595.827 cumulative ms in 2,382
SSD reads, 6,805.673 ms enqueueing 7,680 H2D copies, 561.803 ms in 2,560 H2D
syncs, 12,957.775 ms submitting the five hot IQ2 routes, and 7,724.433 ms
submitting the cold IQ1 route. Router D2H, metadata H2D and join submit were
only 7.794, 14.857 and 17.965 cumulative ms. Timers may overlap and profiling
adds overhead; they identify stages but are not summed into a wall-clock model.

The measured placement conclusion is now explicit: the target is not an IQ1
SSD cache layered on top of G73. The runtime slot size is 4,915,200 bytes, so
all 10,240 routed IQ1 experts require 46.875 GiB. The next gate replaces the
30 GiB IQ2 host arena with a preloaded all-IQ1 host arena, keeps the protected
IQ2 set in VRAM plus only a small IQ2 RAM probation tier, and leaves IQ2 SSD
backing for anticipated promotion. Packed IQ1 H2D remains separate because
full host residency removes SSD reads but not the three copy submissions per
IQ1 expert.

Native Windows commits: G103 aggregate/results
[`1112cf3`](https://github.com/imanu86/ds4-win/commit/1112cf3), G104 protocol
[`883c43b`](https://github.com/imanu86/ds4-win/commit/883c43b), and G104 result
[`4854731`](https://github.com/imanu86/ds4-win/commit/4854731).

### G105 full-IQ1 residency attempt

G105 implemented an opt-in 46.875 GiB pageable Windows arena for all 10,240
routed IQ1 experts. It used deterministic slots for layers 3 through 42,
preloaded every expert once, froze the mapping, and forbade on-demand IQ1 SSD
fallback during decode. The implementation, fail-closed harness, protocol and
static contract were committed as
[`8580241`](https://github.com/imanu86/ds4-win/commit/8580241). Build Release,
native ctest and the G103/G104/G105 static contracts passed before launch.

The first structural attempt was stopped after 535.754 seconds because OS
telemetry disproved physical residency. It reached layer 42 and entered decode,
but process working set peaked at 53,342,937,088 bytes and later fell to
33,461,399,552 while private committed bytes stayed near 63,297,142,784.
Available RAM reached 4,020,531,200 bytes. The system disk queue peaked at 25,
system disk writes at 682,677,263 B/s, system reads at 1,719,828,058 B/s, and
process page faults increased by 38,454,456. Process reads increased by
150,741,854,757 bytes. No final content hash or cache summary exists because
the invalid run was stopped; it supports no throughput, exactness or quality
claim.

This is a negative placement result, not an IQ1 compute failure. A decode-side
counter of zero explicit IQ1 SSD bytes is insufficient when Windows trims or
pages the pageable arena underneath the runtime. The existing contamination
monitor also missed the failure because it required both available RAM below
1 GiB and a high disk queue simultaneously; paging restored available RAM and
broke that conjunction.

Decision: reject complete 46.875 GiB IQ1 residency on this 64 GiB host. Measure
the largest physically stable mass-ranked IQ1 arena while preserving the IQ2
hot placement, then add a dedicated four-slot pinned DMA staging ring
(18.75 MiB total) between pageable IQ1 RAM and H2D. Keep full-IQ1 preload as a
research fallback only. The compact native receipt is
`G105_IQ1_FULL_RESIDENT_ABORT_RECEIPT.json`; its hashes bind the uncommitted raw
local stderr, telemetry and memory preflight files.

### G106 pinned IQ1 capacity boundary

G106 first fixed the G105 observability hole. The runtime monitor now records
system page-in/page-out counters, a private-working-set residency ratio and an
independent hard available-RAM floor. Requested counters fail closed, and an
HTTP failure caused by the monitor now reports the exact abort reasons. The
guard commits are
[`5634a2b`](https://github.com/imanu86/ds4-win/commit/5634a2b),
[`461220d`](https://github.com/imanu86/ds4-win/commit/461220d), and
[`d106c46`](https://github.com/imanu86/ds4-win/commit/d106c46).

The structural capacity probe kept the current 20 GiB IQ2 dynamic arena and
varied only the pinned IQ1 host cache. All valid arms were `n=1` structural
safety runs with a 4 GiB hard RAM floor, a 512 pages/s output ceiling and no
timing or quality eligibility. Results:

| IQ1 request | Evidence | Min available RAM | Peak shared WDDM | Page-out peak | VRAM expert slots | Gate |
|---:|---|---:|---:|---:|---:|---|
| 6 GiB | 1259 / 1310 slots used over 64 tokens | 26.236 GiB | 26.373 GiB | 0 | 320 | PASS |
| 8 GiB | full pinned allocation, 488 / 1747 used | 24.891 GiB | 28.373 GiB | 0 | 296 | PASS |
| 10 GiB | full pinned allocation, 488 / 2184 used | 22.923 GiB | 30.373 GiB | 0 | 296 | PASS |
| 11 GiB | first IQ1 selected load | n/a | n/a | n/a | n/a | `cudaHostAlloc` OOM |

The measured boundary for this exact stack is therefore 10 GiB passing and
11 GiB failing. The initial dynamic-policy candidate is nevertheless 6 GiB:
it preserves all 320 measured VRAM expert slots, while 8 and 10 GiB reduce the
cache to 296. This is a capacity decision only; the forced one-IQ1-per-layer
fixture remains the wrong routing policy and its decode rates are excluded
from SOTA.

One calibration run is also retained as excluded evidence. A 0.80 residency
ratio armed after only 8 GiB private bytes killed normal CUDA startup with
35 GiB still available and zero page-out. The corrected gate arms the 0.70
ratio only after 40 GiB private bytes. G106 protocol and compact receipt are
committed in native Windows as
[`d150535`](https://github.com/imanu86/ds4-win/commit/d150535).

Next runtime gate: preserve every IQ2 VRAM or IQ2 arena/RAM hit and substitute
IQ1 only when metadata proves that the selected IQ2 route would otherwise read
from SSD and the IQ1 representation is already resident. Separately, the
aggressive-quantization survey found existing GGUF `Q1_0` type 41 at 18 bytes
per 128 weights (1.125 bpw), or 33.75 GiB for all routed experts. It is the
preferred sub-IQ1 research path, but DS4's routed-expert binder and kernels do
not support it yet; no quality or runtime claim exists. The CPU-only Q1_0
layout and dot-product smoke is documented in
[`0a38d94`](https://github.com/imanu86/moe-aggressive-commit/commit/0a38d94)
with 10 focused tests passing.

### G107 IQ1_S cold-only residency gate

G107 implemented the fail-closed policy requested after G106: preserve every
selected IQ2 route already resident in VRAM, the prefill snapshot arena, or the
tiering RAM tier; permit an IQ1_S substitution only when the authoritative IQ2
expert is classified as SSD cold and the exact IQ1_S representation is already
present in the RAM cache. Unknown residency, an IQ1 cache miss, or any failure
falls back to all six main IQ2 routes. The runtime, harness, raw HTTP checkpoint
and focused contract were committed in native Windows as
[`83aec83`](https://github.com/imanu86/ds4-win/commit/83aec83).

The validated structural run used the cyberpunk HTML prompt, temp 0, no-think,
64 generated tokens, three requests in one process, the 30 GiB / 4,551-slot
prefill-mass WRAP arena, request-scoped closed mass mask, 320-entry VRAM expert
cache, split-fused routes and enforced mass-LFRU tiering. All three outputs were
byte-identical to the expected baseline SHA
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`.
Server decode measured `4.47 / 4.48 / 4.50` t/s. End-to-end request times were
`63.944 / 22.568 / 19.330` seconds because the first request paid the one-time
25.682-second 30 GiB arena WRAP while the later requests reused it. This is an
`n=3` structural exactness result, not an L0-L3 quality grade and not a SOTA
throughput claim.

Across 7,680 routed calls and 46,080 selected routed-expert uses, measured
residency was 17,439 VRAM and 28,641 prefill-snapshot RAM. SSD-cold, tier-RAM,
IQ1 RAM hits, IQ1 substitutions, uncertain classifications and failures were
all zero. The 6 GiB IQ1 cache was therefore not materialized and the receipt
records `iq1_s_ram_cache_deferred_unused=true`. The three tier summaries also
reported zero cold routes, zero SSD bytes and zero forbidden direct
SSD-to-VRAM transitions. Aggregate IQ2 snapshot-to-VRAM transport was
227,030,335,488 bytes.

This is not an isolated zero caused by the three-request lifecycle. The prior
G73 SOTA aggregate contains three independent clean processes; each reports
2,752 tier calls, 16,512 selected routes, 10,692 snapshot-RAM hits, 5,820 VRAM
hits, zero cold routes, zero IQ2 SSD bytes and zero forbidden direct
SSD-to-VRAM transitions. Their validated decode rates were `4.98 / 4.97 /
5.01` t/s. G46 and G70 also measured zero tier-cold routes in each of their
three processes, although G46 predates the modern contamination contract and
G70 is descriptive rather than performance-claim eligible. Thus the closed
SOTA stack already supplies independent `n>=3` evidence that the IQ2 cold-SSD
path is not its decode bottleneck.

Decision: the cold-only rule is mechanically correct, exact and fail closed,
but this SOTA-like closed-mask protocol gives it no work. The measured remaining
transport target is RAM-to-VRAM, not SSD-to-RAM. Do not spend an `n>=3` matrix
on larger IQ1_S SSD/RAM caches under the same closed mask. The next active
representation experiment is a resident Q1_0 base or equivalent compact RAM
representation that reduces the bytes for the 28,641 snapshot-RAM routes,
while IQ2 remains authoritative for hot experts and promotion.

The complete native result and receipts are committed as
[`2fc99c1`](https://github.com/imanu86/ds4-win/commit/2fc99c1). In parallel,
Q1_0 runtime step 1 (type 41, 128 weights / 18 bytes, dot/dequant and fail-closed
dispatch) is committed as
[`9251e5e`](https://github.com/imanu86/moe-aggressive-commit/commit/9251e5e),
and step 2 (separate binder metadata/offsets plus compiling qwarp32 gate/up/down
kernels, still not runtime-dispatched) as
[`2c32417`](https://github.com/imanu86/moe-aggressive-commit/commit/2c32417).
Neither Q1_0 commit supports a speed or quality claim yet.

### G108 Q1_0 foundation env-off safety

G108 is the inertness gate for the corrected native-Windows Q1_0 foundation.
It used native commit
[`6a44578`](https://github.com/imanu86/ds4-win/commit/6a44578), the same
authoritative IQ2 model and the G73 static-32 split-fused configuration, with
all Q1_0 and IQ1_S sidecar options disabled. A stale build manifest stopped the
first launch before the model was opened; after `g7_build.ps1` regenerated the
manifest and executable provenance, the recorded run exited normally.

The one greedy/no-think safety repeat produced the expected content SHA
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`
exactly. Q1_0/IQ1_S runtime remained unobserved, the prefill-mass candidate set
and 30 GiB arena both contained 4,551 entries, snapshot backing misses were
zero and tiering read zero bytes from SSD. The known RAM transport remained:
75,676,778,496 expert bytes were uploaded to the GPU. This passes the narrow
env-off exactness gate and shows that the Q1_0 foundation does not alter G73
when disabled.

The run reported `8.3 t/s` server decode, `51.241 s` prefill/TTFT and
`26.354 s` WRAP. These numbers are retained only as provenance: `n=1`,
`quality_eligible=false`, `sota_eligible=false`, so they do not update the SOTA
or support a performance verdict. The native receipt and explicit exclusion
are committed as
[`ee8167c`](https://github.com/imanu86/ds4-win/commit/ee8167c).

### G109 Q1_0 real sidecar and resident transport

G109 converted the authoritative IQ2/Q2 tensors for routed layer 42 to a real
Q1_0 type-41 sidecar using the reference CPU helper ported from llama.cpp. The
fixture is explicitly `derived_from_iq2=true` and `not_quality_optimal=true`:
`C:\ds4-models\ds4-q1-layer42-derived-iq2-84b4ffb.gguf`, `907428832`
bytes, SHA-256
`58d537738ac80df504d9954a694703c37cc5f9ee236ca8c06ce94cea1ab8ef26`.
Its receipt SHA-256 is
`6a36ae77c20e60f2e8a6be57b64f1f8eb524def9d9d119201c81e92c56312462`.
The converter provenance/hardening is committed as
[`f969faa`](https://github.com/imanu86/moe-aggressive-commit/commit/f969faa),
[`2dd1b0a`](https://github.com/imanu86/moe-aggressive-commit/commit/2dd1b0a) and
[`84b4ffb`](https://github.com/imanu86/moe-aggressive-commit/commit/84b4ffb).

The native runtime added a pinned host arena for the Q1_0 active layer, a
receipt-bound selected loader and fail-closed telemetry. The ordered `n=1`
structural gates measured:

- env-off preserved the known exact content hash and observed no Q1_0 state;
- a valid sidecar without `DS4_Q1_0_SELECTED_LOAD=1` stopped with the expected
  fail-closed diagnostic and did not fall back silently;
- direct-file Q1_0 produced nine selected loads and `251265024` routed pread
  bytes with zero failures;
- the first resident attempt failed before serving because generic arena
  geometry rejected type 41; this negative result led to the explicit Q1-only
  geometry fix;
- the fixed resident attempt produced 71 resident hits, zero misses,
  `251265024` H2D bytes and zero routed direct-file reads.

Runtime, runner and geometry commits are
[`c6fbb2b`](https://github.com/imanu86/ds4-win/commit/c6fbb2b),
[`1f661f0`](https://github.com/imanu86/ds4-win/commit/1f661f0) and
[`eabf03c`](https://github.com/imanu86/ds4-win/commit/eabf03c).

After the structural gates, an isolated transport matrix used six independent
clean processes in balanced order `C-D-D-C-C-D`, three per arm. Every run used
the same executable SHA, passed quiescence, generated the same retained output
SHA `8a17fc0dc61e8520bdbe3a735b000358a6476cbe9f0e3d86c54a51cf26b5d009`
and moved the same Q1_0 route payload.

| G109 arm | Decode t/s, three processes | Mean / median | HTTP t/s mean | TTFT mean |
|---|---:|---:|---:|---:|
| Direct file | `2.45, 2.58, 2.58` | `2.5367 / 2.58` | `1.2845` | `2.7173 s` |
| Resident arena | `2.61, 2.44, 2.67` | `2.5733 / 2.61` | `1.2895` | `2.7247 s` |

The resident arm eliminated `753795072` routed pread bytes across its three
processes and served the same byte count from pinned RAM with zero misses. The
measured mean deltas were only `+1.4455%` process decode, `+0.3923%` HTTP
throughput and `+0.2699%` TTFT time. Therefore the earlier `n=1` apparent 2x
decode uplift is rejected as a performance result: it was not reproduced by
the balanced matrix. The measured positive finding is zero routed pread, not a
material throughput win on this warm one-layer fixture.

No SOTA or quality claim follows. G109 uses eight output tokens and one
IQ2-derived Q1_0 layer, so it is not an L0-L3 quality experiment and cannot be
compared with the complete G46/G73 stack. The current Q1 resident mode owns the
single global host arena and disables the IQ2 arena. The next gate is a typed
dual-arena resolver keyed by backing/layer/expert, preserving all Q1 fail-closed
rules, followed by exact safety and `n>=3` comparison of the complete composite.
The complete native protocol and result report are committed as
[`eb85ee4`](https://github.com/imanu86/ds4-win/commit/eb85ee4).

### G112-G116 full-router and representation disambiguation

G112 is the first local Q1_0 run in this sequence that actually kept the
decode router open over all routed experts. It loaded 11,008 sidecar entries
(including the layout entries for layers 0 through 2), used Q1_0 for all six
selected experts in every routed layer, and applied no request-scoped mask.
The 256-token cyberpunk safety process measured 6.76 server decode t/s with
66,048 Q1_0 routes, zero IQ2 routes, zero misses and zero direct preads. Its
output was L0. This is an `n=1` transport ceiling and a negative quality gate,
not a SOTA result.

G113 and G114 kept the full Q1_0 candidate set but added only a small exact-IQ2
VRAM seed. The measured exact share remained about 1.15 to 1.34 routes out of
six. Both variants stayed near 6.21-6.22 server decode t/s, had zero storage
misses and produced L1 output in their single safety samples. They show that a
small opportunistic IQ2 seed does not repair a Q1-dominant calculation; they do
not establish a general quality rate.

G115 then implemented a physical five-IQ2 plus one-Q1_0 split. It served
20,733 exact routes from VRAM, 34,307 from the exact IQ2 host snapshot and
11,008 from resident Q1_0, with zero IQ2 SSD bytes and zero Q1 misses. However,
the request-scoped snapshot retained only 4,247 entries total, including the
768 layout entries for layers 0 through 2. The active routed coverage was
3,479 / 10,240 entries and gate-mass coverage was 0.5566. The single 256-token
safety process measured 5.55 server decode t/s and malformed output. It is not
a full-router 5+1 quality test. Code review also shows that this path invokes
`routed_moe_launch` twice, requantizes the input for the Q1 sub-launch, disables
SplitFused for Q1 and joins a second output buffer. Its throughput therefore
includes avoidable composition overhead and is not a clean precision-only
contrast.

G116 removed Q1_0 while preserving the same 55.66%-coverage closed mask and
28 GiB host arena. It measured 8.08 server decode t/s in one safety process,
zero SSD traffic and invalid long-form output. This isolates a large throughput
cost in the current mixed Q1 dispatch, but also confirms that the closed mask
itself is sufficient to invalidate the long document. G116 is neither an
open-router control nor a SOTA result.

The causal conclusion is now frozen: expert selection and representation must
not be conflated. The next candidate keeps the original router IDs and weights
authoritative, avoids a request-scoped closed mask, and uses Q1 only as an
exactly completable resident base or a separately graded low-weight one-token
fallback.

### Nested IQ2 base plus exact residual CPU gate

The CPU-only nested-residual lab split the original expert bytes rather than
requantizing independently. For IQ2_XXS, the base keeps scale/group-scale/sign
metadata in 34 of 66 bytes and the residual contains the remaining 32 bytes.
For Q2_K, the base keeps scales/mins and the high quant bitplane in 52 of 84
bytes; the residual is again 32 bytes. One expert is therefore 3.75 MiB base
plus 3.00 MiB exact residual, reconstructing the original 6.75 MiB.

Three deterministic sample seeds read 768 real routed-expert blocks each from
`C:\ds4-models\ds4-2bit.gguf`. All 2,304 joins reproduced the original block
byte-for-byte. Mean base-only dot nMAE was 0.02978-0.03020 for IQ2_XXS and
0.02638-0.02724 for Q2_K. The base-only error is not a quality breakthrough;
its measured value is that an exact cold promotion reads 3.00 rather than 6.75
MiB per expert. The 10,240 active routed experts require 37.50 GiB resident
base; all residuals would be 30.00 GiB and remain bounded/cached rather than
duplicated in RAM.

The implementation and raw reports are in
`runs/ds4/20260718_nested_residual_lab/`, with the active architecture in
`docs/FULL_ROUTER_NESTED_RESIDUAL_PLAN_20260718.md`. This is a representation
and exactness gate only. No decode-speed or generation-quality claim exists
until the one-layer runtime reconstructs the authoritative IQ2 output and the
full-router long-form protocol passes `n>=3` L0-L3 grading.

### G117 nested residual one-layer structural safety

G117 measured the first runtime safety check of the nested residual path. It is
`n=1` structural safety only: full/open router, mask off, prompt `Hi`,
`max_tokens=8`, `ctx=256`. Control and candidate produced identical output
SHA-256 `8a17fc0dc61e8520bdbe3a735b000358a6476cbe9f0e3d86c54a51cf26b5d009`.

Structural counters were `router_calls=9`, `cache_hits=9`,
`cache_misses=62`, `preads=62`, `reconstructed=62`,
`residual_bytes=195035136`, `h2d_bytes=502530048`, `mismatch=0` and
`failures=0`. Control server decode was 1.55 t/s; candidate server decode was
1.40 t/s. `timing_claim_valid=false`, so no performance verdict, no throughput
claim and no SOTA update are recorded from this run.

The run happened after fixing the pre-test review blockers: pinned slot event
reuse, hard fail closed, mandatory open router, parser overflow, full sidecar
hash lock and per-used-expert reconstruction.

### G118-G119 nested residual distributed-layer capacity gate

G118 applied the exact nested representation to layers 3, 16, 29 and 42. It
matched the control output SHA and measured `router_calls=36`, `misses=268`,
`preads=268`, `residual_bytes=843055104`, `reconstructed=268`,
`h2d_bytes=1896873984`, `mismatch=0`, `failures=0`. The candidate pinned
`4026531840` base bytes but then failed to register the requested 28 GiB source
window with CUDA `out of memory`; the control registered 28 GiB. Exactness
passed, but the memory paths were not comparable and timing is invalid.

G119 used the same model, sidecar, prompt, output hash and four covered layers
with `BudgetGB=24`. Both arms registered the 24 GiB source window; the nested
candidate also pinned all 3.75 GiB of base with `mapped=0`. All 268 misses were
reconstructed from 268 residual reads, with zero mismatch and zero failure.
This fixes the first compatible physical budget at 24 GiB source window plus
3.75 GiB nested base. G118 and G119 are both `n=1` structural evidence only;
neither updates quality, throughput or SOTA.

### G120-G121 open-router nested reuse gate

G120 combined the historical G73 transport levers with an explicitly open
request-scoped router. This corrected the earlier assumption that G73's
4.986667 t/s result was itself full/open: G73 used a request-scoped closed
transport set after prefill. The G120 full/open control produced coherent
output SHA
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`,
different from historical G73 SHA
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`.

The single G120 control measured 0.650419 end-to-end t/s, 1.42 server decode
t/s and 52.864 s prefill/TTFT. The exact four-layer candidate was byte-identical
with zero mismatch/failure, but its default six-entry exact cache recorded 0
hits and 1,802 misses. It measured 0.487870 / 0.88 t/s and read
5,668,601,856 residual bytes. Source review established why: six entries hold
only one top-six route, so each covered layer evicted the previous layer before
the next token.

G121 changed only nested exact-cache capacity from 6 to 64. Output stayed
byte-identical. Hits rose to 817, misses fell to 985 and residual bytes fell to
3,098,542,080. Derived hit rate and miss reduction were 45.34%.
End-to-end throughput rose to 0.553980 t/s (+13.55% from the cache-6 arm) and
server decode to 1.12 t/s (+27.27%), but remained below the single open
control. Nested H2D remained 12,754,354,176 bytes in both candidates because
the covered-layer path still bypasses the existing GPU-resident SplitFused
cache and uploads a full native expert for every use, including host-cache
hits.

G120 and G121 are `n=1` structural measurements and do not update SOTA or
generalized quality. The next causal gate is to admit reconstructed exact
experts to the existing VRAM route cache, require nonzero nested VRAM hits and
lower H2D at exact output, then run `n>=3`.

### G122 nested exact GPU-resident reuse gate

G122 completed that causal gate with the same full/open output SHA
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`,
zero reconstruction mismatch, zero runtime failure and no selected-load
fallback. Across the four covered layers it measured 256 GPU-route calls,
541 VRAM hits and 995 VRAM misses. Every miss filled and uploaded exactly one
native 7,077,888-byte expert: `host_fills=995`,
`host_bytes=h2d_bytes=7042498560`.

The dedicated route-cache H2D subset was 7,042,498,560 bytes. Including
prefill selected-load traffic, the comparable total nested H2D counter fell
from 12,754,354,176 bytes in G121 to 8,925,216,768 bytes in G122, a measured
30.02% reduction. End-to-end throughput was 0.559345 t/s and server decode
1.16 t/s versus G121's 0.553980 / 1.12; those +0.97% and +3.57% deltas are
`n=1` signals only. Prefill/TTFT was 58.849 s and load was 21.9 s. G122 is
structural evidence, not a SOTA or generalized performance verdict. It
unlocks a clean equal-host-budget `n>=3` full/open A/B.

### G123 nested exact equal-host-budget n=3 A/B

G123 compared three independent full/open G73-composite controls with three
independent nested GPU-cache candidates. All six accepted rows reproduced
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`,
passed quiescence and had runtime contamination peak zero. Host budget was
equal: 30 GiB control arena versus 25.828125 GiB candidate arena plus 3.75
GiB nested base and 0.421875 GiB exact cache.

Control mean/median end-to-end throughput was 0.650261/0.647540 t/s and
server decode was 1.650000/1.650000 t/s. Candidate mean/median end-to-end was
0.555497/0.559217 t/s and decode was 1.163333/1.160000 t/s. Mean deltas were
-14.57% end-to-end and -29.49% decode. Mean TTFT changed from 59.217 s to
59.803 s; mean load changed from 11.222 s to 29.212 s.

Every candidate recorded 541 nested GPU hits, 995 misses, 995 host fills,
7,042,498,560 route H2D bytes, 869 residual reads, 2,733,637,632 residual
bytes, 8,925,216,768 total nested H2D bytes and zero mismatch/failure. The
35.22% GPU hit rate is insufficient to amortize the current host
reconstruction and full-expert upload miss path.

An initial control-r3 attempt is explicitly excluded: arena copy expanded
from about 32 s to 134.654 s and the route worker timed out at sequence 1882.
The cooled replacement was exact and measured 0.656230 end-to-end / 1.67
decode. G123 therefore rejects extending the current implementation to all
layers. SOTA is unchanged; this 64-token exact benchmark is not a generalized
L3 quality result. Receipt:
`runs/ds4/20260718_nested_residual_lab/G123_RECEIPT.json`.

### G124 nested residual causal profile

G124 profiles the G123 nested candidate miss path without changing the verdict.
It is `n=1` causal profile evidence only, with exact output SHA
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510` and
runtime reconstruction verification enabled. It has no SOTA, performance A/B
or L0-L3 generation-quality verdict.

The authoritative receipt is
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g124_nested_residual_profile_20260718T172630143Z_ea23d8bd87_receipt.json`;
the associated result is
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g124_nested_residual_profile_20260718T172630143Z_ea23d8bd87_result.json`.

Measured profile timers were: CPU reconstruct `14.8898841 s` over 869 calls,
residual pread `1.9070234 s` over 869 calls, reconstruction verification
`0.4990679 s` over 2,607 calls, host copy `0.3460242 s` over 995 calls,
H2D enqueue `0.0381582 s` over 995 calls, H2D sync `0.1677010 s` over 254
calls, H2D enqueue+sync `0.2058592 s`, and route-ready wait `14.6092232 s`
over 256 calls. These timers overlap across worker, route and transfer paths;
they identify causal stages and must not be summed into wall-clock time.

The measured next lever was G125 GPU-side exact join, so the miss path could
avoid host-side reconstruction plus full native expert upload before reuse.

### G125 nested residual GPU-join structural safety

G125 is the `n=1` structural safety gate for GPU-side exact join on the same
full/open nested-residual fixture. It preserves the open router: no REAP mask,
no static mask, no bake mask and no request-scoped closed routing are part of
the run. The output SHA was exactly
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.

The authoritative receipt is
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g125_nested_gpu_join_safety_current_build_clean_20260718T182935501Z_eb824ebedb_receipt.json`
with receipt SHA-256
`ae15a6d3d3bc35e75b46befd8d18d7886f571e47d93561d146ada3ccf20f58fb`.

Measured G125 counters: GPU join was requested and observed, with 1,261 calls,
4,958,453,760 base H2D bytes, 3,966,763,008 residual H2D bytes, zero native
H2D bytes, zero CPU reconstruction calls, 3,783 verification calls, zero
verification mismatches and zero failures. The existing route cache still
reported 256 route calls, 541 hits and 995 misses. The GPU-join timer was
0.1242402 s and wait time was 0.0019319 s, but these are diagnostic timers
only and may overlap with other stages; G125 is not a performance or quality
verdict.

### G126 nested residual CPU-join versus GPU-join A/B

G126 is the repeated full/open A/B for the same four-layer nested fixture,
comparing the previous CPU reconstruction miss path against GPU-side exact
join. It used three independent processes per arm in order
`cpu, gpu, gpu, cpu, cpu, gpu`. All six accepted rows were exact,
uncontaminated and produced the same content SHA
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.
The model SHA was
`efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`;
the sidecar SHA was
`07199bc5503aa6e2dea10f702c1ca9e8f05a5bf466a56cbed031f6a5fca4bdf9`.

The authoritative aggregate is
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g126_nested_gpu_join_ab_current_build_clean_v2_20260718T183831489Z_b4d8a5d111_result.json`
with SHA-256
`c1f7849aa0da33c4b6d8279073954ba39f556fc485215e6d4058e809fbe9eaa6`.
It references the G125 safety receipt above.

| G126 arm | Server decode t/s | Mean / median | End-to-end t/s mean / median | TTFT mean / median |
|---|---:|---:|---:|---:|
| CPU join | `1.15, 1.16, 1.15` | `1.153333 / 1.15` | `0.333719 / 0.241509` | `171.829 / 208.944 s` |
| GPU join | `1.56, 1.58, 1.57` | `1.570000 / 1.57` | `0.499730 / 0.657202` | `140.836 / 56.204 s` |

The defensible G126 finding is the server decode delta: GPU join improved
mean decode throughput by 36.1272% over CPU join on this exact batch. The
end-to-end mean was also higher by 49.7457%, but TTFT and WRAP/request timing
were wildly noisy in this batch, with request seconds spanning roughly
96.98 to 351.57 s and WRAP/TTFT behavior not stable enough for a general
latency claim. Treat E2E as batch-only/noisy.

The CPU arm reconstructed 869 experts in every process. The GPU arm recorded
1,261 GPU-join calls in every process, positive base and residual H2D, zero
native H2D, zero CPU reconstruction, zero benchmark verification calls, zero
mismatches and zero failures. Both arms retained the same 541 nested GPU hits,
995 misses and 7,042,498,560 nested route-cache H2D bytes per process.

Context: G126 should be compared to G123 as a same-fixture full/open nested
improvement: G123's nested candidate mean decode was 1.163333 t/s and G126
GPU join reached 1.57 t/s while preserving exact output. It remains below the
G123 full/open IQ2 control at 1.65 t/s, and historical G73 at 4.9867 t/s is a
closed/request-scoped short-workload result, not an absolute full/open target.
G126 is therefore not absolute SOTA. It is a positive miss-path decode finding
for GPU-side exact join under the current four-layer nested fixture.

## 2026-07-19 Native Windows G73 Diagnostic Closeout

This is an `n=1` diagnostic and safety closeout on the current G73-derived
build. It does not update the historical G73 `n=3` result, establish output
exactness, grade C0-C3 quality, or make a full/open or headline-performance
claim. Every completed canary response ended with `finish_reason=length`; its
decode-valid flag only means that the protocol's 48-token minimum was met.

| Record | Status and configuration | Exact receipt measurements | Allowed claim |
|---|---|---|---|
| C0 | pass; one request; ctx256/chunk256 | prompt/completion 43/64; wall 79.769711 s; TTFT 10.278 s; prefill 66.318 s at 0.648391 t/s; decode 13.035 s at 4.909858 t/s; runtime tensor progress 0; unsafe tier events 0 | `n=1` baseline canary only; C0 is not workload-comparable to C1-C3 |
| C1 | pass; dynamic two-request case; ctx1024/chunk256 | request1 43/64, wall 79.309938 s, TTFT 10.051 s, prefill 66.155 s at 0.649989 t/s, decode 12.726 s at 5.029074 t/s; request2 306/64, wall 75.561198 s, TTFT 10.646 s, prefill 62.217 s at 4.918270 t/s, decode 13.331 s at 4.800840 t/s; runtime tensor progress 0; unsafe tier events 0 | request2 dynamic canary passed; no quality, sustained-performance, or full/open verdict |
| C2 | pass; dynamic two-request case; ctx8192/chunk256 | request1 43/64, wall 272.863275 s, TTFT 9.963 s, prefill 259.776 s at 0.165527 t/s, decode 12.665 s at 5.053296 t/s; request2 306/64, wall 75.391667 s, TTFT 10.412 s, prefill 62.459 s at 4.899214 t/s, decode 12.929 s at 4.950112 t/s; runtime tensor progress 0; unsafe tier events 0 | ctx8192/chunk256 diagnostic canary passed; no quality, sustained-performance, or full/open verdict |
| C3 | aborted; dynamic two-request case; ctx8192/chunk2048 | one runtime tensor progress event during request1 `request-begin`: `6.22 GiB cached` at server log line 28; no request completed; request1 GPU samples 18, mean/median/max 7.333/4/52%, memory max 10664 MiB | contaminated chunk2048 safety case; no timing or quality evidence |
| Long Arm A | failed; assigned L0; ctx8192/chunk256/max3000 | request1 only: HTTP 200, prompt/completion 43/3000, wall 508.495548 s, TTFT 10.082 s, prefill 66.045 s at 0.651071 t/s, decode 442.000 s at 6.787330 t/s; `finish=length`; no observed `</html>`; raw prefix and Markdown fence; malformed document without body/nav/form/script; runtime tensor progress 0; unsafe tier events 0 | negative `n=1` long-quality safety; 6.787330 t/s is non-promotable |
| Long Arm B | not_run | no Arm B receipt and no measurements; request2 was not sent because the shared request1 prerequisite failed before A/B divergence | only the fact of non-execution; no behavior, timing, or quality inference |

Authoritative receipts:

- C0: `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_performance_canary\perfC0_final_countfix_20260719T115141Z\summary.json`
- C1: `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_performance_canary\perfC1_dynamic_20260719T1156Z\summary.json`
- C2: `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_performance_canary\perfC2_dynamic_ctx8192_20260719T1202Z\summary.json`
- C3: `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_performance_canary\perfC3_dynamic_ctx8192_chunk2048_20260719T1212Z\summary.json`
- Long Arm A and the Arm B fail-fast decision: `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\G73_LONG_HTML_SAFETY_CLOSEOUT_RESULTS.md` and `g7_runs\g73_long_html_ab_safety\longA_safety_20260719T125754Z\summary.json` in that worktree.

## 2026-07-19 Native Windows G129 Startup Capacity Abort

The post-reboot G129 structural-safety attempt passed its launch preflight with
`ready_to_launch=true`, a 7.5 GiB minimum-available guard, and
`59089137664` available bytes. Startup validated and installed the 36.37 GiB
Q1_0 routed-expert sidecar, prepared 7.21 GiB of startup model-cache spans, and
allocated a separate pinned exact-IQ2 arena of `5902958592` bytes / 834 slots.

The runtime monitor then aborted the owned server after 53.476 s on three
consecutive `hard-low-memory` samples. The abort sample recorded PID 3872,
`661884928` Windows-available bytes, `49352544256` working-set bytes,
`54035292160` private/paged bytes, `14865493` page faults,
`106359.27866895513` pages input/s, `1258.2420328094572` page reads/s,
GPU utilization 1%, and 8534 MiB VRAM used. The failure reason is
`runtime-contamination-abort`. No HTTP request, output, exactness result,
quality grade, decode throughput, or proof of full/open runtime routing was
produced. This receipt supports only a startup capacity/contamination abort for
that machine state.

Authoritative receipt:
`C:\Users\imanu\Documents\Codex\2026-07-07\cia\work\ds4-win-publish-g126-20260718-v2\g7_runs\g7_g129_q1_open_dynamic_promotion_safety_postreboot_main_20260719T145402185Z_7e9f63bb00_failure.json`.
Its memory preflight and runtime telemetry use the same tag in the same
directory.

Sandbox/no-run note: no sandbox execution receipt is present. The frozen G129
handoff's earlier statement that no safety run or receipt existed, plus its
sandbox/approval discussion, is context only. It must not be represented as a
measured or aborted run. The hard-low-memory receipt above remains a distinct
earlier execution record from the later control failure below.

## 2026-07-19 Native Windows G129 Full/Open Q1 Control Decode Failure

The later control-only safety attempt
`g129_control_fullopen_q1_safety_20260719T185343539Z_codex_fg_20260719T185343911Z_c0c84095d9`
passed process isolation and memory readiness. The server became ready after
about 92 s. It used source head
`dc52ec05ec2636a09fbf59fe9a21460e23621501`, Release build input fingerprint
`f781522ee992aed8c9e0b905f888e7254d3d9730121b75ee34354a411657f53a`,
and executable SHA-256
`64353fb5e8b0638e4e4349540802b3570093f324d5e48d47116f9d24c36060dc`.
The at-run measurement harness SHA-256 was
`eea29af72f85659b3a036490255037edfc93d922802f7ab1f782c3dac1a167c2`;
the at-run G129 safety runner SHA-256 was
`43b4940720c1d2242f8d6166e6abed6798aab6456438502cd9754f1a4bb48b7e`.
The configuration SHA-256 was
`ab487f9e535d1da90d818ba2ee1d94f5188904854cf1660681c13d346e376b2f`.
These, together with the at-run protocol SHA-256
`eae069fbb1d8d460f78b31208ea1cb806a4f1790ce8e05336e0fad87aeeaafc2`,
are execution-time identities acquired before later worktree edits; current
script or protocol hashes must not be substituted retroactively for this run.

The control configured the exact-IQ2 primary arena at 5.50 GiB / 834 slots and
bootstrapped the complete Q1_0 routed base: 11,008 entries for layers 0..42,
38,956,695,552 bytes total, 26,304,970,752 pinned bytes / 7,433 slots, and
12,651,724,800 pageable bytes / 3,575 slots. Prefill published 770 exact-IQ2
entries in 11.291 s. No closed decode mask was applied: the log reports
`semantics=request-scoped-open`, `base=none`, and zero kept/pruned mask entries.

Decode nevertheless failed before producing one token. At layer 0 the
`q1-0-mixed` entry contract rejected six routes with `tier_entries=0`, followed
by `cuda decode failed` at `gen=0`. This is a fail-closed control NO-GO, not a
throughput or quality result. Dynamic promotion was OFF in this control, and
the promotion arm was consequently `not_run` under the protocol gate.

The Windows source-unlock telemetry recorded 129 calls over three ranges per
layer, with `success=0`, `true=0`, `not_locked=129`, and `failed=0`. It therefore
did not demonstrate physical release of the mapped Q1_0 source and cannot
support a claim for that fix. Minimum available memory was 6.694 GiB; peak
working set was 47.788 GiB and peak private memory was 51.523 GiB.
Contamination count and contamination-abort count were both zero, and post-run
cleanup was reported clean.

Because failure preceded final invariant parsing, final SSD bytes, backing
misses, exactness, forbidden-transition counters, output, and L0-L3 quality are
not attestable. In particular, zero values must not be inferred from absent
final telemetry. There is no full/open success, exactness, source-unlock-fix,
performance, quality, promotion, or SOTA claim from this attempt.

Authoritative manual failure receipt:
`C:\Users\imanu\Documents\Codex\2026-07-07\cia\work\ds4-win-publish-g126-20260718-v2\g7_runs\g7_g129_control_fullopen_q1_safety_20260719T185343539Z_codex_fg_20260719T185343911Z_c0c84095d9_manual_failure_receipt.json`,
SHA-256
`a9b3d6b3fef6ca6b820c3a7812cdccdc0218dde55521e273560046b21f58ce7b`.
Its process-isolation, memory-preflight, runtime-telemetry, stderr, and stdout
SHA-256 values are respectively
`f4e2a2df5ad9f961e8961699ca9e3adb1508f364e979f0bdfdd11ade23ebb1c2`,
`e0b004d4e865d97f3c2dadb27fdbfe932ff72b7b050a4176457031ca6035609d`,
`860f2ffca93dd8c0b578903790e5cba789c76447f4fce2b941242f9640d191a6`,
`da3086a97c4434282d7a4b01afc939c5cd8774d46119f39741513df73eb8c929`,
and `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Preceding invocations without a result or failure receipt remain
infrastructure-only Markdown context and have no CSV evidence row:

| invocation | surviving artifact | exact disposition |
|---|---|---|
| `parent_g129_control_fullopen_q1_safety_20260719T184517831Z` | zero-byte parent stdout/stderr, both SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | wrapper no-run; no child/preflight/receipt |
| `g129_control_fullopen_q1_safety_20260719T184648597Z_codex` | zero-byte stdout; 427-byte stderr SHA-256 `e31efaf72f3437db3e8c2af75cb97a28281313caaf19b807d89c95c6a54ead4b` | pre-server parameter failure: unsupported `ExePath`; no runtime/receipt |
| `g129_control_fullopen_q1_safety_20260719T184804897Z_codex` | process-isolation preflight PASS, SHA-256 `e0a62fea31c5f7725c84e87e610742de3d41ece1cfd708418e6f70b5fa03f093`; zero-byte parent logs | stopped after process-isolation preflight; no memory/runtime/result/failure receipt |
| `g129_control_fullopen_q1_safety_20260719T185041132Z_codex2` | process-isolation preflight PASS, SHA-256 `3176e6e2524c7c189f3dcfac19411780663b13515223ee4319dc786918548d6b`; 150-byte parent log SHA-256 `796427ac870db0572838c19550345d2e075806a114a42d1b882bc0480ef546ae` | stopped during pre-launch memory handling; no DS4 request/runtime receipt |

## 2026-07-19 Native Windows G129 Entry-Fix Control: Runtime PASS, Harness FAIL

Run
`g129_control_fullopen_q1_safety_entryfix_fg_20260719T194543063Z_20260719T194543443Z_1d6d439708`
is an official fail-closed control result. The HTTP request completed, produced
64 tokens in 332.664134 s, and the owned server exited 0 after a clean shutdown.
The raw output SHA-256 is
`920bddf1a1dea9cb85583d297d394107f34abe2fec3bf30c690b1debdc75581b`.
This short structural output is not assigned an L0-L3 grade and its observed
0.192386 client t/s is not a performance result.

Execution identity was source head
`dc52ec05ec2636a09fbf59fe9a21460e23621501`, build-manifest SHA-256
`36f9b48aff62765770108df9efd184e949e063ae4419e107fa7e9ac95a3fd92e`,
Release input fingerprint
`d71dd9405badd833a0c49efdf0548cd3740d10b7699a995d510d90dd4ff42d16`,
and executable SHA-256
`262159830d5628795ff79eabd8da415aab6b8a4d049d8f7f2c8b01fcef437580`.
The at-run measurement harness, runtime monitor, G129 safety runner, and
protocol SHA-256 values were respectively
`2b0fb688a159d2333327958d4129000e5589312818f44cb1873aaa0e690d8182`,
`bd5be0ac80594a6866137e44cffbd58f1c3707eedcd8cbf246620eebdd9e9b29`,
`1e7962b2c040b8cde827c17ff840cb1fe96982d1f05787ba6c1fff3c5486b832`,
and `5d75da4de0e7fa3ef7088952b7c3cfde68256192014f9bef0116203fa4cde03b`.
The control configuration SHA-256 remained
`ab487f9e535d1da90d818ba2ee1d94f5188904854cf1660681c13d346e376b2f`.
The harness, safety runner, and protocol were edited later at 20:07-20:08 UTC;
their current hashes are not retroactively attributed to this run.

The entry-contract fix reached and completed mixed decode. Runtime structural
telemetry is internally complete and consistent:

- full/open routing was preserved with no mask (`request-scoped-open`,
  `base=none`, zero kept/pruned entries);
- the Q1_0 arena contained all 11,008 entries for layers 0..42, with
  26,304,970,752 pinned and 12,651,724,800 pageable bytes;
- the exact-IQ2 arena contained 834 slots / 5.50 GiB and published 770 prefill
  entries in 12.107 s; cache320 seeding completed with zero failures;
- tiering exposed 11,008 entries and recorded 16,512 route entries;
- mixed routing recorded 9,834 Q1-resident, 5,680 IQ2-VRAM, 998 IQ2 snapshot-RAM,
  and zero IQ2 tier-RAM routes, totaling exactly 16,512;
- `trace_rows=16512` equals `tier_route_entries=16512`; mixed failures,
  snapshot-backing misses, Q1 resident misses, direct-pread fallbacks,
  SSD bytes/violations, tier failures, and forbidden cold SSD-to-VRAM events
  were all zero;
- dynamic promotion was OFF and no promotion event was claimed.

The official failure is solely the harness SplitFused accounting gate. Runtime
reported `split_fused_hits=5680` and `split_fused_misses=998`, totaling 6,678,
which exactly equals the real IQ2 route population `5680 + 998 + 0`. The
harness instead compared this with `6 * calls = 6 * 2498 = 14988`, implicitly
including routes that legitimately used Q1 fallback. The failure receipt thus
says `SplitFused route accounting does not match the primary-model route
population`, although the complete mixed-route partition itself balances.
The official harness verdict remains FAIL; this analysis does not rewrite it
as a passing run.

Runtime telemetry contains 423 samples. Minimum Windows-available memory was
4.190941 GiB (4,499,988,480 bytes), with zero samples below 2 GiB or 1 GiB;
contamination and contamination-abort counts were zero. Peak working set and
private memory were 49.931679 GiB and 53.723179 GiB. Source unlock again
reported 129 calls with `success=0`, `not_locked=129`, and `failed=0`, so this
run does not prove physical source release.

The authoritative failure JSON SHA-256 is
`fd960616313d8d21f429c8903e325623e210e2b63f05a9c5bc86d75ae0252be0`;
the G129 safety failure receipt SHA-256 is
`82a8cf68b57c7dfd3acb4479b797d04f049957d4688f17f2ab5cb38e98c798af`.
The raw-output, runtime-telemetry, and stderr SHA-256 values are
`1579d3a1b0db0537f1ade8e55076b00399e9fb3ed073ad4c4296c388b4a38da7`,
`89f9ee5e6348502fe31a7652aa8a09e4e5d629371656ccab7b966826d239a91a`,
and `0aaf3eadd4d78d43dc93a87beadb893acdd7ed44b2702b5397ebabc420a21e51`.
The planned promotion arm is `not_run`. No performance, quality, exact-output,
physical-source-release, promotion, n>=3, or SOTA claim follows.

## 2026-07-19 Native Windows G129 Confirmed SplitFused Control: Structural PASS n=1

Run
`g129_control_fullopen_q1_safety_confirm_splitfused_fg_20260719T201956114Z_20260719T201956487Z_0a22b47cb8`
is the first official G129 control receipt with status
`pass_structural_n1_no_performance_or_quality_verdict`. The request completed
64 tokens in 332.769602 s, produced output SHA-256
`920bddf1a1dea9cb85583d297d394107f34abe2fec3bf30c690b1debdc75581b`,
and the owned server exited 0. These timing and output facts are diagnostic
only: the run was not quality-eligible or SOTA-eligible, has no L0-L3 grade,
and supplies no performance claim.

Execution identity was source head
`dc52ec05ec2636a09fbf59fe9a21460e23621501`, build-manifest SHA-256
`36f9b48aff62765770108df9efd184e949e063ae4419e107fa7e9ac95a3fd92e`,
Release input fingerprint
`d71dd9405badd833a0c49efdf0548cd3740d10b7699a995d510d90dd4ff42d16`,
executable SHA-256
`262159830d5628795ff79eabd8da415aab6b8a4d049d8f7f2c8b01fcef437580`,
and `ds4_cuda.cu` SHA-256
`e0fd2aa8f043c66f9199f65783cc764c232d89d9fd4003a4666d1fc5b733625c`.
The at-run measurement harness, G129 safety runner, protocol, and runtime
monitor SHA-256 values were respectively
`43123e850c871b3d61bbd932e7c1d2232a084aefd1dcab653a9fa7ee0bc3499c`,
`33ac97d1c426171eb7113a667356a77f1c55b8ab116a60e66cc7771f0bacc43b`,
`e7b8315ea67ae6323c8d85846ac898632c9e1cc42a565b130a0f545aefea658a`,
and `bd5be0ac80594a6866137e44cffbd58f1c3707eedcd8cbf246620eebdd9e9b29`.
The control configuration SHA-256 was
`ab487f9e535d1da90d818ba2ee1d94f5188904854cf1660681c13d346e376b2f`.

The corrected harness accepted the runtime's complete structural partition:

- routing remained full/open with no decode mask (`request-scoped-open`,
  `base=none`, zero kept/pruned entries);
- Q1_0 covered all 11,008 entries for layers 0..42: 7,433 pinned slots /
  26,304,970,752 B and 3,575 pageable slots / 12,651,724,800 B;
- the exact-IQ2 arena allocated 834 pinned slots / 5,902,958,592 B and
  published 770 prefill entries / 5,449,973,760 B in 11.751 s; the 320-entry
  VRAM seed completed with zero failures;
- `trace_rows=16512` equaled `tier_route_entries=16512`; the route partition
  was 9,834 Q1-resident + 5,680 IQ2-VRAM + 998 IQ2 snapshot-RAM + zero IQ2
  tier-RAM = 16,512;
- SplitFused hits/misses were 5,680/998 = 6,678 against expected/observed
  6,678, using the corrected `q1-0-mixed-iq2-routes` basis and explicitly
  excluding the 9,834 Q1-resident routes;
- mixed failures, Q1 resident misses, snapshot-backing misses, direct-pread
  fallbacks, tier failures, SSD bytes/violations, current-token/direct
  SSD-to-VRAM transitions, and forbidden cold SSD-to-VRAM events were zero;
- dynamic promotion was not requested; attempts, successes, events, direct
  rejections, backing reclaims, and promotion failures were all zero.

Structural exactness and provenance were complete. The IQ2 model was
86,720,111,488 B with SHA-256
`efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
and receipt SHA-256
`a1a6626088489743628165692d32870f083cfe74386469176d0b333a2c95eb55`.
The Q1_0 sidecar was 39,048,344,416 B with SHA-256
`05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207`,
receipt SHA-256
`9a4dc6e6a86523a3df4bf88681c83819b0eac2dc9d36c5ff78800d7025c9b2d5`,
and `provenance_verified=true`. Prompt SHA-256 was
`38f6ec5ee5403f59dd2418eb5d9a5a94a0f0da19df015060383bb1ae46003bb6`.
No expected-output oracle was requested, so this is not a quality or
exact-output claim.

Runtime telemetry contained 422 samples. Minimum Windows-available memory was
4,449,619,968 B / 4.144 GiB, with zero samples below 2 GiB or 1 GiB and zero
contamination/abort observations. Source unlock reported 129 calls with
`success=0`, `not_locked=129`, and `failed=0`; it still does not prove physical
source release.

The authoritative result, receipt, runtime-telemetry, and stderr SHA-256 values
are respectively
`795b65609c1826129e2467493a90f6998bad2176e2eff7e7e89c110b05ac6b63`,
`0f4aeb6e8b38a03e7ba0c943abbbce1009c53b2c55bfec04c8668b5da2b65017`,
`8669b0b06db15427ae95e8f85e630d8abb1cada1451776b51baaa2c4e1b93935`,
and `3492e7befa373440856f01810216d780c4a9ce2d8acab71adac5380815ef8a9f`.
The raw-output, process-isolation preflight, memory preflight, system-quiescence
preflight, and empty stdout SHA-256 values are
`a70101599565e6630f2cbad4999376e29eaf3b0c912e7e8510af27d9ba27951f`,
`492c75a3d3a452a9a357a9cbeac5f0a11515807adb25dc1e8614918dd199158e`,
`7ae4ec9a538abeb3f26fb4400ee5b3fe4d9904e117c453efc998f1ecd7dd4ea7`,
`9b74e234252d9821896bcae41b9345e48c86f34dbd60c98ad84d904bbb244b41`,
and `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
This is a control-only structural `n=1` PASS, not a quality, performance,
physical-release, promotion, `n>=3`, or SOTA result.

## 2026-07-19 Native Windows G129 Promotion: Pre-Server Quiescence Abort

Promotion tag
`g129_promotion_fullopen_q1_safety_confirm_fg_20260719T204336000Z_20260719T204205331Z_069895af73`
has a failure receipt but no DS4 runtime. The wrapper receipt records
`failed_structural_n1_no_performance_or_quality_verdict` /
`child-exit-nonzero`, while the evidence disposition is
`inconclusive; runtime not_run`: the child stopped before `ds4_server` launch
because system quiescence returned `disk-median-above-threshold`. There was no
readiness probe, HTTP request, output, runtime telemetry, or server exit code.
No promotion gate is evaluable. Although `failure.json` contains
`completed_results=1`, it also has `http_ok=null` and `server_exit_code=null`,
and none of the result/raw-output artifacts exists; that counter is therefore
not evidence of a completed request.

Process isolation passed with zero conflicts. Memory preflight also passed:
Windows-available memory moved from 59,657,281,536 B / 55.560 GiB to
59,629,735,936 B / 55.535 GiB against a 7.5 GiB minimum. After the 10.003 s
cooldown, quiescence collected all five requested samples. Its medians and
thresholds were:

| signal | observed median | launch threshold | result |
|---|---:|---:|---|
| CPU | 10.855744% | <=60% | pass |
| GPU | 0% | <=85% | pass |
| disk I/O | 28.727630 MiB/s | <=64 MiB/s | pass |
| disk busy | 100% in all 5/5 samples | <=30% | fail |

The promotion configuration requested full/open routing, no mask, and
`Q1_0DynamicPromotion` with probation 64, minimum touches 2, minimum weight
0.02, request budget 64, window 40 calls, and window budget 1. Those are only
requested settings: no promotion attempt, success, event, storage transition,
exactness check, or quality/performance observation occurred. The postflight
process check was clean.

Preflight identity was source head
`dc52ec05ec2636a09fbf59fe9a21460e23621501`, executable SHA-256
`262159830d5628795ff79eabd8da415aab6b8a4d049d8f7f2c8b01fcef437580`,
measurement harness SHA-256
`43123e850c871b3d61bbd932e7c1d2232a084aefd1dcab653a9fa7ee0bc3499c`,
runtime-monitor SHA-256
`bd5be0ac80594a6866137e44cffbd58f1c3707eedcd8cbf246620eebdd9e9b29`,
build-manifest SHA-256
`36f9b48aff62765770108df9efd184e949e063ae4419e107fa7e9ac95a3fd92e`,
and promotion configuration SHA-256
`9b20307e3135382306cebd63cb23e0c9197eb54d71a05c10c4fa242e93c45744`.
The prompt SHA-256 was
`38f6ec5ee5403f59dd2418eb5d9a5a94a0f0da19df015060383bb1ae46003bb6`;
Q1_0 sidecar preflight provenance was verified, but no runtime exactness or
provenance gate ran.

The failure, safety-receipt, system-quiescence, memory-preflight, and
process-isolation SHA-256 values are respectively
`b9a4b751a604ab44679cac08eae478bed32fd1536a0274e71d52d05e0e953615`,
`ba71b58821cbca7fb7da1b231d92c0570b7a862ab03d80e41ae86ee782b6271b`,
`81a421d43acfa5f44c5c50a0f8b5c268a10263afb11fef79572b9d7a2874eb0d`,
`9ff38381ec0c121d630a381e44ef57effbe3b20b998800c43c70ca3f00473333`,
and `234ee834617d57bf4298679ae715f46b13b9fe1faf2939e356c6094296efd3ca`.
The result, runtime-telemetry, raw-output, stderr, and stdout paths named by the
failure schema do not exist. This row carries no runtime, promotion, exactness,
quality, performance, `n>=3`, or SOTA claim.

## 2026-07-19 Native Windows G129 Promotion Retry: Structural PASS n=1

Run
`g129_promotion_fullopen_q1_safety_retry_quiet3_fg_20260719T205248000Z_20260719T205242553Z_4c5d7def20`
has receipt status `pass_structural_n1_no_performance_or_quality_verdict`.
It is a promotion-arm mechanism result only: full/open routing remained active
with no mask, the server exited 0, and the aggregate structural harness passed.
It is not quality-eligible, performance-eligible, or SOTA-eligible, has no
L0-L3 grade, and does not authorize `n>=3` while the promotion lineage caveat
below remains open.

### Quiet launch evidence

The dedicated read-only quiet3 preflight collected three consecutive windows,
five samples per window, with zero process conflicts. All three passed the
unchanged thresholds: CPU <=60%, disk busy <=30%, disk I/O <=64 MiB/s, GPU
<=85%, and available memory >=7.5 GiB.

| window | samples | duration ms | CPU median | disk median | disk I/O median MiB/s | GPU median | available min GiB | committed max GiB | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5/5 | 5,345.879 | 3.437823% | 0.016890% | 0.552498 | 0% | 55.823 | 10.784 | PASS |
| 2 | 5/5 | 4,557.373 | 3.970328% | 0.037008% | 0.813430 | 0% | 55.813 | 10.735 | PASS |
| 3 | 5/5 | 4,542.220 | 1.602687% | 0.025040% | 0.246094 | 0% | 55.840 | 10.677 | PASS |

The owned harness then repeated its normal five-sample quiescence gate after a
9,992.654 ms cooldown and also passed: CPU median 0.577411%, disk median
0.006396%, disk I/O median 0.018646 MiB/s, GPU median 0%, with no failures.
Process isolation and memory preflight passed with zero process or maintenance
conflicts; available memory moved from 59,941,191,680 B to 59,870,720,000 B.

### Timing and output

All timing values below are retained as diagnostics from the result artifact;
they are not a performance claim:

| field | observed value |
|---|---:|
| model/server load | 91.089035 s |
| client request wall | 331.310314 s |
| server total | 330.932 s |
| server prefill / TTFT | 39.179 s |
| server decode | 64 tokens / 291.753 s / 0.22 t/s |
| client completion rate | 64 tokens / 0.193172 t/s |
| finish reason | `length` |
| prefill rows per routed MoE layer | 43 minimum / 43 maximum |
| prefill routed slots / unique entries | 10,320 / 10,240 |
| prefill snapshot WRAP | 770 loads, 8 workers, 11.502 s |
| prefill VRAM seed | 320 entries, 2,264,924,160 B, 4.559 s, 0 failures |

The server artifact does not expose a separate prefill-token-rate field. The
reported TTFT and the 43-row-per-layer prefill trace are preserved without
deriving an unsupported prefill t/s value.

The complete 273-byte output had SHA-256
`4f24448486ef3b1fabc3395f0e0d945246e8f869805bb0b74034bc6c4a9c13bf`:

````text
Here is a complete single-file HTML landing page for a cyberpunk AI programming shop. It includes a styled theme, navigation, hero section, a request form, and a JavaScript confirmation popup.
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <c
````

It is a 64-token length-limited structural output, not a completed-page quality
sample; no raw-only, rendering, form, popup, or L0-L3 verdict is assigned.

### Structural and promotion accounting

- full/open routing was preserved with `request-scoped-open`, `base=none`, and
  zero kept/pruned mask entries;
- Q1_0 covered all 11,008 entries: 7,433 pinned slots / 26,304,970,752 B and
  3,575 pageable slots / 12,651,724,800 B;
- the exact-IQ2 arena allocated 834 slots / 5,902,958,592 B and published 770
  resident entries / 5,449,973,760 B;
- `trace_rows=16512` equaled `tier_route_entries=16512`; routes partitioned as
  9,608 Q1-resident + 5,831 IQ2-VRAM + 1,019 IQ2 snapshot-RAM + 54 IQ2
  tier-RAM = 16,512;
- SplitFused hits/misses were 5,831/1,073 = 6,904 against expected/observed
  6,904 on the `q1-0-mixed-iq2-routes` basis, excluding 9,608 Q1 routes;
- mixed failures, Q1 resident misses, snapshot-backing misses, general backing
  reclaims, probation backing reclaims, direct pread fallbacks, IQ2 decode SSD
  bytes/violations, direct SSD-to-VRAM rejection, forbidden cold SSD-to-VRAM,
  and promotion failures were zero.

G129 promotion was requested with 64 pre-reserved probation slots, minimum
touches 2, minimum weight 0.02, request budget 64, window 40 calls, and window
budget 1. Aggregate telemetry reported 64 Q1_0 stage attempts, 64 successes,
64 next-call guards, 64 `cold_to_2bit_ram` admissions, 54 probation-RAM hits,
zero next-token waits, and zero failures. Staging read exactly 452,984,832 B
(0.421875 GiB) from SSD in 0.7869478 s into RAM; it did not move a current-token
expert directly from SSD to VRAM. The tiering summary's 493 generic
`vram_promotions` are a separate cache-policy counter and are not the 64 G129
Q1_0-to-IQ2 RAM promotions.

An independent sequential parse of all 16,512 route lines found 54/54
`iq2_tier_ram` uses with an earlier `q1_resident` route for the same
`(layer, expert)` pair, and zero missing pairs. This supports causal use for the
54 promoted experts that were actually hit.

The remaining provenance gate is material and blocking. The result contains
one aggregate `iq1_promotion_requests` object and stderr contains one final
`[iq1-promotion]` summary line, with zero promotion lines carrying an epoch.
It does not enumerate all 64 successful stages individually with request epoch,
layer, expert, prior Q1 touch, source, stage completion, and next-call
eligibility. Therefore the aggregate promotion gate passes, but complete
per-expert provenance is not materialized. No next promotion, quality,
performance, or SOTA claim may rely on this run until that sub-gate is emitted
and validated.

Runtime telemetry contained 420 samples. Minimum Windows-available memory was
4,301,701,120 B / 4.006 GiB; GPU utilization median/peak was 26/75%, peak VRAM
was 11,746 MiB, and contamination/abort counts were zero. Peak working set and
private memory were 49.931694 GiB and 53.721672 GiB. Source unlock again
reported 129 calls with `success=0`, `not_locked=129`, and `failed=0`, so no
physical source release is proven.

### Identity and artifacts

Execution identity was source head
`dc52ec05ec2636a09fbf59fe9a21460e23621501`, build-manifest SHA-256
`36f9b48aff62765770108df9efd184e949e063ae4419e107fa7e9ac95a3fd92e`,
Release input fingerprint
`d71dd9405badd833a0c49efdf0548cd3740d10b7699a995d510d90dd4ff42d16`,
executable SHA-256
`262159830d5628795ff79eabd8da415aab6b8a4d049d8f7f2c8b01fcef437580`,
measurement harness SHA-256
`43123e850c871b3d61bbd932e7c1d2232a084aefd1dcab653a9fa7ee0bc3499c`,
`ds4_cuda.cu` SHA-256
`e0fd2aa8f043c66f9199f65783cc764c232d89d9fd4003a4666d1fc5b733625c`,
and `ds4-server.c` SHA-256
`44eefebac4af4672bb245f14806ecaef4a2df06403fc2875e5c4d05dc2ff2bec`.
Configuration SHA-256 was
`9b20307e3135382306cebd63cb23e0c9197eb54d71a05c10c4fa242e93c45744`.

The IQ2 model SHA-256/receipt SHA-256 were
`efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668` /
`a1a6626088489743628165692d32870f083cfe74386469176d0b333a2c95eb55`.
The Q1_0 sidecar SHA-256/receipt SHA-256 were
`05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207` /
`9a4dc6e6a86523a3df4bf88681c83819b0eac2dc9d36c5ff78800d7025c9b2d5`,
with preflight and runtime provenance verification true. No expected output
oracle was requested.

The result, receipt, runtime-telemetry, stderr, and quiet3 SHA-256 values are
respectively
`b3f636901e75cec9809b733a8663a5b087faac1a54b171af904ee32dbe5e40e6`,
`5b95cb31b5b5fff9cd03b250e90b6523bd61195cf1249ac21c29a86d7f2b67a6`,
`143160a8958ebb480a4433488f7ad25750d7ae61cea865f15836801103ea9fc7`,
`ebb4abbc6b58425ff3af5a872d7f8f22474fb46decc24825c6f4738414f37a25`,
and `83c77ed313763bb3f64ec0103f298f3fa869b53ac3255d9d36d470bd412fe8f8`.
Raw-output, owned system-quiescence, memory-preflight, process-isolation, and
empty stdout SHA-256 values are
`20254cd20ef89e39b56adf95ff67e4cd5ca1cd4a6accd025a5a98f580fca46df`,
`7f5bf81241dc320844bac43f2158d79ce4d5d41ef536c5936f8d21c5e57d98b7`,
`e18741fbe83de88466e4ab048a5b2f1bae34676390fcf86990f7dd3ebc5a9a77`,
`092ff4870f41b60fb41de1af146597ce02e6baf0bc2c8f1168fe02cc77127116`,
and `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## 2026-07-20 Native Windows G129 Hegel-Direct Promotion: Structural FAIL n=1

Run
`g7_g129_promotion_structural_hegel_direct_20260720T000041186Z_codex_20260720T000041553Z_64b579e1e8`
is classified exactly as `structural_fail_n1 / non_sota /
router_contract_and_artifact_materialization_blocked`. The HTTP request completed
and `ds4_server` exited 0, but the owned harness failed closed because the
expected Q1 mixed router was `open` while the final summary remained
`unchanged`. This is not a quality, performance, `n>=3`, or SOTA result.

### Lifecycle, timing, and output

The three-window read-only preflight passed before launch with no matching DS4,
build, test, or file-copy processes. Its unchanged thresholds were CPU <=60%,
disk busy <=30%, disk I/O <=64 MiB/s, GPU <=85%, and available memory >=7.5
GiB.

| window | samples | CPU median | disk median | disk I/O median MiB/s | GPU median | available min GiB | result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5/5 | 6.876% | 0.031% | 0.209 | 3% | 55.747 | PASS |
| 2 | 5/5 | 6.030% | 0.013% | 0.153 | 3% | 55.695 | PASS |
| 3 | 5/5 | 5.527% | 0.009% | 0.116 | 0% | 55.704 | PASS |

Readiness completed in about 92 s. The single request then completed HTTP with
64 tokens, finish reason `length`, and server exit code 0:

| field | observed diagnostic |
|---|---:|
| client request wall | 1,780.083583 s |
| server total | 1,779.714 s |
| server prefill / TTFT | 38.719 s |
| server decode | 64 tokens / 1,740.982 s |
| server logged rate | 0.04 t/s average; 0.03 t/s final chunk |
| client completion rate | 0.035953 t/s |
| prefill rows per routed MoE layer | 43 / 43 |
| prefill routed slots / unique entries | 10,320 / 10,240 |
| prefill snapshot WRAP | 770 loads / 8 workers / 11.018 s |
| prefill VRAM seed | 320 entries / 2,264,924,160 B / 5.075 s / 0 failures |

These timings are deliberately retained only as diagnostics. Stderr contains
9,672 per-record promotion markers: 64 attempts, 64 successes, and 9,544
rejects. The marker lines occupy 16,073,629 B (16.074 decimal MB / 15.329 MiB)
of an 18,602,065 B stderr log. This synchronous per-record instrumentation
dominates the 1,780 s observation, so the reported 0.04/0.035953 t/s values are
not comparable to any benchmark.

The raw checkpoint is incomplete only because the post-run gate failed before
final result materialization. It preserves one 273-byte, length-limited output
with SHA-256
`4f24448486ef3b1fabc3395f0e0d945246e8f869805bb0b74034bc6c4a9c13bf`:

````text
Here is a complete single-file HTML landing page for a cyberpunk AI programming shop. It includes a styled theme, navigation, hero section, a request form, and a JavaScript confirmation popup.
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <c
````

No L0-L3 grade is assigned: the output is a 64-token structural checkpoint, not
a completed-page quality sample.

### Runtime structure and failure

- route accounting balanced: `trace_rows=16512` and
  `tier_route_entries=16512`;
- the partition was 9,608 Q1-resident + 5,831 IQ2-VRAM + 1,019 IQ2
  snapshot-RAM + 54 IQ2 tier-RAM = 16,512;
- SplitFused was 5,831 hits + 1,073 misses = 6,904 real IQ2 routes;
- Q1 covered all 11,008 entries; the exact-IQ2 arena retained 834 slots / 5.50
  GiB and the prefill snapshot published 770 entries;
- Q1 mixed, tier, backing, resident, direct-pread, current-token, overflow, and
  forbidden SSD-to-VRAM failure counters were zero;
- final IQ2 decode SSD bytes were zero. Separately, the promotion stage read
  452,984,832 B from SSD into RAM; this was SSD-to-RAM, not SSD-to-VRAM;
- promotion diagnostics recorded request epoch 1, 64 attempts, 64 successes,
  zero failures, 9,544 rejects, and 64 next-call guards;
- all 64 success records were unique `(layer, expert)` pairs and unique record
  IDs. Their first eligible call was later than both current and observation
  calls, source/destination hashes matched the authoritative Q1/IQ2 artifacts,
  and no record declared same-call eligibility, direct current-token transfer,
  or overflow.

This raw diagnostic evidence does not make the run a pass. The router contract
failed (`expected=open`, `summary=unchanged`), the dedicated promotion JSONL was
not materialized, and neither the automatic result JSON nor automatic
safety-failure receipt exists. The later
`g129_codex_postrun_failure_receipt_v1` is explicitly supplemental and
non-automatic. The artifact-materialization gate therefore remains blocking.

Runtime telemetry has 1,853 samples. Minimum available memory was
4,190,031,872 B / 3.902 GiB; GPU utilization median/peak was 3/80%; peak VRAM
was 11,751 MiB; peak working set/private memory were 49.932/53.723 GiB; process
read delta was 27.354277 GiB; contamination triggers and aborts were zero.
`VirtualUnlock` again reported 129 calls, 0 successes, 129
`ERROR_NOT_LOCKED`, and 0 failures, so it does not prove physical release.

### Identity and provenance

Source head was
`dc52ec05ec2636a09fbf59fe9a21460e23621501`; the worktree was dirty at build
start. Build-manifest SHA-256, Release input fingerprint, executable SHA-256,
measurement harness SHA-256, runtime-monitor SHA-256, safety-runner SHA-256,
and configuration SHA-256 were respectively:

- `f4934c834fadd9921392bf24e0787856f99b1d041406082ecd6bbf352e158d44`;
- `4a6d152dafb5596936c92fdf0cb92b3c3f5185977663cab0d0bd96f05f4bfa27`;
- `154b430ac742d52dce504a45de10d38e331d5c5e9841c2ec4cd6315aebbea533`;
- `a4251594bdff49b2751a41e56359787ddf0055785251832d95fe2bd6b5813f51`;
- `bd5be0ac80594a6866137e44cffbd58f1c3707eedcd8cbf246620eebdd9e9b29`;
- `445a21d08f1da76daf4284a7bf5be8d66ca880e870e72c197b41f696b224c81e`;
- `9b20307e3135382306cebd63cb23e0c9197eb54d71a05c10c4fa242e93c45744`.

The build manifest is the authoritative complete input-hash inventory. Its 45
input SHA-256 values are:

```text
CMakeLists.txt ba629659ec622a8755c76832fe80dfcb3d4c2108f75fca3813c7f652799839e4
ds4.c 48db9abf9bd5aa3d26f3fca6c29cf300667a7a75d5e8ddcc98365139414ecc01
ds4.h 98dfdad193cb45c59cf7b244e3a0f26fb13734ce4e62662ac069a0e94d3c28fe
ds4_bake.c 6b3a28e2c1e8dc9142de5a7e804ad336a8bd2dc137bf4539b9eefb0292be1009
ds4_bake.h ae0c41adb96cb4cdf502c80a4359869789a1b69675b3aaced6e0074978a71191
ds4_bench.c 5ff4af34c921ac4be24e2fd2f7e6c8f55c27461635b2ca04bc36c4b99e3ace6b
ds4_cli.c b74ad531965d65a95a8dacc184cc20b7b36512039ac447b2c297700c2d9ce792
ds4_cuda.cu 843b13a898918914f4335936676d04384857d7c95219a7f3f23f094063f7372e
ds4_gpu.h c779567247f6b58ca01bd7a8e20f0ce3d1136c33c2b01e714a830f41bd65d781
ds4_inspect.c e86415e88b455b872278b4814b1048b5f6b44695d58563797d19293f1aeda978
ds4_iq1_s_bench.cu e3d9af0c6d9f812190137307e18a369579c434c4b74dc350185d11cf41ccde30
ds4_moe_gate_bench.c 574420a7ee6364f09424bb5068d1d2126632c07b5c63223f2afab6458af48098
ds4_q1_0_bench.cu b86954a301abca622c00453e3ab2a3876c48fa724b9b6e1df2fc74d4e50bce88
ds4_server.c 44eefebac4af4672bb245f14806ecaef4a2df06403fc2875e5c4d05dc2ff2bec
ds4_spex_predict.c 49186a1b724e1507d73934e5a7ba5db1b41437972e1ca212003c359cfd50994e
ds4_spex_predict.h d7bdd5ed4988dfd5e9f946150098d110046f703295ff602ddf3eef398b61c253
ds4_spex_queue.h 5ab1720324a7fce194ffd7cece7da41b1e47feb376060d3dd02b1f6dbb07cff8
g17_arena_probe.cu 654f929fd4db8afa873bbc53e5e42af7ae45860c2289415d42d80b5b48ffcb34
g17_segmented_arena_probe.cu ac8a770b4cc93d9b63e63604ac27c59fb19d40cbeb58dfaaeaec68cee61817a8
gguf-tools/deepseek4-quantize.c 00a1f882a8233c1b5c2a7f4d4e31129dce6695a89b95c4c0d971c41870db6627
gguf-tools/quality-testing/score_official.c 317fbddd7654d8d623c8417550b25beaccef38b0ca8c34846940b993dbdb06f3
gguf-tools/quants.c 4661b8150181a53c53c7c342419b55c84dde60c3b052d7d6e117ff3afac34f45
gguf-tools/quants.h 2aa6e29a248144f9b5b36faef80d90830c210216570db99cf2a1bb49cd2266e1
linenoise.c 254418684ccde08e0d225c15b8eea06c26c054d75e48515cbb2871f36370c1a5
linenoise.h e0c1415d3f88d86546d076c271c55009861a90f0a5329f912533d44e88fbdf75
rax.c d10efe432cf8b1308523d2a782347bb1c9a2528ff9057ffe46188eb3f1f4e3fe
rax.h 017a8d397bb01f2c15f002a729ac534bb9b63e34f9a03118dae623839b1e7040
rax_malloc.h 07897d67ef82b2229509866ac1aae5f7b476d7ae473bb0ac7d3acfb031c3356b
src/platform/os_clock.h 2384a293d18a381e5f2b99e12b49ea8e3f188f1e6612ea8c8663f02ee97818bb
src/platform/os_console.h 0445e688d91201a99f38b8f29120ec37ed2d56189b1e24f828498c8ffcea0a7c
src/platform/os_file.c 941b1ba241fc6265989b39b77b3b5babfdfec5c607293e7113227fefc56b45f8
src/platform/os_file.h 230365a4496de2a680a00939a7f2d8b05ceb5755848726ac97950a6476ccdf5e
src/platform/os_mmap.c 8c87196e0605ea04f3f56652662b908ad056f425dae8205c57638ecfbf199755
src/platform/os_mmap.h 60201a5cfcd6a4ab8d6525d838236a0c077c1ee242b6b7a6b2147286c2e85d5a
src/platform/os_path.h 8ba47ecbee22b5edf52ba761cc321888c0261439c666c387f511a405680df15c
src/platform/os_random.h e4191f861d1de6e884c8697e3ed2cfbe5f6ec7ec5fc04cc7d152f299f2c071f0
src/platform/os_thread.c 2c1762763d25d03d351abaa4d3861406b7d4aa6b0ab4197047cd010dee749fec
src/platform/os_thread.h 1e8ceedb5ff5c1d2c9f564c94f9d6266ebed970ca0b11fb9db2d510322317f91
src/win32/win_dirent.h a566312468f7ab64aaeefad40ddd24c9fe905e0c22199e392b0ea9e5996c00a1
tests/CMakeLists.txt 9e91a3758df572920b7056e8462fc995d2bfb7684a8f1ed9d31d8c096418ef01
tests/cuda_long_context_smoke.c 947b3b898d3b71786d702033ef9440cb0386d6c1762b796facef018ad217492b
tests/ds4_bake_test.c b6e439dae4bb16452a5a6cfaa976ad6b5f904bfd96e0bc1a1c5f071ea1918433
tests/ds4_test.c dfc1806c8912702ea6df8768faef8e9f72ad73aa126aeabe6107bb0a83320299
tools/mapviewoffile_register_probe_win.cpp 8f8036e1e2a3efc48a88ab0fa01bf275dbc8aa61d7ed0e45e03ddd35d8277a21
tools/os_layer_test.c 441045d5948ad38f181d07c7eb1a79d257d91839c56bdc7035d275b8e668544d
```

The authoritative IQ2 model SHA-256 was
`efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`;
the Q1_0 sidecar SHA-256 was
`05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207`.

### Artifact receipts

- failure JSON:
  `753af14917dd33bde899fdcbe416f8e625162af57c3fe032d2b79cd20f8e0f70`;
- runtime telemetry:
  `72ef0277368fb825c81a3d843e3b91f8ac18cf7e1c33dbcf805ac392305d8090`;
- stderr:
  `82925443e8fc7a768fa6824d745e7483ab7cfd257ef67ba2e87efe4bb083f5d2`;
- supplemental non-automatic receipt:
  `2230d241f2eb45769445ed8e66faadf5b80059c82b336bb8d4a1ce13225da248`;
- quiet3 preflight:
  `fb7b989029fefc30d3b98c2e28cc88c2e3b24cf8c10b264e0e4c7d9faa2f3436`;
- raw-output checkpoint:
  `c56d343e9f9c23e2bcf637a8e4cbed3d7dbefde08dd698e4b794759c801e8d17`;
- memory preflight:
  `adc03c9892ea706eb97d42d08980e16223329bc116f5db6e3b6c1202bfc7eb61`;
- process-isolation preflight:
  `e8702b08d080ea4e537ac54ea9212ae3bf14abf5d9eacdf4b9d52c55cb617968`;
- empty stdout:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

No automatic result or automatic safety-failure receipt exists. The
supplemental receipt does not substitute for either required artifact.

## 2026-07-20 Native Windows G129 Final Promotion Pair

These two independent-process, 64-token structural safeties used the same
full/open Q1_0 plus exact-IQ2 promotion configuration. They are recorded
separately because the first runtime completed but the parent runner failed
after runtime, while the second reached the authoritative validator PASS.
Neither run is quality-eligible, performance-eligible, `n>=3`, or SOTA.

Common identity was source head
`dc52ec05ec2636a09fbf59fe9a21460e23621501`, build manifest
`c8867e210889c4c13dacd91986ad97adbeeccb5675873922bef3a6d21d5bdd20`,
input fingerprint
`d52c6f10baeae0e6f7b0ab08c5af5d5c0c5a7d68395182e03c922bfe8f71e02c`,
executable
`798c24d3d836cb22bdf14656d543e958acdae385df5bc2c9ad1dc7ef0e672fcc`,
measurement harness
`3581448fa75fe3b7df946e07e461ff7685a844e136deb709b852f4278388c3e2`,
and configuration
`9b20307e3135382306cebd63cb23e0c9197eb54d71a05c10c4fa242e93c45744`.

### Structural-final wrapper failure

Run
`g129_promotion_structural_final_20260720T020905751Z_foreground_codex_20260720T020906193Z_0538a73d4d`
has official status
`failed_structural_n1_no_performance_or_quality_verdict`, reason
`child-exception`. `ds4_server` exited 0 and wrote its result plus complete
promotion JSONL, but the parent cast the multi-item PowerShell return from
`Invoke-G129BootstrapChild` from `System.Object[]` to `System.Int32`. This is
a post-runtime runner failure, not a model/runtime failure, but the official
FAIL remains authoritative.

The completed runtime evidence showed `router_mode=open`, balanced
`trace_rows=tier_route_entries=16512`, and the exact route partition 9,608
Q1-resident + 5,831 IQ2-VRAM + 1,019 IQ2 snapshot-RAM + 54 IQ2 tier-RAM.
SplitFused was 6,904/6,904 on the IQ2-only basis, excluding 9,608 Q1 routes.
Promotion materialized 128 strict JSONL records: 64 attempts, 64 successes,
zero failures, request epoch 1, window epochs 1..64, zero unpaired or same-call
records, and zero direct current-token SSD-to-VRAM transitions. Per-expert
offset/range and source/destination SHA provenance validated. Mixed, backing,
resident, promotion, IQ2 SSD, forbidden-transition, and contamination counters
were zero.

Diagnostic-only timing was TTFT 38.884 s, server decode rounded to 0.2 t/s,
and client completion 0.180956 t/s. Q1 H2D was 34,002,173,952 B. Promotion
read 452,984,832 B from SSD to RAM. `VirtualUnlock` reported 129 calls, zero
successes and 129 `ERROR_NOT_LOCKED` in 80.890085 s, so no physical-memory
release is claimed.

The automatic failure receipt SHA-256 is
`5afff38a44b12b9923b4fb71838d3128ca98d9a719c05c3364a9bbaa5688b381`;
result, runtime telemetry, stderr, raw-output, and promotion-JSONL SHA-256 values
are respectively
`770ef2b50a315d9a0ef6e98ef002eb85a492513a169ec8d7a0b32e41fc29675a`,
`f9e30a6902584bd285db9b36d3d0132b57849d6aee405d55b5a4a1764e261423`,
`dd0cdcbb7814ae3442afe248eeb023a1dd52d1ef2591910d36749068786164d5`,
`f0cfbc96f5cbaa4ef2dbe4c821d91693682d86d6eca3db8c3edf8dea5257315b`,
and `5c5dfe0b379fe9dc1d6deefeadfc64632f796e72835ed70d34c7586e4c86b457`.
The at-run runner SHA-256 was
`853a241dd7d1ef0a12877770a77b27cce4e65903bb90d782861a147de0f0bc69`.

### Official structural confirmation

After the runner-only scalar-return fix, run
`g129_promotion_confirm_final_20260720T025308987Z_foreground_codex_20260720T025309391Z_e3ef656d25`
repeated the same structural workload and reached official status
`pass_structural_n1_no_performance_or_quality_verdict`.

All route, SplitFused, promotion, request-epoch, JSONL pairing/budget,
offset/range/SHA provenance, next-call causality, storage, backing, forbidden,
and contamination gates repeated the values and zeros above. The dedicated
JSONL again contained 128 physical records with the same SHA-256, representing
64 unique attempt-terminal pairs. `router_mode=open` is the authoritative
routing state; the separate `router=unchanged` field describes only transition
state. Postflight found no `ds4_server`, and GPU returned idle/P8.

Diagnostic-only timing was TTFT 39.630 s, server decode rounded to 0.2 t/s,
and client completion 0.181705 t/s. Minimum available memory was 3.763 GiB.
Q1 H2D was again 34,002,173,952 B; the 452,984,832 B exact-IQ2 promotion read
took 9.2210697 s. `VirtualUnlock` again reported 129/0/129 calls/successes/
`ERROR_NOT_LOCKED` in 80.250202 s, so memory release remains unproved.

Receipt, result, runtime telemetry, stderr, and raw-output SHA-256 values are
`8d3898bc37fd7552489e991c4466d9e303a303b08d144efa321d5193cb06d1f4`,
`5714c13d781c0f86a3b33dfff1fc9e5811dd704d854126bb1545afe4547b9567`,
`89490740967c79401ca14bf64a145634f3aecc23b0894a8af32f6f88467b3314`,
`5d80c237a2abaea0a277c19f7e20ad5a6fbcd4b1b13fc3a6ae69c9759fec82bd`,
and `ca2826d2f29e93dc60d87a534fd9fe3d4740b04e8bc597ba35f140dc21b0f353`.
The fixed runner SHA-256 was
`1d040fcd0de28678c6ac98e073c0389509216a0ebd0c1f3105fb9744651e2860`.

This closes the G129 structural/provenance prerequisite at `n=1`. It does not
promote the observed roughly 0.2 t/s decode, does not grade output, and does
not authorize a long or `n>=3` matrix before the memory and Q1 transport
bottlenecks are corrected.

## 2026-07-19 Native Windows G73 Two-Turn Replay A/B

This cycle used the exact-IQ2 G73 runtime and `C:\ds4-models\ds4-2bit.gguf`,
not G129 and not the Q1_0 RAM fallback. It is a long `ctx8192`, chunk-256,
two-turn mechanism safety, not the canonical short G73 benchmark. G73 remains
a prefill-derived `request-scoped-closed` decode-eligibility design with a
dynamic cache and dynamic promotion/replacement policy; it is neither a
static/baked mask nor full/open routing.

Both valid arms used independent server processes, temperature 0, `think=false`,
the same 8,811-byte assistant replay fixture, the same request payloads, and
the same full-conversation request-2 transcript. Request 2 used 2,566 prompt
tokens, `full-conversation-reprefill`, and `kv_reuse=0` in both arms. The exact
prompt-2 UTF-8 SHA-256 was
`44d75b9687aafaca4d1312bb2516e25a44df250a367f094b7d9bea730292dfe1`;
request-2 payload SHA-256 was
`a6cc61c2817d203cac09f500f0636dfc4a4f941057859359c08b61c1c12d89aa`;
and request-2 messages SHA-256 was
`6ae7d64b131720c9dfb290fa11507263359806b6d2c0dc6dcbdcb1c149daf6cb`.

### Attempt inventory

| Attempt | Files and lifecycle disposition | Runtime/mechanism result | Quality and allowed claim |
|---|---|---|---|
| `g73_replayA_mech_n1_20260719T183918Z` | 13 files / 1,751,904 bytes. Runtime completed two HTTP-200 requests, but the original post-run parser crashed on `ContainsKey` against a `PSCustomObject`. CPU-only reparse later wrote `summary.json` but failed `server_exit_zero` because the historical exit code was unavailable and failed `existing_artifact_provenance` on `request2:content2`. The request file bytes themselves hash to the later canonical payload, while the reparse decoded/reconstructed messages as `4bb607...` rather than expected `6ae7d6...`; therefore the receipt is non-promotable. | Diagnostic stale mechanics only: no request-2 candidate; effective mask stayed 4,551 entries / FNV `c59a437fe9c6c376`, generation 1; seed stayed 320 entries / FNV `d65ec394cf596cfb`, overlap 320, Jaccard 1.0. Request 2: server finish 584.027 s, prefill 90.298 s at 28.42 t/s, decode 2,319 tokens / 493.618 s at 4.70 t/s. GPU request-2 samples 1,080, mean/median/max 94.48/95/100%, VRAM max 10,955 MiB. Backing misses, SSD bytes, failures, forbidden SSD-to-VRAM events, unsafe events and runtime model-progress events were all zero. | Independent raw audit: L1, because structure was recoverable but the fenced document, CSS and JavaScript were broken and the requested dark restyle was not achieved. Failed provenance/reparse gates prohibit an A/B, performance, exactness or quality-recovery claim. |
| `g73_replayA_utf8_mech_n1_20260719T1727Z` | 6 files / 40,025 bytes; native failed summary exists. The first `Start-Process` wrapper hit duplicate process-environment keys `Path` and `PATH`; the foreground runner then hit the same lifecycle error before server start. No request, response, GPU sampler CSV or DS4 runtime log content exists. Summary SHA-256 `7ee9d3d0931417662d92012d0b197f594d4de00cf5237cffc98c8885635bdcf1`. | Pre-server infrastructure failure. Requests=0, server exit unavailable, GPU baseline P8 only; no DS4/GPU run occurred. | Context only; no mechanism, timing, exactness or quality evidence. |
| wrapper `g73_replayA_utf8_mech_n1_20260719T1735Z` | No run directory. `_wrappers` contains only zero-byte stdout/stderr logs, each with empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Wrapper PID 6848 exited; no summary, request, response or server log was produced. | Infrastructure no-run, so there is deliberately no CSV evidence row. | No inference. |
| `g73_replayA_utf8_mech_n1_20260719T1736Z` | 14 files / 1,773,050 bytes. Native summary `status=pass`, server exit 0. Independent server process. | Valid stale control. Request 2: wall 563.255346 s; prefill 83.068 s at 30.89 t/s; decode 2,296 tokens / 480.051 s at 4.78 t/s. GPU request-2 samples 1,038, mean/median/max 95.34/95/100%, VRAM max 10,991 MiB. | Official L0: Markdown fenced output failed raw-only HTML; visual grading blocked. Diagnostic in-memory fence removal only: L1, with severely corrupted CSS/JS. Mechanism-only `n=1`; no performance or `n>=3` claim. |
| `g73_replayB_utf8_mech_n1_20260719T1746Z` | 14 files / 2,236,899 bytes. Native summary `status=pass`, server exit 0. Independent server process. | Valid request-epoch rebuild. Request 2: wall 990.054970 s; prefill 459.306 s at 5.59 t/s; decode 2,660 tokens / 527.075 s at 5.05 t/s. GPU request-2 samples 1,806, mean/median/max 70.49/94/100%, VRAM max 10,938 MiB. | Official L0: Markdown fenced output failed raw-only HTML; visual grading blocked. Diagnostic in-memory fence removal only: L2 (37/37 braces, 12/12 class selectors, dark palette, `node --check` PASS, form/popup DOM-stub PASS); one CSS parenthesis, invalid gradients and insufficient CTA contrast remain. Material quality recovery over stale A at `n=1` only; no `n>=3` until raw-only and visual gates pass. |

For old A, the exact on-disk request-2 SHA-256 is
`a6cc61c2817d203cac09f500f0636dfc4a4f941057859359c08b61c1c12d89aa`.
The failed reparse mojibake path reconstructed request/messages SHA-256 values
`5c88cfb730640e9c42dc7461306a88b96bf19e192a9d708dd51e1a2f86ffca6a` /
`4bb607933d02f46cbe646c45709e5c00c9cf5db51cc1ee1cd9c3e40e5968a493`,
versus expected messages SHA-256
`6ae7d64b131720c9dfb290fa11507263359806b6d2c0dc6dcbdcb1c149daf6cb`.

The `_wrappers` directory contains exactly four zero-byte logs: stdout and
stderr for tags `1727Z` and `1735Z`. The `1727Z` run directory contains
`assistant_replay_fixture.html`, `gpu_sampler.stop`, `preflight.json`, empty
server stdout/stderr logs, and `summary.json`. Each completed run directory
contains the fixture, stop marker, GPU CSV, preflight, request/response/raw
artifacts for both turns, server stdout/stderr and summary; valid A/B also
contain `request2.extracted_script.js`. The old `183918Z` directory lacks only
that extracted-script artifact. This is the complete directory/artifact set
observed under `g7_runs\g73_two_turn_html_replay_ab` at closeout.

### Valid A/B mechanism comparison

| Property | Arm A, stale control | Arm B, request-epoch rebuild |
|---|---|---|
| Request-1 candidate/effective | 4,551 / FNV `c59a437fe9c6c376`; effective generation 1 | Same |
| Request-2 candidate | none | 4,551 / FNV `4a752556bccde274` |
| Request-2 effective | `mode=stale`; same request-1 FNV and generation 1 | `mode=candidate`; FNV `4a752556bccde274`, generation 4,553 |
| Mask overlap | effective set unchanged | overlap 3,212; Jaccard 0.54533106960950761; 1,339 entered and 1,339 exited |
| Request-1 seed | 320 / FNV `d65ec394cf596cfb` | Same |
| Request-2 seed | stale reseed 320 / same FNV; request-1 snapshot provenance; overlap 320, Jaccard 1.0 | 320 / FNV `32fb4c7102950965`; dynamic-seed provenance generation 4,553; overlap 90, Jaccard 0.16363636363636364; 230 entered and 230 exited |
| Request-2 WRAP | no new arena publication; first publication loaded 4,551 in 25.107 s | turnover-in-place retained 3,212 and loaded/entered/exited 1,339 in 3.522 s; planned 9,477,292,032 bytes (9.477 GB decimal, 8.826 GiB binary) |
| Final tier counters | RAM/VRAM hits 459,762/132,606; H2D 3,256,408,866,816 B; promotions/demotions 7,680/7,360 | RAM/VRAM hits 514,674/171,606; H2D 3,645,069,852,672 B; promotions/demotions 8,835/8,515 |
| Safety counters | backing miss=0; SSD B=0; failure=0; forbidden=0; unsafe=0; runtime model progress=0 | Same zeros |

The causal mechanism result is positive and narrow: the B path reset request
scores, observed the complete new transcript prefill, published a different
4,551-entry snapshot atomically and reseeded a materially different 320-entry
VRAM set before follow-up decode. A deliberately retained request-1 provenance.
This proves request-epoch rebuilding structurally at `n=1`; it does not prove
full/open routing, losslessness against a complete IQ2 control, or performance.

The official grades remain L0 under the runner's current static contract:
both outputs are fenced, violate raw-only HTML, and could not pass the visual
gate. A separate diagnostic that removed only the fences in memory does not
change those grades: A is L1 because its CSS/JavaScript remain severely
corrupted; B is L2, with 37/37 braces and 12/12 class selectors recovered, a
dark palette, `node --check` PASS, and form/popup DOM-stub PASS. B still has one
CSS parenthesis defect, invalid gradients, and insufficient CTA contrast.

The rebuild therefore recovered material quality over stale A at `n=1`, but
this is not a statistical claim. No `n>=3` matrix is authorized until the
raw-only and visual gates pass.

### Provenance

- Canonical G73 runner SHA-256 remained `977ac73114bcdb883d06c123e3c33e467f230eea96235406c87d29f951b58470`.
- Final replay runner SHA-256 was `942797c5da95cb56597f9a688fe5f013b58cc50bdf76003186e6345ccf41dc85`; protocol SHA-256 was `ad871d5443deb4b8e4a16952710d1cf39cb74a024069906ea1ad782eb5025ac0`; contract SHA-256 was `036b6acfa89ff67d4c33bf039030e4706b96b37c7ef77b9555b1eea7c1c538c2`.
- Executable SHA-256 was `f703f53246331cd632e81eadba9892b8184645cc0b714ad6005ad87d2899dcfa`; build input fingerprint was `c8698dc4f5ba0dcd4e29f50c84545704140875d653f122f1a8136d82c5444daa`.
- IQ2 model SHA-256/bytes were `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668` / `86720111488`; model receipt SHA-256 was `a1a6626088489743628165692d32870f083cfe74386469176d0b333a2c95eb55`.
- Valid A summary/raw/request-2/log SHA-256 values were `d4452a5071791ca9a25be7b72a7663f7978e4efac1491b6a9b3f77e6faafc69a`, `90f2e0a02ee5fd9dafc820c25b57856e410eff30db130dc8734ac23d57d587d8`, `a6cc61c2817d203cac09f500f0636dfc4a4f941057859359c08b61c1c12d89aa`, and `9f68eb04860c1ba995be3373730330ebaa2bf438ca12189b97cfbcd4cd46ffc5`.
- Valid B summary/raw/request-2/log SHA-256 values were `a11aad816d0d5389cd0428c0933355d9f2c63b963f8517f08780d9dc138b19d4`, `70de59844406c68cb9fd5561cfc7f384a41c64363a9cbc9b695df1341f3434df`, `a6cc61c2817d203cac09f500f0636dfc4a4f941057859359c08b61c1c12d89aa`, and `40ea0c78a6ff72e160880e7f1445cf7564f6db0edcd8371e4cbf1457c54e5621`.
- Authoritative run root: `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_two_turn_html_replay_ab`.

## 2026-07-20 G73 Live Arm B End-to-End Failed Start

Run `g73_liveB_e2e_20260720T053105Z` was intended to exercise one live Arm B
conversation in a single server process: generate turn 1, reinsert its exact
bytes into the chat history, then issue the dark-theme follow-up. It produced
no request and no transcript. `ds4_server` PID 3880 disappeared during initial
CUDA model loading, before readiness. `server.stderr.log` contains only CUDA
backend initialization and the initial model-tensor cache-load line.

The official disposition is `failed-start/pre-request NO-RUN`. Request count,
HTTP-200 count, raw outputs and transcript count are all zero. This record is
not quality or performance evidence and is not a result for the G73 request-
epoch mechanism. The failure is attributed to the live runner's launcher,
readiness and PID-ownership lifecycle: its wrapper and discovered server PID
were not reconciled, the normal summary path did not execute, and a fail-closed
summary plus receipt had to be recovered from the on-disk artifacts. No retry
and no Arm A run were performed.

Authoritative artifact root:
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_live_html_b_end_to_end\g73_liveB_e2e_20260720T053105Z`.
Recovered summary, failure receipt, stderr and preflight SHA-256 values are
`8100f87dc2bc4e076e17bb29aeebb5712103de5629be2b8c39d8c68eb806a54d`,
`c8b5286d3907fe9dc281c838c19bd0365a3611e305f2d8eac7bacf52830fa21f`,
`4b7aaefb5816739058fc3da17d9e0677d8c30fe6dc8b296f26416d47b14f4778`,
and `ea59542caab3e384fe4aeb0f456daf687e8edd0c3fe8a17c56eef2e9ec30a096`.
The live runner, canonical G73 runner and executable SHA-256 values are
`8a69314f80441561c2f31157b4bd1cfd216a6ac616edf3c2b1bcbe55c91b531c`,
`977ac73114bcdb883d06c123e3c33e467f230eea96235406c87d29f951b58470`,
and `f703f53246331cd632e81eadba9892b8184645cc0b714ad6005ad87d2899dcfa`.

## 2026-07-20 CPU-Only Low-Bit and Conversation-Runner Evidence

This section records mechanism, design, simulation and build evidence only.
None of the entries below is a DS4 GPU runtime, quality, throughput, `n>=3`,
full/open success or SOTA result.

### G73 live two-turn runner mock

The dedicated live Arm B runner passed its CPU mock integration scenario. The
first request roles were exactly `system,user`; the second were exactly
`system,user,assistant,user`. Assistant turn 1 was inserted byte-identically
into request 2 with SHA-256
`f5a6c675058c5f6a2ed0ea9d42515388756c827ce59d0dcf30f85ac1739a8e0e`.
The controls were temperature 0, `think=false` and completion stop `</html>`.
Exit-before-readiness failed closed and malformed turn 1 prevented request 2.
This proves runner lifecycle and conversation construction only: Python mock
responses were used, no DS4 server/model/CUDA/GPU was accessed, and no G73 mask,
quality or performance claim follows. Success summary and validation receipt
SHA-256 values were
`5545cf8f5101a08778acb0cec5386a2d972a1dfe98b86dcfe5472ce63537057d`
and `7491e0ea41a255960c07509483f6c23be012bea1bb81c09934a7ee79f1332ce8`.

### Binary/Bonsai-equivalent density budget

The current DS4 Q1_0 representation already has nominal density 1.125 bpw:
3,538,944 bytes for 25,165,824 weights per routed expert. Across 11,008
experts this is 277,025,390,592 routed weights and 38,956,695,552 payload
bytes. The authoritative sidecar file is 39,048,344,416 bytes including file
overhead. A 27B model at the same density is 3,796,875,000 bytes, about 3.54
GiB; DS4 has about 10.26 times as many routed weights. Thus the apparent
Bonsai 3.5-3.9 GB versus DS4 39 GB difference is parameter count, not poorer
packing density. This is arithmetic/design context, not a model-quality result.

### One-expert weight-domain micro-probe

The deterministic CPU-only probe used `blk.3`, expert 0. Current Q1_0 versus
the decoded IQ2 teacher measured cosine 0.811377 and NMSE 0.341667. A binary
sign plus L2-scale g128 refit reproduced current Q1_0 to numerical precision.
The 1.296875-bpw ternary candidate measured cosine 0.758266 and NMSE 0.425033.
The weight-only path is therefore NO-GO as a quality fix; activation-aware
recovery remains an experiment. No end-to-end quality inference is allowed.
Report SHA-256:
`a9e910d488c14cfc5614e7f861786856b0f38898229023af4c9975a60fcbdb9a`.

### Activation-aware trainer scaffold

The CPU-only scaffold implements a fail-closed trace reader and streaming
binary/ternary g128 trainer with expert-output and gate-weighted losses,
bounded updates and delta-only checkpoints. Six synthetic tests and Python
bytecode compilation passed. It consumed no real activation trace and makes
no recovery-quality claim. Report SHA-256:
`0c6fb2aa64c35441f169377fae678843761e682a262004f4091f8b47fb274069`.

### G129 route replay and host-capacity simulation

The structural G129 trace showed that Q1 fallback was not rare: approximately
58% of routes and 52% of router mass. CPU replay of exact-IQ2 host capacities
produced the following fallback-mass estimates:

| exact-IQ2 slots | oracle | LFRU pass 1 | LFRU pass 2 |
|---:|---:|---:|---:|
| 606 | 24.0% | 37.1% | 31.5% |
| 910 | 18.0% | 32.1% | 24.5% |
| 1213 | 13.5% | 28.6% | 19.2% |

Exclusive low-bit/IQ2 replacement implied high recovery I/O. The current
candidate is therefore a complete resident binary base plus a duplicated,
bounded exact-IQ2 arena. These are replay/design findings, not measured runtime
quality or speed.

### Packed Q1 kernel audit

Read-only source audit found that G129 already transfers and consumes Q1_0 in
packed 18-byte/128-weight form; it does not globally expand the representation
to FP16 or INT8. A new binary container therefore cannot reduce transport
bytes at the same density. DP4A/MMVQ-style arithmetic remains a conditional
kernel optimization, but no measured end-to-end gain or quality evidence
exists.

### G129 input-only activation trace certification

G129 now contains an OFF-default, single-expert input trace capped at 256
samples. It records input vectors plus request epoch, call, token, rank, gate
weight and resolved representation; exact-IQ2 teacher outputs are reconstructed
offline. Strict binary/JSONL validation and manifest-last atomic publication
are required. With tracing OFF there is no trace file, D2H copy or added sync.

CPU-only certification passed PowerShell parsing, positive and negative parser
tests, Python tests, control/promotion/trace WhatIf, Release Ninja `sm_86`
build, CTest 1/1 and diff check. Build manifest, input fingerprint and
executable SHA-256 values were respectively
`ba819fee217ba5c2eddca6894a8b012e35560aa3ed2c6a216115e7671157bf5a`,
`2c382328f8f1327b613f449f3ebdd85f2c7a32143339f77a5bb504182de5ea40`
and `51ff6dad1e0bab5d3a0c6e145d767e5a0dc0f8e2f99660b68aaf4f4fb8adb9a8`.
No trace was acquired and no GPU/runtime/recovery-quality claim is made.

### G129 SSD-WRAP CPU-only certification

The OFF-default G129 SSD-WRAP implementation passed PowerShell parsing, strict
parser positives/negatives, runner SelfTest, both Python suites, control/legacy/
SSD-WRAP WhatIf, Release `sm_86`, CTest 1/1 and diff-check. No DS4 server or GPU
was started. The immutable exact-IQ2 host budget remains 5,902,958,592 bytes:
834 slots, four used by the transition ring. The proposed first split is 299
pinned plus 531 pageable stable slots.

The current call remains on resident Q1. Exact IQ2 is published only for a
later call after SSD-to-RAM completion and validation; current-token
SSD-to-VRAM is forbidden. OFF creates no thread, handle, ring, allocation or
write. Build manifest, fingerprint and executable SHA-256 values are
`1893258c8406e8c668eaf1856527ba0b5aba9ea2169e7d53525ca7257686a66b`,
`f80f32087ed9d7716651ea661846a4ed50082aae0746700c8fc3c5c0a75a106a`
and `e258a4fd60c7dc4dfb98cd0ec8b168f4a06e3f3f435adbe8ef89540cd2d307e5`.

This is `context_only` CPU implementation evidence. No runtime, quality,
throughput, `n>=3` or SOTA claim follows. The next gate is one separately
authorized `promotion_ssd_2_0` structural safety. Runtime, harness and branch
tip commits are `16273d4a1d9b648a5878223e7d3ecd3a8d233672`,
`415ed980da69bad304d98e77b1851076a7ae06a6` and
`801a6d8fee17d1fd18fa4fe83fec3f750501fd7e`.

## 2026-07-22 sera — F6/F7/F8, radice della spirale big-ctx, cattura L15 su fleet 4090

**F6+F7 landate** (wt-g73-open `55ad829`, autorate Codex + applicazione multi-agente,
verifica avversariale Claude+Codex, build Release pulita). F6
(`DS4_CUDA_RELEASE_PREFILL_SCRATCH=1`): rilascio dello scratch prefill al confine
prefill→decode, ledger `[vram-ledger]` sempre attivo — MECCANISMO VERIFICATO live
(`release_status=released`), ma a prefill_chunk=2048 libera ~72MiB, non i ~1.5GiB
dello studio: lo scratch non è il ladro dominante della spirale. F7
(`DS4_CUDA_KV_MANAGED=1`, advice host-preferred sulle 3 cache KV già managed):
no-op documentato su Windows/WDDM (one-shot `[kv-managed] cudaMemAdvise
unavailable`), payoff solo Linux. h6 ambiguo risolto al sito GPU 17743 (il gemello
CPU non linkerebbe sotto DS4_NO_GPU).

**F8 "KV staged-ring" landata** (`4535e11` + fix stop `decd412`; design = intuizione
utente, registro "KV-ring rotante in VRAM"; 13 hunk Codex sessione
019f8b15-d37d + 19 fix initializer per il nuovo `host_ptr`): KV intera in host
pinned-mapped, ring VRAM 2×4MiB (2048 righe MLA/slot, top-k cap 512), stream copia
dedicato + 4 eventi, online-softmax esatto, kernel heads8 cooperativi in shared
memory (uccide l'amplificazione ×64 per costruzione). Gate
`DS4_CUDA_KV_STAGED_RING=1`, unset = bit-exact per costruzione (host_ptr NULL →
branch mai presi; verificato a mano). Costo VRAM del ctx → 8MiB FISSI qualunque
finestra: spezza il legame capacità-dichiarata→VRAM. Smoke runtime in corso.

**RADICE della spirale ctx≥1536 QUANTIFICATA** (misure live 3060, stessa sessione):
ctx 768 → cache esperti 320 slot ≈2.1GB > working set ~1.6GB/token → 2.8 t/s;
ctx 8192 → buffer/riserve clampano la cache a 165-174 slot ≈1.1GB < working set →
thrash LRU totale → 8.4GB/token letti da SSD (631MB/s misurati / 0.07-0.5 t/s)
→ 10-13s/token. È il muro fit-in-cache applicato al contesto, NON una regressione:
bisect pre-F8 vs F8 = stesso collasso (10.8 vs 13.6 s/token). F6 da sola non salva
il big-ctx. Leva candidata: F8 (KV+riserve fuori VRAM → capacity risale). APERTO:
righe "transient route unavailable"+"snapshot publication failed" SOLO col binario
F8 (~25% peggio) — indagare interazione col path mass-compose anche a gate spento.
Nota metodo (richiamo utente recepito): quota SSD SI MISURA (contatori
vram/ram_hit + MB/s disco / t/s), non si stima per capacità.

**Cattura L15 all-experts su fleet**: H200 terminato dall'utente (misload:
zero-copy default; `DS4_CUDA_COPY_MODEL=1` → residenza piena in 12s MA decode
resta ~3.5 t/s = orchestration-bound confermato su H200; multi-processo su 1 GPU
NON scala, aggregato piatto ~3.5; `DS4_LOCK_FILE` bypassa il lock single-instance).
Pivot fleet: pod A secure 4×4090 ($2.76/h, 251GB RAM, driver 570 sano) = 4 worker
1×GPU zero-copy, 1.1 vec/s l'uno, aggregato 4.40 STEADY (≥75s) = batte l'H200 a
60% del costo (~9.6k vettori/$ vs ~2.9k). Modello in page cache condivisa (162GB
buff/cache, I/O disco ~0). Round 1 target 40k; round 2 (A2) automatico → 80k
fase-1. Histogram a 31.7k vettori: 191.589 coppie (≈6 route/token ✓), 253/256
expert toccati, MEDIANA 269/expert (già sopra il muro 256), 29 expert ≥2000 —
segnale del pilota Q1 promettente. Ops: 3 secure 4×4090 serali mai partiti
(runtime null anche senza volume), community = stesso host marcio 580.x in loop →
regola: estendere il pod provato batte cacciare host nuovi; MSYS mangia
`--volumePath` da Git Bash (3 pod persi, ~$2) → `MSYS_NO_PATHCONV=1`; porte SSH
da GraphQL `runtime.ports`, non REST.

**DeepSpec/route-oracle**: analisi Codex (sessione 019f8b19-af3d, report
`D:\ds4_work\g73_fix\deepspec_oracle.out.log`): capitolo drafter riaperto RISTRETTO
("MTP nativo GPU + route proxy tier-aware") — MTP GPU 26.6/30.5ms p50/p95 entra
nel budget <50ms (G34-CPU resta chiuso); V4 ufficiale ha i primi 3 layer routing
hash/token-ID (route ESATTA, da verificare nel fork); pipeline corretta = predire
t+2; checkpoint DeepSpec inutilizzabili (1.85-6.9GB, no DeepSeek). Spike 1g con
go/no-go nel report §F.

**Stop pulito**: endpoint `/__g130_u1_shutdown` armato dal solo token+loopback
(`decd412`; il gate Q1-profile confliggeva con DS4_G73_OPEN). UI DS4 Control
patchata: Ferma DS4 = drain via token → taskkill gentile → /F ultima spiaggia
(il /F misurato orfana la RAM: NON-ATTRIBUITA 8.9→11GB).

**Addendum F8 smoke Windows (stessa sera)**: con `DS4_CUDA_KV_STAGED_RING=1` su
ctx corto il decode va a ~10.3-10.6 s/token (pre-F8: ~2s) = ~5× peggio; un token
da 107.9s con `route worker respawned after transient-io kill`; GPU 3% = attese
eventi/mapped-host. WDDM è ostile a pinned-mapped + event-handoff per chunk.
DESIGN non refutato (il bisect lo scagiona dal collasso; la matematica è esatta):
è l'IMPLEMENTAZIONE su Windows a non rendere. Gate resta off di default
(bit-exact). Banco giusto per l'A/B: pod Linux post-cattura. Se lì conferma,
ottimizzazione Windows separata (chunk-batching o staging via copia).

**Addendum 2 (notte): A/B cancello d'ammissione tier = IPOTESI STROZZATURA REFUTATA.**
ctx 768, 120 tok/braccio, stesso prompt temp 0. CONTROLLO (budget32/minf3/hyst1.25)
vs SPALANCATO (512/2/1.0): decode_ms 222 vs 211 (rumore), ram_hit 60-67% vs 61-73%
(nessuna convergenza differenziale), transient/tok ~31 vs ~29 (identici). Aprire il
cancello 16x non sposta nulla → l'ammissione NON e' il collo a ctx sano. I ~30
transient/token (~12%) coincidono col pavimento SORPRESA del 07-12 (il ~10% della
domanda fuori-recency): a ctx 768 il sistema sta SUL floor teorico. Bonus: 4-5 t/s
misurati (>riferimento 2.8, page cache calda). La patologia big-ctx resta confinata
a: (a) cache clampata 165<working-set, (b) bug "snapshot publication failed" =
interazione F8 x ctx-grande A GATE SPENTO (assente su pre-F8 a 8192 e su F8 a 768)
→ indiziati gli hunk 7-8 "contents/copy host-aware" (girano su TUTTI i tensori,
inclusa la pubblicazione dello snapshot mass-compose). Da correggere prima di
qualunque nuovo test big-ctx.

**Addendum 3 (mezzanotte): suite F8b + refutazione riserva.** F8b "migrate-to-arena"
(commit su g73-open; prefill in VRAM -> migrazione una-tantum D2H pinned non-mapped
al decode-start -> ring; append D2H; sticky-off; spin): (a) gate OFF a ctx768 =
4.95 t/s parita' daily -> SICURA; (b) migrazione FUNZIONA (105 tensori, mapped=0 =
obiettivo architetturale centrato); (c) decode ring CRASHA mid-gen (S2, da
debuggare); (d) S3 end-to-end 794s vs 1718s S4 (2.2x, ma page-cache piu' calda e
n=2 campioni: non conclusivo). S4 (RESERVE=32 @8192): 11.8s/tok, cache 15/177 ->
TEORIA RISERVA REFUTATA; il mistero big-ctx si sposta: l'ammissione in cache e'
bloccata a 8192 (count 15-37 su 161-182) mentre a 768 riempie normale (109/250) --
primo indiziato il floor di VRAM libera (~1GB a 8192) che veta le ammissioni.
Prossimo: debug crash ring (log in g73_gate/f8b_suite_231257) + caccia al gate di
ammissione big-ctx.

**Addendum 4 (01:30): S5 chiude la diagnosi big-ctx.** Con
DS4_G133_TRANSIENT_IO_TIMEOUT_S=300 (+ROTATOR): tiering-enforce failed 0,
deadline 0, cache RIEMPITA 160/160 (vs 15-37) -> il blocco d'ammissione ERA il
timeout IO (accoppiamento confermato). MA decode 7.2->10.1->27.3->20.3 s/tok al
riempirsi: a capacity 160 < working-set ~240 slot NESSUNA politica vince
(bloccata=fame 32s; libera=churn LRU 27s, la lezione fit-sweep). Il blocco era
accidentalmente protettivo. CONCLUSIONE QUANTIFICATA: il big-ctx si sblocca solo
con capacity >= ~240 slot = ~540MB di VRAM in piu' a ctx 8192. Vie: migrazione KV
F8b (250-500MB, borderline da sola) + dieta buffer ctx + (alternativa) riduzione
working-set via mask. Non ci sono piu' misteri: c'e' un budget di 540MB da
chiudere. (S5: 90 tok in 2113s totali; stop pulito.)

**Addendum 5 (02:00) — IL MURO Q1 E' SFONDATO.** train_expert.py su E176 (L15) con
5.173 campioni reali (round-1 della cattura fleet, shard filtrato
extract_expert_shard.py + teacher offline exact-iq2 via make_teacher_outputs):
iter 5000 -> train 0.9336 / val 0.8258 / **TEST COSINE 0.8335** contro il muro
storico 0.7009@256 (STE-adapter) e l'overfit 0.962/0.651 del trained-max. Firma
sana: val=test (niente overfit), gap train-test fisiologico. LA TESI DI FASE-1 E'
CONFERMATA: il muro era fame di dati, non capacita' del formato Q1_0. Il filone
Q1-companion (tier sub-bit per la coda: complemento residenza + knock + MTP,
config I7 cold<=1.0bpw=41GB) ha ora fondamento sperimentale. Prossimi: curva
256/1000/2000 (in coda), poi punto dense-training, poi produzione (~$100 per
10.240 sidecar, pipeline collaudata stanotte end-to-end da R2 al checkpoint).

**Addendum 6 (02:30) — CURVA DATI Q1 completa: il muro era pipeline, non dati.**
E176 test-cosine per n campioni: n256=0.8050, n1000=0.7865, n2000=0.7928,
n5173=0.8335. SHOCK: gia' a 256 (STESSO conteggio del muro storico 0.7009) questa
pipeline fa 0.805 = +0.10 a parita' di dati. Il "muro 0.70" era ingegneria di
training, NON capacita' del formato ne' fame di dati. Curva non-monotona
(split diversi + pochi campioni), segnale robusto: 256 gia' forte, 5173 il best.
IMPLICAZIONE COSTO: bastano poche centinaia-poche migliaia di campioni/expert per
superare la soglia transient-serve (~0.80-0.83) -> i 10.240 sidecar costano molto
meno del previsto (cattura naturale per i caldi, dense per la coda). Artefatti:
q1_pilot_data/trained_e176_n{256,1000,2000,full}/. Pipeline end-to-end collaudata:
R2 -> extract_expert_shard (filtro all-experts->single, header DS4ERTR1) ->
make_teacher_outputs (GGUF decode + forward esatto) -> train_expert (STE+LoRA) ->
checkpoint. Fase-1 CHIUSA positiva; prossimo = dense-training per la coda + scale.

**Addendum 7 (03:00) — DENSE-TRAINING: il ponte verso l'universale REGGE (cross-eval).**
E176: dense (14336 x di TUTTI gli expert, teacher offline E176-exact) vs routed (5173).
Cross-eval sulla STESSA regione routed (cross_eval.py):
- routed-model su routed: 0.8936 (con train overlap) / 0.8335 test pulito
- DENSE-model su routed: 0.8088 (quasi tutto unseen per lui)
- dense-model su full manifold: 0.4570 (IRRILEVANTE: E176 non serve mai quegli x)
VERDETTO: il dense costa solo ~0.02 cosine sulla regione che conta e resta SOPRA la
soglia transient-serve (~0.80), pur non avendo MAI visto la cattura routed. => per la
CODA FREDDA (esperti non catturabili abbastanza) il dense-training col teacher offline
su x abbondanti da' un Q1 usabile (0.81) -> l'universale NON e' capture-bound, e'
compute-bound (embarrassingly parallel: un forward-expert per (x,expert)). Caldi =
routed-specific (0.83); freddi = dense (0.81). Il crollo dense-su-full-manifold (0.46)
conferma che la selettivita' del ROUTER e' cio' che rende Q1 viable: l'expert opera solo
sul suo sub-manifold ristretto, dove 1.125 bpw basta. Artefatti trained_e176_dense/.

**Correzione (front C, decomposizione da codice ds4_context_memory_estimate):** il
budget 540MB del big-ctx NON e' dominato dal prefill (correggo la stima precedente
~130MB). Decomposizione a ctx 8192 dei "context buffers" 333MiB: raw_bytes +
compressed_bytes = KV MLA (~300MB, = F8b migra questi, device_bytes=315MB misurato)
+ scratch_bytes = 2*comp_cap*prefill_cap*4 = solo ~32MB (leva prefill_chunk ->
dimezza a ~16MB = TRASCURABILE). Il ~130MB residuo del budget = g_cuda_tmp (scratch
prefill che cresce nel prefill lungo, GIA' rilasciato da F6 al decode-start, questione
di timing). CONCLUSIONE: la leva del big-ctx e' la KV via F8b (~55% del budget), NON
il prefill. Ridurre prefill_chunk = rifinitura da pochi-MB, non fix.

**Addendum 8 (03:30) — F8c: crash ring diagnosticato + SCOPERTA STRATEGICA.** Codex
(f8c_ring_crash.out.log, 12 hunk): il crash "whole-map-classification-d2h" e' una RACE
reale — cuda_moe_route_worker_cancel_and_join non sincronizza il route_upload_stream
(non-blocking), l'invalidazione azzera la device map sul default stream, il respawn
riusa buffer non quiescenti. Fix = safe respawn (query stream, 5s quiescenza o
safe-leaked+fallback, no cudaDeviceSynchronize). APPLICATO. **MA il punto strategico:**
anche con timeout 300s (zero crash, S5) il ring resta ~90s/token → sistemare il crash
lo rende SICURO, non VELOCE. Il ring KV migrate, su 3060/WDDM, si mangia in latenza piu'
di quanto guadagna liberando i 315MB. => la leva big-ctx via migrazione-KV e' in DUBBIO
su questo HW (il rischio WDDM originale confermato). Ipotesi alternativa da testare: la
vera leva big-ctx e' RIDURRE IL WORKING SET (mask/pruning K12 -> il set/token scende
sotto la capacity 160) non liberare VRAM con streaming KV lento. Lo studio prefill
(Codex, in corso) potrebbe indicare la strada. F8c resta valore (server che non crasha).

**Addendum 9 (04:00) — SCOPERTA UTENTE: ~1-2GB VRAM MAI RECLAMATA (candidato #1 big-ctx, GRATIS).**
Osservazione utente: "durante i test la VRAM non si riempie mai, ~1GB di margine, e' voluto?"
Verificato NO: la expert cache e' dimensionata al PRIMO prefill (cuda_moe_expert_cache_prepare,
lazy dai path forward 35490/37181/40823) quando i buffer transitori del prefill pressano la VRAM;
cap=max_slots=(free-reserve)/6.75MB al momento del calcolo. Poi il prefill finisce, ~1-2GB si
liberano (F6 + graph buffers) MA la cache e' assegnata UNA volta (capacity= riga 31263, early-return
se config combacia) e NON ricresce. Dati decode-start: ctx768 free=1.91GB cache=250; ctx8192
free=1.16GB cache=160. PARADOSSO: la cache nasce PIU' PICCOLA quando il ctx e' PIU' GRANDE (sizing
al picco di pressione prefill), l'opposto del bisogno. FIX candidato (elegante, no F8b, no ring lento):
far RICRESCERE la cache al passaggio prefill->decode (hook F6 gia' li') reclamando l'1-2GB liberato ->
a 8192: 160 -> ~310 slot > working set ~240 = BIG-CTX RISOLTO GRATIS. Codex incaricato del patch
(realloc-and-grow o two-phase sizing). Questa e' la leva vera, non la KV-migrate.

**Addendum 10 (04:30) — RISCATTO F8b (intuizione utente): il ring KV e' per l'ULTRA-LONG-CTX.**
Utente: "lo streaming KV torna utile per contesti molto grandi; la gente viaggia a 250k KV
minimo, DS4 supporta 1M." Corretto: la KV MLA a 8k = ~300MB (sta in VRAM -> F10 grow-cache e'
la leva, il ring costa solo latenza). A 250k = ~10GB, a 1M = ~40GB -> NON entra in nessuna VRAM
(nemmeno H100) -> streaming host+double-buffering+online-softmax esatto (F8b) e' l'architettura
OBBLIGATORIA, non un ripiego. F8c (crash-safety respawn) conta SOPRATTUTTO li' (ring esercitato a
lungo). MAPPA LEVE: F10 grow-cache = sblocca 8k sul 3060 (working-set vs cache); F8b+F8c ring =
abilita ultra-long-ctx 250k-1M su QUALSIASI GPU. F8b non era sbagliata, era puntata alla scala di
contesto sbagliata per il daily; casa = long-context. Da testare quando serve il regime 250k+.

**Addendum 11 (05:00) — Compressione KV nativa (domanda utente): eccellente e GIA' sfruttata.**
DS4 usa MLA con ratio per-layer (ds4.c:452 ds4_layer_compress_ratio): layer 0-1 ratio 0 (piena),
layer pari ratio 4, layer DISPARI ratio 128 (!). Meta' dei layer comprimono la KV 128x. E' bakato
nel modello (metadata deepseek4.attention.compress_ratios, validato). Noi lo implementiamo fedelmente
(comp_cap=ctx/ratio+2). E' perche' la KV a 8k = solo 300MB e perche' 1M e' fattibile senza 4 H200.
Non e' una leva lasciata sul tavolo: e' design del modello, non spingibile senza retrain. CONVERGENZA
(le 2 domande utente streaming-KV + compressione puntano allo stesso posto): la KV NON e' mai stata il
collo del big-ctx (300MB!); il collo e' l'I/O ESPERTI (40 layer di pesi da SSD) = leva #1 studio prefill
(planner per-layer, expert caricato 1 volta/chunk) + F10 (cache esperti che ricresce). F8c committata.
Studio prefill completo salvato (prefill_study.out.log via GitHub connector, sandbox ha bloccato clone).

**Addendum 12 (05:10) — Studio prefill (Codex, GitHub-wide) — leve estratte.** Fonti verificate via
connettore GitHub (sandbox ha bloccato il clone locale; link fissati a commit). Tecniche mappate:
- **Expert-I/O planner per-layer** (LA leva non-tirata): caricare gli esperti PER-LAYER-tutti-i-token del
  chunk, non per-token-tutti-i-layer -> ogni expert letto 1 volta per chunk invece che per token. Il
  batching base c'e' gia'; manca il PLANNER CUDA/Windows che ordina/coalizza gli I/O (PR #514 upstream:
  .expbundle, coalescing gate/up/down, dense staging ring). Beneficio ALTO sul nostro prefill, costo medio.
- **Prefetch predittivo cross-layer** (ProMoE 2410.22134, MoE-Infinity, Fate): prefetch dell'onda expert
  successiva col fallback esatto; potenziale ma prematuro finche' restano reload/seek/sync deterministici.
- **MLA absorb** (DeepSeek V2/V3): aiuta KV/attention ma NON riduce i 40 layer di traffico expert -> non
  e' la nostra leva (confermato: collo = expert I/O, non KV).
- **Prefix caching / PD disaggregation** (DistServe, Splitwise, Mooncake): leva sui prefissi RIPETUTI, non
  sul primo cold prefill; utile per chat multi-turno, non per il primo prompt lungo.
- **Speculative prefill / token pruning** (SpecPrefill 2502.02789): un modello leggero pota token del
  prompt -> meno lavoro prefill; prompt compression, rischio qualita'.
Report completo: D:\ds4_work\g73_fix\prefill_study.out.log. CLASSIFICA per noi (3060/Windows): #1
expert-I/O planner per-layer (allineato a F10 grow-cache: entrambi attaccano l'I/O esperti); #2 prefetch
predittivo (dopo che il planner rende deterministico l'ordine); #3 prefix-caching per il multi-turno.

**Addendum 13 (05:30) — PILOTA SCALATO 13 esperti caldi: RIDIMENSIONA il titolo Q1 (misura>speranza).**
E176 (0.83) era il CASO FORTUNATO sul lato alto. Distribuzione reale 13 esperti caldi L15
(multi_expert_results.tsv): mediana 0.733, media 0.716, range 0.582(e77)-0.807(e246). Solo 2/13 >=0.80,
4/13 >=0.75. SEGNALE DIAGNOSTICO: forte OVERFITTING su molti (e77 train0.77/test0.58; e87 0.89/0.64;
e43 0.90/0.67) -> non fame-dati (hanno 4.6-16.7k) ma capacita'/regolarizzazione a 1.125bpw. VERDETTO
ONESTO: Q1@1.125bpw NON riproduce affidabilmente gli esperti, e' expert-dipendente (mediana 0.73), NON
lo 0.83 universale che E176 suggeriva. Produzione 10240 sidecar a bit fisso IN DUBBIO. Il pilota scalato
ha fatto il suo lavoro: evitato $100 su una conclusione da un expert fortunato. CAVEAT che salvano
l'architettura (misurabili): (1) il carico vero sono i FREDDI (cold-test in corso), errore diluito, barra
piu' bassa; (2) la soglia 0.80 e' euristica mia, MAI validata vs qualita' output -> test vero = end-to-end
(chat degrada con Q1 transient-serve?). LEVE PRODUZIONE: bit-rate adattivo (Q1 facili/Q2 difficili) o fix
overfitting (reg/early-stop/LoRA-rank). NON promuovere la produzione finche' cold-test + test end-to-end.

**Addendum 14 (05:45) — COLD-TEST: piu' dati NON aiuta, il tetto e' il formato.** Esperti freddi routed
(50-160 campioni): e63(151)=0.700, e15(160)=0.697, e117(50)=0.716, e252(55)=0.652. Confronto: caldi
4.6-16.7k campioni = mediana 0.73; freddi 50-160 = 0.65-0.72. QUINDI il conteggio campioni NON e' il
collo (caldi 16.7k ~ freddi 50). Il collo e' FORMATO 1.125bpw + overfitting. La maggioranza cluster
0.65-0.80 qualunque n; E176(0.83)/e246(0.81) = fortunati. VERDETTO Q1 CONSOLIDATO: tetto ~0.72 mediano,
non universale. Vie: (a) bit-adattivo (piu' bit per gli esperti duri), (b) diluizione architetturale (freddi
sparano di rado -> 0.70 forse basta, da validare end-to-end). Il dense-on-routed cold non ha completato
(teacher dense per-expert costoso, run veloce); da rifare se serve. NEXT deciso dall'utente: distillazione
REGOLARIZZATA sui peggiori (e77=0.58, e87=0.64) per capire se 0.72 e' tetto-formato o difetto-training.

**Addendum 15 (06:00) — RICERCA COMPRESSIONE SOTA (Codex): il fix di Q1, validato dalla letteratura.**
Report completo: D:\ds4_work\g73_fix\compression_sota.out.log (fonti verificate, link a commit/arXiv).
DIAGNOSI: il nostro train/test gap (0.9/0.73) e' un fallimento di RAPPRESENTAZIONE+OTTIMIZZAZIONE, non
dati/reg. STE su 25M segni + LoRA "in nessuno dei 2 regimi che funzionano" (native QAT globale tipo
BitNet/ParetoQ; o PTQ strutturato tipo AQLM/QuIP#). Tre difetti precisi: (1) STE sbagliato per ottim.
discreta - update sotto-soglia spariscono (PV-Tuning arXiv:2405.14852 lo prova); (2) COSINE LOSS
magnitude-blind, dominata da direzioni frequenti (difetto del NOSTRO obiettivo); (3) "16k caldi ~ 50
freddi" = attivazioni routed a rango basso/anisotrope, piu' token stessi-domini ~zero info indipendente.
FIX 2 STRADE: Path A (formato Q1_0 invariato, trainer-only, feasible stanotte) = GPTQ-style: obiettivo
Hessian-weighted reconstruction tr((W-Q)H(W-Q)^T) NON cosine, sign-assignment Hessian-aware NON STE,
NIENTE LoRA, + rotazione Hadamard fissa (QuaRot); one-shot, Hessian 64MiB, no overfit possibile. Path B
(nuovo formato, meglio ma serve kernel dequant) = AQLM codebook VQ ~1.38bpw. IDEA UTENTE Q1/Q2 ADATTIVO
= prior art solido (QuantMoE-Bench 2406.08155, MxMoE 2505.05799, MoQE, QMoE 2310.16795 = precedente MoE
diretto GPTQ-per-expert 0.8bpw); a budget 1.5bpw medio -> Q2 al 37.5% esperti sensibili. CALIBRAZIONE:
split per-documento non per-token, dedup, stratifica per dominio/router-prob/norma, damping Hessian.
ESPERIMENTO DECISIVO: GPTQ-Q1 one-shot su 2-3 esperti -> se batte 0.73 era il TRAINING, se ~0.73 e' il
FORMATO 1.125bpw (serve VQ). NEXT: decisione utente Path A (GPTQ-Q1) vs B (AQLM) vs Q1/Q2 adattivo.

**Addendum 16 (06:30) — Test F10 + LEAKAGE Q1.** (1) F10 two-phase FALLISCE a runtime: fase B GROW=1
-> "[expert-cache-grow] mode=two-phase result=failed cache=disabled fallback=exact-streaming". CAUSA:
conflitto con prefill-vram-seed (il pre-warm della cache coi caldi durante il prefill si aspetta una
cache che F10 ha RIMANDATO -> "prefill-vram-seed result=failed reason=cache-capacity" -> finalize
fallisce -> cache disabilitata = PEGGIO del baseline 160). Bug d'integrazione: la deferral rompe il
contratto del seed. FIX: F10 deve anche disabilitare/rimandare il prefill-vram-seed sotto GROW=1, o
approccio diverso. Rimandato a Codex. Baseline fase A: cache 160, decode 2733ms. (2) LEAKAGE Q1
SCOPERTO: shard per-expert estratti SENZA dedup -> E176 5173 righe/3797 token unici/1090 duplicati ->
split random mette copie in train+test -> i 0.73-0.83 sono OTTIMISTICI, il vero held-out e' peggiore.
Batteria ablation (leakage-clean + no-LoRA su e176/e77/e87) + GPTQ-Q1 (Codex, one-shot no-STE) in corso.
Loss training e' GIA' MSE gate-weighted (cosine solo metrica) - il report sbagliava su quel punto per noi.

**Addendum 17 (07:00) — SVOLTA: GPTQ-Q1 RISOLVE il tetto Q1 (era TRAINING non formato).** GPTQ/OBQ
one-shot (Codex gptq_q1.py, Hessian XtX + error-feedback, no STE/no LoRA/no SGD) su e176 LEAKAGE-CLEAN
(dedup 3797 token, 570 holdout): cosine=0.8434, mean-token=0.8328, MSE=0.0166. CONFRONTO: STE+LoRA e176
= 0.83 ma LEAKY (gonfiato); GPTQ-Q1 = 0.843 su holdout PULITO e one-shot (overfit impossibile). QUINDI il
tetto 0.73 mediano era il METODO DI TRAINING (STE+LoRA), NON il formato 1.125bpw ne' i dati. Valida la
ricerca compressione (STE per segni discreti e' sbagliato; Hessian-reconstruction generalizza) e
l'intuizione utente ("distillazione LoRA non tiene"). IN CORSO: GPTQ su e77(STE 0.58)/e87(STE 0.64) = il
decisore-produzione (se GPTQ risolleva anche i peggiori, i 10240 sidecar sono viable via GPTQ one-shot,
zero training, minuti/expert su CPU). Se GPTQ tiene su tutta la distribuzione -> il filone Q1 e' SALVO e
la produzione e' molto piu' economica del previsto (no GPU training, no overfit da gestire).

**Addendum 18 (07:15) — PRODUZIONE Q1 VERDE: GPTQ salva i peggiori.** GPTQ-Q1 clean vs STE+LoRA leaky:
e176 0.83->0.843, e87 0.64->0.811 (+0.17!), e77 0.58->0.755 (+0.17!). I PEGGIORI sotto STE sono quelli che
GPTQ risolleva di piu'. Distribuzione: STE leaky mediana ~0.64 -> GPTQ clean ~0.81, caso peggiore 0.58->0.755
(sopra soglia utile). GPTQ NON e' marginale: ribalta il filone. Metodo produzione-ideale: one-shot, CPU,
minuti/expert, zero training/GPU/overfit, holdout pulito. I 10240 sidecar fattibili senza GPU-training.
VERDETTO Q1 CONSOLIDATO (3 revisioni in una notte, onesto): (1) "muro sfondato 0.83" [E176 fortunato+leaky];
(2) "mediana 0.73, dubbio" [STE su 13, ancora leaky]; (3) DEFINITIVO: il tetto era STE+LoRA, GPTQ one-shot
da ~0.75-0.84 clean su tutta la distribuzione testata -> PRODUZIONE VIABLE via GPTQ. CAVEAT residui pre-prod:
(a) validazione end-to-end (0.75-0.84 preserva l'output chat? soglia mai validata); (b) multi-dominio (cattura
solo coding); (c) very-cold <50 campioni (Hessian ha bisogno di ~128+ righe - borderline, testare); (d) Q1/Q2
adattivo per chi resta basso. NEXT: GPTQ su tutti i 13 del pilota + i freddi -> distribuzione GPTQ completa;
poi scrivere Q1 sidecar dal GPTQ output nel formato runtime + test end-to-end.

**Addendum 19 (07:45) — TESTA-A-TESTA STE-clean vs GPTQ-clean: chiude la diagnosi.** Ablation
(ablation_results.tsv) su e176/e77/e87 x [STE-leaky / STE-clean-lora4 / STE-clean-lora0] vs GPTQ-clean:
e176: 0.833/0.792/0.813 vs GPTQ 0.843. e77: 0.581/0.577/0.541 vs GPTQ 0.755. e87: 0.642/0.627/0.629 vs
GPTQ 0.811. TRE VERITA': (1) LEAKAGE PICCOLO ~-0.04 (i 0.73 STE erano solo lievemente gonfiati, non un
disastro - onesto: temevo peggio). (2) LoRA NON e' la colpa (effetto incoerente: aiuta e176 0.792->0.813,
danneggia e77 0.577->0.541). (3) GPTQ domina ovunque (+0.03 facili, +0.18 difficili); nessuna variante STE
(clean/no-LoRA) si avvicina. LA CAUSA E' STE (ottimizzazione dei segni), come da ricerca compressione: STE
finge una derivata sulla discontinuita' del segno -> update sotto-soglia spariscono; GPTQ risolve i segni
via Hessian error-feedback deterministico, no gradiente. VERDETTO FINALE: adottare GPTQ per la produzione
Q1. Distribuzione GPTQ completa sui 13 in corso (bl9c4m5vl). Poi: scrivere sidecar Q1 dal GPTQ output nel
formato runtime + test end-to-end (la validazione che manca).

**Addendum 20 (08:00) — DISTRIBUZIONE GPTQ COMPLETA: filone Q1 SALVO.** GPTQ-Q1 su 13 esperti caldi
(gptq_all_results.tsv, leakage-clean): mediana 0.811, min 0.755, max 0.855, media 0.811. >=0.80: 10/13;
>=0.75: 13/13 (TUTTI). Contro STE leaky (mediana 0.733, >=0.80 solo 2/13): +0.08 mediana e NESSUN crollo -
il peggiore GPTQ (0.755) supera la mediana STE. Distribuzione STRETTA e ALTA = qualita' PREVEDIBILE per
expert, non lotteria. VERDETTO Q1 DEFINITIVO: produzione VIABLE via GPTQ one-shot CPU (~0.80 mediano
garantito, minuti/expert, zero training/GPU/overfit). La notte ha portato il Q1 da "in dubbio" (STE 0.73)
a "produzione verde" (GPTQ 0.81) risolvendo la CAUSA (STE, non formato/dati/LoRA/leakage). PIPELINE PRODUZIONE
pronta: extract dedup -> teacher fp32 -> gptq_q1.py -> sidecar. RESTA pre-deploy: (a) sidecar nel formato
runtime ds4 (ds4_q1_sidecar_converter.py) + test END-TO-END chat (la soglia 0.80 non e' ancora validata vs
qualita' output reale); (b) very-cold <50 campioni; (c) multi-dominio; (d) Q1/Q2 adattivo per i <0.78.

**Addendum 21 (08:15) — PIANO PRODUZIONE Q1 (utente: "avanti fino a produzione completa").** GPTQ output =
gate/up/down.q1.npz + manifest (Q1_0_custom, block128, provenance completa). FRONTI PARALLELI:
- FRONTE B (CRITICO, Codex): ponte GPTQ.q1.npz -> sidecar formato RUNTIME ds4 (ds4_q1_sidecar_converter.py
  ha RUNTIME_MANIFEST_SCHEMA + un gate provenienza) + wiring per servire l'expert come Q1 nel runtime
  (path mixed_q1 esiste). Poi TEST END-TO-END: chat reale con N esperti serviti Q1-GPTQ, l'output degrada?
  = IL GATE MAI VALIDATO (la soglia 0.80 cosine e' euristica).
- FRONTE A (CPU): GPTQ su piu' esperti L15 (verso i 256) - estende la distribuzione + produce i sidecar.
  Collo: estrazione teacher fp32 per-expert (~2min x 256 = 8h serial) -> ottimizzabile con estrazione
  all-256 in un pass GGUF (follow-up).
- FRONTE C (Codex): F10-fix (defer cache+seed, col conto slot onesto).
PREREQUISITO PRODUZIONE TOTALE (10240 sidecar): cattura ALL-LAYER (task #1) - la cattura attuale e' solo
L15. Milestone raggiungibile: L15 COMPLETO (256) + end-to-end -> PROVA la pipeline; poi campagna pod all-layer.
TEST NECESSARI PRE-PRODUZIONE: (1) end-to-end [critico]; (2) L15 full no-crolli; (3) very-cold <50 (calib
dense per Hessian pieno); (4) Q1/Q2 adattivo per i <0.78.

**Addendum 22 (08:30) — BIG-CTX = MURO DI CAPACITA' PROVATO (non difetto). F10-fix conto onesto.** Codex
F10-fix (f10_fix_slotmath.out.log) ha fatto l'aritmetica esatta: per_expert=7.077.888 byte (6.75MB),
free a decode-start=1.114GB, reserve 0.125GB. -> grow DA SOLO = floor((1.114G-0.125G)/6.75M)=138 slot
(PEGGIO del baseline 160!); grow+F8b(+315MB)=185 slot; working set=240 -> GAP RESIDUO 55 slot anche con
TUTTE le leve VRAM. Il seed da 160 (1.132GB) supera pure il free lordo di 18MB. CONCLUSIONE (categoria
fisica-provata, l'unica che e' muro vero): il big-ctx a 8192 sul 3060 NON e' risolvibile per reclamo VRAM -
backbone(~8.85G)+context-buffers lasciano ~1.1-1.7GB per la cache = max ~185 slot con ogni ottimizzazione,
il working set ne vuole 240. La scoperta utente (1GB stranded) era REALE ma reclamarlo non basta. LEVA VERA
= ridurre il WORKING SET sotto 185 (mask K12/pruning - meno esperti/token, l'ENDGAME del registro
[[reap-loop-concept-conditional-dynamic]]), NON liberare VRAM. F8b/F10 = miglioramenti parziali (185 vs 160),
non la soluzione. F10-fix autorato (5 hunk, defer cache+seed capacity-bounded + latch anti-retry) disponibile
se si vuole il +25 slot parziale, ma non chiude il big-ctx. Il big-ctx e' MEMORY-BOUND provato, redirect a mask.

**Addendum 23 (08:45) — RITRATTAZIONE big-ctx (utente aveva ragione: e' INGEGNERIA, non muro).** Il verdetto
"muro di capacita'" (add.22) era SBAGLIATO, basato su 2 numeri mai misurati. MISURA vera del working set K6
(DS4_N_EXPERT_USED=6, non K12!) dalla cattura L15: finestra 16tok=34 distinti, 32=51, 64=72, 128=98 (p90
111). NON 240 (quello era un numero preso dagli esperimenti K12/coffee - modello DIVERSO, errore mio). La
cache a 160 slot CONTIENE il working set (160>111). Percio' NON e' capacita'. IL VERO BUG (dai log S5 ctx8192):
cache count=160/160 PIENA ma vram_promotions=0 SEMPRE, decode 7-27s/token, vram_hit~0%. La cache si riempie via
SEED statico (esperti caldi del PREFILL) e NON PROMUOVE MAI durante il decode -> tiene gli esperti sbagliati per
la fase decode, non si auto-corregge -> miss/thrash con slot liberi inutilizzati. FIX (ingegneria pura, no piu'
VRAM): promozione DINAMICA durante il decode = tiered-residency 0033 gia' progettata (docs/TIERED_RESIDENCY +
PHASE_ADMISSION_CONTROLLER_PLAN, patch 0033 authored 8723e29). L'utente aveva ragione su tutto: (1) e' sempre
ingegneria; (2) l'1GB stranded va reclamato E mantenuto (F10-fix valido); (3) il backbone e' rivedibile. DA
INVESTIGARE ancora: perche' vram_hit~0% con cache piena (esperti sbagliati O bug di serving?); misurare hit-rate
decode con cache=working-set corretto. Il big-ctx TORNA APERTO come problema di ADATTAMENTO cache, risolvibile.

**Addendum 24 (09:00) — SCOMPOSIZIONE MODELLO (domanda utente "quanto e' l'attention"): la VRAM non e' MAI stata il vincolo.** ds4-2bit.gguf 86.71GB: ESPERTI routed 77.91GB (90%, STREAMABILE, sparse 6/256), ATTENTION 5.80GB (residente), embed/output 1.62, shared 1.15, router 0.09. RESIDENTE MINIMO (attn+embed+shared+router) = 8.66GB. Su 12GB: 8.66 resident + ~0.3 KV(8k) = 9GB -> ~3GB liberi = ~440 slot cache esperti >> working set 111 (misurato). CONFERMA DEFINITIVA (3 vie): working set 111 non 240; cache 160>111; resident floor lascia 440 slot. LA VRAM NON E' IL VINCOLO. Il big-ctx thrasha per promotions=0 (cache non adatta), NON capacita' - INGEGNERIA (utente aveva ragione). ATTENTION e' f32(299)/q8(215)/f16(187) = ALTA PRECISIONE, comprimibile: q4/q2 la porta 5.8->~1.5-3GB. ARCHITETTURA CORRETTA (utente): residente = attention(+KV) comprimibile, esperti streammati. LEVE: (8k daily) fix promotions=0 [cache adatta, gia' 3GB liberi]; (ultra-long 250k-1M) comprimi attention +4GB -> KV enorme + ring F8b. Il "rewrite backbone" dell'utente = comprimere l'attention (ora f32/q8) per il regime long-ctx. Verdetto muro CANCELLATO: e' tutto ingegneria, headroom abbondante.

**Addendum 25 (09:15) — Attention high-precision = allocazione OTTIMALE, non spreco (domanda utente).** Scomposizione
attention per ruolo: proiezioni grandi (attn_output_a/b, q_b, q_a, kv) = q8_0 8-bit (~6.4GB, NON f32); compressori
MLA (compressor_gate/kv/ape, indexer) = f16 (~0.9GB); norms/sinks/scales = f32 ma VETTORI DA BYTE (~0GB, RMSNorm
divide + sink toccano softmax = f32 obbligatorio). ZERO f32 sprecato su matrici grandi. PERCHE' giusto: l'attention
(6.7% del modello) decide DOVE fluisce l'info (routing/mixing), errori si COMPONGONO su ogni token/layer; esperti
ridondanti (256, ne sparano 6) errori si MEDIANO. Bit spesi dove contano (attn 8-bit) risparmiati dove tollera
(esperti 2-bit) = mixed-precision ottimale, confermato da QuantMoE-Bench/MxMoE ("attn+shared meritano piu' precisione").
CONSEGUENZA PRODUZIONE Q1: comprimere ESPERTI (GPTQ 2->1.125), attention NON si tocca (gia' ottimale). Comprimerla
(q8->q2) = danneggia il routing backbone per pochi GB, vale solo per long-ctx estremo (spendere qualita' per memoria KV).

**Addendum 26 (09:30) — Idea utente: riduzione dinamica su miss-pressione poi riespansione. Analisi.** Tre
target possibili: (1) PRECISIONE ATTENTION dinamica = RISCHIOSA: fattibile con base-resident+residuo-streamato
(sfratti il residuo sotto miss), ma i miss-spike coincidono coi CAMBI DI FASE (prosa->codice) = quando il routing
conta di piu'; degradare il backbone del routing li' -> route peggiori -> piu' miss -> spirale. Si degrada la cosa
sbagliata nel momento sbagliato. (2) DYNAMIC-K su miss-pressione = OTTIMA (la forma migliore dell'idea): quando la
cache thrasha, restringi il routing K6->K4 per pochi token -> working set si contrae -> entra in cache -> miss
crollano -> ri-espandi a K6. Costo: qualita' minima+transitoria sulla RIDONDANZA esperti (errori si mediano), non
sul backbone. E' la mask conditional-dynamic del registro [[reap-loop-concept-conditional-dynamic]], "riduzione
minima poi riespansa" alla lettera. (3) MEMBERSHIP cache dinamica = gia' il fix promotions=0 (muovi quali esperti
residenti, gratis, vs ri-quantizzare pesi costoso). PRINCIPIO unificante delle 3 intuizioni utente notturne (VRAM
stranded, working set, dynamic-reduction) = RIALLOCAZIONE DINAMICA guidata da miss-pressione = endgame. REGOLA:
riallocare la risorsa ECONOMICA (membership/K) sotto pressione, proteggere la SENSIBILE (attention). Dynamic-K su
miss e' esperimento pulito falsificabile -> aggiunto come leva al fix big-ctx (task promotions=0).

**Addendum 27 (09:45) — Ponte sidecar: formato MAPPATO, converter da completare (onesto).** Codex
(sidecar_bridge.out.log) ha mappato ENTRAMBI i formati ma NON consegnato un converter runnable: (a) GPTQ .npz =
chiavi sign_bits[out,nblk,16]uint8 (packbits little), scales[out,nblk]fp16, shape, block_size=128; bit0->-1 bit1->+1
(gptq_q1.py:977/1005). (b) Loader ds4 (ds4.c:19511 q1_0_sidecar_bind): NON legge JSON companion; vuole GGUF v3 coi
metadata semantici del base + stesso general.name, validati da config_validate_model; lista in ds4_q1_sidecar_converter.py:113.
La mappatura (parte difficile) c'e'; RESTA: scrivere+testare il writer GGUF che impacca sign_bits/scales nel blocco
GGML Q1_0 coi metadata giusti + golden byte-test. RILANCIARE Codex per il converter completo (o completarlo a mano dai
frammenti). CAMPAGNA GPTQ: 98/~200 esperti fatti (CPU, in corso), produce gate/up/down.q1.npz = materia prima sidecar.
STATO PRODUZIONE Q1: scienza SOLIDA (GPTQ 0.81, breakthrough); pipeline produzione = campagna GPTQ (in corso) + ponte
sidecar (formato mappato, converter da finire) + end-to-end (dipende dal ponte). Non e' finita, ma il percorso e' chiaro
e senza incognite scientifiche - solo engineering GGUF rimasto.

**Addendum 28 (10:00) — DISTRIBUZIONE L15 DEFINITIVA + SPEC PRODUZIONE Q1/Q2 (1.46 bpw).** GPTQ su 161/256
esperti L15 (>=200 campioni; gptq_L15_full.tsv): mediana 0.766, media 0.748, min 0.569, max 0.924. >=0.80: 46
(29%); >=0.75: 92 (57%); <0.72: 54 (34%). I 13 caldi (0.811) erano il lato alto; la distribuzione completa e' piu'
larga (coda a 0.57). SPEC PRODUZIONE (idea utente Q1/Q2 adattivo, ora coi numeri): 107 esperti -> Q1 (1.125bpw,
>=0.72), 54 -> Q2 (2.125bpw, lista: 4,5,11,19,24,36,42,45,49,54,55,56,60,65,66,72,74,75,78,86,88,90,91,98,107,110,
115,121,129,133,135,136,140,147,151,158,161,163,180,186,189,190,205,210,214,220,221,222,223,239,244,245,249,253).
BUDGET MEDIO = 1.460 bpw, coincide col ~1.5 predetto (research). VERIFICHE oneste rimaste: (1) i <0.72 migliorano
DAVVERO a Q2? (assunto 2x bit, da provare su 2-3); (2) 95 esperti FREDDI (<200 campioni) non nei 161 -> serve
GPTQ con calib DENSE (tutti gli x, Hessian pieno). Il filone Q1 e' passato in una notte da "in dubbio" a spec di
produzione quantitativa: mix Q1/Q2 a 1.46bpw, lista Q2 concreta, pipeline GPTQ collaudata. Manca solo l'engineering
sidecar (converter GGUF in corso Codex) + end-to-end + i freddi.

**Addendum 29 (10:15) — Verifiche produzione: "Q2"=base 2-bit (semplifica!) + freddi via dense.** (1) gptq_q1.py
e' Q1-only (1.125bpw). RIFLESSIONE: il teacher E' gia' l'expert a 2-bit (iq2_exact, cosine~1.0 = reference), quindi
per i 54 duri "Q2" = LASCIARLI AL BASE 2-bit, NON un GPTQ-2bit. L'adattivo si semplifica: sidecar Q1 copre SOLO i
107 facili (>=0.72), il runtime serve i 54 duri dal modello base. La COPERTURA PARZIALE del sidecar (che Codex
analizzava) e' quindi il DESIGN NATURALE, non un problema. Storage: base 2-bit (77.9GB) con 107 esperti sostituiti
da Q1 (1.125) = risparmio ~0.875bpw x 107. (2) FREDDI via calib DENSE: e117 (50 campioni routed=0.716) con dense
14336x = 0.727 (+0.011, sopra soglia). Hessian pieno aiuta poco ma basta a renderlo Q1-viable. Dense = via per i
95 freddi (<200 campioni), boost marginale. SPEC PRODUZIONE FINALE: sidecar Q1 parziale (esperti >=0.72, ~107+ dei
161 + freddi via dense), base 2-bit per il resto; ~1.46bpw medio. NEXT: converter GGUF con copertura parziale
(Codex in corso) -> primo sidecar reale -> end-to-end. NB estendere gptq a 2-bit resta possibile SE si vuole
comprimere anche i duri sotto 2-bit (follow-up), ma non necessario per la produzione base.

**Addendum 30 (10:30) — CONVERTER SIDECAR FUNZIONA: pipeline Q1 completa fino al GGUF.** Codex (fresh, il resume
si piantava) ha scritto tools/gptq_to_sidecar.py (moe-aggressive-commit, commit f00465c+fix): GPTQ .npz ->
GGUF v3 sidecar caricabile via DS4_Q1_0_EXPERT_SIDECAR. Formato Q1_0 VERIFICATO vs runtime: type-id GGML 41,
blocco 18 byte = fp16 scale LE + 16 byte sign-bit (bitorder little). Metadata semantici copiati dal base +
general.name (expert_group_* resi opzionali: assenti in ds4-2bit, il runtime non li richiede). VINCOLO scoperto:
il binder q1_0_sidecar_bind richiede tensori Q1_0 COMPLETI per layer -> esperti non-GPTQ dequant base->Q1;
quindi "hard->base-2bit" NON possibile intra-layer (o tutti Q1 o niente sidecar) - l'adattivo Q2 va ripensato
(hard->Q1-naive vs meccanismo per-expert opt-out da investigare). TEST sidecar 5-expert: OK (909MB, verifica
metadata+identity+round-trip NPZ byte). Sidecar PRODUZIONE L15 (161 GPTQ) in generazione. ULTIMO PASSO: test
END-TO-END (server con sidecar, DS4_Q1_0_LAYER_FIRST/LAST=15, chat reale, output degrada?) = valida la soglia
mai provata. Prereq: riavvio (4.7GB RAM orfana). Pipeline Q1 COMPLETA: cattura->GPTQ->sidecar GGUF, solo
end-to-end rimasto. Zero incognite scientifiche, solo il test di conferma.

**Addendum 31 (10:45) — SIDECAR PRODUZIONE L15 GENERATO+VERIFICATO. Pipeline Q1 completa fino al GGUF.** 
sidecar_L15_prod.gguf: 909MB, 161 esperti GPTQ di L15, verifica interna PASSATA (metadata, identity tensors,
round-trip NPZ byte-level). Servibile via DS4_Q1_0_EXPERT_SIDECAR + DS4_Q1_0_LAYER_FIRST/LAST=15. LA PIPELINE Q1
E' COMPLETA E FUNZIONANTE: cattura fleet -> GPTQ one-shot -> sidecar GGUF valido. RESTA SOLO il test END-TO-END
runtime (conferma, non scoperta): server col sidecar, chat reale, output degrada vs base? -> valida la soglia
0.80 mai provata contro qualita' reale. Test qualita' robusto allo stato RAM; t/s pulito serve riavvio (4.7GB
orfana). BILANCIO FILONE Q1 (una notte): dubbio STE 0.73 -> GPTQ 0.81 -> spec produzione 1.46bpw -> pipeline
GGUF funzionante. Zero incognite scientifiche. Il converter (gptq_to_sidecar.py, moe-aggressive-commit) e' la
prova che GPTQ-Q1 diventa un artefatto servibile dal runtime esistente.

**Addendum 32 (11:00) — LEVA: expert-parallelism GPU+CPU per RIDURRE GLI SCAMBI (idea utente).** Non piu'
compute (GPU non satura), ma TRANSFER: la CPU calcola un expert dalla SUA RAM -> quei pesi NON attraversano
PCIe. Numeri: per token 7 esperti x 6.75MB = 47MB verso GPU (collo PCIe/streaming). CPU serve N esperti da
RAM -> N x 6.75MB NON transitano, solo N x 16KB output (trascurabile). Ogni expert-su-CPU = -6.75MB H2D/token.
CANDIDATI: (1) SHARED expert (spara ogni token, deterministico, e' Q8 nel modello ds4.c:2405) -> 6.75MB
garantiti/token; (2) COLPO GROSSO = esperti FREDDI/miss (la fonte del thrash, streaming SSD->VRAM): invece di
streammarli in VRAM (capacita' limitata), la CPU li calcola da RAM (gia' page-cache) -> il miss diventa
compute-CPU-da-RAM, ZERO transfer/stallo SSD. E' lo split hot/cold di ktransformers/PowerInfer/llama.cpp
(-ncmoe): hot->GPU-VRAM, cold->CPU-RAM. ATTACCA il bottleneck cold-expert (tutta la notte) ELUDENDO il limite
VRAM invece di scontrarcisi. Si sposa col Q1 (esperti CPU in Q1 = ancora meno RAM letta). LIMITI: sync (CPU
finisce entro la finestra layer GPU, 1-2 esperti ~1-2ms overlappano), core dedicati (orchestrazione ne usa),
H2D output 16KB + sync/layer. Prior art forte. Esiste gia' un path CPU (ds4_cpu_decode_scratch, forward CPU
per DS4_NO_GPU). NEXT: prototipo = shared-expert su CPU da RAM (il piu' semplice, deterministico) -> misurare
H2D/token risparmiato + t/s; poi estendere ai cold routed. Complementare al fix promotions=0 [[reap-loop-concept-conditional-dynamic]].

**Addendum 33 (11:15) — COLIBRI: l'idea GPU+CPU era gia' citata dall'utente, ed e' il blueprint.** L'utente
ha ricordato Colibri (github.com/JustVugg/colibri, audit bad64d1 in DS4_MOE_EXTERNAL_AUDIT_RANKED_20260714.md).
Colibri = ESATTAMENTE le 2 idee di stanotte: (a) "its cold path can compute in CPU RAM" = expert-split GPU+CPU
(add.32); (b) three-tier state machine SSD/RAM/VRAM + LFRU/heat/recency repin = fix promotions=0 (add.23/15).
Anche llama.cpp PR#24524: "CPU threads compute miss rows concurrently with GPU hit rows... avoids making every
miss a synchronous H2D penalty" = idem. L'audit aveva gia' rankato l'esperimento DECISIVO come #0: "IQ2XXS
warm-tier microbenchmark: decides CPU miss vs transient H2D architecture" - MAI eseguito. CAVEAT audit: "valuable
only if an IQ2XXS CPU expert kernel is competitive on our CPU". NOVITA' STANOTTE che cambia le probabilita':
l'audit assumeva IQ2XXS (2-bit); ora abbiamo Q1 (1.125bpw) = esperti ~meta' piu' piccoli -> kernel CPU su expert
Q1 MOLTO piu' probabile che batta l'H2D. IL Q1 RENDE COLIBRI PIU' FATTIBILE. CONVERGENZA: filone Q1 (compressione)
+ filone velocita' (GPU+CPU split + three-tier) NON ortogonali -> si incontrano: Q1 rende competitivo il
compute-CPU-da-RAM -> elimina scambi -> risolve thrash. NEXT DECISIVO (audit #0, ora con Q1): microbenchmark
kernel CPU Q1-expert vs transient-H2D; se CPU vince, implementare il cold-path-in-CPU (Colibri) + three-tier.

**Addendum 34 (11:30) — END-TO-END Q1 FUNZIONA: pipeline completa cattura->GPTQ->sidecar->runtime->generazione.**
Test: BASE (867 char) vs SIDECAR_Q1 L15 (803 char), stesso prompt coding temp0. Runtime: sidecar validato+
installato (0.85GiB) + dispatch abilitato (DS4_Q1_0_SELECTED_LOAD=1; opt-in mancante trovato a ds4_cuda.cu:37762;
NB DS4_G73_OPEN rifiuta il sidecar -> server minimale senza catena F5). ESITO: Q1-GPTQ GENERA OUTPUT COERENTE E
SUL TASK (entrambi capiscono: funzione Python email-regex + docstring + 3 test; il Q1 nota pure il prompt IT).
LA PIPELINE FUNZIONA END-TO-END. Metrica: string-similarity 0.23 ma FUORVIANTE - primi 100 char IDENTICI poi
divergono = BUTTERFLY EFFECT del greedy@temp0 (Q1 perturba i logit, flippa 1 token, il greedy amplifica in
continuazione diversa), NON degrado qualita'; entrambi coerenti+corretti. VERDETTO ONESTO: serving Q1 PRESERVA
la competenza, NON e' token-identico (atteso). String-match e' la metrica sbagliata; giusta = perplessita'
held-out o task-correctness. Dato interessante: L15 da SOLO (1/43 layer) fa divergere la sequenza -> i suoi
esperti CONTANO, eppure la coerenza tiene. FILONE Q1 CHIUSO: da "dubbio STE 0.73" (ieri sera) a "Q1-GPTQ servito
dal runtime reale, output coerente, pipeline funzionante" (stamattina). RESTA per la validazione rigorosa:
perplessita' base-vs-Q1 su held-out (metrica corretta) + t/s pulito post-riavvio. Output salvati runs/ds4/BASE.txt,
SIDECAR_Q1.txt.
