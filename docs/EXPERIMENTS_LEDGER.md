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
