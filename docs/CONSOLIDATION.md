# Confidence-Scheduled Predictive Expert Offload — Documento di Consolidamento

**Obiettivo:** far girare un MoE gigante DeepSeek-class (235B-A22B) su una singola RTX 3060 (12 GB) a **qualità piena** e per **uso generale**, tramite offload predittivo degli esperti, schedulato per confidenza e guidato dalla saliency.

**Tesi in una riga:** i mattoni (expert prediction, offload, mixed-precision, informed pruning, confidence calibration) sono tutti noti e citati; il palazzo — il *transfer* della macchina confidence-scheduled di DSpark sulla cache reattiva di DwarfStar, resa **predittiva** (non reattiva), **saliency-guidata** (non a frequenza), **dinamica** (non calibrazione statica), e **misurata con controlli onesti** (random, massa, reactive) — è il contributo.

---

## 1. Tabella dei Findings (numeri esatti + verdetto novelty)

| ID | Claim | Numeri esatti | Verdetto | Prior art principale |
|----|-------|---------------|----------|----------------------|
| **F1** | ~50% esperti prunabili near-lossless su dominio; selezione informata ≫ random | 30B dominio K50 saliency **ppl 5.50 vs full 5.56** (near-lossless) | **GIÀ-NOTO** (replichiamo) | EASY-EP [2504.06792], PreMoE [2505.17639], REAP [2510.13999], NAEE [2402.14800] |
| **F2** | FT col router **sbloccato disperde** il routing; è il **DOMINIO** (non il FT) a concentrare; FT meno prunabile; cross-mask(base) ≥ ft-mask | N_eff **49.9 → 56.9**; FT retention field-acc K25 **0.675** / K50 **0.663** / K70 **0.595** (base near-lossless); random FT collassa K50 **0.356** / K70 **0.000** | **NOVELTY-CANDIDATO** (misura con controlli; da replicare seed/distribuzione) | ESFT [2407.01906], Demons-in-Detail [2501.11873], Guo [2505.22323], ReMoE [2412.14711] (contro-caso), ST-MoE [2202.08906], confounders [2604.09780], [2601.03425], Mixtral [2401.04088] |
| **F3** | Dominio-vs-generale: il divario cresce con la scala | 30B: dom K50 **1.13x**, gen K50 **2.14x**, K70 **3.56x**. 235B: dom K50 **4.32x**, gen K50 **21.9x** (ppl **97.6 vs 4.46**) | **GIÀ-NOTO** (quantifichiamo) | EASY-EP Table 4 [2504.06792], PreMoE [2505.17639], Less-is-MoE [2606.05538] |
| **F4** | Predizione cross-layer next-expert + **LOOP ADATTIVO** confidence-scheduled | Static recall @25%: hidden **0.986**, Markov **0.865**, random **0.25**. Loop miss-rate @25% (30B gen): random **0.209** / reactive-LRU **0.154** / **adaptive 0.117** (adaptive vs random **−44%**, dominio **−51%**; vs reactive **13–25%** quando confidente). Markov naive **perde** contro reactive | **NOVELTY-CANDIDATO** (transfer + loop; da validare end-to-end) | Fate [2502.12224], Pre-gated [2308.12066], SiDA [2310.18859], ExpertFlow [2410.17954], Mixtral-offloading [2312.17238] |
| **F5** | Offload predittivo 235B su 12 GB (regime estremo) | Working set ~22B attivi + hot-set salienti in VRAM; resto RAM/SSD int4/int2 | **MECCANISMI NOTI**, regime estremo | ProMoE [2410.22134], MoE-Infinity [2401.14361], PowerInfer-2 [2406.06282], HOBBIT [2411.01433] |
| **F6** | Mixed-precision esperti freddi | int4 cold **LOSSLESS** (K50 e K70); int2 gen **1.42x** (K50) / **1.97x** (K70) vs DROPPED **2.33x** / **5.42x** | **GIÀ-NOTO** (confermiamo) | MC-MoE [2410.06270], MxMoE [2505.05799], MoQE [2310.02410], MoPEQ [2509.02512] |
| **F-ctrl** | **Massa < random a scala** (il controllo che smaschera) | 235B K50 per **massa ppl 18.7 (4.32x)** > **random 9.2 (2.12x)** → la frequenza è criterio **peggiore del random** a 235B. Su 30B saliency > massa nel verso normale (dom K50 **5.50 vs 6.26**; gen K50 **9.36 vs 12.23**) | **NOVELTY-CANDIDATO** (esposto solo dal controllo random; confound int4) | How-to-Score-Experts [2606.15716], DeepSeekMoE [2401.06066], REAP [2510.13999] |

### Attribuzione: dirette vs convergenti

- **Ispirazioni DIRETTE** (l'utente le conosceva solo da video, non dai paper): **DwarfStar** (`antirez/ds4`) e **DSpark** (`deepseek-ai/DeepSpec`). Da qui viene l'idea generativa del transfer.
- **Prior art CONVERGENTE** (scoperto in fase di verifica, allinea/precede singoli mattoni ma non il palazzo): tutti i paper arXiv elencati sopra.
- **Confound noti e dichiarati:** 30B in **bf16** vs 235B in **GPTQ-int4** (il reversal massa<random è 235B-only, int4-only); **N=152**, seed singolo, in-distribution FT per F2; predittore hidden **non ancora dentro** il loop reale.

---

## 2. Citazioni complete (con LINK arXiv), raggruppate per tema

### Tema A — MoE expert pruning (F1, F3, F-ctrl)
- **EASY-EP** — Domain-Specific Pruning of Large MoE Models with Few-shot Demonstrations — Dong et al., 2025 — https://arxiv.org/abs/2504.06792
- **PreMoE** — Proactive Inference for Efficient MoE — Pei et al., 2025 — https://arxiv.org/abs/2505.17639
- **REAP** — REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression — Lasby et al., 2025 — https://arxiv.org/abs/2510.13999
- **NAEE** — Not All Experts are Equal: Efficient Expert Pruning and Skipping — Lu et al., ACL 2024 — https://arxiv.org/abs/2402.14800
- **Less-is-MoE** — Trimming Experts in Domain-Specialist LMs — He et al., 2026 — https://arxiv.org/abs/2606.05538
- **How-to-Score-Experts** — Unified Formulation and Selection Principle for One-Shot MoE Pruning — Liu et al., 2026 — https://arxiv.org/abs/2606.15716
- **DeepSeekMoE** — Towards Ultimate Expert Specialization — Dai et al., ACL 2024 — https://arxiv.org/abs/2401.06066

### Tema B — Expert specialization & routing (F2)
- **ESFT** — Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning — Wang et al., EMNLP 2024 — https://arxiv.org/abs/2407.01906
- **Demons-in-Detail** — On Implementing Load-Balancing Loss for Specialized MoE — Qiu et al., ICML 2025 — https://arxiv.org/abs/2501.11873
- **Advancing Expert Specialization for Better MoE** — Guo et al., NeurIPS 2025 (Oral) — https://arxiv.org/abs/2505.22323
- **ReMoE** — Fully Differentiable MoE with ReLU Routing (contro-caso) — Wang et al., ICLR 2025 — https://arxiv.org/abs/2412.14711
- **ST-MoE** — Designing Stable and Transferable Sparse Expert Models — Zoph et al., 2022 — https://arxiv.org/abs/2202.08906
- **The Myth of Expert Specialization** — Routing Reflects Geometry (confounder) — Xi Wang et al., 2026 — https://arxiv.org/abs/2604.09780
- **The Illusion of Specialization** — Domain-Invariant "Standing Committee" (confounder) — Yan Wang et al., ACL 2026 — https://arxiv.org/abs/2601.03425
- **Mixtral of Experts** — Jiang et al., 2024 — https://arxiv.org/abs/2401.04088

### Tema C — Offload & prefetch per MoE serving (F5)
- **ProMoE** — Fast MoE-based LLM Serving using Proactive Caching — Song et al., 2024 — https://arxiv.org/abs/2410.22134
- **MoE-Infinity** — Efficient MoE Inference with Sparsity-Aware Expert Cache — Xue et al., 2024 — https://arxiv.org/abs/2401.14361
- **PowerInfer-2** — Fast LLM Inference on a Smartphone — Xue et al., MLSys 2025 — https://arxiv.org/abs/2406.06282
- **HOBBIT** — Mixed Precision Expert Offloading System — Tang et al., 2024 — https://arxiv.org/abs/2411.01433

### Tema D — Expert prediction (F4)
- **Fate** — Fast Edge Inference of MoE via Cross-Layer Gate — Fang et al., 2025 — https://arxiv.org/abs/2502.12224
- **Pre-gated MoE** — Algorithm-System Co-Design for Fast MoE Inference — Hwang et al., ISCA 2024 — https://arxiv.org/abs/2308.12066
- **SiDA-MoE** — Sparsity-Inspired Data-Aware Serving — Du et al., MLSys 2024 — https://arxiv.org/abs/2310.18859
- **ExpertFlow** — Predictive Expert Caching and Token Scheduling — He et al., 2024 — https://arxiv.org/abs/2410.17954
- **Mixtral-offloading** — Fast Inference of MoE LMs with Offloading — Eliseev & Mazur, 2023 — https://arxiv.org/abs/2312.17238

### Tema E — Mixed-precision quantization di esperti freddi (F6)
- **MC-MoE** — Mixture Compressor for MoE LLMs — Huang et al., ICLR 2025 — https://arxiv.org/abs/2410.06270
- **MxMoE** — Mixed-precision Quantization for MoE, Accuracy/Performance Co-Design — Duanmu et al., 2025 — https://arxiv.org/abs/2505.05799
- **MoQE** — Mixture of Quantized Experts — Kim et al., 2023 — https://arxiv.org/abs/2310.02410
- **MoPEQ** — Mixture of Mixed Precision Quantized Experts — Chitty-Venkata et al., 2025 — https://arxiv.org/abs/2509.02512

---

## 3. Ispirazioni dirette + la LEVA non tirata

### DwarfStar — `antirez/ds4` — https://github.com/antirez/ds4
Motore d'inferenza per DeepSeek-V4 in C (Metal/CUDA/ROCm), 17.4k stelle. Streaming **reattivo** degli esperti da SSD: cache in-RAM + load-on-miss, calibrazione **statica** stile imatrix, quantizzazione **2-bit uniforme** (IQ2_XXS / Q2_K), hot-preload **per frequenza**.

> **LEVA NON TIRATA:** nessun prefetch **predittivo** degli esperti in DECODE (solo miss-hide reattivo + prefill layer-ahead). Ha il draft **MTP** ma lo usa solo per speculare **token**, mai per prefetchare **esperti**. La leva predittiva è già dentro il motore e resta non tirata.

### DSpark — `deepseek-ai/DeepSpec` — https://github.com/deepseek-ai/DeepSpec
Speculative decoding semi-autoregressivo. Fornisce la macchina di calibrazione: **confidence head** σ(w·[hidden ; Markov-embedding]) che predice l'accettazione; **Sequential Temperature Scaling (STS)** — la *temperatura come calibrazione*, minimizzando l'ECE su held-out; **scheduler hardware-aware** che massimizza il throughput. DSpark punta questa macchina sui **token**, non sugli **esperti**.

### Il cerchio chiuso (contributo utente)
Trasferire la predizione confidence-scheduled di **DSpark** sulla cache reattiva di **DwarfStar**:
- **predittivo** (non reattivo) — il confidence head diventa un *prefetch verifier* per esperti;
- **saliency-guidato** (non a frequenza) — perché a 235B la massa è *peggiore del random* (18.7/4.32x vs 9.2/2.12x);
- **dinamico** (non calibrazione statica) — STS al posto dell'imatrix statico → regge cross-distribution;
- **draft MTP ripurposato** da speculazione token a prefetch esperti.

Risultato: collo di bottiglia dei cache-miss ridotto del **13–25%** vs il baseline reattivo DwarfStar-class (e **−44%/−51%** vs random), a qualità piena. + F2 (dissociazione FT-router) + massa<random-a-scala.

---

## 4. Skeleton del paper (incorpora le bozze delle sezioni)

**Titolo:** *Confidence-Scheduled Predictive Expert Offload: Running a 235B MoE at Full Quality on 12 GB*

- **Abstract** — [bozza pronta] regime 235B-on-12GB full-quality general; tre contributi C1/C2/C3; headline: mass-worse-than-random @235B (18.7/4.32x vs 9.2/2.12x), hidden recall 0.986, adaptive miss-rate 0.117 (−44%/−51%, 13–25% vs reactive); int4 lossless / int2 gen 1.42x/1.97x.
- **§1 Introduction** — [bozza pronta] §1.1 il bottleneck e la leva non tirata (DwarfStar reattivo, MTP inutilizzato per esperti); §1.2 cosa trasferiamo e da dove (DSpark confidence head + STS); §1.3 contributi C1 (cosa tenere residente, perché la frequenza è sbagliata a scala), C2 (predizione next-expert ante-gate), C3 (loop adattivo confidence-scheduled).
- **§2–3 Ablation protocol + confounder controls (C1 / F1 / F-ctrl)** — protocollo con controllo **random** esplicito; saliency vs massa; il reversal massa<random @235B; near-lossless in-domain 5.50 vs 5.56.
- **§F2 — Fine-tuning dissociates the router** — [bozza pronta] FT disperde (N_eff 49.9→56.9); dispersione costa prunabilità (K25 0.675 / K50 0.663 / K70 0.595; random collassa 0.356 / 0.000); cross-mask(base) ≥ ft-mask; disarmo di ReMoE (separare obiettivo dalla distribuzione); confounder geometry [2604.09780] + standing-committee [2601.03425].
- **§4 — Cross-layer next-expert predictor (C2 / parte di F4)** — [bozza pronta §4.2] recall @25%: hidden 0.986 / Markov 0.865 / random 0.25.
- **§F4 — The Adaptive Confidence-Scheduled Prefetch Loop (C3)** — [bozza pronta] la leva DwarfStar; loop reactive vs Markov-naive vs adaptive (0.209 / — / 0.117); transfer della macchina DSpark (confidence head → prefetch verifier; STS → miss-cost dial; hw-scheduler → budget-aware eviction); perché saliency-guidato e perché serve il controllo random.
- **§5 — Combining the Findings: three-tier system** — [bozza pronta] gerarchia VRAM/RAM/SSD; placement per saliency (non massa); residency = loop predittivo; precisione int4-default/int2-cold; perché il regime estremo è *favorevole* (gap dominio-generale cresce con la scala); tabella di posizionamento vs DwarfStar (5 sostituzioni).
- **§6 — Related Work** — [bozza pronta] cinque thread convergenti per tema + le due ispirazioni dirette con delimitazione onesta di cosa prendiamo e cosa aggiungiamo.
- **§7 — Limitations, Confounders, Future Work** — [bozza pronta] N=152/single-seed; confound 235B-int4 vs 30B-bf16; in-distribution FT per F2; predittore non ancora nel loop; implementazione DSpark-fedele ancora da fare; verdetti di novelty onesti (GIÀ-NOTO / NOVELTY-CANDIDATO / palazzo-non-mattoni).

**Nota di merge (dalle bozze):** l'id `2401.14088` che compare in un cluster di citazioni C1 è un refuso — correggere in `2401.04088` (Mixtral) o `2202.08906` (ST-MoE) al merge.

---

## 5. Next steps — implementare il loop DSpark-fedele

Priorità in ordine di rischio-per-la-tesi:

1. **[BLOCCANTE tesi] Portare il predittore hidden DENTRO il loop reale.** Oggi il recall statico (0.986) è misurato col predittore forte; il loop (0.117) è misurato con predittori più economici. Unire i due nel regime target (235B GPTQ-int4, offload su RTX 3060 12 GB, latenze SSD/PCIe reali) e verificare che il gap regga pagando il costo di calcolare il predittore hidden abbastanza presto da nascondere la load-latency dietro il compute.
2. **Implementazione DSpark-fedele.** Costruire l'esatto confidence head σ(w·[hidden ; Markov-embedding]); tunare **STS** minimizzando l'ECE su uno split held-out reale; costruire lo scheduler hardware-aware contro le figure misurate di memory-bandwidth e PCIe/SSD del 3060. Finché non esistono questi tre pezzi, "abbiamo trasferito la calibrazione di DSpark" è un'architettura proposta, e il 13–25% è ciò che la simulazione predice, non un risultato di produzione.
3. **Sciogliere il confound 235B-int4 vs 30B-bf16.** Replicare il controllo massa<random su **235B in bf16** per rendere pulito il claim di ordinamento-criterio (mass < random). È il finding più esposto al confound di precisione.
4. **Multi-seed + set più grande.** Portare N oltre 152 e più seed prima di promuovere i NOVELTY-CANDIDATE da "osservato sui nostri dati" a "stabilito". Prerequisito per tutte le CI strette.
5. **Cross-distribution FT per F2.** Eseguire un FT held-out cross-distribution per confermare che il router-unlock disperde anche fuori distribuzione (oggi F2 è in-distribution).
6. **Repurpose reale del draft MTP di DwarfStar.** Collegare la testa MTP già presente in `ds4` al prefetch degli esperti (non solo speculazione token) e misurare il guadagno end-to-end sul motore vero.
7. **Riportare tok/s misurato.** Sostituire gli "sketch" di footprint RAM/throughput con numeri GB e tok/s misurati sul 3060; finché mancano, tenere il **miss-rate** come risultato primario hardware-independent (scelta corretta ed onesta).

---

## Appendice — le due voci github aggiunte alle citazioni
- **DwarfStar (ds4)** — antirez — DeepSeek-V4 inference engine in C — https://github.com/antirez/ds4
- **DSpark (DeepSpec)** — deepseek-ai — semi-autoregressive speculative decoding — https://github.com/deepseek-ai/DeepSpec
