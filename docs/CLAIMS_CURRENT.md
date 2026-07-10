# CLAIMS_CURRENT — single source of truth

**Ultimo aggiornamento: 2026-07-10 (post-T1: controllo positivo full su pod + retro-grade funzionale L0-L3 + non-determinismo greedy prompt-dependent; corretta la riga STABILITA con grading)**

Questo file e l'UNICA fonte di verita sullo stato dei claim di reap-loop.
Se un altro documento del repo contraddice questa tabella, questo file vince.

Legenda stato: **CLOSED** = validato, difendibile; **OPEN** = misurato ma non concluso / CI attraversa 1.0 / lead-time da confermare; **RETRACTED** = non replica o artefatto, non citare; **SUPERSEDED** = rimpiazzato da framing migliore.

Numeri di riferimento hardware: "pod" = cache-1024/RAM-calda (pod cloud); "3060" = RTX 3060 12GB WSL (piu lento, I/O/capacity-bound). K su 256 esperti/layer salvo diverso avviso.

---

## Contributo (attuatori e metodo)

| Claim | Stato | Evidenza / numero corrente | Dove vive |
|---|---|---|---|
| Attuatore mask REVERSIBILE (0011) | **CLOSED** | Gate-bias -1e9 sui potati. ZERO violazione aderenza: le selezioni sono subset di keep, pruned=0 su tutte le mask. Il router calcola comunque tutti i 256 esperti; il bias agisce solo sulla selezione. | src/ attuatore mask (patch 0011), docs/paper/PAPER.md |
| Eval funzionale GRADUATO L0-L3 | **CLOSED** | Scala: L0 = non-parse / non-apre; L1 = apre ma feature rotta; L2 = presente con difetti minori; L3 = pieno + pulito. Rivela cio che ppl / repeat-rate non catturano. Metodo, non numero. | docs/paper/PAPER.md, harness eval graduato |
| Adaptive-K via COPERTURA di massa-gate | **CLOSED** | Non si fissa K, si fissa la copertura % della massa-gate; K = conseguenza. Manopola task-INDIPENDENTE. Ranking per MASSA-gate (non frequenza). | src/ (DS4_SPEX_TRACE_ROUTING_WEIGHTS), docs/paper/PAPER.md |
| Session-learning (mask live-calibrata) | **CLOSED** | Osserva W token wide (no mask) -> costruisce mask da massa-gate osservata (top-K/layer) -> congela. Live-calibrato, NON pre-addestrato sul dominio. | src/ session-learning, docs/paper/PAPER.md |

## Findings validati (numeri correnti)

| Claim | Stato | Evidenza / numero corrente | Dove vive |
|---|---|---|---|
| GINOCCHIO scala col task (cold-static) | **CLOSED** | JSON keep-20 (7.8%) = L3 esatto; Python keep-32 (12.5%) = L3 (keep-28 rompe); Frontpage >32 collassa a ogni K cold (L0 loop); Frontpage-HARD (todo-app) full = L1 (soffitto capacita). Random control collassa ovunque -> conta il RANKING di salienza. "La potatura tollerabile e funzione della larghezza del task, non costante." | docs/paper/PAPER.md, trace cold-static |
| SESSION-LEARNING riscatta il cold-collapse | **OPEN (knife-edge freeze-point)** | Cold-static keep-23 = L0 uniforme (loop); il riscatto session-learned resta reale ma NON generale. Replay pod 2026-07-10 (cache1024, prompt compatto): W50 riproduce il regime utile (fase2 14.60 t/s, repeat=0, `</html>`/form/script presenti, 1 restart), W130 FALLISCE (fase2 16.24 t/s ma repeat=1, no `</html>`, loop document.addEventListener). Fail confounded dal punto di taglio del freeze: il re-prefill di fase-2 tagliato a meta CSS induce document-restart (nota J44) -> la vecchia scala W (W=50/130 L3, W=80/110 L2, W=150 L1) e knife-edge sul freeze-point, non monotona in W. n=1 greedy. Si richiude solo con W-sweep a freeze su boundary sicuri (T4, piano 2026-07-10). | docs/paper/PAPER.md, runs/ds4/20260710_pod_cache1024_warmup_replay/README.md, docs/EXPERIMENTS_LEDGER.md (nota J44) |
| ADAPTIVE-K COVERAGE 90% universale | **CLOSED** | Copertura 90% -> JSON L3, Python L3, Frontpage L2. K-auto NON diverge tra task (~36-39 a cov90); diverge il LIVELLO raggiunto. Trace locale frontpage: cov80 K~30, cov85 ~38, cov90 ~49, cov95 ~67. | docs/paper/PAPER.md, trace coverage |
| TIMING SEGMENTATO (TTFT / 1-64 / 65-256 / 257+) | **CLOSED** | Cache=400 (~12GB): full 2.05->0.98->0.76 t/s (DEGRADA); keep-8 12.95->23.55->25.82 (ACCELERA, entra in cache); keep-32 ~3.4-4.3 STUCK (working-set 1280 > 400). Velocita vera nel segmento 257+, decisa dal FIT in cache. TTFT ~14-18s, prefill-bound. | docs/paper/PAPER.md, timing segmentato |
| COMPLETAMENTO (frontpage L3 ~500 tok) | **CLOSED (caveat W130)** | session W=50 L3 picco 13.6 t/s comp ~65s (2.5x full); session W=130 L3 comp ~81s; adaptive cov-90 (K~39) L2 picco 6.7 comp ~99s; full L3 picco 3.4 comp ~164s. Warmup corto (>=50) vince sul completamento; coverage costa ~1.5x il tempo di K23-fissa. Numeri pod cache-1024/RAM-calda; su 3060 piu bassi. Caveat replay 2026-07-10: W50 riconferma il regime (fase2 14.60 t/s, repeat=0); W130 NON replica pulito (repeat=1) -> il punto W=130 eredita il knife-edge freeze-point della riga SESSION-LEARNING. | docs/paper/PAPER.md, completamento, runs/ds4/20260710_pod_cache1024_warmup_replay/README.md |
| VELOCITA config funzionanti (>=L2) | **CLOSED** | Python cov-80 (K~23) L3 = 9.6 t/s; Python keep-32 L3 = 8.3; JSON keep-20 L3 = 7.2; JSON cov-80 (K~24) L3 = 5.8. | docs/paper/PAPER.md |
| FEEDBACK slope-S1 (segnale di loop) | **OPEN** | S1 = frazione di massa-router sugli esperti POTATI (router calcola tutti 256; bias solo su selezione). Assoluto cronico ~0.75 -> NON distingue; solo lo SLOPE e usabile. Locale: S1 sale +0.058 (0.722->0.781) entrando nel loop (K91: 0.73->0.81 prima del collasso). Controllore (alza coverage su slope-S1) = engine da costruire. Caveat: banda stretta, lead-time da confermare. | docs/paper/PAPER.md, trace S1, K91 |
| CONTROLLO POSITIVO full (T1 pod) | **CLOSED** | Full no_pace (nessuna mask) MAI degenerato: 13/13 repeat=0, CSS sempre coerente. Cyberpunk L0 a 800 E 2000 tok (greedy E sampled) per BUDGET, non per config -- il prompt elicita un CSS cosi verboso che `<body>` arriva solo oltre ~2000 tok; pagina completa L2 a 3498 tok (finish=stop). Prompt compatto (coffee): L1 a 800-length (2/3) e L3 a ~785-819 tok quando chiude (finish=stop). Sampling 0.7/0.95 seed42 NON cambia il fenotipo del cyberpunk. => i test qualita' sul cyberpunk richiedono ~4000 tok (o il prompt compatto); il budget <=800 NON e' un banco di prova valido per NESSUNA config su quel prompt. Corollario: il 2-bit non e' indiziato (produce L2/L3 a budget sufficiente). | runs/ds4/20260710_pod_t1_full_positive_control/README.md |

## Negativi onesti / ritrattazioni

| Claim | Stato | Evidenza / numero corrente | Dove vive |
|---|---|---|---|
| MIXED-PRECISION alza il soffitto | **RETRACTED (lose-lose)** | q2 e q2-q4 entrambi L1 sul task hard -> il soffitto e la TAGLIA del modello, non i bit. q2-q4 (98GB) +19% tempo vs q2 (81GB). ds4 fa gia asimmetrico (routed 2-bit IQ2_XXS/Q2_K, attention AProjQ8 + shared SExpQ8 + output OutQ8 a 8-bit, imatrix, opzione q2-q4-last6). Per alzarlo: full-q4 / >=200GB disco o modello piu grosso. | docs/paper/PAPER.md, run q2 vs q2-q4 |
| PREFETCH SPEX-dense = velocita ovunque | **OPEN (regime-dipendente)** | recall != velocita. Hidden-probe wired (0015, funziona); recall 0.616-0.807 costante-su-cache (batte n-gram 0.33-0.56). t/s reale: RAM-served (pod) 2.5-4.5x PIU LENTO; SSD-bound (gate cold-start) 3.2x PIU VELOCE (0.24->0.77). MISURATO 3060-locale (v3, cold-start drop_caches, N=2): sulla config PRATICA (mask keep-23 + cache 5GB + leva-RAM, baseline 1.27 t/s = solo MILDLY-SSD-bound, working-set gia parzialmente cache-ato) il prefetch RALLENTA ~1.55x (0.82 vs 1.27). Converte recall->velocita SOLO se DEEPLY-SSD-bound (gate full/no-mask baseline 0.24). -> NON aiuta il caso pratico del 3060 (overhead > beneficio quando il working-set e piccolo/cache-ato). 0003 selection-continuity + 0004 markov anche nel binario. | docs/SPEX_spec.md, patch 0015/0003/0004 |
| Asimmetria causale HOT/COLD ("94% vs 69%") | **RETRACTED** | Multiseed N=3 non replica; era mask-inerte + n=1. rep-rate hot ~ cold. | docs/paper/PAPER.md (retraction) |
| DYNAMIC staircase / PACE learn-live | **RETRACTED (refutato)** | Avvelena la cache; il direct-descent vince. Path dinamico OFF. | src/ path dinamico (patch 0014), note ENG-BUG |
| reap/full "near-lossless" (loop) | **OPEN** | 1.009x CI[0.972, 1.035]. Il CI attraversa 1.0 -> NON dire "near-lossless" secco per il loop. (Distinto da F1 statico, vedi NUANCE.) | docs/paper/PAPER.md |
| Contrasto PAIRED rand/reap | **CLOSED** | 1.345x CI[1.270, 1.423], NON sovrapposto a reap/full. Stessa GPU, confound-clean. Contrasto ordinale pulito: conta il RANKING di salienza. | docs/paper/PAPER.md, log paired |
| "frontier / DC-scale unlock" | **RETRACTED come tesi** | Prior-art fino a 1026B (EASY-EP, PreMoE, REAP-Cerebras 2510.13999). Posizione onesta = EDGE / single-stream. | docs/paper/PAPER.md |

## Engineering / 3060

| Claim | Stato | Evidenza / numero corrente | Dove vive |
|---|---|---|---|
| Bug reserve cache esperti | **CLOSED (fix noto)** | DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB default 16 -> capato a VRAM/2 = 6GB -> cache esperti DISABILITATA su 12GB. reserve=1 la riabilita. Trappola (vale anche per antirez: default strozza le 12GB). | note locali 3060, src/ streaming cache |
| 3060 I/O / capacity-bound | **OPEN** | Working-set keep-23 = 920 esperti / 6.07 GiB > cache VRAM 854; + non-expert ~5.4GB -> 12GB pieni. keep-23 warm dipende dalla cache esperti (sweep 2026-07-09): 3.03-3.34 t/s a cache128 (best warm della sweep; su prompt html800 static arriva a 3.39, riga STABILITA), cache258 warm 3.07-3.23, cache64 1.69-2.31 -> il vecchio ~1.6-1.7 e compatibile col regime cache-strozzata (cache64 warm 1.69-1.85 su html), non col tetto del 3060. cache-size zero-sum; mlock non ingaggia su WSL. | note locali 3060, runs/ds4/20260709_local_cache_sweep_k23_RESULTS.md |
| STABILITA lunga 3060: K23 rotate32 (W100 non salva il direct) | **OPEN (L0-L3 misurato; ctx8192 n=1, replica=M1a)** | CORRETTO da retro-grade L0-L3 + controllo positivo T1 (i vecchi verdetti "regge / repeat=0" erano artefatti del proxy): (a) a <=800 tok rotate32 E static K23 sono ENTRAMBI L0 funzionale -- il repeat=0 di rotate32 era un'illusione del proxy (loop CSS header/.grid/.card, corruzioni `#l Lime`/`##0f0`, mai `<body>`), static K23 L0 con repeat=1: stessa fine, diverso proxy. (b) Il grade L0 a <=800 sul cyberpunk NON e' attribuibile alla mask: budget-confound dimostrato da T1 (full no_pace L0 sul cyberpunk a 800 E 2000 tok, greedy e sampled, 13/13 repeat=0 e CSS sempre pulito e mai loop; pagina completa L2 solo a 3498 tok, finish=stop). (c) Cio' che RESTA attribuibile alla mask e' la FIRMA loop/corruzione: il full non loopa MAI (13/13 repeat=0), le mask K23/rotate32 loopano dentro `<style>`. (d) A 2000 tok emerge separazione: W100 rotate32 = L1 (apre con `<div>` ma niente bottone, struttura sporca), W50 rotate32 = L2 (body + card + `<button onclick>`, tronca prima dello `<script>`); a ctx8192 il W50 e' il primo output dell'intero corpus (0/89 prima) a emettere `</html>` ma la pagina resta rotta, il W100 loopa -- n=1, replica = M1a. La metrica giusta a basso budget e' la firma di degenerazione, non il livello L da solo. t/s invariati (static 3.03-3.39 / rotate32 2.61-3.03 a c128/c256; costo rotate 0.36-0.45). | runs/ds4/20260710_retro_grade_l0l3/REPORT.md, runs/ds4/20260710_pod_t1_full_positive_control/README.md, docs/HANDOFF_CODEX.md (M1a), runs/ds4/20260709_requested_breath_rotation_RESULTS.md, runs/ds4/20260710_w100_rotate32_k23_cache256_html2000/, docs/DS4_EXPERIMENT_LEDGER_20260710.md |
| Architettura modello 2-bit | **CLOSED (fatto)** | 158B tot / 13B attivi, 43 layer (3 densi 0-2 + 40 MoE 3-42), 256 esperti/layer, top-6, 6.75 MiB/esperto, 81GB SSD. "40 vs 43" era falso allarme (i densi non hanno esperti). | note architettura, docs/ |
| Greedy NON riproducibile run-to-run | **CLOSED (misurato, con nuance)** | Locale (3060): divergenza a tok~75 a config/prompt identici, temp 0 (rollout indipendenti, NON prefisso+coda: coerente con non-associativita' FP/CUDA che ribalta un argmax quasi-pari e poi cascata). Sul pod (3090) e' prompt-dependent: cyberpunk greedy bit-identico n=3 a 800 e n=2 a 2000, ma coffee greedy diverge a ~char346 (~tok100) con grade L1/L1/L3. Il determinismo dell'output dipende da build/hardware/prompt, non e' proprieta' garantita ne' esclusa di ds4-CUDA. => ogni verdetto da n=1 e' UN rollout; n>=3 obbligatorio anche per la qualita' (non solo per la t/s). | runs/ds4/20260710_w50_rotate32_k23_cache256_html4000/ANALYSIS.md, runs/ds4/20260710_pod_t1_full_positive_control/README.md |

## Posizionamento

| Claim | Stato | Evidenza / numero corrente | Dove vive |
|---|---|---|---|
| Nome "REAP" | **OPEN (collisione)** | Collide con REAP-Cerebras (arXiv 2510.13999). Pruning statico e prior-art (EASY-EP, PreMoE, NAEE). | README, docs/paper/PAPER.md |
| Contributo netto del progetto | **CLOSED** | = loop live-calibrato + eval graduato L0-L3 + adaptive-K coverage + negativi onesti. Costruito su ds4 / DwarfStar (antirez, MIT). | README, docs/paper/PAPER.md |

---

**NUANCE (non confondere con l'asimmetria e col near-lossless del loop, entrambi ritrattati/OPEN):**
Il finding **F1** — pruning statico ~50% esperti su dominio, saliency >> random — resta **CLOSED / valido come prior-art** (30B K50 ppl 5.50 vs full 5.56; EASY-EP / PreMoE / REAP / NAEE). Il "near-lossless" di F1 e legittimo. Il "near-lossless" ritrattato/OPEN qui sopra si riferisce SOLO al loop (reap/full 1.009x, CI attraversa 1.0).

---

**REGOLA anti-regressione: nessun doc del repo puo affermare come CORRENTE un claim marcato qui RETRACTED o SUPERSEDED, ne presentare come CLOSED un claim marcato OPEN. In caso di conflitto, questo file vince. Aggiornare QUI per primo prima di cambiare qualunque altro documento.**
