# Inventions Ledger — recupero + audit di novelty (2026-07-06)

> Recupero a tappeto delle invenzioni/intuizioni dell'autore, da 3 fonti (repo-doc, memoria, transcript di sessione),
> con verifica avversariale della novelty. Workflow `w3fkb00fd` (24 agenti). Scopo: **consolidare per non perdere** +
> **sapere cosa si può rivendicare nel paper** (reviewer target: antirez → i meccanismi noti vanno citati, non rivendicati).
> `[R]` = RISCOPERTA (prior-art, NON rivendicare). `[M]` = ORIGINALE-MANU.

## Verdetto in una riga (onesto)
Recuperate ~19 idee distinte. **La maggioranza sono RISCOPERTE** di prior-art applicato al regime estremo 12GB
(SPEX, skip-on-miss, axis-flip, temperature-quant, batched-decode, leva-RAM). Il **contributo difendibile non sono i
meccanismi** ma *il transfer al regime 12GB + i controlli onesti + i negativi misurati*. **Rischio-perdita BASSO**:
ogni idea ha almeno un doc nel repo (internal note + EXPERIMENTS_LEDGER fanno il loro lavoro) — tranne 1 gemma trovata solo
in chat (vedi §Recuperi). Le eventuali altre gemme solo-in-chat sono nei transcript Codex/Claude, fuori da questo sweep.

## ⭐ AGGIORNAMENTO 2026-07-06 (pm) — REAP-loop v1 CHIUSO E2E e FUNZIONA (chat K91)
La leva che qui era `[M] IDEA-SOLA` è ora **misurata end-to-end con esito positivo** → passa a **`[M] IMPLEMENTATA-v1`** ed è **il contributo-headline ORIGINALE-MANU più forte del progetto**. Branch `reap/k91-coding-vram`, 11 commit, patch 0011, giornata $-.
- **Asimmetria salita/discesa (finding nuovo)**: potare un modello *già in delirio* NON recupera (il contesto sporco vince); ma **partire full + stringere a scalini** (keep-64→40→23) tiene il **codice pulito fino a keep-9%** — dove il keep-9% *a freddo* collassa all'istante. È il cuore del loop.
- **Mask imparata dalla sessione** (primi 150 token) copre **62% vs 15%** delle statiche, gira **~2× più veloce** (4.5 vs 2.4 t/s, hit 0.79 vs 0.57).
- **Correzione onesta**: il collasso K91/K96 era soprattutto il **criterio di selezione** (teneva specialisti rari), non solo il numero → tenere i "cavalli da tiro" ad alta massa regge 700 token dove prima crollava in 40.
- **Sensore distress** gratis (dal routing) → all'onset **rewind**, non solo allargare.
- **Ricetta v1**: full (contesto sano + osserva routing) → a ~150 tok mask-dalla-sessione + stringi a caldo (0011) → scala a gradini a keep-9% (veloce, in VRAM, coerente) → sensore acceso → se degenera, rewind.
- **Caveat**: 700 token di CSS coerente ≠ sito completo con JS verificato (il verdetto "produzione" vuole gen completa + render).
- **Novelty check FATTO** (`docs/REAP_LOOP_NOVELTY.md`): **PARZIALE** — la composizione closed-loop + l'asimmetria + il segnale gate-mass-su-potati sono difendibili; i mattoni singoli sono prior-art (GRIFFIN/Zhu-Gupta/μ-MoE). Citazioni verificate reali.

### AGGIORNAMENTO D3–D5 (2026-07-06 sera) — rafforza la novelty E valida la velocità
- **D3** (`69567e1`): il working-set a keep-estremo è **PER-SESSIONE, non di dominio** → mask statiche **morte**, session-mask **4×**. È il **perché causale**: la potatura statica (il prior-art, GRIFFIN incluso) **NON PUÒ** raggiungere keep-9% coerente perché il set giusto cambia per-sessione → il loop session-learned è **necessario, non solo migliore**. Questo *rafforza* la linea difendibile (spiega perché il prior-art statico fallisce dove il loop regge).
- **D4** (`87d2a93`): scalini session-mask CLEAN fino a keep-23 (=keep-9%); la statica **per-massa** ("cavalli da tiro") regge (conferma la correzione criterio-di-selezione).
- **D5** (`05044ac`): **sito COMPLETO** a scalini → chiude il caveat "700 token ≠ sito", a **9.2 t/s / hit 0.91**. ⚠️ **9.2 t/s è sul pod 3080Ti** (stessa arch 12GB, non il 3060); **l'hit-rate 0.91 è la metrica che TRASFERISCE**. Valida l'ipotesi "REAP-loop = leva di velocità più grossa" (da riconfermare sul 3060 con mediana-3).
- **D6-D8 (sera) — due scoperte grosse + il Controllo B FATTO**:
  - **23.6 t/s misurati** su keep-9 DINAMICO (`1b231b6`) — **SOPRA** il tetto PCIe ~11, perché il working-set piccolo è **VRAM-resident** → **conferma che il REAP-loop SCHIVA il PCIe** (non lo accelera). Pod 3080Ti; l'hit-rate trasferisce, l'assoluto no.
  - **"Respiro" (D6b, idea utente)**: ciclo [~400 tok K91 → ~80 tok breath keep-64 + **re-learn della mask**] → il respiro periodico **resetta la deriva** prima che si accumuli (dove D5 scivolava nel comment-loop, D6b restava sano). Il loop non è "scendi e tieni" ma "scendi → tieni → respira+ri-impara". Rafforza la novelty (loop periodico di ri-apprendimento, non mask statica).
  - **CONTROLLO B (ppl downstream) FATTO** (uno dei 3 che avevo mandato): session-mask **ppl 1.22×** vs full, domain-mask **3.5×** → session **near-lossless**, domain **distrugge**. Metrica quantitativa, non più "renderizza pulito".
  - **D8-turbo** (discesa SECCA a keep-23 + respiro = ricetta di produzione candidata, ~11-13 t/s) in misura.
- **PROVE COMPLETE (`79bd20b`) — refinement grosso + controlli**:
  - ⭐ **D3 RAFFINATO + TRANSFER POSITIVO**: il working-set è per **TIPO-DI-OUTPUT**, non strettamente per-sessione. La session-mask **TRASFERISCE** (ppl **1.19×**) a un altro task html/css/js. La domain-mask fallisce (3.5×) perché "coding" mescola python/sql/rust/docker. Gerarchia: **output-type 1.2× ≪ domain-multilang 3.5× ≪ random 9.3×**. → conferma il design "learn-once-reuse for the task" (un output-type coerente riusa la stessa mask) e **affina la novelty** (claim più pulito di "per-sessione"). RANDOM passato (9.34×).
  - **Controllo A** (stessa-mask hot/cold): asimmetria confermata sull'orizzonte LUNGO (2600 tok: hot 94%/9.2 t/s vs cold 69%/2.3 t/s), invisibile a 700 tok. n=1.
  - **Controllo C: NEGATIVO** (onesto, e tocca la nostra idea del "termometro"): il sensore routing-Jaccard **lag-1 NON anticipa** (resta 0.36 mentre la coda entra in comment-loop, periodo ~8 tok). Il detector TESTUALE n-gram resta l'unico pratico. **Lag-k routing DA PROVARE** → decide se il termometro-router (e il controller adattivo) è viabile.
  - Breath fisso previene la degenerazione di coda (~2 t/s costo); adattivo bloccato dal sensore (C). keep-9 dinamico CLEAN 726 tok @ **23.6 t/s**; direct basta (scala non serve). Onset collasso **TASK-dipendente** (snake 46% < sito 94%). Domain-shift (#4) non eseguibile senza harness a 2-prompt; rollback-via-prompt fallisce → serve rewind KV.
- **BATCH SERA-2 (`ab3e038`→`f91b5ee`→`de45172`) — head-to-head, ritrattazione, north-star**:
  - ⭐ **HEAD-TO-HEAD VINTO 3.7×**: il REAP-loop batte la cache reattiva iso-VRAM di **3.7×** (la prova dello scettico, il #5) → l'apparato è giustificato.
  - 🔧 **0011-V3 (bug cold-start + fix)**: la V2 in COLD-START applicava la mask al token ~1, poi i range del model-load la SOPPIANTAVANO → mask **silenziosamente OFF** (85% selezioni sui potati = di fatto spenta). Fix v3: **re-apply ogni 32 token**. LEZIONE: verificare l'aderenza dell'attuatore su OGNI config d'uso, non solo sul primo smoke hot. → relayato al Fable-build (usare V3, non V2).
  - **RITRATTAZIONI oneste**: "statica per-massa regge" era full-travestito; asimmetria caldo/freddo → i bracci cold erano rotti (rerun v3 in corso). SALVI: tutte le **ppl** (attuatore file-based indipendente), la fase-1, gli hot-apply (aderenza 0-0.1%).
  - **lag-k: S2 MORTO a TUTTI i lag (1-16)**, S1 = deriva reale ma LENTA → **il termometro-router per il lead-time è morto**; il **detector TESTUALE resta l'unico sensore** ("il loop testuale è un attrattore nello spazio-testo, invisibile dal top-6 del router"). Il controller-adattivo router-side è quindi improbabile; il fixed-clock/textual è la via.
  - **NORTH-STAR structured-extraction task IN VOLO** (`de45172`): task VERO, scoring **the rubric-scored eval set** — full vs K50 vs K91-reuse vs **two-step (idea utente)**, con **learn-once-reuse** (mask imparata 1 volta da 4 warm, riusata COLD su 10 draft) = il test del transfer 1.19× sul task reale. Risultati ~40-50 min.
- **REPORT FINALE K91 (`1706271`, pod spenti, giornata ~$5.9)** — tutti i test tranne #4-shift:
  - 🎯 **NORTH-STAR structured-extraction task VALIDATO**: 10 item held-out domain set, learn-once-reuse (mask da 4 warm). field-acc **LENIENT: full 0.70 = k50/k91/two-step 0.70** → **K91 aggressivo PARI al full in CONOSCENZA**. STRICT: full 0.583 vs masked 0.08-0.23 → il danno è **SOLO DI FORMATO** (chiavi sfaldate, JSON verboso che sfora -n 200), NON di conoscenza → fix produzione banale: +50 tok o JSON-repair. Con hit 0.89-0.99, **K91-on-the-task ~pari al full operativamente. LEARN-ONCE-REUSE CONFERMATO.**
  - **HEAD-TO-HEAD (#5)**: loop session cache-1024 (6.9GB) **9.22 t/s hit 0.91** vs stock reattiva cache-2048 (13.8GB) 2.51 t/s hit 0.60 → **3.7× con METÀ VRAM**.
  - **Sensore (#1)**: S2-Jaccard morto a tutti i lag; **S1 (0012) = deriva REALE 0.80→0.89 ma LENTA, niente ginocchio → indicatore, non allarme.** Controller pratico = **detector testuale n-gram + pendenza-S1**. Trigger-adattivo netto router-side = morto.
  - **Two-step (#3, idea utente): NON-risultato onesto** — pari alle altre config a n=10 su draft corti.
  - **Asimmetria (ctrl A, mask VIVA)**: la **qualità**-asimmetria è REALE (HOT 94% clean vs COLD onset 70%); l'**efficienza**-asimmetria era artefatto della mask rotta (cold-vivo è veloce, 11.4 t/s).
  - **RITRATTAZIONE finale**: la statica-per-massa VERA (col fix v3) **collassa all'8% hit 0.99** → **solo la session-mask funziona** (rafforza D3: session-learned NECESSARIO, non solo migliore). Regola: aderenza verificata su OGNI run mascherata, mai fidarsi del log.
  - **Per il BUILD**: `ds4_session_rewind` (rewind KV) **serve per il recovery LIVE** (C1: contesto avvelenato vince; rollback-via-prompt riparte da zero 3/3) → il PREVENTIVO non ne ha bisogno, la CURA sì. `0011-v3` + `0012` nel repo, buildate sm_86.

## I 3 pilastri difendibili per il paper (in ordine di solidità)
1. **REAP bias-mask come protocollo di eval surgery-free** — l'unica invenzione IMPLEMENTATA + misurata lossless
   (K50/dom ppl 1.010× vs full, CI95 [0.996,1.025]; random 1.388×; dose-response K50/K67/K70; 0/11280 violazioni V0).
   Rivendicabile: scrivere il −1e9 nel `exp_probs_b` auxiliary-loss-free di DeepSeek (entra solo nella selezione, mai
   nei pesi) → equivalenza esatta col pruning fisico → eval weight-exact su runtime **stock**, senza surgery.
   **Presentare come protocollo di riproducibilità, non come saliency nuova** (la saliency è REAP/Cerebras 2510.13999).
2. **Suite di NEGATIVI onesti** (valore metodologico alto per un reviewer come antirez):
   - **F1**: markov-naive < reactive-LRU quando cache≥top-k → giustifica perché SPEX predice *loading* non *gating*.
   - **Axis-flip** [PENDING]: il prefetcher ID-only cross-layer crolla su DS4-Flash 2-bit (top6 0.0245) → motiva SPEX. Da replicare (oggi singolo prompt degenere).
   - **H4/H5 skip-on-miss**: "devi fetchare, non droppare" (59× ppl); sweet-spot ibrido non-lossless (plateau 1.3×).
   - **Confidence-cascade Gate#2 NO-GO**: pre-registrato, con controllo random, negativo misurato. Raro e credibile.
3. **By-products riusabili**: il **trigger-router black-box** (AUROC 0.984, il pezzo vivo della cascata refutata) e il
   **framing predittivo-vs-reattivo** (SPEX su expert / cascata su memoria).

## Le idee GENUINAMENTE tue (ORIGINALE-MANU) — quelle da coltivare
| Idea | Stato | Perché è tua | Fonte |
|---|---|---|---|
| **[M] REAP-loop dinamico** | IDEA-SOLA | keep-set di saliency-dominio che si allarga/stringe a runtime (sensore n_eff/deriva, attuatore bias-mask). In letteratura il domain-pruning è **sempre statico** → il loop chiuso è tuo. **Se chiuso E2E = candidato headline.** | (internal note); REAP_DS4_design §7.bis; PAPER_DRAFT §5.5 |
| **[M] Attuatore-stabile-in-ID-space** | insight | usare la bias-mask (non la surgery) mantiene gli expert-**ID originali** → Markov/.spex/hotlist/trace SPEX restano validi attraverso i cambi di mask → **REAP-loop componibile con SPEX** senza re-dump. È il collante delle due leve. | verify #23 |
| **[M] Triage-come-segnale-di-prefetch** | IDEA-SOLA | nella extraction pipeline il triage (sempre caldo) carica il working-set dello specialista **durante il proprio decode** (SSD idle) → latenza 15ms/expert nascosta, burst specialista già warm. | (internal note) |
| **[M] Static-is-the-collo** | osservazione (confermata) | l'intuizione che il muro dei 12GB è lo **statico residente**, non la cache-expert. Ora confermata da 3 tracce (backbone/K91/memoria). | sweep + (internal note) |
| **[M] Rete neurale della conversazione** ⭐ NUOVA | IDEA-SOLA (solo-in-chat) | invece della cascata RAG a costo crescente, **addestrare una piccola rete sulla conversazione** che va "dritta al punto" recuperando il contesto giusto senza rung intermedi. Emersa nella chat SPEX-memory Fase 2, **mai consolidata**. | sessione "SPEX-memory Fase 2" (transcript) |

## Ledger completo
| Invenzione | Stato | Origine | Valore | Fonte chiave |
|---|---|---|---|---|
| REAP bias-mask (protocollo surgery-free) | IMPLEMENTATA | `[R]` saliency + sliver-M | MEDIO-ALTO | `scripts/reap_bias_mask_ds4.py`; `runs/reap/2026-07-05_eval_biasmask_v2/` |
| SPEX (prefetch predittivo expert) | SIMULATA | `[R]` (Pre-gated/AdapMoE/HOBBIT/SP-MoE) | MEDIO | `docs/SPEX_spec.md`; `EXPERIMENTS_LEDGER` H1/H2; `patches/ds4/ds4_spex_predict.c` |
| REAP-loop dinamico | IDEA-SOLA | **[M]** | MEDIO(-ALTO se chiuso) | `REAP_DS4_design §7.bis` |
| Confidence-cascade memory | **REFUTED** (Gate#2 NO-GO) | `[R]` sistema; rung-0 sliver | MEDIO | `experiments/cascade_memory/FINDINGS.md` |
| Axis-flip finding | BOUNDED [PENDING] | `[R]` meccanismi, misura tua | MEDIO | `runs/ds4_routing_trace_smoke/`; `PAPER_DRAFT §4.1` |
| Skip-on-miss a perdita limitata | REFUTED (puro) | `[R]` (AdapMoE/HOBBIT/BuddyMoE) | BASSO | `src/msc/residency/miss_modes.py`; `EXPERIMENTS_LEDGER:289` |
| Leva-RAM (NO_DIRECT_IO+KEEP_PAGES) | BOUNDED | `[R]` (page-cache) | MEDIO | `project_ds4_realtest_inflight.md` |
| Batched decode (→≥10 t/s aggregati) | IDEA-SOLA | `[R]` (gap motore ds4) | MEDIO | (internal note) |
| Sinergia union-load × spec-dec (−49% IO) | SIMULATA | MISTA | MEDIO | sweep; guardia `ds4.c` streaming |
| Decomposizione-impatto (fetch alto-impatto / drop coda) | SIMULATA | `[R]` principio | MEDIO | `EXPERIMENTS_LEDGER` double-loop |
| Temperature-per-expert quant | IDEA-SOLA | `[R]` (mixed-precision MoE) | BASSO | doc repo |
| Prune-per-sottrazione (r=freq_unwanted/wanted) | IDEA-SOLA | `[R]` | BASSO | `SUBTRACTION_PRUNING_analysis` |
| Precisione per-sessione (sub-IQ2 del complemento) | IDEA-SOLA | `[R]` | BASSO | sweep |
| FT-router per concentrazione/predicibilità | REFUTED (concentrazione) | `[R]` | BASSO | verify #22 (FT disperde n_eff 49.9→57.1) |
| Ri-idratazione sorted-parallel | IDEA-SOLA | `[R]` | MEDIO | (internal note) |
| Triage-come-segnale-di-prefetch | IDEA-SOLA | **[M]** | MEDIO | (internal note) |
| Attuatore-stabile-in-ID-space | insight | **[M]** | MEDIO | verify #23 |
| Static-is-the-collo | osservazione | **[M]** | MEDIO | (internal note) |
| Rete neurale della conversazione | IDEA-SOLA (solo-chat) | **[M]** | da valutare | transcript Fase 2 |

## Riscoperte — NON rivendicare nel paper (con il riferimento)
1. **SPEX (meccanismo)** = Pre-gated MoE (2308.12066), AdapMoE/HOBBIT/Mixtral-offloading (gate L+1 sull'hidden di L), SP-MoE (2510.10302). Lo smentisce il tuo stesso `PRIOR_ART.md` (F4/F5). Difendibile: la calibrazione STS cross-layer + confidence-head nel regime cache≪N di dominio stretto, e il negative F1.
2. **REAP saliency** = REAP (2510.13999, Cerebras); mascherare un gate a −inf è il primitivo standard top-k. Sliver: l'equivalenza esatta via `exp_probs_b` come *protocollo di eval*.
3. **Confidence-cascade (sistema)** = MemGPT/Letta + FLARE/Self-RAG + StreamingLLM/H2O + RECOMP/Infini + cascade-ranking. Già refutata da te (Gate#2). Sopravvive il trigger-router (AUROC 0.984).
4. **Skip-on-miss** = AdapMoE (2408.10284), HOBBIT (2411.01433), BuddyMoE (2511.10054). Smentito in casa (H4 59×).
5. **Axis-flip (meccanismi)** = Consecutive Tokens Pattern (2606.15453) + Pre-gated/SiDA/ExpertFlow. Originale è solo l'osservazione empirica dell'**inversione** dei due assi su target 2-bit vs proxy Qwen (da rafforzare: oggi 1 prompt degenere).
6. **Temperature-quant** = mixed-precision MoE noto.

## Recuperi + prossimo passo
- **Gemma solo-in-chat recuperata**: "Rete neurale della conversazione" (§sopra) — mai in un doc, ora qui.
- **Dove cercare altre gemme non consolidate** (fuori da questo sweep): i transcript Codex/Claude (internal session note).
  Uno sweep dedicato lì è il recupero prioritario residuo.
- **Da confermare prima di citarlo nel paper**: l'identificazione "DSpark = deepseek-ai/DeepSpec" è dichiarata ma **non verificata**.
