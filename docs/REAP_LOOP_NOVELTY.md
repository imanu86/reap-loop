# REAP-loop v1 — verdetto novelty (per il paper, reviewer target: antirez)

> **[SUPERSEDED in parte — 2026-07-07: l'asimmetria HOT/COLD citata sotto è stata RITRATTATA dalla replica multiseed N=3; per lo stato corrente dei claim vince docs/CLAIMS_CURRENT.md. La linea difendibile resta composizione closed-loop + segnale gate-mass-su-potati, SENZA il finding causale dell'asimmetria.]**

> Novelty check avversariale a 3 fronti (workflow `w8ms2k2n1`). **Verdetto: PARZIALE.**
> ✅ **CITAZIONI VERIFICATE (2026-07-06, WebFetch/WebSearch) — tutte REALI, nessuna confabulata**:
> LoopGuard `2604.10044` ("Breaking Self-Reinforcing Attention Loops via Dynamic KV Cache Intervention"),
> LPSR `2604.18567` ("Latent Phase-Shift Rollback: Inference-Time Error Correction via Residual Stream Monitoring and KV-Cache Steering"),
> ST-MoE `2606.15453` ("A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference" — conferma anche l'axis-flip del ledger),
> SpecRA (OpenReview `xVO4BqmzVD`, "Monitor Degenerative Repetition in LLM Agents using Randomized FFT"). Il do-not-claim REGGE.
>
> ⭐ **SHARPENING dalla verifica (rafforza il tuo contributo)**: ogni prior-art di detection usa un **segnale DIVERSO** —
> LoopGuard = attention/KV, LPSR = residual-stream, SpecRA = testo/vocab (FFT), ST-MoE = pattern di prefetch (loading).
> **Nessuno** usa il tuo segnale — **gate-mass-sugli-esperti-potati, causalmente legato all'AZIONE di potatura** — né lo
> accoppia a un attuatore bias-mask sul routing. Questa è la linea difendibile, ora più netta di prima.

## Verdetto in una riga
**Nessun singolo mattone è nuovo.** L'unica novità che regge a un reviewer severo è la **COMPOSIZIONE closed-loop** + il **finding causale dell'asimmetria** + il **segnale gate-mass-su-potati**. REAP-loop v1 è un'**integrazione originale**, non una primitiva nuova → si rivendica come **sistema + finding causale**, MAI come "abbiamo inventato X".

## La frase esatta rivendicabile (il contributo)
> "Un loop di potatura-per-dominio a inference-time, **training-free e senza update dei pesi**, che opera sugli esperti
> routati di un MoE reale (DeepSeek-V4-Flash, 256 expert/layer) dentro un motore di expert-offload: sul singolo stream
> (a) parte dal modello FULL con contesto sano, (b) apprende una bias-mask dagli esperti effettivamente instradati nei
> primi ~150 token, (c) la stringe a gradini fino a keep-9% via bias −1e9 al gate, (d) la riallarga/rewinda quando la
> **massa di gate che ricade sugli esperti potati** segnala l'onset di degenerazione. Finding causale centrale:
> un'**ASIMMETRIA** misurata — la stessa mask a keep-9% raggiunge coerenza se la sparsità è stretta a gradini mentre lo
> stato di contesto/KV è sano, e collassa all'istante se applicata a freddo; la leva è **la salute del contesto**, non un
> recovery via gradiente."

## Cosa NON rivendicare (o antirez lo smonta) — con riferimento
| Pezzo | Prior-art | Nota |
|---|---|---|
| "impara la mask dalla sessione / primi token" | **GRIFFIN** (arXiv:2404.01365, ICML'24) | fa esattamente questo, training-free |
| "graduale/a gradini evita il collasso" (schedule) | **Zhu & Gupta** (arXiv:1710.01878) | folklore IMP/layer-collapse |
| "test-time/dynamic pruning per prompt" | **μ-MoE** (arXiv:2505.18451) | già rivendicato lì |
| "adaptive top-k / dynamic-k per token" | DynMoE/AdaMoE (2406.13233), Ada-K, LExI (2509.02753) | **tu NON lo rivendichi — tienilo così** |
| "working-set di sessione a runtime" | MoE-Infinity (2401.14361), HOBBIT (2411.01433), AdapMoE | ⭐ ma per **prefetch/cache/offload, NON toccano il routing** → **questa è la tua linea di demarcazione, marcala netta** |
| "rewiring/re-routing on-the-fly" | arXiv:2510.14853 | |
| "detect onset → rewind/KV-edit" | LoopGuard (2604.10044), LPSR (2604.18567) | ✅ reali; ma segnale attention/KV & residual-stream, NON routing |
| "periodicità via autocorrelazione" | SpecRA (OpenReview xVO4BqmzVD) | ✅ reale; ma FFT sul testo, NON Jaccard sul routing |
| "entropia/spettro attivazioni = health-signal" | EigenTrack (arXiv:2509.15735) | gate-entropy = folklore MoE-training |
| "REAP" come naming | arXiv:2510.13999 | è compressione one-shot **statica** = l'opposto; non spacciare continuità di metodo |

## Framing verso antirez + i 3 controlli obbligatori
**Apri DISARMANDO** le sue obiezioni: *"Sappiamo che GRIFFIN impara già la mask dalla sessione, che il gradual-pruning è vecchio come Zhu&Gupta 2017, e che detect-onset→rewind esiste. Non rivendichiamo nessuno di questi. Rivendichiamo tre cose specifiche e verificabili: la composizione closed-loop, l'asimmetria causale, e il segnale gate-mass-su-potati."*

**PER SOPRAVVIVERE servono 3 controlli** (senza, è confounder/cherry-picking e antirez ha ragione):
- **(A) Stessa mask a caldo vs a freddo** → isola "salute del contesto" da "mask migliore". Senza questo la causalità dell'asimmetria è aneddotica.
- **(B) Metrica di qualità DOWNSTREAM** (perplexity o task-acc, NON solo Jaccard/hit-rate) alla soglia keep-9%. Senza, "coerente" è cherry-picking.
- **(C) Lead-time del sensore** → quanti token PRIMA di un repetition-detector testuale (TTR) si accende la gate-mass-su-potati. Se non **precede**, il sensore non è un contributo.

→ Questi 3 controlli sono il work-item che rende il REAP-loop v1 pubblicabile. Sono misurabili sul setup K91 esistente.
