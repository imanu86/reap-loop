# MoE ECOSYSTEM SURVEY — strategie oltre ds4 (2026-07-11)

**Scopo.** Setacciare l'ecosistema MoE-inference 2025-2026 *fuori* dai fork di
`antirez/ds4` (quello lo copre `docs/FORK_SURVEY_20260710.md`) — engine di serving
(vLLM/SGLang), PR mainline llama.cpp, lineage PowerInfer/KTransformers, e la
ricerca accademica su pruning/caching/prefetch/rewind — per tecniche riusabili
sulle **nostre leve** (`docs/SOTA_ROADMAP.md`): `mask/keep-K` (S2), `cache/
resident-hit` (S4), `prefetch` (S3-S4, delta-prefetch 0021 / SPEX), `tiering-quant`
(S4/E-LAT), `boot-probe` (HW-adaptive, gate P2), `recovery/rewind` (S1-slope 0020 /
rewind 0022 / admission 0026 / stopper), `ctx-lungo + KV`.

**Config di riferimento (REAP-LOOP).** DeepSeek-V4-Flash "DwarfStar", 158B MoE,
256 esperti/layer, top-6, routed 2-bit (IQ2_XXS/Q2_K, attention/shared/output a
8-bit), expert ~6.75 MiB, RTX 3060 12GB, **mask keep-K session-learned** (bias
−1e9 al gate) **+ streaming SSD**.

> **ONESTÀ — leggere prima di citare qualunque numero.** Tutti i numeri di questo
> documento sono **`[CLAIM]` degli autori, NON verificati da noi**: HW, prompt,
> quant e metodo di misura sono i loro, quasi sempre su GPU/APU diverse dal nostro
> 3060 12GB e senza il nostro n=3/ABAB (`SOTA_ROADMAP.md` P1). Nessuno è headline
> finché non lo replichiamo sotto `docs/DS4_RUNNER_PROTOCOL.md`. In conflitto vince
> `docs/CLAIMS_CURRENT.md`. I nostri numeri (S1, curve cold/warm, ecc.) NON portano
> il tag `[CLAIM]` perché misurati sotto il nostro protocollo.

---

## 1 — Tabella tecniche (per leva)

### 1A — `mask/keep-K` (costruzione + sizing della maschera)

| Tecnica | Fonte | Cosa | Numeri `[CLAIM]` | Fit leva |
|---|---|---|---|---|
| **Unified scoring MAN/MSAN** | arXiv 2606.15716 | REAP/MoNE/SEER = casi speciali di `freq × gate × activation`; propone score **gate-free** (Mean/Mean-Sq Activation Norm) task-agnostic | +8.8 pt downstream vs REAP in task-agnostic | Upgrade *a parità di sforzo* del nostro score REAP: cambia solo la formula, ingredienti già calcolati |
| **CoX-MoE — EAS** | arXiv 2605.17889 | keep-K **offline** via clustering embedding + profiling frequenza; preload statico | 1.7-2.4× vs MoE-Lightning, 3.4-7.1× vs FlexGen | Contro-esempio per il paper: mask **offline fissa** vs nostra **session-learned** che segue il drift di dominio; non testato <48GB |
| **PreScope — LLaPor** | arXiv 2509.23638 | predittore **per-gruppo-di-layer** (shallow near-input/output vs deep middle); hot-table offline | hit top-4 94-99%; +141% vs Klotski | Suggerisce keep-K **non uniforme per layer** — i layer near-input/output hanno pattern diversi dai middle |
| **MoBiLE — big/little** | arXiv 2510.12357 | **K dinamico per importanza-token**: set pieno sui token difficili, metà sui facili | 1.60-1.72×, degrado accuratezza trascurabile | K per-token da segnale (entropia/confidenza router) invece di keep-K fisso; streaming SSD = fallback per il set "big" |
| **fMoE / FineMoE** | arXiv 2502.05370 (EuroSys'26) | "expert map" indicizzata su traiettorie storiche + **grafo di successione** (X→quali seguono) | −47% latenza, +36/39% hit | Far evolvere la mask da *top-K-per-frequenza* a *top-K-per-frequenza + grafo di transizione* |
| **Attribution+Coverage pruning** | arXiv 2606.18304 | pruning **channel-level dentro** gli esperti tenuti + quant 4-bit | 5.27× mem su Qwen3-30B a 50/75% pruning | Gli esperti tenuti da K91 (6.75 MiB @2-bit) hanno ancora ridondanza interna comprimibile |
| **REAP su GGUF quantizzato** | llama.cpp PR #20454 | profiler REAP a runtime su modello **già quantizzato** (no BF16); pruning slicing asse-expert preservando i blocchi quant come **byte grezzi** | keep-20% → Nemotron 22GB→4.5GB; profiling ~15GB GPU | Materializzare la nostra keep-K come **file GGUF fisico più piccolo** (hard-prune) senza doppio giro di quant-error |

### 1B — `cache/resident-hit` (residenza + eviction)

| Tecnica | Fonte | Cosa | Numeri `[CLAIM]` | Fit leva |
|---|---|---|---|---|
| **CUDA MoE expert cache** | llama.cpp PR #24524 (RFC #24528) | cache LRU slot-pool in VRAM idle; batched matvec sugli hit mentre CPU calcola i miss; **bail-out auto** se più lento del baseline campionato | top-10% expert = **80-81% hit**, top-30% = 95-96%; +10..57% su 13 modelli; **GTX1080Ti 11GB: REGRESSIONE** | Conferma indipendente della **power-law di riuso** (= la forma della nostra keep-K). Il **bail-out** è un pattern per il boot-probe. ⚠ su GPU piccole può PEGGIORARE → probe, non assumere |
| **Two-tier cache + SLRU** | llama.cpp issue #20757 | 3 tier (VRAM/RAM/SSD); **SLRU** 20% probationary / 80% protected + ammissione al **2° miss** (esperti freddi di prefill non inquinano) | RTX PRO 2000 8GB: cold 1.9-2.5 t/s hit 48-56% → steady 12-14 t/s hit 98-100% | **SLRU separa** gli esperti "da provare" (probationary) dalla mask session-learned **stabile** (protected): un domain-shift non cancella la mask consolidata (fix diretto del knife-edge) |
| **Incremental offload — LFRU** | vLLM RFC #38256 | eviction **LFRU** `score=freq/(clock−last_access+1)` per proteggere gli "hub expert" durante domain-shift; tier disco mmap | 375× riduzione expert-load a 3K ctx; RTX PRO 2000 8GB, 97-100% hit | LFRU pesa recency+freq insieme → upgrade del keep-K puramente freq-based; tier disco mmap ≈ nostro streaming SSD (limiti: `--enforce-eager`, single-GPU) |
| **Expert-centric slot cache** | vLLM #41447 | N slot fisici GPU, mapping slot↔expert O(1) (mirror del KV-block manager) | A10-40GB Gemma-4-26B: prefetch 16→21.7 / 32→33.9 / 72→60.3 t/s (**~lineare**) | Curva di riferimento: allargare "K" residente dà throughput **~lineare** fino a saturazione → sizing keep-K vs VRAM |
| **MoE-Infinity** | arXiv 2401.14361 | activation-tracing expert cache; ~15-20% esperti → ~80% token | 3.1-16.7× latenza/token vs vLLM/Ollama/DeepSpeed; target 24-48GB | Baseline maturo "activation-aware caching" contro cui posizionare la nostra mask session-learned (loro tracciano+sostituiscono continuo) |
| **HybriMoE — impact-driven** | arXiv 2504.05897 | cache/prefetch **impact-driven** (guadagno atteso, non solo recency/freq) | 1.33× prefill, 1.70× decode | Criterio di scoring alternativo per il keep-K (impact ≠ hit-count puro); baseline comune di PreScope/DALI |

### 1C — `prefetch` (nascondere la latenza SSD dietro il compute)

| Tecnica | Fonte | Cosa | Numeri `[CLAIM]` | Fit leva |
|---|---|---|---|---|
| **Residual-based prefetch (DALI)** | arXiv 2602.03495 | corregge l'hidden-state del layer N con un **residuo offline-calibrato** per predire il routing di N+1 prima del forward; assegnazione greedy 0-1 CPU/GPU | greedy 92% dell'ottimo @5% overhead; residual pred 93.3% vs 77.6% HybriMoE; prefill 7.62× vs llama.cpp | Segnale **residual-stream** (non solo logit router) per lanciare la read SSD **in anticipo** sul keep-K corrente; greedy placement = modello cheap VRAM-vs-SSD |
| **Speculating Experts (YALIS)** | arXiv 2603.19289 | predice gli esperti di un layer futuro da rappresentazioni interne **prima** che il router finisca | (numeri non estratti in modo pulito) | Esattamente "lancia la read SSD prima che l'output del router sia definitivo" |
| **Cross-Layer Gate / Fate** | arXiv 2502.12224 | predice il routing da **input del gate di layer adiacenti** + quant per esperti in cache | prefill 4.5× vs on-demand, hit 99%, scala su budget mem | Segnale cross-layer economico + quant integrata: match più vicino a mask+cache+2-bit potenziato |
| **DuoServe-MoE** | arXiv 2509.07379 | MLP a 7 layer (popolarità+affinità+path) predice gli esperti del layer succ. | TTFT 1.78-5.34×; predittore ~0.6 ms / 300 MB VRAM | Predittore **leggerissimo** innestabile a valle della mask; overhead 300MB trascurabile sui 12GB |
| **Pre-Attention Prediction** | arXiv 2511.10676 | 2 funzioni lineari sulla attivazione **pre-attention** dello stesso layer (ranking-preserving loss) | acc 93.03% (DSV2-Lite) / 94.69% (Qwen3-30B) / 97.62% (Phi) | Probe **cheap a freddo**, senza storico → boot-probe per inizializzare la keep-K dal 1° token |
| **MoE-SpeQ** | arXiv 2511.14102 | speculative decoding + prefetch proattivo di esperti **quantizzati**; il ciclo draft-verify nasconde l'IO | 2.34× vs offload SOTA su Phi-MoE | Il "draft in avanti" nasconde la nostra latenza streaming SSD; si accoppia ai 2-bit |
| **ggml async prefetch layer n+1** | llama.cpp PR #21067 | overlap transfer CPU→GPU del layer n+1 durante il compute del layer n (richiede `--no-mmap`) | guadagni a ubatch 512-2048, maggiori a batch piccoli; **incerto a batch=1** | Prefetch al confine CPU/GPU; stratificabile (SSD→RAM n+2, RAM→VRAM n+1). ⚠ batch=1 interattivo = guadagno incerto → probe |

### 1D — `tiering / quant` (fallback economico al mask-miss)

| Tecnica | Fonte | Cosa | Numeri `[CLAIM]` | Fit leva |
|---|---|---|---|---|
| **Expert substitution (SMoE)** | arXiv 2508.18983 | su miss di esperto a **basso score**: **sostituisce** con l'esperto simile già residente (invece di caricarlo/aspettarlo); eviction score-based; **online, no profiling offline** | hit >60%, decode −24..−48% (A6000); **RTX 3080 Ti 12GB** (nostra fascia), α unico iperparametro | ⭐ terza opzione al mask-miss: **approssima** con il residente più simile invece del round-trip SSD; serve tabella similarità (già calcolabile dagli score REAP) |
| **FluxMoE — shadow copies** | arXiv 2604.02715 | disaccoppia *residenza* da *bisogno*; copie **shadow compresse** GPU-resident anche per esperti non tenuti; placement bandwidth-aware | fino 3.0× vs vLLM in regime memory-intensive | Formalizza il nostro mask-GPU + SSD-cold; la shadow compressa abbassa il costo di un mask-miss (fallback low-bit vs SSD pieno) |
| **SpecMD — degrado selettivo** | arXiv 2602.03921 | prefetch speculativo "Least-Stale"; cache **multi-precisione** (fp16..int2) con **degrado selettivo** sui bottom-60° percentile su miss | Least-Stale hit 88-92% a 5% cache; TTFT −10.7..−34.7% (OLMoE) | Su miss del keep-K: copia **molto compressa** (1-bit/pruned) dei freddi in VRAM/RAM come fallback rapido invece dello streaming SSD |
| **Tensor bundling su disco** | AlexChen31337/gemma4-moe-offload | co-loca su disco i tensori co-attivati (gate+down) per **letture contigue** invece che sparse; DRAM window scorrevole; TurboQuant KV | −50..65% letture NVMe (stimato); WIP, target ≥10 t/s | ⭐ **basso costo, IOPS gratis**: se up/gate/down del nostro expert 6.75 MiB non sono già bundled fisicamente, guadagno diretto sullo streaming SSD; TurboQuant KV libera VRAM per ctx |
| **KTransformers — shared-on-GPU + AMX** | github kvcache-ai/ktransformers ; SOSP'25 | shared-expert sempre su GPU, routed offload CPU con kernel AMX; **Expert Deferral** (posticipa esperti sul residual → overlap CPU/GPU) | prefill 4.62-19.74×, decode 1.25-4.09×; Deferral +1.45×, drop ≤0.5% | `--kt-num-gpu-experts` = knob keep-K esplicito 1:1; **Expert Deferral** = il nostro stesso bet (tollerare stantìo sul residual) ma sul *compute* |
| **PowerInfer-2 — 5-stage pipeline** | arXiv 2406.06282 | Predict→GatherIO→GatherCompute→UpdateIO→UpdateCompute; hot/cold neuron residency; planner **una-tantum offline** | miss medio 3.5% (P99 18.9%); decode 25.4× vs llama.cpp (smartphone) | Pipeline con **"Predict" esplicito** riusabile per lo streaming SSD (oggi più reattivo che predittivo); planner-offline = contrappunto alla nostra mask session-learned |
| **ik_llama.cpp — soglia adattiva** | knightli.com / gist | soglia switch prefill/decode = `32 · total_exp / active_exp` invece di 32 fisso (scala con la sparsità) | (solo formula) | Idea cheap: il nostro punto di switch mask-only↔streaming scali con `active/total` (top-6/256 è molto più sparso di top-8/128) |

### 1E — `boot-probe` (calibrazione a freddo) e `ctx-lungo / KV`

| Tecnica | Fonte | Cosa | Numeri `[CLAIM]` | Fit leva |
|---|---|---|---|---|
| **ESS — LRU-Warmup + FlashTrans UVA** | arXiv 2512.10576 | pre-riscalda la cache con le top-2K entry delle ultime 32 finestre di prefill; UVA per transfer a **granularità fine** (656B) | H2D 0.79→37 GB/s; +69% ctx32K, +123% ctx128K | LRU-Warmup = **boot-probe letterale**; UVA rilevante se i nostri transfer SSD→VRAM sono frammentati; guadagno cresce col ctx |
| **pshard — plan discovery + tier runtime** | llama.cpp PR #22691/#22692 | plan offline per bs∈{1,16,512}, runtime sceglie il tier più piccolo che copre il carico | Qwen3.5-397B: 12-13× prefill a ctx lungo. ⚠ "troppo grande per il merge" | Multi-piano precomputato + hot-swap invece di probe singolo; il grosso è **prefill a ctx lungo**. ⚠ warning scope-creep architetturale |
| **OD-MoE — cacheless** | arXiv 2512.03927 | nessuna cache residente, solo predittore quasi-perfetto JIT | pred 99.94%; ~75% velocità full-cache a 1/3 mem; <1GB VRAM | Limite inferiore "solo predizione": modalità fallback per la **finestra a freddo** prima che la mask di sessione converga |
| **Apple TurboQuant KV** | (come 1D bundling) | KV compresso PolarQuant + JL 1-bit; approx `--cache-type-k/v q4_0 --flash-attn` | (parte del progetto WIP) | Libera VRAM per ctx più lungo a budget 12GB fisso senza toccare il budget esperti/mask |

### 1F — `recovery / rewind` (deriva residua + qualità in generazione lunga)

| Tecnica | Fonte | Cosa | Numeri `[CLAIM]` | Fit leva |
|---|---|---|---|---|
| **ExpertFlow — real-time correction** | arXiv 2410.17954 | RPP (T5) predice il **piano di routing completo pre-layer-0**; PLEC rialloca gli slot per-request; modulo di **correzione runtime** sui misprediction | −93.72% picco mem GPU; 10× throughput; hit 91.96%; pred fino 95% | Il "Real-time Correction" è il nostro lever recovery/rewind già formalizzato; RPP full-plan = versione più forte della mask statica |
| **LPSR — phase-shift rollback** | arXiv 2604.18567 | monitora il **residual stream**, rileva phase-shift (cosine+entropia), rollback KV + inietta steering vector; layer-detect ≠ layer-correct (14 vs 16) | MATH-500 8B: 44.0% vs 28.8% AR; batte Best-of-16 con 5.4× meno token | Pattern più vicino al nostro **rewind S1-guided**: trigger su segnale interno + rewind KV + correzione. Il "detect≠correct" è un dato di design riusabile |
| **LoopGuard — KV intervention** | arXiv 2604.10044 | rileva collasso attenzione (entropia bassa), pota le KV in loop e ricostruisce attorno agli **anchor token** | overhead ~2-5%, recovery ~95% ppl `[CLAIM da fetch]` | Rewind **fine-grained** a livello KV (vs nostro rewind a checkpoint S1 a monte); layer aggiuntivo pre-rewind |
| **Recurrent state rollback** | llama.cpp PR #25004 | snapshot per-sequenza a **shift-register** con `rs_valid_depth` (profondità rollback bounded); un solo `ggml_cpy` | Qwen3.6 spec-decode 1.72× (solo sotto concorrenza) | Forma-dati esatta per un **rewind O(1)** della mask/keep-K: tieni N snapshot, torna indietro invece di rigenerare la mask |
| **GhostServe — shadow KV checkpoint** | arXiv 2605.00831 | erasure-coding dei parity shard in host memory **in background** (non blocca il forward) | checkpoint 2.7×, recovery 2.1× più veloci; MLSys'26 oral | Rende il checkpoint del rewind quasi gratuito vs snapshot completo periodico |
| **Concordia — delta-checkpoint** | arXiv 2606.23521 | handler JIT di **delta-checkpoint** per regione (KV-block/adapter-page) | (solo architettura) | Salva/ripristina solo i **delta** di KV-cache dal checkpoint S1 invece dell'intero stato |
| **SGLang — EEP redundant + EPLB** | lmsys blog 2026-03 / sglang docs | esperti **ridondanti** + repair-path su rank falliti; EPLB ricalcola il placement ogni ~1000 richieste | interruzione <10s; TBO fino 2× | Margine di esperti ridondanti oltre il keep-K minimo per assorbire shock di dominio; **ricalibrazione periodica** vs update solo incrementale |
| **Tarragon — per-expert recovery** | arXiv 2601.01310 | recovery a **grana di singolo expert** invece di rollback del modello intero | `[CLAIM]` non estratti | Se la mask individua un expert "corrotto/degenerato" in sessione: recovery granulare invece di rewind globale |
| **Edit-1-neuron anti-loop** | arXiv 2606.13705 | editing di **un'attivazione** a inference-time per rompere i loop; testato su Gemma-4 **MoE (A4B)** | `[CLAIM]` non estratti; modelli "noloop" pubblicati | Molto più cheap del rewind SE generalizza; da leggere: il neurone è nel **router** o nel backbone? compat 2-bit? |
| **DRY + XTC sampler** | llama.cpp PR #10803 / #9742 | DRY penalizza ripetizioni **multi-token**; XTC esclude i top-choice per rompere loop **semantici** (non letterali) | (production-tested, no numeri) | Prima linea cheap a livello logit **prima** di un rewind costoso. ⚠ **il nostro dato lo depriorizza** (91ea3bd: sampling-under-mask n=3 → fenotipo invariato) |

### 1G — Baseline esterni (righe di confronto oneste, non tecniche)

| Baseline | Fonte | Numeri `[CLAIM]` | Uso |
|---|---|---|---|
| **llama.cpp `--n-cpu-moe`** | knightli.com | RTX 3060 12GB Qwen3.6-35B-A3B Q4_K_M, ncmoe=32, ctx 64K: **33-36 t/s** | Baseline replicabile **sulla nostra scheda** per la tabella velocità del paper (invece di soli confronti vs full-dense) |
| **PR #25294 — stream expert da disco** | llama.cpp | streaming 5.3× prefill / 2.4× decode vs mmap+CPU-MoE; **output bit-exact**; O_DIRECT; wave-partitioned prefill | Equivalente mainline del nostro streaming SSD: riferimento numerico + 2 trucchi (O_DIRECT, wave-partition) |
| **PR #23440 — MoE offload Metal** | llama.cpp | M3 Pro: 32 slot = −16.6GB per −42% throughput (degrado **ripido** sotto soglia) | Punto-dati per dimensionare il tiering keep-K: sotto un certo #slot il degrado non è lineare |
| **GPT-OSS-120B su RTX 5090** | millstoneai / x.com | 47 t/s decode (RAM-bound) / 473 t/s prefill; gap ~10× dominato dal transfer pesi | Riferimento consumer "attention-GPU + expert-RAM"; contestualizza le nostre claim PRELIMINARY |

---

## 2 — Cosa fanno gli altri che noi NO

Temi ricorrenti presenti nell'ecosistema e **assenti** dal nostro stack attuale:

1. **Predittore appreso del routing.** RPP T5 (ExpertFlow), MLP-7layer (DuoServe),
   LLaPor per-gruppo-layer (PreScope), 2 funzioni lineari pre-attention (2511.10676),
   cross-layer gate (2502.12224). **Noi** usiamo **statistiche di frequenza** di
   sessione, nessun predittore addestrato. (Caveat nostro: E-CAL ha già dimostrato
   che la *coverage* NON separa sopravvivenza/collasso → un predittore va giustificato
   contro il **collapse-rate**, non contro l'hit-rate.)
2. **Grafo di transizione/successione tra esperti.** fMoE indicizza "se si attiva X,
   quali seguono". La nostra mask è **top-K per frequenza**, senza catena di Markov.
3. **Sostituzione dell'esperto mancante** con il residente più simile (SMoE) o con una
   **shadow compressa** GPU-side (FluxMoE, SpecMD bottom-60°). **Noi** su mask-miss
   facciamo (presumibilmente) round-trip SSD pieno o rewind — **nessun fallback low-bit**.
4. **Policy di eviction sofisticate a runtime**: LFRU (recency+freq, vLLM), SLRU
   (probationary/protected, llama.cpp), impact-driven (HybriMoE). La nostra keep-K è
   **frozen per sessione** (o admission CUSUM 0026), **senza** una eviction che pesi
   recency+freq+impact insieme.
5. **Tiering multi-precisione con degrado-su-miss** (SpecMD, FluxMoE, DyMoE): copia
   intermedia 1-bit tra 2-bit e SSD. **Noi**: 2-bit uniforme → SSD, **nessun tier
   intermedio**.
6. **keep-K non uniforme per layer** (PreScope): near-input/output ≠ middle. La nostra
   keep-K è **uniforme per layer**.
7. **K dinamico per-token per importanza** (MoBiLE big/little). La nostra K è **fissa**
   (session-learned costante). *(Nota: DynMoE/AdaMoE già lo rivendicano — `REAP_LOOP_NOVELTY.md`
   dice di NON rivendicarlo; resta però una leva di velocità non sfruttata.)*
8. **Speculative/draft-forward per nascondere l'IO** (MoE-SpeQ, Speculating Experts).
   **Noi** abbiamo l'infra MTP (`spec_frontier_snapshot`) ma la usiamo per la
   generazione, **non** per nascondere la latenza streaming SSD.
9. **Bundling dei tensori co-attivati su disco** (Apple flash) e **UVA fine-grained**
   (ESS): letture contigue vs sparse. I nostri expert 6.75 MiB potrebbero essere
   frammentati.
10. **Checkpoint KV delta/erasure-coded** (GhostServe/Concordia). Il nostro rewind
    usa lo snapshot-frontier completo (via MTP), **senza** ottimizzazione delta.
11. **Difesa di prima linea a livello sampler** (DRY/XTC). **Noi** rewindiamo; nessuna
    difesa cheap pre-rewind. *(Ma 91ea3bd suggerisce che il sampler non sia la leva.)*

---

## 3 — Le nostre unicità (cosa facciamo che loro NO)

Contrappunto onesto della §2. Ogni sistema sopra apprende un working-set per
**cache/prefetch/offload**; nessuno tocca il **routing**. La nostra linea di
demarcazione (già affilata in `docs/REAP_LOOP_NOVELTY.md`):

- **U1 — Session-learning sul ROUTING, non sul caching.** La bias-mask (−1e9 al gate)
  è un **attuatore sulla selezione del router**, appresa **live per sessione** dai
  primi ~150 token, **non** domain-pretrained (GRIFFIN è il vicino più prossimo, ma
  MoE-Infinity/HOBBIT/AdapMoE toccano solo prefetch/cache). Tutti i planner offline
  della §1 (PowerInfer-2, CoX-MoE EAS, PreScope hot-table) fissano il piano al boot;
  **noi lo aggiorniamo durante la sessione**.
- **U2 — Segnale S1 pre-bias (gate-mass sugli esperti potati).** Il router calcola
  tutti i 256; il bias agisce solo sulla selezione → **S1 = frazione di massa-router
  che ricade sugli esperti potati** è osservabile e **causalmente legata all'AZIONE
  di potatura**. Nessun detector altrui usa questo segnale: LoopGuard=attention/KV,
  LPSR=residual-stream, SpecRA=FFT-testo, ST-MoE=pattern-di-loading. Il nostro è
  l'unico legato all'attuatore. (Onestà: S1 assoluto è cronico ~0.75 → solo lo
  **slope** è usabile, e compra lead **solo nel regime slow-erosion**, `E-DET` OPEN.)
- **U3 — Rewind bit-exact.** `spec_frontier_snapshot`/`restore` (ds4.c:26766/26800)
  + `ds4_session_rewind` (ds4.c:30560) danno un rewind KV **bit-uguale**, e sono
  **già esercitati ogni token** dalla speculazione MTP. Gli altri **approssimano**
  (GhostServe erasure, Concordia delta, LPSR steering-vector); noi riusiamo primitivi
  **testati** per un rewind **esatto** (gate R1 = smoke di bit-equality).
- **U4 — Eval graduato L0-L3.** Scala funzionale (L0=non-parse; L1=apre-ma-rotto;
  L2=difetti-minori; L3=pieno-pulito) che **rivela ciò che ppl/repeat-rate non
  catturano** — ha smascherato l'illusione "repeat=0" di rotate32 (`CLAIMS_CURRENT.md`
  STABILITA). Quasi tutta la letteratura §1 riporta ppl/hit-rate/accuracy; **nessuno**
  grada l'artefatto funzionale. È il nostro strumento di onestà.
- **U5 — Negativi onesti + protocollo n=3 ABAB + portabilità adimensionale (P1-P4).**
  Ritrattiamo (asimmetria HOT/COLD, mixed-precision, scala-W). Le leggi di controllo
  usano **solo invarianti adimensionali**, mai un MB/s cablato. Questo rigore è
  assente nella letteratura CLAIM-heavy.

---

## 4 — Candidate adozioni (ordine valore/effort)

**Principio P4 — identificazione parametri, non enumerazione config.** Ogni test
minimo **misura un parametro / rapporto / legge** (e la sua soglia adimensionale
dove serve, P2), **non** fa uno sweep di configurazioni. Vincoli dai nostri dati:
sampler non è la leva (91ea3bd); coverage non separa (E-CAL); S1-slope guadagna
lead solo in slow-erosion → ogni segnale va testato **per-regime** (slow-erosion vs
aggressive-rotate).

| # | Candidato | Leva | Valore/Effort | Test minimo (parametro, non config) |
|---|---|---|---|---|
| **1** | **Tensor bundling su disco** (co-locare up/gate/down dell'expert contigui nel GGUF) | prefetch/streaming | **ALTO/BASSO** | Parametro = **frazione di read contigue vs random** per expert sul nostro GGUF attuale. Misura una volta: gli expert 6.75 MiB sono già fisicamente bundled? Se la frazione-random è alta → repack offline. 1 numero, nessuno sweep |
| **2** | **Expert substitution su mask-miss** via similarità REAP (SMoE/FluxMoE) | recovery/cache | **ALTO/MEDIO-ALTO** | Parametro = **quality-drop-per-sostituzione** a K costante. Legge: sostituendo il miss bottom-percentile col residente più simile (grafo similarità dagli score REAP che **già abbiamo**), la mediana L0-L3 resta ≥L2? Misura la legge drop-vs-similarità, non quali expert |
| **3** | **SLRU/LFRU sull'admission 0026** (probationary/protected) | cache/mask | **ALTO/MEDIO** | Parametro = **ritenzione del segmento protected sotto burst di dominio**. Legge: dopo N token off-domain, quale frazione della keep-K consolidata sopravvive? (attacca il knife-edge freeze-point). 1 curva ritenzione-vs-N, non sweep di rapporti 20/80 |
| **4** | **Score gate-free MAN/MSAN** vs REAP (2606.15716) | mask/keep-K | **MEDIO-ALTO/BASSO-MEDIO** | Parametro = **quale dei 3 fattori** (freq × gate × activation) separa sopravvivenza/collasso nelle **nostre** trace. Sappiamo che coverage NON separa (E-CAL); testa se la norma gate-free separa a keep-K fisso. 1 delta L0-L3 misurato |
| **5** | **O_DIRECT + wave-partition + cross-check bit-exact** (PR #25294) | streaming/rewind | **MEDIO/BASSO** | Parametro = **pollution della page-cache OS** = Δ byte-letti-da-SSD con/senza O_DIRECT su rollout fisso. Identifica se la page cache aiuta o thrasha (1 rapporto); e valida il nostro claim bit-exact contro il mainline |
| **6** | **Segnale residual-stream per il prefetch** (DALI / Speculating Experts / cross-layer) | prefetch | **MEDIO/MEDIO** | Parametro = **lead-time del prefetch** = quanti token/layer avanti il segnale residual predice il prossimo set di expert con acc ≥X. Misura il punto operativo acc-vs-lead sulle nostre trace, **per-regime** (E-DET vincola: slow-erosion vs rotate) |
| **7** | **Shadow tier a bassa precisione su miss** (SpecMD/FluxMoE/DyMoE) | tiering/quant | **MEDIO/MEDIO-ALTO** | Parametro = **bit-floor del fallback** = a quale bit-width la shadow dei freddi tiene ≥L2 come fallback rapido invece dello streaming SSD. Misura la legge bit→livello-L (si accoppia a #2) |
| **8** | **Materializza keep-K come GGUF hard-pruned** (PR #20454) | mask/tiering | **MEDIO/MEDIO** | Parametro = **quality-delta materializzazione** = L0-L3 del file hard-pruned vs mask-runtime a keep-K identico. Legge: lo slicing raw-byte dell'asse-expert perde qualità? 1 delta, per un checkpoint per-dominio più leggero |
| **9** | **Checkpoint KV delta/erasure per il rewind** (GhostServe/Concordia) | recovery/rewind | **MEDIO/MEDIO** | Parametro = **costo del checkpoint** = tempo/byte del delta-snapshot vs frontier-snapshot completo al gate R1. Legge: il delta rende il rewind quasi-gratis mantenendo la bit-equality? |
| **10** | **Sampler DRY+XTC come prima linea** | recovery | **BASSO/BASSO** | Parametro = **collapse-rate sampler-only vs rewind**. ⚠ **depriorizzato**: 91ea3bd (sampling-under-mask n=3) mostra fenotipo invariato → il test serve solo a **falsificare** definitivamente, non ad adottare |

**Le 5 da fare per prime (valore/effort):** #1 (bundling, gratis) → #2 (substitution,
attacca lo stall SSD dominante riusando gli score REAP) → #3 (SLRU, fixa il
knife-edge) → #4 (MAN/MSAN, mask migliore a parità di sforzo) → #5 (O_DIRECT +
bit-exact, hygiene + valida il claim).

---

## 5 — Provenienza

Scouting 2026-07-11 (5 aree: lineage KTransformers/PowerInfer; PR mainline
llama.cpp; pruning/caching/prefetch accademico; engine vLLM/SGLang; recovery/rewind
qualità). Citazioni **non ri-verificate una-a-una** in questa passata — priorità
alle sole che entrano nelle Candidate adozioni §4 va data prima dell'adozione
(gate P2/P1). Complementare a `docs/FORK_SURVEY_20260710.md` (fork ds4) e
`docs/PRIOR_ART.md`.
