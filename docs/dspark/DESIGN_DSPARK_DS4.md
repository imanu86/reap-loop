# DESIGN — DSpark su ds4: confidence + STS + scheduler sopra l'MTP nativo

> Track DSpark, step 3 del brief. Presuppone `docs/dspark/RECON_MTP_DS4.md` (mappa
> file:riga @ `80ebbc3`, tutte le citazioni qui sotto sono allo stesso pin).
> Paper: `docs/references/DSpark_paper.txt`. Stato: DESIGN — nessun codice ancora.

## 0. Tesi (aggiornata dopo la correzione del brief, commit `563d7d3`)

**Ciò che ds4 ha oggi con `--mtp` È il baseline MTP-1 del paper** (quello battuto del
60-85%), non DSpark. DSpark vero = (1) drafter semi-AR ADDESTRATO (backbone parallelo +
testa sequenziale, checkpoint open-sourced + repo DeepSpec) e (2) verifica
confidence-scheduled. Questo design implementa la **Strada A** del brief: tenere l'MTP
nativo come drafter (autoregressivo ricorsivo: T_draft cresce linearmente con la
profondità — il contrario del backbone parallelo del paper, e il costo va modellato così
nella cost-table) e costruirci sopra la componente (2): confidence per-posizione
calibrata via STS + scheduler hardware-aware. La **Strada B** (portare i checkpoint
DSpark in formato ds4/GGUF) resta per dopo, se la A non basta.
Il *twist paper-worthy* nostro: nel regime streaming-MoE il costo di verifica si
ammortizza col blocco (−49% IO expert a k=8, misurato) → il termine expert-IO entra nel
profilo di throughput dello scheduler e sposta la lunghezza ottima del draft più in alto
che nel paper denso (§2.2).

## 1. Gap-analysis puntuale

| Meccanismo DSpark (paper) | Cosa fa ds4 oggi | Gap |
|---|---|---|
| Confidence head: c_k = σ(w·[h_k; W1[x_{k-1}]]) per posizione, stima P(sopravvivenza \| prefisso accettato) (Eq. 7) | Margine logit top1−top2 dell'ULTIMO draft, solo per draft_n==2, non calibrato (`ds4.c:27295-27341`) | Nessun segnale per-posizione; nessuna semantica probabilistica; inutilizzabile per profondità >2 |
| STS: calibrazione sequenziale dei prodotti cumulati Π c_i, grid-search 1D per posizione, min-ECE (§3.2.1) | Niente | Il margine grezzo non è una probabilità; senza calibrazione non si può stimare τ atteso |
| Scheduler hardware-aware (Alg. 1): massimizza Θ = τ·SPS(B) con curva di costo profilata | Profondità FISSA `--mtp-draft` (default 1, "production depth is two", `ds4.c:27441`); margin gate on/off | Nessuna nozione di costo del passo di verifica; nessun adattamento per-token/dominio |
| Verifica batch con blocco lungo | C'è (`verify_suffix_tops`, 16 righe max via `spec_logits`, `ds4.c:11169`) | Solo greedy match; e in streaming è VIETATA (`ds4.c:25685`) |
| Rejection sampling lossless a T>0 | Solo greedy (`ds4_cli.c:483`) | Fuori scope per ora: il regime 3060 usa greedy |
| Drafter semi-AR addestrato (backbone parallelo + testa sequenziale, T_draft ~ O(1) nel blocco) | MTP-1 ricorsivo autoregressivo: T_draft ∝ profondità | = Strada B (checkpoint DSpark→GGUF, repo DeepSpec); NON in questo design. La Strada A assorbe il costo lineare del draft nella cost-table dello scheduler |

## 2. Adattamento dei tre meccanismi alla realtà ds4

### 2.1 Confidence per-posizione senza training (DSpark-lite)
Il paper allena una testa lineare sull'hidden del backbone. Non abbiamo l'infra di
training del drafter, ma abbiamo un segnale per-posizione già oggi: i logits MTP di ogni
passo di draft (`metal_graph_eval_mtp_draft_from_hc` scrive `s->mtp_logits` quando
richiesto, `ds4.c:27271`). Definiamo:

    c_k = σ( (m_k − b_k) / T_k )      con  m_k = logit_top1 − logit_top2 al passo k

e calibriamo (b_k, T_k) per posizione con la STESSA procedura STS del paper: grid-search
1D sequenziale sinistra→destra che minimizza l'ECE del prodotto cumulato a_j = Π_{i≤j} c_i
contro l'acceptance empirica. Il dataset di calibrazione è GRATIS: ogni run con
`DS4_MTP_CONF_LOG=1` emette (margine, drafted, committed) per ciclo
(`ds4.c:27288-27292, 27485-27494`) — il run pod di oggi produce già le prime coppie.
La macchina di fit esiste: `src/msc/spex/spex_loop.py` implementa la logica STS
(dominio expert-prefetch, stessa matematica); si estrae in un modulo `dspark_sts.py`.

Upgrade path (fase 2, opzionale): sostituire m_k con la prob. softmax del token draftato
(più informativa del margine) — costo identico, stessa calibrazione. Un vero head
allenato resta possibile più avanti via DeepSpec (il paper open-sourcia i checkpoint
DSpark per V4-Flash: verificare se il GGUF MTP pubblicato da antirez potrà incorporarli).

### 2.2 Scheduler Alg. 1 ridotto a R=1 (single-request)
Il 3060 serve UNA richiesta: l'Alg. 1 degenera in una scelta scalare della lunghezza di
verifica ℓ ∈ {0..γ}:

    ℓ* = argmax_ℓ  (1 + Σ_{j≤ℓ} a_j) / T_step(ℓ)

dove T_step(ℓ) = T_draft(ℓ) + T_verify(ℓ) è la curva di costo PROFILATA, l'equivalente
della cost table SPS(B) del paper (§3.2.2, "profiled once during engine initialization").
Nel regime streaming T_verify(ℓ) è dominato dall'IO expert e NON è lineare in ℓ: i byte
da caricare crescono come gli expert unici del blocco (misurato: 10.6/16.4/24.3 per
k=2/4/8 → `runs/ds4_routing_trace_smoke/`), cioè sublineare. È esattamente ciò che rende
il blocco lungo conveniente in streaming anche con acceptance moderata — il twist nostro
rispetto al paper, dove la curva di costo modella la capacità batch, non l'IO.
Implementazione: early-stop del draft loop quando il guadagno marginale atteso
a_{ℓ+1}·Δ(1/T) diventa negativo — identico in spirito all'early-stopping causale di
Alg. 1 riga 13 (il margine m_k al passo k è noto PRIMA di draftare k+1: nessuna
violazione non-anticipating).

### 2.3 Punti d'innesto (file:riga @ 80ebbc3)
1. **Draft loop** `ds4.c:27264-27287`: per ogni passo k leggere m_k (top2 dei logits MTP;
   oggi `logits_top2` esiste già, `ds4.c:27289`) → c_k calibrata → a_k cumulata;
   early-stop del draft quando lo scheduler dice basta. Il cap `draft_cap`
   (`ds4.c:27209-27214`) diventa il γ massimo, non la profondità effettiva.
2. **Margin gate** `ds4.c:27295-27341`: sostituito dal criterio dello scheduler (il gate
   attuale è il caso particolare ℓ*∈{0,1} con soglia fissa). Compat: `--mtp-margin`
   resta come fallback quando la calibrazione non è caricata.
3. **Parametri STS**: file binario/testo piccolo caricato a init engine, pattern identico
   al loader `.spex` di `patches/ds4/0004-spex-markov-loader-prefetch.patch`; env
   `DS4_DSPARK_STS_FILE` (b_k, T_k per k=1..16 + versione + hash modello).
4. **Cost table**: profilata a init o al primo N cicli (T_draft per passo, T_verify(ℓ)
   per ℓ=1..γ), salvata/caricata via env; in streaming rifittata quando cambia il regime
   cache (la curva dipende dal hit-rate expert-cache).
5. **Sblocco streaming (il 49%)** — due modifiche upstream-shaped:
   a. rimuovere la guardia `ds4.c:25685` dietro un flag esplicito (es.
      `DS4_MTP_STREAMING=1` finché sperimentale);
   b. in `metal_graph_verify_suffix_tops` (`ds4.c:21107-21200`), quando
      `g->ssd_streaming`: avvolgere il loop layer (21146-21153) con la stessa
      orchestrazione del prefill split_commands (`ds4.c:20540-20704`):
      `metal_graph_stream_map_layer[_decode]` + prepare/readahead per layer. A quel
      punto `encode_layer_batch` → `encode_layer_ffn_batch` → union-load batch
      (`ds4.c:18938` → `ds4.c:14336` → `ds4_cuda.cu:3176`) scatta DA SOLA: il gating
      `metal_graph_decode_cuda_selected_slots_expected` (`ds4.c:13774`) richiede
      esattamente `ssd_streaming && !quality && n_expert_used==6 && n_expert>=128` e
      quant Q4_K/IQ2 — tutte vere nel nostro regime 2-bit Flash.
   c. il modello MTP (3.8GB, mmap separato — `model_open` dedicata a `ds4.c:25689`) NON
      passa dallo stream mapper: zero-copy dalla page-cache host; 3.8GB stanno nei 28GB
      del 3060. Da verificare a runtime: pressione page-cache vs cache expert.
6. **Rollback/partial accept in streaming**: `spec_frontier_snapshot/restore` e
   prefix-1 capture (`ds4.c:13166`, `ds4.c:24129`) toccano lo stato del compressor —
   da smoke-testare sotto streaming prima di qualsiasi benchmark (rischio principale
   della fase C; se emergono problemi, prima iterazione = full-accept-or-restart).

## 3. Fasi di lavoro (ognuna con misura committata)

- **Fase A — misura (in corso)**: acceptance MTP su pod 3090 stock
  (`runs/dspark/20260705_mtp_acceptance_pod3090/`): probe per-token, drafted/committed a
  profondità 2 e 4, margini per la calibrazione. Trasferisce: acceptance. Non trasferisce: t/s.
- **Fase B — DSpark-lite non-streaming**: STS fit offline dai log A (+ eventuale corpus
  più largo su pod, stesso protocollo); patch scheduler (innesti 1-4); validazione su pod:
  acceptance invariata per costruzione (lossless: la verifica resta greedy exact-match),
  τ/ciclo e t/s vs `--mtp-draft` fisso. Deliverable: patch `0001-dspark-*.patch` in
  `patches/ds4/` + run committati.
- **Fase C — sblocco streaming**: innesti 5-6; misura su 3060 reale (regime DwarfStar,
  `--cuda --ssd-streaming`, coordinandosi per la GPU contesa): byte SSD/token accettato e
  t/s vs baseline streaming senza MTP. Qui vive il claim 49% → obiettivo ~2× sul collo IO.
- **Fase D (= Strada B del brief, solo se la A non basta)**: portare il drafter DSpark
  vero (checkpoint open-sourced V4-Flash + repo DeepSpec) in formato ds4/GGUF: backbone
  parallelo + testa Markov + confidence head addestrata. Progetto a sé (conversione
  formato, kernel per il blocco parallelo); da valutare coi numeri della Fase B/C in mano.

## 4. Interazione col track SPEX (solo interfaccia, non toccare)
Complementari, stessa cache expert: SPEX predice gli expert del token FUTURO (prefetch
speculativo); il verify a blocco rende NOTI (post-router, per layer) gli expert di k
token insieme → union-load deterministica. Composizione naturale in fase C+: durante il
verify del blocco al layer L, SPEX può seminare L+1. Nessuna dipendenza di codice oggi:
l'unico punto di contatto è la expert-cache CUDA, già condivisa.

## 4-bis. PERCHÉ ESISTE LA GUARDIA ds4.c:25685 (indagine 2026-07-05, paletto 1 del go-ahead)

Indagine svolta PRIMA di scrivere codice, come richiesto. Tre fonti concordanti:

1. **Upstream non documenta una race**: il clone locale è shallow (8 commit), ma su GitHub
   l'issue **antirez/ds4#495** (aperta 2026-07-04 da terzi, con la nostra stessa tesi:
   4060Ti disk-bound, 2.2 t/s gen vs 28.7 t/s prefill → stima 4-6 t/s con MTP) chiede
   esattamente "cosa blocca --mtp con --ssd-streaming?" e **non ha risposta del
   maintainer**. Il testo del messaggio d'errore dice "yet": non-implementato, non vietato
   per principio.
2. **Il blocco concreto è identificato dalla PR upstream #497** (aperta, one-liner):
   sotto streaming il verifier MTP emette encode batch **single-position** (es. re-verify
   del prefisso con `commit_drafts==1`, ds4.c:27630); l'early-return `n_tokens <= 1` in
   `metal_graph_cuda_stream_prefill_batch_selected_load` (ds4.c:14345) salta la
   selected-load → il MoE ricade sui range expert RESIDENTI → "CUDA model arena alloc
   failed for moe_gate (1792.00 MiB chunk)" → "MTP verifier failed". Fix della PR:
   `n_tokens <= 1` → `n_tokens == 0` (con commento esplicativo). Diff verificato da noi.
3. **Il nostro probe streaming (patch misura) ha già dimostrato** che drafting MTP,
   registrazione della mappa MTP (3.55GiB full anche sotto streaming) e decode streaming
   convivono: i gap sono SOLO nel path di verify batch.

**Conclusione (paletto 1)**: la guardia protegge un percorso non finito, non una race
documentata. L'assunto di residenza rotto è quello del punto 2. Restano DA VERIFICARE in
smoke (non escludibili dal solo sorgente): (a) mapping dei pesi densi/attn nel batch
verify sotto streaming — ipotesi: coperti dal device cache statico (3.94GiB q8) come nel
prefill batch streaming, quindi nessun map mancante; (b) snapshot/restore della frontiera
compressor su partial-accept sotto streaming; (c) protocollo evento upload
(`DS4_SELECTED_UPLOAD_EVENT`): la selected-load compatta del verifier è sincrona
(end/begin commands attorno, ds4.c:14369-14396) quindi safe di default, ma con l'evento
attivo va rispettato `ds4_gpu_wait_selected_upload()` prima dei kernel MoE consumatori.

**Piano innesto rivisto (compone con upstream, non duplica)**:
- serie patch dal **0008** (0006-0007 riservate al track SPEX; il vecchio probe-patch
  rinumerato 0008): 0008 = probe di misura (già usato); **0009** = one-liner della PR
  #497 (accreditata nel commento) + rilascio guardia dietro `DS4_MTP_STREAMING=1`
  (default OFF) + wait-event nei punti di consumo se l'evento è attivo; 0010+ = eventuali
  fix emersi dallo smoke (mapping, frontiere).
- branch `dspark/unlock-streaming` da `bee9eb3` in un **git worktree separato**
  (`/root/ds4-dspark`) per non toccare il tree condiviso né i binari dello sweep SPEX.
- validazione (paletto 3): niente diff dei token (greedy CUDA non bit-riproducibile,
  finding REAP); **`ds4-eval` appaiato** ON vs OFF (harness nativo, supporta
  --ssd-streaming) su set fisso di domande + coerenza distribuzioni acceptance/τ.
  PPL ds4-nativa non esiste nel repo (quella REAP è pipeline HF/torch): se serve PPL
  vera, da concordare col track SPEX-main.
- smoke: 3060 SOLO dopo fine sweep (~20:00-20:30) o pod 3090 $-/h (il pod A è stato
  terminato a baseline completa come da playbook: se serve prima, se ne rideploya uno
  equivalente).

## 4-ter. PAESAGGIO UPSTREAM (aggiornato 2026-07-05 sera — riposizionamento fase B)

Upstream si è mosso OGGI su tre fronti (segnalazione SPEX-main, PR lette):
- **PR #480 (audreyt)**: runtime DSpark REALE su Metal + converter checkpoint + partial
  commits via prefix-checkpoint. **PR #482 (machiabeli)**: B2 rejection sampling temp>0
  (bug noto: draft via argmax ⇒ non-lossless; fix di audreyt in review) + **adaptive
  block sizing `DS4_DSPARK_ADAPTIVE=1`**: euristica REATTIVA (accepted/drafted del ciclo
  precedente in `ds4_session`, block 2→5 dopo full-commit, rollback su partial). Niente
  confidence head, niente calibrazione, niente modello di costo. **PR #502** le riconcilia.
  Nota dolori upstream documentati (lobanov §7): divergenza argmax batch-vs-decode nel
  verifier (fp order — fenomeno già noto a noi da ds4.c:27342 e dal finding REAP),
  replay KV dominante (~75ms/ciclo anche su full-accept nel loro runtime).
- **PR #504 (iCreil)**: per-size-class expert caches + warm attraverso i prefill in
  ds4_cuda.cu → è LA base per il nostro problema VRAM/cache (M8): NON scriviamo un fix
  nostro; SPEX-main la testa domani. La 0011 (no device-cache statics support model)
  resta complementare.
- **Issue #468**: thread sul checkpoint V4-Flash-DSpark — i nostri τ misurati
  (5.18/5.00/2.70) sono materiale per l'outreach.

**Riposizionamento del contributo (deciso col track SPEX-main):**
1. **Ramo CUDA+STREAMING = esclusiva nostra**: upstream è Metal-only su DSpark runtime;
   la guardia #495 l'abbiamo sbloccata solo noi (0009/0010 non duplicano nulla, verificato).
2. **Twist expert-IO nella cost-table = esclusiva nostra**: upstream ragiona dense/Metal;
   l'ammortamento union-load in streaming (misurato: −49% @ blocco-8, slots=12→~10 live)
   non esiste nel loro modello di costo.
3. **Fase B on-device = ESTENSIONE di DS4_DSPARK_ADAPTIVE, non alternativa**: stessa
   interfaccia/points d'innesto; la nostra politica diventa una modalità in più (es.
   `DS4_DSPARK_ADAPTIVE=sts`): sopravvivenze STS-calibrate × cost-table IO-aware al
   posto della sola euristica reattiva. Così il patch resta upstream-able qualunque
   esito abbia la #502.

## 5. Rischi aperti
1. Snapshot/rollback della frontiera compressor sotto streaming mai esercitato (fase C).
2. Costo readback logits MTP per passo di draft (host round-trip): mitigazione = kernel
   top2 on-GPU o readback dei soli 2 valori.
3. Nondeterminismo del verifier batch vs decode esatto (documentato upstream a
   `ds4.c:27342-27345`): in streaming va rimisurato il tasso di divergenza greedy.
4. Il margin gate default (3.0) potrebbe già catturare parte del guadagno dello
   scheduler a profondità 2: il confronto onesto in fase B è vs `--mtp-draft 2` con gate
   attivo, non solo vs gate spento.
