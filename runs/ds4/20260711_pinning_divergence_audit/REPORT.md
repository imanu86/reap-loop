# AUDIT — la mask REAP PINNA i keep-expert in VRAM, o VIETA solo i non-keep? (intento vs impl)

Data: 2026-07-11. Sorgenti: patches/ds4/ (repo reap-loop) + live /root/ds4 (ds4.c, ds4_cuda.cu; WSL sola lettura). Modo: statico su codice + transcript/run.

## VERDETTO SECCO

**(A) La mask pinna i keep in VRAM? -> NO.** La mask (0011) scrive `-1e9` SOLO sul router bias `ffn_exp_probs_b`. Non tocca mai la residenza VRAM.

**(B) Divergenza CONFERMATA.** Intento = pinna-keep (residenza VRAM warm). Impl = vieta-non-keep (bias router) + cache-LRU generica keep-cieca. I due sottosistemi (router-mask <-> streaming-cache) sono COMPLETAMENTE SCOLLEGATI.

**(C) Un pinning-VRAM dei keep non e' mai esistito.** Nessuna versione persa. Cio' che assomiglia a "pin" e' tutto pinning-RAM (WRAP mlock, KEEP_MODEL_PAGES), non VRAM.

**(Rotazione) La residency-rotation voluta dall'utente non e' MAI stata implementata.** Ogni rotate testato (1/4/16/32 + demand-admit) opera sulla MASK (`g_reap_mask_pruned`) o pagina in RAM, mai su un set pinnato in VRAM.

---

## A — Cosa fa davvero la mask (0011 `ds4_reap_mask_apply`)

Due sole scritture, entrambe di SELEZIONE, nessuna di RESIDENZA:

1. **CPU** (`layer_topk_selected_experts_from_probs`, ds4.c): il bias mascherato sostituisce il bias router -> `selection[i] += bias[i]`. Pura selezione top-k.
2. **GPU** (`ds4_gpu_model_range_update`, ds4_cuda.cu:15505): upsert di un device-range PER IL TENSORE BIAS `ffn_exp_probs_b` (DS4_N_EXPERT * sizeof(float) = 256 float/layer). Aggiorna il BIAS DEL ROUTER, NON i pesi degli expert (`ffn_gate/up/down_exps`), NON la streaming-cache.

    g_reap_bias_masked[slot][e] = orig[e] + (g_reap_mask_pruned[il][e] ? -1.0e9f : 0.0f);   // ds4.c 0011

Prova di scollegamento (decisiva): in ds4_cuda.cu la ricerca di `rmass|pace_|reap_mask|keep_set|g_reap` restituisce ZERO occorrenze. L'unico simbolo REAP nel .cu e' `ds4_gpu_model_range_update` (l'upsert del bias). La streaming-cache e' CIECA rispetto al keep-set.

### Chi decide la RESIDENZA VRAM (e non sa nulla del keep-set)

- **Streaming expert cache** (`g_stream_selected_cache`, `cuda_stream_selected_cache_begin_compact_load`): LRU-style con eviction (`evictions++`), riempita on-demand dagli expert selezionati. Il resident-hit (0024) e' solo "era-gia-in-cache?", guidato dall'uso, non dal keep.
- **Hot-tier promote** (`cuda_expert_tier_promote`, `hot_count`, `preloaded`): promuove in cache gli expert PIU' FREQUENTI NEL PROMPT (frequenza d'uso), con cap e eviction proprie. Anche questo keep-cieco.

Nessuno dei due riceve la keep-mask. -> I keep sono serviti DALLA STESSA cache di tutti. NESSUN PINNING. Questo spiega perche' K basso non ha mai accelerato: K cambia QUALI expert sono selezionabili, non COME sono serviti.

## B — WRAP / delta-prefetch: RAM, non VRAM

- **0013 WRAP** (`ds4_reap_prefetch_working_set`): "Host-side only (mmap touch + madvise WILLNEED)... no CUDA stream interaction". Pagina i keep nella PAGE-CACHE RAM. `DS4_REAP_WRAP_LOCK=1` -> `mlock()` = pinning-RAM (immune a eviction di pagina RAM), NON VRAM.
- **0021 delta-prefetch**: sul rotate pagina in RAM SOLO gli expert ENTRATI (delta della mask), via le stesse pagein-worker host-side. RAM, guidato dal delta-MASK.
- **KEEP_MODEL_PAGES**: env che SALTA la madvise-free delle pagine mmap del modello -> ritenzione pagine RAM. E' pinning-RAM, non VRAM. (Chiarito: la leva-RAM NON e' pinning-VRAM.)

## C — Ricerca storica: e' mai esistito un pin-VRAM dei keep?

- Grep su TUTTE le patch `patches/ds4/*.patch` per pin/keep/resident/cudaMalloc-keep/stream_cache-mask: unico match = 0024, ed e' solo il testo "keep the resident cache alive" della LRU generica (non keep-expert).
- Grep transcript Codex (`~/.codex/sessions`) e Claude (`~/.claude/projects`) per pinnare/residente-in-vram/pin-keep-vram/lock-in-vram: le occorrenze di "pin" riguardano UI/altro; NESSUN collegamento keep->residenza-VRAM.
- Conclusione: il pinning-VRAM dei keep non e' mai stato scritto ne' perso. E' il PEZZO MANCANTE, non un regresso.

---

## Rotazione — risposta ai 3 punti (mask vs residenza)

**A) rotate32 (0015 `DS4_PACE_ROTATE`) cambia la MASK, non la residenza.**
`ds4_pace_rotate_maybe` -> `ds4_pace_apply_keep_acc(..., g_pace.rmass, cur_keep, "rotate")` -> `ds4_pace_learn_mask(rmass, keep)` RISCRIVE `g_reap_mask_pruned` e ri-applica via `ds4_reap_mask_apply` (bias router `ffn_exp_probs_b`). Tocca il ROUTER/SELEZIONE. Per questo destabilizza la traiettoria -> collasso AN-1 (E-CAL: a pari coverage ~0.79, K23 STATIC sopravvive, K23 ROTATE collassa ~gen126).

**B) NON esiste una rotazione della sola RESIDENZA.** Tutti i meccanismi ruotano la MASK:
- 0015 rotate32 -> `g_reap_mask_pruned` (re-rank wholesale da rmass).
- 0020 s1-slope-trigger -> FORZA il path rotate 0015 (stessa scrittura mask).
- 0021 delta -> dopo la scrittura mask, pagina in RAM i soli entrati.
- 0026 demand-admission -> `g_reap_mask_pruned[il][vic]=1; g_reap_mask_pruned[il][adm]=0` (scambio mirato DELLA MASK), swap fisico = delta-prefetch RAM 0021.

I due sono FUSI: rotate = ruota-mask. La streaming-cache VRAM non e' mai ruotata su domanda: resta LRU keep-cieca.

**C) Micro-patch giusta = residency-rotation (SPEC sotto). E' il pezzo mancante per la velocita', e NON collassa** (mask bit-identica -> traiettoria intatta).

### Prova rotate-1 (i "test vuoti" dell'utente)

Run `runs/ds4/20260710_eadmit_demand_admission/REPORT.md`: rotate-1 (E), rotate-1+isteresi (E-prime), demand-admit (C) sono SCAMBI DI MASK ("mask a K costante dove un expert VIETATO viene AMMESSO, sfrattando il keep meno usato"), misurati in COVERAGE (qualita'), swap fisico = delta-prefetch RAM 0021 (GiB/100tok). NESSUN set pinnato in VRAM. Quindi il rotate-1:
- NON operava sulla residenza (impossibile: nessun set pinnato esiste),
- operava sulla MASK (1 expert selezionabile in piu'/meno = tocca l'ANCORA/traiettoria) o nel RAM-paging.

-> Confermato: TUTTI i rotate (1/4/16/32) hanno toccato la mask o la RAM, MAI la residenza pinnata. I test rotate-1 erano "vuoti" nel senso preciso che mancava il PREREQUISITO: un set residente in VRAM da ruotare. "Ruota-1-expert" ha senso SOLO sopra un set pinnato (sostituisci il piu' freddo residente col piu' richiesto), prerequisito MAI esistito.

---

## SPEC micro-patch 0031 pin-keep-experts (residency-rotation)

**Obiettivo:** disaccoppiare QUALITA' e VELOCITA'. La keep-K mask resta STATICA e bit-identica (router invariato: nessun collasso). In parallelo un set PINNATO in VRAM dei keep, eviction-immune, che RUOTA sulla domanda rmass (pin il caldo-richiesto, evict il freddo) SENZA toccare `g_reap_mask_pruned`.

Dove agire: interamente in ds4_cuda.cu (streaming-cache), NON in ds4.c/router.
1. **Pin-tier**: allocazione VRAM residente dedicata (slab) per i keep gate/up/down; slot marcati NON-EVICTABLE in `cuda_stream_selected_cache` / `g_stream_selected_cache` (nuovo flag pinned).
2. **Freeze**: al primo freeze mask, cudaMemcpy H2D dei keep nella slab e set pinned=1.
3. **Rotazione residenza**: riusa rmass 0020 + CUSUM 0026 come segnale, ma l'azione e' PIN/UNPIN VRAM (memcpy nella slab + toggle flag), NON un flip della mask. Cooldown anti-thrash 0026.
4. La mask e il router restano INVARIATI -> traiettoria intatta -> nessun AN-1.

**Fattibilita': MEDIO-ALTA.** Il sensore (rmass 0020) e il segnale di scambio (CUSUM 0026) esistono gia'; il lavoro nuovo e' la SLAB PINNATA + FLAG NON-EVICT nella CUDA cache e il redirect dell'azione da mask->residenza.
**Gate vincolante = budget VRAM del 3060 (12 GiB):** pinnare TUTTI i keep ~= keep(23) x n_layer(~58) x 6.75 MiB ~= ~9 GiB, non entra accanto al modello residente. Percio' la variante realistica e' PIN DI UN SOTTOINSIEME CALDO CHE RUOTA (non tutti i keep): esattamente la residency-rotation, non un pin-all statico. Coerente col motivo per cui un pin-all-keep non e' mai stato costruito.

**E' il pezzo mancante per la velocita' di K basso?** Si', plausibilmente: oggi la residenza e' una LRU keep-cieca il cui overhead domina (finding 2026-07-11), e K non la tocca. Legare keep->residenza-pinnata-che-ruota e' l'unico percorso che rende la velocita' funzione del set caldo SENZA perturbare la traiettoria. Resta da confermare live (A/B S3) che il pin-tier riduca le direct-load H2D/token.
