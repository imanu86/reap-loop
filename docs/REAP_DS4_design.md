# REAP → ds4 gguf — design doc (track `reap/ds4-domain-prune`)

> Deliverable 1 del `docs/briefs/BRIEF_REAP_DS4.md`. Tutto ciò che segue è verificato su:
> (a) sorgente ds4 upstream pin `80ebbc3` (clone di analisi, righe citate come `ds4.c:NNNN`);
> (b) il gguf reale `models\ds4\DeepSeek-V4-Flash-IQ2XXS-imatrix.gguf` (86,720,111,488 byte),
> header/directory letti con `scripts/gguf_inspect_ds4.py` → `runs/reap/gguf_flash_expert_geometry.txt`.

## 0. TL;DR — le 4 decisioni

1. **Due stadi.** Stage A = *bias-mask* (scrivi `-1e9` nel bias di selezione dei pruned dentro una
   copia del gguf): equivalenza **esatta** col pruning fisico sui layer 3..42, gira su ds4 **stock**,
   costo ~zero → è lo stadio con cui si fanno TUTTE le eval di qualità (pruned vs random vs full).
   Stage B = *surgery* (rimozione fisica degli slab + rimappatura indici): si fa UNA volta, solo
   quando la mask è già validata in Stage A, e richiede una patch runtime ds4 (per-layer expert count).
2. **Hash layer 0-2 esclusi dal pruning** (v0): routing per token-id via `ffn_gate_tid2eid`
   (I32 6×129280, `ds4.c:3637`), assegnazione fissa non rinormalizzabile, e infatti niente
   `exp_probs_b` su quei layer (verificato sul file: 40 bias per 43 layer). Costo: ~5 GiB non potati.
3. **Saliency = media condizionale del gate-weight (g-only)** dalla trace ds4 (patch 0006), MAI
   massa/frequenza come criterio primario (anomalia F-ctrl, ledger C8/G7). Il fattore `‖f‖`
   dell'Eq.9 vera non è estraibile dalla trace: approssimazione validata offline sul 30B
   (retention 0.90 a K50, § 4.2) e coperta dai controlli in eval.
4. **Drop 50% sui 40 layer non-hash → file ~47.0 GiB (≈50.5 GB)**: dentro il target 64GB RAM con
   margine per KV/OS. Numeri esatti in `runs/reap/gguf_flash_expert_geometry.txt`.

## 1. Anatomia verificata del gguf ds4 (Flash)

Dal file reale (`runs/reap/gguf_flash_expert_geometry.txt`) e dal sorgente:

| Tensore (per layer `blk.N.`) | dims | tipo | byte/expert | note |
|---|---|---|---|---|
| `ffn_gate_exps.weight` | [4096, 2048, 256] | IQ2_XXS (16) | 2,162,688 | expert = dim[2], slab contigui |
| `ffn_up_exps.weight`   | [4096, 2048, 256] | IQ2_XXS (16) | 2,162,688 | idem |
| `ffn_down_exps.weight` | [2048, 4096, 256] | Q2_K (10) | 2,752,512 | idem |
| `ffn_gate_inp.weight`  | [4096, 256] | F16 (1) | 8,192/riga | router; riga per expert |
| `exp_probs_b.bias`     | [256] | F32 (0) | 4 | **solo layer 3..42** (40 tensori) |
| `ffn_gate_tid2eid.weight` | [6, 129280] | I32 (26) | — | **solo layer 0-2** (hash) |

- **Totale expert: 6.750 MiB/expert, 1.6875 GiB/layer, 72.56 GiB sui 43 layer.**
- Metadata: `deepseek4.expert_count=256`, `expert_used_count=6`, `expert_weights_scale=1.5`,
  `expert_weights_norm=true`, `hash_layer_count=3`, `expert_gating_func=4`.
- La shape è una tabella fissa nel runtime (Flash/Pro, `ds4.c:180-286`) selezionata dai metadata e
  poi validata campo-per-campo: `config_expect_u32("expert_count", ...)` (`ds4.c:3949`) e
  `tensor_expect_routed_expert(..., DS4_N_EXPERT)` (`ds4.c:3626-3628`). **Un gguf con
  `expert_count≠256` viene rifiutato dal runtime stock** → serve la patch di §3.3 per Stage B.
- **MTP è un gguf separato** (32 tensori `mtp.0.*`, architettura `deepseek4_mtp_support`,
  expert_count proprio = 256, con gate/bias propri). v0: eval senza MTP; v1: stessa mask su `mtp.0.*`.

### 1.1 Matematica del router (`ds4.c:7328-7431`) — perché la rinormalizzazione è automatica

```
probs[i]  = sqrt(softplus(logit_i))          # per-expert, NESSUNA softmax globale (ds4.c:7337)
selection = probs + exp_probs_b              # SOLO per scegliere i top-6 (ds4.c:7415-7420)
weights   = probs[selected] / Σ probs[selected] * 1.5   # normalizzati sui 6 scelti (ds4.c:7422-7430)
```

Conseguenze chiave:
- i `probs` sono indipendenti per expert → rimuovere expert non cambia i probs dei superstiti;
- i pesi sono normalizzati **sul top-6 selezionato**, qualunque esso sia → il "router rinormalizzato
  sui sopravvissuti" chiesto dal brief è **gratis**: top-6 sui superstiti ⇒ pesi già rinormalizzati.
- il bias `exp_probs_b` entra SOLO nella selezione, mai nei pesi → è la leva dello Stage A.

## 2. Stage A — bias-mask (pruning logico, runtime stock)

**Meccanismo.** Su una copia del gguf, per ogni layer 3..42 scrivi `-1e9` (F32) nelle posizioni del
`exp_probs_b.bias` corrispondenti agli expert da potare. `selection = probs + (-1e9)` non entra mai
nel top-6 (probs ≥ 0, bias reali O(1)); i pesi dei superstiti si rinormalizzano da soli (§1.1).

**Equivalenza col pruning fisico**: identica selezione, identici pesi, identici FLOP utili — cambia
solo che i byte dei pruned restano nel file (mai letti dallo streaming: un expert mai selezionato
non viene mai caricato). Quindi Stage A misura **la stessa qualità** del modello fisicamente potato.

**Cosa NON dà Stage A**: la riduzione del file (deliverable finale). E il footprint page-cache resta
sparso sul file grande. È uno strumento di *eval*, non il prodotto.

**Attuazione**: `scripts/reap_bias_mask_ds4.py` (da scrivere, ~100 righe: parser gguf già
prototipato in `scripts/gguf_inspect_ds4.py`): input = mask json (§5), output = patch in-place con
backup dei 40×1024 byte originali (reversibile, si può fare full→masked→restore→random sulla stessa
copia; su pod evita il doppio download).

## 3. Stage B — surgery (pruning fisico + rimappatura)

### 3.1 Riscrittura del file

Tool Python `scripts/reap_prune_ds4_gguf.py` (da scrivere) derivato dallo scheletro upstream
`gguf-tools/mixed/splice_mixed_expert_layers_gguf.py` (parser header/KV/directory + streaming dei
payload senza dequantizzare — già gestisce IQ2_XXS/Q2_K/Q4_K/F16/F32/I32). Per ogni layer con
keep-list `S_l` (ordinata ascendente = nuova numerazione):

1. `ffn_{gate,up,down}_exps.weight`: copia i soli slab `e ∈ S_l` (slab = byte/expert della tabella
   §1, offset = `e × slab`); nuova dim[2] = `|S_l|`.
2. `ffn_gate_inp.weight`: copia le sole righe `e ∈ S_l` (8,192 byte/riga F16).
3. `exp_probs_b.bias`: copia i soli float `e ∈ S_l`.
4. Layer 0-2 (hash): copiati INTATTI (tid2eid compreso).
5. KV `deepseek4.expert_count`: **resta 256** (= conteggio architetturale massimo); il conteggio
   reale per-layer lo dà `dim[2]` del tensore. Così un runtime NON patchato fallisce rumorosamente
   (fail-loud, niente risultati silenziosamente sbagliati).
6. Directory tensori riscritta con offset ricompattati (allineamento 32, come upstream).

### 3.2 Rimappatura indici (il punto delicato del brief)

Gli expert-ID diventano **posizionali**: il vecchio id `e` diventa `rank di e in S_l`. Artefatti in
vecchio ID-space che si ROMPONO senza rimappa:

| Artefatto | Impatto | Rimedio |
|---|---|---|
| hotlist statica di default (`ds4_streaming_hotlist.inc`, caricata in `ds4.c:19821`) | preload di id sbagliati | disattivarla per file potati (env già esistente per hotlist file esterno) o rimappare |
| file `.spex` Markov (patch 0004, shape-check `L=43 E=256`) | rifiutato/id sbagliati | rigenerare la trace SUL modello potato (coerente col TODO ledger #10 "REAP-then-SPEX") |
| trace CSV vecchie, expert-profile JSON | analisi incoerenti | la mask json (§5) contiene `old2new` per convertire |

La mask json ship-a la mappa `old→new` completa per layer (§5) — è la fonte unica di verità.

### 3.3 Patch runtime ds4 (necessaria SOLO per Stage B)

`0007-reap-per-layer-expert-count.patch` (da scrivere quando Stage A avrà validato la mask):
introduce `ds4_layer_n_expert(il) = ffn_gate_exps->dim[2]` e lo usa al posto del `DS4_N_EXPERT`
globale nei punti: layout check (`ds4.c:3626`), loop probs router (`ds4.c:7336`), bound del top-k
(`ds4.c:7420`), kernel select GPU (`ds4_gpu_router_select_tensor`, chiamata `ds4.c:15566`), bounds
del profiler/trace. Gli array statici sono già `DS4_MAX_EXPERT=384` (`ds4.c:122`) → nessun rischio
di overflow con count minori. Le tabelle streaming/cache derivano già i byte dai dims dei tensori
(`ds4.c:3261-3277`) → invariate.

## 4. Saliency dalla trace ds4

### 4.1 Cosa logga ds4 e cosa serve

- Patch 0005 (nostra): CSV `pos,layer,n,e0..e5` — **manca il gate-weight** → Eq.9 non calcolabile.
- Patch 0006 (deliverable 2, questo track): con `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1` il CSV diventa
  `pos,layer,n,e0..e5,w0..w5`. I pesi sono già su host nei 3 call-site di
  `metal_graph_spex_note_selected` (readback bloccante di `g->router_selected` dallo stesso kernel
  che scrive `g->router_weights`, `ds4.c:14804-14822` mostra il pattern) → costo = una
  `ds4_gpu_tensor_read` da 24 byte per layer/token, zero se env assente.
- Cross-check zero-patch: il profiler upstream (`--expert-profile` / env `DS4_EXPERT_PROFILE`,
  `ds4.c:25616-25624`) accumula già `hist` + `weight_hist` per (layer, expert) e con env
  `DS4_EXPERT_HOTLIST` scrive la distribuzione COMPLETA `layer expert hits weight`
  (`ds4.c:1075-1137`). Da verificare sul pod che si popoli sotto i nostri flag streaming
  (call-site unico a `ds4.c:15589`); se sì, `weight/hits` = stessa saliency, doppio conteggio
  indipendente della stessa run.

Saliency per (layer, expert): **S'ₗₑ = Σw / conteggio sui token in cui e è selezionato** (media
condizionale del peso, frequency-agnostic come l'Eq.9). Expert mai selezionati → S'=0 (più freddi).

### 4.2 Quanto costa l'approssimazione g-only (manca ‖f‖) — misurato

L'Eq.9 vera è `mean(g·‖f‖)` (output dell'expert incluso, `scripts_pod/reap_saliency.py`); dalla
trace ds4 si ha solo `g`. Validazione offline su 30B dominio (stessi 152 prompt della reference):
`scripts/reap_gonly_vs_eq9_30b.py` → `runs/reap/gonly_vs_eq9_30b.json`:

- **retention@K64 (50%): 0.902 ± 0.109** — potando col ranking g-only i superstiti trattengono il
  90% della saliency-vera trattenuta dal prune ottimo; overlap grezzo bottom-64: 0.754.
- unico layer degenere: layer 2 (retention 0.191) — sul ds4-Flash i layer 0-2 sono comunque esclusi
  (hash). freq (solo hits) è sistematicamente ≤ g-only.
- caveat dichiarati nel json: g ricostruito dai probe-scores (fedeltà top-k 99.8%, ledger E8), non
  dai logits router veri; sul ds4 la trace 0006 darà i g **esatti**, quindi questi numeri sono un
  lower bound della fedeltà.

Mitigazioni residue del gap ‖f‖: (a) controllo RANDOM sempre (già obbligo brief); (b) ablation
gratuita freq-vs-g-only dalla stessa trace (se g-only ≫ freq sul ds4, la pesatura conta e il rank è
sano); (c) fallback se l'eval delude: patch CPU-path che logga `‖down‖` per expert selezionato
(`layer_routed_moe_one`, `ds4.c:7437-7520`, il per-expert output è materializzato lì) — trace lenta
ma Eq.9 completa.

### 4.3 Nota quantizzazione

La saliency misurata sul modello IQ2 è saliency **dell'artefatto che effettivamente potiamo** (il
gguf 2-bit), non del modello bf16: per questo deliverable è un pregio, non un confound. Confound
reale da dichiarare nel paper: non confrontabile 1:1 con la serie 30B bf16.

## 5. Formato mask json (`reap_mask_ds4_domain.json`)

```json
{
  "tag": "ds4_flash_domain_K50",
  "model": "DeepSeek-V4-Flash-IQ2XXS-imatrix.gguf",
  "n_layer": 43, "n_expert": 256, "hash_layers": [0,1,2],
  "method": "gonly_conditional_mean",
  "source_trace": "runs/reap/<data>_trace_dominio/trace_weights.csv",
  "keep_frac": 0.5,
  "keep": {"3": [/* id VECCHI ordinati asc dei superstiti */], "...": []},
  "old2new": {"3": {"7": 0, "12": 1}, "...": {}},
  "random_control": {"seed": 0, "keep": {"3": []}},
  "est_file_gib": 47.01
}
```

`keep` ordinato ascendente = anche nuova numerazione (old2new ridondante ma esplicito, anti-errore).
Il controllo random è **nella stessa mask json** (stesso K per-layer, seed loggato) così l'eval non
può disallinearsi.

## 6. Footprint atteso (file: `runs/reap/gguf_flash_expert_geometry.txt`)

| drop sui 40 layer non-hash | risparmio | file risultante |
|---|---|---|
| 50% (keep 128/256) | 33.75 GiB | **47.01 GiB (≈50.5 GB)** |
| 54% (keep 118/256) | 36.45 GiB | 44.31 GiB |
| 60% (keep 102/256) | 40.50 GiB | 40.26 GiB |

Il file include già backbone e statico; a K50 restano ~17 GiB di headroom nei 64 GiB per
KV/OS/processi. v1 opzionale: potare anche i hash layer via rimappa tid2eid (fino a −2.5 GiB a K50,
ma richiede riassegnare i token dei pruned → rischio qualità non quantificato, NON in v0).

## 7. Rischi

1. **‖f‖ mancante nella saliency** — quantificato §4.2 (retention 0.90); mitigato da random control
   + ablation freq/g-only + fallback CPU-trace. Il rischio residuo si vede nell'eval, non prima.
2. **Trace dominio corta/di parte** — prompt reali del task di estrazione ≥500 token decode, più prompt diversi
   (coord. col task "trace multi-dominio pod"); la trace smoke (65 token, "Conta da 1 a 200")
   NON è una base per la mask.
3. **Distribuzione hash-layer** — i layer 0-2 non compaiono nella trace (verificato nella smoke:
   layer 3..42) e non servono: non li potiamo.
4. **Profiler upstream non si popola sotto flag streaming** — solo cross-check, il primario è 0006;
   verifica V1 sul pod (righe attese = token×40).
5. **Stage B rompe artefatti ID-based** — gestito con `old2new` + hotlist off (§3.2).
6. **MTP** — v0 senza MTP; per la produzione serve la stessa mask su `mtp.0.*` (v1) o accettare
   MTP full (3.8GB, file separato).
7. **Eval generativa costosa su pod** — il piano eval (deliverable 4) usa ppl forward-only e budget
   token ridotto; la field-accuracy N=152 completa va fatta una sola volta sulla config vincente.

## 7.bis Bias-mask come ATTUATORE del REAP-loop dinamico (gap per il toggle in-memory)

Contesto (mandato SPEX-main 2026-07-05): l'obiettivo di progetto è un motore
`ds4flash-dwarf-spex-reap_loop` dove la mask segue la deriva del dominio a runtime.
La bias-mask è candidata ad attuatore. Stato e gap:

**Oggi (Stage A come implementato):** la mask si applica **scrivendo il FILE gguf**
(`scripts/reap_bias_mask_ds4.py`, byte-patch dei 40×1KB `exp_probs_b`) e serve un
**riavvio del processo** perché il bias viene letto al load e caricato sulla GPU.
Apply/remove ≈ secondi (40KB scritti), ma il ciclo completo costa un restart
(re-init CUDA + re-warm cache ≈ minuti su hardware piccolo).

**Vantaggio strutturale della bias-mask come attuatore** (vs surgery Stage B): gli
expert-ID restano quelli ORIGINALI — niente rinumerazione `old2new`, quindi Markov
`.spex`, hotlist, trace e ogni stato SPEX restano validi attraverso i cambi di mask.
È la proprietà che rende il REAP-loop componibile con SPEX senza re-dump (il
tier-mixing naive che backfirava in E7 nasceva proprio dal doppio ID-space).

**Gap per il toggle in-memory (proposta patch `0008-reap-runtime-mask`):**
1. Host: array override per-layer `float bias_override[DS4_MAX_LAYER][DS4_MAX_EXPERT]`
   caricato da `DS4_EXPERT_MASK_FILE` (stesso formato mask json); consultato nel path
   CPU al posto di `tensor_data(model, ffn_exp_probs_b)` (`ds4.c:7415-7418`).
2. GPU: il bias vive in un buffer residente usato da `ds4_gpu_router_select_tensor`
   (`ds4.c:15566`); serve una `ds4_gpu_tensor_write` dei 1KB/layer al toggle
   (banda irrisoria, 40KB totali) — stessa primitiva già usata a `ds4.c:14124`.
3. Trigger: endpoint su ds4-server (o SIGUSR1 + reread del file) che ricarica la mask
   e riesegue l'upload dei bias. Nessuna invalidazione necessaria: la mask agisce SOLO
   sulla selezione; expert pruned eventualmente ancora in cache GPU/RAM sono innocui
   (mai più selezionati → evicted naturalmente via LFU).
4. Attenzione all'unico stato derivato: il **profiler/le stats hit-rate** contano per
   expert-id — ok, ID-space invariato. Il KV-cache non dipende dagli expert. Il punto
   delicato è il prefetch SPEX in-flight al momento del toggle: al più prefetcha expert
   appena mascherati (spreco una tantum, non errore).

Effort stimato: patch piccola (~100 righe) sopra 0006; nessun cambio di formato file.
Fino ad allora il loop può già funzionare in modalità "coarse" (apply→restart) usando
il tool attuale, con il costo del re-warm.

## 8. Sequenza operativa (stato → brief)

1. ✅ Design (questo doc) + validazione g-only (`runs/reap/gonly_vs_eq9_30b.json`).
2. Patch `0006-spex-routing-trace-weights.patch` (deliverable 2) — build check su pod insieme al 3.
3. Pod 3090 (~$1, playbook): download modello, build 0001..0006, trace dominio (prompt del task di estrazione,
   500+ token, 2 run + warm-up scartata), `DS4_EXPERT_PROFILE`+`DS4_EXPERT_HOTLIST` attivi come
   cross-check → scp CSV/JSON in `runs/reap/<data>_trace_dominio/` + `meta.json` (GPU, cmdline,
   costo) → **podTerminate + verifica pods vuoto**.
4. Saliency g-only → `reap_mask_ds4_domain.json` (+ random control embedded) + stima GiB.
5. Piano eval (deliverable 4): Stage A bias-mask, ppl dominio pruned/random/full su pod.
