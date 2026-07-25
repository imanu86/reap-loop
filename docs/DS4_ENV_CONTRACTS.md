# DS4_ENV_CONTRACTS — inventario dei contratti delle variabili d'ambiente del runtime DS4

**Data**: 2026-07-25
**Sorgenti analizzati**: `D:/ds4_work/wt-converge/ds4_cuda.cu` (42.6k righe, mtime 2026-07-24 22:44), `ds4.c`, `ds4_server.c`, `ds4_cli.c`
**Preset di riferimento**: `D:/ds4_work/wt-g73-open/tests/g73_open.env.ps1`
**Metodo**: sola lettura del codice. Nessun binario eseguito, nessun `ds4_server` avviato, nessun sorgente modificato.

> **Nota sui numeri di riga.** Tutte le righe `cu:NNN` / `ds4.c:NNN` in questo documento sono state **rilette sul sorgente corrente**. Il file `ds4_cuda.cu` è stato toccato il 2026-07-24 22:44, quindi alcune righe citate in briefing precedenti sono scalate di ~20 righe. Dove il briefing citava `cu:26049-26057`, il parser attuale è a `cu:26073-26081`. **Fidarsi di questo documento, non delle righe storiche.**

---

## Sommario

1. [I sei modi in cui una variabile fallisce](#1-i-sei-modi-in-cui-una-variabile-fallisce)
2. [Tassonomia dei parser](#2-tassonomia-dei-parser--la-tabella-che-serve-per-prima)
3. [Grafo dei vincoli e modi validi](#3-grafo-dei-vincoli-e-modi-validi)
4. [Contratti per famiglia](#4-contratti-per-famiglia)
5. [Letterali hardcoded nei log (campi che mentono)](#5-letterali-hardcoded-nei-log--campi-che-mentono)
6. [Fattibilità di `--validate-config`](#6-fattibilità-di---validate-config)
7. [Copertura di questo documento](#7-copertura-di-questo-documento)
8. [Come usare questo documento](#8-come-usare-questo-documento)

> **In tre righe, se hai fretta:**
> **(a)** Su **273** variabili, **139 (51%)** usano un parser a sola presenza: `DS4_X=0` le **ACCENDE**. Per disattivare, `unset`. Sempre.
> **(b)** Le variabili non sono indipendenti: 34 gate riducono 1,3×10¹⁰ combinazioni a **33 modi di servizio**. I tre modi già toccati sono E24 (G73-OPEN), E14 (IQ2 puro) ed E09 (full-Q1 esclusivo).
> **(c)** **23 campi di log sono letterali hardcoded** travestiti da stato misurato. Un campo `chiave=valore` è prova solo se `valore` è un `%`-specifier.

---

## 1. I sei modi in cui una variabile fallisce

Tutti e sei **riverificati sul sorgente corrente**. Le righe sono aggiornate.

| # | modo di guasto | caso verificato | riga attuale | verdetto |
|---|---|---|---|---|
| 1 | **Valore rifiutato** | `DS4_CUDA_PREFILL_TIER_COMPOSE=0` | parser `cu:26073-26081` | **CONFERMATO**. Il parser accetta solo l'assenza (→0) o `"1"` (→1). Qualunque altro valore, **incluso `"0"`**, stampa `ds4: invalid DS4_CUDA_PREFILL_TIER_COMPOSE=%s; expected 1` e **ritorna -1**. Il -1 entra in `cuda_moe_tiering_prepare()` a `cu:26921`, cade nel test `if (compose_requested < 0 ‖ …) return 0;` a `cu:26932-26934`, che fa fallire `cuda_moe_expert_cache_ensure` a `cu:31549` con `ds4: expert tiering preparation failed` → `return NULL`. **Per disattivarlo: NON settarla.** |
| 2 | **Si applica ma NON è binding** | `DS4_CUDA_STREAMING_EXPERT_CACHE_N=160` | parser `cu:25985-25995`, capacità `cu:31305-31316` | **CONFERMATO**. Il parser restituisce il valore (clampato a 512). Ma la capacità effettiva è `cap = min(max_slots, requested)` dove `max_slots = (free_b - reserve) / per_expert` con `free_b` da `cudaMemGetInfo` — cioè **la VRAM libera è quasi sempre il vincolo binding, non la variabile**. La riga `resident expert cache ready: N/M` a `cu:31596` stampa `cap` e `requested` **entrambi**: se N < M la variabile non è binding. |
| 3 | **Si applica ma non produce l'effetto** | `DS4_CUDA_STREAM_RUNTIME_RESERVE_MB=256` | parser `cu:6218`, log `cu:31612-31616` | **CONFERMATO**. La variabile è letta in `cuda_model_stream_reserve_bytes()` **solo se** `g_model_streaming_active && g_model_expert_cache_ready` (`cu:6217`); altrimenti si ricade su `DS4_CUDA_STREAM_RESERVE_MB` (`cu:6220`). Il log `CUDA stream runtime reserve activated: %.2f GiB -> %.2f GiB` è **prova di esito reale** (confronta `startup_reserve` calcolato prima di `g_model_expert_cache_ready=1` con `runtime_reserve` calcolato dopo, `cu:31592-31594`) — ma prova solo che la *reserve* è cambiata, non che il guasto a valle sia risolto. |
| 4 | **No-op per costruzione** | `DS4_CUDA_KV_MANAGED` | `cu:6839`, alloc `cu:6820-6831` | **CONFERMATO**. `ds4_gpu_tensor_alloc()` chiama `cudaMallocManaged` **incondizionatamente** (`cu:6824`): la memoria è managed comunque. La variabile gate solo due `cudaMemAdvise` (`SetPreferredLocation` a `cu:6855`, `SetAccessedBy` a `cu:6860`) dentro `ds4_gpu_tensor_alloc_kv_cache`. Su Windows/WDDM `cudaMemAdvise` **fallisce sempre**, e il codice lo dice: `ds4: [kv-managed] cudaMemAdvise unavailable (%s); managed KV stays migratable` (`cu:6866-6870`). **La presenza di `[kv-managed]` è prova di INTENTO, non di esito.** |
| 5 | **Gated da una condizione mai soddisfatta** | `DS4_Q1_VRAM_LRU_SLOTS` | parser `cu:30329-30347`, gate `cu:36992` | **CONFERMATO**. Il consumatore è `cu:36991-36995`: `cuda_q1_0_resident_transport_requested() && cuda_q1_vram_lru_prepare(...) > 0 && capacity >= compact_count`. `cuda_q1_0_resident_transport_requested()` (`cu:2422-2425`) è `RESIDENT_ARENA ‖ SNAPSHOT_BACKING`: **senza una delle due la variabile è inerte, il parser non viene nemmeno chiamato.** |
| 6 | **Il campo di log è un letterale hardcoded** | `iq2_vram_cache=q1-routes-bypass`, `preloaded=0` | `cu:4578`, `cu:7652`; `cu:13016` + 10 altre | **CONFERMATO e ALLARGATO**. Vedi [§5](#5-letterali-hardcoded-nei-log--campi-che-mentono): ne ho trovati altri, incluso `router=unbiased mask=off` hardcoded sulla riga di **successo** di `[reap-mass-wrap]` (`cu:12497`). |

---

## 2. Tassonomia dei parser — la tabella che serve per prima

Ogni variabile del runtime cade in **una** di queste 9 classi. La classe determina da sola la risposta alla domanda «come si disattiva», che è quella che costa i run.

| classe | forma nel codice | `=0` disattiva? | valore invalido | esempio canonico |
|---|---|:---:|---|---|
| **P1 — strict-`1` silenzioso** | `v && strcmp(v,"1")==0` | **sì** (e anche qualunque altro valore) | **ignorato silenziosamente** → OFF | `DS4_Q1_0_RESIDENT_ARENA` `cu:1354` |
| **P2 — strict-`1` con -1** | `!v‖!v[0]`→0; `=="1"`→1; **altrimenti warn + `return -1`** | **NO — `=0` è INVALIDO** | **-1 che propaga** → abort a monte | `DS4_CUDA_PREFILL_TIER_COMPOSE` `cu:26073` |
| **P3 — bool strict 0/1** | accetta `"0"`/`"1"`, warn e fallback su altro | **sì** | warn + fallback (di solito OFF) | `DS4_G73_OPEN` `cu:1388`, helper `cuda_g133_bool_env` `cu:26056` |
| **P4 — bool permissivo** | `v && v[0] && strcmp(v,"0")!=0` | **sì** | **ignorato**: qualunque stringa non-`"0"` → ON | `DS4_CUDA_MOE_GPU_RESIDENT_ROUTES` `cu:25895` |
| **P5 — presenza** | solo `getenv(x) != NULL` | **NO — `=0` ATTIVA** | n/d (il valore non è letto) | `DS4_CUDA_MOE_CACHE_STATS` `cu:26002`, `DS4_CUDA_NO_TF32` `cu:15232` |
| **P6 — numerico con range** | `strtoul` + controllo `*end=='\0'` + `[min,max]` | dipende: spesso `"0"` è il valore-spegnimento **esplicito** | warn + **`return -1`** oppure `return 0` (fallire il chiamante) | `DS4_CUDA_PREFILL_VRAM_SEED_PER_LAYER` `cu:25920`, helper `cuda_moe_tiering_u32_env` `cu:26110` |
| **P7 — numerico permissivo (parse parziale!)** | `strtoull(e,&end,10); if (end != e) mb = v;` | no: `=0` significa *reserve 0*, non "unset" | **accetta il parse parziale**: `"256xyz"` → 256; `"abc"` → resta il default, **senza warning** | `DS4_CUDA_STREAM_RESERVE_MB` `cu:6221` |
| **P8 — enum di stringhe** | catena di `strcmp` | dipende dall'enum (spesso `"off"` o `"0"`) | warn + **`return -1`** (tiering) oppure **ignorato** (cache policy) | `DS4_EXPERT_TIERING` `cu:26027`, `DS4_CUDA_PREFILL_TIER_ROUTER` `cu:26083` |
| **P9 — path / sha / testo** | usata come nome file o confrontata con uno sha atteso | unset = disattivata | file mancante = fatale o fallback, dipende | `DS4_REAP_MASK_FILE` `ds4.c:6236`, `DS4_MODEL_SHA256` `cu:1976` |

### Censimento verificato — e il risultato che conta più di tutti

Classificazione automatica **per variabile** (non per sito) su `ds4_cuda.cu` + `ds4.c`, guardando **tutti** i siti di lettura di ciascun nome:

| classe | variabili | quota | `=0` disattiva? |
|---|---:|---:|:---:|
| **P5 — presenza** | **139** | **51%** | **NO — `=0` ATTIVA** |
| P6/P7 — numerico | 61 | 22% | dipende |
| P1 — strict-`1` silenzioso | 29 | 11% | sì |
| P4 — bool permissivo | 17 | 6% | sì |
| P6/P7 con spegnimento testuale | 10 | 4% | sì |
| P3 — bool 0/1 | 10 | 4% | sì |
| P8 — enum | 6 | 2% | dipende |
| P2 — strict-`1` con -1 | ≥1 | — | **NO — `=0` ABORTA** |
| **totale variabili uniche con `getenv` diretto** | **273** | | |

> ## ⚠️ **Il risultato principale: in più della metà del runtime, `=0` accende.**
>
> **139 variabili su 273 (51%) usano un parser a sola presenza.** Per queste, `DS4_X=0` **attiva** `DS4_X`. E siccome ~40 di esse si chiamano `DS4_CUDA_NO_*` o `DS4_CUDA_DISABLE_*`, la grafia `DS4_CUDA_NO_TF32=0` — che ogni lettore interpreta come «non disabilitare TF32» — **disabilita TF32**.
>
> Nessuna di queste variabili emette un warning, perché dal loro punto di vista non c'è niente di sbagliato: il valore non viene nemmeno letto.

Nota metodologica: il classificatore automatico sottostima **P2** (conta 1) perché le varianti che ritornano `-1` usano messaggi diversi (`expected 1`, `expected closed or open`, `reason=invalid-enabled`, `ds4: invalid %s=%s` dagli helper `cuda_moe_tiering_u32_env`/`_double_env`/`cuda_q1_0_ssd_wrap_*_env`). Contando gli helper, le variabili il cui valore invalido **aborta** sono ~30 (tutte quelle passate a quei quattro helper). Sono elencate una per una in [§4](#4-contratti-per-famiglia).

### Regola pratica derivata

> **Se non sai la classe, NON settare `=0` per disattivare: rimuovi la variabile.**
> `=0` è sicuro solo per P1, P3, P4 (56 variabili su 273). È **rovinoso** per P5 (139 variabili: attiva invece di disattivare) e per P2 e per gli helper con range (~30 variabili: aborta l'avvio).
> `unset` è **sempre** sicuro: ogni parser ha un ramo per l'assenza.

### Il caso `DS4_CUDA_MOE_CACHE_POLICY` (correzione al briefing)

Il briefing lo elencava insieme a `PREFILL_TIER_COMPOSE=0` come «valore rifiutato». **Non è così, ed è peggio.** L'unico lettore è `cuda_moe_expert_cache_layer_top1()` a `cu:25995-25998`:

```c
static int cuda_moe_expert_cache_layer_top1(void) {
    const char *env = getenv("DS4_CUDA_MOE_CACHE_POLICY");
    return env && strcmp(env, "layer-top1") == 0;
}
```

È una **P1 travestita da enum**: l'unico valore con significato è `layer-top1`. `lru` non produce nessun errore, nessun warning, e nessun effetto — è **silenziosamente ignorato**. Nel preset `g73_open.env.ps1` è settato a `'lru'`, cioè **è una riga di preset senza effetto** (il comportamento LRU è quello di default del rimpiazzo slot, non selezionato da questa variabile).
Nota: esistono `policy=lru` (`cu:5883`), `policy=global-lru` (`cu:30468`), `policy=layer-lru` (`cu:33700`) nei log — sono **letterali** di sottosistemi diversi, non l'eco di questa variabile. Vedi [§5](#5-letterali-hardcoded-nei-log--campi-che-mentono).

### P10 — la classe che mancava: **parser divergente fra `ds4.c` e `ds4_cuda.cu`**

**32 variabili sono lette da entrambi i file.** Per 7 di esse i due parser hanno contratti *diversi*, e questo è un settimo modo di guasto che non era nella lista.

| VAR | `ds4.c` | `ds4_cuda.cu` | conseguenza |
|---|---|---|---|
| `DS4_Q1_0_RESIDENT_ARENA` | `ds4.c:2457` **P3-con-(-1)**: `0`/`1` ok, altro → warn + `-1` → **`ds4_engine_open` fallisce** (`ds4.c:19541`) | `cu:1354` **P1 strict-1 silenzioso**: tutto ciò che non è `"1"` → 0, nessun warning | `=true` / `=yes` / `=ON` è **fatale** all'apertura engine ma sarebbe stato *silenziosamente OFF* nel layer CUDA |
| `DS4_Q1_0_DUAL_ARENA` | `ds4.c:2467` idem | `cu:1359` idem | idem |
| `DS4_Q1_0_DUAL_SPARSE_COMPANION` | `ds4.c:2477` idem | `cu:1364` idem | idem |
| `DS4_Q1_0_MIXED_COLD_ONE` | `ds4.c:2487` idem | `cu:1369` idem | idem |
| `DS4_Q1_0_SNAPSHOT_BACKING` | `ds4.c:2497` idem | `cu:1374` idem | idem |
| `DS4_Q1_0_PAGEABLE_OVERFLOW` | `ds4.c:2507` idem | `cu:1379` idem | idem |
| `DS4_Q1_0_DYNAMIC_PROMOTION` | `ds4.c:2517` idem | `cu:1384` idem | idem |

Buona notizia: per tutti e 7 **`=0` è accettato da entrambi**. Cattiva notizia: qualunque altra grafia («true», «on», «1 » con spazio) produce comportamenti *diversi* nei due strati, e nel layer CUDA non lascia traccia.

Altre due divergenze, di natura diversa:

- **`DS4_CUDA_DYNAMIC_ARENA_GB`** — `ds4.c:20186` fa `strtod(env, NULL)` **senza alcuna validazione**: `"abc"` → `0.0` in silenzio, cioè arena disattivata senza dirlo. Il layer CUDA la ri-legge con controlli propri.
- **`DS4_CUDA_STREAMING_EXPERT_CACHE_N`** ha **tre consumatori con tre semantiche**: (a) `cu:25985` capacità della expert-cache; (b) `ds4.c:1719-1725` `accelerator_has_dedicated_expert_cache()` = predicato di *presenza-non-zero*; (c) `ds4.c:10039-10046` **disabilita il prefetch SPEX K1** se non-zero. Cioè: alzare `CACHE_N` ha l'effetto collaterale non documentato di spegnere il prefetch SPEX.

---

## 3. Grafo dei vincoli e modi validi

### 3.1 Il risultato

Le variabili **non sono indipendenti**. Ho estratto **34 predicati di rifiuto** dal codice (ogni `fprintf` con `requires`/`rejects`/`expected`/`invalid` seguito da `return 0`/`return -1`), li ho codificati come vincoli logici con provenienza `file:riga`, e ho enumerato lo spazio con backtracking + pruning.

| grandezza | valore |
|---|---|
| variabili che compaiono in almeno un gate | **33** |
| spazio grezzo (prodotto dei domini) | **12.884.901.888** |
| configurazioni che superano **tutti** i gate | **48.912** |
| fattore di riduzione | **263.430×** |
| **modi nominali** (per 11 variabili portanti) | **39** |
| **modi effettivi** (collassando le portanti accettate ma non consumate) | **33** |

> **La scoperta**: il problema non è combinatorio. Sono **33 modi di servizio**, catalogabili. La ricerca passa da esplorazione esponenziale a caratterizzazione di un catalogo finito. Dentro ogni modo le variabili *libere* (elencate sotto) sono quelle testabili **una alla volta senza rompere il modo** — quella è la vera matrice di test.

Lo script di enumerazione è riproducibile e vive accanto a questo documento: **`docs/ds4_env_modes.py`** (`python ds4_env_modes.py`). È sola analisi statica: non apre il modello, non tocca CUDA, non avvia nulla. Ogni vincolo porta il suo `file:riga`, quindi è anche la lista di controllo da ri-verificare quando il sorgente cambia.

### 3.2 Il grafo — archi «richiede»

Notazione: `A ⇒ B` = «se A è attiva, B deve valere, altrimenti il runtime rifiuta».

```
                          DS4_CUDA_DYNAMIC_ARENA_GB > 0
                                      ▲
                                      │ C10 cu:27175
                          DS4_EXPERT_TIERING=enforce
                            ▲         ▲          ▲
             C8b cu:27007   │         │ C9       │ C13 cu:31192
                            │         │ cu:27085 │
   DS4_CUDA_PREFILL_TIER_COMPOSE=1  G133=1   MOE_GPU_RESIDENT_ROUTES=1
        ▲        ▲       ▲                        ▲
 C3     │        │ C11   │ C14 cu:31203           │
 cu:26957        │ cu:27184                       │
        │        │                                │
 ROUTER=open   PREFILL_MASS_OBSERVE=1        (tutti i tiering != off)
        ▲       + PREFILL_MASS_WRAP=1
        │       + POLICY=mass-lfru
        │       + NESSUN observer REAP
        │
   DS4_G73_OPEN=1  ──C4 cu:26963──▶ {G133, enforce, compose, router=open,
        │                            reserve_slots>0, gpu_routes}
        └──C4b cu:1478──▶ NESSUNA var Q1/IQ1 di arena/promotion/serving

   DS4_Q1_0_DYNAMIC_PROMOTION=1
        ├──C17f ds4.c:19554──▶ RESIDENT_ARENA=1 ∧ DUAL_ARENA=1 ∧ ¬sparse ∧ ¬cold-one ∧ ¬snapshot
        ├──C6   cu:26987   ──▶ enforce ∧ compose ∧ router=open ∧ reserve==probation ∧ sidecar
        └──C7a  cu:26155   ──▶ Q1_PROMOTION_PROBATION_SLOTS>0 ∧ IQ1_PROBATION==0

   DS4_Q1_0_DUAL_ARENA ──C17a ds4.c:19544──▶ RESIDENT_ARENA
                       ──C18c ds4.c:20234──▶ CUDA_DYNAMIC_ARENA_GB>0
   DS4_Q1_0_RESIDENT_ARENA ──C17g ds4.c:19558──▶ sidecar Q1_0 ∧ SELECTED_LOAD=1
                           ──C18a ds4.c:20217──▶ (Q1_0_DYNAMIC_ARENA_GB>0 ∨ CUDA_DYNAMIC_ARENA_GB>0)
   DS4_Q1_0_DUAL_SPARSE_COMPANION ⇔ DS4_Q1_0_MIXED_COLD_ONE   (C17b/C17c, ds4.c:19545-19548)
   DS4_CUDA_PREFILL_VRAM_SEED_* ──C14 cu:31203──▶ enforce ∧ compose ∧ gpu_routes
   DS4_CUDA_PREFILL_VRAM_SEED_FLOOR_PER_LAYER ──C15b cu:31214──▶ SEED_TOTAL>0
   DS4_IQ1_S_MIXED_COLD_K=1 ──C19b ds4.c:19593──▶ sidecar IQ1_S
   DS4_NESTED_RESIDUAL_EXACT ──C20a ds4.c:19492──▶ router=open ∧ nessun sidecar quant
```

### 3.3 Il grafo — archi «esclude»

| A | esclude | provenienza | nota |
|---|---|---|---|
| arena Q1 **esclusiva** (`RESIDENT ∨ SNAPSHOT`, senza `DUAL`) | **tutto** il tiering: `TIERING≠off`, `COMPOSE`, `ROUTER_OPEN`, `probation`, `RESERVE_SLOTS` | `cu:26935-26942` | messaggio: `reason=mixed-host-resolver-not-implemented` |
| `DS4_Q1_0_SNAPSHOT_BACKING` | `RESIDENT_ARENA`, `DUAL_ARENA` | `ds4.c:19552-19553` | |
| `DS4_CUDA_PREFILL_TIER_COMPOSE` | `DS4_CUDA_REAP_MASS_OBSERVE`, `DS4_CUDA_REAP_MASS_WRAP` | `cu:27195` | **compose e REAP-mass sono mutuamente esclusivi** |
| `TIERING=enforce` **senza** compose | **ogni** observer (prefill-mass, REAP-mass, arena-observer) | `cu:27200-27206` | enforce nudo richiede arena idle |
| `DS4_CUDA_MOE_SPLIT_HIT_MISS` | `DS4_CUDA_MOE_SPLIT_FUSED` | `cu:37919-37923` | con `G73_OPEN=1` hit/miss è **forzato a 0** (`cu:37912`), quindi il conflitto non può nemmeno presentarsi |
| `DS4_CUDA_PREFILL_VRAM_SEED_PER_LAYER` | `DS4_CUDA_PREFILL_VRAM_SEED_TOTAL` | `cu:31211-31213` | |
| `DS4_Q1_0_EXPERT_SIDECAR` | `DS4_IQ1_S_MIXED_COLD_K=1` | `ds4.c:19586-19592` | |
| `DS4_G73_OPEN=1` | **44 variabili** Q1/IQ1 (lista a `cu:1420-1477`) | `cu:1478-1512` | escape parziale: `DS4_G73_ALLOW_Q1_SIDECAR=1` esenta 6 var di sola serving |
| `DS4_NESTED_RESIDUAL_EXACT` | `DS4_REAP_MASK_FILE`, sidecar IQ1_S/Q1_0 | `ds4.c:19477-19500` | |

### 3.4 Archi «inerte senza abilitatore» (la variabile viene ignorata, non rifiutata)

Questo è il gruppo che costa i run, perché **non produce nessun messaggio**.

| variabile | inerte se… | provenienza |
|---|---|---|
| `DS4_Q1_VRAM_LRU_SLOTS`, `DS4_Q1_VRAM_LRU_RESERVE_MB` | `¬(RESIDENT_ARENA ∨ SNAPSHOT_BACKING)` | `cu:36992` |
| `DS4_CUDA_STREAM_RUNTIME_RESERVE_MB` | `¬(g_model_streaming_active ∧ g_model_expert_cache_ready)` | `cu:6217` |
| `DS4_CUDA_PREFILL_TIER_COMPOSE` | `TIERING ≠ enforce` — **accettato dai gate, mai consumato** | `cu:27184` |
| `DS4_CUDA_PREFILL_TIER_ROUTER=open` | compose inerte | `cu:13086` |
| `DS4_G133_*` (KNOCK_X/Y, DECAY, PROMOTE_BUDGET, SEED_DYNAMIC) | `DS4_G133_TIER ≠ 1` | `cu:27055-27065` |
| `DS4_EXPERT_TIER_HYSTERESIS/CLOCK_CALLS/…` | `TIERING = off` (il parser non viene raggiunto) | `cu:27017-27046` |
| `DS4_CUDA_MOE_CACHE_POLICY` | sempre, salvo il valore esatto `layer-top1` | `cu:25995` |
| `DS4_CUDA_KV_MANAGED` | sempre su Windows/WDDM (`cudaMemAdvise` fallisce) | `cu:6866` |
| `DS4_CUDA_NO_MODEL_PREFETCH`, `DS4_CUDA_MODEL_PREFETCH_SYNC` | `_WIN32` (ritorno anticipato) | `cu:4280-4282` |
| `DS4_CUDA_NO_DIRECT_IO`, `DS4_CUDA_KEEP_MODEL_PAGES` | non-Linux (compilate fuori) | `cu:14704`, `cu:4376/4394` |
| `DS4_CUDA_NO_TOPK_CHUNKED` | `DS4_CUDA_NO_TOPK2048` settata (che già chiude quel ramo) | `cu:19872` |
| 7 `DS4_METAL_GRAPH_*` di test | raggiungibili solo da `--metal-graph-*-test`, che abortisce su build CUDA | `ds4.c:19646-19651` |

### 3.5 Il catalogo dei 33 modi effettivi

Colonna «libere» = variabili che assumono **più di un valore** all'interno del modo, cioè quelle che si possono muovere una alla volta senza uscire dal modo. Sono la matrice di test.

| # | modo | config | variabili LIBERE dentro il modo |
|---|---|---:|---|
| E01 | `observe` | 13824 | POLICY, RESERVE_SLOTS, PF_MASS_OBS/WRAP, REAP_MASS_OBS/WRAP, Q1_SIDECAR, Q1_SEL_LOAD, IQ1_SIDECAR, ARENA_GB, Q1_ARENA_GB, SPLIT_HIT_MISS, SPLIT_FUSED |
| E02 | `tiering-OFF` | 9216 | come E01 + GPU_ROUTES |
| E03 | `Q1-dual-arena / observe` | 6912 | POLICY, RESERVE_SLOTS, gli observer, Q1_SPARSE_COMP, Q1_COLD_ONE, Q1_PAGEABLE, IQ1_SIDECAR, Q1_ARENA_GB, SPLIT_* |
| E04 | `Q1-dual-arena / tiering-OFF` | 4608 | idem + GPU_ROUTES |
| E05 | `IQ1-mixed / observe` | 3456 | POLICY, RESERVE_SLOTS, observer, Q1_SEL_LOAD, ARENA_GB, Q1_ARENA_GB, SPLIT_* |
| E06 | `IQ1-mixed / tiering-OFF` | 2304 | idem + GPU_ROUTES |
| E07 | `Q1-snapshot-ESCLUSIVA / tiering-OFF` | 1152 | POLICY, GPU_ROUTES, observer, Q1_PAGEABLE, IQ1_SIDECAR, Q1_ARENA_GB, SPLIT_* |
| E08 | `nested-residual / observe` | 1152 | POLICY, RESERVE_SLOTS, observer, Q1_SEL_LOAD, ARENA_GB, Q1_ARENA_GB, SPLIT_* |
| E09 | **`Q1-resident-ESCLUSIVA / tiering-OFF`** ← **modo C di stanotte** | 864 | POLICY, GPU_ROUTES, observer, IQ1_SIDECAR, ARENA_GB, Q1_ARENA_GB, SPLIT_* |
| E10 | `enforce-nudo` | 384 | POLICY, RESERVE_SLOTS, REAP_MASS_WRAP, Q1_SIDECAR, Q1_SEL_LOAD, IQ1_SIDECAR, Q1_ARENA_GB, SPLIT_* |
| E11 | `Q1-dual-arena / enforce-nudo` | 384 | POLICY, RESERVE_SLOTS, REAP_MASS_WRAP, Q1_SPARSE_COMP, Q1_COLD_ONE, Q1_PAGEABLE, IQ1_SIDECAR, Q1_ARENA_GB, SPLIT_* |
| E12 | `enforce+compose (router chiuso)` | 384 | RESERVE_SLOTS, Q1_SIDECAR, Q1_SEL_LOAD, IQ1_SIDECAR, Q1_ARENA_GB, **SEED_PER_LAYER, SEED_TOTAL, SEED_FLOOR**, SPLIT_* |
| E13 | `enforce+compose (router chiuso) / G133` | 384 | idem E12 |
| **E14** | **`enforce+compose+router-open`** ← **modo B, i ~4 t/s sani** | 384 | RESERVE_SLOTS, Q1_SIDECAR, Q1_SEL_LOAD, IQ1_SIDECAR, Q1_ARENA_GB, SEED_PER_LAYER, SEED_TOTAL, SEED_FLOOR, SPLIT_HIT_MISS, SPLIT_FUSED |
| E15 | `enforce+compose+router-open / G133` | 384 | idem E14 |
| E16-E19 | le quattro varianti `Q1-dual-arena` di E12-E15 | 384 ea. | + Q1_SPARSE_COMP, Q1_COLD_ONE, Q1_PAGEABLE |
| E20 | `enforce-nudo / G133` | 192 | RESERVE_SLOTS, REAP_MASS_WRAP, Q1_SIDECAR, Q1_SEL_LOAD, IQ1_SIDECAR, Q1_ARENA_GB, SPLIT_* |
| E21 | `Q1-dual-arena / enforce-nudo / G133` | 192 | + Q1_SPARSE_COMP, Q1_COLD_ONE, Q1_PAGEABLE |
| E22 | `IQ1-mixed / enforce+compose+router-open` | 144 | RESERVE_SLOTS, Q1_SEL_LOAD, **IQ1_PROBATION**, Q1_ARENA_GB, SEED_*, SPLIT_* |
| E23 | `IQ1-mixed / enforce+compose+router-open / G133` | 144 | idem E22 |
| **E24** | **`G73-OPEN / enforce+compose+router-open / G133`** ← **modo A** | 128 | Q1_SIDECAR, Q1_SEL_LOAD, Q1_ARENA_GB, SEED_PER_LAYER, SEED_TOTAL, SEED_FLOOR, SPLIT_HIT_MISS, SPLIT_FUSED |
| E25 | `IQ1-mixed / enforce-nudo` | 96 | POLICY, RESERVE_SLOTS, REAP_MASS_WRAP, Q1_SEL_LOAD, Q1_ARENA_GB, SPLIT_* |
| E26-E27 | `IQ1-mixed / enforce+compose (router chiuso)` ± G133 | 96 ea. | RESERVE_SLOTS, Q1_SEL_LOAD, Q1_ARENA_GB, SEED_*, SPLIT_* |
| E28-E29 | `nested-residual / enforce+compose+router-open` ± G133 | 96 ea. | RESERVE_SLOTS, Q1_SEL_LOAD, Q1_ARENA_GB, SEED_*, SPLIT_* |
| E30-E31 | `Q1-promotion / enforce+compose+router-open` ± G133 | 96 ea. | Q1_PAGEABLE, IQ1_SIDECAR, Q1_ARENA_GB, SEED_*, SPLIT_* |
| E32 | `G73-OPEN / nested-residual / …/ G133` | 64 | Q1_SEL_LOAD, Q1_ARENA_GB, SEED_*, SPLIT_* |
| E33 | `IQ1-mixed / enforce-nudo / G133` | 48 | RESERVE_SLOTS, REAP_MASS_WRAP, Q1_SEL_LOAD, Q1_ARENA_GB, SPLIT_* |

### 3.6 Verifica degli ancoraggi

Tutti e tre i modi già toccati **sono nel catalogo**, e tutte le configurazioni rifiutate stanotte **risultano insoddisfacibili nel modello** — con il vincolo esatto che le uccide:

| ancoraggio | esito | modo |
|---|---|---|
| **A** — `G73_OPEN=1` + costellazione completa, Q1/IQ1 disabilitati | ✅ valida | **E24** |
| **B** — `G73_OPEN=0` + enforce + compose + router open, IQ2 puro | ✅ valida | **E14** |
| **C** — full-Q1 residente esclusivo, tiering off, seed off | ✅ valida | **E09** |

| configurazione rifiutata stanotte | il modello la rifiuta? | vincolo che la uccide |
|---|---|---|
| `PROMOTION_SSD_WRAP=1`/`DYNAMIC_PROMOTION` senza la costellazione | ✅ sì | C17f `ds4.c:19554` + C6 `cu:26987` + C7a `cu:26155` |
| `RESIDENT_ARENA=1` da solo con tiering enforce | ✅ sì | **C2 `cu:26935`** (`reason=mixed-host-resolver-not-implemented`) |
| `COMPOSE=1` senza enforce | ✅ sì | C8b `cu:27007` |
| `G133_TIER=1` senza enforce | ✅ sì | C8a `cu:27002` + C9 `cu:27085` |
| `G73_OPEN=1` con sidecar Q1 residente | ✅ sì | C2 `cu:26935` + C4b `cu:1478` |
| compose + observer REAP insieme | ✅ sì | **C11 `cu:27195`** |
| seed VRAM con tiering off | ✅ sì | C14 `cu:31203` |

Nessun falso positivo: il grafo è coerente con l'evidenza sperimentale.

### 3.7 Il preset `g73_open.env.ps1` letto attraverso il grafo

Il preset seleziona **E24**. Due righe risultano però **senza effetto**:

| riga del preset | verdetto |
|---|---|
| `DS4_CUDA_MOE_CACHE_POLICY = 'lru'` | **no-op**: l'unico valore letto è `layer-top1` (`cu:25995`). `lru` è silenziosamente ignorato. |
| `DS4_CUDA_STREAMING_EXPERT_CACHE_N = '320'` | **richiesta, non capacità**: il binding è `min(320, (free−reserve)/per_expert)`. Effetto collaterale non ovvio: essendo non-zero **disabilita il prefetch SPEX K1** (`ds4.c:10044-10046`). |
| `DS4_Q1_0_PROMOTION_SSD_WRAP` (rimosso dal preset) | irrilevante: con `G73_OPEN=1` l'ssd-wrap è **forzato a 1** a monte del parser (`cu:1516`). |

---

## 4. Contratti per famiglia

Legenda colonna **riga di prova**:
- **esito** = stampa un valore OTTENUTO (utilizzabile per falsificare)
- **intento** = prova solo che un ramo è stato preso
- **nessuna** = non esiste output correlato
- **MENTE** = campo che sembra stato calcolato ma è un letterale hardcoded

Legenda **come si disattiva**: `=0` / `unset` (= `=0` NON disattiva) / `entrambi`.
★ = presente nel preset `g73_open.env.ps1`.

### 4.1 `DS4_CUDA_STREAMING_EXPERT_CACHE_*` e `DS4_CUDA_MOE_CACHE_*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| ★`DS4_CUDA_STREAMING_EXPERT_CACHE_N` | `cu:25985` **P6** `strtoul`, clamp a 512 | interi; qualunque testo → `return 0` = "non richiesta", **senza warning** | `=0` (= cache off) | `cu:31139`; **NON binding**: `cap=min(req,(free−reserve)/per_expert)` `cu:31307-31311` | numero di slot esperti residenti in VRAM | **esito** — `ds4: CUDA resident expert cache ready: %u/%u experts, %.2f MiB/expert, %.2f GiB total` `cu:31596`. **Se il primo numero < il secondo, la variabile NON è binding.** Fallimento: `resident expert cache disabled: free %.2f GiB <= reserve %.2f GiB` `cu:31313` | **no** — dipende da `cudaMemGetInfo` |
| ★`DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` | `cu:26015` **P6** `strtod`, `>=0` | double ≥0; invalido → **ignorato**, default 0.5 GiB | unset (`=0` = riserva zero) | clamp interno: `reserve = min(reserve, total/2)` `cu:31306` | VRAM lasciata libera dalla expert-cache | **esito indiretto** via `cu:31313` (stampa la reserve applicata) | no |
| ★`DS4_CUDA_MOE_CACHE_POLICY` | `cu:25995` **P1 travestita da enum** | solo `layer-top1` ha effetto; **ogni altro valore (incluso `lru`) è ignorato in silenzio** | qualunque valore ≠ `layer-top1` | nessuno | politica top1-per-layer nella cache esperti | **nessuna** | **sì** |
| `DS4_CUDA_MOE_CACHE_STATS` | `cu:26002` **P5 presenza** | qualunque valore, **`=0` ATTIVA** | **unset** | nessuno | abilita le statistiche periodiche | intento | sì |
| `DS4_CUDA_MOE_CACHE_STATS_INTERVAL` | `cu:26003` **P6** | interi >0; invalido → **fallback 128 silenzioso** | unset | richiede `MOE_CACHE_STATS` settata `cu:26002` | cadenza delle stat | esito | sì |
| ★`DS4_CUDA_MOE_GPU_RESIDENT_ROUTES` | `cu:25895` **P4 bool permissivo** | `"0"`/vuoto → OFF; **ogni altra stringa → ON** | `=0` | **richiesto** da qualunque `TIERING≠off` (C13 `cu:31192`) | route resolver GPU-resident | **esito** — `CUDA GPU-resident route resolver active: exact miss worker, %u slots` `cu:31601` | no |
| ★`DS4_CUDA_MOE_ROUTE_NO_DEFAULT_SYNC` | `cu:25910` **P4** | idem | `=0` | nessuno | elimina la sync sullo stream di default | nessuna | sì |
| `DS4_CUDA_MOE_ROUTE_PACKED_COPY` | `cu:25915` **P1 strict-1** | solo `"1"`; altro → OFF silenzioso | entrambi | nessuno | copia H2D impacchettata gate/up/down | **esito** — `CUDA route packed H2D copy active: gate_bytes=%llu down_bytes=%llu …` `cu:31604` | no |
| ★`DS4_CUDA_MOE_SPLIT_HIT_MISS` | `cu:25900` **P4** | idem | `=0` | esclude `SPLIT_FUSED` (C16a `cu:37919`); **forzato a 0 se `G73_OPEN=1`** `cu:37912` | percorso split hit/miss legacy | intento | sì |
| ★`DS4_CUDA_MOE_SPLIT_FUSED` | `cu:25905` **P4** | idem | `=0` | esclude `SPLIT_HIT_MISS` | overlap split-fused | intento | sì |
| `DS4_CUDA_EXPERT_CACHE_GROW` | `cu:25545` **P1 strict-1** | solo `"1"` | entrambi | `capacity==0 ∧ !decode_started` `cu:31243` | rinvia l'allocazione della expert-cache a decode-start | **esito** — `[expert-cache-grow] mode=two-phase result=ok capacity=%u requested=%u residents=%u` `cu:25627`. Sul ramo failed `cache=disabled fallback=exact-streaming` è **MENTE** (letterale) `cu:25638` | no |
| `DS4_CUDA_MOE_NO_SELECTED_LOAD` | `cu:37882`, `cu:37926` **P5 presenza** | **`=0` ATTIVA** | **unset** | esclude nested-residual `cu:37882` | disabilita il caricamento selettivo degli esperti | nessuna | sì |
| `DS4_CUDA_MOE_PROFILE` | `cu:37930` **P5 presenza** | **`=0` ATTIVA** | **unset** | nessuno | profilo MoE | esito | no |

> **Trappola nota (modo di guasto 2).** `CACHE_N=160` «non cambiava nulla» perché il binding era la VRAM. La riga di prova esiste ed è precisa: `resident expert cache ready: 148/320`. Il **secondo** numero è la richiesta, il **primo** è ciò che si è ottenuto.
> **Effetto collaterale non documentato**: `CACHE_N` non-zero **disabilita il prefetch SPEX K1** (`ds4.c:10039-10046`).

### 4.2 `DS4_CUDA_STREAM_*RESERVE*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| ★`DS4_CUDA_STREAM_RESERVE_MB` | `cu:6220-6221` **P7 numerico permissivo** | `strtoull`; **accetta il parse parziale** (`"256xyz"`→256); `"abc"` → resta il default 2048 MiB **senza warning** | unset (`=0` = reserve 0) | letta solo se `RUNTIME_RESERVE_MB` è assente/vuota o se lo streaming non è ancora attivo | VRAM riservata all'allocatore di range streamed | **esito** — `CUDA stream reserve startup=%.2f GiB runtime=%.2f GiB switch=expert-cache-ready` `cu:15199` (stampata solo se `RUNTIME_RESERVE_MB` è settata) | no |
| `DS4_CUDA_STREAM_RUNTIME_RESERVE_MB` | `cu:6218` **P7**, idem | idem | unset | **`g_model_streaming_active ∧ g_model_expert_cache_ready`** `cu:6217` — prima di quel punto è inerte | abbassa la reserve dopo che la expert-cache è pronta | **esito** — `CUDA stream runtime reserve activated: %.2f GiB -> %.2f GiB` `cu:31612`, stampata **solo se** `startup_reserve != runtime_reserve` (`cu:31611`): il confronto è fra due chiamate reali attorno a `g_model_expert_cache_ready=1` (`cu:31592-31594`) | no |
| `DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB` | `cu:2732` **P6** | numerico | unset | maschera REAP attiva | budget RAM per lo streaming mascherato | esito | no |

> **Trappola nota (modo di guasto 3).** La riga `runtime reserve activated: 1.00 GiB -> 0.25 GiB` è **prova di esito vera** — la reserve È cambiata. Ma dimostra solo il cambio di reserve, non che il guasto a valle sia stato risolto: sono due proposizioni diverse.

### 4.3 `DS4_CUDA_PREFILL_*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| ★`DS4_CUDA_PREFILL_TIER_COMPOSE` | `cu:26073` **P2 strict-1 con -1** | assente→0; `"1"`→1; **tutto il resto, `"0"` incluso, → warn + `-1`** | **⚠️ NON settarla.** `=0` è INVALIDO e **aborta** | consumata solo con `TIERING=enforce` `cu:27184`; richiede `PREFILL_MASS_OBSERVE+WRAP`, `mass-lfru`, e nessun observer REAP (C11) | fonde lo snapshot di massa del prefill nel tiering | intento (assenza dell'abort) | **sì** |
| ★`DS4_CUDA_PREFILL_TIER_ROUTER` | `cu:26083` **P8 enum + -1** | assente/`closed`→0; `open`→1; **altro → warn + `-1`** | `=closed` **o** unset. `=0` è INVALIDO | richiede `COMPOSE` (C3 `cu:26957`) | router non biased in prefill | intento | **sì** |
| ★`DS4_CUDA_PREFILL_TIER_RESERVE_SLOTS` | `cu:26093` → `cuda_moe_tiering_u32_env` `cu:26110` **P6 [0,512]** | interi 0..512; invalido → warn `ds4: invalid %s=%s` + `-1` | `=0` | con promotion deve **eguagliare** i probation slots (C5/C6) | slot d'arena riservati | esito (via il messaggio d'errore di capacità) | **sì** |
| ★`DS4_CUDA_PREFILL_MASS_OBSERVE` | `cu:12047` **P3 bool 0/1, warn+disable** | assente/vuoto/`"0"`→0; `"1"`→1; altro → warn UNA volta `[prefill-mass] invalid observe value '%s'; observer disabled` + **0** (non -1) | `=0` | escluso da `enforce senza compose` (C12) | osserva la massa per esperto in prefill | esito | sì |
| ★`DS4_CUDA_PREFILL_MASS_WRAP` | `cu:12033` **P3** idem, `[prefill-mass-wrap] invalid value` | `=0` | richiede `PREFILL_MASS_OBSERVE` | pubblica lo snapshot di residenza | **esito parziale** — `[prefill-mass-wrap] result=%s reason=%s candidate=%u loads=%u workers=%u seconds=%.3f … resident_before=%u resident_after=%u generation=%llu preloaded=0 router=unbiased mask=%s` `cu:13016`. **`preloaded=0` e `router=unbiased` MENTONO** | no |
| `DS4_CUDA_PREFILL_MASS_LAYER_FULL_EVERY` | `cu:12066` **P6 [1,40]** | 1..40; invalido → `[prefill-mass-layer-stripe] result=failed reason=invalid-stride` + **-1** | `=0` | richiede l'observer | stride di layer a copertura piena | esito | sì |
| `DS4_CUDA_PREFILL_MASS_LAYER_FULL_PHASE` | `cu:12079` **P6 [0,stride-1]** | deve essere `< stride`; altrimenti `reason=invalid-phase` + **-1** | unset | richiede `FULL_EVERY` | fase dello stripe | esito | **sì** (coerenza fra le due) |
| `DS4_CUDA_PREFILL_VRAM_SEED_PER_LAYER` | `cu:25920` **P6 [1,32]** | 1..32; `"0"`/assente→0; altro → warn + **-1** | `=0` | richiede enforce+compose+gpu_routes (C14); esclude `SEED_TOTAL` (C15a) | pre-semina N esperti/layer | **esito** — `[prefill-vram-seed] result=… entries=%u bytes=%llu` (sul ramo failed `entries=0 bytes=0` sono letterali, `cu:31068`) | no |
| `DS4_CUDA_PREFILL_VRAM_SEED_TOTAL` | `cu:25952` **P6 [1,512]** | idem | `=0` | idem | budget globale di semina | esito | no |
| `DS4_CUDA_PREFILL_VRAM_SEED_FLOOR_PER_LAYER` | `cu:25969` **P6 [0,32]** | 0..32 | `=0` | richiede `SEED_TOTAL>0` (C15b `cu:31214`) | pavimento per layer | esito | **sì** |
| `DS4_CUDA_PREFILL_WAVES` | `cu:33763` **P5 presenza** | **`=0` ATTIVA** | **unset** | rifiutata se un dump di debug è attivo (`cu:35363-35369`); esclude nested-residual (`cu:37883`) | prefill a ondate | **intento** — `[prefill-waves] refused reason=debug-dump-active layer=%u tokens=%u compact=%u` `cu:35367` (non dice *quale* dump) | sì |
| `DS4_CUDA_PREFILL_WAVE_DOUBLE_BUFFER` | `cu:33768` **P5 presenza** | **`=0` ATTIVA** | **unset** | richiede `PREFILL_WAVES` | doppio buffer fra ondate | nessuna | sì |

### 4.4 `DS4_EXPERT_TIER*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| ★`DS4_EXPERT_TIERING` | `cu:26027` **P8 enum + -1** | `off`/`0`/assente → OFF; `observe`; `enforce`; **altro → `ds4: invalid DS4_EXPERT_TIERING=%s` + `-1` → abort** | `=off` **o** `=0` (entrambi validi) | `enforce` richiede arena idle (C10) e GPU routes (C13) | modalità di tiering esperti | esito | **sì** |
| ★`DS4_EXPERT_TIER_POLICY` | `cu:26046` **P8 enum + -1** | assente/`second-touch`; `mass-lfru`; **altro → warn + `-1`** | unset (default `second-touch`) | `mass-lfru` **obbligatorio** con G133 (C9) e con compose (C11) | politica di rimpiazzo | esito | **sì** |
| ★`DS4_EXPERT_TIER_CLOCK_CALLS` | `cu:27033` → `u32_env` **P6 [1,1000000]** | invalido → `ds4: invalid %s=%s` + il chiamante `return 0` → **abort del prepare** | unset (default 430) | letta solo con `TIERING≠off` `cu:27017` | periodo del clock di tiering | nessuna | **sì** |
| ★`DS4_EXPERT_TIER_REPLACEMENT_BUDGET` | `cu:27036` **P6 [1,512]** | idem | unset (default 16) | idem | rimpiazzi per clock | nessuna | **sì** |
| ★`DS4_EXPERT_TIER_MIN_FREQUENCY` | `cu:27039` **P6 [2,1000000]** | idem — ⚠️ **il minimo è 2**: `=1` è INVALIDO e aborta | unset (default 3) | idem | frequenza minima per promuovere | nessuna | **sì** |
| ★`DS4_EXPERT_TIER_HYSTERESIS` | `cu:27042` → `double_env` **P6 [1.0,100.0]** | idem — `=0` è INVALIDO | unset (default 1.25) | idem | isteresi di sostituzione | nessuna | **sì** |
| `DS4_EXPERT_TIER_ADAPTIVE_BUDGET_MIN/MAX/STEP` | `cu:27097+` **P6 [1,512]** | idem | unset | richiedono il budget adattivo attivo | budget adattivo | esito | **sì** |
| `DS4_EXPERT_TIER_ADAPTIVE_PRESSURE_THRESHOLD` | `cu:27100+` **P6** | idem | unset | idem | soglia di pressione | esito | **sì** |

### 4.5 `DS4_G133_*` e `DS4_G73_*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| ★`DS4_G133_TIER` | `cu:3332` **P3 bool 0/1, cached in `static`** | `0`/`1`; altro → warn `invalid DS4_G133_TIER=%s; expected 0 or 1` + OFF | `=0` | richiede `enforce` + `mass-lfru` (C8a/C9); e `ctx ≤ CUDA_G133_EPOCH_POSITION_MASK` (`cu:26317`) | tiering advisory G133 | esito | **sì** (tranne il limite di contesto) |
| ★`DS4_G133_KNOCK_X` | `cu:27050` **P6 [1,STREAK_MAX]** | invalido → abort del prepare | unset (default 3) | solo con `G133=1`; **`X+Y ≤ STREAK_MAX`** (`cu:27073-27082`) | soglia di knock | esito | **sì** |
| ★`DS4_G133_KNOCK_Y` | `cu:27054` **P6** idem | idem | unset (default 5) | idem | soglia di knock | esito | **sì** |
| ★`DS4_G133_DECAY` | `cu:3367` **e** `cu:27058` **P6 doppio parser** | `0 < decay < 1` **stretto**; `=1.0` e `=0.0` INVALIDI → `must satisfy 0 < decay < 1` | unset (default 0.98) | validata **due volte**: a `cu:3367` (dispatch init, prima del prepare) e a `cu:27068` | decadimento della massa | esito | **sì** |
| ★`DS4_G133_SEED_DYNAMIC` | `cu:27061` → `cuda_g133_bool_env` `cu:26056` **P3** | `0`/`1`; altro → `invalid %s=%s; expected 0 or 1` + **abort** | `=0` | solo con `G133=1` | semina dinamica | nessuna | **sì** |
| ★`DS4_G133_PROMOTE_BUDGET` | `cu:27063` **P6 [1,512]** | invalido → abort | unset (default 8) | solo con `G133=1` | promozioni per clock | esito | **sì** |
| ★`DS4_G133_ROTATOR_IO_TIMEOUT_S` / `DS4_G133_TRANSIENT_IO_TIMEOUT_S` | `double_env` **P6** | double con range | unset | solo con `G133=1` | timeout I/O rotator/transient | esito | **sì** |
| ★`DS4_G73_OPEN` | `cu:1388` **P3 cached** | `0`/`1`; altro → warn `invalid DS4_G73_OPEN=%s; expected 0 or 1` + OFF | `=0` | **la costellazione completa** (C4 `cu:26963`) + ambiente ermetico (C4b `cu:1478`) | dispatch `routed_moe_launch_impl<true>` | esito — `[g73-open] terminal ready slot_bytes=%llu device_slots=%u …` `cu:7400` (ma `acquisition=`/`source=` sono letterali) | **sì** |
| `DS4_G73_CONSERVATION_STRICT` | `cu:1402` **P3** | `0`/`1`; altro → warn + OFF | `=0` | nessuno | conservazione stretta | esito | **sì** |
| `DS4_G73_ALLOW_Q1_SIDECAR` | `cu:1478` **P1 strict-1** | solo `"1"` | entrambi | esenta 6 variabili di serving Q1 dal filtro ermetico | escape hatch E1 | intento (assenza del rifiuto) | **sì** |
| ★`DS4_G73_PAGEABLE_OVERFLOW_GB` | `cu:1591` → `ssd_wrap_double_env` `cu:1527` **P6 [0.0,64.0]** | double finito nel range; invalido → `[q1-0-ssd-wrap] result=failed reason=invalid-config name=%s value=%s` + **abort** | `=0` | solo con `G73_OPEN=1` `cu:1588` | chunk pageable oltre l'arena pinnata | esito | **sì** |

### 4.6 `DS4_Q1_0_*` / `DS4_Q1_*` / `DS4_IQ1_*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| `DS4_Q1_0_RESIDENT_ARENA` | **DUE parser** — `ds4.c:2457` **P3+(-1) fatale** / `cu:1354` **P1 silenzioso** | `0`/`1`; altro → **fatale in `ds4.c`**, ignorato in `cu` | `=0` | sidecar + `SELECTED_LOAD=1` (C17g); arena GB>0 (C18a); **arena esclusiva ⇒ tiering off** (C2) | arena Q1_0 residente | esito — `[q1-0-resident-arena] result=bound … iq2_vram_cache=q1-routes-bypass` `cu:7652` (**quel campo MENTE**) | **sì** |
| `DS4_Q1_0_DUAL_ARENA` | idem (`ds4.c:2467` / `cu:1359`) | idem | `=0` | richiede `RESIDENT_ARENA` (C17a) e `CUDA_DYNAMIC_ARENA_GB>0` (C18c) | seconda arena per IQ2 esatto | esito | **sì** |
| `DS4_Q1_0_DUAL_SPARSE_COMPANION` | idem (`ds4.c:2477` / `cu:1364`) | idem | `=0` | ⇔ `MIXED_COLD_ONE` (C17b/C17c) | companion sparse 5+1 | esito | **sì** |
| `DS4_Q1_0_MIXED_COLD_ONE` | idem (`ds4.c:2487` / `cu:1369`) | idem | `=0` | ⇔ `DUAL_SPARSE_COMPANION` | fixture 5 hot + 1 cold | esito | **sì** |
| `DS4_Q1_0_SNAPSHOT_BACKING` | idem (`ds4.c:2497` / `cu:1374`) | idem | `=0` | **esclude** `RESIDENT` e `DUAL` (C17e); richiede `ARENA_GB>0` (C18b) | backing a snapshot sparse | esito | **sì** |
| `DS4_Q1_0_PAGEABLE_OVERFLOW` | idem (`ds4.c:2507` / `cu:1379`) | idem | `=0` | `SNAPSHOT ∨ (RESIDENT ∧ DUAL)` (C17d) | overflow pageable | esito | **sì** |
| `DS4_Q1_0_DYNAMIC_PROMOTION` | idem (`ds4.c:2517` / `cu:1384`) | idem | `=0` | **la costellazione completa** (C17f + C6 + C7a) | promozione dinamica esperti | esito | **sì** |
| `DS4_Q1_0_PROMOTION_SSD_WRAP` | `cu:1517` **P2-like con -1** | assente/vuoto/`"0"`→0; `"1"`→1; **altro → `[q1-0-ssd-wrap] result=failed reason=invalid-enabled value=%s` + `-1`** | `=0` | **`G73_OPEN=1` lo forza a 1 prima di leggere l'env** (`cu:1516`); altrimenti richiede `DYNAMIC_PROMOTION` (`cu:1583`) | rotazione SSD dell'arena | esito | **sì** |
| `DS4_Q1_0_PROMOTION_PROBATION_SLOTS` | `cu:26142` → `u32_env` **P6 [0,512]** | invalido → `-1` | `=0` | richiede `DYNAMIC_PROMOTION=1` (C7b `cu:26165`); deve eguagliare `RESERVE_SLOTS` | slot di probation | esito | **sì** |
| `DS4_IQ1_PROMOTION_PROBATION_SLOTS` | `cu:26132` **P6 [0,512]** | idem | `=0` | richiede mixed IQ1_S + costellazione (C5); **incompatibile con promotion Q1_0** (C7a) | probation IQ1 | esito | **sì** |
| `DS4_Q1_0_IQ2_PINNED_GIB` | `cu:1613` → `ssd_wrap_double_env` **P6 [0.125,64.0]** | invalido → `reason=invalid-config` + abort | unset (default 1.5) | solo se ssd-wrap attivo **e non** `G73_OPEN` | GiB pinnati per IQ2 | esito | **sì** |
| `DS4_Q1_0_SSD_WRAP_*` (QUEUE_CAPACITY, MAX_AGE_CALLS, `{PREFILL,DECODE}_WAVE_{COUNT,MIB}`) | `cuda_q1_0_ssd_wrap_u32_env` `cu:1549` **P6 con range** | invalido → `[q1-0-ssd-wrap] result=failed reason=invalid-config name=%s value=%s` + abort | unset | ssd-wrap attivo | parametri di trasporto SSD | esito | **sì** |
| `DS4_Q1_VRAM_LRU_SLOTS` | `cu:30329` **P6 [0,65535] + enum di spegnimento** | `0`/`off`/`false`/`disabled` → 0; interi ≤65535; **invalido → `invalid DS4_Q1_VRAM_LRU_SLOTS=%s; cache disabled` + 0** | `=0` (o `off`/`false`/`disabled`) | ⚠️ **`cuda_q1_0_resident_transport_requested()`** = `RESIDENT_ARENA ∨ SNAPSHOT_BACKING` (`cu:36992`). Senza, il parser **non viene chiamato** | cache LRU di slot Q1 in VRAM | **esito** — `[q1-vram-lru] result=summary requested=%u capacity=%u count=%u … hits=%llu misses=%llu` `cu:30360`, ma **solo se `DS4_Q1_0_PROFILE` è attiva** (`cu:30358`) | **sì** (il gate è puro env) |
| `DS4_Q1_VRAM_LRU_RESERVE_MB` | `cu:30352` → `cuda_parse_mib_env` `cu:3034` **P6 MiB** | testo non numerico → `present=0` → default 512 MiB, **silenzioso** | unset | stesso gate | reserve della LRU | esito | **sì** |
| `DS4_Q1_0_EXPERT_SIDECAR` | `ds4.c:19510` **P9 path** | path non vuoto; backend≠CUDA o modello baked → **fatale** (`ds4.c:19515`) | unset | esclude `IQ1_S_MIXED_COLD_K` (C19a) | sidecar GPTQ-Q1 | esito — `ds4: Q1_0 expert sidecar source: %s` `ds4.c:19525` | no (apre il file) |
| `DS4_Q1_0_SELECTED_LOAD` | `ds4.c:19537` **P1 strict-1** | solo `"1"` | entrambi | **obbligatorio** con qualunque arena Q1 (C17g) | caricamento selettivo esperti | intento | **sì** |
| `DS4_Q1_0_PROFILE` | `cu:3924` **P1/P5** | — | — | nessuno | **sblocca le righe di summary** di più sottosistemi Q1 | — | sì |
| `DS4_IQ1_S_MIXED_COLD_K` | `ds4.c:2527` **P1 strict-1** | solo `"1"` | entrambi | richiede sidecar IQ1_S (C19b); esclude sidecar Q1_0 (C19a); esclude `G73_OPEN` (C4) | fixture mixed IQ1_S | esito — `IQ1_S mixed decode fixture enabled: hot_main=5 cold_iq1=1 prefill=main` `ds4.c:19606` (**tutti e tre i campi sono letterali**) | **sì** |

### 4.7 `DS4_CUDA_DYNAMIC_ARENA*` e `DS4_CUDA_ARENA_WRAP_*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| ★`DS4_CUDA_DYNAMIC_ARENA_GB` | **TRE parser** — `cu:14612` **`atof()` nessuna validazione**; `ds4.c:20186` **`strtod(env,NULL)` nessuna validazione**; più i controlli a valle | `"abc"` → **0.0 in silenzio** = arena disattivata senza dirlo | `=0` | richiesto da `enforce` (C10), `DUAL` (C18c), `SNAPSHOT` (C18b) | dimensione dell'arena dinamica pinnata | **esito** — `CUDA registered %.2f GiB contiguous host window (of %.2f GiB model) …` `cu:14629` | **parziale** (una grafia errata è rilevabile staticamente) |
| `DS4_CUDA_DYNAMIC_ARENA_MIN_AVAILABLE_GIB` | `cu:10026` **P6 [0.0,64.0]** | invalido → warn `[arena-cap] invalid min_available_gib '%s'; dynamic arena disabled` + **-1.0** (disattiva l'arena!) | `=0` | nessuno | soglia di RAM disponibile | esito | **sì** |
| `DS4_CUDA_DYNAMIC_ARENA_OBSERVED_WINDOW` | `cu:8794` **P6 [1,256]** | invalido → `[arena-observe] invalid window '%s'; policy disabled` + **0** | `=0` | nessuno | finestra dell'observer | esito | **sì** |
| `DS4_CUDA_DYNAMIC_ARENA_OBSERVED_MIN_HITS` | `cu:8810` **P6 [1,256]** | invalido → warn + **fallback 1** | unset | richiede la finestra | soglia di hit | esito | **sì** |
| `DS4_CUDA_DYNAMIC_ARENA_GROW_INTERVAL` | `cu:8826` **P6 [1,256]** | invalido → `growth disabled` + 0 | `=0` | idem | cadenza di crescita | esito | **sì** |
| `DS4_CUDA_DYNAMIC_ARENA_CARRY_ACROSS_REQUESTS` | `cu:13686` **P3-like** | — | `=0` | arena legata | mantiene la residenza fra richieste | **esito parziale** — `[arena-carry] request=%llu mode=%s snapshot=%llu resident=%u lookup=%s observer=frozen` `cu:13754`; nel ramo *prime* (`cu:13746`) `snapshot=0 resident=0` sono **letterali** | no |
| ★`DS4_CUDA_ARENA_WRAP_SCHEDULE` | `cu:9996` **P8 enum, warn+fallback** | assente/`0`/`expert-major`; `source-parts`/`g44-source-parts`; **altro → warn UNA volta + fallback `expert-major`** (non aborta) | `=0` o `=expert-major` | `sequential/random file` richiedono `source-parts` (`cu:10945-10950`) | ordine di riempimento dell'arena | esito — `[arena-wrap-profile] … schedule=%s` `cu:10953` | **sì** |
| ★`DS4_CUDA_ARENA_WRAP_TRUST_WORKER_CHECKSUM` | `cu:10090` **P3 warn+disable** | `0`/`1`; altro → warn + OFF | `=0` | nessuno | si fida del checksum del worker | intento | **sì** |
| ★`DS4_CUDA_ARENA_WRAP_UNLOCK_SOURCE_RANGES` | `cu:10134` **P3 warn+disable** | idem | `=0` | nessuno | sblocca i range sorgente | intento | **sì** |
| ★`DS4_CUDA_ARENA_WRAP_UNLOCK_WAVE_GIB` | `cu:10745` **P6** | numerico con range | unset | richiede `UNLOCK_SOURCE_RANGES` | ampiezza dell'onda di unlock | esito | **sì** |
| `DS4_CUDA_ARENA_WRAP_SEQUENTIAL_FILE` / `_RANDOM_FILE` | `cu:10016` / `cu:10021` **P4 bool permissivo** | `"0"`/vuoto → OFF; **altro → ON** | `=0` | richiedono `SCHEDULE=source-parts` **e** il file valido, altrimenti `reason=file-source-requires-source-parts` / `file-source-unavailable` (`cu:10950`) | sorgente di lettura | esito | **sì** (la coerenza con SCHEDULE) |
| `DS4_CUDA_ARENA_WRAP_SEQUENTIAL_WORKERS` / `_FILE_QD` | `cu:10047` / `cu:10070` **P6** | range | unset | idem | parallelismo I/O | esito | **sì** |
| `DS4_CUDA_ARENA_WRAP_PART_PROFILE` / `_TRIM_BETWEEN_PHASES` / `_LAYOUT_PROFILE` / `_SLOW_PART_MS` | `cu:10105`/`cu:10120`/`cu:9526`/`cu:10765` **P3/P6** | `0`/`1` o numerico; altro → warn + disable | `=0` | `LAYOUT_PROFILE` richiede `source-parts` (`cu:10940`) | diagnostica di wrap | esito | **sì** |
| ★`DS4_REAP_PREFETCH_THREADS` | `cu:8843` **P6 [1,32]** | fuori range o testo → **fallback 8 silenzioso** | unset | nessuno | worker di prefetch | esito | **sì** |

### 4.8 `DS4_CUDA_KV_*`

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| `DS4_CUDA_KV_STAGED_RING` | `cu:6766` **P1 strict-1, cached** | solo `"1"` | entrambi | `!g_kv_staged_sticky_off` `cu:6773` | sposta le row-cache KV in un'unica arena host pinned a decode-start | esito | no |
| `DS4_CUDA_KV_MANAGED` | `cu:6839` **P1 strict-1, cached** | solo `"1"` | entrambi | ⚠️ **NON controlla l'allocazione**: `ds4_gpu_tensor_alloc` fa `cudaMallocManaged` **incondizionatamente** (`cu:6824`). Gate solo su due `cudaMemAdvise` (`cu:6855`, `cu:6860`), **che falliscono sempre su Windows/WDDM** | in teoria: preferred-location CPU + accessed-by device | **intento** — `[kv-managed] cudaMemAdvise unavailable (%s); managed KV stays migratable` `cu:6866`. La presenza del tag `[kv-managed]` prova che il ramo è passato, **non** che l'effetto ci sia — anzi, il messaggio dice l'opposto | **sì** (che sia un no-op) |

> Nota: `DS4_CUDA_KV_STAGED_RING_BYTES/SLOTS/TOPK_CAP/ARENA_ALIGN` (`cu:71-74`) **non sono variabili d'ambiente**: sono valori di un `enum` di compilazione. Il prefisso identico li fa sembrare env.

### 4.9 `DS4_METAL_*` — attenzione: **non sono morte su Windows**

Contro l'aspettativa: il prefisso `metal_graph_*` è **naming legacy**. Il blocco è protetto da `#ifndef DS4_NO_GPU` (`ds4.c:9233`…`16412`), **non** da `__APPLE__`. `DS4_NO_GPU` non è definito nel `CMakeLists.txt` del worktree, che compila `ds4_cuda.cu` e linka `ds4_cuda`. `ds4_backend_uses_graph()` (`ds4.c:91`) è vero anche per `DS4_BACKEND_CUDA`. **Quindi il "metal graph" È il percorso di generazione CUDA su questa macchina.**

**Eccezione**: 7 variabili sono raggiungibili solo dalle modalità CLI `--metal-graph-*-test`, che su build non-Apple abortiscono a `ds4.c:19646-19651` (`Metal backend requested but this build is linked with CUDA, not Metal`). Quelle sono **effettivamente morte**: `DS4_METAL_GRAPH_TRACE_LAYERS`, `_TEACHER_FORCE`, `_TRACE_STAGE_LAYER`, `_PROMPT_TOKENS`, `_DUMP_LOGITS`, `_TRACE_CACHE`, `_TRACE_COMP`.

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| `DS4_METAL_PREFILL_CHUNK` | `ds4.c:7293` **P6** `strtol`, clamp a `prompt_len` | `≤0` → nessun chunking; testo → **ignorato**, default 2048 | unset | nessuno | dimensione della ubatch di prefill | **esito** — `using chunked GPU prefill (%u-token chunks for %d prompt tokens)` `ds4.c:17655` | sì |
| `DS4_METAL_GRAPH_DUMP_PREFIX` | `ds4.c:10537` **P9 path + master-enable** | stringa non vuota; path non scrivibile → **ignorato senza messaggio** | **unset** (`=0` dà prefisso `"0"` e ATTIVA) | nessuno | dump binari dei tensori; ogni dump **sincronizza** | esito — `dumped %s layer %u pos %u to %s` `ds4.c:10572` | sì |
| `DS4_METAL_GRAPH_DUMP_NAME` | `ds4.c:10540` filtro `strstr` | substring; nessun match → zero dump silenzioso | **unset** (`=0` filtra via TUTTO) | richiede `DUMP_PREFIX` | restringe i tensori dumpati | esito | sì |
| `DS4_METAL_GRAPH_DUMP_LAYER` | `ds4.c:10543` **enum{all} ∪ numerico** | `all` o intero; testo → `strtoul`=0 → **solo layer 0** | unset o `=all` (`=0` seleziona il layer 0) | richiede `DUMP_PREFIX` | filtro layer | esito | sì |
| `DS4_METAL_GRAPH_DUMP_POS` | `ds4.c:10547` **P6** | testo → 0 → **solo pos 0** | unset | richiede `DUMP_PREFIX` | filtro posizione | esito | sì |
| `DS4_METAL_DISABLE_HC_FUSION`, `_KV_FUSION`, `_QKV_NORM_FUSION`, `_COMPRESSOR_PAIR_PROJ`, `_HC_NORM_FUSION`, `_SHARED_DOWN_HC_FUSION`, `_ATTN_OUT_HC_FUSION` | helper `ds4.c:11013` **P4 bool permissivo, latch `static`** | tutto tranne `"0"`/`""` → ON | `entrambi` | `_HC_NORM_FUSION` ridondante se `_HC_FUSION` già ON (`ds4.c:11194`) | disattivano i kernel fusi | **nessuna** | sì |
| `DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION` | `ds4.c:11778` **P5 presenza** — ⚠️ **NON usa l'helper come i fratelli** | qualunque valore, **`=0` ATTIVA la disabilitazione** | **unset** | `!g->quality` `ds4.c:11777` | separa gate/up dallo SwiGLU | nessuna | sì |
| `DS4_METAL_DECODE_STAGE_PROFILE` | `ds4.c:11184` **P5** | **`=0` ATTIVA** | **unset** | nessuno | **serializza la pipeline** (chiude/riapre il command buffer per stadio): altera i tempi che misura | esito — `metal layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms` `ds4.c:13313` (**`tokens` è il letterale `1`**, vedi §5) | no |
| `DS4_METAL_INDEXER_STAGE_PROFILE` | `ds4.c:11337`, `ds4.c:13363` **P5** | **`=0` ATTIVA** | **unset** | prefill: molti stadi richiedono `ratio==4` (`ds4.c:14222`) | profila l'indexer sincronizzando | esito — `ds4.c:13287` | no |
| `DS4_METAL_LAYER_STAGE_PROFILE` / `_Q_STAGE_PROFILE` / `_GRAPH_TOKEN_PROFILE` / `_GRAPH_PREFILL_PROFILE` | `ds4.c:13364`/`13365`/`14952`/`15418` **P5** | **`=0` ATTIVA** | **unset** | nessuno | profili di stadio | esito | no |
| `DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE` | `ds4.c:15410` **P5** | **`=0` ATTIVA** | **unset** | **implica** `split_commands=true` (`ds4.c:15417`) | ⚠️ non solo osserva: **cambia la struttura di esecuzione** (un command buffer per attn/ffn per layer) | esito `ds4.c:15559`, ma il campo `execute` dell'embed è finto (§5) | no |
| `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS` | `ds4.c:13015` **P6 [0,43]** | fuori range/testo → **ignorato**, default 4 | `=0` (disabilita il flush intermedio) | `allow_split_flush` `ds4.c:12990` | punto di flush del command buffer | nessuna | sì |
| `DS4_METAL_GPU_BATCH_EMBED_MIN` | `ds4.c:13208` **P6** | testo → **ignorato**, default 512 | **unset** (`=0` fa l'opposto: forza SEMPRE l'embed GPU) | `tokens != NULL` | soglia CPU/GPU per l'embedding | nessuna | sì |
| `DS4_METAL_NO_PREFILL_KERNEL_WARMUP` | `ds4.c:13241` **P5** | **`=0` ATTIVA** | **unset** | anche `static bool warmed` e `n_tokens<=8` cortocircuitano | salta il warm-up prima del prefill misurato | nessuna | sì |
| `DS4_METAL_GRAPH_OUTPUT_ROW` | `ds4.c:15448`, `15603` **P6 [0,n_tokens-1]** | fuori range → **ignorato**, default `n_tokens-1` | **unset** (`=0` seleziona il primo token) | `ds4.c:15448` richiede `!split_commands` | ⚠️ **cambia quale riga alimenta l'output head → cambia i logit** | **nessuna** (la riga scelta non è mai stampata) | sì |
| `DS4_METAL_GRAPH_RAW_CAP` | `ds4.c:16143` **P6** clamp `[raw_window, min(ctx,8192)]` | `≤0`/testo → **ignorato**, default calcolato | entrambi | nessuno | righe della KV cache raw (SWA window) → **cambia l'allocazione VRAM** | nessuna | sì |
| `DS4_METAL_RESUME_PREFILL_MIN` | `ds4.c:16168` **P6** | `≤0` → `UINT32_MAX` = feature spenta; testo → default 32 | `=0` | checkpoint di sessione valido `ds4.c:20522` | soglia per il resume-prefill | nessuna | no |
| `DS4_METAL_MEMORY_REPORT` | `ds4.c:17674` **P5** | **`=0` ATTIVA** | **unset** | il sito `ds4.c:16276` è morto (prompt test) | report VRAM a 3 checkpoint | **esito** — `CUDA memory report %s: free %.2f MiB total %.2f MiB` `cu:15226` | no |
| `DS4_METAL_DUMP_PREFILL_LOGITS` | `ds4.c:17696` **P9 path** | path non vuoto; **scrittura fallita → `return 1` che propaga e aborta la generazione, senza alcun messaggio** | **unset** | prefill riuscito | scrive i logit di fine prefill | esito — `wrote GPU prefill logits to %s` `ds4.c:17703` | sì (il contratto di parsing) |

### 4.10 Resto della famiglia `DS4_CUDA_*` (weight cache, model load, kernel switch)

Osservazione strutturale che vale per **quasi tutta** questa famiglia: sono **P5 (presenza)**. Cioè **`=0` le ATTIVA**. È il gruppo più pericoloso del runtime perché contiene una trentina di `NO_*` e `DISABLE_*`, dove l'intuizione «`NO_X=0` significa non disattivare X» è **esattamente sbagliata**.

| VAR | parser | valori / invalido | disattiva | gate | effetto | riga di prova | verif. statica |
|---|---|---|---|---|---|---|---|
| `DS4_CUDA_WEIGHT_CACHE` | `cu:2866`, `4285`, `6477` **P5** | **`=0` ATTIVA** | **unset** | nessuno | blocca HMM-direct + prefetch ATS + copia intera → forza la cache per-range | **nessuna** | no |
| `DS4_CUDA_WEIGHT_PRELOAD` | `cu:2867`, `4286`, `6478` **P5** | **`=0` ATTIVA** | **unset** | nessuno | identico a `WEIGHT_CACHE`. **Non esiste alcun codice di "preload"**: è solo un flag di soppressione | nessuna | no |
| `DS4_CUDA_WEIGHT_CACHE_VERBOSE` | 14 siti (`cu:2927`…`36217`) **P5** | **`=0` ATTIVA** | **unset** | nessuno | diagnostica cache/arena. **Effetto collaterale**: a `cu:4220` **disabilita la progress bar** di caricamento | esito | no |
| `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` | `cu:6144` **P7** | suffisso testuale accettato; parse fallito → `gb=0` → **`UINT64_MAX` = ILLIMITATO** | unset — ⚠️ **`=0` significa illimitato, non "off"** | nessuno | tetto ai byte cacheati | **nessuna** | sì (parsing) |
| `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB` | `cu:6156` **P7 clamp [256,8192]** | `"512xyz"` → 512; `0`/testo → **default 1792 silenzioso** | unset | può essere sovrascritto da `align-up(need)` `cu:6165` | chunk del `cudaMalloc` d'arena | esito (verbose) | sì |
| ★`DS4_CUDA_NO_Q8_F16_CACHE` | `cu:3159` **P5 invertita** | **`=0` NON riabilita la cache** | **unset** | `!g_quality_mode`, `!disabled_after_oom` | disattiva globalmente la cache Q8→F16 | esito in negativo (assenza di `cu:3264`) | sì |
| `DS4_CUDA_Q8_F16_CACHE_MB` | `cu:3048` → `cuda_parse_mib_env` `cu:3034` **P6 MiB** | suffisso o testo → `present=0` → **ignorato**, limite `UINT64_MAX` | `=0` (limite 0 ⇒ cache spenta) | nessuno | tetto della cache F16 | **esito** — `CUDA q8 fp16 cache %s; using q8 kernels (request=%.2f MiB cached=%.2f GiB limit=%.2f GiB …)` `cu:3092` | sì (parsing) |
| `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` | `cu:3054` **P6 MiB** | testo → default `max(5% total, 4096 MiB)` | **unset** (`=0` = riserva zero, più aggressiva) | nessuno | VRAM lasciata libera dalla cache F16 | esito | no |
| `DS4_CUDA_COPY_MODEL` | `cu:14569`, `4284`, `6476` **P5 non-vuota** | `""` è OFF; **`=0` ATTIVA**; con sparse bake → `return 0` che propaga (`cu:14570`) | **unset** o `=""` | `!g_sparse_bake_active`; `cudaMalloc(model_size)` | copia l'intera immagine modello in VRAM | **esito** — `CUDA model copy complete in %.3fs` `cu:14587` | no |
| `DS4_CUDA_COPY_MODEL_CHUNKED` | `cu:14667`/`14671` **P5** | **`=0` ATTIVA** | **unset** | `!g_sparse_bake_active`; su POSIX il fallimento fa fallback a prefetch HMM | copia a chunk | **esito** — `CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)` `cu:6554` | no |
| `DS4_CUDA_COPY_CHUNK_MIB` / `DS4_CUDA_MODEL_COPY_CHUNK_MB` | `cu:4363`/`4364` **P6 clamp [16,4096]** | invalido/0 → **default 64 MiB silenzioso** | entrambi | il secondo è alias legacy, letto **solo** se il primo è assente/vuoto | chunk di staging pinned | **nessuna** | sì |
| `DS4_CUDA_DIRECT_MODEL` | `cu:2870`, `6476` **P5** | **`=0` ATTIVA**; ⚠️ `""` è OFF a `cu:2871` ma ON a `cu:6476` (**incoerenza fra i due siti**) | **unset** | `!g_model_device_owned ∧ !g_model_registered` | serve i pesi come puntatore host diretto | nessuna | no |
| `DS4_CUDA_RELEASE_PREFILL_SCRATCH` | `cu:25577` **P1 strict-1** | solo `"1"` | entrambi | `g_cuda_tmp != NULL` | libera lo scratch a decode-start | **esito** — `[vram-ledger] phase=%s free_bytes=%llu … released_bytes=%llu release_status=%s` `cu:25557` | no |
| `DS4_CUDA_REAP_MASS_OBSERVE` / `_WRAP` | `cu:12104`/`12118` **P3 warn+disable** | `0`/`1`; altro → warn una volta + OFF | `=0` | **esclusi da `COMPOSE`** (C11 `cu:27195`); richiedono arena pronta | observer/policy di massa REAP | esito, ma la riga `cu:12621` ha `router=unbiased mask=off policy=…` **letterali** | no |
| `DS4_CUDA_REAP_MASS_WINDOW` | `cu:12132` **P6 [1,256]** | invalido → `observer disabled` + 0 | **unset** (`=0` è INVALIDO e spegne l'observer) | richiede OBSERVE o WRAP | lunghezza del ring | esito | sì |
| `DS4_CUDA_REAP_MASS_GROW_INTERVAL` | `cu:12147` **P6 [1,256]** | invalido → **abort dell'observer** | **unset** (`=0` invalido) | letta solo se `wrap` | cadenza di crescita | esito | sì |
| `DS4_CUDA_REAP_MASS_HYSTERESIS` | `cu:12163` **P6 [1.0,100.0]** | invalido → abort | **unset** (`=0` < 1.0 ⇒ invalido) | letta solo se `wrap` | isteresi | esito | sì |
| `DS4_CUDA_NO_TF32`, `_NO_Q8_DP4A`, `_NO_WARP_ROUTER_SELECT`, `_NO_PARALLEL_ROUTER_SELECT`, `_NO_Q8_BATCH_WARP`, `_NO_TOPK1024/2048/_CHUNKED`, `_NO_INDEXED_HEADS8`, `_NO_INDEXER_WMMA`, `_NO_INDEXER_DIRECT_ONE`, `_NO_CUBLAS_ATTENTION(_OUTPUT_A)`, `_NO_WINDOW_ATTENTION`, `_NO_F16_PAIR_MATMUL`, `_NO_ORDERED_F16_MATMUL`, `_NO_FD_CACHE`, `_NO_MODEL_COPY`, `_NO_MODEL_PREFETCH`, `_NO_DIRECT_IO`, `_NO_ATTENTION_OUTPUT_F16_CACHE`, `_NO_ATTN_Q_B_F16_CACHE`, `_NO_Q8_F32_CACHE` | tutte **P5 presenza invertita** | **`=0` NON riabilita** | **unset**, sempre | vari (dimensioni tensori, `n_tokens`, `g_quality_mode`, piattaforma) | disattivano un kernel/percorso e forzano un fallback | **quasi tutte: nessuna prova** | in gran parte no (dipendono da metadati/`n_tokens`) |
| `DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED`, `_DISABLE_Q8_HC_EXPAND_FUSED`, `_DISABLE_QKV_RMS_FUSED`, `_DISABLE_SHARED_GATE_UP_PAIR` | **P5** | **`=0` DISATTIVA il fuso** | **unset** | nessuno | spengono kernel fusi | nessuna | sì |
| `DS4_CUDA_Q8_F16_ALL`, `_Q8_F32_ALL`, `_Q8_F32_LARGE`, `_Q8_F32_PRELOAD`, `_ATTN_Q_B_F32_CACHE`, `_ATTENTION_OUTPUT_PRELOAD` | **P5** | **`=0` ATTIVA** | **unset** | i rispettivi `NO_*` hanno precedenza | forzano la cache dequantizzata | esito gated da `WEIGHT_CACHE_VERBOSE` | no |
| `DS4_CUDA_SERIAL_F16_MATMUL`, `_SERIAL_ROUTER`, `_INDEXED_TWOPASS`, `_WINDOW_ATTENTION`, `_EMBED_ROW_STAGING`★, `_KEEP_MODEL_PAGES`, `_MODEL_PREFETCH_SYNC`, `_MODEL_COPY_VERBOSE`, `_SEL_PROFILE` | **P5** | **`=0` ATTIVA** | **unset** | vari | vedi tabelle di dettaglio | miste | miste |
| `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN` | `cu:21701` **P6 [2,4095]** | fuori range/testo → **ignorato**, default 2 | entrambi | `!quality ∧ cublas_ready ∧ n_tokens≥min` | soglia token per cuBLAS su attn_output_a | nessuna | no |
| `DS4_CUDA_DUMP_LOGITS`, `_DUMP_PREFILL_LOGITS`, `_DUMP_TENSORS`, `_GRAPH_DUMP_PREFIX` | `cu:33801-33804`, lettura `cu:33807` **P5 non-vuota** | **`=0` ATTIVA** | **unset** | con `PREFILL_WAVES` e `n_tokens>1` → **fail-closed** | ⚠️ bloccano le prefill-waves. `_GRAPH_DUMP_PREFIX` in `ds4_cuda.cu` è **solo una sentinella**: nessun dump viene prodotto | **intento** — `[prefill-waves] refused reason=debug-dump-active layer=%u tokens=%u compact=%u` `cu:35367` (non dice quale) | no |

> **Nota di build**: `DS4_CUDA_ATTENTION_SCORE_CAP`, `DS4_CUDA_ATTENTION_RAW_SCORE_CAP`, `DS4_CUDA_TOPK_MERGE_GROUP`, `DS4_CUDA_UNUSED` (`cu:48-60`) e `DS4_CUDA_KV_STAGED_*` (`cu:71-74`) **non sono variabili d'ambiente**: sono `enum`/`#define`. Il prefisso identico le fa comparire in ogni grep ingenuo.

---

## 5. Letterali hardcoded nei log — campi che mentono

Il briefing ne segnalava 2. **Ne ho verificati 23**, tutti riletti sul sorgente. Sono ordinati per pericolosità: in cima quelli su una riga di **successo**, in mezzo a campi genuinamente misurati — la forma più ingannevole.

### 5.1 Categoria A — letterali su riga di SUCCESSO, fra campi misurati (i più pericolosi)

| # | riga | campo che mente | perché è grave |
|---|---|---|---|
| **L1** | `cu:12781` | **TUTTI**: `[prefill-mass-compose-mask] result=applied reason=ok layers=0 kept=0 pruned=0 semantics=request-scoped-open base=none existing_layers=0` | La `fprintf` **non ha nemmeno un argomento variadico**. Cinque contatori a zero su una riga `result=applied`. Non prova assolutamente nulla oltre al fatto che il ramo è stato preso. |
| **L2** | `cu:13016` | `preloaded=0`, `router=unbiased` | Riga di **successo** di `[prefill-mass-wrap]`, con 13 campi genuinamente misurati accanto (`candidate`, `loads`, `workers`, `seconds`, `snapshot_*`, `resident_*`, `generation`) e `mask=%s` calcolato. Solo `preloaded` e `router` sono costanti. Confronto: a `cu:11862` `[arena-observe]` stampa `preloaded=%u` **calcolato** — stessa parola, semantica opposta. |
| **L3** | `cu:12497` | `router=unbiased`, `mask=off` | Riga di **successo** di `[reap-mass-wrap]`, 13 campi reali. `mask=off` è affermato anche quando `DS4_REAP_MASK_FILE` è settata. |
| **L4** | `cu:12621` | `router=unbiased`, `mask=off`, `policy=free-then-mass-victim` | `[reap-mass-wrap] armed`: `grow_interval`, `hysteresis`, `capacity` sono reali; gli altri tre no. |
| **L5** | `cu:4578` | `iq2_vram_cache=q1-routes-bypass` | `[q1-0-sidecar] result=summary`: 11 contatori reali, questo campo è una stringa fissa che sembra uno stato di cache osservato. |
| **L6** | `cu:7652` | `iq2_vram_cache=q1-routes-bypass`, `backing=q1_0` | `[q1-0-resident-arena] result=bound`. Stesso campo, secondo sito. |
| **L7** | `cu:5595` | `native_h2d_bytes=0` | `[nested-residual-gpu-join] scratch-ready`: tre `%llu` reali (`max_base`, `max_residual`, `max_native`) e un quarto campo che sembra della stessa famiglia ma è la costante `0`. |
| **L8** | `cu:7400` | `acquisition=boot-only`, `source=dedicated-plain-fd`; inoltre `host_bounce_bytes` **riceve `slot_bytes`** (`cu:7403`), cioè duplica il primo campo | `[g73-open] terminal ready` — è la riga di conferma del modo G73. |
| **L9** | `cu:9168` | `storage=pageable`, `source=pread`, `router=unchanged` | `[q1-0-dual-sparse] result=staged`, in coda a `entries`/`bytes`/`workers`/`seconds`/`candidate_fnv1a64` reali. |
| **L10** | `cu:12851` | `prefill_mass=disabled snapshot=full-bootstrap router=unchanged dynamic_masks=not-implemented` | Quattro campi di stato, **zero argomenti**. |
| **L11** | `cu:12616` | `semantics=…`, `transport=packed-router-d2h` | `[reap-mass] armed`. `transport=` suggerisce un canale negoziato; è costante. |
| **L12** | `cu:25638` | `cache=disabled`, `fallback=exact-streaming` | `[expert-cache-grow] result=failed`: asserisce un percorso di fallback che in quel punto non è verificato. |
| **L13** | `cu:13746` | `snapshot=0 resident=0` | `[arena-carry] mode=prime`: nel ramo gemello (`cu:13754`) `resident` è `cuda_dynamic_arena_active_count()` **calcolato**. Qui è asserito. |

### 5.2 Categoria B — misure strutturalmente vuote (peggio di un letterale: sembrano numeri veri)

| # | riga | problema |
|---|---|---|
| **L14** | `ds4.c:15505-15506` | ```const double t_embed_encoded = profile ? now_sec() : 0.0;```<br>```const double t_embed_done    = profile ? now_sec() : 0.0;```<br>Due `now_sec()` consecutivi **senza alcun lavoro in mezzo**. Il campo `execute=%.3f ms` stampato a `ds4.c:15512` è per costruzione la differenza fra due letture di clock adiacenti: **non misura nulla**, ma è un `%.3f` che varia da run a run — indistinguibile da una misura vera. |
| **L15** | `ds4.c:11188` | `tokens=%u` riceve il **letterale `1`** dalla macro decode. Il campo stampato a `ds4.c:13313` sembra un conteggio osservato. |
| **L16** | `ds4.c:15844` | `chunk=%u` stampa `chunk_cap` (il cap *richiesto*, `ds4.c:15715`), non il chunk reale: i chunk effettivi sono ridotti dai clamp di boundary (`ds4.c:15742-15750`) e l'ultimo vero sta in `last_chunk_tokens` (`ds4.c:15751`), **mai stampato**. Stessa famiglia del modo di guasto 2 («richiesta ≠ ottenuto»). |
| **L17** | `cu:3112` | `cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit)` passa `free/total/reserve` come **zeri letterali**; la funzione li stamperebbe come `free=%.2f GiB reserve=%.2f GiB total=%.2f GiB` (`cu:3086`, `cu:3096`). Oggi è salvato dal test `free==0 && total==0 && reserve==0` a `cu:3075` che seleziona il ramo corto. **È una bomba a orologeria**: se quel test cambia, tre zeri diventano VRAM misurata. |
| **L18** | `ds4.c:17644` | `using GPU graph generation with layer-major graph prefill` è stampata **incondizionatamente all'ingresso**, prima che la strategia sia scelta. Se `prefill_cap < prompt->len` il codice esegue `metal_graph_prefill_chunked` (`ds4.c:17684`), **non** layer-major. La riga dichiara un percorso che può non essere quello preso. |

### 5.3 Categoria C — letterali su rami di fallimento (legittimi, ma da non confondere)

`cu:13061`, `13100`, `13113`, `13132`, `13327`, `13341`, `13359`, `13383`, `13398`, `13435` — tutte le varianti `[prefill-mass-wrap] result=failed` con `candidate=0 loads=0 workers=0 seconds=0.000 generation=0 preloaded=0 router=unbiased mask=off`. Sul ramo di fallimento gli zeri sono **veri** (non è stato fatto nulla), ma `seconds=0.000` accanto a `seconds=%.3f` altrove, e `router=`/`mask=`, restano letterali. Idem `cu:31063`/`31068` (`entries=0 bytes=0 … prior_mass=0` accanto a `seconds=%.3f` misurato), `cu:10431-10433` (9 contatori a zero nel ramo `result=unsupported` non-Windows), `cu:10953` (`begin=0.000 copy_checksum=0.000 finish=0.000 publish=0.000`), `cu:28045` (`hard_fault_source=external-sampler`), `cu:39825` (`mode=cold-one-upload-overlap`), `cu:40275` (`source=primary_iq2_pinned`), `ds4.c:19606` (`hot_main=5 cold_iq1=1 prefill=main`).

### 5.4 La regola da applicare d'ora in poi

> **Un campo `chiave=valore` in un log DS4 è prova solo se il `valore` è un `%`-specifier.**
> Prima di usare una riga come evidenza: apri la `fprintf`, conta i format specifier, conta gli argomenti. Se un campo non ha uno specifier, **non è stato calcolato**.
> Corollario: `iq2_vram_cache=`, `router=`, `mask=`, `preloaded=`, `storage=`, `source=`, `transport=`, `policy=`, `acquisition=`, `semantics=`, `fallback=` sono **sempre letterali** in questo codice. Non li ho mai trovati calcolati.

---

## 6. Fattibilità di `--validate-config`

Domanda: **quanto della validazione si può fare senza caricare il modello** — parsing + coerenza incrociata, poi uscita prima di allocare?

### 6.1 (a) Gate di puro env — verificabili a costo zero

Sono la maggioranza dei vincoli. Tutti e **34 i predicati** del [§3](#3-grafo-dei-vincoli-e-modi-validi) tranne tre dipendono **solo** da `getenv`, e la prova è che li ho già eseguiti su CPU in `modes.py` senza toccare il runtime.

| categoria | cosa si valida | quante variabili |
|---|---|---|
| **Sintassi del valore** | ogni parser P2/P3/P6/P7/P8 ha un dominio esplicito nel codice (range numerici, enum). Un validatore può replicarli 1:1 | 71 numeriche + 6 enum + 10 bool 0/1 |
| **Classe del parser** | `=0` è sicuro? **Qui sta il grosso del valore.** Un warning «`DS4_X` usa un parser a presenza: `=0` la ATTIVA; per disattivarla rimuovila» coprirebbe da solo **139 variabili su 273** | **139 P5** + ~30 con abort su valore fuori range |
| **Vincoli «richiede»** | C3, C4, C4b, C5, C6, C7a, C7b, C8a, C8b, C9, C14, C15b, C17a-g, C18a-c, C19a-b, C20a-b, C21a | 31 su 34 |
| **Vincoli «esclude»** | C15a, C16a, la parte env di C2 e C11 | — |
| **Inerzia** | tutte le righe di [§3.4](#34-archi-inerte-senza-abilitatore-la-variabile-viene-ignorata-non-rifiutata) il cui abilitatore è a sua volta una env (es. `Q1_VRAM_LRU_SLOTS` senza `RESIDENT_ARENA`/`SNAPSHOT_BACKING`) | ~12 |
| **Divergenza di parser** | le 7 variabili con contratto diverso fra `ds4.c` e `ds4_cuda.cu` ([§2 P10](#p10--la-classe-che-mancava-parser-divergente-fra-ds4c-e-ds4_cudacu)) | 7 |
| **Piattaforma** | `_WIN32` / `__linux__` / `O_DIRECT` sono note a compile-time: si può dire «questa variabile è inerte su questo build» | ~10 |
| **No-op noti** | `MOE_CACHE_POLICY≠layer-top1`, `KV_MANAGED` su WDDM, `WEIGHT_PRELOAD` (non esiste il preload) | 3+ |

### 6.2 (b) Gate che dipendono da stato runtime — NON simulabili

Solo **tre** vincoli, più i valori-soglia:

| vincolo | dipendenza | perché non si simula |
|---|---|---|
| **C10** `cu:27175-27182` — `enforce` richiede *un'arena pinnata idle e legata* (`host_base`, `n_layer==CUDA_MOE_LAYER_COUNT`, `n_expert==256`, `slots` non vuoti) | geometria del modello + allocazione riuscita | si può validare la **condizione necessaria** `DS4_CUDA_DYNAMIC_ARENA_GB>0`, non quella sufficiente |
| **C11** `cu:27184-27199` — `compose` richiede `snapshot_generation != 0`, observer prefill *finalizzato e pubblicato* | il prefill deve essere già girato | validabile solo la parte env (`PREFILL_MASS_OBSERVE`+`WRAP`, `mass-lfru`, no REAP) |
| **C2** `cu:26935` — `cuda_q1_0_exclusive_arena_active()` include `snapshot_generation != 0` | l'arena deve essere stata popolata | validabile la parte env (`(RESIDENT ∨ SNAPSHOT) ∧ ¬DUAL`) |
| *capacità* — non è un gate ma è il binding vero | `cudaMemGetInfo` | `STREAMING_EXPERT_CACHE_N`, `*_RESERVE_*`, `DYNAMIC_ARENA_GB`, `Q1_VRAM_LRU_SLOTS`: la richiesta è verificabile, l'**esito** no |
| *geometria del modello* | `n_expert`, `head_dim`, `in_dim/out_dim`, uniformità della geometria esperti (`cu:7457`, `cu:7610`) | serve almeno l'header del GGUF |
| *sidecar* | esistenza + sha256 + binding (`ds4.c:19510`, `cu:28334`) | serve aprire il file — **ma l'apertura di un sidecar è ordini di grandezza più economica del modello da 81 GB** |
| *contesto* | `DS4_G133_TIER` richiede `ctx ≤ CUDA_G133_EPOCH_POSITION_MASK` (`cu:26317`) | il ctx viene dalla CLI, quindi in realtà **è validabile** se il flag riceve anche gli argomenti CLI |

**Nota importante**: una parte di ciò che sembra runtime è in realtà *quasi statico*. La geometria esperti si legge dall'header GGUF senza mappare i pesi. Un `--validate-config` in due livelli (`--validate-config` = solo env; `--validate-config=deep` = env + header del modello + apertura sidecar) coprirebbe quasi tutto restando sotto il secondo.

### 6.3 (c) Dove inserire il punto di uscita

Il codice ha già la topologia giusta: **le funzioni `*_requested()` sono pure** (leggono `getenv` e ritornano, senza toccare CUDA). Questo è il fatto abilitante.

Tre candidati, in ordine di rapporto valore/rischio:

1. **`ds4_cli.c`, subito dopo il parsing degli argomenti e PRIMA di `ds4_engine_open`.**
   Qui si può chiamare l'intera batteria dei parser `q1_0_*_requested()` (`ds4.c:2457-2527`) e replicare il blocco di coerenza `ds4.c:19541-19570` + `ds4.c:20193-20240`. Copre tutti i vincoli C17*/C18*/C19*/C20a. **Costo: zero allocazioni, zero CUDA.**
   ⚠️ Limite: i parser `cuda_*_requested()` vivono in `ds4_cuda.cu` e non sono esportati. Servirebbe **una sola** nuova funzione `extern "C" int ds4_gpu_validate_env(void)`.

2. **`ds4_cuda.cu`, una `ds4_gpu_validate_env()` che esegue la parte pura di `cuda_moe_tiering_prepare` (`cu:26917-27100`).**
   Il blocco `cu:26921-27011` è **già interamente puro**: legge solo env e ritorna 0/-1. Basterebbe estrarne un `cuda_moe_tiering_validate_env()` chiamato sia dal nuovo flag sia (invariato) da `cuda_moe_tiering_prepare`. Copre C2(parte env)/C3/C4/C4b/C5/C6/C7/C8/C9/C11(parte env)/C12(parte env).
   Anche il blocco parametri `cu:27017-27100` (clock/budget/frequency/hysteresis/G133) è puro: valida ~14 numeriche con i loro range.

3. **`cuda_g73_validate_hermetic_environment()` (`cu:1420-1513`) è già esattamente questo**: una validazione di sole variabili d'ambiente, chiamata da `cuda_g73_route_dispatch_init` (`cu:3358`). **È il precedente da imitare** — il pattern esiste già nel codice.

**Raccomandazione**: punto 2, con firma `extern "C" int ds4_gpu_validate_env(int ctx_size)`, invocata dal punto 1. Il `ctx_size` permette di validare anche il limite G133.

### 6.4 (d) Quanti dei sei (ora sette) guasti sarebbero stati intercettati

| # | guasto | intercettato? | come |
|:-:|---|:-:|---|
| 1 | `PREFILL_TIER_COMPOSE=0` invalido | ✅ **sì** | puro parsing: il parser `cu:26073` è già puro, ritorna -1 senza toccare nulla. Il flag stamperebbe lo stesso messaggio in 50 ms invece che dopo 2-6 minuti di caricamento |
| 1b | `MOE_CACHE_POLICY=lru` no-op | ✅ **sì** | confronto statico col solo valore letto (`layer-top1`). Richiede una *whitelist* di valori significativi per gli enum-travestiti-da-P1 |
| 2 | `STREAMING_EXPERT_CACHE_N=160` non binding | ⚠️ **parziale** | non si può sapere `free_b` senza CUDA. **Ma**: (i) si può stampare la formula e il `per_expert` atteso data la geometria del modello (`--validate-config=deep`); (ii) si può **avvisare** che l'effetto collaterale è disabilitare il prefetch SPEX K1 — quello è puro env |
| 3 | `STREAM_RUNTIME_RESERVE_MB` si applica ma non risolve | ⚠️ **parziale** | il flag può dire «questa variabile è inerte finché `g_model_expert_cache_ready` non è 1» (gate noto, `cu:6217`), il che spiega *quando* si applica. Non può dire se risolverà il guasto — quella è una domanda sperimentale, non di configurazione |
| 4 | `KV_MANAGED` no-op per costruzione | ✅ **sì** | è una proprietà **statica** del codice: `cudaMallocManaged` incondizionato + `cudaMemAdvise` che fallisce su WDDM. Va nella tabella dei no-op noti per piattaforma |
| 5 | `Q1_VRAM_LRU_SLOTS` inerte | ✅ **sì** | il gate `cuda_q1_0_resident_transport_requested()` è **puro env** (`cu:2422`): `RESIDENT_ARENA ∨ SNAPSHOT_BACKING`. Verificabile a costo zero |
| 6 | letterali hardcoded nei log | ❌ **no** | non è un problema di configurazione. Va risolto **nel codice** (vedi sotto) |
| 7 | parser divergente `ds4.c` vs `ds4_cuda.cu` | ✅ **sì** | il validatore chiamerebbe **entrambi** i parser e confronterebbe: se divergono, la config è ambigua |

**Bilancio: 5 su 7 intercettati completamente, 2 parzialmente, 1 fuori scope.**
Tradotto in tempo: i guasti 1, 1b, 4, 5, 7 sono costati run da 2-6 minuti ciascuno e sarebbero stati diagnosticati in meno di un secondo.

### 6.5 Due interventi non-`--validate-config` con rapporto valore/costo migliore

1. **Uniformare i parser booleani.** Il costo di questo inventario nasce dall'esistenza di **9 classi** di parser per la stessa domanda «acceso o spento?». Un unico helper `ds4_env_bool(name, default)` con contratto `0`/`1`, warn + fallback su tutto il resto, eliminerebbe da solo le classi P1, P2, P3, P4, P5 e i modi di guasto 1 e 7. Le ~40 variabili **P5 (presenza)** sono la trappola più grande: `NO_X=0` che *attiva* `NO_X` è controintuitivo per chiunque.

2. **Vietare i campi letterali nei log strutturati.** Una regola meccanica (`grep -nE '"[^"]*[a-z_]+=[a-zA-Z0-9._-]+' | grep -v '=%'`) trova i 23 casi di [§5](#5-letterali-hardcoded-nei-log--campi-che-mentono) in un secondo. Se un campo non è calcolato, o si calcola o si toglie dal log. Finché restano, ogni riga di prova va letta con la `fprintf` accanto — che è esattamente il costo che questo documento cerca di eliminare.

---

## 7. Copertura di questo documento

| famiglia | variabili trattate | note |
|---|---:|---|
| `DS4_CUDA_STREAMING_EXPERT_CACHE_*` + `DS4_CUDA_MOE_*` | 13 | §4.1 |
| `DS4_CUDA_STREAM_*RESERVE*` | 3 | §4.2 |
| `DS4_CUDA_PREFILL_*` | 13 | §4.3 |
| `DS4_EXPERT_TIER*` | 9 | §4.4 |
| `DS4_G133_*` + `DS4_G73_*` | 12 | §4.5 |
| `DS4_Q1_0_*` / `DS4_Q1_*` / `DS4_IQ1_*` | 19 | §4.6 |
| `DS4_CUDA_DYNAMIC_ARENA*` + `DS4_CUDA_ARENA_WRAP_*` | 17 | §4.7 |
| `DS4_CUDA_KV_*` | 2 | §4.8 |
| `DS4_METAL_*` | 28 | §4.9 (35 totali, 7 morte su Windows) |
| resto `DS4_CUDA_*` | ~60 | §4.10 (le ~40 `NO_*`/`DISABLE_*` in blocco, essendo tutte P5 con lo stesso contratto) |
| **totale** | **~176** | |

**Universo totale**: **273** variabili uniche raggiunte da `getenv` diretto (numero verificato, coincide con il conteggio indipendente del coordinatore) + **72** raggiunte **solo** tramite gli helper indiretti (`cuda_moe_tiering_u32_env`, `cuda_moe_tiering_double_env`, `cuda_g133_bool_env`, `cuda_q1_0_ssd_wrap_{u32,double}_env`, `cuda_parse_mib_env`, `cuda_nested_residual_env_flag`, `cuda_iq1_s_env_flag`, `metal_graph_env_flag`) — che un `grep getenv` **non trova**. Su ~350 nomi `DS4_*` totali nei sorgenti, una manciata sono `enum`/`#define` e non variabili.

**Non ancora coperte** (bassa priorità, nessuna nel preset `g73_open.env.ps1`): `DS4_EXPERT_RECOVERY_*` (11), `DS4_NESTED_RESIDUAL_*` (~10), `DS4_SPEX_*`, `DS4_CPU_*`, `DS4_MTP_*`, `DS4_BAKE_*`, `DS4_BATCHED_*`, e le restanti `DS4_IQ1_S_*` di dettaglio. Per tutte vale comunque la [tassonomia §2](#2-tassonomia-dei-parser--la-tabella-che-serve-per-prima): la classe del parser si determina in 10 secondi guardando le 6 righe attorno al `getenv`.

---

## 8. Come usare questo documento

**Prima di lanciare un run**, tre controlli nell'ordine:

1. **Il modo.** La config appartiene a uno dei 33 modi di [§3.5](#35-il-catalogo-dei-33-modi-effettivi)? Se non ci si riconosce, il runtime la rifiuterà — e [§3.6](#36-verifica-degli-ancoraggi) dice con quale messaggio.
2. **Le variabili che si stanno muovendo sono LIBERE in quel modo?** Se non lo sono, si sta cambiando modo, non facendo un esperimento controllato.
3. **Ogni `=0` che si scrive: la variabile è P1/P3/P4?** Se è P5 (139 su 273) `=0` la accende; se finisce in un helper con range, `=0` può abortire l'avvio. Nel dubbio: `Remove-Item Env:\DS4_X`.

**Dopo il run**, prima di credere a una riga di log: contare i `%` nella `fprintf` ([§5.4](#54-la-regola-da-applicare-dora-in-poi)). `iq2_vram_cache=`, `router=`, `mask=`, `preloaded=`, `storage=`, `source=`, `transport=`, `policy=`, `acquisition=`, `semantics=`, `fallback=` non sono **mai** calcolati in questo codice.

