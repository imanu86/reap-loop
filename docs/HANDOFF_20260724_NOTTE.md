# HANDOFF — sessione 23→24 luglio 2026 (sera/notte)

*Continua da `HANDOFF_20260723_COMPLETE.md`. Branch: `research/ds4-iq1-subbit-tier-planner` (pushato, `b040302`).
Rami runtime nel bare `D:\ds4_work\m1-admin.git`: `g73-q1serve`, `g73-promo0033`, `c7-capture-alllayer`.*

---

## 1. IN UNA FRASE

Il gate qualità di Goal A è **PASSATO** (Q1 vs base, rubric cieco), la cattura di calibrazione è passata da 1 layer a **40/40**,
e sul fronte serving abbiamo scoperto che **tutte le misure big-ctx del progetto erano fatte con la KV deliberatamente
parcheggiata in RAM host** — ma il rimedio architetturale (KV-ring) atterra sullo stesso numero del difetto (0.55 t/s),
quindi il collo di ctx8192 **non è il trasporto KV** ed è tuttora non attribuito.

---

## 2. RISULTATI SOLIDI (con numeri)

### GATE QUALITÀ Q1 — PASSATO (chiude il blocker #1 di Goal A)
Rubric cieco, 4 prompt × 3 seed × 2 modi, temp 0.7, stack identico, `DS4_CUDA_NO_Q8_F16_CACHE=1`,
grading su file anonimizzati (mapping sigillato in `rubric_20260723_135650/anon_map.json`):

| | L0 (degenere) | L1 | L2 |
|---|---|---|---|
| **base 2-bit** | 1 | 2 | 9 |
| **Q1-L15** | **0** | **0** | **12** |

Nessuna caduta di mediana su nessun prompt; le uniche degenerazioni sono nel **base**. La soglia 0.80-cosine
— l'assunzione non validata a più alto rischio dell'handoff precedente — **regge alla prova di output**.
*Caveat*: `max_tokens=200` troncava il reasoning di entrambi i bracci (disegno mio); confronto paired valido.

### Perplessità (riprodotta pulita)
BASE 3.013584 → Q1-L15 3.151018 = **+4.56%**. Riprodotto alla sesta cifra con igiene e assert anti-contaminazione.
**Attribuzione onesta**: il sidecar L15 è 161 esperti GPTQ + ~95 **naive-Q1** (L15 ha 256 esperti; il converter
ri-quantizza rozzamente i non listati, `gptq_to_sidecar.py:8-10`) → il +4.56% misura la **miscela**, non il GPTQ puro.
La catena F completa **non sposta la ppl di un bit** (base fedele identico a 6 decimali) → la residenza è bit-exact anche qui.

### Velocità — i numeri che mancavano
- **base-G73 ctx768 = 4.24 t/s** (steady, n=140): il numero base pulito che il progetto non aveva mai avuto.
- **q1-G73 = 0.20 t/s (−95%)**, ma **q1-plain = 0.48 ≈ base-plain 0.48**: a stack pari **il formato Q1 costa zero**.
  Tutta la penalità era integrazione col trasporto, non il formato.
- Colpevole trovato e corretto: `cuda_q1_0_route_arena()` non restituiva **mai** l'arena resident pubblicata
  (il bootstrap dei 256 esperti completava, il consumatore era irraggiungibile). Fix `2318753`, provato sul 3060:
  bootstrap 256/256 pinned 905MB, `route_pread=disabled`.

### AUDIT #0 — CPU vince, Colibri validato al microbench
| | per-esperto |
|---|---|
| CPU Q1 dequant+GEMV @4 thread | **294 µs** |
| GPU H2D+GEMV, base 2-bit | 766 µs |
| GPU H2D+GEMV, Q1 | 448 µs |
| solo H2D (base) | 293 µs |

Il **solo trasferimento** costa quanto l'intero calcolo CPU → il transient H2D non può vincere. `tools/audit0_microbench/`.

### Campagna calibrazione: da 1 layer a 40
Cattura all-layer implementata (sentinel `LAYER=43`) + **fix SIGSEGV Linux** (`c2d536b`: 8.5MiB di stato
materializzati sullo stack a ogni reset, anche a tracing spento — MSVC perdonava, gcc no).

Stato dati **al momento di questo handoff** (round-3 in corso):
- round-1 (22k/layer) + round-2 parziale = **~51k vettori/layer su 40/40 layer**, backup su `D:\ds4_work\pod_captures_*`
- round-3 in cattura per riportare tutti i layer a **≥77.8k** (soglia pilota)

---

## 3. IL FRONTE SERVING — cosa abbiamo scoperto e cosa resta aperto

### La KV era parcheggiata in host per configurazione
`DS4_CUDA_KV_MANAGED=1` (in **tutte** le catene di test, quindi in tutte le misure storiche) alloca la KV in
memoria managed. Il log stampava sempre:
```
[kv-managed] cudaMemAdvise unavailable (invalid device ordinal); managed KV stays migratable
```
Codex ha spiegato entrambi i pezzi: (a) Windows/WDDM riporta `concurrentManagedAccess=0` → `SetAccessedBy`
non è supportato (non è un ordinale sbagliato); (b) **l'advice di preferred-location seleziona esplicitamente la CPU**
→ la KV è host-backed *by design*, e le 64 teste MLA la rileggono dal PCIe.

### Matrice completa a ctx8192 (600 token, stessa richiesta)
| KV config | t/s |
|---|---|
| managed (default storico) | 0.56 |
| VRAM pura | **OOM** (10.7/12 GB, telemetria) |
| ring v0 | **crash** — illegal memory access |
| ring v1 (fix null-ptr) | 0.085 |
| **ring v2 (overlap riparato, 8M×3)** | **0.55** |

**Ring e managed convergono a ~0.55** ⇒ il trasporto KV **non è il collo**. Il ring resta l'architettura
necessaria per il regime 250k-1M (dove la KV non entra in VRAM), ma non spiega la lentezza a 8k.

### Due bug veri riparati nel ring (`9182f99`, `7baabee`)
1. Il quantizzatore FP8 partiva su `(float*)x->ptr` di un tensore migrato con `ptr==NULL` → accesso illegale asincrono.
2. Tre difetti di overlap: fence inutile; **le chiamate a un batch riusavano sempre lo slot 0** (double-buffering mai
   esistito); l'attention indicizzata sincronizzava il top-k su host e sparava **fino a 512 copie H2D sparse per layer**
   (×40 layer ≈ 20.000 microcopie/token). Sostituito con gathering device-side. Geometria ora da env.

### Il fix promotions funziona ma non basta (`2c1116a`)
`DS4_CUDA_DECODE_PROMOTION=1`: `vram_promotions>0` in 154/434 finestre (prima **0 sempre**), `vram_hit` da ~0% a 22-38%.
**Ma la velocità non si muove**: 0.56 → 0.59. E il numero che spiega perché: **media 2.3 promozioni/finestra,
tetto `PROMOTE_BUDGET=8`**, mentre ogni token seleziona **240 esperti**. Nel 64% delle finestre non promuove nulla.
La rotazione per massa esiste (`policy=mass-lfru`, knock 3/5, decay 0.98) ma è **strozzata a un rubinetto**.
*Manopole mai provate*: `PROMOTE_BUDGET` 8→64, `KNOCK_X/Y` 3/5→1/2.

### DOVE VA IL TEMPO A ctx8192 — non lo sappiamo
Attribuzione per-token: `decode_ms=1776`, `upload_sync_wait_ms=33` (1.9%), **`residual_ms=1776` = 100%**.
Tutti gli span strumentati sono a zero perché coprono solo il path **Q1-mixed**, non l'IQ2 standard.
Telemetria durante il decode: **GPU 29-39%, disco 0.5-92 MB/s, CPU 10-24%** — *niente è saturo*: firma
latency/serialization-bound. **Il prossimo passo obbligato è strumentare il path IQ2 standard.**

### Il diroppo ctx (misurato, non spiegato)
| ctx | t/s | KV in VRAM | slot cache |
|---|---|---|---|
| 768 | **4.91** | 75.9 MiB | 163/320 |
| 2048 | 0.59 | 207 MiB | 197/320 |
| 4096 | 0.59 | 263 MiB | 212/320 |
| 8192 | 0.56 | 333 MiB | 199/320 |

Ipotesi **tutte falsificate**: lunghezza generazione (a 600 token fissi: 4.91 vs 0.56), KV in byte, arena esperti
(identica 6546 a ogni ctx), slot cache (ctx8192 ne ha **di più**). Il salto 768→2048 resta **da spiegare**.

---

## 4. ERRORI DI QUESTA SESSIONE (perché non si ripetano)

1. **File toccati con i server vivi** → il fail-closed ha cancellato ~40% dei vettori round-2 dal pod.
   Un `pgrep` singolo aveva dato un falso 0. **Regola: mai rename/truncate su file di cattura; morte verificata
   con 3 check separati; la finalizzazione la fa il runtime.** (Il backup su D: ha salvato il 58%.)
2. **Conteggio vettori sbagliato di 6×**: le righe di `samples.jsonl` sono 6 per vettore (all-experts scrive
   1 vettore-input + 6 righe di routing). Il numero autorevole è `sample_count` nel manifest.
3. **Confound ctx/max_tokens**: variati insieme nella prima serie; corretto poi con 600 token fissi.
4. **Etichette bugiarde**: echo hardcoded `ctx=8192` mentre il flag era 4096; label `mode=promo` su un binario
   privo del fix promotions.
5. **`MSYS_NO_PATHCONV=1` obbligatorio** su `runpodctl create`: senza, `/workspace` diventa
   `C:/Program Files/Git/workspace` e **il container non parte** (tutti i pod creati da Git Bash nascevano morti).
6. **rclone → R2 via S3 API è strozzato** (0.36 MB/s, stalli al 95%): usare **URL presigned + curl -C -** (60-116 MB/s).

---

## 5. STATO OPERATIVO (aggiornato in corsa)

- **Pod 1** `ds4-camp-v2-155412` (`8740zkq3kxexjg`, 4×4090, **256 vCPU**, 1007GB RAM, $2.76/h): round-3 in cattura,
  prefisso `all43c_w*`, 4 worker all-layer, driver a **1800 token/richiesta**, target +26k/layer (56% alle 01:15).
  È anche la macchina designata per il GPTQ: ha modello, 256 core **e i vettori sul proprio volume** (zero trasferimenti).
  ⚠️ Il pod si era **stoppato per credito esaurito** e sembrava sparito dall'API: era solo in pausa,
  `runpodctl start` l'ha resuscitato **col volume intatto** (modello 86GB, build, captures round-1).
- **Pod 2** `ds4-cap-4090a3-migration` (`sxwbfmt49kw98k`): **SPENTO**. La sua rete verso R2 andava a **1 MB/s**
  (20h per il modello) ed era comunque ridondante — pod 1 ha il doppio dei core e i dati in locale.
- **Port teacher Linux CONSEGNATO** (`b040302`): `--backend {auto,msvc,gcc}`, equivalenza **provata byte-a-byte**
  su L15/e176, 0.241s/esperto di dequant. `tools/run_gptq_campaign_linux.sh` pronto (resumable, xargs-parallel).
- **Strumentazione path IQ2** — in lavorazione da Codex (`task-mrycpjlz`, ~26 min): è il blocco #3 dei prossimi passi.
- **Workflow "atlante delle leve"** — 10 agenti in parallelo sui sottosistemi del runtime (76k righe, 277 env var,
  51 test) → produce catalogo completo, leve mai tirate, matrice dei conflitti, e la sezione *caccia aperta* su
  ciò che scala col ctx. Run `wf_000bc1c4-494`.
- **Harness raccolta profilo pronto**: `iq2_profile_collect.sh` + `iq2_profile_sweep.sh`. Strategia scelta:
  **molte richieste corte** (40×64 token ≈ 2.500 token profilati per sessione) invece di poche corse lunghe →
  potenza statistica vera, e chiude il problema "n=3 senza potere" dell'handoff precedente. Sweep minimo:
  **8192, 768, 1200** (il plateau 2048/4096/8192 è già noto, misurarlo tutto sarebbe ridondanza pagata).
- **3060**: libero, in attesa della build strumentata.

---

## 6. PROSSIMI PASSI, IN ORDINE

1. **Finire round-3** → tutti i layer ≥77.8k → finalizzare **col runtime** (non a mano).
2. **Campagna GPTQ full-model** sui 128 vCPU del pod 2 (`run_gptq_campaign_linux.sh`), poi sidecar multi-layer
   (`gptq_to_sidecar.py --layer-spec`, `dc77c2d`) → **ppl full-model = il gate Step-1 vero**.
   ⚠️ Vincolo noto: il binder pretende il range contiguo → un sidecar joint su layer sparsi implica riempimento
   naive dei layer in mezzo (3..39 = 33.5GB, 33 layer naive). Con la copertura su **tutti** i 40 layer il problema sparisce.
3. **Strumentare il path IQ2 standard** (i 1776ms non attribuiti) — è il blocco che impedisce qualunque
   ottimizzazione ragionata del big-ctx.
4. **Provare le manopole della rotazione**: `PROMOTE_BUDGET` 8→64, `KNOCK` 3/5→1/2.
5. Spiegare il diroppo 768→2048 (la sola ipotesi non ancora testata: cosa cambia nella composizione arena/seed).
