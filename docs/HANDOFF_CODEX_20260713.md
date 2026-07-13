# HANDOFF per Codex — 2026-07-13 (crediti Claude esauriti a metà V2)

Usa questo come prompt di ripresa. Lavora sul branch `spex-predictive-mask-study-2026-07-12`
del repo `imanu86/reap-loop`. Prima `git pull`, poi leggi:
- `docs/PUNTO_2026-07-12.md` (sintesi trasversale della giornata precedente)
- `docs/DS4_LEVE_CATALOG.md` **se esiste** (censimento leve — vedi §Thread 3, potrebbe essere assente)
- `runs/ds4/20260712_virtual_bake/RUN_LOG.md` + `arm_self60b_run1/` (il run che RENDE)

## STATO INFRA / SOLDI (verificato)
- **Pod RunPod `znec1r9osf2k74` TERMINATO** da me (era RUNNING orfano dopo la morte agenti; ~$2 totali spesi, di cui ~$1.6 idle). Nessun pod nostro acceso. Codex `99xyqm02gke4xg` intatto.
- **WSL crash root-cause TROVATA**: `.wslconfig` dà 62GB alla VM su host da 64GB → ~2GB a Windows → crash-loop `E_UNEXPECTED`. **DA FARE (utente)**: abbassare a ~56-57GB. La finestra bake60 (40.6 GiB) ci sta comunque.
- Crediti Claude esauriti (session limit reset 10:00 Roma; Fable reset 5:00). Gli agenti sono morti a metà.
- Disciplina kill VINCOLANTE: MAI `pkill` (self-match + ammazza server altrui). Solo `kill $(cat <run>/server.pid)`.
- GPU serializzata con `flock /tmp/ds4-gpu.lock`.

## IL QUADRO (dove siamo)
La velocità è capita e stabile; **la QUALITÀ è il problema, ed è risolta dalla LARGHEZZA giusta.**
- Tutte le maschere STRETTE (K8/K23/K32) collassano (word-salad). L'inseguimento controfattuale (braccio A) rende ma si dissolve verso K0 (unione 49.6%, churn no-decay). Il soft-mask (braccio B) contiene ma affama (skeleton 518 char).
- **BAKE60 RENDE** (`runs/ds4/20260712_virtual_bake/arm_self60b_run1`, commit `b0511ac`): mask STATICA che tiene il **top-60%/layer per MASSA** (154/256, dalla traccia K0 VERA `runs/ds4/20260711_k0_fullmodel_baseline/route_k0_cyberpunk.csv.gz`), finestra ~40GiB in RAM. Documento cyberpunk COMPLETO 11.7KB, chiude `</html>`, 0 tag-mismatch, JS funzionante, grade L2. **avg 1.95 t/s, regime 2.45-2.52** (path pread).
- Mass-coverage (offline, verificata): keep 55/60/65% perde solo 3-6% di massa held-out (sotto la soglia near-lossless 7.5%). Ma **cross-sotto-dominio FALLISCE** (coffee/json/python instradano su set disgiunti): il design è **UNA finestra PER-DOMINIO + switch dinamico per-prompt** (fattorino scalda la finestra al cambio dominio), non una finestra-famiglia unica.

## IL DESIGN EMERSO (la "doppia maschera")
- **Maschera ESTERNA** = bake statico 60% per dominio (selezione, near-lossless, non ruota mai in-run).
- **Promozione INTERNA** = la nostra macchina (pin-by-mass 0040/0042, pressione 0039, fattorino 0043) decide chi sta in VRAM (residenza, bit-exact, un miss costa ms non qualità). Arbitra il 15:1 VRAM(400)/finestra(~6160).
- Serve una modalità **rating-only** della livemask (osserva+pubblica massa per il pin, MA non scrive mai la mask esterna) — da cablare (env + guard).

## THREAD 1 — V2 ZERO-COPY (patch 0050) — LA PRIORITÀ, con diagnosi già avviata
Obiettivo: eliminare il pedaggio pread+staging+sync su ogni fetch RAM (tetto attuale 2.5 t/s → target 5-8). Concetto = chripell (fork MIT `github.com/chripell/ds4-rtx3090`, base 80ebbc3) ma sui SOLI range keep (~40GiB, l'intero 81 non entra in 62 RAM).
**DIAGNOSI CRITICA lasciata dall'ultimo agente opus (parti DA QUI):**
> Il fast-path zero-copy **non è MAI scattato**: `cuda_masked_pin_covers` ritornava false ogni volta → probabile **mismatch di identità del `model_map`** tra il momento della REGISTRAZIONE e quello dell'expert-load. Quindi lo 0.10 t/s osservato NON era nemmeno il path zero-copy. Inoltre: il "2× più veloce" di misure precedenti potrebbe essere un **confound di page-cache** (OFF girato cold-disk, ON warm) → servono run BACK-TO-BACK controllati.
> Prossimo passo che stava per fare: leggere i siti di registrazione (`ds4_gpu_set_model_map*`, `cuda_model_range_register_mapped` ~ds4_cuda.cu:1192) e di consumo (`cuda_model_copy_to_device_streamed` ~4845, chiamata da `cuda_stream_expert_cache_load_slot`) e verificare che il puntatore `model_map` registrato sia LO STESSO usato al load.
Cablaggio CORRETTO (variante A): la registrazione cambia solo la SORGENTE della copia H2D (host-registered invece dello staging buffer), il GEMM continua a leggere la cache VRAM. Variante B SBAGLIATA (=lo 0.10): dare al GEMM un device-pointer sulla RAM host → letture PCIe fini.
Lavoro in `/root/ds4-v2-work` (build md5 `c1635e74`, log `build_0050c.log`). NON toccare `/root/ds4-fullstack`. Artefatti attesi in `runs/ds4/20260712_v2_zerocopy/` (crea), patch `0050-stream-from-ram-masked.patch`.
Validazione: bit-exact ON==OFF (coffee, temp0, 60tok), poi A/B t/s con config bake60. Register a gradini 5→10→20→40GB (MemAvailable mai <7GiB).
REGOLA ABORT (utente): un run che parte lento e RAMPA è ok (freddi storici 0.2-0.6→2.5); un valore FUORI SCALA (0.1) o SENZA rampa dopo ~100 tok → ABORT + diagnosi, mai lasciar strisciare.

## THREAD 2 — MATRICE QUALITÀ n=3 (pod) — DA RIFARE
Il retest n=3 sul pod NON è mai andato in porto: campagna di ieri fallita in 5 min (0 char = OOM al primo token nell'envelope 12GB NATIVO — serve ~12.7GB; il nostro 3060-WSL regge perché il path mmap-register non è disponibile su WSL → footprint più magro). Decisione presa: girare a **24GB pieni** (token identici per l'invariante residenza≠selezione → i verdetti L0-L3 valgono per il 12GB; i t/s hanno caveat di regime). Il pod è terminato → **ri-provisionare** e rifare: `bake60_self×3` (run1 temp0 fail-fast, run2-3 temp0.7) → `bake65×3` → `family60×1` (atteso degradare) → `K0×1`. Modello da HF (hash sha256 `efc7ed607ff...e616668`, size 86720111488) o R2 (sdoganato, usa il più veloce). Masks in `runs/ds4/20260712_virtual_bake/masks/`.

## THREAD 3 — CENSIMENTO LEVE ds4 (workflow fallito, DA RIFARE) — richiesta esplicita utente
Motivazione utente: "decine di volte c'era una leva da tirare e non lo sapevamo perché non abbiamo scandagliato la doc del modello". Il workflow `ds4-variable-census` è fallito alla fase DISCOVER (session limit). Script: `...workflows/scripts/ds4-variable-census-wf_0319281f-003.js` (resumabile). Obiettivo: grep ESAUSTIVO di ogni `getenv`/flag CLI/`#define`-soglia in `/root/ds4-fullstack/{ds4.c,ds4_cuda.cu,ds4.h,ds4_gpu.h,ds4_ssd.c,Makefile}` + nostre patch → catalogo per categoria + **shortlist leve rilevanti MAI sfruttate** (velocità/memoria/qualità sul 3060/bake/zero-copy) + interazioni pericolose. Output → `docs/DS4_LEVE_CATALOG.md`. Leve già scoperte per caso da tenere presenti: `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB` (default 1792, min 256 — usare 256), il tier GB10/ATS del commit upstream `15f42aafd` (HBM-resident hot/cold), KV-su-disco (first-class).

## REGOLE DI PROTOCOLLO (adottate, vincolanti)
- Cap giusto (max_tokens ~4000+stream), stop SOLO su `</html>` / degenerazione oggettiva / budget. MAI cap corti.
- FAIL-FAST: run1 di 3 degenera → STOP braccio (niente 2/3). Run1 passa → completa tutti e 3 (run2-3 a temp 0.7 per potere statistico vero, temp0 è deterministico).
- WATCHDOG esterno su ogni run: stream_live.txt incrementale + server.pid; kill chirurgico sul PID; tag-salad/loop = kill, CSS lungo coerente = NON kill; in dubbio non killare.
- ABORT su anomalia velocità (sopra).
- Ogni run salva: prompt/request/env/commit/patch-chain/hash-modello/cache/ctx/output/log/motivo-stop. Dati da misure, non deduzioni.
- Micro-smoke può SCARTARE una policy rotta, mai PROMUOVERLA.

## CATENA PATCH (branch)
...0043 → 0044/0045 (SPEX Codex) → 0046-counterfactual-admission → 0047-no-whole-mmap-register → 0048-prefill-overlap-s1 (S1 validato -10%/-21% TTFT; S2 refutato su WSL2) → 0049-soft-breakthrough-log → 0050-stream-from-ram-masked (WIP, in diagnosi).

## PRIMA MOSSA CONSIGLIATA
Thread 1 (V2): verifica l'ipotesi `model_map` identity mismatch → è quasi certo il perché il fast-path non scatta. Aggiungi il contatore diagnostico (copy H2D calls/MiB nel path ON) per attribuire QUALUNQUE numero prima di misurarlo. Poi il t/s vero, con run back-to-back (no confound page-cache).
