# 2026-07-11 — K0 FULL-model NO-mask baseline (positive control + TRUE unmasked route trace)

**Doppio scopo.** (1) Controllo positivo = *soffitto di qualità* del modello pieno,
DS4_PACE=0, nessuna reap-mask, tutti i 256 Expert eleggibili, routing nativo top-6.
(2) Emettere la **traccia di routing VERA non-mascherata** (quali Expert sceglie il
router per-token per-layer) come **ground-truth** per l'analisi phase-segmented —
rimuove il *caveat-proxy* di `runs/ds4/20260711_pin_viability_and_gaps` (che stimava
la distribuzione d'uso su trace full **filtrati ai keep**, un proxy della decode
mascherata; qui la trace è la decode reale, unbiased).

## Setup

- **Pod:** RunPod `0htxln87674tjq` (pod4-worker-canonical-v2), **RTX 3090 Ti 24GB**
  (sm_86, come il 3060), driver 575.51.03, image cu1290, 220 GB RAM → **regime
  RAM-hot** (modello 81 GB interamente in page-cache; t/s NON confrontabili col 3060).
  Pod **RIPRESO** (non creato): era EXITED; al resume il disco container era vuoto
  (community volumeInGb=0 → il modello non sopravvive allo stop).
- **Gate-check PRE-download PASS:** `nvidia-smi -L` = RTX 3090 Ti;
  `torch.cuda.is_available()=True`, device_count=1.
- **Modello:** `ds4-2bit.gguf` (86 720 111 488 B) pull da R2 (`r2:ds4-models`),
  **sha256 `efc7ed60...616668` VERIFICATO** (`model.sha`). NB: il primo pull era stato
  corrotto da un kill prematuro (rclone `--multi-thread-streams` pre-alloca la size,
  quindi `stat -c%s` = target prima che i chunk siano scritti); re-pull con attesa
  dell'uscita pulita rclone (RC=0 = integrity-check ETag R2) + sha256 indipendente.
- **Binario:** `ds4_sm86_canonical-62ed2e71-v2` da R2 (CLI, sm_86; ds4.c md5
  `62ed2e71`, identico all'ultima catena livetree pace0028; base canonical committata).
- **Path pulito:** `DS4_CUDA_NO_Q8_F16_CACHE=1` (cache uniforme 2-bit — il path senza
  segnale di divergenza oltre il rumore, cfr. `20260711_local_clean_lowK/BITEXACT.md`).
- **K0 full no-mask:** nessun `DS4_REAP_MASK_FILE`, nessun `DS4_PACE` (=off). Il
  router gira top-6 su tutti i 256 → **unbiased**, la trace cattura l'intero vocabolario.

## Recipe (esatta, riproducibile — `run_k0.sh`)

    DS4_CUDA_NO_Q8_F16_CACHE=1 \
    DS4_SPEX_TRACE_ROUTING=route_k0_<name>.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
    /root/bin/ds4 -m ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
      --ssd-streaming-cache-experts 1024 -c <ctx> --nothink --temp 0 \
      -n <ntok> --prompt-file <prompt>

greedy temp0; cyberpunk: -n 4000 -c 6144; coffee: -n 1000 -c 3072.

## Risultati (n=1 per prompt; t/s = POD RAM-hot, DIAGNOSTICI, non trasferiscono)

| Prompt | tok | finish | L0-L3 | doctype | body | </html> | form | button | popup/alert | repeat | t/s prefill | t/s gen (pod) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **cyberpunk** (WIDE) | 4000 (cap) | length | **L1** | 1 | 1 | 0 | 1 | 2 | 1 (popup JS) | **0** | 6.34 | 4.09 |
| **coffee** (NARROW) | ~734 | **stop** | **L3** | 1 | 1 | **1** | 1 | 1 (wired) | 2 (alert) | **0** | 13.85 | 4.21 |

- **cyberpunk@4000 = L1, MA pulito e coerente, ZERO degenerazione** (repeat=0, CSS
  valido). Rende: DOCTYPE + head + CSS completo + `<body>` (hero, `<form>` contatti,
  2 bottoni, struttura popup + JS `getElementById/showPopup`), troncato a metà JS del
  popup dal budget. `<body>` arriva **~tok 3200** (CSS ancora più verboso del T1) → non
  chiude `</html>` entro 4000. **Identico fenotipo del controllo T1**
  (`20260710_pod_t1_full_positive_control`: full L0/L1 budget-bound, repeat=0 ovunque,
  chiude solo con budget ~3500+): **il FULL NON degenera** — il grado basso a budget
  fisso è *budget-confound*, non collasso. La firma LOOP/CSS-corrotto resta quindi
  attribuibile alle mask (il full non loopa mai).
- **coffee@1000 = L3, pagina completa e chiusa**: nav (Home/Menu/Contact), hero
  `<h1>Bean & Brew</h1>` + sottotitolo, `#order` cablato con addEventListener ->
  `alert("Thank you for your order!")`, `<form action="/submit">` con name/email/submit
  + onsubmit `preventDefault()` + alert di conferma, `</html>` chiuso. Zero difetti.

## Traccia di routing VERA (ground-truth)

Header CSV: `pos,layer,n,e0..e5,w0..w5` (per-token per-layer: 6 Expert selezionati +
6 gate-weights). **40 layer MoE tracciati (3..42)** — i layer densi 0-2 non instradano.

| file | righe | tok | Expert distinti totali | per-layer distinti (min/med/max) |
|---|---:|---:|---:|---|
| `route_k0_cyberpunk.csv.gz` | **159 961** | ~3720 gen (max_pos 4063) | **256 / 256** | 203 / 232 / 247 |
| `route_k0_coffee.csv.gz` | 31 601 | ~734 | (narrow) | — |

- **Tutti i 256 Expert compaiono** nella trace cyberpunk (router unbiased confermato);
  per-layer 203-247/256 attivi su 4000 tok → il full esercita l'intero set. Questa è la
  **decode reale non-mascherata**, non un proxy filtrato.
- File **gzippati** per snellezza repo; `gunzip` per riuso. Sono i CSV `route_k0_*.csv`
  richiesti (nome del deliverable rispettato, `.gz` solo storage).

## PRONTA PER PHASE-SEGMENTATION

`route_k0_cyberpunk.csv(.gz)` è la **traccia non-mascherata VERA** per il re-run
phase-segmented: sostituisce i trace full-filtrati-ai-keep usati come proxy in
`runs/ds4/20260711_pin_viability_and_gaps` (COMPITO 1 / GAP **G2**). Il caveat-proxy
è rimosso: la concentrazione d'uso e la segmentazione per fase possono ora essere
misurate sulla decode reale del full a budget largo (4000 tok, fase-larga CSS->body->JS).

## Costo / stato pod

- Pod **ripreso** (non creato) -> a fine lavoro **STOP -> EXITED** (stato in cui l'ho
  trovato; il modello NON persiste comunque su community). Spesa ~$0.27/h per la durata.
  Balance pre-lavoro $23.12.
- Artefatti: `gen_k0_*.txt` (output integrali), `diag_k0_*.txt` (t/s), `progress.log`,
  `model.sha`, `run_k0.sh`, prompt, `route_k0_*.csv.gz`.
