# REAP K91 coding — "quanto posso strizzare un 158B finché non entra in un 3060"

> Track `reap/k91-coding-vram`. Esperimento: una mask REAP **estrema K91** (tieni ~9% degli
> expert, 23/256 per layer) guidata dalla saliency del **dominio CODING** fa entrare il
> working-set del DeepSeek-V4-Flash 2-bit **interamente nella VRAM di una GPU 12GB** — e il
> modello sa ancora scrivere codice? **STATO: IN CORSO** (risultati sotto man mano).

## Ipotesi e matematica (da verificare, non ancora confermata)

- 12GB VRAM − backbone (~5.8 GiB atteso, **da misurare**) = ~6.2 GiB liberi → cache expert.
- K91 = keep **23/256** (`round(256*0.09)=23`), drop 233, sui **40 layer non-hash** (3..42).
  Layer 0-2 = hash routing, mai potati (design doc §0.2).
- Expert residenti a regime = 23 × 40 = **920 expert** × 6,75 MiB/expert (=7.077.888 B, geometria
  `runs/reap/gguf_flash_expert_geometry.txt`) = **6,06 GiB**.
- **"Entra in VRAM"** = PROVA che, con la bias-mask K91 attiva e cache VRAM ≤ budget 12GB, dopo
  warm-up `DS4_SPEX_STATS` mostra `direct_loads` a plateau (0 nuovi stream) e hit-rate → ~1.0.
  La bias-mask NON rimpicciolisce il file (86GB mmap): mette -1e9 sui bias dei potati → solo
  23/layer vengono MAI selezionati → resident-set effettivo ~6 GiB.

## Tre esiti possibili (tutti da riportare onestamente)

- **(A)** entra + codifica bene = risultato forte.
- **(B)** entra ma codifica male = K91 troppo aggressivo per un dominio largo (datapoint curva K-qualità).
- **(C)** non entra = tieni la matematica backbone.

## Contesto: curva K-qualità già misurata (a real structured-extraction task, NON coding)

Da `runs/reap/2026-07-05_eval_biasmask_v2/` (H200, ppl teacher-forced, CI95):
K50 dom **1.010×** (lossless), K67 **1.076×**, K70 ~1.11×, random **1.388×**. Nessun punto
oltre K70, nessun dominio coding, nessun target 12GB → **questo esperimento è nuovo su tutti e tre**.

## Setup (verificato)

- **GPU: RTX 3080 Ti 12288 MiB (sm_86)** = stessa architettura del RTX 3060 dell'utente, e VRAM
  reale 12GB → il tetto 12GB è **hardware vero, non un budget forzato** (risultato più forte).
- Pod RunPod community, dettagli/costi in `meta.json`.
- ds4 `80ebbc3` + patch **0001-0007**, `make cuda CUDA_ARCH=sm_86` → BUILD_OK,
  `make cuda-regression` → "cuda long-context regression: OK".
- Modello `DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf` (86.720.111.488 B, byte-identico al
  daily-driver locale), da HF `antirez/deepseek-v4-gguf`.
- Trace coding: **generata sul pod** dai 12 prompt `coding_en/c00..c11` (i CSV coding non
  esistevano in repo; solo i prompt) con `DS4_SPEX_TRACE_ROUTING`+`DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`
  (patch 0006).

## Findings operativi misurati (pod 3080Ti 12GB)

- **Backbone reserve = 5.82 GiB** (misurato: ds4 logga `reserve 5.82 GiB` nel sizing cache) →
  **conferma la stima del task (~5.8 GiB)**.
- **Il full model va in OOM con i default a 12GB**: ds4 prova cache 512→cap 398 expert, poi
  `CUDA model arena alloc failed for q8_0 (1792 MiB): out of memory` (le compute-arena q8_0/q8_hc
  non entrano oltre backbone+cache). Il full genera 1 token ("Here") e crolla.
- **Nuance importante per la matematica-fit**: la stima ingenua "12 − 5.8 = 6.2 GiB → ~910 expert"
  ignora le **compute-arena** (~1.8 GiB q8_0). Serve un budget cache esplicito. Config funzionante
  full-model: `--ssd-streaming-cache-experts 256 -c 2048` → gira pulito, **1.38 t/s**.
- Velocità full-model in streaming SSD (24GB RAM < 86GB modello): **~0.46-1.4 t/s**, NON trasferibile
<!-- redacted: internal cost/infra note -->

## Metodo (playbook §4)

1. Trace coding (12 prompt, greedy temp0, `--nothink`, `-n` fisso, weights) → CSV.
2. Saliency g-only (`reap_saliency_ds4.py`) → mask **K91** (keep-frac 0.09) + **K50-coding**
   (keep-frac 0.50, controllo qualità intermedio) + random control embedded.
3. Fit: bias-mask K91 applicata, `DS4_SPEX_STATS=1`, cache VRAM entro 12GB → hit-rate/direct_loads.
4. Qualità: prompt fisso sito web HTML+CSS+JS, stesso seed, config **K0 (full) / K50-coding / K91-coding**.
5. Report qui + `meta.json`.

## Risultati

_PENDING — compilati man mano che le misure arrivano dal pod._

### Fit (entra in VRAM?) — MISURATO

**Config di misura** (stesso prompt BST, `-n 90`, cache=380, `--ssd-streaming-cold --prefill-chunk 512 -c 2048`, `DS4_SPEX_STATS=1`). V0 meccanismo: `checked=5760 violations=0` (ogni expert selezionato ∈ keep-list K91).

| Config | keep/layer | working set | **hit_rate** | direct_loads | gen t/s |
|---|---|---|---:|---:|---:|
| **K0 full** | 256 | illimitato | **0.35** | 2670 (cresce) | 2.16 |
| **K91** (9%) | 23 | ~920 | **0.67** | 993 (≈plateau) | 3.67 |
| **K96** (3.5%) | 9 | ~360 | **0.96** | 631 | **12.02** |

**Capacità cache reale su 12GB = ~407 expert** (non ~910): backbone 5.82 GiB + arene `q8_0`
~1.8 GiB + ctx/CUDA overhead lasciano solo ~2.7 GiB. `≥440` va in OOM sull'arena. Indipendente
dal `prefill-chunk` (è backbone-bound, cap sempre ~407).

**VERDETTO FIT → esito (C) per K91, con datapoint costruttivo:**
- **K91 NON entra del tutto in 12GB reali**: working set 920 > cache 407 → hit_rate 0.67, non 1.0.
  La matematica del task (`12−5.8=6.2 GiB → 910 exp`) **ignorava ~2 GiB di arene/overhead ds4** →
  la cache vera è ~2× più piccola.
- **MA K91 funziona lo stesso come REAP**: bounda il working set (`direct_loads` plateau ~993 =
  920 unici caricati una volta, vs full 2670 **in crescita illimitata**) e va **~8× il full**.
- **Il K che ENTRA DAVVERO in un 3060 è ~K96** (keep 3.5%, working set 360 ≤ cache 407 →
  **hit_rate 0.96, 12 t/s = 33× il full**). Questo è il vero "quanto devi strizzare un 158B per un 3060".

### Qualità (siti web) — MISURATO

Stesso prompt (`sites/` → `website.txt`: nav + hero + form con validazione + toggle tema),
greedy `--temp 0`, `-n 2600`, config safe cache 64. **1 run generativa per config** (greedy CUDA
NON è bit-riproducibile, playbook §6 → trend chiaro, singolo campione). Output grezzi in `sites/`.

| Config | keep | render | tag reali | verdetto qualità |
|---|---|---|---|---|
| **K0 full** | 100% | ✅ **sito completo e pulito** (nav MySite+4 link, hero "Build Something Amazing"+CTA, form Name/Email/Message, toggle 🌙 Dark, footer) | 9 `<div>`, 0 pseudo | **codifica bene** — 0 errori console |
| **K50** | 50% | ⚠️ nav ok, **body degrada a pseudo-HTML** | 0 `<div>`, 5 pseudo (`<toggle>`,`<error>`,`<button "..`) | CSS valido ma markup rotto |
| **K91** | 9% | ❌ nessun HTML | 0 | **degenera**: ripete "The answer is a single-file HTML document..." all'infinito |
| **K96** | 3.5% | ❌ nessun HTML | 0 | **degenera**: `index_html` × 88 |

Render K0 verificato via snapshot DOM (nav+hero+form+toggle+footer tutti presenti, 0 errori console).
Degradazione **monotòna** con la potatura: K0 pulito → K50 degradato → K91 loop-prosa → K96 loop-token.

### VERDETTO A/B/C — onesto, entrambi i lati misurati

- **K91 (il target del task) = esito (C) + fallimento qualità**: NON entra del tutto in 12GB reali
  (hit_rate 0.67, working set 920 > cache 407) **E** non sa più scrivere codice (degenera in loop di
  meta-testo). Il peggio dei due mondi.
- **K96 (il K che ENTRA in 12GB) = esito (B)**: entra (hit_rate 0.96, 12 t/s) ma è **troppo potato per
  codificare** (degenera). Ciò che entra in un 3060, non codifica.
- **K0/K50 = esito "codifica ma non entra"**: K0 pulito, K50 già degradato — sul dominio **generativo
  coding** anche keep-50% inizia a rompersi (diverso dalla ppl del task strutturato lossless a K50: la coerenza
  generativa long-form è più fragile della ppl su dominio stretto).

**Conclusione (il "quanto strizzo un 158B per un 3060"):** per il DeepSeek-V4-Flash 2-bit sul dominio
**largo del coding generativo**, **non esiste un K che entri in 12GB reali E scriva ancora codice**.
C'è un muro netto: quello che entra (~K96) è già oltre il precipizio di coerenza; K91 non entra e non
codifica. La causa strutturale doppia: (1) le arene ds4 lasciano solo ~407 expert di cache reale
(non 910), (2) il coding è un dominio largo → intollerante alla potatura estrema (a differenza del
dominio stretto (a real structured-extraction task) dove K67 gira e K50 è lossless in ppl). K91 è nella terra di nessuno.
