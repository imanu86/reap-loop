# Domain-calibrated mask (cyberpunk-session) — does matching calibration to task cure the collapse?

**3060 locale, 2026-07-11.** Esperimento decisivo di chiusura campagna.

## Domanda
La campagna aveva stabilito (`20260711_highK_sweetspot`, commit be829c6) che **nessuna
mask-coffee (W50) rende il cyberpunk**: K48/64/91 collassano tutte a **L0** in un loop CSS
`body{background}`, senza mai raggiungere `<body>`, mentre la *stessa* mask rende il coffee
in-dominio. Verdetto di allora: il soffitto è il **match di dominio della calibrazione**, non
la magnitudine di K — ma "una mask calibrata sul dominio wide" **non era stata testata**.
Qui la si costruisce e la si testa: **una mask calibrata SUL cyberpunk cura il collasso?**

## Setup (regime pulito, IDENTICO al baseline highK)
- Bin `/root/ds4/ds4`, modello `/root/models/ds4-2bit.gguf` (86.7 GB), WSL sm_86, 3060 12GB.
- `--cuda --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 32 --nothink
  --temp 0.0` (greedy, **cache32**).
- Env: `DS4_CUDA_NO_Q8_F16_CACHE=1`, `DS4_CUDA_NO_DIRECT_IO=1`, `DS4_CUDA_KEEP_MODEL_PAGES=1`,
  `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1` (no cap 256->154, no abort reserve=16),
  `DS4_SPEX_STATS=1`.
- **Coesistenza:** `DS4_LOCK_FILE=/tmp/ds4_domain_calibration.lock` (proprio; ds4 è CLI puro,
  nessuna porta; UI:8000 mai toccata). GPU lasciata libera a fine lavoro.
- **Prompt tenuto COSTANTE** = `prompt_cyberpunk_wide.txt` del baseline (HTML-primed), così la
  matrice varia **solo la mask**, apples-to-apples col controllo coffee-mask. (La calibrazione K0
  usava il prompt cyberpunk BARE senza stub HTML: stesso task/dominio, lo stub semina solo lo
  scheletro.)

## STEP 1 — Costruzione mask calibrate-cyberpunk (offline, no GPU)
Sorgente = traccia K0 VERA non-mascherata `20260711_k0_fullmodel_baseline/route_k0_cyberpunk.csv.gz`
(159 960 righe, full model che HA reso il cyberpunk, **256/256 Expert**, per-layer 203-247 distinti).
Ranking **weighted (massa-gate)** per-layer sull'intera decode, top-K/layer — *stessa ricetta*
di `build_session_mask_canonical.py` usata per la W50-coffee, cambia solo la sorgente.

| mask | keep eff. | layer tappati <K | note |
|---|---|---|---|
| `sessCyber_K23` | **23.0** | 0/40 | pieno (vocabolario-cyber largo) |
| `sessCyber_K48` | **48.0** | 0/40 | pieno |
| `sessCyber_K64` | **64.0** | 0/40 | pieno |

A differenza della W50-coffee (tappata a ~75/layer dal vocabolario stretto), la calibrazione
cyberpunk vede >=203 esperti/layer -> i keep sono **genuinamente top-K**, mai tappati.

**Diversità di dominio (predittore di specificità):** cyber-K64 vs coffee-K64, per-layer:
intersezione mediana **33.5/64**, **Jaccard 0.36**; il **~48% degli esperti tenuti dalla
cyber-mask sono esperti che la coffee-mask POTA** (e viceversa). Sono due mask davvero diverse.

## STEP 2+3 — MATRICE confermativa (mask-dominio x task x [rende? grado? chiude? t/s])

### A. Velocità warm (probe -n300, warm-controllato: warmup scartato cold 1.14, poi back-to-back)
| mask x task | gen t/s | prefill t/s |
|---|---|---|
| sessCyber_K23 x cyberpunk | **3.44** | 0.50 |
| sessCyber_K48 x cyberpunk | **3.49** | 8.18 |
| sessCyber_K64 x cyberpunk | **2.87** | 8.53 |

**Piatta ~3 t/s, identica al baseline coffee (K48=3.60 / K64=3.06).** La velocità è
**indipendente da dominio E da K** (già noto: il driver è warmth+fit, non il numero di esperti).
Swap <=4 MB -> keep-set caldo in RAM. **La velocità non è mai stata la domanda**; lo è la qualità.

### B. CELLA CHIAVE — mask-cyberpunk x task-cyberpunk (full budget -n5500, ctx8192, trace ON)
| mask | grado | raggiunge `<body>`? | chiude `</html>`? | firma / dove muore | n |
|---|---|---|---|---|---|
| **sessCyber_K48** | **L1** | **SI (~char 7180)** | **NO** | rende TUTTA la pagina, poi loop di commento JS `// ma` | **n=2** |
| sessCyber_K64 | **L0** | no | no | glyph-loop corrotto DENTRO `<title>` (243->504 char), mai a `<style>`/`<body>` | n=1 |

**sessCyber_K48 SUPERA il punto-collasso CSS** dove ogni mask-coffee moriva. Rende, in ordine:
`<!DOCTYPE>` + head/meta/title + **CSS completo e VARIATO** (`</style>` chiuso: reset, container
neon `#00f0ff`, `grid-offerte` responsive auto-fit, card, popup-overlay, keyframes) + `<body>` +
hero + **card prodotti** + **`<div class="modulo-box">` modulo contatti** (nome/email/textarea/
button — *elemento richiesto*) + **popup modale** `#popupOver` "richiesta inviata al server cyber"
(*elemento richiesto*) + `<script>` con **3 `addEventListener` FUNZIONANTI** (`btnInvia`->
`e.preventDefault()`+show popup; `popupChiudi`->hide; overlay->close). JS ben formato
(`function(e){` corretto, vs il `(e) {` rotto della coffee-mask in-dominio).
**Poi**, nella coda JS, un commento `// ma resta...` **degenera in `// ma` x~90** e NON emette
`</script></body></html>`. Difetti minori (glyph emoji corrotti, typo `word-breut`/`CYERPUNK`).

**RIGORE n=2 (STEP 3):** re-run byte-per-byte **NON identico** (nondeterminismo greedy-streaming
noto, prefetch flippa i tie argmax: run2 diverge sui colori/valori CSS a metà) **MA
fenotipo-identico**: run2 attraversa il CSS coerente, raggiunge `<body>` (~char 7230), rende
lo *stesso* modulo contatti, e cade nell'**identico loop `// ma`** senza chiudere. Il residuo di
fase terminale è **stabile e riproducibile**, non un artefatto di un singolo run.

### C. SPECIFICITÀ — la mask-cyberpunk (K48, quella che RENDE) rompe il coffee?
| mask x task | grado | body? | chiude? | gen t/s |
|---|---|---|---|---|
| **sessCyber_K48 x coffee** | **L1-L2** | SI | **SI (`</html>`)** | 2.95 (stop naturale) |

**NO — non rompe il coffee: lo RENDE e lo CHIUDE.** Pagina coffee completa: colori coffee
(cream `#faf3e3`, brown `#8b5e3c`), nav Home/Menu/Contact, hero `<h1>Bean & Brew</h1>` + subhead,
`<button id="order">Order Now</button>` cablato con `addEventListener`+`alert`, `<form>` con
handler `preventDefault`+alert, **`</body></html>` chiuso** (rc=0, stop naturale). Difetti JS
minori (typo `Thank us`, `..addEventListener`). **La mask-cyberpunk chiude il coffee ma non
chiude il cyberpunk** -> la specificità **NON è simmetrica** (vedi verdetto b).

### D. CONTROLLO — coffee-mask x cyberpunk = collasso (baseline, ri-confermato)
Baseline diretto `20260711_highK_sweetspot/sweep/qual_K64`: **sessK64-coffee x cyberpunk = L0**,
loop CSS `background: radial-gradient.../#000`, gen 2928 char, mai `<body>`, no `</html>` — stesse
mask/config/giorno. Un re-run same-session è stato avviato ma **abbandonato**: dopo i run cyber
la page-cache era satura dei loro esperti, e caricare il working-set coffee ha innescato un
thrash I/O cold-stream (0.25 char/s) — artefatto di caching, non di qualità. Il controllo baseline
resta valido e non ambiguo.

## VERDETTO
**(a) La mask calibrata-cyberpunk rende il cyberpunk a >3 t/s?**
Velocità: **SI** (K23=3.44, K48=3.49 t/s warm; K64 2.87 dentro il rumore) — piatta, come sempre.
Qualità: **PARZIALMENTE SI, ed è il risultato grosso.** sessCyber_K48 **cura il collasso-CSS**
che uccideva OGNI mask-coffee: attraversa CSS+body+JS e rende una pagina cyberpunk **sostanziale
e funzionante** (modulo contatti + popup + 3 handler JS cablati) = **L1**, riproducibile n=2.
**MA non chiude `</html>`**: un loop di commento JS terminale (`// ma`) resta. E **K64-cyber
collassa del tutto** (L0, glyph-loop nel title) -> la resa statica è **fragile rispetto a K**.

**(b) È domain-specific (rompe il coffee)?**
**NO, non nel senso simmetrico ipotizzato.** La mask-cyberpunk che rende il cyberpunk **rende e
CHIUDE il coffee**. La specificità è **DIREZIONALE = copertura di calibrazione**, non un
fingerprint di dominio: la decode cyberpunk esercita 256/256 esperti (largo) -> il suo top-K
**sussume** il task-coffee stretto; la decode coffee (59-100/layer, stretto) **non sussume** il
task-cyberpunk largo -> collassa. `wide contiene narrow`, non `wide ortogonale narrow`. La
previsione naive "la cyber-mask rompe il coffee" è **refutata**; ciò che conta è che il keep
**contenga gli esperti che il task richiede**, cosa che una calibrazione larga garantisce per un
task stretto ma non viceversa.

**(c) Il domain-match statico BASTA, o resta residuo di collasso-di-fase (controller dinamico)?**
**NON basta da solo — serve ancora ammissione dinamica per l'ultimo miglio.** Il domain-match è
**necessario** (la coffee-mask non rende affatto il cyberpunk) e porta **quasi tutto il percorso**
(body + JS funzionante), ma **NON è sufficiente**: (i) residuo di fase terminale **riproducibile
(n=2)** — il loop `// ma` nella fase-commento JS, che una mask-cyberpunk *statica* non copre;
(ii) **fragilità rispetto a K** — K64 collassa dove K48 rende. Entrambi indicano che **le fasi
DENTRO il cyberpunk ruotano oltre ciò che una singola mask statica cattura**: la fase CSS/body/
early-JS è coperta (grande vittoria del domain-match), ma la fase terminale JS deriva -> **serve
il controller dinamico** (phase-admission) per chiudere in modo affidabile e robusto.

## Traccia mascherata-cyberpunk emessa (feed phase-seg cross-ref)
`route_maskedCyber_K48_cyberpunk.csv.gz` — 114 141 righe, 40 layer, **enforcement perfetto:
0.0000% pick su esperti potati** (684 846 pick, 0 violazioni); distinti-usati/layer = **48 = keep**
(i 6-di-6 si ridistribuiscono su TUTTO il keep). È la decode mascherata-cyber reale su cui
cross-referenziare le fasi (dove il loop `// ma` cade nella fase-JS terminale).
Anche `route_maskedCyber_K64_cyberpunk_COLLAPSED.csv.gz` (28 164 righe, il collasso-title).

## Artefatti
- `masks/sessCyber_K{23,48,64}.txt(.json)` — mask weighted massa-gate, sorgente cyberpunk K0.
- `sweep/` — probe warm K23/48/64 (t/s). `qual_K48_cyber/`, `qual_K64_cyber/`,
  `qual_K48_cyber_run2/` (n=2), `spec_K48cyber_coffee/`, `ctrl_K64coffee_cyber/` (gen+diag+mem).
- `route_maskedCyber_K48_cyberpunk.csv.gz` (+ K64 collassato). `progress.log`.
- Script: `run_one.sh`, `phaseA_probe.sh`.
