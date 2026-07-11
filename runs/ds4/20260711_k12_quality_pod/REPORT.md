# Does domain-matched K12 render? — min-K-that-renders = the speed/quality target

**Pod RunPod (RTX 3090, RAM-hot), 2026-07-11.** Domanda-qualità decisiva della campagna.

## Domanda
La ricetta-3060 punta a **K12** (keep-12/layer): il footprint più stretto che *sta* nei 12 GB reali.
La chiusura `20260711_domain_calibration` (commit d046a71) aveva mostrato che una mask calibrata SUL
cyberpunk **cura il collasso-CSS** che uccideva ogni mask-coffee: **sessCyber_K48 RENDE il cyberpunk
a L1** (arriva a `<body>`, non chiude `</html>`, loop-commento `// ma` terminale), mentre
**K64-cyber COLLASSA** (L0) → la resa è **fragile e NON-monotona rispetto a K**. Ma il K della
ricetta-3060 (K12) **non era mai stato testato in qualità**. Qui: **una mask domain-matched a K12
RENDE ancora il cyberpunk, o K12 è troppo stretto anche col match di dominio?** Decide se
"ricetta 3060 = K12" regge, o se serve K più alto / ammissione-di-fase.

## Setup (regime pulito, IDENTICO al riferimento d046a71)
- **Pod** `7qgalm9sasqnr7` (ds4-podD, **RTX 3090 24GB**, sm_86 come il 3060), **RIPRESO** (già
  RUNNING, idle post-smoke-0031): modello + binario già presenti, nessun download da 80 GB.
- **Gate-check PASS**: `nvidia-smi -L`=RTX 3090, `torch.cuda.is_available()=True`, count=1.
- **Modello** `/root/models/ds4-2bit.gguf` (86 720 111 488 B), **sha256 `efc7ed60…616668`
  RI-VERIFICATO indipendentemente** (`sha256sum -c` = OK). Binario `/root/ds4/ds4` (canonical).
- **Regime RAM-hot** (233 GB liberi, modello interamente page-cached): **t/s NON confrontabili col
  3060** (diagnostici) — ma **la QUALITÀ trasferisce tra GPU** (è la domanda). Il controllo K48 su
  QUESTO pod àncora il transfer al riferimento-3060.
- **Config** (esatta di d046a71): `--cuda --ssd-streaming --ssd-streaming-cold
  --ssd-streaming-cache-experts 32 -c 8192 --nothink --temp 0.0 -n 5500` (greedy, cache32).
  Env: `DS4_CUDA_NO_Q8_F16_CACHE=1` (path 2-bit pulito), `DS4_CUDA_NO_DIRECT_IO=1`,
  `DS4_CUDA_KEEP_MODEL_PAGES=1`, `DS4_SPEX_STATS=1`, `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1`,
  lock proprio (`/tmp/ds4_k12q.lock`, mai toccata la UI:8000). Codex pod `99xyqm02gke4xg` NON toccato.
- **Prompt COSTANTE** = `20260711_highK_sweetspot/prompt_cyberpunk_wide.txt` (HTML-primed), lo stesso
  del riferimento → la matrice varia **solo la mask**.

## STEP 1 — Costruzione mask domain-matched K12/K16 (offline, no GPU)
Sorgente = traccia K0 VERA non-mascherata `20260711_k0_fullmodel_baseline/route_k0_cyberpunk.csv.gz`
(159 960 righe, full che HA reso il cyberpunk, 256/256 esperti, 203-247 distinti/layer). Ranking
**weighted massa-gate** per-layer, top-K — **STESSA ricetta** di `build_session_mask_canonical.py`
usata per K23/K48/K64, cambia solo K.

| mask | keep eff. | layer pieni | note |
|---|---|---|---|
| `sessCyber_K12` | **12.0** | 40/40 | genuinamente top-12 (mai tappato: ≥203 visti/layer) |
| `sessCyber_K16` | **16.0** | 40/40 | genuinamente top-16 (bisezione, costruita ma non tirata) |

## STEP 2 — MATRICE (mask-dominio × cyberpunk × [rende? grado? body? chiude?])
Config identica; trace ON per enforcement. **n=2** su K12 (richiesto). K48 = controllo positivo
cross-hardware. Grader canonico `scripts/functional_grade.py frontpage` (node --check per il JS).

| run | mask | grado | `<body>`? | chiude `</html>`? | firma / dove muore | gen t/s (pod RAM-hot) |
|---|---|---|---|---|---|---|
| **K48_ctrl** | sessCyber_K48 | **L1** | **SÌ** | **NO** | rende TUTTA la pagina, poi loop `//` (commento JS) | 2.78 |
| **K12_r1** | sessCyber_K12 | **L1** | **SÌ (~char 700)** | **NO** | body+form+popup, JS rotto, poi loop-CSS `#popup .popup-content` | 2.70 |
| **K12_r2** | sessCyber_K12 | **L1** | **SÌ (~char 700)** | **NO** | **fenotipo IDENTICO a r1**: body+form+popup, **stesso loop-CSS terminale** | ~2.7 |

- **K48_ctrl = L1** (grader canonico): riproduce ESATTAMENTE il riferimento-3060 d046a71 (body sì,
  close no, loop terminale). **→ transfer di qualità cross-GPU confermato su questo pod**: ciò che
  rende sul 3060 rende qui, stesso fenotipo. (wall 2003s, prefill 4.04 / gen 2.78 t/s, 5499 tok.)
- **K12_r1 = L1 — è il risultato**: K12-domain-matched **SUPERA il punto-collasso-CSS** dove ogni
  mask-coffee moriva. Arriva a `<body>` **prestissimo (~char 700)** — stile **inline compatto** (non
  il grande `<style>` verboso del K48) — e rende, in ordine: `<!DOCTYPE>` + head/title/meta + `<body>`
  + `<header>`/`<h1>AI Programming Store</h1>` (hero) + sezione prodotti + **`<form id="contactForm">`
  completo** (input Nome, input Email, `<textarea>` Richiesta, `<button type="submit">Invio` —
  *elemento richiesto*) + **`<div id="popup">` modale "✓ Richiesta inviata"** (*elemento richiesto*)
  + `<script>`. **Enforcement mask PERFETTO**: distinti-usati/layer = **12 = keep** su tutti i 40
  layer (min=max=12.0, 0 violazioni). (wall 2050s, prefill 5.64 / gen 2.70 t/s, 5499 tok.)
- **K12_r2 = L1** (grader canonico, profilo identico a r1): **stesso fenotipo, riproducibile** —
  arriva a body ~char 700, rende lo stesso modulo contatti + popup + submit, cade nell'**identico
  loop-CSS terminale** `#popup .popup-content p:first-child { … }` senza chiudere. Il residuo di fase
  terminale è **stabile** (come il K48 in d046a71), non un artefatto di un singolo run. (Run interrotta
  a ~tok 4800 allo stop-pod — fenotipo già pienamente catturato: body+form+popup+loop; trace off.)

**GRADO FINALE: K12 = L1, n=2 (r1=L1, r2=L1) = K48-control (L1). K12_r2 CONFERMA L1.**
- **Difetti K12 (reali, il costo del keep stretto):** (i) **JS rotto** — dentro `<script>` mette
  regole CSS invece di JS, e l'unico "handler" è la riga-garbage `document.addEventListener:
  DOM/Full/Notify;` → il popup NON è cablato (**peggio del K48**, che aveva 3 `addEventListener`
  funzionanti); (ii) **tag di chiusura malformati** (`section>`, `footer>`, `main>`, `div>` senza
  `</`); (iii) **loop-CSS terminale** → **NON chiude `</html>`** (stesso esito-di-fase del K48).

## STEP 3 — Bisezione? NON necessaria
La bisezione (K16/K23) serviva **solo se K12 NON rendesse ma K48 sì**. **K12 RENDE (L1, n=2)** → il
minimo-K-che-rende domain-matched è **≤ 12**. E la velocità è **PIATTA vs K** (stabilito ripetutamente:
il driver è warmth+fit, non il numero di esperti) → nessun motivo-velocità per alzare K. Il K minimo
che rende **è anche il target-footprint 3060**. Mask K16 costruita e pronta, non tirata.

## VERDETTO (8 righe)
1. **K12-domain-matched RENDE? SÌ — L1, n=2.** Arriva a `<body>` (presto, ~char 700), **passa il
   punto-collasso-CSS**, rende **modulo contatti + popup "richiesta inviata"** (entrambi gli elementi
   richiesti). Stesso grado del K48-domain-matched.
2. **Chiude `</html>`? NO** — loop-CSS terminale, esattamente come il K48 (loop-commento). Nessuna
   delle due mask *statiche* chiude.
3. **Qual è il K minimo che rende? ≤ 12** (K12 rende → niente bisezione). Non 16, non 23: **12 basta**.
4. **Bersaglio-velocità/footprint realistico = K12.** "Ricetta 3060 = K12" **REGGE** al livello L1:
   il footprint più stretto rende la pagina cyberpunk (struttura + elementi richiesti).
5. **Velocità = FIT-IN-VRAM (correzione CLAIM-008, misura parallela sul 3060).** La t/s piatta vs K
   vista *qui* è **artefatto RAM-hot del pod** (working-set sempre caldo in page-cache). Sul **3060
   REALE** la velocità è governata dal **FIT**: keep-8 che ENTRA = **25 t/s**, keep-32 che NON entra =
   **3.4 t/s** (stesso 3060). → il valore di K12 è **doppio**: **rende (L1)** *E* il suo working-set
   (**~516 esperti**) è **più vicino a entrare** nei **~394 slot** reali del 3060 di quanto lo sia K23
   (**~989**) → K12 tende al regime-che-entra (veloce), K23 no. Il vero bersaglio-fit è il set
   **per-fase (~240)**, ben sotto i 394 slot → l'**ammissione-di-fase (punto 6) non serve solo a
   chiudere `</html>`: serve anche a far ENTRARE il working-set** e raggiungere il regime veloce.
   K12-che-rende + fit-vicino = il compromesso velocità/qualità reale per il 3060.
6. **Serve ammissione-di-fase per la coda? SÌ**, stessa conclusione di d046a71 — il domain-match
   statico porta *quasi tutto il percorso* (body + elementi richiesti) ma **non chiude**; il residuo
   terminale (loop) e il JS-fine sono la fase che una mask statica non copre. A K12 il JS **peggiora**
   (handler rotti vs funzionanti del K48) → l'ammissione-di-fase serve **di più** a K12, non di meno.
7. **Fragilità-K non-monotona confermata**: K12 rende (L1), K48 rende (L1), **K64 collassa (L0,
   d046a71)**. Il "più stretto" NON è peggio del "più largo" per la resa strutturale; anzi K12,
   compatto, arriva a body prima e rende gli elementi richiesti a budget minore.
8. **Caveat**: pod RAM-hot (t/s diagnostici, la qualità trasferisce); greedy CUDA non
   bit-riproducibile (n=2 mostra fenotipo stabile, non byte-identico); grader euristico L0-L3 +
   lettura fenotipica. BONUS lock-tail-freddo dinamico NON eseguito (fuori budget; verdetto già netto).

## Artefatti
- `masks/sessCyber_K12.txt(.json)`, `masks/sessCyber_K16.txt(.json)` — weighted massa-gate, sorgente
  cyberpunk-K0. (K23/K48 riusati da `20260711_domain_calibration/masks/`.)
- `K48_ctrl/`, `K12_r1/`, `K12_r2/` — `gen.txt` (output integrale) + `diag.txt` (t/s).
- `route_maskedCyber_K12_cyberpunk.csv.gz` — decode mascherata-K12 reale (219 960 righe, 40 layer,
  enforcement 12/12 su tutti i layer, 0 violazioni) per cross-ref fasi.
- Script `run_one.sh`, `phase1.sh`, `phase2.sh` (bisezione, non usata). Grader `scripts/functional_grade.py`.
