# Retro-grade funzionale L0-L3 — output ds4 archiviati

**Data:** 2026-07-10 · **Grader:** `scripts/functional_grade.py` (portato da
moe-aggressive-commit branch `reap/k91-coding-vram`, rubrica **INVARIATA**) ·
**Runner:** `scripts/retro_grade_l0l3.py` · **Dati:** `graded.csv` (105 righe).

## Nota metodologica (leggere prima dei numeri)

Questi grade sono **retroattivi** (grader v-k91) su output **n=1 greedy** gia' su disco.
Il repeat_flag / render heuristico precedente **resta a ledger**: questo retro-grade e' una
**colonna NUOVA di evidenza funzionale**, NON sostituisce i replay ne' il ledger esistente.
La rubrica frontpage e' tenuta identica al repo gemello per comparabilita'.

Scala (frontpage HTML single-file: hero + nav + bottone-che-fa-qualcosa + form + CSS/JS):
- **L0** = non fa parse / non si apre (nessun `<body>`/`<div>`/`<section>` reale)
- **L1** = si apre ma feature critica rotta (bottone assente/non wired, JS con errori)
- **L2** = feature presenti, difetti minori (manca un elemento non critico)
- **L3** = pieno e pulito

Per i prompt **code / code_mini** solo syntax-level (`compile()` Python -> `py_syntax_ok`,
"syntax-only, no unit tests"): **nessun L-level** (la rubrica frontpage non si applica).

Popolazione: **105** `content_measured.txt` (89 HTML, 1 code, 15 code_mini). Le due dir
non tracciate `20260710_{w100,w50}_rotate32_..._html4000` sono ESCLUSE: non hanno prodotto
`content_measured.txt` (run non completata su disco).

## Risultato headline

Distribuzione HTML: **L0=87, L1=1, L2=1** (n=89). Solo **2 run su 89** producono una
struttura `<body>`/`<div>` reale: **entrambi i 2000-token rotate32 W-run** (W100=L1, W50=L2).
Tutto il resto (<=800 token, e anche W100-direct/compact a 2000 token) **muore dentro
`<head><style>`**, tipicamente in un **loop di ripetizione del CSS**, prima di emettere
`<body>`, bottone, form o `<script>`.

Verifica strutturale grezza sull'intero corpus HTML (89 file): `<body>` presente in **1**,
`</html>` in **0**, `<button>` in **1**, `<div>/<section>/<main>` in **2**. Il modello ds4
in questi budget **quasi mai chiude il documento**: il grade L0 non e' un artefatto del
grader, e' la misura di questo fatto.

## Domande decisionali

### (a) rotate32 800tok (requested4 cache128 + cache256, variante rotate32) — L quanto?

**L0** entrambe. `requested4_html800_cache256/html_local_k23_rotate32_cache256` e
`requested4_html800_cache128/html_local_k23_rotate32_cache128`: **L0, repeat_flag=0**.
Il cache256 e' un **loop CSS**: rigenera all'infinito il blocco `header/.grid/.card/.btn`
(con corruzioni: `#l Lime`, `##0f0`, `radial-gradient circle at center`), **non arriva mai
a `<body>`**, niente bottone/form/`<script>`. Il candidato "meta' qualita'" del test T2 del
piano, misurato funzionalmente, e' **catastrofico (L0)** — e il proxy repeat=0 lo dava per
pulito.

### (b) static K23 stessa famiglia — L quanto?

**L0**, `repeat_flag=1`. `requested4_html800_cache256/html_local_k23_cache256` e
`_cache128/html_local_k23_cache128`: entrambi **L0**, loop CSS (`/* Reset */ * { margin:0
padding:0 box:border font:new */ ...` ripetuto, CSS malformato). Anche i pod static K23
(`pod_e7w4_static64`, `_static128`) sono **L0, repeat=1** (loop `body { background:#dark
color:#light box-sizing:h */ ...`). Static K23 e rotate32 a 800 token **finiscono entrambi
L0**; la differenza e' solo *quale* proxy li becca (static -> repeat=1, rotate32 -> repeat=0).

### (c) W-run del 2026-07-10 (2000 token)

| run | L | repeat_flag | body | button | button_wired | form | popup(alert in `<script>`) |
|---|---|---|---|---|---|---|---|
| `w100_direct_k23_cache256_html2000` | **L0** | 1 | no | no | no | no | no |
| `w100_rotate32_k23_cache256_html2000` | **L1** | 0 | si (`<div>`) | no | no | no | no |
| `w100_rotate32_..._html2000_compact_prompt` | **L0** | 1 | no | no | no | no | no |
| `w50_rotate32_k23_cache256_html2000` | **L2** | 0 | si (`<body>`+`<div>`) | si | si (onclick) | no | no |

Lettura: **solo rotate32 a 2000 token arriva al body**. `w100_direct` collassa in un loop
`font-s: 2; font-s: 2; ...` gia' a 2000 token -> **L0**. Il **compact_prompt non aiuta**: L0
(loop). Il **W50 rotate32 (L2)** e' l'unico output "buono" del corpus (nav/hero/griglia di
card con `<button onclick="showAlert()">`), ma **tronca prima del `<script>`** -> `showAlert`
non e' mai definita e non c'e' `<form>`: L2 = "feature presenti, difetti minori" (grade
leggermente generoso, vedi Spot-check #1). Il **W100 rotate32 (L1)** apre (ha `<div>`) ma il
bottone e' assente e la struttura e' sporca (doppio `</style>`, commento `<!-- HTML -->`).

### (d) Distribuzione L0-L3 per famiglia (solo prompt HTML)

| famiglia | n | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| exchange_matrix (b1-b6+smoke) | 21 | 21 | 0 | 0 | 0 |
| breath | 4 | 4 | 0 | 0 | 0 |
| descent_prebreath | 14 | 14 | 0 | 0 | 0 |
| stepdown_pace (+pace_advanced) | 14 | 14 | 0 | 0 | 0 |
| sota_candidates | 4 | 4 | 0 | 0 | 0 |
| cache_sweep | 6 | 6 | 0 | 0 | 0 |
| requested4 (T2) | 8 | 8 | 0 | 0 | 0 |
| pod (static/rotate) | 6 | 6 | 0 | 0 | 0 |
| k23_unit / direct-vs-stepdown | 4 | 4 | 0 | 0 | 0 |
| trace_ab | 4 | 4 | 0 | 0 | 0 |
| **w_runs (2000 tok)** | 4 | 2 | 1 | 1 | 0 |
| **TOT HTML** | **89** | **87** | **1** | **1** | **0** |

Nessuna famiglia a <=800 token produce un output >=L1. L'unica leva che sposta l'ago e' il
**budget di token** (2000) **combinato con rotate32**. Le leve pace/breath/descent/stepdown/
cache non cambiano il livello funzionale in questo corpus: sono tutte L0.

Prompt code/code_mini (16): `py_syntax_ok=0` per **tutti e 16**, `has_code_fence=0` per tutti
— vedi bias sotto (sono review in prosa, non funzioni Python: il syntax-check e' lo strumento
sbagliato, non un verdetto di qualita').

### (e) Correlazione repeat_flag vs L-level (quanto il proxy repeat sovrastima)

Matrice di confusione (89 HTML, 83 con repeat_flag disponibile dal summary.csv):

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| **repeat_flag=0** (32) | 30 | 1 | 1 | 0 |
| **repeat_flag=1** (51) | 51 | 0 | 0 | 0 |

- **repeat=0 ma L0-L1: 31** (30 L0 + 1 L1). Cioe' **31/32 (97%)** dei run che il proxy
  chiama "puliti" sono in realta' **funzionalmente rotti**. Il repeat=0 **non implica
  qualita'**: nell'intero corpus **un solo** run con repeat=0 arriva a L2 (W50 rotate32 2000tok).
- repeat=1 e' invece **precisissimo nel senso negativo**: 51/51 repeat=1 sono L0 (il loop di
  ripetizione, quando c'e', e' sempre catastrofico).
- Conclusione: **repeat_flag e' un rilevatore di loop, non un misuratore di qualita'.** Cattura
  i collassi conclamati (repeat=1 -> L0) ma ha un tasso di falsi-"ok" enorme (repeat=0 -> L0/L1
  nel 97% dei casi qui).

### Bug has_popup del runner (bonus, confermato quantitativamente)

Il flag `has_popup` del runner e' **1 in 81 su 83** HTML con summary. La colonna nuova
`alert_in_script` (che cerca `alert(`/`popup` **solo dentro i `<script>` parsati**) e' **1 in
0**. Cioe': **nessun output ha davvero un popup nel codice**, ma il runner ne segnala 81 —
100% falsi positivi, perche' `has_popup` matcha l'**eco del prompt** ("...un popup JS che
dice richiesta inviata"). `alert_in_script` e' il segnale corretto da usare al posto suo.

## Spot-check (8 output letti a mano)

| # | run | grade | verdetto manuale |
|---|---|---|---|
| 1 | W50 rotate32 2000tok | **L2** | Il piu' completo del corpus: header/hero/`<section>` con 6 `<div class=card>` e `<button onclick="showAlert()">`. MA troncato mid-card prima del `<script>` -> `showAlert` **mai definita**, niente `<form>`. Grade L2 **corretto ma leggermente generoso**: `button_wired` accetta l'`onclick` anche se l'handler non esiste (vedi Bias #2). Funzionalmente L1-L2 borderline. |
| 2 | W100 rotate32 2000tok | **L1** | Apre (ha `<div class=glass-card>`) ma struttura sporca (doppio `</style>`, `<!-- HTML -->` a meta'), **nessun `<button>`**. L1 corretto. |
| 3 | W100 direct 2000tok | **L0** | Anche con 2000 token degenera in `font-s: 2; font-s: 2; ...` all'infinito, mai body. L0 corretto; repeat=1 lo becca. |
| 4 | requested4 cache256 **rotate32** (T2) | **L0** | Loop CSS `header/.grid/.card/.btn` con corruzioni (`#l Lime`, `##0f0`); mai body/bottone/form/script. L0 corretto; **repeat=0 lo aveva mancato**. |
| 5 | requested4 cache256 **k23 static** | **L0** | Loop `/* Reset */ * { margin:0 padding:0 box:border font:new */`, CSS malformato, mai body. L0 corretto; repeat=1. |
| 6 | pod_e7w4 static64 | **L0** | Loop `body { background:#dark color:#light box-sizing:h */`, valori spazzatura, nessun elemento reale. L0 corretto. |
| 7 | exchange_b1 html160 direct_k23 | **L0** | Parte bene poi collassa a `box border: *;` ripetuto a ~token 160. Budget troppo corto per chiudere. L0 corretto. |
| 8 | sota html_direct_k64 | **L0** | Regola CSS `body{...}` ma `color:#cron`, troncato su `background-image: url 'data:image/svg+x`; mai un `<body>` reale. L0 corretto; **repeat=0 lo aveva mancato**. |

**Conclusione spot-check:** 8/8 grade sensati. L'unico attrito e' il caso #1 (W50), dove
`button_wired` e' troppo permissivo (onclick verso funzione inesistente conta come "wired"):
il grader puo' sovrastimare di ~mezzo livello i rari output che arrivano al body. Non tocco la
rubrica (mandato) — lo documento come Bias #2.

## Bias / limiti del grader su questo corpus (rubrica NON modificata)

1. **Il corpus e' quasi tutto pre-body -> il grader collassa a binario.** Con <=800 token il
   modello muore in `<head><style>`, quindi `has_body_struct=False` -> L0 domina e le
   distinzioni fini della rubrica (nav vs hero vs form) **non vengono mai esercitate**. Il
   grade L0/non-L0 qui coincide di fatto con "ha raggiunto il body: si/no". Non e' un difetto
   del grader ma va tenuto presente leggendo la tabella (d): l'assenza di L1/L2/L3 sotto 2000
   token misura il **troncamento**, non una differenza fine di qualita' tra le leve pace.
2. **`button_wired` troppo permissivo.** Un `onclick="showAlert()"` conta come wired anche se
   `showAlert` non e' mai definita nel `<script>` (o lo `<script>` e' troncato). Puo' spingere
   a L2 output il cui popup non funziona (caso W50).
3. **Prompt code/code_mini: strumento sbagliato.** Questi prompt sono **review di pseudocodice
   in prosa** ("List the main correctness risks and a minimal patch plan"), non "scrivi una
   funzione Python". Gli output sono **prosa inglese coerente e sensata** (spot-letto:
   `sota code_mini_direct_k64` produce un'ottima lista di rischi + patch plan). Il
   `py_syntax_ok=0` su tutti i 16 **non e' un verdetto di qualita'**: significa solo "non c'e'
   Python da compilare". Un eval reale di questi richiederebbe un LLM-judge sulla qualita'
   della review, fuori scope qui. Colonne `py_syntax_ok`/`has_code_fence` lasciate come diagnostica.
4. **La rubrica frontpage (nav/hero/form/button/popup) e' coerente col prompt cyberpunk**
   (che chiede esplicitamente modulo contatti + popup di conferma), quindi non ho riscontrato
   il bias "rubrica pensata per un altro prompt". `alert_in_script` e' stata aggiunta apposta
   per correggere il falso-popup del runner senza toccare la rubrica.

## File

- `runs/ds4/20260710_retro_grade_l0l3/graded.csv` — una riga per output, con `level`, tutte
  le sotto-colonne `det` del grader, `alert_in_script`, `py_syntax_ok`/`has_code_fence`, e i
  metadati agganciati dal summary.csv (variant, prompt, completion_tokens, avg_tps,
  repeat_flag, has_popup, doctype, html_balance, s_init_count, content_chars).
- `scripts/retro_grade_l0l3.py` — generatore (rieseguibile: `python scripts/retro_grade_l0l3.py`).
- `scripts/functional_grade.py` — grader v-k91 (rubrica invariata).
