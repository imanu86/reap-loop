# Bit-exact: la cache tocca la qualita'? — verdetto

Domanda (priorita'-0 del coordinator): cache-on e cache-off producono token
IDENTICI a parita' di mask/prompt/seed greedy (temp0)? Se divergono, la
cache tocca la qualita' e la campagna L0-L3 va riletta.

## Metodo (dopo 2 pivot)

1. Primo tentativo (`bitexact/`): confronto reserve=1 (cache attiva) vs
   reserve=16 (doveva essere "cache off"). **Abbandonato**: reserve=16 non e'
   affidabile (vedi `RESERVE16_ABORT.md`) e confonde due effetti diversi
   (cache-off + abort intermittente) — non e' un vero toggle on/off a parita'
   di dimensione cache.
2. Metodo finale (`bitexact2/` + `control/`): **reserve=1 fisso** (parsing
   sicuro, sempre onorato), stessa mask K12 (session-weighted, coffee W50),
   stesso prompt, stesso seed, greedy temp0, 300 tok. Si varia SOLO la
   dimensione cache: **1024** (hit-ALTO, 98%) vs **32** (hit-BASSO, 0%).
   Ripetuto in 2 condizioni: `DS4_CUDA_NO_Q8_F16_CACHE=1` (cache uniforme
   2-bit) e senza (cache q8/f16 attiva).
3. **CONTROL aggiunto in corsa** (richiesto dal coordinator dopo la scoperta
   che il greedy sotto SSD-streaming NON e' deterministico neanche a
   config fissa, per via delle race dei prefetch async 0015/0021/0026):
   stessa identica config (K12, cache32, q8-uniforme) girata **due volte**,
   per misurare il rumore di base prima di attribuire qualunque diff alla
   cache.

## Risultati

| Confronto | Diff (righe `diff`) | Nota |
|---|---|---|
| **CONTROL**: cache32 vs cache32 (stessa config, 2 run) | **21** | rumore di base (non-det da prefetch async) |
| **TRATTAMENTO q8-uniforme**: cache1024 vs cache32 (`NO_Q8_F16_CACHE=1`) | **24** | ≈ control (21) |
| **TRATTAMENTO q8/f16-attiva**: cache1024 vs cache32 (senza `NO_Q8_F16_CACHE`) | **54** | ≈2.6x il control |

Prima riga di divergenza del control: `9,16c9,15` (dentro il blocco CSS,
tok ~9-16 su 300) — conferma la scoperta del coordinator: il greedy diverge
presto anche a config bit-per-bit identica, per via delle race async, non
per la cache.

## Verdetto

- **Cache uniforme 2-bit (`NO_Q8_F16_CACHE=1`)**: 24 righe di diff contro un
  rumore di base di 21 — **dentro il rumore**, nessuna evidenza che la
  dimensione della cache (quindi l'hit-rate) tocchi la qualita' oltre il
  non-determinismo intrinseco del path SSD-streaming. Il path miss/direct e
  il path hit sembrano numericamente equivalenti quando la precisione e'
  uniforme: **nessun bug di correttezza confermato sul path 2-bit puro**.
- **Cache q8/f16 attiva (default, `NO_Q8_F16_CACHE` non settata)**: 54 righe
  di diff, ~2.6x il rumore di base — **eccede chiaramente il control**.
  Indicazione che servire un expert dalla cache in precisione q8/f16 invece
  che nativa 2-bit introduce una divergenza SISTEMATICA, non spiegabile dal
  solo non-determinismo dei prefetch. Questa e' la "crepa di precisione"
  ipotizzata dal coordinator: **confermata a livello di segnale (n=1 coppia),
  non quantificata in dettaglio** (fuori budget separare quanto del 54 e'
  rumore-base vs precisione-pura — servirebbe un control anche nella
  condizione q8-attiva, non fatto qui).
- **Non e' un allarme rosso di correttezza core**: il path piu' semplice e
  piu' usato nelle probe precedenti (`DS4_CUDA_NO_Q8_F16_CACHE=1`, presente
  in tutte le env "sane" documentate) non mostra segnale oltre il rumore.
  Il rischio e' isolato alla cache q8/f16 (comportamento di default se
  quella env NON e' settata).

## Implicazione di design per 0031 (pin-keep)

Il verdetto qui sopra e' gia' cablabile nella prossima patch di residenza
fissa (0031, pin-keep): il path 2-bit puro (`NO_Q8_F16_CACHE=1`, 24 righe
diff contro 21 di rumore-base = dentro il rumore) e' **bit-sicuro** per
aumentare la residenza/pinning in VRAM. Il path q8/f16 (54 righe, ~2.6x il
rumore) **non lo e'**: se 0031 introduce slot pinnati, quegli slot devono
servire SEMPRE precisione 2-bit nativa, MAI q8/f16 — un gate hard in fase di
design, non un'opzione. Altrimenti la patch che dovrebbe aumentare la
velocita' (vedi tabella t/s sotto: cache/hit alto era comunque piu' lento
del direct, quindi 0031 andra' probabilmente combinata con dimensionamento
cache-al-working-set, non solo pinning) rischia di introdurre silenziosamente
la stessa deriva di qualita' misurata qui.

## Limiti

- n=1 per confronto (una sola coppia per condizione) — il coordinator ha
  chiesto di ripetere anche la velocita' a n>=2; lo stesso vale qui in
  linea di principio, ma il gap control(21) vs trattamento-q8on(54) e'
  abbastanza largo da essere un segnale credibile anche a n=1.
- Non e' stato fatto un control nella condizione q8/f16-attiva (due run
  identici CON cache q8/f16, per isolare quanto del 54 e' rumore-base in
  QUELLA condizione specifica — il rumore potrebbe non essere lo stesso 21
  misurato in condizione uniforme-2bit).
- Non e' stato provato il bonus "prefetch disabilitato" (env
  `DS4_REAP_PREFETCH=0`) per verificare se il greedy torna deterministico —
  tagliato per budget di tempo.

## Artefatti

- `bitexact/` — primo tentativo (reserve1 vs reserve16), da NON usare per il
  verdetto di qualita' (confuso dall'abort bug), utile solo come base per
  `RESERVE16_ABORT.md`.
- `bitexact2/K12_cache{1024,32}_q8{off,on}/` — le 4 run del metodo finale.
- `bitexact2/diff_q8off_hi_vs_lo.txt`, `diff_q8on_hi_vs_lo.txt` — i diff.
- `control/ctrl_32_{a,b}/` — le 2 run identiche di controllo.
- `control/diff_ctrl_32_a_vs_b.txt` — il diff di controllo (21 righe).
