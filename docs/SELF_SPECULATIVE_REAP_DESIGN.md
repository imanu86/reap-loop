# Self-speculative decoding con mask REAP come draft — design study

**Data:** 2026-07-11. **Stato:** DESIGN STUDY offline, sola lettura (no pod, no build). Nessun
numero qui e' un claim misurato live: sono stime da (a) grep del sorgente `ds4.c` live in WSL
(`/root/ds4/ds4.c`, sola lettura) e (b) analisi statica dei trace di routing gia' raccolti in
`runs/ds4/`. Da validare con la sonda proposta in §5 prima di investire ingegneria pesante.

## L'idea (sintesi)

Self-speculative decoding senza modello draft separato: lo STESSO modello, con due bias-mask
diverse, fa sia da drafter che da verifier.

- **DRAFT** = forward sotto mask K12 (aggressiva, pochi esperti ammessi) propone N token.
- **VERIFY** = forward sotto mask K-alto/full in un singolo passo batched verifica i draft.
- Concordano -> si tengono i token veloci; divergono -> il full corregge e la divergenza
  stessa e' anche un segnale di deriva.

Variante "compatibile-3060" dello speculative di DSpark: zero VRAM extra perche' non c'e' un
secondo modello, solo un secondo stato di bias sullo stesso peso residente.

---

## 1. Fattibilita' motore

### 1.1 Cosa esiste gia' (riusabile)

`ds4.c` ha gia' uno scaffolding MTP-speculative completo e non banale:

- `ds4_spec_frontier` (righe ~26759-26834): snapshot/restore della cache raw SWA speculativa,
  con commit-prefix parziale — la state machine di accept/reject/rollback che serve a QUALSIASI
  draft-then-verify, non solo a MTP.
- `metal_graph_verify_decode2_exact` e `metal_graph_verify_suffix_tops` (chiamati ~30147,
  ~30253): verificano il SUFFISSO di draft in un **singolo passo batched** contro il target —
  non e' un ciclo di verify sequenziali, e' vera ammortizzazione. Il fallback sequenziale
  esiste solo come safety net se il micro-verifier fallisce (~30478-30536, commento esplicito:
  "deliberately slow and should not be selected during normal --mtp operation").
- `DS4_MTP_PROBE` (~29894, ~29934): un contatore hit/miss (`mtp_probe_hit`/`mtp_probe_total`)
  che logga se il draft ha indovinato il token realmente campionato, SENZA commitare nulla —
  cioe' la sonda diagnostica "draft-vs-verify agreement" esiste gia' come pattern per MTP.
  Vedi §5: e' l'analogo esatto di cio' che serve per validare l'idea REAP-draft.

### 1.2 Cosa NON esiste (il gap da colmare)

Il drafter oggi e' cablato in modo rigido a un **secondo modello vero**, non a "stesso peso,
mask diversa":

- `e->mtp_model` / `e->mtp_weights` sono una `ds4_model` distinta, caricata da un secondo
  `.gguf` (`model_open(&e->mtp_model, opt->mtp_path, ...)`, ~28460), con la propria cache raw
  SWA (`s->graph.mtp_n_raw`) e un head dedicato piccolo (`hc_head_fn/scale/base/norm`, poche
  tensor — non l'intero stack di layer). Non c'e' un punto di iniezione per "usa il modello
  base con bias B invece del modello mtp".
- La mask REAP (`g_reap_mask_pruned` / `g_reap_bias_masked`, ~7386-7560) e' **globale,
  process-wide, single-slot**: un solo stato attivo per volta, ricaricato via file-poll
  (`DS4_REAP_MASK_FILE`, controllato ad ogni token decode, ri-applicato ogni 32 poll) con un
  upsert di range di memoria device per-layer + una `fprintf` ad ogni apply. Non e' pensato per
  essere alternato piu volte per ciclo di generazione: oggi cambia al piu una volta ogni ~32
  token, non due volte per token (draft-mask poi verify-mask poi ritorno).

### 1.3 Il vincolo architetturale che invalida l'intuizione "K12 = draft economico"

Punto centrale, confermato sia dal grep sia da `docs/CLAIMS_CURRENT.md` riga "Attuatore mask
REVERSIBILE (0011)": **"Il router calcola comunque tutti i 256 esperti; il bias agisce solo
sulla selezione."** In `layer_topk_selected_experts_from_probs` (~7578-7605) il bias REAP si
somma ai logit di routing PRIMA del `topk_desc`, ma il numero di esperti effettivamente
eseguiti e' `DS4_N_EXPERT_USED` — una **costante di shape del modello** (`g_ds4_shape`, non
influenzata dalla mask). La mask K cambia QUALI esperti sono candidabili, non QUANTI vengono
calcolati per token.

Conseguenza diretta: un forward sotto K12 costa **esattamente gli stessi FLOP** di un forward
sotto K91/full — stesso numero di layer, stessa attention, stesso numero di esperti FFN
eseguiti. A differenza del vero MTP head (poche tensor, non l'intero stack) o di un vero
draft model piu piccolo, **il draft REAP-mask non e' computazionalmente piu economico del
verify**. Vedi §2 per cosa questo implica sul numero.

### 1.4 Stima complessita'

Media-alta, non un semplice "repoint":

1. Nuovo path per far si' che il "drafter" chiami il forward del modello BASE con un bias
   diverso invece di `e->mtp_model` — richiede toccare la state machine di
   `ds4_session_eval_speculative_argmax` (~29970 in poi) per disaccoppiarla dal presupposto
   "drafter = secondo modello con cache propria".
2. Un meccanismo di toggle-mask RAPIDO e locale al ciclo di generazione (non il poll a 32
   token) — oggi ogni apply fa un upsert di range GPU per-layer + fprintf; farlo 2x per ogni
   step di generazione e' overhead nuovo, non gratuito, e va misurato.
3. **Nessun risparmio FLOP garantito** da (1.3): il case per investire questo lavoro regge
   solo se il guadagno viene da localita' di cache (esperti K12 piu piccoli restano caldi tra
   step di draft consecutivi), non da FLOP — ipotesi non verificata, vedi §4.

---

## 2. Accelerazione attesa (numero)

### 2.1 Metodologia (proxy, sola lettura)

Non abbiamo trace di logit accoppiati draft-vs-verify (richiederebbero un run live con due
mask alternate, che non esiste). Come proxy uso i trace di routing FULL-model gia' raccolti
in `runs/ds4/20260711_podA_narrow_traces/{a_coffee_full,b2_json_long_full,c2_python_long_full}`
(route.csv pesato, 40 layer MoE, top-6 esperti/posizione, 150-300 token/cella, nessuna mask).
Per ogni K in {12,23,38,50,64,91} costruisco una keep-mask per layer (rank per massa-gate
cumulata, stessa logica di `build_session_mask.py` usata in
`runs/ds4/20260710_pod_cache1024_warmup_replay`) e misuro, per ogni posizione, se il forward
K-mascherato avrebbe selezionato GLI STESSI esperti del forward full osservato — condizione
**sufficiente ma non necessaria** per token identico (se gli esperti selezionati sono
identici, il layer produce output bit-identico, visto che i pesi-esperto usano le probs
originali non biasate; se differiscono, il token PUO' comunque restare identico se lo scarto
di massa e' irrilevante — quindi questo proxy e' un limite inferiore, non la vera acceptance).

Due varianti di mask: **warmup** (costruita sui primi 50 token, applicata al resto — replica
il regime online reale a due fasi) e **oracle** (costruita sull'intera traccia, upper bound
in-dominio). Due varianti di soglia: **top6-all40layer** (condizione stretta, sufficiente per
identita' esatta) e **top1-all40layer** (solo l'esperto dominante per layer, proxy piu morbido).

### 2.2 Risultati

| K | top6-all40 (warmup) | top1-all40 (warmup) | top1-layer medio (warmup) | top1-all40 (oracle) | top1-layer medio (oracle) |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.00% | 0.00-0.40% | 52.4% | 0.00-7.7% | ~75% |
| 23 | 0.00% | 0.00-0.40% | 66.6% | 0.5-33.8% | ~87% |
| 38 | 0.00% | 0.81-1.94% | 76.8% | 14.0-60.2% | ~94% |
| 50 | 0.00% | 0.81-9.78% | 81.0% | 25.5-71.9% | ~97% |
| 64 | 0.00% | 0.81-17.4% | 83.7% | 48.5-80.9% | ~98% |
| 91 | 0.00-10.7% | 0.81-25.8% | 86.8% | 77.9-96.6% | ~99.7% |

(range = valori sulle 3 celle coffee/JSON/python; script `scripts` in scratchpad, non
committato — riproducibile dai route.csv citati.)

Lettura: la condizione stretta (top6, sufficiente per identita' esatta) e' **~0% ovunque**
nel regime realistico — anche a K91 (circa meta' del budget di 256 esperti/layer tenuto). La
condizione morbida (top1, l'esperto dominante) sale con K ma nel regime **realistico
(warmup)** resta bassa (0-26% anche a K91); solo nel regime **oracle in-dominio** (il
contenuto generato e' lo stesso su cui la mask e' stata costruita, il caso migliore possibile)
si avvicina a valori utili (78-97% a K91).

Nota di onesta': il progetto ha gia' un finding che tempera (con cautela) il pessimismo del
proxy stretto — saliency-K50 on-domain ppl 3.8604 vs full 3.8111 = 1.013x [CI 0.995-1.028],
CI che attraversa 1.0 (statisticamente indistinguibile da full su N=4 chunk), contro random
control 1.365x [CI 1.280-1.455] che non attraversa 1.0. Il claim difendibile e' il **contrasto
saliency << random (CI non sovrapposti), non "1.013x quasi-lossless"** — cosi' come
`docs/paper/PAPER.md` (riga ~416) corregge esplicitamente se stesso contro l'overclaim.
Tradotto per qui: differenze di selezione esperto layer-per-layer non sempre si propagano a
differenze di token forti come il proxy stretto suggerisce, ma il margine e' piu' piccolo di
quanto "quasi-lossless" implichi, e comunque parla di PPL aggregata, non di top-1 token match
posizione-per-posizione. La vera acceptance resta **non quantificabile senza la sonda di §5**.

### 2.3 Perche' il numero e' comunque negativo: draft e verify costano uguale

Qui la matematica standard dello speculative decoding (Leviathan & Chen 2023) e' decisiva.
Con blocco di draft di lunghezza gamma e acceptance i.i.d. alpha, il numero atteso di token
per ciclo e' `(1 - alpha^(gamma+1)) / (1 - alpha)`; il costo per ciclo (in unita' di "un
forward target") e' `gamma*c + 1`, dove `c` = costo relativo di un passo di draft rispetto a
un passo target. Lo speedup e':

```
speedup = [(1 - alpha^(gamma+1)) / (1 - alpha)] / (gamma*c + 1)
```

Il §1.3 stabilisce che per un draft REAP-mask **c = 1** (stesso FLOP del target: router calcola
sempre tutti gli esperti, top-k fisso). Con c=1, gamma=2 (profondita' di draft di produzione
di ds4, `mtp_draft_tokens` default 1, tipicamente 2 in uso — ds4.c ~28331):

| alpha | speedup (c=1, gamma=2) |
|---:|---:|
| 0.2 (realistico, K12-K38 warmup) | 0.41x (piu' lento) |
| 0.5 | 0.58x |
| 0.8 (oracle K64-91, caso migliore) | 0.81x |
| 0.95 | 0.95x |
| 0.99 | 0.99x |
| 1.00 (limite teorico) | 1.00x (pareggio, mai vittoria) |

**Con costo draft=verify, lo speedup e' matematicamente limitato sopra da 1.0x per QUALSIASI
alpha < 1 — non c'e' scenario in cui questo meccanismo, cosi' come proposto (mask-swap sullo
stesso stack di layer), batte la baseline sui FLOP.** I nostri numeri migliori (§2.2, oracle
K91) danno alpha nell'ordine di 0.8-0.97 -> speedup atteso 0.81-0.97x, cioe' **pareggio o
rallentamento**, non accelerazione. Nel regime realistico (warmup, K12-38, come proposto
dall'idea originale "K12 aggressiva") alpha e' 0-0.3 circa -> speedup **0.0-0.5x, nettamente
piu' lento**.

L'unica via d'uscita da questo tetto e' `c < 1`, cioe' un draft genuinamente piu' economico in
FLOP — che qui non c'e' (a differenza di LayerSkip/Zhang, che saltano LAYER, o del vero MTP
head di ds4, che e' un mini-head e non l'intero stack) — oppure un guadagno **fuori da questa
contabilita' FLOP**: localita' di cache (working-set K12 piu' piccolo, meno cold-load durante i
gamma step di draft). Questo e' l'unico argomento a favore residuo, ed e' empiricamente aperto
(§4).

---

## 3. Il doppio ruolo: la divergenza come segnale di deriva

Trovata la connessione piu' forte del design study: il team ha **gia' voluto misurare
esattamente questo segnale** e non ci e' riuscito per un blocco tecnico che il self-speculative
REAP-draft rimuoverebbe.

Da `runs/ds4/20260711_instrumented_collapse/REPORT.md` (riga "ACCEPTANCE (MTP)"):

> "ACCEPTANCE (MTP): NOT MEASURED — hardware/tooling block (declared). [...] Acceptance
> remains the theoretically strongest early-warning (no calibration window, cross-model) but
> is currently unmeasurable on-pod."

I tre blocchi elencati nello stesso report sono TUTTI aggirati da un draft same-model:

1. `--ssd-streaming` e `--mtp` sono hard-guarded incompatibili sul build in uso, senza bypass
   funzionante -> un draft che riusa il modello gia' residente (nessun secondo `.gguf`) non ha
   motivo strutturale di collidere con lo streaming allo stesso modo (da verificare in
   implementazione, ma il conflitto oggi e' specificamente "due modelli in streaming", non
   "una mask in piu'").
2. Il path non-streaming va in OOM su 24GB con un modello da 86GB -> irrilevante se non serve
   un secondo modello.
3. Il `.gguf` MTP da 3.8GB e' assente sia sul pod sia su R2 -> irrilevante, la REAP-mask non
   richiede pesi aggiuntivi, solo un secondo stato di bias sui pesi gia' caricati.

Quindi, **anche se il caso per l'accelerazione e' negativo (§2.3), il caso per costruire
questo meccanismo SOLO per finalmente misurare il segnale acceptance/divergenza resta in
piedi** — e' la via piu' economica per sbloccare una misura che il progetto ha gia' segnato
come "la piu' forte in teoria".

### 3.1 Confronto con l'entropia (unico segnale gia' misurato)

Dallo stesso report: entropia z-onset a pos 125, **+57 token prima** del primo garbage
palese (pos 182, `data-gposcate`), +82 prima del repetition-lock; baseline sano entropia
media 0.023 / std 0.081 — un pavimento pulito, quasi zero rumore nel regime sano.

La divergenza draft-verify e' concettualmente un segnale diverso e potenzialmente piu' precoce
(non richiede un collasso di entropia per attivarsi, basta un disaccordo token-per-token), ma
i nostri numeri di §2.2 dicono che il suo **pavimento nel regime sano non e' pulito**: anche
nel caso oracle migliore (K91, in-dominio) il proxy stretto mostra 3-22% di "disaccordo" gia'
in condizioni sane, e nel regime realistico (warmup) il disaccordo di base e' 74-100%. Un
segnale con un pavimento di rumore cosi' alto in regime sano e' un rivelatore molto piu' debole
dell'entropia (pavimento ~0) finche' non viene calibrato — non e' un banale sostituto,
richiede soglia/EWMA come nel caso di S1 (vedi CLAIMS_CURRENT.md riga "FEEDBACK slope-S1":
"assoluto cronico ~0.75 -> NON distingue; solo lo SLOPE e' usabile"). Ipotesi di lavoro:
e' probabile che anche l'acceptance-rate REAP-draft sia utile solo come SLOPE (variazione
rispetto alla propria baseline locale), non come valore assoluto — non misurabile da qui,
richiede §5.

---

## 4. Onesta'

- Lo speculative decoding, in qualunque forma, accelera la produzione di un output che il
  modello avrebbe comunque generato: **non cambia COSA viene generato**. Non cura il collasso,
  al massimo ci arriva piu' in fretta (se funzionasse — §2.3 dice che qui probabilmente non
  funziona sui FLOP) o lo segnala prima (§3, ipotesi aperta).
- Il collo di bottiglia misurato questa notte sul 3060 e' la cache/residenza degli esperti, non
  il decode compute (piu'-hit=piu'-lento). Un meccanismo che alterna DUE working-set di esperti
  diversi (K12 draft, K-alto verify) **ad ogni ciclo di generazione** aggiunge transizioni di
  working-set esattamente dove il collo di bottiglia gia' vive — rischio concreto di
  peggiorare, non solo di non migliorare, finche' non e' misurato.
- Conclusione: questa leva vale SOLO dopo che qualita' (S3, es. entropy-triggered
  widen/rewind) e cache (S4, es. K23/rotazione/residenza) sono avanti — non prima. Anche nella
  lettura piu' ottimista di §2.3 (oracle, alpha~0.9+) lo speedup sui FLOP e' al massimo un
  pareggio; l'unico guadagno plausibile e' indiretto (segnale di deriva, §3) e va pesato contro
  il rischio di cache aggiuntivo.

**Posizione in roadmap: post-S3 (qualita') e post-S4 (cache).** Prerequisito per anche solo
iniziare: la sonda minima di §5, che e' quasi-gratis rispetto al motore di speculative pieno.

---

## 5. Test minimo di validazione

Prima di costruire la state machine draft/verify pesante (§1.4), costruire SOLO la sonda,
sul modello di `DS4_MTP_PROBE` gia' esistente (~29894-29939):

**`DS4_REAP_DRAFT_PROBE`** (nuovo env, shadow-eval, nessun commit/rollback):
1. Ad ogni token decode gia' verificato dal path normale (mask verify attiva, es. K-alto o
   full), esegui in ombra un secondo forward con la mask K12 (bias alternativo, stessi pesi
   residenti) sullo STESSO prefisso — senza modificare checkpoint/cache reale.
2. Confronta il top-1 del forward K12 col token realmente accettato; logga hit/miss
   cumulativi (stesso pattern `mtp_probe_hit`/`mtp_probe_total` + fprintf).
3. Logga in parallelo l'entropia del passo (gia' strumentata da patch 0030,
   `runs/ds4/20260711_instrumented_collapse`) per poter allineare acceptance-rate e entropia
   sullo stesso asse posizione, come fatto per entropia/margin/S1 nello stesso report.

Costo: un forward extra per token (stesso ordine di grandezza del costo gia' pagato da
`DS4_MTP_PROBE` per il vero MTP), nessun secondo `.gguf`, nessuna nuova cache, nessuna state
machine di accept/reject — puramente diagnostico. Un singolo run pod (cyberpunk o coffee
prompt, stesso protocollo di `20260711_instrumented_collapse`) basta a rispondere a:

- alpha reale (non proxy) su K12 e, per confronto, su un K meno aggressivo (es. K64/K91);
- se alpha ha uno slope pre-collasso comparabile o migliore dell'entropia (+57 tok di lead);
- se alpha assoluto e' abbastanza alto da rendere sensato investire in §1.4 (soglia indicativa
  dalla matematica di §2.3: serve alpha > ~0.9 con gamma=2 e c=1 solo per avvicinarsi al
  pareggio, quindi la sonda deve mostrare qualcosa di vicino a quella soglia per giustificare
  il motore pieno — altrimenti la conclusione resta "usalo solo come sensore, non come
  acceleratore").

---

## Riepilogo

| Domanda | Risposta |
|---|---|
| Fattibile sul motore? | Scaffolding draft/verify/batched-verify gia' c'e' ed e' riusabile; il drafter e' pero' cablato a un secondo modello vero, non a una mask sullo stesso stack — serve nuova ingegneria (mask-toggle rapido in-processo), complessita' media-alta. |
| Speedup atteso? | Con costo draft=verify (confermato: router calcola sempre tutti gli esperti, top-k fisso), lo speedup e' matematicamente <=1.0x per ogni acceptance <100%; proxy da trace porta a stime 0.0-0.5x nel regime realistico (K12-38) e 0.8-0.97x nel caso oracle migliore (K91 in-dominio) — **pareggio o rallentamento**, non accelerazione. |
| La divergenza e' anche segnale? | Si', concettualmente, e con un ancoraggio forte: il progetto aveva gia' marcato "ACCEPTANCE" come il segnale early-warning teoricamente piu' forte ma NON MISURABILE per blocco hardware/tooling (`ssd-streaming` vs `--mtp`, gguf mancante) — un draft same-model rimuove quel blocco. Ma il pavimento di rumore in regime sano e' alto nel nostro proxy: va calibrato (probabile solo-slope, come S1), non e' pulito come l'entropia (+57 tok di lead, pavimento ~0). |
| Posizione in roadmap? | Dopo S3 (qualita') e S4 (cache): anche nel caso migliore non accelera, e alternare working-set rischia di aggravare il collo di bottiglia cache gia' identificato. Prerequisito: la sonda di §5 (quasi-gratis) prima di qualunque motore pesante. |

## Prior art

- Leviathan, Chen et al., "Fast Inference from Transformers via Speculative Decoding" (ICML
  2023) — la formula costo/accettazione usata in §2.3.
- Zhang et al., "Draft & Verify: Lossless Large Language Model Acceleration via
  Self-Speculative Decoding" (ACL 2024, aclanthology.org/2024.acl-long.607) — self-speculative
  via layer-skip sullo stesso modello, nessun training extra: il draft e' economico perche'
  SALTA layer, non perche' cambia quali esperti sono ammessi.
- Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
  (Meta, ACL 2024, arXiv:2404.16710) — training con layer-dropout progressivo + early-exit
  head condiviso; drafter = layer iniziali, verifier = stack intero; speedup misurato 1.34x-
  2.16x (fino a 2.16x su summarization, 1.82x coding, 2.0x semantic parsing). Stesso principio
  di "stesso modello, meno compute nel draft" che manca alla proposta REAP-mask (§1.3).
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding
  Heads" (arXiv:2401.10774) — non same-model in senso stretto (teste extra addestrate), ma
  stesso principio di costo: head leggere, non l'intero stack; top-1 acceptance ~60%, top-5
  ~80%, speedup 2.2-3.6x grazie a candidati ad albero + head economiche.
- CLaSp / SWIFT (arXiv:2505.24196, arXiv:2410.06916) — layer-skip dinamico senza training,
  1.3-1.7x e 1.3-1.6x: stesso pattern, il guadagno viene sempre da MENO compute nel draft, mai
  da una semplice ri-eleggibilita' di esperti a top-k costante.

Il filo comune a tutto il prior art self-speculative: il draft e' economico perche' fa MENO
lavoro (meno layer, head piccola), non perche' e' "un'altra vista" dello stesso lavoro. La
REAP-mask in ds4, per come e' costruita oggi (router su tutti gli esperti, top-k fisso), non
riproduce quel meccanismo — e' l'unico motivo per cui questa idea, concettualmente elegante,
non traduce automaticamente in uno degli speedup citati sopra.
