# Substitution-similarity feasibility — mask-miss substitution via co-activation (2026-07-11)

**Mode:** OFFLINE, trace-only (no GPU/WSL/pod) · **Script:** `scripts/analyze_substitution_similarity.py`
**Retrial pointer:** `runs/ds4/20260711_substitution_archaeology/REPORT.md` (verdetto: mai provata
la vera substitution, retrial giustificato) · **Candidato:** `docs/MOE_ECOSYSTEM_SURVEY_20260711.md` §4 #2

## Domanda

Su un mask-miss (il router vuole un expert potato dalla session mask keep-23), sostituire
con l'expert **kept più simile** ha senso solo se il "want" (l'expert fuori-mask che il
router chiede) è **funzionalmente vicino** a un kept — non ortogonale. Questo probe misura
quella vicinanza dai trace pesati già su disco, senza assumere nulla a priori.

## Metodo

1. **Grafo di co-attivazione** (proxy di similarità funzionale, per layer): pool di TUTTI i
   token routati da TUTTE le sorgenti full-model; ogni coppia di esperti co-selezionati nello
   stesso top-6 è un evento di co-attivazione. `sim(layer,i,j) = co-occorrenze(i,j) /
   sqrt(sel(i)·sel(j))` (overlap-coefficient, [0,1]).
2. **Insieme "want"**: per ogni sorgente, split token generati in WARMUP (primi ≤50, rispecchia
   il warmup reale DS4_PACE che congela la mask di sessione) e REST (il resto). `keep-23` è
   costruito dalla sola massa-gate del WARMUP (stessa `rank_keep()` del builder di produzione,
   importata direttamente — nessuna reimplementazione). `want[layer]` = (top-23 per massa nel
   REST) − (keep-23 da WARMUP): esperti che il router ha DIMOSTRABILMENTE richiesto nella
   continuazione e che la mask congelata non ha. Non è un insieme teorico, è il miss reale che
   un run mascherato incontrerebbe.
3. **Substitutability**: per ogni want *w*, `best_kept_sim` = similarità massima verso i 23
   kept del warmup della stessa sorgente. Confrontata contro un **null order-matched**
   (`random-max`): la similarità best-of-23 di un expert scorrelato scelto a caso verso lo
   stesso keep-23 pooled — stessa procedura di massimizzazione su 23 candidati dei want reali,
   per isolare il segnale "w è davvero simile" dal puro effetto statistico del max-su-23.

## Sorgenti (18 trace full-model pesati, `e0..e5/w0..w5`)

2 html (W50/W130, `20260710_pod_cache1024_warmup_replay/`) + 5 narrow (coffee/json/python,
`20260711_podA_narrow_traces/`) + 11 coding (`k91_coding_vram/trace_coding.tgz`). 17/18
usabili per l'estrazione want (≥20 token nel REST); esclusa solo `narrow_b_json_full`
(37 tok totali, rest=19 < soglia).

**Controllato ed escluso:** `runs/ds4/20260710_scope_divergence_pod/r1/s1_r1.csv.gz` — schema
`pos,layer,pruned_mass,total_mass`, nessuna identità per-expert, non utilizzabile per un grafo
di co-attivazione (verificato leggendo header e manifest).

## Risultati

| Distribuzione | n | mediana | p25 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| **want → best-kept** (segnale reale) | 7964 | **0.169** | 0.120 | 0.236 | 0.319 |
| random-max order-matched (null, best-of-23) | 8895 | 0.118 | 0.078 | 0.178 | 0.258 |
| kept-kept pairwise (riferimento, NON order-matched) | 10120 | 0.050 | 0.014 | 0.116 | 0.210 |
| random pairwise (riferimento, NON order-matched) | 8000 | 0.000 | 0.000 | 0.018 | 0.046 |

**AUC = 0.673** = P(want_sim > random_max_sim), effect size Mann-Whitney via rank-sum
(nessuna dipendenza scipy). AUC 0.5 = nessuna separazione, 1.0 = separazione totale — 0.673 è
una separazione **reale ma debole-moderata**, non uno strappo netto. Concretamente: **76%**
dei want superano la mediana del null (atteso ~50% sotto H0). Solo lo **0.01%** dei want ha
similarità *esattamente* zero (mai co-attivato con alcun kept) — quasi nessun want è
totalmente isolato.

Il segnale è **uniforme sui 40 layer** (mediana per-layer 0.130–0.221, nessun layer
anomalo — `per_layer` in `stats.json`).

**Nota metodologica:** il confronto naive want-vs-kept-kept-pairwise (0.169 vs 0.050, 3.4×)
è fuorviante da solo — `want_sim` è un MAX su 23 candidati, un solo pairwise kept-kept no.
Il null corretto (random-max, stessa procedura di massimizzazione) porta il confronto onesto
a 0.169 vs 0.118 (AUC 0.673), molto meno drammatico ma comunque positivo.

## VERDETTO

> **SUBSTITUTION-RUNTIME GIUSTIFICATO, segnale debole-moderato (AUC 0.673, non uno strappo
> netto).** I want fuori-mask NON sono ortogonali ai kept — sono misurabilmente più vicini al
> keep-23 di quanto lo sarebbe un expert scorrelato scelto a caso, con la stessa procedura
> best-of-23. Non è una prova schiacciante: il 24% dei want resta sotto la mediana del caso
> random, quindi la sostituzione aiuta *in media*, non *sempre* — serve una soglia minima di
> similarità sotto cui il miss va comunque sul path SSD/rewind invece di forzare una
> sostituzione scadente.

### Formula di scoring proposta (se si procede al retrial runtime, P4 minimo)

`sim(layer,i,j) = co-occorrenze(i,j) / sqrt(sel(i)·sel(j))`, precalcolata **dalla finestra di
WARMUP della sessione corrente** (stesso segnale già raccolto per costruire la mask, nessun
trace aggiuntivo) come lookup table layer→(N×N sparsa, solo tra kept e loro vicini osservati).
Su un miss dell'expert *w*: `sostituto = argmax_{k ∈ kept[layer]} sim(layer, w, k)`, **con
soglia minima** (es. `sim ≥ p75(random-max) ≈ 0.18` da questo run, da ricalibrare in-session)
sotto cui si rifiuta la sostituzione e si cade sul path esistente (SSD/rewind). Costo: O(23)
lookup per miss, nessun overhead di trace aggiuntivo — il grafo di co-attivazione emerge
gratis dallo stesso `DS4_SPEX_TRACE_ROUTING_WEIGHTS` già usato per costruire keep-23.

**Retrial runtime minimo (dal report di archaeology, P4):** K23-costante n=3 ABAB vs
baseline L-STATIC/L-ROTATE su prompt wide, osservando il segnale-AN-1 (repeat/S1) — il
meccanismo di danno del rotate bocciato (discontinuità hidden/KV da swap di *membership*)
non tocca la substitution (membership invariata), ma resta da falsificare dal vivo, non solo
offline.

### Riprodurre
```
python scripts/analyze_substitution_similarity.py \
  --reap-loop-root <reap-loop> --moe-root <moe-aggressive-commit>
```
Output: `stats.json` (aggregati + verdetto), `want_records.csv` (7964 righe, per audit).
