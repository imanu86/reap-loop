# S2 locale 3060 — T4 W-sweep freeze-safe + T5 weighted-vs-unit (coffee, due-fasi)

Esecuzione: 2026-07-10 18:09 → 2026-07-11 ~02:20, WSL Ubuntu-24.04, binario
live-tree `/root/ds4/ds4` (post-0018, quello dei run M1), modello
`/root/models/ds4-2bit.gguf`, CLI-direct due-fasi (ricetta pod replay), cache
256, ctx 4096/4096, `--total 1200` (fase2 = 1150-1170 tok), temp 0, n=3.
Harness: `scripts/run_w_sweep_freeze_safe.py` (+ fix fence `92406ce`/`b91188d`),
mask: `scripts/build_session_mask_canonical.py`, grading:
`scripts/functional_grade.py` (node ASSENTE su WSL → check JS euristico,
consistente per i confronti interni).

**Co-residenza dichiarata:** sulla GPU girava per tutto il batch il server UI
dell'utente (porta 8000, ~2.9 GB VRAM, idle, MAI toccato; lock dedicato
`/tmp/ds4_s2.lock` per i run). **Tutti i tempi sono "co-resident UI server —
non confrontabili come speed pulita"; S2 misura QUALITÀ.**

Riordino per valore decisionale (coordinator): T4(50, 30, 130) → T5 ABAB →
coda W{90,70,110,150} **DESCOPATA su pod2** (`runs/ds4/20260710_pod2_t4_tail/`);
localmente esistevano solo skip-marker, rimossi — la coda NON è stata eseguita
in locale (nota per il coordinator che la credeva completata dal driver).

## T4 — tabella W × seed (mask weighted, freeze-safe)

| W | run | boundary | freeze_class | tok_est | L0-L3 | doc-restart | loop(repeat) | p2gen t/s | chars |
|---|---|---|---|---|---|---|---|---|---|
| 30 | r00 | `>` | clean | 14 | **2** | 1 | 0 | 1.74 | 1938 |
| 30 | r01 | `>` | clean | 14 | **0** | 1 | 1 | 1.84 | 5518 |
| 30 | r02 | `>` | clean | 14 | **1** | 1 | 1 | 2.51 | 3213 |
| 50 | r00 | `;` | clean | 50 | **2** | 1 | 0 | 1.62 | 2386 |
| 50 | r01 | `}` | clean | 40 | **0**† | 0 | 0 | — | 161 |
| 50 | r02 | `;` | clean | 50 | **2** | 1 | 0 | 1.74 | 2397 |
| 130 | r00 | `;` | clean | 108 | **1** | 1 | 1 | 1.74 | 3812 |
| 130 | r01 | `;` | clean | 106 | **2** | 1 | 0 | 1.79 | 2376 |
| 130 | r02 | `;` | clean | 108 | **2** | 1 | 0 | 1.85 | 2772 |

† r01-W50 = **anomalia infrastrutturale**, non datapoint di qualità (vedi §Anomalie).

- `freeze_class`: *clean* = taglio su boundary strutturale; *raw* = nessun
  boundary sicuro (`freeze_boundary=none`). **9/9 righe clean, 0 raw** — dopo il
  fix fence il taglio-lotteria J44 è eliminato dal disegno. Non esiste quindi
  una scala "raw" locale da leggere separatamente (i raw della patologia
  prosa-prima-di-fence sono un fenomeno del pod S5).
- `doc-restart` := `doctype>=2` nel deliverable (il flag `restart` del grader
  sottoconta sulle righe L0/L1: `grade_frontpage` lo valorizza solo sul
  percorso L2/L3).

### Mediane e monotonia

| W | mediana L | doc-restart | loop |
|---|---|---|---|
| 30 | **1** | 3/3 | 2/3 |
| 50 | **2** (mediana {2,0,2}=2, regge anche col †) | 2/3 | 0/3 |
| 130 | **2** | 3/3 | 1/3 |

**Monotonia: SÌ** — mediana L non-decrescente (1 → 2 → 2) sui 3 W misurati
(W90/70/110/150 su pod2). **Spread = 1** (< 2 richiesto dal criterio runbook) e
**restart_majority ≠ 0 a ogni W** → per i criteri pre-registrati del runbook
la tabella W **NON è riabilitata come scala pulita**, ma **nemmeno collassa
piatta** (la lettura "solo lotteria" avrebbe voluto livelli piatti senza
restart).

### Lettura corretta (il risultato scientifico di T4)

1. **Il freeze sicuro NON elimina il document-restart in fase-2: 8/9 righe
   valide ripartono** con un documento fresco (fence + `<!DOCTYPE`) ignorando
   il prefisso frozen — anche con tagli perfetti su `}`/`;`/`>`.
   → **Il restart è un attrattore del re-prefill `[istruzione]+[HTML parziale]`
   in sé (a cache256/K23 locale), non (solo) del taglio dentro una
   dichiarazione CSS.** La spiegazione J44 "W=80/110/150 rotti perché il cut
   cadeva male" è quindi **insufficiente**: il cut sicuro non basta a impedire
   il restart.
2. **La qualità scala comunque con W attraverso la resistenza al loop**: W30
   degrada (2/3 loop → mediana L1), W50/W130 tengono (0-1/3 loop → mediana L2).
   Il documento riscritto in fase-2 si completa pulito quando la mask di
   sessione è stata costruita su una finestra W sufficiente.
3. **Pavimento W ≈ 50**: primo W con mediana L2, replicato indipendentemente
   dal braccio T5-weighted (3 × W50: {L2, L1, L2} → mediana L2, 4/6 L2 sul
   totale W50-weighted). Sotto (W30) si scende a L1.
4. Caveat W30: il boundary sicuro più grande ≤ 30 tok cade a ~14 tok stimati
   (head HTML povero di boundary) → la cella W30 testa di fatto un prefisso
   frozen più corto del nominale. Non cambia la lettura (la mask resta
   costruita su ~46 tok osservati), ma va ricordato per i confronti.

## T5 — weighted OFFLINE vs unit in-engine (W=50, ABAB, n=3 per braccio)

| round | braccio | L0-L3 | doc-restart | loop | p2gen t/s | chars |
|---|---|---|---|---|---|---|
| r0 | weighted | **2** | 1 | 0 | 1.70 | 2021 |
| r0 | unit | **1** | 1 | 1 | 1.69 | 5293 |
| r1 | weighted | **1** | 1 | 0 | 1.78 | 1834 |
| r1 | unit | **1** | 1 | 0 | 1.05 | 2020 |
| r2 | weighted | **2** | 1 | 0 | 1.57 | 2338 |
| r2 | unit | **2** | 1 | 0 | 1.24 | 1701 |

**Verdetto T5: WEIGHTED > UNIT.** Mediana weighted **L2** (n=3: {2,1,2}) vs
mediana unit **L1** (n=3: {1,1,2}). Criterio
runbook soddisfatto: mediana weighted strettamente più alta al W testato, mai
più bassa, e p2gen nella stessa banda (mediane 1.70 vs 1.24 t/s — weighted
**non** paga la qualità con velocità; il warning J45/J46 sul weighted-in-engine
non si trasferisce al weighted-OFFLINE). **Il metodo storico a massa-gate
resta il costruttore giusto per la mask di sessione; il relearn in-engine
dovrebbe calcolare massa cumulativa, non conteggio unitario.** (Un solo W
testato: estensione W∈{90,130} = coda naturale su pod.)

## Anomalie

1. **r01-W50 (T4): crash infrastrutturale muto di fase-2** — ds4 morto a
   `gpu prefill layer 14/43` senza alcun messaggio (niente OOM-killer in dmesg,
   niente error string nel diag); trest vuoto, deliverable = solo frozen (161
   ch) → L0 **non-quality**. Riga tenuta nel CSV come da protocollo (mediana
   W50 invariata). Non ricorso: 14/15 fasi-2 completate.
2. **Doc-restart universale nonostante freeze sicuro** (14/14 fasi-2 completate
   con doctype=2, T4+T5): caratterizzato sopra — attrattore del re-prefill, non
   del cut. Il deliverable contiene [frozen + documento riscritto completo]; il
   grader seleziona il documento più completo e cappa a L2 per il restart.
   Difetti tipici 2-bit nel doc riscritto: `#faaq`, `#brown`, `1.0.5rem`,
   `alert('Conferycation!')`.
3. **Emissione fence non deterministica a temp 0**: 12/15 fasi-1 aprono con la
   fence markdown, 3/15 no (stesso comando greedy) — conferma il
   non-determinismo dell'engine SSD-streaming già visto in M1a. Gestita dal
   fix `92406ce`+`b91188d` (0 righe raw).
4. **Fase-1 t/s degradante nella notte** (p1_gen 0.29-0.39 → 0.14-0.17 t/s
   tra le 18 e le 01): co-residenza + probabile thermal; ennesima ragione per
   cui i tempi di questo batch non sono confrontabili come speed.
5. **Nota operativa**: il primo gruppo W30 (driver v1, ordine vecchio) è stato
   ucciso a ~20 min per il riordino decisionale e ri-eseguito da zero col
   driver v2; la sessione dell'esecutore è morta per esaurimento crediti alle
   ~00:30 ma il driver nohup è sopravvissuto e ha completato il piano senza
   perdite. Smoke di validazione in `runs/ds4/_smoke_s2/` (non committata).

## Implicazioni CLAIMS (da applicare a cura del coordinator — QUI NON TOCCATE)

- **SESSION-LEARNING (OPEN, nota J44):** aggiornare la meccanica: il knife-edge
  del cut point è RIMOSSO dal freeze-safe (9/9 clean) ma il restart di fase-2
  persiste col re-prefill → la vecchia tabella W era confusa da DUE effetti
  (cut-lottery + attrattore restart); la scala W reale è la loop-resistance
  (pavimento W≈50, mediana L2 con restart-cap). Il gate S2 "L2+ mediano senza
  interventi runtime" è raggiunto a W50/W130 su coffee (L2 mediano) — non L3:
  il cap è il restart, candidato naturale per 0025/continuation-prompting.
- **COMPLETAMENTO / budget-confound (T1):** coerente — coffee completa
  ampiamente dentro fase2 1150 tok (deliverable 1.9-2.8k ch quando non loopa);
  l'estensione cyberpunk 4000 tok NON è stata eseguita localmente (tempo:
  batch notturno già ~8h con co-residenza).
- **T5:** nuova evidenza per la riga sul costruttore mask: weighted-OFFLINE >
  unit-in-engine a parità di tutto (n=3, ABAB, stesso W): il relearn PACE
  dovrebbe accumulare massa-gate.

## Artefatti

- `t4_W030|t4_W050|t4_W130/` (summary.csv + summary_median.csv + VERDICT.txt +
  manifest.json + celle W×run complete: route.csv, tw, frozen, sess, p2prompt,
  trest, deliverable.html, diag)
- `t5_{weighted,unit}_r{0,1,2}/` (idem, n=1 per round, ABAB)
- `summary_all.csv` (fuso, con `freeze_class` e `doc_restart`), `s2_aggregate.py`
- `driver.log`, `s2_driver.sh` (v1), `s2_driver_v2.sh` (riordinato)
- probe velocità low-K separata: `runs/ds4/20260711_local_lowK_tps/`
