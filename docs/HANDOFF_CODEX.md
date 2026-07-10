# HANDOFF CODEX — reap-loop

> **Sistema di lavoro:** l'utente lancia Codex con un prompt fisso che punta a questo file.
> Questo file è la fonte del mandato. Prima di iniziare: `git pull` e leggere i commit recenti
> di altri (Claude/utente) — **non lavorare in silo**. A fine mandato: aggiornare §LOG,
> committare, fermarsi. Non uscire dal mandato senza chiedere.

## §REGOLE PERMANENTI (violarle = risultato non valido)

1. **Ground truth**: `docs/CLAIMS_CURRENT.md` (con la sua regola anti-regressione) e
   `docs/NEXT_STEPS_PLAN_20260710.md`. Se un finding cambia lo stato di un claim,
   aggiornare CLAIMS **per primo**.
2. **Il greedy NON è deterministico run-to-run** (divergenza misurata a tok~75 a config
   identica: `runs/ds4/20260710_w50_rotate32_k23_cache256_html4000/ANALYSIS.md`).
   Ogni run è un rollout indipendente → **nessun verdetto qualità o velocità da n=1**:
   usare `--runs 3` (runner, commit `d0ad967`) e riportare i risultati **per-seed** più mediana.
3. **repeat_flag / ngram basso ≠ qualità** (97% dei run repeat=0 sono comunque L0-L1:
   `runs/ds4/20260710_retro_grade_l0l3/REPORT.md`). Il verdetto qualità è SOLO
   `scripts/functional_grade.py` (L0-L3) + note strutturali. L'ngram serve per onset e
   diagnosi, mai per dire "promettente" in corso d'opera.
4. **`prompt_s` misura il prefill del prompt, non il warmup**: mai attribuire differenze di
   prompt_s alla manopola W. Prima di ogni claim di velocità, controllare lo stato cache
   (ordine di esecuzione, warm/cold) o appaiarlo esplicitamente.
5. Igiene misura: trace off nei bench; manifest per ogni run; server spenti a fine; t/s pod
   marcati "non confrontabile col 3060"; commit in inglese stile repo, staging selettivo.
6. La serie patch canonica è `patches/README.md`; numeri nuovi partono da 0019.

## §GROUND TRUTH (puntatori, non copiare numeri qui)

- `docs/CLAIMS_CURRENT.md` — stato dei claim (single source of truth)
- `docs/NEXT_STEPS_PLAN_20260710.md` — piano fasi/test/leve
- `runs/ds4/20260710_retro_grade_l0l3/REPORT.md` — retro-grade L0-L3 dei 105 output (87/89 HTML = L0)
- `runs/ds4/20260710_w50_rotate32_k23_cache256_html4000/ANALYSIS.md` — non-determinismo greedy + repetition-lock
- `runs/ds4/20260710_pod_t1_full_positive_control/README.md` — T1: budget-confound dimostrato; firma loop resta della mask; test cyberpunk validi solo a ~4000 tok o prompt compatto
- `patches/README.md` — mappa canonica patch

## §MANDATO CORRENTE — M1 (aperto 2026-07-10)

**Contesto**: il run W50 ctx8192 è il **primo output dell'intero corpus a emettere `</html>`**
(prima: 0/89). È il punto di partenza reale — ma è n=1, e a ctx8192 il W100 ha comunque
loopato nello script. Obiettivo di M1: trasformare quel segnale in un risultato replicato e
verificare se lo stopper anti-ripetizione sblocca pagine **funzionali**.

**M1a — Replica n=3 di W50 vs W100 a ctx8192** (richiede la scheda locale: coordinarsi con
l'utente su quando è libera).
Config: `ctx=8192`, `server_max_tokens=4096`, `max_tokens=4000`, `cache-experts=256`, K23,
rotate32 (`every=32`, `decay=0.98`), prompt `html` standard, W∈{50,100}, **`--runs 3`**,
`--order abab`. Grading per-seed con `functional_grade.py`.
Deliverable: per ogni seed {emette `</html>` sì/no, L-level, coherent_until, onset loop} +
mediana; verdetto su "W50>W100?" e "ctx8192 fa completare il documento?" come
**distribuzioni**, non punti singoli.

**M1b — Anti-repetition stopper + retry** (leva §5(a) di ANALYSIS.md).
Implementare lo stop su tripla ripetizione verbatim (n-gram n=3, finestra 120, o blocco-riga
ripetuto ≥3x) — in engine se pulito, altrimenti prima versione lato client/runner; cintura
secondaria: stop-string su `</html>`. Poi rieseguire M1a-W50 **con stopper attivo** (n=3) e
misurare: token risparmiati, L-level, e se stop→retry (1 retry) produce una pagina funzionale
dove il run liscio degenerava.

**M1c — Correggere l'artefatto prefill nel ledger.**
I 266s di `prompt_s` del W100 ctx8192 erano prefill a cache fredda (ordine di esecuzione),
non "warmup K0 che riempie la VRAM" (il W50 prefilla lo stesso prompt in 57s; il warmup
agisce DOPO `prompt done`). Correggere la nota dove registrata; d'ora in poi misure di
velocità solo a stato cache appaiato.

**M1d (opzionale)** — Aggiungere la colonna `level` (L0-L3) alle righe nuove del master
ledger, sostituendo il readout repeat-based.

**Definition of done**: run committati con summary + grade per-seed; §LOG aggiornato con
esiti, deviazioni e tempo speso; CLAIMS toccato solo se un finding cambia stato (e per primo).

## §LOG MANDATI

- **M1** — aperto 2026-07-10, in corso. (Chiuderlo qui con: esiti M1a-d, commit hash, deviazioni.)
