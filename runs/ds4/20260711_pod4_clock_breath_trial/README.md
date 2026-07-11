# Clock-breath fair trial (D6b, leva L6) — orizzonte lungo, in-engine — 2026-07-11

**Domanda:** il "respiro" originale dell'utente (mask keep-K statica + finestra a
orologio che apre a K0, ri-impara la mask dalla domanda onesta, richiude) riabilita
il task largo a orizzonte lungo, dove la statica collassa? E la previsione del
decision model (fa35dd6) — "breath mai raccomandato; a K23-wide marginale;
K12+breath PERDE" — regge alla validazione diretta?

**Verdetto secco: il clock-breath è una PERDITA PURA a orizzonte lungo.
Beneficio qualità = ZERO ESATTO (output bit-identici alla statica, 8/8 md5
uguali, incluso cadenza 128); tassa velocità 48-72% (vs 13-17% assunta da D6b).
La previsione del decision model è CONFERMATA e RAFFORZATA: era ancora troppo
generosa col breath (assumeva hazard dimezzato + tassa 15%; misurato: hazard
invariato + tassa 3-5x).**

## Setup

- Pod4: RunPod **community RTX 3090 Ti** `0htxln87674tjq` (machine `bfqzozxd2xav`),
  sm_86, driver 575.51.03, 42 vcpu / 72 GB RAM, **$0.27/h**. Gate-check CUDA
  PASS (compute reale torch). Due deploy 3090 precedenti = STESSO host UVM-rotto
  `d5r94zi8wuwh` riassegnato (memoria confermata: cambio gpuTypeId → host sano al
  primo colpo). RAM 72 GB < modello 81 GB ⇒ page-cache parziale; t/s = numeri
  pod, diagnostici, NON confrontabili col 3060.
- Binario: **canonical v2** `ds4_sm86_canonical-62ed2e71-v2` da R2 (sha256
  verificata vs .meta; build dello switchover, commit pod2 `bfa987e`). Modello
  `ds4-2bit.gguf` da R2, sha256 `efc7ed60…` verificata. Boot-test PASS (32 tok
  coffee, PACE on, rc=0).
- Attuatore: **in-engine PACE clock-breath** (stream continuo, KV intatto — il
  D6b fedele). Ciclo verificato nel pace.jsonl:
  `prefill_apply keep=23` → `breath(clock) keep=0` → `relearn` → `breath_end
  keep=23`. `DS4_PACE_DRIFT=99` spegne il trigger n-gram ⇒ **solo orologio**.
  Config: WARMUP=50, KEEP=MIN=MAX=23 (pinned), BREATH_EVERY=450, BREATH_LEN=70,
  BREATH_KEEP=0 (K0 pieno), RELEARN=1, DECAY=0.3, WRAP=1, greedy temp 0,
  `--ssd-streaming-cache-experts 1024`, trace off.
- **Ricostruzione offline RIGETTATA con evidenza** (`_offline_validate_rejected/`):
  il multi-fase re-prefill `[prompt+HTML parziale]` fa RIPARTIRE il documento
  (doctype×5, L0 su coffee banale) — artefatto da re-prefill assente
  nell'attuatore continuo. Non è un canale fedele per il trial multi-respiro.
- **Caveat dichiarato (provenienza mask):** in questo binario il PACE impara la
  mask dal routing del PREFILL (`prefill_apply` a tok 0), non dai primi 50 tok
  GENERATI come la two-phase offline. Mask più dura: collasso wide a ~126 tok
  (vs MTTC ~1-2k delle statiche offline d3b4614/pod3). Vale per ENTRAMBI i
  bracci ⇒ il confronto breath-vs-static resta internamente valido; i livelli
  assoluti NON sono confrontabili con le baseline offline.

## Bracci (16 run)

| arm | prompt | ctx | budget | cadenza | keep | n |
|---|---|---|---|---|---|---|
| A1_coffee_breath | coffee | 4096 | 1200 | 450 | 23 | 3 |
| A1_coffee_static | coffee | 4096 | 1200 | — | 23 | 3 |
| A2_cyber_breath | cyberpunk | 8192 | 4050 | 450 | 23 | 3 |
| A2_cyber_static | cyberpunk | 8192 | 4050 | — | 23 | 3 |
| A2b_cyber_breath_early | cyberpunk | 8192 | 4050 | **128** | 23 | 2 |
| A3_cyber_k38_breath | cyberpunk | 8192 | 4050 | 450 | **38** | 2 |

Grading `functional_grade.py` L0-L3; collasso = primo blocco ≥24 char ripetuto
≥3× O sequenza contatore (≥12 numeri consecutivi — nuova firma, vedi F6).

## Risultati per-seed (GRADED.csv / SUMMARY.json)

| arm | L per run | collasso@tok | </html> | breaths | gen t/s med |
|---|---|---|---|---|---|
| A1_coffee_breath | 0,1,0 (med **L0**) | 123, —, 43 | 0,1,0 | 2,0,2 | 13.5 |
| A1_coffee_static | 1,0,1 (med **L1**) | —, 32, — | 1,0,1 | 0 | 16.4 |
| A2_cyber_breath | 0,0,0 | 126×3 | 0 | 7×3 | 12.5 |
| A2_cyber_static | 0,0,0 | 126×3 | 0 | 0 | **24.3** |
| A2b_cyber_breath_early | 0,0 | 126×2 | 0 | 20×2 | 6.7 |
| A3_cyber_k38_breath | 0,0 | 121×2 | 0 | 7×2 | 4.3 |

## Finding chiave

- **F1 — Beneficio del breath = ZERO ESATTO sul task largo.** Gli 8 run
  cyberpunk K23 (breath-450 ×3, static ×3, breath-128 ×2) sono
  **BIT-IDENTICI** (md5 `63aba520…` ×8). Venti finestre K0 + venti relearn non
  cambiano un byte. A3 K38: coppia bit-identica separata (`92beb7eb…`).
- **F2 — Il loop è sostenuto dal CONTESTO, non dalla mask (C1 in forma
  massima).** Il collasso parte a tok ~126, PRIMA di ogni orologio pratico. Il
  breath-128 fira a tok 128 — 2 tok dopo l'innesco, n-gram ancora 0.107 — apre
  a K0 pieno per 70 tok e la continuazione greedy è comunque identica. Quindi
  anche a K0 il modello continua il loop: nessuna cadenza può salvare.
- **F3 — "Demand evaporation" raffinata:** durante le finestre K0 `hit=1.0000`
  sempre — la domanda "onesta" del gate a mask aperta cade INTERAMENTE dentro
  il keep-set. La domanda non "riappare" al breath perché il contesto avvelenato
  instrada dentro la mask comunque; il relearn dalla finestra impara il loop.
- **F4 — Tassa di velocità 3-5× oltre l'assunzione D6b:** static 24.3 t/s vs
  breath-450 12.5 (**-48.5%**) vs breath-128 6.7 (**-72%**); wall 183s → 340s →
  616s a parità di budget E di output. Puro overhead di attuazione (WRAP
  bulk page-in alle transizioni K) per delta-output nullo.
- **F5 — Sul task stretto il breath è ≤0 anche lì:** coffee static med L1 (2/3
  chiudono da sole, button+form cablati); coffee breath med L0 — l'unico run
  pulito (r01) ha chiuso PRIMA che il primo respiro firasse; i 2 run respirati
  non chiudono mai. Nessun run collassato è mai stato recuperato da un respiro.
- **F6 — Nuova firma di degenerazione: il CONTATORE.** Coffee r02 degenera a
  tok ~43 in "211, 212, 213, …": ogni token è diverso ⇒ **invisibile sia al
  detector n-gram del motore (EWMA=0.0000 per tutto il run) sia al
  block-repeat del grader** (aggiunto detector dedicato). I detector attuali
  hanno un buco sulle sequenze monotone.
- **F7 — Determinismo:** cyberpunk greedy bit-identico run-to-run su questo
  host (8×+2×), coffee non-deterministico (3 esiti diversi a comando identico)
  — replica esatta del pattern del pod T1.

## Verdetto vs D6b e vs decision model (fa35dd6)

1. **D6b non si estende all'orizzonte lungo.** Il segnale positivo storico
   (160/320 tok, repeat=0, tassa contenuta) era semplicemente PIÙ CORTO
   dell'orizzonte di collasso. A ≥1200 tok il ciclo
   [tight→breath+relearn→tight] non previene, non ritarda e non recupera il
   collasso; paga solo la tassa. Il "reset della deriva" di D6b non esiste
   quando la deriva è già nel contesto.
2. **Previsione del decision model: CONFERMATA, con margine più netto del
   previsto.** Il modello (fa35dd6) diceva "breath mai raccomandato" ma lo
   modellava ancora generosamente come *hazard dimezzato + 15% tassa*
   (K23-wide breath "marginale", K12+breath 1.29 < K48-static 1.56). Misurato:
   **hazard INVARIATO (identità bit-level) + tassa 48-72%** ⇒ ogni riga
   "*+breath" della recovery-ladder è sovrastimata; il verdetto qualitativo
   (mai breath) esce rafforzato a fortiori.
3. **Corollario che motiva il pivot rewind (exp #2 del modello, patch 0022):**
   se anche K0 a 2 token dall'innesco continua il loop, l'unica uscita è un
   **edit del contesto** (rewind), non un edit della mask. Questo trial è
   l'evidenza empirica diretta della scelta "rewind SÌ, breath NO".

## Costi e stato pod

- Pod4 `0htxln87674tjq` (3090 Ti, $0.27/h): deploy 22:52Z, batch 23:30→00:55Z.
  ~2 fail-deploy 3090 (stesso host rotto) ~$0.01. Costo trial ≈ **$0.6-0.7**
  (entro cap $3). **Pod LASCIATO RUNNING** (regola utente: worker di riserva)
  con a bordo: modello verificato, binari canonical-v2 verificati, harness
  scripts, prompts — pronto al prossimo mandato.
- Lavoro 1 del mandato originario (build canonical v2): **già fatto da
  pod2-redeploy** (commit `bfa987e`, binari su R2 con .meta) — verificato e
  SALTATO senza duplicare (usato qui come binario del trial = smoke reale).

## File

- `A*/r0*/gen.out|gen.err|pace.jsonl|status.txt` — output grezzi, diag ds4,
  log eventi PACE per run; `batch_progress.log`; `run_batch.sh` — script
  eseguito sul pod; `GRADED.csv` / `SUMMARY.json` — grading per-seed e
  aggregato; `_offline_validate_rejected/` — evidenza del rigetto del canale
  offline (doctype×5 restart artifact).
