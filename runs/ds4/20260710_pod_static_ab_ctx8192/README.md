# 2026-07-10 Pod — static A/B ctx8192 (W50 two-phase STATIC K23 / K38) + ponte S5

Bracci STATICI mancanti del quadro M1: M1a (locale 3060) ha misurato solo
rotate32 a ctx8192/4000 (collasso 0/6 `</html>`, L 0,0,0 / 0,1,0); qui la
STESSA domanda con mask **statica ben costruita** (two-phase weighted, freeze
sicuro) a parità di prompt (cyberpunk), budget fase-2 4000, ctx8192, n=3.
Più il primo data-point del **ponte di portabilità S5** (coffee W50 K23,
fase2 1200, stessa config del T4 locale in corso).

## Setup

- Pod: RunPod `pdpgc8nck480gd` (COMMUNITY RTX 3090 24GB, machine `p1c7qqs0r6i6`,
  ES, $0.22/h, 125 GB RAM host, disco 160 GB). Il pod adottato
  `1o7eg981j8gmp4` (SECURE, EU-CZ-1) NON è ripartito: 5 tentativi di resume
  API tutti HTTP 500 "not enough free GPUs on the host machine" (era stato
  stoppato alle 13:17:31Z, ~2h prima di questa sessione; resta intatto in
  STOP). Deploy fresco da ricetta R2 (docs/POD_R2_CACHE.md): secure 3090 /
  A5000 / A6000 / A40 tutti "no instances available" → community 3090.
- **Gate CUDA: PASS** (torch `device_count=1`, matmul reale OK — nessun
  problema UVM; la classe di guasto community descritta nel T1 README non si
  è presentata su questa macchina).
- RAM host 125 GB ⇒ il modello 81 GB è page-cached dopo il primo load:
  regime RAM-hot come gli altri pod-run del 2026-07-10.
  **Ogni t/s qui è "pod, NON confrontabile" col 3060 locale.**
- Binario: `ds4_sm86_livetree-771a39a8` dal bucket R2 (CLI; rotate32+sensor+
  PACE; supporta `DS4_REAP_MASK_FILE` + trace pesata). Modello
  `ds4-2bit.gguf` da R2 (86 720 111 488 byte, size verificata).
- Harness: `scripts/run_w_sweep_freeze_safe.py` **@ 92406ce (fence-strip)**
  (two-phase, freeze sicuro `scripts/freeze_boundary.py`, mask
  `scripts/build_session_mask_canonical.py --mode weighted`, grading
  `scripts/functional_grade.py` L0-L3). Greedy temp 0, trace routing SOLO in
  fase-1 (necessaria alla mask), niente trace in fase-2, manifest per arm.
- **Abort & relaunch (16:08→16:10Z):** il primo lancio usava l'harness
  pre-92406ce; verificato sul pod che la fase-1 del binario live-tree esce
  incapsulata in ```` ```html ```` (od -c su `tw.txt`: `` ` ` ` h t m l \n
  < ! D O C T Y P E ``) ⇒ senza fence-strip lo scanner di freeze è cieco
  (raw-cut lottery J44) e il ponte S5 non sarebbe confrontabile col T4
  locale (che usa lo script fixato). Batch killato a ~2 min dal via, `out/`
  azzerato, bundle rigenerato con lo script @ 92406ce, batch rilanciato.
  Nessun risultato del primo lancio è stato conservato.
- Prompt cyberpunk = byte-esatto `PROMPTS["html"]` del runner M1a
  (`cyberpunk_prompt.txt`, 199 byte). Prompt coffee = l'esatto
  `frontpage_prompt.txt` 819 byte del replay.

## Bracci (tutti n=3, W=50, two-phase, freeze sicuro)

| arm | prompt | K | fase2 budget | ctx p2 | cache | outdir |
|---|---|---|---|---|---|---|
| (s5) ponte S5 | coffee | 23 | 1200 (`--total 1250`) | 3072 | 256 | `s5_coffee_k23/` |
| (a) static K23 | cyberpunk | 23 | 4000 (`--total 4050`) | 8192 | 256 | `armA_k23/` |
| (b) static K38 | cyberpunk | 38 (cov90 E-CAL) | 4000 (`--total 4050`) | 8192 | 256 | `armB_k38/` |

## Risultati per-seed

### (s5) ponte S5 — coffee W50 static K23, fase2 1200, n=3 (DONE 16:42Z, rc=0)

| run | freeze | freeze_tok | L | restart | `</html>` | repeat | button+form wired | chars | p2 gen t/s (pod) |
|---|---|---:|---|---|---|---|---|---:|---:|
| r00 | `;` | 50 | **L2** | 1 | 1 | 0 | sì+sì | 2155 | 1.28 |
| r01 | **none** | 70 | **L1** | 0 | 0 | 1 | no+no | 6389 | 1.32 |
| r02 | `;` | 50 | **L2** | 1 | 1 | 0 | sì+sì | 1538 | 1.29 |

Mediane: L=2, p2_gen 1.29 t/s (pod, NON confrontabile), restart_majority=1.

**Anomalia harness r01 (per il confronto col T4 locale):** la fase-1 di r01
esce con **prosa PRIMA della fence** (`Here's a compact, complete HTML page…`
+ ```` ```html ````): lo strip @92406ce copre solo la fence *leading* ⇒
backtick nel testo ⇒ scanner cieco ⇒ `freeze=none` (taglio grezzo a 70 tok,
fence inclusa nel prefisso congelato) ⇒ fase-2 degenerata (repeat=1, L1,
niente `</html>`). r00/r02 (fence leading, strip OK) chiudono L2 con pagina
completa e popup. NB: greedy temp0 ma le 3 fasi-1 DIFFERISCONO (streaming
asincrono ⇒ non-determinismo, come M1a): la prosa-prima-della-fence è
stocastica, non da seed. Il fix suggerito (strip fence anche dopo un
prologo in prosa, o prompt "Output ONLY HTML" rispettato) è DA NON APPLICARE
in-batch: parità di config col T4 locale @92406ce.

Nota fedeltà: r00/r02 hanno `restart=1`/`doctype=2` — il deliverable
contiene il doctype del prefisso congelato E un secondo doctype emesso dalla
fase-2 (attrattore restart J44 parzialmente presente anche con freeze `;`
su questo binario/pod), ma il modello richiude comunque la pagina (L2).

## Confronto con M1a rotate32 (per-seed)

M1a (locale 3060, ctx8192, html 4000, cache256, rotate32 decay 0.98):

| arm | r01 | r02 | r03 | `</html>` | loop onset (est) |
|---|---|---|---|---|---|
| M1a W50 rotate32 | L0 | L0 | L0 | 0/3 | —, 118, 757 |
| M1a W100 rotate32 | L0 | L1 | L0* | 0/3 | 617, 469, 586 |

(*r03 stream_failed a 1799 eventi.) Tutti 6/6 senza `</html>`; loop
ngram3_window120 in 5/6.

TBD confronto.

## Verdetto

TBD

## Costo / stato pod

TBD
