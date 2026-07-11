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

**Secondo abort mirato (solo armA/armB, 17:10→17:28Z):** sul prompt
cyberpunk (italiano) la fase-1 apre SEMPRE con prosa italiana prima della
fence («Ecco una landing page cyberpunk…», confermato su prep-smoke e armA
r00) ⇒ lo strip @92406ce (solo fence *leading*) non scatta MAI su questo
prompt ⇒ armA r00 congelato con taglio grezzo DENTRO un attributo
(`content="width=device-width, initial`) = patologia J44 ⇒ fase-2 in loop
(`'scp', 'scp', …`). Poiché il mandato per (a)/(b) è esplicitamente «freeze
sicuro via freeze_boundary», l'harness sul pod è stato esteso in modo
POD-LOCALE (splice della sola `strip_markdown_fence`: salta un prologo in
prosa ≤600 char fino alla prima riga fence-open; py_compile OK; test
prose/lead/plain/closed passati; freeze `>` within-target sul caso prose) e
armA+armB rilanciati da zero alle 17:28:38Z. L'r00 rotto è conservato come
evidenza (`armA_k23_rawcut_evidence/`). L'arm S5 NON è stato rifatto: la sua
ragione d'essere è la parità harness col T4 locale @92406ce. Fix da
integrare upstream dall'owner dell'harness (utile anche al T4 locale sui
prompt che elicitano prosa).

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

### (a) static K23 — cyberpunk W50, fase2 4000, ctx8192, n=3 (DONE 20:34Z, rc=0)

| run | freeze | freeze_tok | L | `</html>` | repeat | form | script | chars | p2 gen t/s (pod) |
|---|---|---:|---|---|---|---|---|---:|---:|
| r00 | `>` | 17 | **L0** | 0 | **1** | 0 | 0 | 13424 | 1.17 |
| r01 | `>` | 17 | **L0** | 0 | **1** | 0 | 0 | 13424 | 1.12 |
| r02 | `>` | 17 | **L0** | 0 | **1** | 0 | 0 | 13424 | 1.02 |

**I 3 deliverable sono BYTE-IDENTICI** (sha256 `bbf29589…` × 3): la mask
statica ripristina il determinismo greedy pieno che rotate32 rompeva (M1a:
3 seed divergenti, onset di loop diversi). Coda del deliverable: loop
`https.com.` ripetuto. Nota interpretativa: dopo lo strip prosa+fence il
prefisso congelato è ~17 token di HTML (fino a `<meta charset>`); la mask
W50 è appresa sui 66 token di fase-1 (in maggioranza prosa italiana) —
stessa distribuzione dei primi ~50 token che vede il warmup in-engine M1a
su questo prompt (parità mantenuta).

### (b) static K38 (cov90 E-CAL) — cyberpunk W50, fase2 4000, ctx8192, n=3 (DONE 00:18:52Z, rc=0)

| run | freeze | freeze_tok | L | `</html>` | repeat | form | script | chars | p2 gen t/s (pod) |
|---|---|---:|---|---|---|---|---|---:|---:|
| r00 | `>` | 17 | **L0** | 0 | **1** | 0 | 0 | 20781 | 1.00 |
| r01 | `>` | 17 | **L0** | 0 | **1** | 0 | 0 | 20781 | 0.90 |
| r02 | `>` | 17 | **L0** | 0 | **1** | 0 | 0 | 20781 | 0.86 |

Anche qui **3 deliverable BYTE-IDENTICI** (sha256 `cfa6b7ff…` × 3). Coda:
loop `background: linear-gradient( 220);`. K38 produce ~55% più testo di
K23 prima/dentro il loop (20781 vs 13424 char) — traiettoria più lunga, ma
stesso esito funzionale L0 senza chiusura.

## Confronto con M1a rotate32 (per-seed)

M1a (locale 3060, ctx8192, html 4000, cache256, rotate32 decay 0.98):

| arm | r01 | r02 | r03 | `</html>` | loop onset (est) |
|---|---|---|---|---|---|
| M1a W50 rotate32 | L0 | L0 | L0 | 0/3 | —, 118, 757 |
| M1a W100 rotate32 | L0 | L1 | L0* | 0/3 | 617, 469, 586 |

(*r03 stream_failed a 1799 eventi.) Tutti 6/6 senza `</html>`; loop
ngram3_window120 in 5/6.

Quadro completo cyberpunk ~4000 tok (L per-seed / `</html>`):

| config | dove | L per-seed | `</html>` | deterministico? |
|---|---|---|---|---|
| FULL no-mask (T1) | pod secure | **L2** @3498 (finish=stop) | **1/1** | sì (greedy ident.) |
| rotate32 W50 K23 (M1a) | 3060 | L0, L0, L0 | 0/3 | **no** (onset 118–757) |
| rotate32 W100 K23 (M1a) | 3060 | L0, L1, L0* | 0/3 | **no** |
| **static W50 K23 (a)** | pod community | L0, L0, L0 | 0/3 | **sì** (sha ident. ×3) |
| **static W50 K38 (b)** | pod community | L0, L0, L0 | 0/3 | **sì** (sha ident. ×3) |

## Verdetto

1. **Lo static ben costruito NON salva il cyberpunk lungo**: a parità di
   prompt/budget/ctx/cache di M1a, keep-23 statico weighted con freeze
   sicuro collassa in loop senza `</html>` su 3/3 seed — stesso fenotipo
   L0 del rotate32. Il collasso ctx8192/4000 sotto mask K è **robusto alla
   politica di selezione** (statica o rotante): non era un artefatto del
   rotate. Il controllo positivo T1 (FULL = L2 con chiusura a 3498 tok)
   esclude il budget-confound.
2. **K38 (pavimento cov90 E-CAL) NON aggiunge chiusura rispetto a K23**:
   solo una traiettoria ~55% più lunga prima dello stesso esito L0. Il
   "floor" di copertura cov90 non è un floor di qualità funzionale su
   questo prompt/regime.
3. **Lo static ripristina il determinismo greedy** che il rotate rompeva:
   6/6 deliverable statici byte-identici fra seed (K23 e K38), contro i
   6 output tutti diversi di M1a. Diagnosi pulita: la varianza per-seed di
   M1a era interamente indotta dalla dinamica rotate/streaming.
4. **Ponte S5 (lato pod)**: coffee W50 K23 statico = L2/L1/L2 con
   `</html>` 2/3 (unico fallimento = seed con freeze `none` da prosa
   pre-fence, difetto harness non di mask). Confronto finale col T4 locale
   3060 da fare a valle (stessa config @92406ce): per il gate S5 contano i
   seed con freeze sicuro.
5. Implicazioni (da NON scrivere in CLAIMS qui — elencate per il
   coordinatore): CLAIM sul collasso M1 va esteso "anche static"; la linea
   0026 (admission mirata) resta l'ipotesi aperta perché qui non testata;
   il cov90=38 come pavimento va retrocesso a "pavimento di copertura, non
   di qualità" su prompt lunghi.

## Costo / stato pod

- Pod `pdpgc8nck480gd` (community 3090, $0.22/h): creato 15:26:14Z 10-lug,
  batch completo 00:18:52Z 11-lug ⇒ ~8.9 h ≈ **$1.96** (sotto il cap $5).
- **Il pod resta RUNNING per ordine coordinatore** (decisione follow-up a
  valle dei verdetti). A bordo, riutilizzabile:
  - modello `/root/models/ds4-2bit.gguf` (86 720 111 488 B, verificato);
  - binari R2 `/root/bin/ds4` (livetree-771a39a8) e build post-0026
    `/root/src/ds4/{ds4,ds4-server}` (catena 0020→0021→0026, sm_86,
    0 warning; anche su R2 come `*_livetree-1db4f799-admit`);
  - sorgenti patchati `/root/src/ds4` + patch chain in `/root/src/`;
  - harness in `/root/work/scripts/` (con estensione prose-fence
    POD-LOCALE), prompt, `run_smoke26.sh` MAI eseguito (smoke 0026
    riassegnato a pod dedicato), risultati grezzi in `/root/work/out/`;
  - rclone configurato per `r2:ds4-models` (upload risultati già fatti:
    `tmp_s5_results.tgz`, `tmp_armA_results.tgz`, `tmp_armB_results.tgz`,
    `tmp_evidence.tgz`).
- Pod adottato `1o7eg981j8gmp4`: mai ripartito (5×HTTP 500 GPU esaurite),
  intatto in STOP con disco/modello/live-tree.
