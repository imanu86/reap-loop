# 2026-07-11 Pod B — temperature sweep sotto mask (estende pod2 SAMPLING-UNDER-MASK)

DOMANDA (identificazione parametro): pod2 (`20260710_pod2_sampling_under_mask`) ha
misurato **un solo punto** (temp 0.7/top_p 0.95, coffee, n=3): mediana L2 = greedy,
loop 0/3 = greedy, nessuna leva. Qui si **estende il sweep di temperatura** —
0.5 e 0.8 sul coffee (pod2 non li aveva coperti) più il **primo punto cyberpunk**
sotto mask campionata (pod2 non aveva mai toccato il prompt duro) — per rispondere:
il greedy amplifica la fragilità sotto mask in modo *temperatura-dipendente*, o la
finestra 0.5–0.8 è piatta come il singolo punto 0.7 di pod2?

## Setup

- Pod: RunPod `d39bqszgmywel2` (**SECURE** RTX 3090 24GB, $0.46/h, machine
  `u367jo12dcl5`). Due deploy COMMUNITY falliti prima di questo (host
  `hyuu5efkuyma` poi `6p17plgwgpsn`): SSH TCP diretto mai arrivato a bind
  (`connection refused`) per >6-20 min su entrambi pur con pod `RUNNING`/porta
  mappata — sintomo di rete host rotta lato community, non un problema
  applicativo (nessun gate-check CUDA raggiunto, quindi non lo stesso guasto
  UVM già noto). Terminati subito, ri-deploy su **SECURE** con `PUBLIC_KEY`
  iniettata esplicitamente come env del pod (oltre a `RCLONE_CONFIG_R2_*`):
  SSH riuscita al primo colpo.
- **Gate CUDA: PASS** (`torch.cuda.is_available()=True`, matmul reale su
  device OK). `nvcc` non nel PATH ma irrilevante: nessun build, solo binari
  R2.
- Binario: `ds4_sm86_livetree-771a39a8` da R2 (stessa lineage post-0018 di
  pod2; sha256 verificato). Modello `ds4-2bit.gguf` da R2, sha256 verificato
  (86 720 111 488 B).
- Harness: `scripts/run_w_sweep_freeze_safe.py` (stesso hotfix fence
  non-leading + `--top-p`/`--sample-p2-only` di pod2): fase-1 GREEDY (trace/
  mask/freeze costruiti come nei bracci greedy), il sampling si applica SOLO
  alla fase-2 mascherata.
- Config comune: W=50, K23 static weighted, two-phase freeze-safe, cache 256,
  `DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1
  DS4_REAP_PREFETCH_THREADS=16 DS4_REAP_PREFETCH_LOCK=1` (come driver
  locale/pod2).
- **Coffee** (frontpage_prompt.txt, 817B): `--total 1200` (fase2 ≈1150 tok),
  ctx 4096/4096, temp ∈ {0.5, 0.8}, top_p 0.95, n=3/temp, seed_base 50
  (0.5→seed 50-52) e 80 (0.8→seed 80-82) — seed diversi da quelli già usati
  da pod2 (42-44 @ temp 0.7).
- **Cyberpunk** (cyberpunk_prompt.txt, 199B, byte-esatto agli altri pod):
  `--total 2500` (fase2 ≈2450 tok — budget scelto per stare nel cap di
  tempo/costo, **non** i 4000 di armA: risultati L0 qui NON sono comparabili
  in "chiusura pagina" con armA, solo nel confronto loop/fenotipo a parità di
  budget ridotto), ctx 8192/8192, temp 0.7, top_p 0.95, n=2, seed_base 70
  (seed 70-71). Pod2 non aveva mai testato cyberpunk sotto sampling.

## Risultati per-seed

### Coffee temp=0.5 (n=3, seed 50-52)

| run | seed | freeze | L | restart | `</html>` | loop(repeat) | onset (tok, stimato) | chars | p2gen t/s (pod) |
|---|---:|---|---|---|---|---|---:|---:|---:|
| r00 | 50 | `;`@50 | **L2** | 1 | 1 | no | — | 2650 | 2.33 |
| r01 | 51 | `;`@50 | **L1** | 0 | 0 | **SI** | ~215 | 5144 | 2.32 |
| r02 | 52 | `;`@50 | **L2** | 1 | 1 | no | — | 2510 | 2.74 |

Mediana **L2**; loop-rate **1/3**; restart (doppio doctype) **2/3**.

### Coffee temp=0.8 (n=3, seed 80-82)

| run | seed | freeze | L | restart | `</html>` | loop(repeat) | onset (tok, stimato) | chars | p2gen t/s (pod) |
|---|---:|---|---|---|---|---|---:|---:|---:|
| r00 | 80 | `;`@50 | **L1** | 0 | 1 | no | — | 1645 | 2.75 |
| r01 | 81 | `;`@50 | **L1** | 0 | 1 | no | — | 1895 | 2.76 |
| r02 | 82 | `>`@44 | **L1** | 0 | 0 | **SI** | ~169 | 3364 | 2.77 |

Mediana **L1**; loop-rate **1/3**; restart **0/3**.

### Cyberpunk temp=0.7 (n=2, seed 70-71, budget ridotto 2500 vs 4000 di armA)

| run | seed | freeze | L | restart | `</html>` | loop(repeat) | onset (tok, stimato) | chars | p2gen t/s (pod) |
|---|---:|---|---|---|---|---|---:|---:|---:|
| r00 | 70 | `>`@17 | **L0** | 0 | 0 | **SI** | ~54 | 5656 | 2.34 |
| r01 | 71 | `>`@17 | **L0** | 0 | 0 | **SI** | ~56 | 5223 | 2.78 |

Mediana **L0**; loop-rate **2/2**; restart 0/2.

*Nota metodologica onset:* stima diagnostica = conteggio parole (split su
whitespace) nel testo del deliverable fino all'inizio del primo pattern
24-160 char ripetuto ×3 (stesso regex del flag `repeat` dell'harness). Non è
un indice di token generati realmente (l'harness CLI-direct non ha
stream-events per-token in questo path) — coerente in spirito con le stime
"loop onset (est)" già usate altrove nel repo (M1a/armA), ma da trattare come
ordine di grandezza, non cifra esatta.

## Confronto con i greedy noti (stessa config W50 K23 static)

### Coffee (budget 1200, ctx4096)

| braccio | temp | L per-seed | mediana L | loop-rate | restart-rate |
|---|---|---|---|---|---|
| **temp 0.5 (questo)** | 0.5 | 2,1,2 | **L2** | **1/3** | 2/3 |
| **temp 0.8 (questo)** | 0.8 | 1,1,1 | **L1** | **1/3** | **0/3** |
| pod2 sampled | 0.7 | 2,2,2 | **L2** | **0/3** | 3/3 |
| greedy locale `t4_W050` | 0 | 2,0,2 | L2 | 0/3* | 2/3 |
| greedy pod1 S5 `s5_coffee_k23` | 0 | 2,1,2 | L2 | 0/3* | 2/3 |

\* Nei greedy noti l'unico L<2 di ciascun gruppo è un **artefatto** non da
loop-sotto-mask (locale: fase-2 vuota/no-tps; S5 r01: fence pre-hotfix,
freeze=none) — non un repeat genuino. Nei due nuovi bracci sampled, invece,
l'L basso (T050 r01, T080 r02) **è un repeat genuino** rilevato dal grader.

### Cyberpunk (budget ridotto 2500 vs 4000 di armA — non stessa profondità)

| braccio | temp | budget fase2 | L per-seed | mediana L | loop-rate |
|---|---|---:|---|---|---|
| **temp 0.7 (questo)** | 0.7 | 2450 | 0,0 | **L0** | **2/2** |
| armA cyberpunk static greedy | 0 | 4000 | 0,0,0 | L0 | 3/3 |

## VERDETTO (grezzo)

- **La temperatura NON riduce i loop sotto mask — anzi, nella finestra
  0.5-0.8 li INTRODUCE dove il greedy validato non ne aveva**: pod2 (temp
  0.7) e i due greedy noti hanno loop-rate genuino 0/3; qui temp 0.5 e temp
  0.8 mostrano **1/3 ciascuno** (un repeat vero per gruppo, non un artefatto
  fence/vuoto). Il singolo punto "buono" di pod2 (0.7, 0 loop) sembra
  un'isola, non l'inizio di un trend — servirebbe più segnale (n maggiore)
  per dire se è il *seed* o la *temperatura* a pesare qui, ma il segnale
  grezzo NON supporta "più sampling = meno loop".
- **Costo in L: sì, misurabile a temp 0.8** — mediana L1 contro L2 di temp
  0.5/0.7/greedy: 3/3 run a temp 0.8 restano "aperti ma con difetto" (L1),
  nessuno raggiunge L2. A temp 0.5 la mediana tiene (L2) nonostante il loop
  isolato.
- **Segnale collaterale (non richiesto ma visibile nei dati): temp 0.8
  azzera il restart (doppio doctype) 0/3**, contro 2/3 (temp 0.5), 2/3
  (greedy noti), 3/3 (pod2 temp 0.7) — l'unica cella di questo sweep dove
  l'attrattore restart non compare mai. Coerente con "la temperatura sposta
  la traiettoria lontano dal cut point pericoloso" ma qui il prezzo è un
  fenotipo L1 (bottone/form non wired) anziché il restart.
- **Cyberpunk: nessuna riduzione di loop dal sampling, fenotipo identico al
  greedy** (loop 2/2 qui, 3/3 in armA — entrambi ~100%, differenza non
  significativa su n così piccoli): conferma T1/pod2, estesa al prompt duro
  — a budget insufficiente per chiudere la pagina (~3500 tok servirebbero,
  qui il budget è 2450) il sampler non salva né peggiora sistematicamente il
  collasso.
- **Onset dei loop (stima):** quando il loop compare, arriva PRESTO — coffee
  ~169-215 parole (sul totale ~1150 tok di fase2), cyberpunk ~54-56 parole
  (su ~2450 tok fase2, appena dopo il freeze `>`@17). Non c'è un pattern
  "il sampling ritarda il collasso": quando collassa, collassa quasi subito
  dopo l'innesco mascherato.
- **In sintesi (risposta alla missione):** il greedy NON viene reso più
  fragile dal sampling in modo monotono con la temperatura in questo range
  — il quadro è **piatto/rumoroso** (n=3 per cella è poco per separare
  seed-variance da temp-effect), ma l'unica cella con costo di qualità
  chiaro è **temp 0.8 → mediana L1** (perdita di un livello) senza guadagno
  compensativo sul loop-rate. Nessuna delle tre celle sampled batte il
  miglior greedy noto (L2, loop 0/3) su entrambi gli assi contemporaneamente.

## Costo e stato pod

- RunPod balance prima del track: **$18.06** → dopo: **$14.09** (calo
  aggregato **-$3.97**, ma sul balance insistono ANCHE gli altri 3 pod
  concorrenti già RUNNING di altri ruoli/sessioni — `pod4-worker-canonical-v2`
  $0.27/h, `pod2-r2-redeploy` $0.46/h, `ds4-static-ab-fresh` $0.22/h — per
  ~1.5h di finestra: non attribuibile a questo track).
- **Spesa di QUESTO track**: 2 deploy community abortiti (SSH mai raggiunta,
  terminati entro 10-20 min, $0.22/h ⇒ ~$0.07-0.11) + pod secure
  `d39bqszgmywel2` uptime ~84 min a $0.46/h ⇒ **~$0.64**. **Totale track
  ≈ $0.75**, ben sotto il cap $3.
- **Pod NON terminato** (regola utente: a fine sessione non stoppare).
  Handoff: `ssh -i ~/.ssh/id_ed25519 -p 40135 root@213.192.2.117`, ds4 in
  `/root/ds4/ds4`, modello in `/root/models/ds4-2bit.gguf`, harness in
  `/root/reap-loop/scripts/`, risultati sorgente ancora su
  `/root/reap-loop/runs/ds4/20260711_podB_temp_sweep/` (già copiati qui).

## Incidente di sicurezza (dichiarato)

Durante il debug del bootstrap R2 su questo pod, un comando diagnostico
(`grep -i RCLONE /etc/environment` seguito da un secondo tentativo su
`/etc/rp_environment`) ha **stampato in chiaro** l'Access Key ID e la Secret
Access Key R2 nell'output di un tool-call della sessione (non in un file di
repo, non in un commit — ma nel transcript della sessione stessa). Corretto
immediatamente lo script per caricare le credenziali senza mai più
echeggiarle. **Raccomandazione: ruotare le credenziali R2 (Access Key ID +
Secret) nel dashboard Cloudflare e aggiornare `cf.txt`.**

## File

Layout harness standard per gruppo (`coffee_T050/`, `coffee_T080/`,
`cyberpunk_T070/`): `W050/r00../rNN/` (route.csv, tw.txt, frozen.txt,
sess.txt, p2prompt.txt, trest.txt, deliverable.html, p1/p2.diag),
`summary.csv`, `summary_median.csv`, `manifest.json`, `VERDICT.txt` (campo
monotonia non applicabile: singolo W). Più: `pod_progress.log` (log harness
completo del pod), `combined_temp_sweep.csv`/`.md` (tabella aggregata
cross-gruppo con stima onset) e `aggregate_onset.py` (script di
post-processing usato per generarle: legge `summary.csv` +
`deliverable.html` per run, stima l'onset in parole fino al primo match di
`(.{24,160})\1\1`; riusabile con
`python aggregate_onset.py <label>:<outdir> ...`).
