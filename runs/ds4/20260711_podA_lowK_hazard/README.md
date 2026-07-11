# 2026-07-11 podA — hazard a K basso (K=12, K=16), two-phase freeze-safe

Mandato: identificazione del parametro hazard-a-K-basso per il decision model
(mai misurato sotto K=23). Protocollo IDENTICO al gruppo W50 di
`runs/ds4/20260710_t4_t5_w_sweep_local` (harness
`scripts/run_w_sweep_freeze_safe.py`: W=50, weighted via
`scripts/build_session_mask_canonical.py`, freeze-safe via
`scripts/freeze_boundary.py`, strip fence non-leading b91188d, greedy, trace
routing solo fase-1, manifest), con la SOLA leva cambiata: keep per layer
23 → 12 e 16.

## Setup

- Pod: RunPod community RTX 3090 `ysegg4bx67yvr3` (machine `hyuu5efkuyma`,
  $0.22/h, 128 vcpu / 251 GB RAM ⇒ **regime RAM-hot: qualita' confrontabile,
  tempi NO** — t/s marcati POD, non confrontare col 3060 locale), image
  cu1290 collaudata, CUDA gate-check PASS al primo colpo.
- Binario: `ds4_sm86_livetree-771a39a8` da R2, sha256 `772c502f…` verificato
  (stesso lineage post-0018 dei bracci W50 locali e pod2). Modello sha256
  `efc7ed60…` verificato.
- Celle: (1) K12 coffee 1200 tok ctx4096 n=3; (2) K16 coffee n=3;
  (3) K12 cyberpunk 2500 tok ctx8192 n=2; (4) K16 cyberpunk n=2.
  Env IO identico al driver locale (`DS4_CUDA_NO_DIRECT_IO=1` ecc.).
- Grading L0-L3 `scripts/functional_grade.py` (pod senza node ⇒ JS check
  euristico, come i run pod precedenti).
- **Onset di collasso**: `onset_probe.py` (in questa dir) — scansione
  post-hoc del deliverable: primo tra (a) secondo `<!doctype` (doc-restart),
  (b) primo match del repeat-regex del harness. Token = char/4 (STIMA, stessa
  convenzione di `freeze_boundary._default_token_len`), colonna
  `onset_tok_est` in `onset.csv` per cella.

## Tabella K x prompt x seed (L e token-onset)

| K | prompt | run/seed | L | onset kind | onset tok (est) | chars | repeat | doctype | p2gen t/s (pod) |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 12 | coffee | r00/s0 | **L1** | doc_restart | 44 | 5396 | 1 | 2 | 3.42 |
| 12 | coffee | r01/s1 | **L1** | doc_restart | 42 | 4908 | 1 | 2 | 3.68 |
| 12 | coffee | r02/s2 | **L1** | doc_restart | 50 | 4287 | 1 | 2 | 3.65 |
| 16 | coffee | r00/s0 | **L2** | doc_restart | 52 | 2031 | 0 | 2 | 3.63 |
| 16 | coffee | r01/s1 | **L0** | doc_restart | 14 | 2136 | 1 | 2 | 3.64 |
| 16 | coffee | r02/s2 | **L1** | doc_restart | 50 | 1489 | 0 | 2 | 3.61 |
| 12 | cyberpunk | r00/s0 | **L0** | doc_restart | 19 | 7526 | 1 | 2 | 3.64 |
| 12 | cyberpunk | r01/s1 | **L0** | doc_restart | 19 | 7526 | 1 | 2 | 3.65 |
| 16 | cyberpunk | r00/s0 | **L0** | doc_restart | 19 | 13644 | 1 | 2 | 3.64 |
| 16 | cyberpunk | r01/s1 | **L0** | doc_restart | 19 | 13644 | 1 | 2 | 3.58 |

Mediane: K12 coffee **L1** (vs K23 coffee L2 storico), K16 coffee **L1**
(spread L0-L2), K12/K16 cyberpunk **L0** unanime.

## Letture secche (per il fit hazard)

1. **Doc-restart universale 10/10** (doctype=2 in ogni run, anche nei
   migliori): a K basso la fase-2 non continua MAI il frozen — riparte.
   Il flag `restart` di summary.csv e' 0 nei run L0/L1 solo perche' il
   grader lo valorizza a livello >=2; fare fede a `doctype` e `onset.csv`.
2. **Onset molto precoce e stabile**: coffee ~tok 42-52 (3/3 K12; 2/3 K16),
   cyberpunk ~tok 19 (4/4). L'outlier K16 coffee r01 (onset 14) e' il freeze
   corto `>`@14: onset ≈ subito dopo il frozen. L'onset coincide col bordo
   frozen→fase-2: l'hazard a K∈{12,16} e' concentrato all'inizio della
   fase-2, non distribuito lungo la generazione.
3. **Loop-rate sale scendendo di K**: repeat 3/3 a K12 coffee vs 1/3 a K16
   coffee (K23 storico: 0/3). Sul cyberpunk repeat 2/2+2/2 come K23
   (il prompt largo collassa gia' a K23 static, T1/static-AB).
4. **Determinismo greedy cross-run sul cyberpunk anche a K basso**: deliverable
   byte-identici r00=r01 sia K12 (7526 B) sia K16 (13644 B) — stesso
   comportamento dei pod static A/B a K23.
5. Gradiente qualita' K: coffee mediana L1(K12) ≤ L1(K16, spread piu' largo
   con un L2) < L2(K23 storico) — monotono debole ma con n=3 il segnale
   forte e' onset+loop, non la mediana L da sola.

## Costo

Pod A community $0.22/h; gruppo A ~1h21m di run effettivo (23:58→01:20 UTC)
+ bootstrap ~25m; spesa track (incl. primo pod fallito senza SSH ~$0.06 e
gruppo B narrow-traces) ≈ **$1.1-1.3** al momento della chiusura report,
entro cap $3. Pod lasciato RUNNING (regola utente).
