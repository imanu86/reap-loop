# 2026-07-10 Pod2 (redeploy) — CODA W-SWEEP T4 freeze-safe (W 70/90/110/150, coffee, n=3)

Coda della griglia T4 spostata dal locale al pod2 (i gruppi locali
`t4_W070/090/110/150` contengono solo SKIPPED-MARKER). Stesso protocollo del
gruppo W50 locale (`runs/ds4/20260710_t4_t5_w_sweep_local/`): two-phase
freeze-safe, prompt coffee 819 B del replay, fase2 K23 static weighted
(`DS4_REAP_MASK_FILE`), `--total 1200` (fase2 ≈ 1100+ tok), ctx 4096/4096,
cache 256, greedy temp 0, trace routing solo fase-1, manifest per gruppo,
grading L0-L3 per-seed (`scripts/functional_grade.py`), harness con hotfix
fence non-leading b91188d.

**NOTA REGIME (obbligatoria): pod 3090 RAM-hot — qualità confrontabile,
tempi NO.** I t/s di questa pagina non sono confrontabili col 3060 locale
(headline per `DS4_RUNNER_PROTOCOL.md`).

## Setup pod

- RunPod `i7dk94f0y05iji` SECURE RTX 3090 24GB ($0.46/h, RAM host 1007 GB,
  256 vcpu), redeploy da ricetta R2 dopo 2 resume falliti del pod2 originale
  `o0gd30ojfacz96` (host community senza GPU libere; resta STOPPED col volume).
- Binario `ds4_sm86_livetree-771a39a8` da R2 (sha256 `772c502f…` verificata) =
  stesso lineage post-0018 del binario WSL del gruppo W50 locale. Modello
  `ds4-2bit.gguf` sha256-verificato. Gate-check CUDA passato al primo colpo
  (kernel_result=42).
- Driver: `tail_driver.sh` (sequenziale W90→W70→W110→W150, resume-safe,
  partito in coda al job sampling; log `tail_driver.log`).

## Tabella W × seed × L (per-seed, greedy)

| W | run | seed | freeze | L | restart | `</html>` | repeat | chars | p2 gen t/s (pod) |
|---:|---|---:|---|---|---|---|---|---:|---:|
| 70 | r00 | 0 | `}`@59 | **L2** | 1 | 1 | 0 | 2160 | 2.42 |
| 70 | r01 | 1 | `}`@59 | **L2** | 1 | 1 | 0 | 1943 | 2.44 |
| 70 | r02 | 2 | `}`@59 | **L2** | 1 | 1 | 0 | 2182 | 2.43 |
| 90 | r00 | 0 | `>`@44 | **L2** | 1 | 1 | 0 | 1828 | 2.45 |
| 90 | r01 | 1 | `;`@81 | **L0** | 0 | 1 | 0 | 995 | 2.38 |
| 90 | r02 | 2 | `;`@81 | **L0** | 0 | 1 | 0 | 1111 | 2.43 |
| 90 | (anchor) | 0 | `;`@85 | **L2** | 1 | 1 | 0 | 2669 | 3.11 |
| 110 | r00 | 0 | `;`@94 | **L2** | 1 | 1 | 0 | 2268 | 2.43 |
| 110 | r01 | 1 | `;`@90 | **L2** | 1 | 1 | 0 | 2595 | 2.70 |
| 110 | r02 | 2 | `;`@91 | **L1** | 0 | 1 | 0 | 1768 | 2.70 |
| 150 | r00 | 0 | `;`@125 | **L1** | 0 | 0 | **1** | 5405 | 2.72 |
| 150 | r01 | 1 | `;`@118 | **L2** | 1 | 1 | 0 | 2442 | 2.41 |
| 150 | r02 | 2 | `;`@118 | **L1** | 0 | 0 | **1** | 3138 | 2.86 |

(anchor) = smoke W90 n=1 pre-crash, config identica (`smoke_W090_anchor/`),
riportato come 4° data-point W90, escluso dalle mediane di gruppo.

Mediane: W70 **L2** · W90 **L0** · W110 **L2** · W150 **L1**.
`freeze_within_target=1` e boundary ∈ {`}` `;` `>`} su TUTTE le celle
(mai `none`): il confound J44 è rimosso ovunque.

## Lettura (con la testa locale della griglia)

Testa locale (3060, stesso harness/config): W30 L 2/0/1 (med **L1**),
W50 L 2/0/2 (med **L2**), W130 L 1/2/2 (med **L2**).

- **La scala W NON è monotona** nemmeno con freeze sicuro: med L1(30) → L2(50)
  → L2(70) → L0(90) → L2(110) → L2(130) → L1(150), spread per-seed alto
  (0-2 quasi ovunque). Conferma la lettura "lotteria del punto di freeze" J44
  SOLO in parte: qui il freeze è sempre safe/within-target, quindi la varianza
  residua è del modello (fase-1 non deterministica + attrattore restart), non
  del taglio.
- **Novità W150: compare la firma loop** (repeat=1 in 2/3, deliverable lunghi
  senza `</html>`) — il prefisso congelato più lungo NON aiuta: oltre ~W130 il
  two-phase degrada verso il loop. Prima firma di loop in tutta la coda.
- **W90 med L0 è l'outlier**: r01/r02 (freeze `;`@81) chiudono la pagina corta
  e "povera" (995/1111 char, `</html>` presente, niente restart, niente loop,
  ma senza i pezzi funzionali richiesti) — non è degenerazione, è
  under-delivery; l'anchor e r00 allo stesso W sono L2 pieni.
- **Cross-HW (portabilità S5)**: al pivot W50, locale 3060 (L 2/0/2) vs pod1
  S5 3090 (L 2/1/2) vs questo pod (sampling greedy-fase1: freeze `;`@50
  identico al locale) — stessa banda di qualità e stessi modi di guasto
  (restart doctype-doppio, fase-2 occasionalmente corta). La qualità del
  two-phase K23 è **portabile tra HW**; i tempi no (qui p2 ~2.4-2.9 t/s
  RAM-hot vs ~1.6-1.9 locale).

## File

Per gruppo `t4_WXXX/`: layout harness standard (r00-r02 con route.csv, tw.txt,
frozen.txt, sess.txt, p2prompt.txt, trest.txt, deliverable.html, p1/p2.diag),
`summary.csv`, `summary_median.csv`, `VERDICT.txt`, `manifest.json`.
`smoke_W090_anchor/` (cella ancora pre-crash), `tail_driver.sh`,
`tail_driver.log`.
