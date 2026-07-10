# 2026-07-10 Pod2 (redeploy) — SAMPLING-UNDER-MASK, probe nuova

DOMANDA (mai testata: T1 provò il sampling solo sul modello FULL): il sampling
minimo (temp 0.7, top_p 0.95, seed variato) applicato **sotto la mask statica**
riduce i loop / migliora il livello L rispetto al greedy?

## Setup

- Pod: RunPod `i7dk94f0y05iji` (SECURE RTX 3090 24GB, $0.46/h, RAM host 1007 GB ⇒
  regime RAM-hot: **qualità confrontabile, tempi NO** — non confrontare i t/s con
  il 3060 locale).
- Binario: `ds4_sm86_livetree-771a39a8` da cache R2 (sha256 `772c502f…` verificato;
  lineage post-0018 = stesso del gruppo W50 locale). Modello `ds4-2bit.gguf`
  sha256-verificato da R2.
- Harness: `scripts/run_w_sweep_freeze_safe.py` con hotfix fence non-leading
  (b91188d) **+ nuove opzioni** `--top-p` e `--sample-p2-only` (questo run):
  la fase 1 resta GREEDY (stessa costruzione trace/mask/freeze dei bracci greedy),
  il sampling si applica SOLO alla fase 2 mascherata — la probe isola esattamente
  "sampling sotto mask".
- Config: W=50, K23 static weighted (`DS4_REAP_MASK_FILE`), coffee prompt 819 B,
  two-phase freeze-safe, `--total 1200` (fase2 = 1150 tok ≈ ~1100), ctx 4096/4096,
  cache 256, temp 0.7, top_p 0.95, seed 42/43/44, n=3, trace routing solo fase-1,
  manifest completo. Env IO: `DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1
  DS4_REAP_PREFETCH_THREADS=16 DS4_REAP_PREFETCH_LOCK=1` (come driver locale).

## Risultati per-seed (sampled, fase2 sotto mask)

| run | seed | freeze | L | restart | `</html>` | repeat | button+form wired | chars | alert |
|---|---|---|---|---|---|---|---|---:|---:|
| r00 | 42 | `;` @50 | **L2** | 1 | 1 | 0 | sì+sì | 1866 | 3 |
| r01 | 43 | `;` @50 | **L2** | 1 | 1 | 0 | sì+sì | 2398 | 2 |
| r02 | 44 | `;` @50 | **L2** | 1 | 1 | 0 | sì+sì | 2503 | 2 |

Mediana **L2**; loop-rate (repeat) **0/3**; restart (doppio doctype) **3/3**.
I tre output DIFFERISCONO (chars/alert diversi) ⇒ i seed sono onorati; le tre
fasi-1 greedy hanno prodotto lo stesso freeze `;`@50 within-target.

## Confronto con i greedy n=3 (stessa config W50 K23 static coffee)

| braccio | HW | L per-seed | mediana L | repeat | restart | note |
|---|---|---|---|---|---|---|
| **sampled (questo)** | pod 3090 | 2,2,2 | **2** | **0/3** | 3/3 | fence-hotfix attivo |
| greedy locale `t4_W050` | 3060 WSL | 2,0,2 | **2** | 0/3 | 2/3 | r01 L0 = fase-2 vuota (161 char, no tps) |
| greedy pod1 S5 `s5_coffee_k23` | pod 3090 | 2,1,2 | **2** | 1/3 | 2/3 | r01 L1 = artefatto fence pre-hotfix (freeze=none) |

## VERDETTO (secco)

- **Il sampling minimo sotto mask NON cambia il fenotipo**: mediana L2 identica ai
  greedy; loop-rate 0/3 contro 0/3 dei greedy validi (l'unico repeat greedy, S5
  r01, era l'artefatto fence poi corretto da b91188d — non un loop da mask).
- **Non riduce i loop** perché a W50/K23/coffee/freeze-sicuro i loop sono già ~0
  nel greedy: non c'è margine da recuperare.
- **Non rimuove l'attrattore restart** (doppio doctype nel deliverable): 3/3 nei
  sampled vs 2/3 nei greedy — il restart J44 residuo è indipendente dal sampling.
- In positivo: **stabilità** — il sampling non degrada (nessun L<2, nessun loop
  introdotto) e dà diversità di output utile (n=3 realmente indipendenti, a
  differenza del greedy che su questo pod/binario può essere bit-identico).
- Implicazione: per migliorare L o togliere il restart, la leva NON è il sampler;
  restano mask/freeze/prompt (coerente con T1: anche sul FULL il sampling non
  cambiava il fenotipo).

## File

Layout harness standard: `W050/r00..r02/` (route.csv, tw.txt, frozen.txt,
sess.txt, p2prompt.txt, trest.txt, deliverable.html, p1/p2.diag),
`summary.csv`, `summary_median.csv`, `manifest.json` (temp/top_p/sample_p2_only
registrati), `VERDICT.txt` (campo monotonia non applicabile: W singolo),
`harness.log`.
