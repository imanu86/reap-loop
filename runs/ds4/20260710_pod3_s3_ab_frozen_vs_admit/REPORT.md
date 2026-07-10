# POD 3 — S3 live A/B: FROZEN vs FROZEN+DEMAND-ADMIT (patch 0026)

**Data:** 2026-07-10 · **Modo:** LIVE su pod (RTX 3090), non simulazione.
Primo A/B dal vivo dell'ammissione demand-driven (0026): chiude l'anello
`mask -> traiettoria -> domanda -> ammissione` che la sim E-ADMIT
(`runs/ds4/20260710_eadmit_demand_admission`) non puo' vedere perche' gira su
trace FISSI (nessun feedback della mask sulla generazione — "limite 1" nell'
header di 0026: «Quality is NOT deducible from the sane-trajectory sim»).

## Verdetto (secco)

- **Coffee: FROZEN batte ADMIT.** Frozen L1/L1/L1 (pulito, 3/3 chiude `</html>`);
  admit L0/L0/L1 (mediana L0), 2/3 degenerano in loop e non chiudono.
- **Cyberpunk: pareggio al ribasso.** Entrambi L0/L0/L0, nessuno chiude, ENTRAMBI
  degenerano in ripetizione (frozen loop corto ` no, no,`; admit loop lungo di
  ~301 char = lista di tag ripetuta). Admit non salva il prompt duro.
- **Conclusione:** il "config C" sim-raccomandato (h=1.2, k_d=0.02, p=2),
  POSITIVO offline (+13.7 pt copertura tardiva, churn basso, ~0 rimbalzi), dal
  vivo NON migliora la qualita' e sul coffee la peggiora attivamente. Il
  gate-qualita' che l'header di 0026 rimanda «to the live A/B S3» e' NEGATIVO.

## Setup

- **Pod:** RunPod community `hfyk1ze2yl9w2x` (machine `8uzldzxxg4jy`), RTX 3090
  24 GB, **$0.22/h**, image `runpod/pytorch:1.0.7-cu1290-torch280-ubuntu2404`
  (CUDA 12.9, driver 580.95.05). Provisionato ex-novo (il pod sibling
  `pdpgc8nck480gd` era occupato dal batch static-K23 di un altro ruolo; non
  toccato). Gate-check CUDA PASS al primo colpo (torch.cuda.is_available()=True +
  alloc reale su device OK: il fallimento tipico dei community 3090
  cudaGetDeviceCount->0 NON si e' verificato). Creds R2 iniettate come env var
  del pod (RCLONE_CONFIG_R2_*) — mai transitate su SSH/CLI/log.
- **Regime:** host 125 GB RAM, 32 vcpu ⇒ modello 81 GB interamente in page-cache
  ⇒ RAM-hot (classe T1/smoke). Tutti i t/s sono numeri-pod RAM-hot: DIAGNOSTICI,
  NON confrontabili col 3060 locale.
- **Modello:** `ds4-2bit.gguf` (IQ2XXS, 86 720 111 488 B) da R2, sha256 verificato.
- **Binario:** `ds4-admit` = `ds4_sm86_livetree-1db4f799-admit` da R2 (catena
  0020+0021+0026 su live-tree md5 1db4f799, buildato dal ruolo POD-2), md5
  d0f37f40..; `strings` conferma 6x DS4_PACE_ADMIT. Dipendenza-binario
  soddisfatta via R2, nessun rebuild.
- **Grading:** `scripts/functional_grade.py` (L0-L3), node presente ⇒ check JS
  reale. Metriche admit dal JSONL (DS4_PACE_LOG, evento "admit").

## Metodo (deviazione dichiarata dal brief)

Il brief chiedeva la mask via two-phase W50 weighted + freeze
(build_session_mask_canonical/freeze_boundary). Non applicabile all'arm ADMIT:
l'hook 0026 ammette solo con controller PACE vivo in PACE_HOLD e reap-mask attiva
(`admit_on && rmass && phase==PACE_HOLD && g_reap_mask_on`, dal sorgente). La
fase-2 two-phase e' mask STATICA via DS4_REAP_MASK_FILE senza controller ⇒ admit
non scatterebbe mai, e il re-prefill spezzerebbe l'anello di feedback oggetto del
test. Usata quindi la costruzione equivalente e piu' fedele al vivo: PACE con
WARMUP=50 (i primi 50 tok accumulano la massa router = top-23 weighted) poi HOLD
della K23 congelata. Un solo delta tra i bracci:

    BASE = DS4_PACE=1 WARMUP=50 KEEP=23 KEEP_MIN=23 KEEP_MAX=96
           BREATH_EVERY=999999 RELEARN=0 ROTATE=0 WRAP=1 WRAP_ROTATE_DELTA=1
    A (FROZEN): + DS4_PACE_ADMIT=0
    B (ADMIT) : + DS4_PACE_ADMIT=1 ADMIT_H=1.2 ADMIT_KDRIFT=0.02 ADMIT_PERSIST=2
                  ADMIT_COOLDOWN=16   (config C; MAX_PER_100=0)

Greedy temp 0, --nothink, trace OFF, --ssd-streaming(-cold) cache-experts 1024.
Ordine ABABAB, n=3/braccio. Coffee: ctx4096, -n1250 (=50+1200). Cyberpunk:
ctx8192, -n4050 (=50+4000). Comando: `run_ab.sh`. Smoke pre-run: admit conferma
di scattare live (PACE_HOLD, keep costante 23, one-in/one-out).

## Risultati per-seed

### COFFEE (compatto, budget 1200, ctx4096) — greedy NON-deterministico
| Cell | arm | L | close </html> | loop | chars | admit | gen t/s (pod) |
|---|---|---|---|---|---|---|---|
| A_r0 | FROZEN | 1 | si | no | 1226 | 0 | 9.0 |
| A_r1 | FROZEN | 1 | si | no | 949 | 0 | 11.0 |
| A_r2 | FROZEN | 1 | si | no | 1203 | 0 | 11.1 |
| B_r0 | ADMIT | 0 | no | SI | 1434 | 9 | 19.2 |
| B_r1 | ADMIT | 0 | no | SI | 5292 | 9 | 18.3 |
| B_r2 | ADMIT | 1 | si | no | 1438 | 19 | 12.5 |

FROZEN L1x3 (mediana L1), 3/3 chiude, 0 loop. ADMIT L0,L0,L1 (mediana **L0**),
1/3 chiude, **2/3 loop**. Firme: B_r1 loop `nav {  background: sans-serif>` a
nastro; B_r0 loop `, 0, 0, 0`. FROZEN produce pagine complete e chiuse (difetti
JS minori ⇒ L1, non L3).

### CYBERPUNK (duro, budget 4000, ctx8192) — greedy DETERMINISTICO (3/3 identici per braccio)
| Cell | arm | L | close | loop | chars | admit | gen t/s (pod) |
|---|---|---|---|---|---|---|---|
| A_r0..2 | FROZEN | 0 | no | SI | 8252 | 0 | 21.6 |
| B_r0..2 | ADMIT | 0 | no | SI | 10816 | 31 | 15.4 |

Entrambi L0, nessuno chiude (coerente con T1: il cyberpunk e' CSS-verboso, chiude
solo >3500 tok). ENTRAMBI degenerano: frozen in loop corto ` no, no,`; admit in
loop lungo (unita' 301 char, lista di tag `header, form, input, ...` ripetuta).
A e B divergono al **char 272** (subito dopo il warmup, alla prima ammissione
~tok68): l'admit cambia la traiettoria immediatamente ma verso un'altra
degenerazione, non verso la chiusura.

## Comportamento admit: LIVE vs SIM E-ADMIT (config C)

| Metrica | Sim E-ADMIT (trace fissi) | LIVE (questo A/B) |
|---|---|---|
| eventi/100 tok post-warmup | **~130** | **~1-7** (coffee 9/9/19; cyber 31 su ~4000 tok) |
| distribuzione temporale | sui confini strutturali (prose->code) | **burst subito dopo il warmup** (coffee tok~40-130, B_r1 9 ammissioni in tok59-78; cyber cluster tok68-212) |
| rimbalzi (admit->sfratto <=100tok) | ~0.3% | **0** osservati |
| effetto qualita' | +13.7 pt copertura tardiva (dedotto) | coffee: **peggiora** (L1->L0, loop); cyber: **nullo** (L0=L0) |

Tasso di ammissione vivo **~20-100x sotto** il predetto: la sim misura la domanda
"bloccata" su trace UNMASKED (tutti i 256 expert visibili, il CUSUM sale in
fretta), ma sotto mask ATTIVA il bias sopprime la probabilita' router dei pruned
⇒ il segnale che guida il CUSUM e' molto piu' debole ⇒ pochissime soglie
attraversate. E il poco che scatta arriva a raffica appena dopo il warmup, non
distribuito sui confini.

## Le tre domande

1. **Admit migliora o non peggiora la qualita'?** NO. Coffee: peggiora
   (mediana L1->L0, loop 2/3). Cyberpunk: nullo (entrambi L0, entrambi loop).
2. **Ammissioni nei punti giusti (confini) come da sim?** Solo in parte: burst
   subito dopo il warmup (coincide col primo confine head/style ma SOVRA-ammette
   li'), non distribuito sui confini come previsto (variante D).
3. **Instabilita' da feedback che la sim non poteva vedere?** SI, risultato
   centrale: (a) coffee — il burst di ammissioni fa collassare la traiettoria
   greedy in loop (2/3); (b) cyber — A e B divergono al char 272 alla prima
   ammissione, verso una degenerazione diversa. La sim su trace fissi non
   re-inietta mai il cambio-mask nella generazione ⇒ per costruzione cieca a cio'.

## Implicazioni (NON tocco CLAIMS/roadmap — solo elenco)

- Il gate-qualita' di 0026 (che l'header rimanda «to the live A/B S3») e'
  NEGATIVO: config C non e' promuovibile cosi'.
- Ipotesi da testare prima di ri-proporre 0026: (i) anti-burst — rate-cap
  ADMIT_MAX_PER_100>0 (qui 0) e/o H piu' alto per spegnere la raffica
  post-warmup; (ii) detector di confine reale (variante D) invece del CUSUM
  libero; (iii) ricalibrare la soglia sulla domanda MASKED, non su trace unmasked
  (spiega il fattore ~20-100x).
- Metodologico: la sim-su-trace-unmasked sovrastima drasticamente il tasso di
  ammissione e non puo' pronunciarsi sulla qualita' — utile solo per churn/
  copertura, non come gate qualita'.

## Costo e stato pod

- Pod on-demand community $0.22/h. Provisioning ~16:58Z; coffee A/B ~17:35-17:42Z;
  cyberpunk ~17:43-18:07Z. Uptime ~1h20 ⇒ spesa ~**$0.30-0.35**, ben sotto il cap $4.
- Pod **terminato** a fine analisi (community: terminate).

## Riproducibilita'

- `run_ab.sh <prompt> <tag> <ctx> <ntok> <nruns>` (in cartella).
- `cells/coffee/` e `cells/cyber/`: out.txt, pace.jsonl, diag.txt per cella.
- `analyze.py <cells_dir>` rigenera tabella + analysis.json (loop-detector con
  finestra 16-350 char per catturare degenerazioni a periodo lungo).
- Prompt: `coffee_prompt.txt` (819 B, byte-esatto al replay), `cyberpunk_prompt.txt` (199 B).
