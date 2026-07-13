# DS4 su Windows nativo: vale la pena? — 2026-07-13

Studio di sintesi. Consolida due filoni misurati il 2026-07-13 sulla stessa
macchina (Ryzen 7 5800X, 64 GiB DDR4, RTX 3060 12 GiB, driver WDDM 596.21):

1. **BANDA** — banda H2D reale su Windows nativo vs throttle WSL2;
2. **PORT + BYPASS** — fattibilita' del port `ds4-win` e aggirabilita' del cap
   pinned WDDM.

Convenzione epistemica: `MISURA` = artefatto/log locale su questa macchina;
`FONTE` = risultato esterno citato; `IPOTESI` = non e' un risultato.

---

## Numeri di partenza (fissati)

| Grandezza | Valore | Tipo |
|---|---|---|
| H2D pinned WSL2 | ~2.97-3.03 GiB/s | MISURA (arena probe WSL, 4 run) |
| H2D pinned Windows nativo | 24.1-24.4 GiB/s | MISURA (2 code path indip.) |
| H2D pageable Windows nativo | 10.4-11.6 GiB/s | MISURA |
| Cap pinned host (64 GiB RAM) | 31 GiB | MISURA (WSL **e** Windows nativo) |
| Modello ds4-2bit pieno | 80.76 GiB | MISURA (86.72 GB su disco) |
| Mask60 (working set) | ~40 GiB | dato di progetto |
| t/s stock (stream pieno) | 2.72 | MISURA |
| t/s partial-cache 24 GiB | 6.9-7.7 | MISURA (questa macchina) |
| t/s chripell, Linux nativo | 10 | FONTE (banda piena) |
| t/s GB10 / DGX Spark | 13.81 | FONTE (hardware diverso) |

---

## Matrice decisionale

| Opzione | Banda H2D | Cap pinned | t/s atteso | Effort | Cosa serve |
|---|---|---|---|---|---|
| **WSL attuale** | ~3 GiB/s — throttle GPU-PV (MISURA) | **31 GiB** (WDDM, MISURA) | 2.72 stock → **6.9-7.7** partial-cache 24 GiB (MISURA) | **zero** (gia' in uso) | niente |
| **Windows nativo (ds4-win)** | **24.1-24.4 GiB/s** pinned / ~11 pageable — **8x** WSL (MISURA) | **31 GiB — INVARIATO** (stesso WDDM; 32 GiB fallisce OOM anche con >52 GiB liberi, MISURA) | **non misurato con modello**; tetto plausibile ~10 (chripell) se banda-bound, ma il decode puo' restare overhead-bound | **XL** — port mai eseguito con un modello reale | chiudere+mergiare PR #2 (draft, CONFLICTING), build MSVC+CUDA 12.6.3, ri-tarare arena, A/B n≥3 |
| **Linux dual-boot** | ~24+ GiB/s, banda PCIe 4.0 piena (FONTE) | **nessun cap WDDM** → RAM piena pinnabile (40 GiB mask60 interi, e oltre) | **10** (FONTE chripell, Linux nativo banda piena) | **medio** — nessun port (DS4 e' gia' nativo Linux) | partizione + bootloader, rebuild nativo, reboot per cambiare OS |

Note sulla riga Windows nativo:
- Il port `ds4-win` (fork `hawkli-1994/ds4-win`, branch `codex/windows-cuda`,
  PR #2) aggiunge un layer `src/platform/` (~660 righe) + modifiche a `ds4.c`,
  `ds4_cuda.cu`, `ds4_server.c`. La CI `Windows CUDA` (run 25801727530) e'
  **verde ma solo di compilazione**: produce `ds4_server.exe`, **nessun log di
  un run con GGUF reale**. PR **draft**, `mergeable=CONFLICTING`,
  `mergeStateStatus=DIRTY`. L'autore dichiara di **non** aver eseguito il build
  Windows localmente. → "compila" non e' "serve un modello".

---

## Verdetto

### (a) La banda H2D nativa Windows aggira il throttle WSL? — **SI, confermato.**

Native Windows H2D **pinned 24.1-24.4 GiB/s** contro WSL **~3 GiB/s** = **~8x**.
Anche il **pageable** nativo (~10.4-11.6 GiB/s, senza alcun pinning) batte il
*pinned* WSL di ~3.5x. Il numero nativo (~24.4 GiB/s ≈ 85% del tetto teorico
PCIe 4.0 x16) e' "PCIe 4.0 che si comporta normalmente"; il tetto ~3 GiB/s non
e' spiegato da generazione/larghezza PCIe, ne' dal pinning, ne' da alcuna
proprieta' di questa GPU. **Riproduce solo sotto lo strato di
paravirtualizzazione GPU-PV di WSL2.** Confermato da **due code path
indipendenti** (CUDA Driver API raw + PyTorch/cudart), concordi entro il
rumore. Uscire da WSL — Windows nativo **o** Linux — elimina il throttle.

### (b) Il cap 31 GiB e' aggirabile? — **NO su Windows nativo; SI solo su Linux nativo.**

Il muro **non** e' RAM di WSL, `ulimit`, swap o page cache. E' la formula WDDM:

```
TotalSystemMemoryAvailableForGraphics = max(TotalSystemMemory / 2, 64 MiB)
```

Su 64 GiB fisici → ~31.9 GiB → l'arena da 31 GiB passa, il 32-esimo blocco da
1 GiB fallisce. **MISURA: identico su WSL e su Windows nativo** — 31 GiB passa,
32 GiB fallisce con `CUDA_ERROR_OUT_OF_MEMORY` anche con >52 GiB di RAM libera.
Leve escluse punto per punto: **TCC** chiuso su GeForce (confermato NVIDIA:
GeForce non supporta TCC; `nvidia-smi -dm` = "Not Supported" anche su RTX
consumer); **MCDM** esiste ma non e' un bypass utile qui; `.wslconfig`, `mlock`,
pagefile, HAGS, Resizable BAR, kernel custom, registro non documentato →
**tutti negativi**.

Conseguenza: mask60 = **40 GiB > 31 GiB pinnabili**. Il cap **resta** su
Windows nativo e **la mask/scheduler lo gestisce**: la porzione eccedente
(~9-16 GiB) stream da RAM pageable — che su Windows nativo e' comunque ~3.5x
piu' veloce del pinned WSL. Il cap sparisce solo con:
1. **Linux nativo** (niente WDDM; `pin_user_pages`/HMM → RAM piena pinnabile), o
2. **128 GiB di RAM fisica** (porta il tetto WDDM a ~64 GiB).

### (c) Raccomandazione

**Windows nativo e' il ROI peggiore dei tre.** Il port e' **XL e non provato**
(mai servito un modello reale: la CI solo compila, PR #2 draft/CONFLICTING).
Compra la **banda** ma **non** compra la **capacita'**: conservi esattamente lo
stesso cap 31 GiB che ha imposto il design mask/streaming. Paghi molto codice
per meta' del beneficio, e la meta' che non ottieni (pinnare l'intera mask) e'
proprio quella che spinge verso Linux.

**Linux dual-boot domina tecnicamente Windows nativo.** DS4 e' **gia' nativo
Linux → nessun port**. Ottieni banda piena **E** niente cap (pinni tutti i 40
GiB mask60, con margine), con il riferimento reale **chripell = 10 t/s**. Il
costo e' solo setup dual-boot (repartizione + bootloader) e reboot per cambiare
OS. Se l'obiettivo e' massimizzare t/s con la mask **interamente residente**, e'
la strada tecnicamente corretta.

**Restare su WSL e' corretto come baseline a effort zero** (2.72 → 6.9-7.7 t/s)
**finche' il prodotto in iterazione e' la mask dinamica SPEX/streaming**, non il
transport. La banda WSL limita solo la porzione streamed; con arena 24 GiB +
cache l'hai gia' in gran parte mitigata (6.9-7.7 t/s).

**Sintesi operativa:**
- Collo di bottiglia = banda H2D sulla porzione streamed **e** vuoi la mask 40
  GiB interamente pinnata → **Linux dual-boot**.
- Vuoi restare in un solo OS e ti basta la banda sulla parte streamed
  accettando il cap 31 → **Windows nativo**, ma **solo dopo** aver chiuso il
  port e un A/B n≥3 (oggi XL / non provato).
- Altrimenti → **resta su WSL** e continua a iterare la mask.

---

## Caveat onesti

- **Correzione al framing "medio effort" di Windows nativo**: la task porting lo
  valuta **XL** e **non provato con modello**. Non e' medio effort finche' il
  port non serve un GGUF reale con throughput misurato.
- **Nessun t/s DS4 misurato fuori da WSL.** chripell 10 t/s e' Linux nativo con
  config/mask potenzialmente diverse: e' un **riferimento**, non una garanzia
  per questa macchina.
- **Tensione banda-bound vs overhead-bound.** Segnale concorrente in memoria:
  nel regime keep-warm il decode risulta *overhead-bound* (~121 ms
  orchestrazione/token), non banda-bound. Se cosi', una H2D 8x piu' veloce aiuta
  **prefill** e la porzione **streamed** ma il tetto t/s da sola banda potrebbe
  restare sotto i 10 di chripell. La banda aiuta molto quando devi streammare
  esperti per-token (mask 40 GiB > 31 pinnabili); aiuta poco quando il working
  set e' gia' caldo in VRAM+pinned.
- **N=1 macchina, N=1 GPU/driver.** Nessuna ripetizione tra reboot.
- **WSL non ri-misurato in questa sessione** (riuso dei numeri
  `20260712_v2_zerocopy`, pinned/single-shot vs protocollo N=12): non cambia
  l'ordine di grandezza.
- GB10/DGX Spark 13.81 t/s e' **hardware diverso**, citato solo come tetto di
  riferimento, non come opzione su questa macchina.

## Artefatti

- Banda: `runs/ds4/20260713_win_native_bw/RESULTS.md` (+ `tools/`, `bin/`,
  `results/h2d_bandwidth_native.jsonl`, `results/torch_h2d_bandwidth_native.stdout.txt`).
- Cap pinned WDDM: `docs/WSL_WDDM_PINNED_LIMIT_20260713.md`.
- Porting/handoff: `docs/HANDOFF_CLAUDE_20260713_0050_PORTING.md`.
