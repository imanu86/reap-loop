# HANDOFF CLAUDE - DS4 0050 ZERO-COPY + WINDOWS/GB10 - 2026-07-13

Aggiornato: 2026-07-13T11:02:48+02:00.

Questo documento consolida due filoni:

1. patch 0050: finestra host pinned su RTX 3060/WSL2 e DMA diretto verso la
   cache esperti in VRAM;
2. task separata `Porting DS4 / DGX Spark - NO GPU heavy`: Windows CUDA
   nativo, limite WDDM, Linux/HMM, DGX Spark/GB10 e alternative runtime.

Regola epistemica: `MISURA` significa artefatto/log locale; `FONTE` significa
risultato esterno citato; `IPOTESI` non va presentata come risultato.

## 0. Sintesi per riprendere

- Hardware: Ryzen 7 5800X, 64 GB DDR4, RTX 3060 12 GB.
- Obiettivo: evitare il pedaggio `pread -> staging -> sync` dei fetch RAM e
  arrivare a 5-8 t/s, mantenendo una mask dinamica per turno. La mask statica
  bake60 serve solo come banco di prova del trasporto, non e' il prodotto
  finale.
- La 0050 registra 24 GiB di range della mmap e usa DMA H2D diretto verso gli
  slot VRAM. Nel gate attuale: `4446/4446`, 23.99 GiB, path DMA attivo, zero
  errori DMA.
- Windows/WSL impongono su questa macchina un tetto pratico di 31 GiB host
  pinned. Windows nativo NON lo aggira: 31 GiB passa, 32 GiB fallisce con
  `CUDA_ERROR_OUT_OF_MEMORY` anche con oltre 52 GiB di RAM disponibile.
- Quindi la soluzione locale Windows non e' 50 GiB pinned. Il design utile e':
  circa 50 GiB in RAM normale/page cache, arena pinned persistente fino a
  24 GiB, 12 GiB VRAM, con scheduler REAP/SPEX e overlap.
- L'exactness della 0050 non e' ancora chiusa. I run moderni coincidono per i
  primi 59 token ma il token 60 ha prodotto tre esiti (`CSS`, em dash, en
  dash). Esiste non-determinismo greedy storico nel runtime streaming. Non
  attribuire il fenomeno a DMA da un singolo OFF/ON.
- Nessun A/B prestazionale controllato n>=3 e' stato completato. I numeri n=1
  mostrano un segnale favorevole per ON24, ma NON sono un verdetto.
- Prossimo gate gia' predisposto: n>=3 intra-arm + inter-arm, ordine alternato,
  prefetch esplicito. Il runner e' stato validato staticamente ma questa nuova
  modalita' non e' ancora stata eseguita.

## 1. Repository, branch e stato operativo

### `reap-loop` - lavoro vivo

- Path: `C:\Users\imanu\source\repos\reap-loop`
- Branch: `spex-predictive-mask-study-2026-07-12`
- HEAD/push noto: `c4765ef` - `misura il tetto WDDM e consolida il piano arena dinamica`
- Commit precedenti rilevanti:
  - `adf3de4` - salvataggio artefatti V2 zero-copy;
  - `1cb0c40` - handoff/diagnosi iniziale 0050;
  - `b0511ac` - bake60 produce documento completo.
- Worktree sporco: molti artefatti di test sono untracked. Non cancellare,
  resettare o aggiungere tutto in blocco.

Modifiche correnti non committate attribuite a questo filone:

- `runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh`
  - opzione `--prefetch on|off`;
  - opzione `--gate-repeats N`;
  - gate ripetuto che distingue `PASS`, `FAIL` stabile e
    `INDETERMINATE_NONREPEATABLE`;
- `runs/ds4/20260712_v2_zerocopy/scripts/measure_stream.py`;
- `runs/ds4/20260712_v2_zerocopy/AB_PROTOCOL.md`;
- `docs/DESIGN_0051_DYNAMIC_ARENA.md`;
- `docs/DESIGN_0052_DYNAMIC_QUANTIZED_COLD_TIER.md`;
- `docs/DS4_LEVE_CATALOG.md`;
- `runs/ds4/20260712_v2_zerocopy/controlled_ab_0050/`.

Ultima verifica runner: `bash -n` + `--help` = `HARNESS_STATIC_OK`. La modalita'
n>=3 e' stata aggiunta dopo gate06 e non e' ancora stata eseguita.

### Worktree DS4 WSL

- Path: `/root/ds4-v2-work`
- Base Git: `da0b3f63d7cc87c1f11c3c876fb57de3e0caca50`
- Binario SHA-256:
  `dc9171b1349453a982f1ed85e4904e08c820e31076a3304409e981d451ee5149`
- Binario MD5: `7ca6f27d6ce251f944be5c079dc471de`
- Modello: `/root/models/ds4-2bit.gguf`
- Modello SHA-256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- Modello size: 86,720,111,488 byte.
- Mask benchmark:
  `runs/ds4/20260712_virtual_bake/masks/mask60_self.txt`.

La patch 0050 e' ancora WIP nel worktree DS4; non esiste ancora un export
canonico `0050-*.patch` completo sotto `patches/ds4`.

### WSL e risorse al momento dell'handoff

`C:\Users\imanu\.wslconfig`:

```ini
[wsl2]
memory=38GB
processors=16
swap=16GB
networkingMode=mirrored
[experimental]
autoMemoryReclaim=disabled
hostAddressLoopback=true
```

Ultimo check:

- GPU lock `/tmp/ds4-gpu.lock`: libero;
- nessun `ds4-server` o harness 0050 attivo;
- RTX 3060: 661/12288 MiB, 0% util;
- WSL: 38,095 MiB totali, 36,901 MiB disponibili.

Il cap 38 GB e' deliberato: con 40 GB WSL il prefill aveva portato Windows a
6.84 GiB disponibili; a 38 GB gate05/gate06 sono rimasti sopra il floor 8 GiB.

### `moe-aggressive-commit` - report porting

- Path: `C:\Users\imanu\source\repos\moe-aggressive-commit`
- Branch: `cascade-memory/harness`
- HEAD: `c1c55d4`, allineato all'origin al momento della ricognizione.
- Report porting, NON committato:
  `docs/DS4_WINDOWS_RUNTIME_ALTERNATIVES_20260713.md`.
- Untracked preesistente da non toccare: `1{print`.

La task porting non ha creato cloni DS4, non ha modificato runtime DS4 e non ha
fatto commit.

## 2. Cosa implementa la patch 0050

La 0050 non esegue GEMM direttamente da host RAM. Registra privatamente range
della mmap del GGUF e, quando un expert richiesto e' coperto, esegue:

```text
pinned mmap range -> cudaMemcpyAsync H2D -> slot cache expert VRAM -> GEMM VRAM
```

Per i range non coperti resta il fallback:

```text
pread -> staging -> H2D -> slot cache expert VRAM
```

Le env principali sono:

```text
DS4_CUDA_STREAM_FROM_RAM_MASKED=<mask>
DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=24
DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1
DS4_REAP_MASK_FILE=<mask>
DS4_CUDA_NO_Q8_F16_CACHE=1
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
```

Il budget e' distribuito per layer. Il gate moderno ON24 misura:

- 4446/4446 range registrati;
- 23.99 GiB pinned;
- `DMA path ACTIVE`;
- decine di migliaia di copie DMA;
- `dma_failed=0`, `miss_empty=0`, `miss_base=0`.

Limite funzionale corrente: la finestra registrata deriva dalla mask letta allo
startup e non e' ancora una arena dinamica che cambia membership durante la
chat o tra turni. Questo e' il ruolo previsto della 0051.

## 3. Archeologia exactness storica

Fonte primaria recuperata:

`C:\Users\imanu\.claude\projects\C--Users-imanu-source-repos-moe-aggressive-commit\c3703740-a63b-4805-bc2e-db79b6e0b46b\subagents\agent-a3b8d839b195fb68a.jsonl`

Trascrizione equivalente:

`C:\Users\imanu\AppData\Local\Temp\claude\C--Users-imanu-source-repos-moe-aggressive-commit\c3703740-a63b-4805-bc2e-db79b6e0b46b\tasks\a3b8d839b195fb68a.output`

### ON2 - PASS esplicito a 24 GiB

- 23.85 GiB pinned, 4271/4296 range;
- 60 token completati;
- OFF/ON MD5 identico:
  `54d1055f2fe0c0ac5446bddcb438858a`;
- `BIT_EXACT_MATCH` alle righe Claude 426-427.

Caveat: questa era una variante precedente. La registrazione era visibile al
path mapped/UVA e la decode era molto lenta (~0.17 t/s). Prova il contenuto
uguale di quel run, non l'attuale private-DMA hot path.

### ON3 - 24 GiB completo, nessun confronto conservato

- 4446/4446, 23.99 GiB;
- 60 token completati in 883.818 s;
- nessun `diff`/hash eseguito prima di ON4.

### ON4 - 24 GiB incompleto

- 4446/4446, 23.99 GiB;
- arriva al prefill, ma non conserva `gen=60 finish`;
- nessun verdetto exactness.

### ON5 - PASS esplicito a 24 GiB, ma DMA non attribuito

- variante private-pin, 4446/4446, 23.99 GiB;
- OFF/ON MD5 identico `54d1055f...`;
- `BIT_EXACT_MATCH` alle righe Claude 657-659;
- il log non mostra `DMA path ACTIVE`: i pin erano registrati, ma non e'
  dimostrato che il fetch hot usasse la nuova copia DMA.

La directory `/root/ds4-v2-work/bitexact/on` veniva riusata. Gli artifact ON2-
ON5 sono stati sovrascritti; i verdetti sopra sopravvivono nei log Claude.

### Coppia moderna 5 GiB

Gli artifact attuali `/root/ds4-v2-work/bitexact/{off,on}` usano:

- stesso binario `dc9171...` del gate05/06;
- budget ON 5 GiB, 1035/1035 range, 4.75 GiB;
- prefetch REAP disattivato;
- DMA attivo e migliaia di copie riuscite;
- OFF/ON SHA-256 identico:
  `81fb5d5f83d91fae4da37bd2df98ba0b37699dfe9b0f9e48fc45c9124a9eff30`.

Questa prova chiude exactness per quel run a 5 GiB, non per 24 GiB.

## 4. Gate moderni 24 GiB

### gate05 - prefetch ON

Path:

`runs/ds4/20260712_v2_zerocopy/controlled_ab_0050/20260713_0050_gate05`

Config condivisa:

- cache expert 400;
- ctx 2048, prefill chunk 512;
- temp 0, 60 completion token;
- `DS4_REAP_PREFETCH=1`, 16 thread;
- 24 GiB budget;
- stesso binario, modello, mask e richiesta nei due bracci.

Risultati:

| Arm | Hash | Ultimo token | Totale | Prefill | Decode |
|---|---|---|---:|---:|---:|
| OFF24 | `c11791215388...` | `CSS` | 147.696 s | 71.9 s | 0.79 t/s |
| ON24 | `81fb5d5f83d...` | em dash | 107.597 s | 48.473 s | 1.01 t/s |

ON attribution:

- 4446/4446, 23.99 GiB in 10.604 s;
- `dma_ok=24810`, 55,822.50 MiB;
- `dma_failed=0`.

I primi 211 byte, cioe' i primi 59 token, sono identici. Differisce solo il
token 60. Gate formalmente `FAIL`. I tempi sono n=1 e non autorizzano un
verdetto prestazionale.

### gate06_pfoff - prefetch OFF

Path:

`runs/ds4/20260712_v2_zerocopy/controlled_ab_0050/20260713_0050_gate06_pfoff`

Unica modifica intenzionale rispetto a gate05: prefetch REAP assente in
entrambi i bracci.

Risultati:

| Arm | Hash | Ultimo token | Totale | Decode |
|---|---|---|---:|---:|
| OFF24 | `81fb5d5f83d...` | em dash | 120.665 s | 1.23 t/s |
| ON24 | `ab601639a791...` | en dash | 90.868 s | 1.40 t/s |

ON attribution:

- 4446/4446, 23.99 GiB in 10.351 s;
- `dma_ok=24837`, 55,883.25 MiB;
- `dma_failed=0`.

Floor memoria:

- OFF Windows min 10.08 GiB, WSL min 36,009 MiB;
- ON Windows min 10.19 GiB, WSL min 35,872 MiB.

Anche qui i primi 59 token sono identici e differisce solo il token 60. Gate
formalmente `FAIL`; i tempi restano n=1.

### Interpretazione ammessa

MISURA:

- DMA ON24 e' attivo e senza errori;
- il token 60 non e' stabile fra le lifecycle osservate;
- sono comparsi almeno tre token finali diversi sotto configurazioni vicine;
- il repo documenta gia' non-determinismo greedy nel path streaming:
  `runs/ds4/20260711_smoke_0024/SUMMARY.md` e
  `runs/ds4/20260711_local_clean_lowK/BITEXACT.md`.

NON dimostrato:

- che DMA causi la divergenza;
- che prefetch causi da solo la divergenza;
- che ON24 sia piu' veloce in modo ripetibile.

## 5. Runner n>=3 predisposto

Il runner corrente supporta:

```bash
bash runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh \
  --execute \
  --campaign-id 20260713_0050_gate07_n3_pfoff \
  --gate-only \
  --gate-repeats 3 \
  --prefetch off
```

Comportamento previsto:

- 3 OFF e 3 ON;
- ordine alternato per ridurre order/cache bias;
- hash intra-arm e confronto inter-arm separati;
- `INDETERMINATE_NONREPEATABLE` se almeno un arm varia internamente;
- `FAIL` solo se i due arm sono internamente stabili ma diversi;
- `PASS` solo se tutti i run sono stabili e uguali.

Prima di eseguire:

```bash
bash runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh \
  --preflight --prefetch off --gate-repeats 3
```

Se n=3 resta non ripetibile solo al token 60, non inseguire il byte finale con
altri n=1. Aggiungere un gate su una lunghezza stabile (per esempio 50/59 token)
e, se possibile, logit/top-2 margin sull'ultimo token. Il gate di trasporto deve
separare numerical repeatability da equivalenza semantica.

Non avviare l'A/B 450-token finche' exactness/repeatability non e' classificata.

## 6. Risultati della task Porting DS4 / DGX Spark

Report completo, 800+ righe e link primari:

`C:\Users\imanu\source\repos\moe-aggressive-commit\docs\DS4_WINDOWS_RUNTIME_ALTERNATIVES_20260713.md`

### Port Windows CUDA recuperato

- Fork: <https://github.com/hawkli-1994/ds4-win>
- Branch: <https://github.com/hawkli-1994/ds4-win/tree/codex/windows-cuda>
- PR: <https://github.com/hawkli-1994/ds4-win/pull/2>
- Commit:
  <https://github.com/hawkli-1994/ds4-win/commit/2fba7fe4d77b4a917ccca353d2d898c4abe9a817>
- CI Windows CUDA:
  <https://github.com/hawkli-1994/ds4-win/actions/runs/25801727530>

Il port aggiunge CMake/MSVC, Winsock/BCrypt, astrazioni Win32 e workflow CUDA
12.6.3 che produce `ds4_server.exe`. La CI prova la compilazione, non un model
load o throughput reale. Al momento della ricognizione il PR era draft,
conflittuale e dirty.

### Probe CUDA Windows nativo

Sorgente:

`runs/ds4/20260712_v2_zerocopy/tools/cuda_pinned_arena_probe_win.cpp`

Runner:

`runs/ds4/20260712_v2_zerocopy/tools/run_pinned_arena_probe_win.ps1`

Binario MinGW statico:

`runs/ds4/20260712_v2_zerocopy/arena_probe_win/probe_20260713_061915_win_mingw.exe`

Log nativi:

`C:\Users\imanu\Documents\Codex\2026-07-13\studio-separato-porting-ds4-dwarf-spex\work\native_win_probe_20260713_090803`

Build:

```bash
x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -Wextra \
  -static -static-libgcc -static-libstdc++ \
  -o probe_20260713_061915_win_mingw.exe cuda_pinned_arena_probe_win.cpp
```

MSVC locale non era completo (`excpt.h` mancante); MinGW ha compilato. Il probe
carica `nvcuda.dll` e usa Driver API ABI v2.

Matrice MISURATA:

| Windows nativo | Esito | Retained | Nota |
|---|---:|---:|---|
| single 24 GiB | PASS | 24 GiB | materializzazione + DMA/checksum OK |
| single 28 GiB | PASS | 28 GiB | OK |
| single 30 GiB | PASS | 30 GiB | OK |
| single 31 GiB | PASS | 31 GiB | OK |
| single 32 GiB | FAIL | 0 | `cuMemHostAlloc` OOM immediato |
| blocks 32x1 GiB | FAIL controllato | 31 GiB | blocco 32 OOM, DMA/checksum sui 31 OK |

Prima del test c'erano ~52.65/63.89 GiB RAM disponibili e GPU 562/12288 MiB.
Quindi non e' esaurimento RAM generale. Dopo cleanup RAM/GPU sono tornate
libere.

Conclusione MISURATA: Windows CUDA nativo conserva lo stesso tetto WDDM di
WSL. Non investire nel port Windows come via per ottenere 50 GiB pinned.

### Misure WSL coerenti

Path:

`runs/ds4/20260712_v2_zerocopy/arena_probe/`

- 24/28/30/31 GiB: PASS;
- target blocks 50 GiB: fallisce dopo 31 GiB retained;
- H2D arena grande: circa 2.97-3.03 GiB/s;
- al fallimento rimanevano RAM libere sia in WSL sia in Windows.

Documento:

`docs/WSL_WDDM_PINNED_LIMIT_20260713.md`.

### DGX Spark / GB10

Finding principale:

- PR HMM: <https://github.com/antirez/ds4/pull/158>
- Commit:
  <https://github.com/antirez/ds4/pull/158/commits/75c458da489ec2c543121da118601966e68d2727>

FONTE esterna:

- su GB10/NVLink-C2C gestisce `cudaHostRegister` non supportato;
- usa mmap/page cache + `cudaMemPrefetchAsync` invece di copie private;
- modello 80.76 GiB;
- ~26 GB process/device usage con page cache reclaimable;
- 94% GPU util;
- 13.81 t/s riportati su DGX Spark.

Stato al censimento: PR aperta e conflittuale. Non testata localmente.

Altri riferimenti utili:

- distributed span fix: <https://github.com/antirez/ds4/pull/317>
- GB10 vs Thor: <https://github.com/antirez/ds4/issues/183>
- SSD streaming GB10 ~2.1 decode t/s:
  <https://github.com/antirez/ds4/pull/349>
- partial weight cache 6.93-7.77 t/s su GPU 24 GB:
  <https://github.com/antirez/ds4/pull/153>
- selected-load streaming: <https://github.com/antirez/ds4/pull/497>
- fast CUDA/VMM: <https://github.com/antirez/ds4/pull/187>

DwarfStar/Spark:

- <https://github.com/yuhai-china/ds4-spark>
- aggiunge Spark/speculation, ma non e' emersa una soluzione distinta al tetto
  WDDM/pinning.

Fork GB10 recente:

- <https://github.com/redhunt07/ds4-DSpark-GB10>
- dichiara ~17.6 generated t/s su GB10; fonte esterna, non misura locale.

### Altre strade censite

- Multi-GPU Linux: <https://github.com/0xfunboy/ds4-multicuda>, 2x3090 warm
  decode 2.8 t/s dichiarati, sotto target e non applicabile alla singola 3060.
- Windows ROCm Strix Halo: <https://github.com/antirez/ds4/pull/512>, ~15-16
  t/s dichiarati con unified RAM, ma richiede hardware/backend AMD diverso.
- DirectStorage non e' GPUDirect Storage. Nessun fork DS4 Windows CUDA con
  DirectStorage/GDS/cuFile utile e' stato trovato.
- GDS e' una pista Linux con hardware/filesystem supportati, non un bypass
  Windows/WDDM per la RTX 3060.

### Decisione architetturale

| Strada | Stato |
|---|---|
| WSL2 RTX 3060 | 31 GiB pinned max MISURATO; usare gerarchia, non 50 GiB pinned |
| Windows CUDA nativo | stesso tetto MISURATO; chiusa come soluzione capacity |
| Windows + 50 GiB RAM normale + 24 GiB pinned + 12 GiB VRAM | design locale piu' sensato; da implementare/misurare |
| Linux bare metal | niente WDDM; capacità >31 GiB/HMM da misurare |
| Pod Linux | modo rapido per validare HMM/arena senza WDDM |
| DGX Spark/GB10 | miglior fit architetturale; 13.81 t/s da FONTE, costo alto |
| Strix Halo ROCm | valida alternativa hardware, fuori dal path NVIDIA locale |

## 7. Direzione tecnica locale consigliata

Non ripetere altri probe per scoprire se 24 GiB passano: e' gia' misurato sia
WSL sia Windows nativo.

La gerarchia target e':

```text
SSD / GGUF
  -> mask del turno resident/hot in RAM normale (~50 GiB)
  -> arena pinned persistente e rimpiazzabile (fino a 24 GiB)
  -> cache esperti VRAM (entro 12 GiB)
  -> GEMM CUDA
```

La 0051 deve evitare register/unregister e allocazioni per miss. Preferire
un'arena pinned preallocata con slot stabili, copie RAM-normale -> arena in
worker CPU, H2D async su copy stream, eventi invece di sync globali, e
eviction/promotion guidate da REAP/SPEX.

Metriche obbligatorie:

- hit/miss VRAM;
- hit/miss arena pinned;
- fallback RAM normale e SSD;
- byte e tempo RAM -> pinned;
- byte e tempo H2D;
- numero/tempo `cudaStreamSynchronize`;
- overlap compute/copy;
- churn/eviction e working set per layer/token;
- TTFT, prefill t/s, decode t/s;
- exactness/repeatability separata dalla qualita' L0-L3.

Non addestrare SPEX su un prompt o su una mask specifica. Il predittore deve
essere online/generalizzabile e servire lo scheduling della residenza, non
forzare una domain mask statica.

## 8. Mandato suggerito per Claude

1. Leggere questo file e il report porting completo.
2. Non fare `git reset`, non cancellare untracked e non duplicare directory.
3. Archiviare in repo gli artifact primari gate05/gate06 e correggere i docs che
   oggi dicono ancora `post-P1 non testata`.
4. Esportare la catena DS4 WIP come patch 0050 riproducibile, con hash/base.
5. Revisionare il diff corrente del runner n>=3 e committare solo i file
   intenzionali in un commit piccolo.
6. Eseguire gate07 n=3 con prefetch OFF, dopo preflight e lock GPU.
7. Se intra-arm non e' ripetibile, classificare il gate come indeterminato e
   aggiungere un gate stabile (50/59 token o logit margin). Non usare n=1.
8. Solo dopo exactness classificata, eseguire A/B prestazionale con almeno 4
   round alternati, stesso prompt/build/mask, warmup e cooldown.
9. Integrare il risultato nella 0051: arena pinned dinamica da 24 GiB sopra RAM
   normale resident, non mask statica baked e non 50 GiB pinned.
10. Committare e pushare documentazione, harness, patch e artifact selezionati;
    riportare esplicitamente test negativi e confondenti.

## 9. Comandi di orientamento

```powershell
git -c safe.directory='C:/Users/imanu/source/repos/reap-loop' `
  -C 'C:\Users\imanu\source\repos\reap-loop' status --short

git -c safe.directory='C:/Users/imanu/source/repos/reap-loop' `
  -C 'C:\Users\imanu\source\repos\reap-loop' log --oneline -12
```

```bash
flock -n /tmp/ds4-gpu.lock -c true
ps -eo pid,ppid,stat,cmd | grep -E '[d]s4-server|run_0050_controlled_ab'

cd /mnt/c/Users/imanu/source/repos/reap-loop
bash runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh \
  --preflight --prefetch off --gate-repeats 3
```

Non usare i tempi ON2-ON5 o i due singoli gate05/gate06 come A/B conclusivo.
Non confondere una build Windows riuscita con il superamento del tetto WDDM.
