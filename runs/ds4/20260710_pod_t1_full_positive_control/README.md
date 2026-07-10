# 2026-07-10 Pod T1 — FULL no-mask positive control (cyberpunk HTML)

Purpose: the missing positive control for the whole quality track. Question:
does the FULL model (no mask, `DS4_PACE=0`, no PACE at all) degenerate at
800-2000 tokens on the cyberpunk HTML prompt? Without this, "K23 breaks the
HTML" is not attributable (NEXT_STEPS_PLAN_20260710 T1; retro-grade
`runs/ds4/20260710_retro_grade_l0l3/` showed ALL 105 archived outputs at
<=800 tok are L0, so the budget confound had to be tested on the full model).

## Setup

- Pod: RunPod `nyx0ubkpva1j9c` (SECURE cloud, machine `xc8zgkahd330`),
  RTX 3090 24GB, $0.46/h, driver 580.159.03, image
  `runpod/pytorch:1.0.7-cu1290-torch280-ubuntu2404` (CUDA 12.9).
  **Left RUNNING at end of T1 by coordinator order** (patch-build handoff);
  termination responsibility transferred to the next agent.
- Host RAM 1007 GB: the 81 GB model is fully page-cached after download ⇒
  **RAM-hot regime** (same class as the other 20260710 pod runs, NOT the
  local 3060 28 GB regime).
- Cache regime: `--ssd-streaming-cache-experts 1024` (server flag). PACE off
  in all runs here, so no PACE cache knobs apply.
- 3 community 3090 pods failed the CUDA gate-check before this one
  (nvidia-smi OK but `cudaGetDeviceCount → 0 "unknown error"`; root cause
  found on 3rd pod: `/dev/nvidia-uvm` open fails with EIO ⇒ host-side broken
  UVM kernel module, unfixable in-container; confirmed independently with
  PyTorch). Secure cloud passed the same gate on first try.
- Runtime: built ON the pod from ds4 `80ebbc3` + canonical reap-loop patch
  series (`patches/README.md`): 0001–0008 and 0011–0014e applied clean
  (after stripping CRLF picked up by the Windows checkout — the exact trap
  patches/README.md rule 1 warns about).
  - 0009/0010 (dspark-MTP) do NOT apply on the clean pinned base (context
    mismatch) — skipped; MTP is unused in T1.
  - 0015/0016-pace + 0018 need struct fields (`prefill_apply`,
    `prefill_wait_wrap`) that exist only in the uncommitted local live tree,
    not in any canonical patch ⇒ **rotation is not available in this pod
    binary** (irrelevant for `no_pace`; blocks only an optional rotate32 A/B,
    which was therefore not run).
- `ds4-server` SHA256:
  `49b31d3c9e41f7db304ee3e397df44eb05d6403bef345c924d74fe27c2972e20`
  (`ds4server.sha256`; differs from the WSL-copied binary of the sibling runs
  `0f1f0c7d…` — same source lineage minus MTP/rotation patches).
- Runner: `scripts/run_ds4_exchange_matrix.py` @ `d0ad967`
  (sha256 `203c3096…`), grader `scripts/functional_grade.py` @ `0d1d269`
  (sha256 `3341f6b7…`), L0-L3 column from `grade_l0l3` (no node on pod ⇒ JS
  check = heuristic fallback).
- Protocol: greedy temp 0, `think=false`, trace off, manifest per run
  (DS4_RUNNER_PROTOCOL). Sampling arm via `run_t1_sampling.py` (in this dir):
  temperature 0.7, top_p 0.95, **fixed seed 42**, non-stream, same server
  shape and manifests.
- Variant: `no_pace` (`DS4_PACE=0`) = full router, all 256 experts/layer
  eligible, no mask ever applied.
- Prompts: runner `html` (cyberpunk, 78 prompt tokens) and `html_coffee`
  (exact compact coffee-shop replay prompt). ctx 3072 (6144 for the 4000-tok
  run), server-max-tokens 1100/2300/4400, prefill-chunk 128.

## Results (13 runs — reported separately, not only medians)

All t/s are **pod numbers, RAM-hot 3090: DIAGNOSTIC ONLY, not comparable with
the local 3060.** Flags counted on `content_measured.txt`; L = `l0l3` grade.
`ident` = byte-identical to the other runs of its group.

| Arm | Run | compl.tok | finish | L | doctype | body | </html> | form | script | button | alert | repeat | wall s | t/s (pod) |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy html 800 | r01 | 800 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 174.3 | 4.91 |
| greedy html 800 | r02 (ident) | 800 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 167.5 | 5.13 |
| greedy html 800 | r03 (ident) | 800 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 166.1 | 5.18 |
| greedy coffee 800 | r01 | 800 | length | **L1** | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 192.1 | 4.80 |
| greedy coffee 800 | r02 | 800 | length | **L1** | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 185.5 | 5.01 |
| greedy coffee 800 | r03 | 785 | **stop** | **L3** | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 0 | 176.9 | 5.19 |
| greedy html 2000 | r01 | 2000 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 426.2 | 4.82 |
| greedy html 2000 | r02 (ident) | 2000 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 420.8 | 4.88 |
| greedy coffee 2000 | r01 | 819 | **stop** | **L3** | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 0 | 203.3 | 4.61 |
| sampled html 2000 | r01 | 2000 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 509.3 | 4.02 |
| sampled html 2000 | r02 (ident) | 2000 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 502.9 | 4.07 |
| sampled html 800 | r01 | 800 | length | **L0** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 201.9 | 4.20 |
| greedy html 4000 | r01 | **3498** | **stop** | **L2** | 1 | 1 | 1 | 1 | 1 | 5 | 4 | 0 | 739.7 | 4.80 |

Key qualitative fact: in **13/13 runs `repeat_flag=0`** — no repetition loop,
no corrupted CSS, ever. The truncated cyberpunk outputs are clean, coherent,
verbose CSS that simply never reaches `<body>` within 800/2000 tokens
(`finish=length`), until at 4000 budget the model closes the whole page
naturally at 3498 tokens (`finish=stop`, L2: complete hero/cards/form/popup
page, minor defects only).

## Verdetto T1 (secco)

- **Il FULL degenera a 800/2000? NO — su nessun prompt.** Nessun loop,
  repeat=0 ovunque, CSS sempre coerente.
- **Il FULL è L0 a 800 sul cyberpunk? SÌ — e anche a 2000** (greedy E
  sampled): il prompt elicita un CSS così verboso che `<body>` arriva solo
  oltre ~2000 token; il documento completo richiede ~3500 token (misurato:
  L2 a 3498 tok, finish=stop).
- **Prompt compatto (coffee): L1 a 800 (2/3), L3 a 785-819 tok** quando
  chiude da solo. La differenza 800-tok L0-vs-L1/L3 tra i due prompt è
  interamente prompt-driven.
- **Sampling (0.7/0.95/seed 42) non cambia il fenotipo** del cyberpunk:
  L0 CSS-only a 800 e a 2000, come il greedy.
- **Il 2-bit NON è indiziato**: a budget sufficiente produce una pagina L2
  completa e pulita sul prompt difficile e L3 sul prompt compatto.

## Implicazione per l'attribuzione a K23

"K23 rompe l'HTML a 800 tok" va spezzato in due claim distinti:

1. **"L'output a 800 tok è L0"** — NON attribuibile a K23 (né ad alcuna
   config): anche il FULL è L0 a 800 e a 2000 sul cyberpunk. È il
   **budget-confound** previsto dal retro-grade, ora dimostrato sul controllo
   positivo. Un grade L0 a <=800 tok sul cyberpunk non è evidenza contro
   nessuna mask.
2. **"L'output K23 mostra loop/CSS corrotto"** — QUESTA resta la firma
   attribuibile alla mask: il FULL a parità di budget e prompt produce CSS
   pulito senza mai loopare (13/13 repeat=0), mentre gli output K23/rotate32
   del retro-grade loopano dentro `<style>` con corruzioni (`#l Lime`,
   `##0f0`). La metrica giusta per giudicare le mask a basso budget è la
   firma di degenerazione (loop/corruzione), non il livello L da solo.
- Corollario operativo: i confronti qualità sul cyberpunk richiedono budget
  ~4000 tok (o il prompt coffee compatto a 800) per separare le config al
  livello funzionale L.

## Determinismo greedy (dato collaterale prezioso)

- **Cyberpunk greedy: bit-identico run-to-run** su questo pod/build — 3/3 a
  800 (2789 char uguali) e 2/2 a 2000 (7158 char uguali). In contrasto con il
  non-determinismo appena misurato in locale sul 3060 (divergenza ~tok 75) e
  con la nota storica del playbook (trace routing divergenti su pod): il
  determinismo dell'OUTPUT dipende da build/hardware/prompt, non è una
  proprietà garantita né esclusa di ds4-CUDA.
- **Coffee greedy: NON deterministico anche qui** — r01/r02/r03 divergono a
  char ~346 (≈ tok 100) e i grade variano L1/L1/L3. I tre output sono
  salvati integralmente e riportati separatamente sopra.
- **Sampled con seed fisso 42: bit-identici tra loro** (r01=r02 a 2000) — il
  seed è onorato dall'endpoint. Nota metodologica: per misurare la
  *diversità* del sampling servono seed diversi; questi n=2 misurano la
  riproducibilità, non la varianza.

## Costi e stato pod

- Spesa totale track al momento della scrittura: **$0.81**
  (balance 25.3130 → 24.5050): ~$0.07 per i 3 community pod falliti al gate,
  ~$0.74 per il secure pod (deploy 07:39Z, T1 ALL_DONE 09:11Z).
- **Pod NON terminato** (ordine coordinator, cap alzato a $8 totali): resta a
  $0.46/h per il prossimo agente, che eredita la responsabilità di terminare.

## Handoff pod (per il prossimo agente)

- Pod `nyx0ubkpva1j9c`, SSH: `root@213.192.2.91 -p 40053` (chiave
  `~/.ssh/id_ed25519` della workstation).
- ds4 checkout: `/root/ds4` (80ebbc3 + patch committati uno-a-uno, vedi
  `git log`); build: `PATH=/usr/local/cuda/bin:$PATH make cuda
  CUDA_ARCH=sm_86 -j16`.
- Modello: `/root/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
  (86.7 GB), symlink `/root/models/ds4-2bit.gguf`. Runner+grader in
  `/root/reap-loop/scripts/`.
- Disco: 81G usati / 50G liberi (130G). VRAM: libera (nessun server attivo a
  fine T1). GPU sanity: gate-check micro-kernel passato (kernel_result=42).

## Files

- `greedy_800/`, `greedy_html2000/`, `greedy_coffee2000/`,
  `sampled_html2000/`, `sampled_html800/`, `greedy_html4000/` — per ogni run:
  `content_measured.txt`, `request/response_measured.json`,
  `runner_manifest.json`, `server_env.json`, `server.std{out,err}.log`,
  (greedy) `stream_events_measured.jsonl`, più `summary.csv` /
  `summary_median.csv` / `summary.json` per arm.
- `run_t1_sampling.py` — harness del braccio sampling (riusa i helper del
  runner); `t1_runs.sh` — sequenza eseguita sul pod; `t1_progress.log` —
  log di avanzamento; `ds4server.sha256` — hash del binario.
