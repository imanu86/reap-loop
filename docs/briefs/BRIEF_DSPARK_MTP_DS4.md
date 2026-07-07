# BRIEF — Track DSpark/MTP su ds4 (chat parallela)

> Sei una chat/agente dedicato al track DSpark. Lavora su **branch `dspark/mtp-spec-dec`**.
> Contesto generale: vedi note interne di progetto. ⚠️ TERMINOLOGIA: **DSpark ≠ SPEX.**
> - **DSpark** = paper DeepSeek (`docs/references/DSpark_paper.txt`): speculative decoding a livello di
>   TOKEN (draft semi-autoregressivo + confidence head + STS + scheduler). QUESTO track.
> - **SPEX** = prefetch predittivo degli EXPERT (altro track, altro branch). Non toccarlo.
>
> ⚠️⚠️ **SECONDA TRAPPOLA TERMINOLOGICA (dal paper, abstract): il `--mtp` di ds4 NON è DSpark — è il
> BASELINE.** ds4 ha `--mtp FILE --mtp-draft N (default 1) --mtp-margin F` = spec-dec naive con la testa
> MTP nativa = esattamente il "**MTP-1**" che il paper usa come production baseline e batte del 60-85%.
> Quindi: misurare `--mtp` = misurare il BASELINE (necessario come riferimento), NON il metodo.
> **DSpark vero = 2 componenti assenti in ds4:** (1) drafter semi-autoregressivo ADDESTRATO (backbone
> parallelo + testa sequenziale; checkpoint open-sourced da DeepSeek + repo DeepSpec) e (2) verifica
> **confidence-scheduled** (lunghezza adattiva per richiesta da prefix-survival probability + STS +
> profilo throughput del motore) al posto di N/margine FISSI.
> **Strada A (prima):** tieni l'MTP nativo come drafter, implementa la verifica confidence-scheduled sopra.
> **Strada B (dopo, se serve):** portare i checkpoint DSpark in formato ds4.
> **Twist nostro (paper-worthy):** nello scheduler, il costo di verifica in streaming-MoE si AMMORTIZZA
> col blocco (−49% IO expert a k=8, misurato dalla trace) → la lunghezza ottima del draft è più lunga
> che nel paper denso; aggiungere il termine expert-IO al profilo di throughput.

## Obiettivo
Portare lo **speculative decoding a blocchi** su ds4 per il regime streaming del 3060. Il razionale è
MISURATO sulla trace reale (`runs/ds4_routing_trace_smoke/`): decodificando un blocco di k token insieme,
gli expert unici per layer sono molti meno di 6·k:

| k (blocco) | expert unici/layer | risparmio IO expert |
|---|---|---|
| 2 | 10.6 | 12% |
| 4 | 16.4 | 32% |
| 8 | 24.3 | **49%** |

Con l'IO expert a ~90% del wall-clock in streaming, un blocco-8 con buona acceptance ≈ potenziale ~2×
sul collo di bottiglia, CUMULABILE con cache-sizing e SPEX.

## Step (in ordine, il primo è RECON non codice)
1. **RECON ds4 upstream:** cosa fa GIÀ ds4 con MTP? Il modello MTP c'è
   (`models\ds4\DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`, 3.8GB; in WSL /root/models se copiato).
   Cerca nel sorgente (`/root/ds4`, copia Windows in scratchpad ds4-src): flag `--draft`/`mtp`/`speculative`,
   come carica il file MTP, se il decode accetta batch di verifica. Deliverable: mappa con file:riga di
   cosa esiste e cosa manca. NON riscrivere ciò che c'è.
2. **Quantifica l'acceptance attesa:** MTP di DeepSeek-V4 dichiara acceptance ~alta sui domini tipici;
   misura con un run reale (se ds4 ha già lo spec-dec: accendilo e misura acceptance + t/s vs baseline —
   coordinati per il 3060, è conteso).
3. **Gap-analysis DSpark:** cosa aggiunge DSpark rispetto all'MTP naive di ds4 = confidence head per
   token drafted + **STS** (calibrazione temperatura, min-ECE) + scheduler hardware-aware (Alg.1) che
   decide QUANTI token draftare per step. Design doc: dove innestarlo in ds4 (file:riga), quali parametri
   servono, come si fitta STS (abbiamo la macchina in `src/msc/spex/spex_loop.py` — la logica STS è la
   stessa, dominio diverso).
4. **Sinergia streaming (l'idea nostra, da verificare):** nello step di verifica del blocco, gli expert
   del blocco si conoscono DOPO il router di ogni layer ma il blocco è batch → le load per layer si
   fanno UNA volta per blocco (union). Verifica che il path batch di ds4 lo faccia davvero (o se
   ricarica per-token) — è lì che vive il 49%.

## Regole
- Branch tuo, commit piccoli, push frequente. Misure solo da log su disco, citati.
- GPU locale contesa: coordina i run (un solo processo ds4 alla volta).
- **Pod autorizzati: budget $1-2 TOTALE per questo track** (RTX 3090 community, sm_86 come il 3060).
  Metodo di misura e disciplina deploy/terminate: segui ALLA LETTERA `docs/briefs/POD_PLAYBOOK.md`.
  Caso d'uso pod perfetto per te: misurare l'**acceptance-rate MTP** (proprietà del modello → trasferisce)
  e verificare l'union-load nel path batch. I t/s del pod NON trasferiscono al 3060: dichiaralo.
- Aggiorna lo stato qui in coda.

## Stato
- [x] recon MTP in ds4 (mappa file:riga) → `docs/dspark/RECON_MTP_DS4.md` (2026-07-05).
      ds4@80ebbc3 ha GIÀ lo spec-dec MTP completo (`ds4_session_eval_speculative_argmax`,
      ds4.c:27167) con margin gate e probe di acceptance (`DS4_MTP_PROBE`). MA
      `--mtp` + `--ssd-streaming` è VIETATO dall'engine (ds4.c:25685) → il 49% vive
      esattamente nella combinazione bloccata.
- [x] verifica union-load nel path batch → SÌ: dedup+compact-load per blocco in
      `ds4_gpu_stream_expert_cache_prepare_selected_batch` (ds4_cuda.cu:3176), invocato dal
      path batch streaming (ds4.c:18938). Il verifier MTP usa lo stesso encode_layer_batch
      → union-load "gratis" una volta sbloccato MTP+streaming. Dettagli nel RECON §1.7.
- [x] misura acceptance baseline MTP-1 COMPLETA (2026-07-05): **code 0.872, math 0.846,
      chat 0.604** (×2 run bit-identici, greedy, trasferisce al 3060). In streaming con
      combo probe draft2+SPEC_DISABLE+UNSAFE (patch `patches/ds4/0006`).
      → `runs/dspark/20260705_mtp_acceptance_pod3090/RISULTATI.md`
- [x] union-load VERIFICATA E QUANTIFICATA a runtime: prefill 670 tok → 4020 slot/layer
      compattati in 132-191 unici (media 163, **-95.9%**), 43 layer. Stesso meccanismo
      che al blocco-8 vale il 49%. Log compact_counts committati.
- [x] **FASE B offline (2026-07-05)**: STS fittata sui 585 cicli reali (holdout valida;
      chat ECE dimezzata) + scheduler R=1 IO-aware simulato sugli esiti veri:
      **dynamic-STS 1.91× IO/token vs fixed-5 1.46×**, 97-99% dell'oracolo; su chat il
      blocco fisso è 0.92× (peggio del liscio) e il dinamico 1.71× — riprodotto il
      motivo per cui la produzione era ferma a MTP-1. Codice `src/msc/dspark/`,
      risultati `runs/dspark/20260705_fase_b_sts/RISULTATI.md`, `sts_params.json`
      pronto per il loader C.
- [x] **FASE C SBLOCCATA (2026-07-05 sera)**: primo spec-dec MTP in streaming su ds4,
      sul 3060 reale. Patch 0009 (guardia env-gated + fix PR#497) e 0010 (slot
      registrazione durevole mappa MTP — bug upstream: il remap streaming la
      de-registrava). 33-34 cicli verify/run, acceptance pos2 0.848, union-load viva
      nel verify (slots=12→~10 unici), spec2 **+9/10% vs baseline** col PAVIMENTO
      (expert cache spenta: le statics MTP device-cachate mangiano il budget su 12GB
      → prossimo innesto 0011). → `runs/dspark/20260705_fase_c_smoke_3060/RISULTATI.md`
- [ ] 0011 (statics support model non cachate) → misura con expert cache vera
- [ ] scheduler+STS on-device (innesti 1-4 design) + ds4-eval appaiato (paletto 3)
- [x] **STRADA B MISURATA (2026-07-05)**: DSpark ufficiale su V4-Flash (2×H200, ~1h, $9):
      τ = **5.18/5.00/2.70** (code/math/chat) su blocco-5; confidence head ottima come
      ranking ma sottocalibrata in profondità (ECE fino a 0.44 su chat) = caso d'uso
      esatto della STS. Dataset calibrazione (585 cicli) + risultati:
      `runs/dspark/20260705_dspark_b_v4flash_2xh200/RISULTATI.md`. Pod terminato+verificato.
- [x] design DSpark (confidence+STS+scheduler) → `docs/dspark/DESIGN_DSPARK_DS4.md`
      (2026-07-05): confidence-lite dal margine MTP calibrata via STS, scheduler Alg.1
      ridotto a R=1 con cost-table IO-aware, sblocco streaming = guardia ds4.c:25685 +
      stream-map del verifier. Fasi A(misura)/B(scheduler)/C(streaming)/D(head).
