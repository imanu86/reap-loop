# DS4 handoff a Claude — chiusura Codex 2026-07-26

## Stato operativo

L'orchestrazione Codex e' conclusa. Non avviare automaticamente altri run.

Postflight finale:

```text
ds4_processes=0
port8000_listeners=0
nsight_processes=0
gpu_used_mib=489
gpu_pstate=P8
ram_free_gib=54.03
```

Sono stati autorizzati ed eseguiti soltanto K0p, K1p, K0c e K1c. K2 non e'
stato eseguito; non e' stata costruita una KV realmente viva a 150k.

## Git e provenienza

Sorgente DS4 consolidata in un worktree sano:

```text
worktree=D:\ds4_work\wt-codex-k1-handoff
branch=codex/ds4-k1-handoff-20260726
base=f64cc89baf5ba890c11317b8a0283e25ace85eae
P0/P1=8ec491b175406c9f95b380b65ab0da5ebdc589ec
K1=eb03f7203d8c4cc95781bc73a181f3fe972a6a12
```

Hash del candidato K1:

```text
ds4.c=BB24C86FFAB9665B4406113363A08D9144FE68C595C8C604EFC1E64FBF3D73D8
ds4_cuda.cu=F64290214FCBF94D1C852386852143012D7DB8E713C7334654A8E17616129C8E
ds4_gpu.h=535CF8AB13FD57933129FE1E7A247C5D0B9F062A4DD51EEBEC9BAF453FFBB4D0
ds4_server.exe=82FA2E60EC778A058E079C794E6D147D370539A502870640ABDC6AE6DD603EDE
manifest_47=CF1150EC0E46197E78B7F8482F8D94A8DA0666D887B291743478AA9DD1F73609
```

Il vecchio `D:\ds4_work\wt-converge` resta dirty e non e' stato alterato
durante la consolidazione. Usare la nuova branch come cronologia Codex.

## PAGEABLE host RAM: correzione del verdetto

Il controllo trace-OFF con 10 GiB pageable ha prodotto:

```text
OFF0 weighted decode=2.302951 t/s
OFF10 weighted decode=2.512794 t/s
delta=+9.111893%
pageable_slots=1517
pageable_hits=2933
allocation_fallback=0
pageable_paged_out_before_copy=0
RAM_min=9.0-9.5 GiB
```

Il delta e' sotto la soglia convenzionale del 10% e ha una sola replica, ma
e' un segnale positivo comparabile alle dimensioni di varie patch precedenti.
PAGEABLE10 e' quindi HOLD/PROMISING, non respinto. PAGEABLE14, storicamente
usato sulla macchina dedicata, resta un candidato separato.

La regressione `-71.799280%` appartiene all'interazione Sampled10 contro
OFF10, con queue age e fallback exact in crescita. Non prova che piu' RAM
pageable rallenti DS4. La futura matrice corretta, se ripresa, e'
PAGEABLE0/10/14 con trace OFF, repliche e hard-fault/pagefile/H2D-per-tier.

## K1: risultato

Verdetto: `RUNTIME_FAIL_CORRECTNESS`; patch non promossa.

K1 raggiunge il risultato meccanico previsto:

```text
K0 migrations/restores=2/2
K1 migrations=1
K1 request_end_retains=2
K1 staged_reuses=1
K1 request_end_restores=0
K1 fallback=0
K1 sticky_off=0
interturn_bytes_avoided=4,218,484,736
```

Il follow-up prestazionale e' neutro:

```text
K0p suffix TTFT=22.213890 s
K1p suffix TTFT=22.044127 s
delta=-0.7642%

K0p turn2 wall=0.922020 t/s
K1p turn2 wall=0.927223 t/s
delta=+0.5643%
```

K1p ha avuto un cold outlier nel primo turno (`mature -24.6427%`), non
riprodotto da K1c. Va mantenuto come failure osservato, senza attribuzione
causale al lifecycle.

K1 libera VRAM al request-end:

```text
K0 request_end_free_vram_mib=0
K1p request_end_free_vram_mib=1370.1/1359.0
K1c request_end_free_vram_mib=1336.3/1334.3
```

Questa e' la base per una futura cache esperti adattiva. Con esperti da
6.75 MiB, 1 GiB addizionale contiene circa 151 esperti. Non consumare tutto il
margine: usare watermark e conservare almeno 512 MiB-1 GiB per transienti.

## Difetto exact circoscritto

Output testuale di entrambi i turni identico in K0/K1, ma il secondo turno non
e' bit-exact:

```text
turn1 logits: K0=7b650de41e6f0a81 K1=7b650de41e6f0a81
turn1 live-KV: K0=4ebf6dc070e81428 K1=4ebf6dc070e81428

turn2 logits: K0=e614bf7154a5ce97 K1=308afa9400eed81f
turn2 live-KV: K0=5104b8dcc73fdb76 K1=0b4184e2392b3ddd
```

La divergenza compare soltanto quando il suffisso breve viene consumato con
authority `HOST_STAGED`; K0 aveva ripristinato la KV su device. Il dominio da
auditare e' quindi il resumed short-suffix staged path/frontier ordering:

1. ordine append D2H e frontier commit;
2. live range raw circolare, inclusa la seconda span dopo wrap;
3. row frontier compresso/indexer prima del suffix;
4. generazione/fingerprint condivisi dalle view;
5. differenza tra ring read staged e device read ripristinato.

Non promuovere K1 sulla sola equivalenza testuale.

## Artefatti

Il bundle committato contiene:

- patch e receipt P1;
- receipt/metriche PAGEABLE10;
- patch, report statico e receipt K1;
- report e receipt runtime K0/K1;
- harness K0/K1 e manifest 47;
- manifest SHA-256 di 111 artefatti runtime.

Hash principali:

```text
DS4_KV_RUNTIME_FINAL_REPORT.md=C04975B76D9BD0D56A9210837FF988E72A062F8C67951032876AA9DD8A2DE576
ds4_kv_runtime_final_receipt.json=315FB368584BD105E1950FD698023B296E51E8799D31244427E3DDC7B8DBBD37
DS4_KV_RUNTIME_ARTIFACT_HASHES.sha256=1BDBFFBD58A2B52638ECC7AD6A2059E79BEC2B71C434DCBE79F7B3CE3399BE04
ds4_kv_phase_aware.patch=C30A274820C356AFF96089E05823B7A46714C5339E1FF6B7306FA58617588BFC
```

Raw run directory:

```text
C:\Users\imanu\Documents\Codex\2026-07-26\ds4-kv-phase-aware-followup\outputs\kv_phase_runtime
```

## Decisione di chiusura

- PAGEABLE10: promettente ma non promosso; non confonderlo con il failure del
  tracer Sampled10.
- K1: meccanismo ingaggiato e VRAM liberata, ma respinto per correctness.
- K2: non eseguito.
- Nessun altro run Codex e' in corso o autorizzato.

