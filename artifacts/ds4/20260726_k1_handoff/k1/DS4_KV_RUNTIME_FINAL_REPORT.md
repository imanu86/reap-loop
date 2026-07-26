# DS4 KV phase-aware — runtime finale K0/K1

Verdetto: **RUNTIME FAIL — exact correctness**. La patch non è promossa.

Sono stati eseguiti esclusivamente i quattro arm autorizzati, nell’ordine
K0p → K1p → K0c → K1c. Nessun K2, KV live150k, Nsight, PAGEABLE sweep o altra
leva. Ogni arm ha usato un solo `ds4_server`, manifest 47, prompt e seed
congelati, shutdown HTTP nativo e postflight pulito.

## Provenance congelata

```text
source_branch=codex/ds4-k1-handoff-20260726
p0_p1_commit=8ec491b175406c9f95b380b65ab0da5ebdc589ec
k1_commit=eb03f7203d8c4cc95781bc73a181f3fe972a6a12
ds4.c=BB24C86FFAB9665B4406113363A08D9144FE68C595C8C604EFC1E64FBF3D73D8
ds4_cuda.cu=F64290214FCBF94D1C852386852143012D7DB8E713C7334654A8E17616129C8E
ds4_gpu.h=535CF8AB13FD57933129FE1E7A247C5D0B9F062A4DD51EEBEC9BAF453FFBB4D0
ds4_server.exe=82FA2E60EC778A058E079C794E6D147D370539A502870640ABDC6AE6DD603EDE
manifest_47=CF1150EC0E46197E78B7F8482F8D94A8DA0666D887B291743478AA9DD1F73609
```

## Performance

Le metriche graph sono ricavate dai record `metal graph token`: turn 1
posizioni 13–140, mature 50–140, suffix 141–159, follow-up decode 160–191.

| Arm | Turn-1 wall t/s | Turn-1 graph t/s | Mature graph t/s | Suffix graph t/s | Turn-2 decode graph t/s | Turn-2 wall t/s | Suffix TTFT s |
|---|---:|---:|---:|---:|---:|---:|---:|
| K0p | 1.128046 | 2.722257 | 3.294191 | 0.993709 | 3.090835 | 0.922020 | 22.213890 |
| K1p | 0.321301 | 0.741649 | 2.482412 | 0.990207 | 3.050100 | 0.927223 | 22.044127 |
| K0c | 1.133151 | 2.725652 | 3.312857 | 1.008919 | 3.051504 | 0.923378 | 21.839922 |
| K1c | 1.150785 | 2.728207 | 3.342339 | 0.991488 | 3.079346 | 0.932680 | 22.039279 |

Delta K1p vs K0p:

| Metrica | Delta | Classificazione |
|---|---:|---|
| Turn-1 wall | -71.5170% | FAIL >10% |
| Turn-1 graph | -72.7561% | FAIL >10% |
| Mature graph | -24.6427% | FAIL >10% |
| Suffix graph | -0.3524% | varianza |
| Turn-2 decode graph | -1.3179% | varianza |
| Turn-2 wall | +0.5643% | varianza |
| Suffix TTFT | -0.7642% | varianza/miglioramento |

K1p ha incontrato un bulk-wrap/load freddo molto lento prima della maturazione:
al token 50 era a 0.33 t/s, ma il chunk 50→100 è risalito a 3.10 t/s e il
100→128 a 3.53 t/s. Il medesimo K1 con checksum (K1c) non ha riprodotto
l’anomalia: tutte le metriche salvo suffix TTFT restano entro ±1.73% rispetto
a K0c. Non attribuiamo causalmente il cold outlier al lifecycle, ma la regola
congelata `regressione >10%=fail` lo registra comunque come FAIL.

Il comparator prestazionale focalizzato sul follow-up ha dato PASS:

```text
turn2_suffix_ttft_delta=-0.007642
turn2_wall_tps_delta=+0.005643
performance_classification=variance
```

## Round-trip KV e memoria

K0 esegue per ciascun turno una migrazione D2H e un restore H2D full-capacity:

```text
device/copy bytes per operazione=2,109,242,368
migrations=2
restores=2
request-end free VRAM=0 MiB
```

K1:

```text
initial live copy=1,306,112 bytes at 13 tokens
live bytes at first request-end=13,194,752
live bytes after suffix=13,463,552
live bytes at second request-end=13,893,632
append D2H cumulative=20,199,936
migrations=1
request-end retains=2
follow-up staged reuse=1
request-end restore count/bytes=0/0
shutdown live restore=13,893,632 bytes
fallback=0
sticky-off=0
```

Il full round-trip tra i due turni è eliminato: 2.109.242.368 byte H2D al
request-end più 2.109.242.368 byte D2H al decode-start successivo, cioè
**4.218.484.736 byte e due operazioni full-capacity evitati**.

| Arm | RAM minima MiB | Memory load max | VRAM max MiB | VRAM free request-end MiB |
|---|---:|---:|---:|---:|
| K0p | 17,773.9 | 72% | 12,041 | 0 / 0 |
| K1p | 17,674.8 | 72% | 10,552 | 1,370.1 / 1,359.0 |
| K0c | 18,013.1 | 72% | 11,999 | 0 / 0 |
| K1c | 18,278.2 | 72% | 10,608 | 1,336.3 / 1,334.3 |

Non emerge un peggioramento RAM/paging: memory load massimo 72% in tutti gli
arm e letture cumulative ~87–88.6 GiB. Il pinned arena resta tuttavia
2.109.242.368 byte full-capacity.

## Correctness exact

Tutti e quattro gli arm hanno prodotto gli stessi hash testuali congelati:

```text
turn1=7f82253a4825191926f56073e40f10a0cff5541a721731bc81d2909dc1a4a65b
turn2=0179556c8e2dbcdc818fad315ca4df78f7537b63816dac276615b314195b13eb
```

Il confronto raw è però fallito al follow-up:

| Checksum | K0c turn 1 | K1c turn 1 | K0c turn 2 | K1c turn 2 |
|---|---|---|---|---|
| logits FNV-1a | `7b650de41e6f0a81` | `7b650de41e6f0a81` | `e614bf7154a5ce97` | `308afa9400eed81f` |
| live-KV FNV-1a | `4ebf6dc070e81428` | `4ebf6dc070e81428` | `5104b8dcc73fdb76` | `0b4184e2392b3ddd` |

Il comparator exact termina con exit 1:

```text
exact logits checksum mismatch
```

La localizzazione è netta: il primo turno, dove K0 e K1 usano entrambi il ring
dopo il decode-start, è exact. La divergenza compare soltanto dopo che K1
consuma il suffix con authority `HOST_STAGED`, mentre K0 aveva ripristinato la
KV su device prima del suffix. Questo circoscrive il difetto al resumed
short-suffix staged path/frontier ordering; non prova ancora quale singola
operazione lo causi.

Output testuale uguale non soddisfa il requisito P0 di logits e live-KV
bit-exact. La patch è quindi **respinta**, nonostante il round-trip sia stato
eliminato e non vi siano fallback visibili.

## Shutdown e ledger

Ogni arm:

```text
shutdown_mode=graceful_http_verified
post_ds4_processes=0
post_port8000_listeners=0
nsight_processes=0
```

Postflight finale: GPU 489 MiB / P8, RAM libera 54.03 GiB.

Il ledger non è stato modificato. Hash prima e dopo la matrice:

```text
B438296B115FEF83C9728C03D10C9BCB5540D3D655CEAB84955CA69B68A6FF36
```

## Verdetto

```text
round_trip_eliminated=true
short_followup_performance=variance
text_output_exact=true
logits_exact=false
live_kv_exact=false
fallback_count=0
sticky_off_count=0
promotion=false
final_status=RUNTIME_FAIL_CORRECTNESS
```

Nessun ulteriore runtime è autorizzato o necessario in questo handoff.

