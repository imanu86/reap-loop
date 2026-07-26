# DS4 — riconciliazione Atlante Claude con la matrice corrente

Data: 2026-07-26

## Fonti verificate

```text
DS4_LEVER_ATLAS_20260724.md
SHA256=1B8BEDBE973E321958CE2AA0D9568EF2ADD9C4483A5F6CFE974A5A1C07B42E8D

DS4_LEVER_ANALYSIS_POST_20260724.md
SHA256=568ABE823C0512BDBE9C80037935E28A7059E17386CDAAC2856C6EBED64855BB
```

Entrambi gli artefatti sono stati letti integralmente e confrontati con le
correzioni successive nel ledger, con il sorgente corrente e con M0-M4.

## Verdetto

Abbiamo già combinato quattro leve decode, ma soltanto all'interno di una
configurazione macro fissa:

```text
M0 = nessuna micro-patch prestazionale
M1 = P3-B route I/O QD4
M2 = M1 + P2-B batched publish
M3 = M2 + P3-A host-selected reuse
M4 = M3 + P2-A packed copy
```

M2 è il migliore valido a `2.463047 t/s`, `+7.3103%` contro M0: sotto la
soglia utente del 10%, quindi varianza e non nuova baseline. M4 ha riprodotto
un singolo fallback P3-A nel follow-up in due tentativi ed è invalido per il
gate, pur con output esatto. Tutte le patch restano conservate.

L'audit dell'Atlante mostra che la campagna è stata troppo stretta, non che
esista una configurazione certa da 5 t/s dimenticata. La baseline recente
`G73_OPEN=1`, solo pinned, chunk250 è coerente con il successivo dato stabile
post-Atlante da circa `2.41 t/s`. Il vecchio `4.85 t/s` con `G73_OPEN=0` non
si riprodusse sul binario successivo (`2.19 t/s` contro `2.41 t/s` open/pinned).

## Leve già assorbite o chiuse

| Leva Atlante | Stato corrente |
|---|---|
| `DS4_G73_PAGEABLE_OVERFLOW_GB` | già 0; il vecchio 14 GiB peggiorò circa 2.41 → 2.07 t/s |
| `DS4_METAL_PREFILL_CHUNK` | già 250; il segno storico non è universale e va ri-A/B sul binario corrente |
| `DS4_CUDA_KV_STAGED_RING` | già 1 |
| `DS4_CUDA_MOE_ROUTE_NO_DEFAULT_SYNC` | già 1; beneficio storico piccolo |
| `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS` | falsificata come leva prestazionale |
| `DS4_CUDA_MOE_IO_QD` | vecchio path non utile; P3-B è un nuovo path exact-decode distinto |
| `DS4_CUDA_KV_MANAGED` | bassa priorità: advice fallito su WDDM e incompatibile col ring staged |

## Gap reale della matrice M0-M4

Due leve dell'Atlante sono presenti nell'esatto sorgente recente ma erano unset:

- `DS4_CUDA_GRAPH_TENSOR_DEVICE=1`: tensori generici da managed a device-only;
  stima statica di `180-276 MiB` di padding evitato. Richiede A/B e verifica
  dell'effettivo sizing della cache, non una promozione per teoria.
- MTP vero: il runner usa `--mtp-draft 1`, quindi non specula. Il codice contiene
  `DS4_MTP_BATCH_VERIFY` per `draft_n=2`. Va misurato con `temperature=0` e
  `think=false` su entrambi i bracci.

Anche `DS4_G73_OPEN` deve essere riaperto, ma soltanto come A/B sul medesimo
binario e con le micro-patch G73-specifiche spente. Il vecchio 7.8x è una
osservazione storica specifica del binario, non una baseline.

## Prossima matrice

1. M5: `P3-B + P2-B + P2-A`, P3-A off.
2. A1: `GRAPH_TENSOR_DEVICE` off/on su una baseline valida.
3. A2: `G73_OPEN` 1/0 su P0, pageable0, tutte le micro-patch off.
4. A3: chunk 250/600 sul ramo vincente; 768 solo se giustificato.
5. A4: greedy draft1 contro draft2 + batch verify.
6. C1: composizione dei soli vincitori con il miglior sottoinsieme P2/P3.

Per ogni braccio: 47/47 base vars più overlay, stesso binario/modello/prompt,
hash output, t/s wall e graph, power/utilization GPU, H2D/D2H, route class,
fallback, arresto nativo e macchina pulita. Delta medi sotto il 10% sono
varianza. Il vincitore rapido a capacità 150k deve poi essere verificato a
posizione KV realmente lunga.

## Direzione

Nessun test mostra banda GPU satura. Le leve sopra possono liberare headroom e
ridurre attese, ma la direzione capace di alimentare davvero la GPU resta:

1. copie H2D e publish compattati per layer;
2. stream demand separato dalle promozioni, sincronizzato con eventi;
3. buffering con ownership `consumer_done`;
4. residency indexer/KV a 150k sotto watermark;
5. solo dopo QD I/O più alta, fusioni e CUDA Graph guidati dalle misure.

Piano operativo locale sincronizzato:

```text
path=C:\Users\imanu\Documents\Codex\2026-07-25\legg\DS4_OPERATIONAL_PLAN.md
SHA256=D8A8E45B26E0695935BB4C86E7EA34F2AA9EAC4AF069F6A5DE822C293CA5F7B6
```

## Orchestrazione attiva 2026-07-26

Sono state create e pinnate sei task separate:

```text
M5=019f9e6b-a7f8-7ac2-bb64-d3114ee0c03a
A1_GRAPH_TENSOR_DEVICE=019f9e6c-1ce2-7c40-89f7-fc0b78a32c35
A2_G73_OPEN=019f9e6c-7b3f-7432-baed-38b492f5eb8a
A3_PREFILL_CHUNK=019f9e6c-e35a-7963-84c2-23a0e9acc763
A4_MTP_BATCH_VERIFY=019f9e6d-375d-7651-996e-8e4a4a7f2f59
P4_DEMAND_PROMOTION_PIPELINE=019f9e6d-9fa8-7792-afeb-1bbe6eab7c34
heartbeat=ds4-lever-orchestrator-5m
```

Politica: un solo run DS4 fisico alla volta; le altre task restano read-only o
preparano harness isolati. Shutdown nativo e postflight pulito sono gate.

M5 è chiuso:

```text
correctness_and_engagement=PASS
performance=FAIL_NOT_PROMOTED
turn1_graph_tps=2.476776
turn1_mature_tps=2.780617
turn2_decode_graph_tps=0.890385
weighted_decode_tps=1.826076
ledger_commit=f5057ccb9f1e948848190fbbcf5e09904c216b8d
```

Nuovi fatti emersi prima di spendere altri run:

- A1 non è sicuro sul binario corrente: 124 tensor state sono inizializzati
  dereferenziando dalla CPU un pointer che il flag rende device-only. È in corso
  correzione minima, static contract e build isolato.
- A2 non è validabile col runner corrente: il manifest fissa OPEN=1, il gate D2H
  è incompatibile con OPEN=0 e un fallimento finale del rotatore può false-pass.
- A3 250→600 muove circa 462,7 MiB di allocazioni note e deve aspettare A2.
- A4 richiede realmente `--mtp <file>` oltre a draft2/batch verify/temp0/no-think;
  il GGUF MTP locale è stato verificato.
- P4a è ora quantificato: circa 603 H2D/token devono diventare al massimo 86,
  con un publish demand per layer; P4b separerà le promozioni solo dopo P4a.

## Checkpoint P0/A1 e nuove leve

Task aggiunte e pinnate:

```text
P0_G73_ROTATOR=019f9e90-b846-7151-a5b4-47f4641a1a0d
P1_SAMPLED_TRACE=019f9e9a-8aa3-7670-be40-103c24ca75c3
P5_LONG_KV_RESIDENCY=019f9e9a-d59d-70a3-813e-8193954ab7b6
```

P0 ha corretto una failure reale del rotator G73: un singolo job I/O fallito
veniva contato due volte e trasformato in latch terminale, nonostante la
rotazione avvenga dopo il serving exact. La correzione rende la failure
job-local, libera ownership/ring, consente submit successivi, conserva un
receipt `degraded` fail-closed per il benchmark e mantiene terminali gli errori
di ownership/generation/commit e il percorso non-G73.

Il primo A1 OFF sul binario P0 e' stato fermato correttamente:

```text
run=20260726_155302_a1_m0_p0_trace-off_packed-off_publish-off_hostsel-off_routeio-off_graphdev-off
serving_exact=PASS
rotator=DEGRADED
rotator_attempts=407
rotator_successes=362
rotator_advisory_failures=45
rotator_advisory_releases=45
rotator_stale_dropped_structural=0
overall_gate=FAIL
graph_tensor_on=NOT_RUN
turn1_wall_tps=1.076165
turn1_graph_tps=2.236573
turn1_mature_tps=2.602229
turn2_wall_tps=0.907693
turn2_suffix_prefill_tps=1.081058
turn2_decode_graph_tps=2.436381
weighted_decode_tps=2.273869
ledger_commit=460f4cf73002d81d5fb6c868265c75fd68b18ad1
shutdown=native_verified
postflight=clean
```

Tutti i 45 eventi hanno `range=0 expected=2162688 actual=0 errno=138`
e durata 0,2-1,2 microsecondi. Non parte alcuna lettura SSD: la deadline
`queued_at + 50ms` e' gia' scaduta quando il worker background acquisisce il
job. P0.1 deve iniziare lo stesso budget da 50 ms a `io_started`, lasciando
`queued_at` ad aging/telemetria e senza rilassare il gate.

P1 ha trovato inoltre un underflow del contatore `pread_bytes` dovuto al reset
nel worker e ha definito telemetry v2 con 178 event CUDA preallocati, capture
Nsight di 1-2 token e hard gate overhead 2%.

P5 ha chiuso il budget live-150k:

```text
raw_plus_indexer_ratio4_plus_attention_ratio128_resident=504206336_bytes
attention_ratio4_staged=1761693696_bytes
target_kv_indexer_h2d=22020096_bytes_per_token
target_kv_indexer_submissions=21_per_token
watermark_target=1_GiB
watermark_floor=512_MiB
capsule=D4CP_v2
```

Il piano operativo persistente aggiornato e':

```text
path=C:\Users\imanu\Documents\Codex\2026-07-25\legg\DS4_OPERATIONAL_PLAN.md
SHA256=5E99F27467B8D032404ABB9996886C75D5BDBEA11C3C1156854423464B0B64A3
```

## A1 valido su P0.1

P0.1 ha separato correttamente queue age e budget I/O: il timeout configurato
resta 50 ms ma parte quando il worker acquisisce ciascun job. Il successivo A/B
GraphTensor e' valido:

```text
OFF_run=20260726_161517_a1_m0_p0_trace-off_packed-off_publish-off_hostsel-off_routeio-off_graphdev-off
ON_run=20260726_161920_a1_m0_p0_trace-off_packed-off_publish-off_hostsel-off_routeio-off_graphdev-on
OFF_rotator=complete_371_of_371
ON_rotator=complete_373_of_373
correctness_engagement=PASS
weighted_decode_OFF_tps=2.294284
weighted_decode_ON_tps=2.296465
weighted_delta=+0.095pct
max_throughput_delta=+6.188pct
performance=NEUTRAL_VARIANCE
observed_decode_start_vram_recovery=42.0_to_96.7_MiB
promotion=NO
baseline=GraphTensorDevice_OFF
ledger_commit=e50abb5240d84635bcf83e6f1109e12b64bf4e45
```

P1 ha ora lo slot sorgente/build esclusivo per correggere l'underflow
`pread_bytes` e implementare il tracing campionato v2 con 178 CUDA timing
events preallocati. Nessun run o capture Nsight e' autorizzato durante la build.

Piano operativo:

```text
path=C:\Users\imanu\Documents\Codex\2026-07-25\legg\DS4_OPERATIONAL_PLAN.md
SHA256=F6A8FC7528DC4A4D1E1568087A87E042164EB26D01DD25A730F8E1D820EF93A8
```
