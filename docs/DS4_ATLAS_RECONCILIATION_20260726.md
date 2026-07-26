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
SHA256=27C39EA9507359CBDEA70B83B93F6C0AA04BB1575E7EC3DBAE9B767A70F24A11
```
