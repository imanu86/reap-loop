# DS4 A4 — MTP batch-verify harness

`HARNESS_READY=yes`

`READY_FOR_RUN_SLOT=yes` — preparazione completata; serve ancora uno slot DS4
esclusivo e l'hash del binario deve restare quello indicato sotto.

`DS4_RUN_EXECUTED=no`  
`BUILD_EXECUTED=no`  
`SHARED_REPOSITORY_MODIFIED=no`  
`SHARED_RUNNER_MODIFIED=no`  
`COMMIT_CREATED=no`

## Identità runtime bloccante

| Artefatto | Percorso | Byte | SHA-256 |
|---|---|---:|---|
| Binario | `C:\Users\imanu\Documents\Codex\2026-07-25\legg\work\wt-hot-reserve\build2\ds4_server.exe` | 12,412,416 | `F2F6A25600EF0B6728BE0DAE77ACFC5CE89ABCEB1446555A7DB946DA26F11DFF` |
| Modello principale | `C:\ds4-models\ds4-2bit.gguf` | 86,720,111,488 | receipt `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668` |
| Modello MTP | `C:\ds4-models\DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | 3,807,602,400 | `AFD481EE689DCE9037F70F39085FCDAE5A5B096D521CDAD43B19FA52BF8F4083` |
| Harness standalone | `run_a4_mtp_lifecycle.ps1` | 59,719 | `7B08D0CDC5207B466422DCE7CB495DFBC8D79E48A458C299135B52DC0903FEB1` |

Il runner calcola integralmente gli hash del binario e del GGUF MTP prima di
raggiungere il preflight di avvio. Per il modello principale conserva il gate
preesistente: dimensione reale più hash receipt nel manifest.

Il source tree condiviso è cambiato durante la preparazione concorrente A1.
Gli hash correnti di `ds4.c`, `ds4_cuda.cu` e `ds4_gpu.h` non coincidono più
con il receipt del binario auditato. Il runner registra
`source_tree_matches_binary_receipt=0` ma usa l'hash del binario come gate
runtime. Qualunque rebuild cambia l'hash del `.exe` e blocca A4 fino a un
nuovo audit; non è consentito aggiornare il valore atteso alla cieca.

## Matrice esatta

Comune a tutti i run confrontabili:

- stesso binario, modello principale, GGUF MTP e manifest;
- `--cuda`, contesto `150000`, `--mtp-margin 3`;
- `temperature=0`, `think=false`, `seed=12345`, stessi prompt e token cap;
- `DS4_MTP_STRICT=1`;
- `DS4_MTP_SPEC_DISABLE`, `DS4_MTP_PROBE`, `DS4_MTP_CONF_LOG`,
  `DS4_MTP_FULL_LOGITS`, `DS4_MTP_MIN_MARGIN`,
  `DS4_MTP_CAPTURE_PREFIX1`, `DS4_MTP_EXACT_REPLAY` e
  `DS4_MTP_FORCE_SNAPSHOT` unset;
- `TraceMode=Off`, `DS4_METAL_GRAPH_TOKEN_PROFILE` unset e
  `DS4_REQUEST_PHASE_TRACE=1`;
- per la fondazione validata: `PackedCopy=Off`, `BatchedPublish=On`,
  `G73HostSelected=Off`, `RouteIoQd=Qd4`.

| Profilo | `--mtp-draft` | `DS4_MTP_BATCH_VERIFY` | Timing/spec log | Uso |
|---|---:|---|---|---|
| Control | `1` | unset | unset | riferimento greedy; MTP caricato ma speculative branch disabilitato |
| Batch2 diagnostic | `2` | `1` | entrambi `1` | prova engagement, drafted/accepted e fallback; escluso dalle misure performance |
| Batch2 performance | `2` | `1` | unset | A/B di throughput; richiede receipt Control e receipt diagnostic |

Con `draft=1`, il gate server `mtp_draft_tokens > 1` non può entrare nella
speculative acceptance. Con `draft=2`, temperatura greedy e
`DS4_MTP_SPEC_DISABLE` assente la rendono attivabile. Con strict presente e
batch-verify assente il codice sceglie `decode2 exact`; con batch-verify
presente evita quel ramo ed entra nel verifier microbatch. Evidenze del source
tree letto durante la preparazione:

- `ds4.c:19677` — log di caricamento MTP con draft;
- `ds4.c:20848-20850` — drafting solo con MTP pronto e draft maggiore di 1;
- `ds4.c:20953` — strict da quality o presenza di `DS4_MTP_STRICT`;
- `ds4.c:20961` — presenza di `DS4_MTP_TIMING`;
- `ds4.c:21079-21080` — `decode2 exact` solo se batch-verify è assente;
- `ds4.c:21273`, `21303`, `21338`, `21379` — contatori micro
  `drafted`/`committed`;
- `ds4.c:21414`, `21476` — fallback micro e verifier sequenziale;
- `ds4_server.c:7476-7478` — gate greedy, draft maggiore di 1 e spec-disable
  assente.

## Gate implementati

Nel runner standalone:

- modalità/isolamento: righe `2-3`, `78`;
- hash binario/MTP: `96-98`, `252-261`;
- receipt Control ed engagement: `312`, `355`, `364`;
- overlay e argomenti MTP: `443-478`;
- `ValidateOnly` prima di qualsiasi `Start-Process`: `520-525`;
- richieste greedy/no-think: `697-700`, `739-742`;
- parsing acceptance/fallback: `914-965`;
- reconciliation ed engagement: `1276-1313`;
- correctness contro request/content hash Control: `1316-1334`;
- receipt Control e diagnostic: `1387-1460`;
- metriche route/H2D per token accettato e shutdown: `1548-1574`.

Il profilo diagnostic passa soltanto con:

- un unico log di caricamento MTP con `draft=2`;
- almeno un ciclo `mtp timing micro`;
- drafted totale maggiore di zero;
- zero `decode2`, margin-skip, sequential e fallback;
- riconciliazione tra token prodotti, draft accettati e cicli loggati. Sono
  ammessi da zero a due cicli terminali senza log, al massimo uno per request,
  quando resta un solo token o viene emesso EOS.

Un diagnostic PASS genera `a4_batch2_engagement.json`. Un Batch2 performance
non può partire senza quel receipt e il receipt Control. Gli hash e tutta la
fondazione devono coincidere. Il logging per-ciclo resta quindi fuori
dall'A/B performance senza perdere il gate di engagement.

Correctness richiede, per entrambi i turni, stessi request hash, completion
tokens, finish reason e SHA-256 del contenuto rispetto al Control. Il gate
complessivo conserva anche lifecycle a due turni, prefix reuse, route/H2D,
assenza di quarantine/decode refusal e shutdown HTTP graceful verificato.

## Validazioni eseguite senza server

`test_a4_mtp_runner_static.ps1`:

```text
A4_STATIC_PASS parser=PASS manifest_count=47 modes=Control|Batch2 greedy_requests=2 think_false_requests=2 fixed_seed_requests=2 validate_before_start=PASS
```

Tre `ValidateOnly` finali hanno dato PASS:

| Profilo | draft | batch verify | timing/spec | `server_started` | processi DS4 prima/dopo |
|---|---:|---:|---|---|---|
| Control | 1 | 0 | unset/unset | no | 0 / 0 |
| Batch2 performance | 2 | 1 | unset/unset | no | 0 / 0 |
| Batch2 diagnostic | 2 | 1 | 1/1 | no | 0 / 0 |

Tutti hanno ricalcolato con esito positivo hash binario/MTP, manifest e
matrice. Nessun preflight GPU, monitor o `Start-Process` è stato raggiunto.

## Sequenza da usare solo dopo assegnazione dello slot

1. Eseguire Control e conservare il suo `a4_control_reference.json`.
2. Eseguire Batch2 con `-DiagnosticMtp -ReferenceReceiptPath <control>`.
   Conservare `a4_batch2_engagement.json`; non usare questo run per il delta.
3. Eseguire l'ordine performance C–T–T–C. Nei due T passare sia
   `-ReferenceReceiptPath` sia `-EngagementReceiptPath`. Usare sempre gli
   stessi flag di fondazione.
4. Passare i quattro `result.txt` a `summarize_a4_mtp_ab.ps1`.

Il riassuntore richiede almeno due osservazioni per braccio, verifica identità,
correctness, engagement, lifecycle e shutdown, quindi calcola mediane,
dispersione intra-braccio, route/H2D per token accettato e delta. Per regola
A4, `abs(delta) < 10%` viene classificato `variance` /
`NO_CONFIRMED_GAIN`; dispersione maggiore del 10% produce
`INCONCLUSIVE_UNSTABLE`. Non viene attribuito alcun guadagno teorico.

## Istruzioni di merge

La patch `a4_runner_vs_a1_snapshot.patch` è un diff di confronto contro uno
snapshot del runner A1 corrente con SHA-256
`358EF45ABDBDC398182441C66658F33FA4BAFF1E735B69192536B64378C15ACD`.
La patch ha SHA-256
`A25AF50C09EE4EA0BE017121396B8D8BBD24574220C6A50072F62F43694E6DE1`.

Non applicarla wholesale: il runner standalone parte da una baseline diversa
e il diff mostra anche divergenze A1, inclusa `GraphTensorDevice`. Eseguire un
merge manuale conservando le modifiche A1:

1. portare parametri `MtpMode`, `DiagnosticMtp`, path/receipt e `ValidateOnly`;
2. portare i gate identità e i receipt Control/engagement;
3. portare overlay, env e argomenti `--mtp`/`--mtp-draft`;
4. fissare le due request a temp 0, think false e seed 12345;
5. portare parser, gate, metriche normalizzate e receipt finali;
6. mantenere intatti i nuovi contatori/gate A1 e aggiornare soltanto dopo
   audit gli hash di un eventuale nuovo binario;
7. rieseguire test statico e i tre `ValidateOnly` prima di chiedere lo slot.

Il file standalone è l'artefatto autorevole per A4; la patch serve a revisione
e merge selettivo, non a sostituzione automatica.
