# DS4 leve preparate ma non eseguite — 2026-07-26

Questi artefatti sono stati prodotti dalle task parallele e vengono
committati per non perdere il lavoro. Non rappresentano risultati runtime.

## A2 — G73 branch A/B

- Harness standalone e report presenti in `A2_G73_HARNESS/`.
- Stato alla chiusura: `HARNESS_READY=yes`, `READY_FOR_RUN_SLOT=no`.
- Motivo: va rifatta la freeze/provenance sulla branch sorgente scelta da
  Claude; nessun benchmark fisico valido e' stato eseguito.

## A3 — prefill chunk

- Report read-only: `A3_PREFILL_CHUNK_REPORT.md`.
- Harness completo: `A3_CHUNK_HARNESS.zip`, con checksum affiancato.
- Candidati predisposti: chunk 250/600/768 e G73 pending/0/1.
- Nessun risultato prestazionale: solo ValidateOnly, test statici e sintetici.

## A4 — MTP batch verify

- Harness, comparatore, static test e receipt ValidateOnly in
  `A4_MTP_HARNESS/`.
- Confronto progettato: draft1 control contro draft2 +
  `DS4_MTP_BATCH_VERIFY=1`, greedy/no-think.
- Nessun run DS4 e nessun dato di acceptance reale.

## P0/P0.1 — SSD rotator

- Diagnosi, patch core, patch service-deadline e receipt statico presenti.
- Queste modifiche sono state incorporate nella linea sorgente consolidata
  `codex/ds4-k1-handoff-20260726`; i file restano come patch portabili.

## P4 — demand/promotion pipeline

- Architettura salvata in `P4_DEMAND_PROMOTION_ARCHITECTURE.md`.
- Nessuna implementazione o benchmark.
- Punti principali: stream demand separato dalle promozioni, eventi
  consumer-done, generazioni fail-closed e buffering multiplo.

## P5 — long-KV residency

- Studio salvato in `P5_LONG_KV_RESIDENCY_REPORT.md`.
- Nessuna implementazione o benchmark live150k.
- Resta distinto da K1: P5 tratta residency selettiva e watermark; K1
  trattava il lifecycle staged tra turni ed e' fallito al gate exact.

## Regola per la ripresa

Prima di eseguire uno di questi harness:

1. scegliere una singola branch sorgente pulita;
2. aggiornare hash di sorgenti, binario, modello e manifest;
3. ripetere ValidateOnly strict;
4. mantenere un solo run slot;
5. non interpretare capacita' ctx 150k come KV realmente viva 150k;
6. registrare ogni test nel ledger con tutte le variabili.

