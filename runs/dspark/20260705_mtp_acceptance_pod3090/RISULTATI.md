# Strada A (pod) — baseline MTP-1 + union-load: misure finali

**Data**: 2026-07-05 · **Pod**: RTX 3090 community (terminato+verificato) · **ds4**: stock `80ebbc3` + solo patch
`patches/ds4/0006-dspark-mtp-streaming-probe-unsafe.patch` (bypass guardia dietro env).
**Modelli**: 2-bit imatrix sha256 `efc7ed60…` ✓ (= oid HF) + MTP `afd481ee…` ✓.
**Regime**: `--cuda --ssd-streaming` (il path non-streaming con `--mtp` va in OOM su 24GB:
vuole residenza device; gira solo su unified memory tipo GB10 — vedi log v2).
**Combo probe** (scoperta chiave, `ds4_cli.c:922-947` + `:483`): `--mtp-draft 2` +
`DS4_MTP_SPEC_DISABLE=1` + `DS4_MTP_PROBE=1` → dispatch nel sampled loop, spec-dec OFF,
drafting ON, verifier mai toccato = misura pura dell'acceptance MTP-1.

## Acceptance top-1 del baseline MTP-1 (149 confronti/run, -n 150, greedy, ctx 2048)

| dominio | run 1 | run 2 | acceptance |
|---|---|---|---|
| code | 130/149 | 130/149 | **0.872** |
| math | 126/149 | 126/149 | **0.846** |
| chat | 90/149 | 90/149 | **0.604** |

Run doppi bit-identici → determinismo greedy confermato; l'acceptance è proprietà del
modello e **trasferisce al 3060** (stesso GGUF 2-bit, stesso motore). I t/s di questi run
(211-418s per 150 tok) sono POD-ONLY e non trasferiscono.

## Union-load nel path batch streaming: verificata E quantificata a runtime

Prefill batch da 670 token (43 layer routed), log `out_v3/unionload_compact_counts.log`:
- slots per layer = 670×6 = **4020** richieste expert
- compact (expert unici caricati) = **132-191, media 163.0** per layer
- riduzione: **95.9%** delle load evitate dalla dedup per blocco
- timing per layer in `out_v3/unionload_prefill.log` (righe `batch selected load`)

Conferma runtime del claim RECON §1.7: `ds4_gpu_stream_expert_cache_prepare_selected_batch`
(ds4_cuda.cu:3176) fa UNA compact-load per layer per blocco. Al blocco-8 del verify la
stessa meccanica vale il 49% misurato dalla trace (24.3 unici vs 48).

## Tabella A-vs-B finale (stessi 3 prompt, stesso modello target)

| dominio | MTP-1 top-1 (baseline, 2-bit ds4) | DSpark pos1 (fp8 H200) | DSpark τ blocco-5 |
|---|---|---|---|
| code | 0.872 | 0.979 | 5.18 |
| math | 0.846 | 0.959 | 5.00 |
| chat | 0.604 | 0.713 | 2.70 |

Ordinamento dominio identico (code>math>chat) su entrambe le misure = segnale coerente.
⚠️ precisioni target diverse (2-bit vs fp8/fp4): confronto di struttura, non al millesimo.

## Note operative
- `--ssd-streaming-cache-experts 12GB` è troppo per la 3090 dopo le altre riserve
  ("available 11.11 GiB <= reserve 11.78 GiB" → cache disabilitata): usare ≤8GB.
- `--simulate-used-memory` fallisce nei container RunPod (RLIMIT_MEMLOCK): per i conteggi
  union-load è irrilevante; per la pressione RAM reale serve il 3060 fisico.
- Nota costo: la maggior parte del costo del track era il pod 2×H200 della Strada B.
