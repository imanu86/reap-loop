# Replica pubblica del REAP domain branch — `prodotto_json`

**Task #16.** Dimostra che il finding **REAP-K50 near-lossless su dominio** NON dipende
dalla a domain-specific corpus, replicandolo su un **corpus pubblico/sintetico** generato da
`scripts/reap_public_replica_corpus.py` (100% regenerabile byte-identico, nessun dato
proprietario). Pod: RTX A6000 sm_86 (RunPod community); ds4 `80ebbc3` + patch 0001-0007;
modello `DeepSeek-V4-Flash-IQ2XXS…imatrix.gguf` (86.7 GB, HF `antirez/deepseek-v4-gguf`).

Corpus scelto: **`prodotto_json`** — estrazione di schede prodotto e-commerce italiane
sintetiche → JSON con campi fissi. È l'analogo pubblico più fedele del task di estrazione
privato (stesso *shape*: testo IT breve semi-strutturato → JSON con campi fissi). Un secondo
candidato `ricetta_json` è committato come fallback (non è servito: `prodotto_json` ha
centrato l'equivalenza al primo giro).

---

## 1. Equivalenza di routing — la tabella (novità metodologica)

Il surrogato è un proxy legittimo del dominio solo se il router di DS4-Flash lo "vede
stretto quanto" the private-domain reference. Firme misurate con `scripts/reap_neff.py` (validato riproducendo
esattamente i numeri the private-domain reference dalla trace committata
`runs/reap/2026-07-05_trace_dominio/trace_dominio.tgz`):

- `n_eff@64` = exp(H) della distribuzione d'uso degli expert per (file, layer) sui primi
  64 token di decode = numero **efficace** di expert distinti (quanto è concentrato il routing).
- `reuse` = recall top-6 prev-token same-layer = frazione di expert riusati dal token prima.

| firma di routing | private-domain reference | surrogato `prodotto_json` (pubblico) | entro criterio? |
|---|---|---|---|
| n_eff@64 (count) | 46.4 | **41.84** | sì — [35, 50] |
| n_eff@64 (gate-weight) | 39.6 | **35.21** | sì — vicino |
| reuse prev-token same-layer | 0.395 | **0.407** | sì — > 0.30 |

**Verdetto equivalenza: VALIDO** (criterio pre-registrato `n_eff64_count ∈ [35,50] AND
reuse > 0.30`). Fonte: `prodotto_json/neff.json`. 20 prompt pubblici, 3899 token decode,
0 righe con pesi NaN, somma pesi/riga ≈ 1.5 (`expert_weights_scale` Flash).

---

## 2. Eval ppl bias-mask (K50) + controllo random

Corpus di eval **disgiunto** dai prompt di trace (item generati dopo dallo stesso stream
seed → anti-overfit della mask sul suo set di calibrazione). ppl bias-mask (design
equivalente al pruning fisico: il bias `exp_probs_b` entra solo nella selezione top-6, mai
nei pesi). Rapporto ppl appaiato per-chunk vs `full`, geomean + **bootstrap CI95
(B=10000, seed 12345)**. 10 dom_chunk appaiati. Fonte: `prodotto_json/eval/results_raw.csv`,
`prodotto_json/eval/replica_ci.json`.

| config | ppl aggregata | geomean vs full | CI95 | attraversa 1.0? |
|---|---|---|---|---|
| full | 2.175 | — | — | — |
| **reap_k50** (saliency g-only) | 2.289 | **1.052×** | **[1.039, 1.066]** | **no** |
| **rand50_s0** (controllo, pari K) | 2.604 | **1.197×** | [1.181, 1.211] | no |

**V0 mechanism-check:** `checked=11280 violations=0 V0_OK` — 0 expert potati mai
selezionati durante il decode = il bias-mask ≡ pruning fisico, **identico a the private-domain reference**.

Scelta di scope: **1 seed random** (s0) invece di 3, per il costo/tempo del regime
SSD-streaming (modello 86.7 GB > 48 GB VRAM dell'A6000, mlock bloccato dal container come
da playbook §7.9.1 → ~7 min/chunk). La robustezza multi-seed del random è già stabilita
sul private-domain branch (`runs/reap/2026-07-05_eval_biasmask_v2`, 3 seed). Il seed 0 riproduce
bit-exact il `random_control` embedded nella mask.

---

## 3. Verdetto — REPLICA PARZIALE

| proprietà del finding the private-domain reference | the private-domain reference | surrogato pubblico | replica? |
|---|---|---|---|
| routing "stretto" (n_eff, reuse) | n_eff 46/40, reuse 0.39 | n_eff 42/35, reuse 0.41 | **sì** |
| bias-mask ≡ pruning (V0) | 0 violazioni | 0 violazioni | **sì** |
| selezione-conta (reap ≪ random) | reap 1.010× vs rand 1.388× | reap 1.052× vs rand 1.197× | **sì** |
| K50 **strettamente lossless** (CI attraversa 1.0) | sì (CI [0.996, 1.025]) | **no** (CI [1.039, 1.066]) | **no** |

**REPLICA PARZIALE.** Tre proprietà su quattro replicano identiche su dati 100% pubblici:
l'equivalenza di routing, il meccanismo bias-mask≡pruning (V0_OK), e **il risultato
centrale "la saliency batte il random a pari K"** (reap 1.052× ≪ rand 1.197×, entrambi i CI
disgiunti e ordinati come on the private-domain reference). La differenza: sul surrogato reap_k50 è **near-lossless
(+5.2%, CI stretto sopra 1.0)** invece che *strettamente* lossless come on the private-domain reference (+1.0%,
CI che attraversa 1.0). Il finding replica quindi in **segno, meccanismo e ordine di
grandezza** senza la a domain-specific corpus; la soglia esatta "lossless" resta dominio-specifica
(plausibile: il JSON prodotto è ancora più regolare/ridondante della the private extraction task, quindi
gli expert potati pesano un filo di più). Nessuna dipendenza on the private data per la
conclusione operativa: *~50% degli expert è potabile per-dominio via saliency con costo ppl
minimo e nettamente inferiore al pruning casuale*.

---

## 4. Pod / costo

<!-- redacted: internal cost/infra note -->
  (3090 sm_86 — scelta playbook — irraggiungibile: unica macchina disponibile con sshd
  rotto, riassegnata a ripetizione; fallback A6000 = **stesso sm_86**, binari confrontabili
  col run the private-domain reference su 3090 sm_86.)
- Immagine `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; ds4 `80ebbc3` + patch
  0001-0007; **BUILD_EXIT=0, REGRESSION_EXIT=0** ("cuda long-context regression: OK").
<!-- redacted: internal cost/infra note -->
  (`myself.pods` non contiene più `91xbin1wey8el3`). Budget autorizzato $4-5: ampiamente dentro.

**Trasferibilità (dichiarata):** routing / saliency / ppl sono proprietà del **modello** →
trasferiscono dal pod A6000 al 3060. I t/s wall-clock **non** trasferiscono (regime
RAM/SSD diverso) e **non** sono riportati.

---

## 5. File (tutti committati, numeri solo da qui)

- `prodotto_json/neff.json` — n_eff surrogato + verdetto equivalenza
- `prodotto_json/reap_mask_prodotto_json_k50.json` — mask saliency g-only K50 + random control
- `prodotto_json/eval/results_raw.csv` — ppl per config/chunk (30 righe, rc=0)
- `prodotto_json/eval/replica_ci.json` — geomean-ratio + bootstrap CI95
- `prodotto_json/eval/eval_direct.log` — run log + V0 mechanism-check (V0_OK)
- `prodotto_json/trace/trace_prod.tgz` — 20 trace_p*.csv (`pos,layer,n,e0..5,w0..5`) + build/regression log
- `prodotto_json/manifest.json`, `prodotto_json/prompts/`, `prodotto_json/corpus/` — corpus pubblico (sha256, regenerabile)
- `prodotto_json/meta.json` — pod, metodo, costo, risultati
- `ricetta_json/` — candidato fallback (non usato)

Generatori: `scripts/reap_public_replica_corpus.py` (corpus), `scripts/reap_neff.py`
(equivalenza), `scripts/reap_saliency_ds4.py` (mask), `scripts/reap_bias_mask_ds4.py`
(eval), `scripts/reap_public_replica_ci.py` (CI).
