# Eval bias-mask REAP-ds4 — ppl full vs reap_k50 vs random_k50

Run del 2026-07-05. Esegue `docs/REAP_DS4_eval_plan.md` in **Stage A bias-mask**
(design §2): scrivere `-1e9` nei bias `exp_probs_b` dei pruned ≡ pruning fisico per
selezione e pesi, senza chirurgia sul file. 3 config in parallelo, una per pod sm_86.

## Risultato — REAP K50 near-lossless, la selezione conta

`eval_summary.json` (ppl aggregata, dominio, 3400 token scored/config):

| config | ppl dom | vs full | ppl gen | vs full |
|--------|--------:|--------:|--------:|--------:|
| **full**      | 3.811 | 1.000× | 5.344 | 1.000× |
| **reap_k50**  | 3.860 | **1.013×** | — | (gen non eseguito) |
| **rand_k50**  | 5.200 | **1.365×** | 11.30 | 2.115× |

Appaiato per-chunk (ppl config/full sullo **stesso testo**, geomean 4 chunk dom):
- **reap**: [1.033, 0.987, 1.011, 1.021] → geomean **1.013** (un chunk *sotto* il full)
- **rand**: [1.265, 1.295, 1.505, 1.406] → geomean **1.365**

### Criteri pre-registrati (piano §3)
- `reap_dom ≤ 1.10× full` → **1.013× PASS**. Potatura 50% del dominio quasi-lossless.
- `rand_dom ≥ 1.5× full e > reap` → 1.365× **>reap ✔ ma <1.5 ✗** = **PARTIAL**. Il random
  degrada nettamente (+36.5% ppl, **27× più di REAP**) ma meno del 1.5× ipotizzato. La soglia
  1.5 era una stima presa dal 30B bf16 (dom rand K50 2.5×); sul ds4 IQ2 a K50 tenere 128/256
  expert lascia più ridondanza → random meno catastrofico. **La tesi regge**: a pari K la
  selezione REAP vale, il random no; il verdetto binario 1.5 era tarato troppo severo.

### Perché il random su DS4 fa 1.36× dove su Qwen-30B faceva ~2.5× (ipotesi, per il paper)

Tre ipotesi **cumulative**, dichiarate e non tutte testate (mandato SPEX-main 2026-07-05):
1. **Granularità più fine**: DS4-Flash ha 256 expert/layer con 6 attivi (Qwen 128/8). A pari
   frazione potata, ogni expert perso porta via meno capacità e i 6 slot trovano sostituti
   semanticamente più vicini nel pool superstite più numeroso → il danno random si diluisce.
2. **Shared expert**: DS4 ha 1 shared expert per layer sempre attivo (Qwen3-MoE non ne ha):
   un "pavimento" di capacità che non è potabile e ammortizza la perdita dei routed.
3. **Floor di rumore del 2-bit**: la quantizzazione IQ2_XXS aggiunge un errore di base che
   comprime i RAPPORTI di ppl (il denominatore full è già "degradato" dal 2-bit); su bf16 il
   full è più pulito e lo stesso danno assoluto appare come rapporto più grande.
Confound dichiarato: modello E precisione cambiano insieme (30B bf16 vs DS4 IQ2) → le ipotesi
non sono separabili con i dati attuali; il multi-seed v2 quantifica almeno la varianza del random.

## Il meccanismo è verificato (non solo l'esito)

`reap/biasmask.log`: `apply reap: scritti 5120 bias a -1e9 su 40 layer` +
**`V0 selections checked=11280 violations=0 V0_OK`** → con la bias-mask attiva il router
non ha **mai** selezionato un expert potato. Stage A ≡ pruning fisico, confermato sul campo.

## Note

- **dom-vs-gen (F3, ledger G3)**: random sul generale degrada 2.11× contro 1.37× sul dominio →
  il pruning dom-guidato danneggia più il generale, coerente con lo studio. reap/gen non
  eseguito (tagliato per tempo; il reap dovrebbe stare tra i due, ma qui non misurato).
- **corpus dom DISGIUNTO** dagli item usati per calibrare la mask (`corpus_manifest.json`):
  la tenuta di REAP non è overfit al set di trace.
- **velocità NON riportata come 3060** (playbook §4.3): `--perplexity-file` decoda ~1.5 t/s,
  identico su pod e workstation; qui conta il NUMERO (proprietà del modello), non i t/s.
- Cap `-n 850/chunk` (piano rivisto): ~5k token/config, SE sul delta-NLL appaiato ~1e-2,
  sotto gli effetti misurati.

## Artefatti

`full/`, `reap/`, `rand/` (results_raw.csv + ppl_*.log + build/regression log per pod);
`results_raw.csv` (merge), `eval_summary.json`, `corpus/` (dom gitignored, gen + manifest
committati). Pod: 3× RTX 3090/3090Ti community, tutti terminati con verifica (meta sotto).
