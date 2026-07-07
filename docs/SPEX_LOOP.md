# SPEX — Loop di prefetch predittivo degli expert, fedele a DeepSeek DSpark

> **Scopo di questo documento:** rendere DUREVOLE e recuperabile-a-freddo tutto il lavoro sul "loop con
> temperatura" (il verificatore confidence-scheduled), verificato sul paper DSpark vero, così che una
> sessione nuova non debba ri-leggere ~1M di token di chat. Fonti primarie nel repo:
> `docs/references/DSpark_paper.txt` (paper DeepSeek), `docs/references/DwarfStar_ds4_README.md`,
> implementazione in `src/msc/spex/spex_loop.py`, baseline storica `src/msc/spex/continuous_cache_sim_ORIG.py`.
> Esperimenti misurati: vedi `docs/EXPERIMENTS_LEDGER.md`. Data: 2026-07-03.

---

## 0. L'obiettivo (parole dell'utente, verbatim dalla chat)

- **Nord**: far girare MoE GRANDI (235B DeepSeek-class) su HW ridicolo (RTX 3060 12 GB), **qualità piena**,
  **uso generale**, accettando di essere lenti — via offload predittivo degli expert.
- *"io voglio partire da due e man mano girarli di continuo col sistema predittivo per vedere se resta
  usabile occupando una frazione ridicola di vram"* → sweep della capacità **da C=2 expert residenti** in su.
- *"markov me l'hai messo in loop... per predizione continua e purging continuo?"* → predict + evict continui.
- *"nel loop chi verifica se stia facendo bene o male? lì sta il punto!! e su questo dovevamo usare un
  concetto di temperatura... quanto stai sbagliando a predire?"* → il **verificatore + STS**.
- *"come hai impostato il loop markov con validazione in stile paper di deepseek?"* + link al PDF DSpark
  → **il loop va fatto fedele a DSpark**.
- *"voglio citare tutti i paper... conoscevo solo dwarf e dspark... non ho letto nessuno dei due, solo video"*
  → citare tutti, framing onesto. **DSpark = `deepseek-ai/DeepSpec` è un'inferenza NOSTRA, da confermare
  prima di citare** (l'utente ha nominato solo "dwarf e dspark" da video + link `antirez/ds4`).

---

## 1. Il metodo DSpark (verificato riga-per-riga sul paper)

Paper: *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*, DeepSeek-AI.
File: `docs/references/DSpark_paper.txt`. Sezioni chiave verificate:

### 1.1 Confidence head — Eq. 7 (righe 287-289)
```
c_k = σ( wᵀ · [ h_k ; W₁[x_{k-1}] ] )
```
Scalare in (0,1) per posizione k. Proiezione **lineare** `w` sulla **concatenazione** dell'hidden di backbone
`h_k` e della **Markov-embedding** `W₁[x_{k-1}]` del token precedente, poi sigmoide. Modella la probabilità
**condizionale** che il draft in posizione k sopravviva alla verifica, dato che i precedenti 1..k-1 sono accettati.

Target (Eq. 8, righe 297-301), supervisione della head:
```
c*_k = 1 − ½ ‖ p^d_k − p^t_k ‖₁      (= 1 − TV(draft, target))
```
Loss confidence (Eq. 11): BCE pesata `w_k = exp(−(k−1)/γ)`; loss totale `0.1·L_ce + 0.9·L_tv + 1.0·L_conf`.

### 1.2 Sequential Temperature Scaling (STS) — §3.2.1 (righe 309-319)
- Ogni `c_i` è **condizionale** → per la chain rule la prob. di accettazione del prefisso fattorizza nel
  **prodotto cumulato** `A_k = ∏_{i≤k} c_i`.
- Su un **held-out**, STS calibra il prodotto cumulato **da sinistra a destra**.
- Per ogni posizione `k ∈ {1..γ}`: **grid-search 1D** della temperatura scalare che **minimizza l'ECE**
  (Naeini 2015) del **prodotto cumulato**, tenendo **congelate** le posizioni già calibrate.
- **Order-preserving**: `σ(z/T)` è monotona in z per T>0 → il RANKING dei candidati NON cambia; la temperatura
  raddrizza solo le probabilità verso i tassi empirici.
- Numeri di riferimento DSpark (Fig. 6, righe 741-772): ECE grezzo **3–8% → ~1%** dopo STS; ROC-AUC 0.81–0.90
  preservata (es. Pos1 5.7%→2.0% AUC 0.818; Pos5 5.8%→0.8% AUC 0.864; Pos7 3.3%→0.4% AUC 0.907).
- **Il paper NON dà**: range/risoluzione della griglia, numero di bin M, dimensione dello split, né scrive
  esplicitamente `σ(z/T)` (dice solo "temperature scalar" + "order-preserving"). Sono scelte NOSTRE da
  dichiarare e riportare.

### 1.3 Hardware-Aware Prefix Scheduler — Algorithm 1 (righe 322-348)
- Survival `a_{r,j} = ∏_{i≤j} c_{r,i}`.
- Spazio candidati `E = {(r,j) | a_{r,j} > 0}`, ordinato **decrescente per a**.
- Incumbent inizializzato al baseline **no-extension** `Θ_best = R·SPS(R)` PRIMA di ogni ammissione (righe 333-335)
  → un'ammissione deve battere il fallback "non prefetchare".
- Ammissione **greedy**: per ogni candidato in ordine, estendi, calcola `Θ = τ·SPS(B)` (τ = accept attesi,
  SPS = steps/sec profilati); se `Θ > Θ_best` tieni, **altrimenti break** (early-stop, unimodale, non-anticipante).
- **NON c'è cap di capacità/banda nello stop del paper** — lo stop è puro-Θ. Un cap è una **guardia NOSTRA (SPEX)**,
  applicata DOPO la regola del paper e da dichiarare come tale.

### 1.4 Semi-autoregressive drafter (§3.1)
- Markov head (Eq. 5): bias di transizione low-rank `B = W₁W₂`, **default r=256** (riga 247). RNN head (Eq. 6)
  per memoria oltre 1 step.

---

## 2. Mappa DSpark → SPEX (posizione k == layer MoE k, γ == L)

| DSpark (token) | SPEX (expert) |
|---|---|
| posizione k nel blocco di γ token | transizione layer k→k+1; blocco = gli L layer MoE di un forward (Qwen3-30B L=48, 235B L=94, OLMoE L=16) |
| anchor x₀ | seed = hidden h₀ al primo layer + expert routati al layer 0 |
| draft token x_k | i top-N expert che il predittore emette per il layer k+1 (mai il router — solo COSA precaricare) |
| c_k = P(token k sopravvive) | c_{k,e} = P(expert predetto e ∈ top-k REALE del router al layer k) |
| verifier = rejection sampling (load-bearing per correttezza) | **il router REALE che spara**. HIT = expert predetto+prefetchato è residente; MISS = fetch on-demand SSD→VRAM che costa SOLO latenza, mai accuratezza |
| SPS(B) steps/sec | modello di costo con condizione di vittoria `t_prefetch(expert) < t_compute(layer sovrapposti)` |

**Conseguenza chiave:** siccome il router reale decide sempre, l'apparato non-anticipante / Appendix-A di
lossless-ness di DSpark **NON serve** alla correttezza di SPEX; si tiene (o il suo analogo async §5.2) solo come
euristica di banda. **Invariante da assert-are nei test: routing SPEX == routing baseline per ogni token/layer.**

**Scelte di fedeltà dichiarate (SPEX, non del paper):**
- Orizzonte prefetch = **1 layer** (L→L+1). Con orizzonte 1 il "prodotto cumulato" di DSpark si riduce a
  temperatura **per-layer** sulla confidence **per-candidato** (standard temperature scaling per-posizione,
  comunque order-preserving + ECE-min + held-out). Il prodotto-cumulato token-sequenziale di DSpark vale per
  l'orizzonte **multi-layer** (estensione documentata, non ancora implementata).
- Confidence head features = **score Markov** (locale). Il **+hidden** (recall 0.93–0.99, §5) si innesta come
  feature aggiuntiva `[h_k ; MarkovEmb]` (Eq. 7 piena) dopo il dump dal pod, senza cambiare il loop.
- Label confidence = **HARD hit-indicator** (expert nel top-k reale del layer successivo, dalle traces).
  Alternativa soft (Eq. 8 TV analogo) = ablation.

---

## 3. Cosa c'era di SBAGLIATO (baseline storica `continuous_cache_sim_ORIG.py`)

Verificato riga-per-riga contro il paper. Questa è l'euristica grezza da NON riproporre come "DSpark":

| Gap (riga nel file ORIG) | Correzione (in `spex_loop.py`) |
|---|---|
| "temperatura" = EMA della frazione di k pre-caricati (riga 37, `conf=(1-α)conf+α·hit_frac`) — è un hit-rate corrente, NON la temperatura DSpark | STS vera: L temperature per-layer via grid-search 1D che minimizza l'ECE, congelando le precedenti; `σ(z/T)` all'inferenza |
| admission `n_pred = round(C·conf)` (riga 46) — split a conteggio fisso | ammissione stile Alg.1: candidati con confidence CALIBRATA > τ, ordinati desc, fino a cap (guardia SPEX) |
| nessuna confidence head | head `σ(wᵀ·feat)` (Eq. 7), BCE verso hit-label, peso `w_k=exp(−(k−1)/L)` |
| nessun prodotto-cumulato / prefix-survival | `a_k = ∏ c_i` lungo la catena layer (reset per doc) — orizzonte multi-layer |
| predittore SOLO Markov | +hidden come feature (dump pod) |
| **split per token** (`n_tok//2`, riga 61) → leakage intra-doc | **split per DOCUMENTO** (train/calib/test via `doclens`) |
| nessun reset a fine doc | stato Markov + cache resettati a ogni confine-doc |
| eviction LRU cieca | (opz.) prediction-aware; default LRU |
| una sola run | ≥3 seed, media±std |

---

## 4. Implementazione: `src/msc/spex/spex_loop.py`

Modulo numpy puro, gira in locale sulle traces `models\spex\*.npz`. Componenti:
- `load` / `split_docs` / `doc_tokens` — carica `experts[T,L,k]` + `doclens`, split 60/20/20 per DOC (seed).
- `build_markov` — conteggi di transizione per layer `C_l[e_cur,e_next]` su TRAIN (bincount vettoriale).
- `fit_confidence` — head per-layer: logistica 1-feature su `log1p(score_markov)` → P(hit), BCE/GD su (candidato,hit).
- `ece` — Expected Calibration Error (Naeini 2015), M=15 bin equi-larghi.
- `fit_sts` — temperatura per-layer via grid-search 1D (0.4..15) che minimizza l'ECE, order-preserving.
- `simulate(policy)` — il loop: cache per-layer capacità `cap`, reset per doc; il router reale (`D[t,l,:]`) è la verità.
  Policy: `random | reactive(LRU) | markov_naive | adaptive_crude | adaptive_raw | adaptive_dspark`.
- `run` — per seed: fit su train, STS su calib, simula su TEST; tabella miss-rate media±std + ECE raw→STS.

**Uso:** `python spex_loop.py traces_q30_domain.npz traces_q30_general.npz --caps 2,4,8,12,16,24,32 --seeds 0,1,2`

### La meccanica del "verificatore + temperatura" (il punto dell'utente)
La cache tiene sempre `C` slot. Ad ogni layer si ammettono predizioni con **confidenza calibrata > τ**:
- predizione **inaffidabile** (generale, `p` bassa) → si ammette POCO → la cache resta piena di residenti
  reattivi (LRU) → **mai peggio del reactive**;
- predizione **affidabile** (dominio, `p` alta) → si ammette TANTO → **batte il reactive**.
La **temperatura (STS) decide QUANTI** ammettere: è il verificatore che misura "quanto stai sbagliando".
Ablation `adaptive_raw` (stessa cosa senza STS): la head grezza è sovra-confidente → ammette troppo →
degrada verso `markov_naive` → **mostra il valore della calibrazione**.

### Nota di correttezza appresa costruendo (2026-07-03)
Poiché la STS è **order-preserving**, calibrare NON cambia *quali* expert prefetchi a cache piena — cambia
*quanti* ne ammetti via la soglia τ. Quindi, nel puro miss-rate a capacità C, il valore della temperatura
sta **tutto nel gate di ammissione** (banda), non nel ranking. Prima versione: la mia eviction
"prediction-aware" buttava i residenti reattivi (bug) → corretta a LRU; e la soglia va tarata perché la
calibrazione separi dominio (ammetti tanto) da generale (ammetti poco).

---

## 5. Dati disponibili (locali) vs cosa serve dal POD

### Traces di routing (LOCALI, sufficienti per il loop Markov + STS)
`models\spex\*.npz`, ciascuna `experts[T,L,k]` int16 (expert-ID) + `doclens` + `n_experts/topk/n_layers`:

| trace | token T | docs | E | k | L |
|---|---|---|---|---|---|
| traces_q30_domain | 74612 | 152 | 128 | 8 | 48 |
| traces_q30_general | 95841 | 200 | 128 | 8 | 48 |
| traces_q235_general | 55258 | 150 | 128 | 8 | 94 |
| traces_olmoe_general | 99152 | 300 | 64 | 8 | 16 |
| traces_olmoe_domain | 75095 | 152 | 64 | 8 | 16 |

**Manca `traces_q235_domain`** (il pod B200 fu terminato prima del dump). Rigeneratore: `scripts_pod/dump_traces.py`.

### +hidden nel loop → richiede un DUMP dal POD
Il predittore hidden (`scripts_pod/hidden_predict.py`) è una **probe lineare per-layer** `W·h_L+b` (BCE, verso
gli expert del layer L+1); recall **0.93–0.99** vs Markov 0.51–0.89. Ma salva **solo il recall aggregato**
(`models\hidden_predict_q30_{dom,gen}.json`), NON le predizioni per-token → il +hidden **non è runnabile in
locale**. Serve un dump: script pronto `scripts_pod/spex_dump_hidden.py` → produce `hidden_scores_<tag>.npz`
con `scores[T,NL-1,E]` (logit confidence per-token, ~0.9 GB) + `experts` + `doclens`. A100 ~$1, ~30-45 min.
Corpora: `domain_eval.jsonl` (held-out domain set), `general_it.jsonl` (wikipedia-it, ricostruibile
da un dumper di corpus generale).

---

## 6. Esperimenti sul loop già fatti (dalla chat, con `continuous_cache_sim_ORIG.py`)

Miss-rate per layer (più basso = meglio), **@25% residenza (C=32/128)**:

| trace | RANDOM | REACTIVE-LRU | ADAPTIVE (EMA) | Adaptive vs random |
|---|---|---|---|---|
| 30B general | 0.209 | 0.154 | **0.117** | −44% |
| 235B general | 0.197 | 0.142 | **0.123** | −38% |
| 30B domain | 0.268 | 0.203 | **0.132** | −51% |

Decomposizione (30B general): REACTIVE−RANDOM ≈ +0.055 (riuso temporale) ; ADAPTIVE−REACTIVE ≈ +0.037
(predizione+temperatura), fino a +0.15 su dominio. **Risultato negativo (parte della novelty):** `markov_naive`
PERDE contro reactive-LRU su general (routing concentrato → LRU cattura già il riuso); l'ADAPTIVE con
verificatore è ciò che lo rende **mai-peggio-del-reactive**. hidden recall @25% 0.986 ≈ Fate (arXiv 2502.12224)
= prior art, NON novel; il pezzo novel = **il loop verificatore-confidence-scheduled + il negativo markov<reactive**.

> ⚠️ Questi numeri sono dell'euristica EMA grezza. La versione DSpark-fedele (`spex_loop.py`, STS vera) è in
> validazione; l'obiettivo è riprodurre/battere questi con la macchina corretta + calibrazione onesta (ECE).

---

## 7. Decisioni aperte (per l'utente)

1. **Confermare DSpark == `deepseek-ai/DeepSpec`** prima di citarlo (inferenza nostra; l'utente ha dato solo
   "dwarf e dspark" da video + link `antirez/ds4`).
2. **Rank Markov**: r=256 (fedele DSpark) o r=64 (più leggero, da dichiarare come deviazione).
3. **Orizzonte prefetch**: 1 layer (semplice, matcha t_prefetch<t_compute) o multi-layer (prodotto cumulato pieno).
4. **+hidden**: approvare il dump dal pod (A100 ~$1) — senza, il loop resta Markov-only in locale.
5. **235B domain trace mancante**: ri-provisionare B200 (GPTQ-int4, costoso/lento) o restare su 235B-general.
6. **Variante async §5.2** (K a 2-layer-prima, no early-stop): implementarla o basta la greedy sincrona.

---

## 8. Prossimi passi (ordine)
1. Chiudere la validazione locale di `spex_loop.py` (Markov + STS) su q30 domain+general, 3 seed → tabella + ECE.
2. Dump +hidden dal pod (`spex_dump_hidden.py`) → innestare la feature hidden nel loop → misurare il lift.
3. Baselines B0/B1/B2 + latency-breakdown + Pareto latenza-vs-VRAM + heatmap SPEX-viable-zone (spec `SPEX_spec.md`).
4. Scrivere il paper SPEX (contributo: verificatore confidence-scheduled trasferito da DSpark su cache reattiva
   DwarfStar; delta cross-distribution; il negativo markov<reactive). Citare tutti (vedi `references.bib`).
