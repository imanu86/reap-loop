# FEED AUDIT — 0026 admit-CUSUM provenance (PRE vs POST bias)

**Domanda unica:** il CUSUM della patch 0026 (demand-admission) è alimentato da
probabilità router PRE-bias (raw, tutti i 256 expert) o POST-bias (schiacciate a
~0 sui pruned dal −1e9)?

## VERDETTO (secco): **PRE-BIAS — feed identico a S1. NON è un bug.**

Il CUSUM di admit legge lo **stesso identico** array `probs[]` del sensore
S1/0012 e della rmass 0020: un'**unica** `ds4_gpu_tensor_read(g->router_probs)`
che fan-out ai tre consumer. Le `router_probs` sono UNBIASED: il router calcola
la probabilità di tutti i 256 expert anche sotto mask; il −1e9 non le tocca (vive
in un tensore diverso e viene sommato solo a una copia usata per la top-k).

Metodo: base WSL `/root/ds4/ds4.c` (md5 `771a39a861e9512fed2fc4528780e080`, =
header 0026) copiata SOLA-LETTURA in git temporaneo + applicati 0020→0021→0026
puliti. Righe sotto = snapshot ricostruito.

## Evidenza (righe, snapshot 771a39a + 0020+0021+0026)

**1. Un solo readback, tre consumer (nessuna divergenza di feed).**
`metal_graph_spex_note_selected`, hook site:
- `16770  ds4_gpu_tensor_read(g->router_probs, 0, probs, N_EXPERT*sizeof(float))`
- `16772  ds4_pace_note_router_probs(il, probs);`   ← 0020 rmass (eviction EWMA)
- `16773  ds4_pace_s1_note_probs(il, probs);`       ← **sensore S1/0012**
- `16774  ds4_pace_admit_note_probs(il, probs);`    ← **CUSUM admit 0026**

Lo stesso buffer `probs[]` va ai tre. L'admit **non può** vedere un segnale
diverso da S1: è byte-identico. `ds4_pace_admit_note_probs` (def `13333`) somma
`probs[e]/tot` sui soli `g_reap_mask_pruned[il][e]` — sulla stessa lettura.

**2. Il sensore S1/0012 legge la stessa cosa ed è verificato PRE-bias.**
Blocco `DS4_REAP_SENSOR_LOG`, righe `16706–16716`:
`16709  ds4_gpu_tensor_read(g->router_probs, ...)` → somma `pruned_mass` sui pruned.
Commento 0012 (testuale): *"il router calcola le prob di TUTTI i 256 anche
mascherato: **il bias agisce solo sulla selezione**"*. Se `router_probs` fosse
POST-bias, `pruned_mass` sarebbe ~0 sempre e S1 misurerebbe rumore: non è così
(Track G, "unbiased router probabilities").

**3. Dove NASCE `router_probs`: senza bias.**
CPU router path `metal_graph_decode_cpu_router` (`17284+`):
- `17312  probs[i] = sqrtf(softplus_stable(logits[i]));`  ← solo dai logit, NO bias
- `17329  ds4_gpu_tensor_write(g->router_probs, 0, probs, ...)`  ← scrive quel probs
GPU path (`18791`): `ds4_gpu_router_select_tensor(..., g->router_probs, ...,
ffn_exp_probs_b_offset, ...)` — il bias entra come argomento di **selezione**,
`router_probs` resta il vettore di gating unbiased. Coerente col commento a
`7384–7385`: *"choose the six experts by biased top-k, but weight them using the
**unbiased router probabilities**"*.

**4. Il −1e9 è su un ALTRO tensore, e solo per la top-k.**
- `7548  g_reap_bias_masked[slot][e] = orig[e] + (pruned? −1.0e9f : 0.0f)`
  → il −1e9 sta in `g_reap_bias_masked`, copia di `ffn_exp_probs_b` (`exp_probs_b.bias`),
  tensore **distinto** da `router_probs`.
- `layer_topk_selected_experts_from_probs` (def `7618`): riga `7626`
  `memcpy(selection, probs, ...)` poi `7627–7631` somma il reap-bias(−1e9) a
  `selection` (copia temporanea) → `topk_desc(selection,...)`; i pesi restano
  `expert_weight[i] = probs[selected[i]]` (probs UNBIASED, riga `7640`).
  Il −1e9 **non modifica mai** `probs`/`router_probs`.

## Confronto col segnale S1
**Nessuna differenza.** Stesso tensore (`g->router_probs`), stessa `ds4_gpu_tensor_read`
(riga `16770`), stesso array `probs[]` (`16772–16774`). L'unica differenza è cosa
ne fanno: S1 logga `pruned_mass/total_mass`; admit accumula CUSUM `S += probs[e]/tot
− k_drift` sui pruned. Feed provenance: **identica**.

## Correzione proposta
**Nessuna.** Il feed è già PRE-bias (corretto, coerente con CLAIM-001 / S1). Non
serve una 0026b: non esiste bug di alimentazione da fixare.

## Implicazione per l'A/B live
**Verdetto A/B VALIDO. Tasso basso = effetto-TRAIETTORIA (fondamentale), non
artefatto di feed.**

L'ipotesi del REPORT — *"sotto mask attiva il bias sopprime la probabilità router
dei pruned ⇒ segnale più debole"* — è **meccanicisticamente errata**: il −1e9 non
tocca `router_probs` (agisce su `g_reap_bias_masked`→selezione), quindi non
"schiaccia" il segnale del CUSUM. Da correggere quella frase nel REPORT.

La causa reale del gap sim(~130/100tok) vs live(~1–7/100tok) è l'altra metà già
citata dal REPORT: la **sim gira su trace UNMASKED** (traiettoria generata SENZA
mask ⇒ hidden states diversi ⇒ il router domanda genuinamente di più i pruned),
mentre il live gira sulla **traiettoria MASKED** dove il router (sempre unbiased
sui valori) domanda genuinamente meno i pruned. È traiettoria/ricalibrazione
(REPORT punto iii: "ricalibrare la soglia sulla domanda MASKED"), **non** un bug
di feed. Quindi l'A/B non va rifatto per motivi di alimentazione: il verdetto
qualità-negativo di config C regge.
