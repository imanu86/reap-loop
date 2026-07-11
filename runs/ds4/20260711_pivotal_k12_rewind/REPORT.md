# PIVOTAL K12+rewind — REPORT (predizione vs misura)

**Data:** 2026-07-11 (~01:00–05:20 UTC), pod D `7qgalm9sasqnr7` (RTX 3090 community,
RAM-hot), binario post-**0022v2** (ds4.c `a88f9dcb`). Cyberpunk 199 B, W50 two-phase
weighted freeze-safe → **mask K12/layer STATIC** (file, 9760 pair), fase2 4000, ctx8192,
greedy temp 0. **Cache 1024** su tutti i bracci (deviazione dichiarata: a cache 256 questo
pod decodifica K12 a ~0,5 t/s ⇒ batch ~18 h/$7, oltre cap; l'hazard è mask-width-driven,
i rapporti in-batch a pari cache trasferiscono, i t/s assoluti NO — pod ≠ 3060).

## VERDETTO SECCO

**Il rewind NON converte il collasso-certo di K12-wide. 0/6 salvataggi (0/7 col probe).
La predizione del decision model (K12+rewind = 3,99 gtps, +156 % vs static) è REFUTATA
in questo regime: misurato ×1,29 (default) / ×1,0 (aggressive), e NESSUN guadagno di
qualità: L0 ovunque, `</html>` 0/8, lock `sfoko` identico al controllo.** Il conteggio
storico "masked run che chiudono a 4000" passa da 0/15 a **0/23**.

| braccio | n | L per-seed | `</html>` | fire/run | useful_frac | gtps (pod) | ratio vs static |
|---|---|---|---|---|---|---|---|
| 1 K12 static puro | 2 | 0,0 | 0/2 | — | 0,04 | 0,89 / 0,94 | 1,00 |
| 2 +rewind default E-DET | 3 | 0,0,0 | 0/3 | 1 | 0,04–0,05 | 1,18 ×3 | **×1,29** |
| 3 +rewind aggressive-CUSUM | 3 | 0,0,0 | 0/3 | 1 | 0,04 | 0,91 ×3 | ×1,00 |
| 2b widen-probe (fuori protocollo) | 1 | 0 | 0/1 | 1 | 0,05 | 0,69 | ×0,75 |
| *predizione modello* | — | *L2-ish* | *sì* | — | — | *3,99 vs 1,56* | *×2,56* |

(gtps = useful_frac × p2_gen_tps; useful_frac da onset del lock, periodicità ancorata
alla coda, `harness/pivotal_metrics.py`. Il ×1,29 dell'arm 2 NON è rescue: stessa
useful_frac, solo lock leggermente più tardo nello stream grezzo + rumore t/s.)

## Timeline del collasso (identica in tutti i run, determinismo statico)

Fase 2 parte a pos 85 (prompt+prefisso congelato). Erosione CSS da char ~401
(`gposcate`, `finto sfondo`, `sfchiesto` ≈ pos 190–215), **lock `sfoko` da char 493 ≈
pos ~215**, poi ~11 600 char di lock fino a 4000 token. Controllo (arm 1): onset 491,
useful 4 %, 2 seed byte-identici.

## Perché il rewind fallisce QUI — catena meccanicistica (il vero yield)

1. **Il detector E-DET FUNZIONA e spara sul collasso reale coi profili default** — prima
   validazione live: ARM pos 249, FIRE pos 260 (`s1_cusum_fire`), 8/8 run rc=0, attuatore
   meccanicamente perfetto (restore+rewind+resume, K pinnato o widen, zero crash).
2. **Ma S1 non dà lead in questo regime**: sale solo AL lock, non durante l'erosione
   (scope-binding già dichiarato in S1_REWIND_DESIGN: regime aggressivo = niente lead).
3. **CALWIN=128 si sovrappone al collasso** (rischio pre-registrato, materializzato):
   la calibrazione σ copre i decode-token 1–128 = pos 85–213, che CONTENGONO l'intera
   rampa di erosione → σ gonfiata, ARM ritardato oltre il lock.
4. **Il checkpoint "sano" congelato è DENTRO il lock**: ultimo snapshot rolling quieto a
   pos 246 = char 609, il lock parte a char 493. Il rewind teleporta da pos 260 a pos 247:
   da dentro-il-lock a dentro-il-lock.
5. **Temp-0 retrace**: il testo rigenerato è **byte-identico** allo span ritratto (45
   char in arm 2, 6 in arm 3, replay `tokens.csv`) — con K pinnato a 12 il rewind è un
   no-op per costruzione (logits d'onset ripristinati + stesso contesto + stessa mask ⇒
   stesso argmax). Le due leve di divergenza della 0022 (widen a keep_max; re-seed a
   temp>0) erano entrambe rimosse dal design del braccio (K12 pinnato, greedy).
6. **Widen-probe (2b)**: ripristinata la leva (fire ⇒ keep 12→96), il run RICADE comunque
   nello stesso lock — conferma **C1: il contesto avvelenato vince** (l'erosione char
   400–609 resta nel KV; il punto di restore è post-lock). E il t/s crolla a 13,1 (mask
   96-wide). Non è (solo) la leva mancante: è l'ancora marcia.
7. **Saturazione CUSUM**: dentro un lock stazionario la baseline laggata raggiunge l'EWMA
   ⇒ z→0 ⇒ il CUSUM non ricrossa MAI FIRE ⇒ 1 solo fire/run ovunque (MAX=2/6 irrilevante,
   BACKOFF mai esercitato). Il detector rileva TRANSIZIONI, non stati.
8. **L'airbag n-gram della 0022 è dead code in OGNI config** (finding pre-run, source-
   verified): `ds4_pace_tick` gira dentro l'eval PRIMA dell'attuatore e consuma lo stesso
   segnale (breath reattivo → phase BREATH blocca il rewind + de-arm; hysteresis mai
   re-armata in loop persistente). Il catcher designato per i regimi senza lead S1 non
   può sparare. Fix per il patch-owner: soglia drift propria dell'attuatore o hoist del
   check airbag prima del ramo breath del tick.

## Implicazioni per la strategia (cosa comprare col prossimo dollaro)

- **La colonna "con airbag rewind" del decision model va sospesa per i regimi wide/fast**:
  il suo presupposto (il rewind cattura il collasso largo) è falso così com'è. K48-static
  resta il fallback del modello per la larghezza wide.
- Per rendere il rewind utile in fast-collapse servono, in ordine di probabilità di ROI:
  (a) **trigger airbag-class funzionante** (n-gram, già misurato affidabile) — richiede il
  fix del punto 8; (b) **ancora di checkpoint garantita-sana** (es. snapshot pinnato a
  fine-prefill/pre-calibrazione + rolling con staleness-guard, invece del solo rolling
  EVERY); (c) **leva di divergenza reale al resume** (substitution/rotate della mask,
  temp-bump temporaneo, o resampling con ban degli n-gram del lock — il widen da solo NON
  basta, punto 6); (d) CALWIN fuori dalla zona a rischio (calibrare sul prefill, non sui
  primi decode-token).
- **Lo scope slow-erosion (K91-family) resta NON testato da questo esperimento** — per
  design: è l'unico regime dove S1 ha lead misurato (CLAIM-011) e dove la scala
  prevention→correction può ancora pagare come progettata.
- Nota di misura per il modello: `CORR_REWIND_TOK` reale in questo regime = 15 (default)
  / 2 (aggressive) token rigenerati — non i 56 assunti — ma il parametro è moot finché la
  conversione è 0.

## Costo / pod

Spesa attribuibile al mandato pivotale: pod D $0,22/h × ~2,8 h ≈ **$0,61** (cap
addizionale $3 rispettato). NB: il balance account (16,72→7,71 nel periodo) è dominato da
**altri 6 pod RUNNING di altri track** (~$1,85/h aggregati, verificato via `myself{pods}`
— non toccati, regola 5). Pod D lasciato **RUNNING**.

## File

`arm1_static/`, `arm2_rewind_default/`, `arm3_rewind_aggr/`, `arm2b_widen_probe/`
(per-run: route/tw/frozen/sess/p2prompt/trest/deliverable(.retracted)/p1-p2.diag +
pace_events.jsonl/tokens.csv nei bracci rewind; summary.csv + pivotal_metrics.json per
braccio), `harness/` (wrapper + arm scripts + analyzer). Retraction-aware: i deliverable
"client-trimmed" sono `deliverable_retracted.html` (replay del sidecar 0028).
