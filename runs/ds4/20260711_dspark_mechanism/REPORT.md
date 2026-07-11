# DSpark/MTP mechanism analysis — rollback reuse + acceptance-rate come segnale pre-garbage

**Data:** 2026-07-11 · read-only sui sorgenti (ds4 live /root/ds4, pin RECON 80ebbc3), offline sui run registrati · nessun pod acceso.
**Scopo:** validare l'intuizione operatore — lo speculative decoding DSpark/MTP di ds4 contiene gia (a) il rollback KV che riusiamo per il rewind, e (b) un early-warning (acceptance-rate) che crolla PRIMA del garbage.

Riferimenti file:riga = albero live patchato /root/ds4/ds4.c (post-canonical + 0027 + 0022v3); i simboli sono stabili anche sul pin 80ebbc3 del RECON (docs/dspark/RECON_MTP_DS4.md).

---

## A) ROLLBACK — il rollback-su-reject speculativo E cio che il rewind riusa. VERDETTO: SI (verbatim in 0027, twin in 0022), con parti riusabili non ancora sfruttate.

**Le primitive.** Il KV-rollback dello speculative sono:
- `spec_frontier_snapshot(f,s)` (ds4.c:26766) / `spec_frontier_restore(f,s)` (ds4.c:26799): copiano le **frontiere del compressore per-layer** (layer_attn_state_kv/score, layer_index_state_kv/score per i layer ratio-4, i contatori layer_n_comp/n_index_comp, e mtp_n_raw della raw-SWA-cache MTP) dentro/da buffer dedicati `spec_*`. E il rollback KV vero.
- `ds4_session_rewind(s,pos)` (live ds4.c:30560): e **solo** il troncamento logico CPU-side (checkpoint.len = pos, invalida mtp_draft). NON tocca lo stato GPU.
- `spec_frontier_commit_prefix1(s)` (ds4.c:26838): il **partial-accept** — riavvolge le frontiere a uno stato intermedio (post-draft[0]) senza restore+replay completo.

Rewind = `spec_frontier_restore` (GPU) **+** `ds4_session_rewind` (logico). Esattamente la coppia che il RECON prevedeva.

**Prova che e lo stesso meccanismo.** `patches/ds4/0027-rewind-exactness-harness.patch` (diagnostica R1) chiama spec_frontier_snapshot/spec_frontier_restore **verbatim** + ds4_session_rewind(p) e prova che un rewind a profondita **k=50-300 token** (ben oltre gli 1-2 token che lo spec-dec esercita) rigenera gli id **bit-identici**. Requisito: grafo costruito **MTP-ON** ("spec_* frontier buffers are MTP-gated"). Il rollback speculativo scala al nostro orizzonte di rewind. Confermato meccanicisticamente.

**Cosa 0022 fa oggi.** `0022-pace-s1-rewind.patch` (l'attuatore di PRODUZIONE) **non** riusa i buffer spec_*: ha clonato la logica in ds4_pace_rewind_snapshot_frontier/restore_frontier ("graph-level twins of spec_frontier_snapshot/restore") su un proprio **ring** spec_rewind_*[DEPTH][LAYER], gated su DS4_PACE_REWIND **indipendente da enable_mtp**. Ragione legittima: il loop greedy CLI alloca con enable_mtp=false, quindi non puo montare sui buffer del verifier.

**Parti riusabili per rendere 0022 piu robusta (non reinventare):**
1. **spec_frontier_commit_prefix1 / il partial-accept.** 0022 riavvolge solo a snapshot fissi (ring ogni "every" token). Il commit-prefix1 da un riavvolgimento **esatto per-N-token intermedio** gia scritto e testato: utile per un rewind fine-grana senza dipendere dalla spaziatura del ring.
2. **Il bookkeeping multi-token accept dello state-machine speculativo** (DS4_MTP_KEEP_ACCEPTED, ds4.c:30044; snapshot/restore attorno al prefisso accettato :30142/:30208/:30282; gestione mtp_n_raw delle righe speculative "invisible garbage"). 0022 dichiara esplicitamente **NON coperto: "MTP-on speculative multi-accept across a rewind"**. Ma il punto B (acceptance come segnale) **richiede MTP-ON** -> allora sia i buffer spec_* del verifier sia il ring spec_rewind_* sono vivi insieme, e comporre rewind + righe speculative in volo e esattamente il problema che lo state-machine speculativo gia risolve. Riusare il suo accept-bookkeeping e la strada per far girare 0022 **con MTP acceso** senza corrompere lo stato.

-> **A: SI, riusabile.** 0027 lo dimostra bit-exact. Per il salto a "0022 + MTP-on" (necessario a B) riusare commit_prefix1 + l'accept-bookkeeping speculativo invece di estendere il twin a mano.

---

## B) ACCEPTANCE COME SEGNALE PRE-GARBAGE — calcolabile per-token a costo ~0. Ma NESSUN dato allineato al collasso esiste -> PROPOSTA di run, non misura.

**Calcolabile per-token a costo ~0: SI, ed e gia strumentato.** DS4_MTP_PROBE (ds4.c:29894-29907): per OGNI token committato confronta il draft MTP del ciclo precedente col token reale e stampa "ds4: mtp probe token=.. draft=.. hit=H/T". E l'**acceptance top-1 di MTP-1 misurata sul flusso greedy reale**, funziona anche con --mtp-draft 1 (**zero commit speculativi** -> traiettoria invariata: e pura misura). Costo = **un forward del blocco MTP per token** (lo stesso che il drafter fa gia). DS4_MTP_CONF_LOG (ds4.c:30015, stampa :30271) aggiunge drafted/committed/margin per ciclo. **Nessuna nuova patch serve per il segnale grezzo.**

**Crolla prima del garbage? NON LO SAPPIAMO — nessun dato allineato.**
- Nel repo moe dspark/mtp-spec-dec l'unico dato di acceptance e runs/dspark/20260705_mtp_acceptance_pod3090 -> **modello PIENO, NESSUNA mask, nessun collasso** (baseline sano: code **0.872**, math **0.846**, chat **0.604**; 149 confronti/run, greedy, bit-identico x2). E la baseline "acceptance = proprieta del modello", **non** una traccia di deriva. runs/ds4/20260711_pregarbage_sensor dichiara che entropia/logit-margin/top-1 **non sono loggati in nessun run**; l'acceptance MTP e un **terzo** segnale, anch'esso mai loggato in regime mask.
- Quindi B = **proposta di run instrumentato**, non allineamento offline.

**Perche e il candidato migliore che S1 non aveva (l'argomento forte):**
1. Il draft e prodotto dal **modello MTP di supporto separato** (--mtp GGUF, proprio MoE 256-exp, RECON sez 1.2). La REAP-mask (g_reap_mask_pruned) agisce sul routing del **main** model, **non** sul blocco MTP. Acceptance = accordo tra **main-mascherato** e un predittore di riferimento quasi-non-mascherato -> **divergenza cross-model**, strutturalmente diversa da S1 (routing-mass intra-model) e dalla self-confidence. Quando la mask spinge il main off-distribution, l'acceptance dovrebbe crollare mentre il riferimento resta ancorato.
2. **Nessuna finestra di calibrazione da 128 token.** E un binario per-token: puo cadere al token 1. Aggira esattamente la cecita strutturale che ha ucciso S1 nel regime aggressivo (pregarbage_sensor: "128 > 42", il detector nasce dopo il garbage). L'acceptance non ha quell'orizzonte.

**Caveat onesti (da non nascondere):**
- La testa MTP consuma l'**hidden del main gia perturbato** dalla mask -> il riferimento non e perfettamente pulito; la divergenza potrebbe non anticipare abbastanza.
- Il baseline sano e **dominio-dipendente** (chat 0.60 vs code 0.87) -> una **soglia assoluta confonde dominio e collasso**. Il segnale valido e il **DROP relativo alla baseline sana del run stesso** (stessa lezione "livello assoluto non separa" del pregarbage_sensor), non il livello.
- **MTP-ON costa VRAM.** Gli statics del blocco MTP (~3.5GB device-resident) sono **il blocker gia quantificato** dalla serie moe 0011: sul 3060 (12GB reali) competono con la expert-cache. -> misurare su **pod RAM-backed** (l'acceptance trasferisce, e proprieta del modello; i t/s no) o sul 3060 con expert-cache ridotta.

-> **B: segnale gratis e gia strumentato, ma dato di collasso ASSENTE.** Serve il run T1.

---

## C) DETECTION DEGENERAZIONE — accept/reject E l'unico segnale; nessun hook qualita dedicato.

Il controller speculativo ha **un solo** aggancio "qualita" oltre all'accept/reject esatto (match argmax): il **margin gate** (margin = logit_top1 - logit_top2 dell'ultimo draft MTP, soglia default 3.0, override DS4_MTP_MIN_MARGIN; usato **solo** per draft_n==2 per decidere se saltare il verifier — log "mtp conf ... margin=" a ds4.c:30271). E una **proto-confidence sul DRAFT**, scalare, non calibrata, un solo cut-off, senza nozione di carico ne di qualita dell'output main. Rispetto a DSpark (RECON sez 1.8): **niente confidence head (Eq.7), niente STS, niente scheduler, nessun hook di degenerazione**. -> **l'accept/reject E il segnale**; l'acceptance-rate ne e la derivata aggregata. Non c'e un monitor di qualita separato da agganciare — motivo per cui e l'**acceptance stessa** il candidato, non un segnale interno preesistente.

---

## SPEC del test/patch che ne consegue

### T1 — Run instrumentato minimo (misura il LEAD; nessuna patch nuova, logging gia presente)
Obiettivo: allineare offline l'acceptance MTP-1 al primo-garbage e misurare LEAD = pos(first-garbage) - pos(acceptance-drop), col metodo di pregarbage_sensor_hunt.py.

- **Piattaforma:** pod RAM-backed (o 3060 con --ssd-streaming-cache-experts <=8GB), MTP-ON.
- **Comando** (combo-probe: drafting-ON, spec-dec OFF, verifier mai toccato = misura pura), env su una riga:
  `DS4_MTP_PROBE=1 DS4_MTP_SPEC_DISABLE=1 DS4_PACE=1 DS4_PACE_S1=1 DS4_REAP_STATIC=<mask K23-wide cyber> ds4 --mtp <MTP.gguf> --mtp-draft 2 --temp 0 --nothink -n ~400 -c 8192 -p <prompt cyberpunk che collassa @gen~42> 2> accept_probe.stderr`
  (--mtp-draft 2 abilita il dispatch della probe senza commit speculativi grazie a DS4_MTP_SPEC_DISABLE=1; la traiettoria resta quella greedy mascherata.)
- **Cattura per l'allineamento:** stderr (mtp probe token/draft/hit) + il trace token per-pos (patches/ds4/0028-spex-trace-tokens.patch) + content.txt per la posizione first-garbage.
- **Analisi offline:** per-token hit = (draft==token); EWMA dell'acceptance; misura il DROP **relativo alla baseline sana del run** (non assoluto); LEAD vs first-garbage-pos; controllo falsi-allarmi su un run gemello che **completa** (prompt coffee), come run3 del pregarbage_sensor.
- **Verdetto:** se LEAD > 0 e separabile dal run sano -> l'acceptance apre di nuovo "allarga-senza-rewind" nel regime aggressivo; se no -> il rewind resta l'unica leva e l'acceptance serve al piu come sorgente-FIRE aggiuntiva.

### T2 — Patch di comodita (opzionale, ~10 righe): DS4_DIAG_ACCEPTANCE_LOG
I valori sono gia calcolati a ds4.c:29900-29907. Un sink JSONL per-token (pos,token,draft,hit) evita di parsare stderr e da la posizione gia allineata; guardato da getenv("DS4_DIAG_ACCEPTANCE_LOG"), accanto alla probe a ds4.c:~29901. Costo compute ~0 (valori gia calcolati).

### T3 — Aggancio al detector del rewind (SOLO dopo che T1 da LEAD>0)
Il miss di acceptance e un binario per-token disponibile in ds4_session_eval_internal esattamente dove ds4_pace_rewind_feed_token e gia chiamato (0022, ds4.c:31288). Aggancio naturale, gemello del garbage-EWMA:
- nuovo EWMA in g_rewind alimentato da (mtp_draft_token != token);
- nuova sorgente ARM/FIRE "acceptance_drop" accanto a s1_cusum / char_garbage, con soglia sul DROP relativo (non assoluto), stessa isteresi del garbage-latch.
- **Gate:** non cablare l'attuatore a un segnale non validato (lezione pregarbage). T1 prima.

---

## Sintesi
- **(A) Rollback riusabile: SI.** spec_frontier_snapshot/restore + ds4_session_rewind = il rollback speculativo; 0027 lo prova bit-exact a k=50-300; 0022 lo replica su buffer propri. Da riusare per robustezza: commit_prefix1 (partial-accept esatto) e l'accept-bookkeeping multi-token per far girare 0022 **con MTP-ON** (che B richiede).
- **(B) Acceptance anticipa il garbage: PROPOSTA, non dato.** Segnale gia strumentato (DS4_MTP_PROBE, per-token, ~0, traiettoria invariata), ma nessuna traccia allineata al collasso esiste (solo baseline sana pod). E il candidato migliore perche cross-model e senza finestra di calib. -> run T1.
- **(C) Detection: accept/reject e l'unico segnale** (+ margin gate proto-confidence sul draft); nessun hook di degenerazione dedicato.
- **Prossimo passo:** T1 (run instrumentato, pod RAM o 3060 cache-ridotta) per misurare il LEAD; T2 opzionale per il logging pulito; T3 solo se T1 e positivo.

## File
- (analisi read-only; nessun sorgente modificato)

## Attribuzione (fissata 2026-07-11, richiesta utente)
Il meccanismo speculative MTP (draft/verify, rollback KV: spec_frontier_snapshot/restore + ds4_session_rewind + commit_prefix1, e l'acceptance esposta via DS4_MTP_PROBE) è **lavoro di ds4/antirez** — presente nel base 80ebbc3 (HEAD antirez 2026-06-17; commit MTP del 2026-06-14..16). È l'MTP speculative-BASE, non il DSpark-pieno DeepSeek (no confidence-head/STS/scheduler). 
NOSTRO contributo distintivo (da citare come tale nel paper, accanto a "REAP saliency = Cerebras; loop = nostro"): (1) uso dell'**acceptance-drop come rilevatore di degenerazione/deriva** (antirez usa accept/reject solo per la velocità — nessun aggancio-qualità); (2) composizione acceptance→ARM/FIRE del rewind (0022) + mask session-learned. Riusiamo il loro rollback; inventiamo l'aggancio-degenerazione e la composizione.
