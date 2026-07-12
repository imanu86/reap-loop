# Prompt di handoff SPEX - 2026-07-12

Usa questo testo come prompt in una nuova sessione:

---

Lavora sul branch `spex-predictive-mask-study-2026-07-12` del repository
`imanu86/reap-loop`. Prima esegui `git pull`, poi leggi integralmente:

- `docs/SPEX_PREDICTIVE_MASK_STUDY.md`
- `docs/SPEX_ADDITIVE_MASK_PIN.md`
- `docs/SPEX_ADAPTIVE_K_STRONG_KNOCK.md`
- `runs/ds4/20260712_spex_adaptive_k_protocol_audit/REPORT.md`
- `runs/ds4/20260712_spex_adaptive_k_protocol_audit/MANIFEST.csv`

Cronologia dei commit da conoscere:

1. [d49fcf1 - prediction replay e hidden alignment](https://github.com/imanu86/reap-loop/commit/d49fcf1e0db8080069b213887acf0bd87843dd12)
2. [ebe73b6 - 0044 provisional SPEX mask/VRAM/WRAP lane](https://github.com/imanu86/reap-loop/commit/ebe73b69eb21c7eea10df00349a5c4736aeeb6a8)
3. [d2267a1 - 0045 cadence e adaptive K strong-knock](https://github.com/imanu86/reap-loop/commit/d2267a1b8b189155a8d2a0aaa912fc041d62fc6c)
4. [9bdbdfd - rimozione del cap errato a 80 token](https://github.com/imanu86/reap-loop/commit/9bdbdfdce837aca7031852f945d885cce882df99)

Sequenza del lavoro svolto:

1. E' stata misurata offline la predizione SPX1 senza addestramento sul prompt o
   sulla mask valutata. Il top4 raw ha recall pesata circa 19.08%; precisione
   additiva circa 4-7%. Il segnale e' reale ma debole.
2. La patch 0044 ha aggiunto una corsia SPEX provvisoria oltre il core: deduplica,
   WRAP sincrono prima dell'ammissione, pin VRAM/mask e lease. Top4 per layer ha
   distrutto la locality della cache256: 3.40% hit e 0.61 t/s nel primo smoke.
3. E' stata aggiunta la cadenza 2/4/8 con supporto Borda tra refresh e ampiezza
   configurabile 1/2/4. A cache256, cadence8 era piu veloce ma l'hit restava circa
   9%; a cache400 l'hit saliva circa 45% ma cadence2 restava molto costosa.
4. Un tentativo di qualita cadence2 ha prodotto due output chiaramente malformati
   a 800 token; il terzo e' stato interrotto a 129. Non e' n=3 e non va presentato
   come verdetto formale.
5. Offline e' stato definito il controller strong-knock: conta globalmente gli
   expert esclusi che il router controfattuale pre-mask mette nei top6 con peso
   normalizzato sopra soglia; usa l'accelerazione rispetto ai 10 token precedenti
   per variare K tra 16 e 50, con update2 e step +4/-1.
6. La patch 0045 ha implementato quel controller e separato il page-touch
   dell'allargamento logico dal WRAP obbligatorio SPEX. Build CUDA sm_86 riuscita.
7. I runtime smoke hanno trovato un reset bug, poi corretto; il prefetch di tutti
   gli ingressi adattivi creava batch fino a 800 expert/5.4 GiB ed e' stato
   disattivato. Add1 era meno distruttivo di add2/add4, ma tutti i confronti
   adattivi erano limitati a 80 token.
8. ERRORE DI PROTOCOLLO: la regola dell'utente era interrompere quando il codice
   degenera, non impostare max_tokens=80. Gli output 0045 sono solo micro-smoke;
   nessuno ha tentato un documento completo. Il runner e' stato corretto a
   max_tokens4000/ctx6144 con stream e stop su `</html>` o ripetizione oggettiva,
   ma non e' stato ancora eseguito.
9. CORREZIONE CONCETTUALE: il session learning W50/W130 storico congelava una
   mask statica; non dimostra il controller dinamico. Inoltre in 0045 il numero
   di strong knockers decide quanto allargare, ma gli ingressi sono ancora scelti
   per massa10: trigger e attuatore sono scollegati.

Regole permanenti:

- Non addestrare SPEX sul prompt o sulla mask valutata.
- Nessun verdetto da n=1 o dal solo repeat_flag: qualita con grader L0-L3 e n>=3.
- Un micro-smoke puo' scartare una politica palesemente rotta, non promuoverla.
- Non introdurre un hard cap breve: stream fino a `</html>`, degenerazione
  oggettiva o budget completo.
- Non chiamare fallimento il CSS ancora coerente solo perche' e' lungo.
- Ogni run deve salvare prompt, request, env completa, commit DS4, patch chain,
  modello/hash, cache, context, max token, log, output e motivo di stop.
- I dati devono provenire da misure, non da deduzioni.
- Non toccare o includere le directory non tracciate
  `20260712_delta_analysis`, `20260712_win30`, `20260712_winsweep`.

Stato tecnico al passaggio di consegne:

- Patch/runtime 0045 compilata, ma nessuna prova di qualita completa.
- Add1 e' solo il candidato meno dannoso nei micro-smoke, non un vincitore.
- Il prossimo fix corretto e' collegare l'attuatore al segnale: quando K cresce,
  ammettere direttamente gli expert esclusi col maggior peso controfattuale
  recente (per esempio media su tre token); usare massa10 solo per proteggere gli
  incumbent e scegliere le espulsioni.
- Solo dopo il fix eseguire il protocollo corretto prima con SPEX off e poi add1.
- La compressione dinamica degli expert resta fuori da questo segmento e non va
  dichiarata completata.

Prima di modificare o lanciare qualcosa, verifica che nessun `ds4-server` sia
attivo e presenta un piano breve. Mantieni un commit separato per ogni step e
aggiorna il report con risultati positivi, negativi e invalidi.

---
