# Nuovo esperimento: ablazione gate medio condizionale

Registrato DOPO lo screen139 e PRIMA di generare/usare maschere con questo criterio. La curva mass_gate resta FALLITA aK128: baseline9/10, ranked7/10, random5/10. Non si prosegue96/64/32 per quel criterio, non si riscorano i due nuovi fallimenti ranked e non si cambia il contratto pubblico per eliminarli.

## Motivazione verificata nel codice del repository

La salienza REAP descritta da scripts/reap_gonly_vs_eq9_30b.py e scripts/analyze_man_score_feasibility.py e `SUM_selected(g * L2(output_expert)) / N_selected`. Il codice g-only in scripts/reap_saliency_ds4.py105-107 usa il denominatore condizionale. Il riferimento scripts_pod/reap_saliency.py non e presente qui: nessuna pretesa di averlo verificato.

Il criterio usato finora, somma delle gate g, favorisce anche la frequenza. Un controllo economico separato e quindi `mean_selected_gate = SUM_selected(g)/N_selected`, con0per esperti mai osservati e parita risolta per ID crescente. E un'ABLAZIONE g-only, NON REAP completo con norme delle attivazioni e NON una prova di recupero della qualita.

## Dati e procedura congelati prima della nuova selezione

Usare SOLO la stessa cattura completa V3 di136: tutti50casi, inclusi i5fallimenti, senza filtri/ripesature; traceSHA20c06be64158ce136a4e2f9ae1ddabead8cbc468bc5fe91435944c802f16970c, modello671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7. Nessun heldout per ranking o selezione.

Nuova directory e ricetta esplicita `mean_selected_gate`; default storico mass_gate invariato. Hash di ricetta distinti dall'hash del protocollo di cattura. Le diagnostiche di massa trattenuta devono continuare a usare SUM(g), non essere rietichettate usando le medie. Il random resta random_matched_pool, seed20260713, K abbinato e stessi ID del controllo storico a pari K; i nuovi file dichiarano quale esperimento controllano.

Curva NUOVA preregistrata:128,96,64,32. Iniziare dai10ID calibration-<family>-0 nello stesso ordine del corpus. Baseline senza mask fresca, nuovo ranked128, random128 fresco; tutti con candidato018c, native/final-tool/V3/greedyseed0,4096/1024,13turni, TRACE/rawOFF. Stessa ricetta server e corpus di139; le modifiche al coordinatore devono riguardare solo validazione/identita della nuova ricetta, non prompt, sampler o modelli.

Dopo ogni livello, review esaustiva degli arm: nessuna nuova violazione critica rispetto al baseline e al massimo1 nuovo caso fallito (mai netting con miglioramenti). Fermarsi al primo ranked non accettabile, anche se il random e peggiore. Scendere96->64->32 solo dopo review positiva del livello precedente. Nessun avvio anticipato del livello successivo.

Il piu piccolo accettabile richiede conferma su tutti50casi con controlli matched prima del lockbox. Nessun export, approvazione heldout o promessa di100/200t/s deriva da questi screen.

## Salienza con attivazioni: percorso distinto ancora non implementato

La fattibilita readonly suggerisce catturare `attn_post_norm` (post RMSNorm e moltiplicatore appreso) al callback finale dei pesi gia esistente, senza nuovi nodi o callback anticipati. Con la cattura corrente il payload F32 sarebbe24,019,599,360byte, circa22.37GiB. Servono nuovo gate numerico, formato binario verificato, completezza e limiti di risorse prima di acquisirlo.

Replay esperti da pesi Q4 dequantizzati e GEMM F32 darebbe inizialmente una norma APPROSSIMATA: CUDA quantizza anche gli input e puo differire per fusione/accumulo/SiLU. Non presentarla come output nativo esatto. Questo lavoro non e autorizzazione a saltare i gate della nuova ablazione; richiedera un piano numerico dedicato prima dei capture reali.
