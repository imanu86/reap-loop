# REAP Qwen3.6-35B: mandato e laboratorio

## Obiettivo autorizzato

Utente: procedere con REAP; primo traguardo100token/s decode, secondo200token/s. Target misurato, NON promessa. Qualita web/DOM/tool calling/MCP/Cockpit necessaria; contesto breve e100k occupati misurati separatamente. Nessun claim basato soltanto su JSON ripetitivo. Vision desiderata, ma fixture multimodali e misura memoria da aggiungere: non coperta dal pilot testuale.

## Autorizzazioni e protezioni

Conferma umana esplicita: lavorare sul ramo esistente `plan/0051-transport-gate-20260713`, senza cambiarlo; runtime in copia separata; sospensioni temporanee del daily per GPU con ripristino; commit e push inclusi28commit precedenti. Preservare CSV modificato `runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv`, file altrui, donor e pesi originali. Nessun telefono/MCP reale, nessun pagamento. GPU gestita solo dal coordinatore.

Daily da ripristinare: Qwen3.5-4B Q4_K_M originale sulla8100, tramite Desktop `Qwen 4B - DSH Cockpit - 128k.bat`. Non Ling e non il vecchio nome modello DSH. Non registrare chiavi nei report.

## Checkpoint verificato nel precedente audit

`D:\models\qwen36moe\Qwen3.6-35B-A3B-Q4_K_M.gguf`,20419565568byte; metadata architectureqwen35moe,40layer,256esperti/top8,hidden2048,FFexpert512/shared512. IQ3locale includeMTP41layer: NON stessa composizione baseline. mmprojQ8locale614194304byte,446571248parametri; da lasciare invariato.

Nonrouted2.448B,routed32.212B. Solo pool-pruning non porta0.5Battivi: trunk>1.41B anche senzaembedding/output/router, routedtop8~1.007B attivi. ExportuniformeP(K)=2427384448+125911040*K senzaMTP:4Bconvision richiedequasiK8, troppoaggressivo comeprimaipotesi. Non fissare4B comecondizione che giustifichi perderequalita.

## Ownership

- Runtime child08cb8dd6: D:\ds4_work\qwen36_reap_lab\source e docs/qwen36_reap/runtime, provenienza/capture/mask. NO build/GPU finche coordinatore non concede finestra.
- Pilot child9339dd38: docs/qwen36_reap/pilot soltanto,50calib/20heldout offline,validator/test. Nessunainferenza.
- Coordinatore: questoregistro,mask_builder e test,gestioneGPU,baseline,integrazione/commit/push.

## Gate

1. Provenienza source + overlayM3/hash, baseline fitVRAMsenzaassumereflagcache, schema tracciaeffettiva.
2. Duefixture: no-mask e capture-only preservanooutput; ID/pesiallineatilayer/token,top8,range/finite/normalizzazione. Recordfaseunknown se non dimostrabile.
3. Tracce50calib;heldout MAI usatoper ranking. Massa-gateperlayer,coverage95/99 o poolprudenti,controllorandoma paripool,top8invariato.
4. Ventiheldout: taskconcluso,schemi,argomenti,DOMgrounding,recupero,no-loop,nosafetyviolations;stratifuturevisionseparati. Ripeterecasocritici.
5. Exportfisico solodopoqualita: quant-blocksafe,routedtensors/routermetadata/remap,loaderroundtrip;maiGGUForiginaleinplace.
6. Timing nonstrumentato,inferenzaesclusivaGPU;100e200t/s solo se workload/campioni/contesto/cache dichiarati,non velocitasoloMCP. Persistono anchelimiti/risultatinegativi.

## Stato iniziale

Implementazione avviata, nessun pruning/inferenza35B ancora eseguito. Primo componente pronto: mask_builder.py legge trace JSONLtop8 effettive, valida geometria/hash/normalizzazione/allineamento40layer, genera keep-list per massa-gate o controllo casuale a pari pool. Nove test CPU passati; sono fixture sintetiche, non attivazioni del modello. Hash checkpoint originale congelato (job parent pwsh-92 completato): Q4 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`; mmproj `904cbf8c8e876220066ab3bf676c7efa40f3da372276fdaf8b01d2fb2a37a51d`. Manifest locale D:\ds4_work\qwen36_reap_lab\coordinator\checkpoint.json. Runtime:3007file verificati donor+3overlayM3, patchOCC12del4B esclusa. Ramo verificato e dirtyCSVpreservato. Audit precedente in moe-aggressive-commit commit ae10382, docs/porto/banco_qwen4b/REAP_QWEN_READINESS.md. Parent goal goal-1321e866-9c01-41e3-ba62-446bd7854a40 attivo.

## Round1

- Build runtime isolata pwsh-95:225target completati, nessuna compilazione CUDA; GGMLoriginale verificato/importato, llama e common ricompilati. Assembly bin-candidate/ABI childpwsh-96 completato:263exportidentici, nessunimportirrisolto. Candidate cambia soltanto llama.dll SHA2565f4bb83abbde1938dabdd8764215315fd1895597b95a0d5a1576192a802e3566; common/GGML/server originali intatti. Per helpernativo useremo commonfreshmatchedheaders suENTRAMBI arms. Nessuna inferenza35B.
- Memoria da catalogo: routed18119393280byte, nonrouted2289183232byte; cache32payload2264924160byte, circa4.24GiB nonrouted+cache, PRIMA di KV/stati/workspace/OS. Fit non ancora misurato.
- Corpus70episodi preparato,15testvalidator ripetuti dal parent PASS: e coerenza dell'oracle, non prestazione del modello. Rimosse/da rimuovere eventuali etichette id/split dai prompt pubblici prima freeze.
- Runner pilot in sviluppo con nativeOpenAItoolcalls simulate, niente dati oracle nei prompt. Gate nativo logitsF32 delegato a1557e537, ownershipdocs/qwen36_reap/gate; no build propria.
- Ledger esterno invariatoSHA2566858dc61e4c5ce19b6dfe6295784c9da4a44eada8ac524335bf134fe22b5a798. Daily8100 healthOK. Nessun target100/200 raggiunto o dichiarato.
