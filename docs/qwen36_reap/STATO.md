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

Gate GPU35B job113/115 PASS: OFF/TRACE/all-kept bit-identici, keep7 respinto, maschera arbitraria128IDpari esclude correttamente (nessuna qualita/riduzione fisica dichiarata). Calibrazione completa v1/512 job116:1/50completati; job119 valida2635920record/65898token per layer e daily healthOK. Baseline NON utilizzabile, niente pruning/export. Diagnosi: troncamenti, ridondanze e conflitto contratto pubblico/schema. Job121 A/B concluso: v1/1024=0/6, v2/1024=0/6, nessuna troncatura; daily ripristinato healthOK. Finalizzazione ancora fallita. Job124 concluso: originale0/2 e candidato018c0/2 senzaTRACE/MASK; prompt, rawcontent e tokenIDENTICI su4turni. I tool superflui sono gia nei token: non conversione errata del parser e non regressione specifica della patch su questi casi. Causa aperta. Round5: bypass126 emettefinal-tool NONdichiarato (errore); sampling ufficiale127 seed0/1/2 tutti0/2, zero troncature. Prossimo controllo pubblicato in FINALIZATION_PROBES.md: final-mode tool opt-in con PUBLICfinalschema, nessun cambiamento al simulatore/risposte attese, nessun rescoring storico. Job129 sulnuovotrasporto:2/2completati, simulatoreimmutato; pubblicato7e4259f primadeltest. NONgatequalita: campione2casi scelti daifallimenti. Job130 concluso42/50; webworkflow0/5 perprima osservazioneridondante,2recovery evidenceform e1currencytroncata. Daily ripristinatohealthOK. Job132CPU validadisponibili2,768,240record (69,206token/layercumulativi),8maschereNONAPPROVATE: rankedK128massa89.02%,K64=66.90%. NessunpruningGPU/export. Nuovaguardia: almeno1completamento/famigliaprimapruning. Prossimocontrollo preregistratoPUBLIC_CONTRACT_AB.md: V2/V3 x1024/2048,contesto6144DIAGNOSTIC_ONLY,budgetcorpus4096immutato. V3pubblicata58b512c prima dell'inferenza. Job134concluso:V2/1024=0/3,V3/1024=3/3,V2/2048=1/3,V3/2048=3/3;prove6144DIAGNOSTIC_ONLY,V3tokenidenticiaiduecaps. SceltoV3/1024; avviatojob136 full50 in coordinator/calibration_50_v3_final_tool_01, candidato018cTRACEON,context4096/greedy0/finaltool/rawON. Job136CONCLUSO45/50, tutte10famiglie>=3completamenti; dailyoriginaleripristinatohealthOK. Job137CPUvalidati2,932,080record(73,302token/layercumulativi),8maschereV3readyforscreen maNONAPPROVATE. PlannerV2/V3esplicito13testPASS. PrimoscreenGPU:baseline10frescauntraced/rawOFF poiK128ranked/random; coordinatoreora22testCPU PASS, ricetta server identica a136 e runtime9fileverificato; wrapperSHA 2b5379c9f926f9204378f1e3422efdab00a0ffd3cac1bb86ec9b69ef4c5caf30. Review50casi/149turni/102azioni:0critiche confermate, nessunrescoring. Job139CONCLUSO: baseline9/10,ranked1287/10,random1285/10. Review30casi/83turni:ranked2nuovifallimentinoncritici,random4nuovifallimentiincluso1fabricated_success. Reducer reviewed REJECTED:STOPmass_gate128,nessun96/64/32/export. DailyoriginaleripristinatohealthOK. Nuovaablazioneg-onlymean_selected_gate preregdef0c73,codee1f3588/41testPASS;142CPUconcluso8maschereNONAPPROVATE inselection_v3_mean_gate_01,recipeSHA7e404ac0d429abaac9a6dabe1df75348ea7089edc972f1597ddea19de25d9132. Somma(g)/N_selected,NONnorme/fullREAP;corpus/cap/validatorinvariati. Wrapper22test/reducer29testPASS,comandoinferenzainvariato,regressione139identica;metodoesplicitoboundallaricetta. NessunaGPUmeanancora. LedgerSHA6858dc61e4c5ce19b6dfe6295784c9da4a44eada8ac524335bf134fe22b5a798 invariato. Job120 fallito prima modello per contratto string/list corretto+testato. Heldout20 NON eseguito. Exporter SOURCEONLY pronto:29test sintetici, gate20casi/baseline16/no nuove violazioni e hashprotocolloidentico obbligatorio; nessun GGUF reale. Vedere NUMERIC_STATUS.md, PILOT_STATUS.md e PILOT_AB_PROTOCOL.md; Round1 sotto storico. Primo componente pronto: mask_builder.py legge trace JSONLtop8 effettive, valida geometria/hash/normalizzazione/allineamento40layer, genera keep-list per massa-gate o controllo casuale a pari pool. Nove test CPU passati; sono fixture sintetiche, non attivazioni del modello. Hash checkpoint originale congelato (job parent pwsh-92 completato): Q4 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`; mmproj `904cbf8c8e876220066ab3bf676c7efa40f3da372276fdaf8b01d2fb2a37a51d`. Manifest locale D:\ds4_work\qwen36_reap_lab\coordinator\checkpoint.json. Runtime:3007file verificati donor+3overlayM3, patchOCC12del4B esclusa. Ramo verificato e dirtyCSVpreservato. Audit precedente in moe-aggressive-commit commit ae10382, docs/porto/banco_qwen4b/REAP_QWEN_READINESS.md. Parent goal goal-1321e866-9c01-41e3-ba62-446bd7854a40 attivo.

## Round1

- Build runtime isolata pwsh-95:225target completati, nessuna compilazione CUDA; GGMLoriginale verificato/importato, llama e common ricompilati. Assembly bin-candidate/ABI childpwsh-96 completato:263exportidentici, nessunimportirrisolto. Candidate cambia soltanto llama.dll SHA2565f4bb83abbde1938dabdd8764215315fd1895597b95a0d5a1576192a802e3566; common/GGML/server originali intatti. Per helpernativo useremo commonfreshmatchedheaders suENTRAMBI arms. Nessuna inferenza35B.
- Memoria da catalogo: routed18119393280byte, nonrouted2289183232byte; cache32payload2264924160byte, circa4.24GiB nonrouted+cache, PRIMA di KV/stati/workspace/OS. Fit non ancora misurato.
- Corpus70episodi preparato,15testvalidator ripetuti dal parent PASS: e coerenza dell'oracle, non prestazione del modello. Rimosse/da rimuovere eventuali etichette id/split dai prompt pubblici prima freeze.
- Runner pilot in sviluppo con nativeOpenAItoolcalls simulate, niente dati oracle nei prompt. Gate nativo logitsF32 delegato a1557e537, ownershipdocs/qwen36_reap/gate; no build propria.
- Ledger esterno invariatoSHA2566858dc61e4c5ce19b6dfe6295784c9da4a44eada8ac524335bf134fe22b5a798. Daily8100 healthOK. Nessun target100/200 raggiunto o dichiarato.
