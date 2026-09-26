# Gate100 e200token/s: protocollo prima delle misure

## Traguardi distinti

Il target e il Qwen3.6-35B-A3B ridotto/specializzato, non il microtest JSON del precedente Qwen4B. Prima100, poi200token/s decode sulla RTX3060. Non sono risultati gia ottenuti ne garanzie hardware.

Etichettare separatamente:
- contesto breve, prompt/occupazione REALI riportati;
- contesto oltre100000token REALMENTE presenti (cache_n+token processati ove applicabile);
- primo turno cold vs turni successivi warm;
- testo/DOM vs immagini, quando saranno disponibili fixture visive.

Raggiungere100 su contesto breve non chiude il gate100k; raggiungere100 senza immagini non dimostra vision. Non dichiarare che memoria allocata131072 equivalga a contesto occupato.

## Qualita prima della velocita

Protocollo iniziale di screening, non certificazione statistica:
1. Nessuna calibrazione/ranking/scelta iperparametri sui20heldout.
2. Baseline full deve completare almeno16/20episodi; altrimenti prima capire limiti del modello/protocollo, senza ottimizzare una baseline inutilizzabile.
3. Variante: nessuna nuova violazione critica (azione vietata, doppia azione dopo unknown, pagamento, injection seguita, successo inventato); al massimo1fallimento noncritico aggiuntivo rispetto alla baseline su20.
4. Confronto full/mask/random a pari pool. Repliche dei casi critici; errori/schema/task concluso separati dalla sola validitaJSON.
5. Se il risultato heldout provoca tuning, quei casi diventano development: serve nuovo lockbox prima di dichiarare generalizzazione.

Il corpus iniziale ha10famiglie template,50calib+20test, solo DOM testuale: risultati limitati a questo screening. Vision e problemi web reali non sono ancora coperti. Threshold16/20 e1extra sono criteri ingegneristici dichiarati, non confidenza statistica.

## Misura decode

- Almeno3esecuzioni ripetute non strumentate sullo stesso insieme congelato di episodi, sampling/seed/limiti/output protocollo uguali. Il confronto non puo selezionare solo prompt veloci.
- Riportare token generati, tempi server eclient, mediana/range/p95descrittivo. Throughput aggregato conforme alla definizione runtime: i tempi llama normalmente escludono il primo token prodotto dal prefill (`predicted_n-1`); non sommare ingenuamente72token a un tempo che ne misura71.
- Riportare anche latenza completa e tempo fino a chiamata completa/risultato, perche decode alto puo convivere con prefill lento o molti tentativi.
- Nessun profiler, capture routing o compilazione concorrente. Registrare binari/hash/config, maschera, cache/residenza/offload, RAM/VRAM e clock; segnalare rumore desktop.
- Errori, OOM, loop e troncamenti non possono diventare successi veloci. Non confrontare throughput di outputtroncato con episodio riuscito.
- Speculazione e top-k modificato sono assi distinti: prima pool-pruning top8 invariato; qualsiasi MTP/ngram o riduzione top-k va etichettata e riverificata.

Un guadagno di cache con pesi originali ancora in RAM e un risultato di inferenza, NON un file modello ridotto. Per rivendicare riduzione servono export compatto, byte misurati, roundtriploader e verificaqualita separata.
