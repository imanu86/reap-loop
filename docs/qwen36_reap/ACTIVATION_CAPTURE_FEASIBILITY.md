# Capture input FFN e salienza: fattibilita readonly, NON implementazione

Audit separati di runtime e replay offline dopo139. Nessuna compilazione, cattura hidden, lettura dei pesi o inizializzazione CUDA eseguita per questo audit. Il prossimo esperimento operativo rimane l'ablazione economica CONDITIONAL_GATE_PLAN.md.

## Formula e punto di osservazione

Il criterio descritto nei riferimenti locali REAP e `sum_selected(g * L2(f_expert(x))) / N_selected`; esperti mai selezionati=>0. Non g^2*norma^2, non divisione per tutti i token e non norma della somma degli esperti. I riferimenti locali sono scripts/reap_gonly_vs_eq9_30b.py3-8, scripts/analyze_man_score_feasibility.py9-21 e il codice g-only scripts/reap_saliency_ds4.py105-107. Il file scripts_pod/reap_saliency.py citato da altri documenti non e presente: audit non esteso a quel file.

Nel sorgente lab qwen35moe.cpp210-214/493-511, `attn_post_norm-{layer}` e x gia DOPO RMSNorm e moltiplicatore appreso, passato al routed FFN; anche il shared FFN lo usa a517-528. Non normalizzare una seconda volta nel replay.

Fattibilita strutturale: ricordare il puntatore su ask senza chiedere uno stop anticipato e copiarlo soltanto nell'esistente callback finale dei pesi, insieme a ID/gate. x e antenato del router, quindi gia calcolato; viene ancora usato dai consumer expert/shared successivi. graph2149-2150 colloca il callback dei pesi prima del reshape2197 e delle matmul2231/2249; i riferimenti di child/view mantengono vivo lo storage (allocator730-817). La fusione RMS_NORM+MUL scrive il MUL finale (cuda4559-4561/norm.cu522), non soltanto l'intermedio RMS.

Queste osservazioni NON sono una nuova prova numerica. Vietati nuovi nodi, CONT, rename, rewiring, flag di output che forzino allocazioni o tap FFNOUT che rompano la fusione. Verificare tipo F32, dimensioni effettive, stride e contiguita: non dedurli dal nome. Limitare inizialmente a testo/singola sequenza/noMTP.

## Formato e risorse da progettare

DefaultOFF; un eventuale flag hidden deve richiedere TRACE e rifiutare mask nella raccolta di riferimento. Formato F32 little-endian con versione, identita modello/runtime/sorgenti, definizione di x, ID batch/decode/layer/token/pos/seq, offset64bit e dimensioni. Output nuovi esclusivi; limiti MAX_BYTES/spazio libero; completezza e hash verificati solo dopo chiusura. Nessuna promozione di tail parziali o index senza tutti40layer.

Al volume di136, 2,932,080righe layer-token x2048 x4 =24,019,599,360byte (~22.37GiB) di soli input. Un chunk128token/layer vale1MiB,40layer40MiB. Non duplicare gli input8volte. Preferire shard per layer per evitare40scansioni dell'intero file; I/O e memoria vanno misurati, non stimati come gratuiti.

Prima di una raccolta grande: OFF/TRACE/TRACE+hidden con entrambe le fixture e single-token; logits completi dell'ultimo token alle tolleranze gia congelate, stesso argmax/token, stessi ID/pesi routing, allineamento40layer, finite/shape, roundtrip binario, graph reuse, chunk prefill senza output e rifiuto degli errori. Il vecchio gate non copre automaticamente il nuovo read.

## Replay offline approssimato

Disponibili senza nuove installazioni: Python3.12.10, NumPy2.2.1/OpenBLAS0.3.28, GGUF0.19.0, Torch2.6.0+cu124 importabile senza inizializzare CUDA. L'audit ha verificato `torch.cuda.is_initialized()==False`; nessuna verifica/dispositivo CUDA eseguita.

GGUF0.19.0 contiene Q4_K.dequantize_blocks in gguf/quants.py475-522. GGML Q4_K: blocchi256pesi/144byte, due f16(d/dmin),12byte scale/min6bit,128byte nibble; ordine per gruppi32, non nibble interleavati arbitrariamente. Q4_K_M e una ricetta di file: controllare SEMPRE il tipo del singolo tensore. Slice dell'esperto PRIMA della dequantizzazione, mai espansione dell'intero modello (~120GiB F32 routed).

Orientamento GGML gate/up[D,H,E], down[H,D,E]; NumPy[E,H,D] e[E,D,H]. Per righe X[B,D], G=X@Wgate.T,U=X@Wup.T,Y=(SiLU(G)*U)@Wdown.T. Per eventuale gate_up fuso, gate prima di up. Verificare bias/scale/clamp/LoRA prima di usare questo schema semplificato. Y deve essere il routed expert individuale dopo down, prima di gate/shared/residual.

F32 dequantizzato+GEMM NON e identico al CUDA quantizzato: mmvq.cu1544-1550 quantizza anche input Q8_1; il CPU quantizzato usa Q8_K. Prefill/MMQ, ordine degli accumuli, fusioni e SiLU possono differire. Anche Torch GPU e solo approssimato senza una validazione dedicata; non attivare TF32/FP16 e chiamarlo esatto.

Elaborare soltanto gli8esperti osservati, per layer/esperto/tile, riducendo subito somme/count scalari. A D2048/H512 sono circa147.6TFLOP, non un tempo misurato; calcolare tutti256 costerebbe32volte tanto. Dequant/gather/I/O non sono inclusi.

Test preliminari: blocchi Q4_K sintetici con tutti i bit scale/min, nibble estremi, segni, subnormal f16, LE/troncature/layout/tail; confronto con oracle C dequant verificato; MLP+SiLU+L2 su piccoli tensori inventati, tile invariance, nessun rerouting/doppia norm. Poi piccolo probe held-IN per quantificare errore di output/norma/ranking rispetto a un riferimento dello stesso backend. Un replay nativo CPU non dimostra identita con il CUDA realmente usato.

Nessun risultato di questa fattibilita approva maschere, export, contesto100k o100/200t/s.
