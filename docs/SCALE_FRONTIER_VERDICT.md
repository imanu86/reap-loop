# Scala-frontiera — verdetto (indagine con verifica avversariale, 2026-07-06)

> **Stato dei claim: vedi [docs/CLAIMS_CURRENT.md](CLAIMS_CURRENT.md) (single source of truth, aggiornato con la retraction multiseed N=3 del 2026-07-07).** Se una cifra qui contraddice quel file, quel file vince.

> Il claim "il REAP-loop scala al frontiere / sblocca modelli da trilioni oggi non-servibili" verificato a terra
> (workflow `w9ev38a9x`, 8 agenti, WebSearch). **Verdetto: regge a METÀ.** Il muro è reale; la scale-invariance no.
> Serve prima di scrivere il paper: protegge da claim che antirez demolirebbe + da una collisione di nome.

## Verdetto in una riga
Il muro "modello ≫ memoria veloce" **esiste al frontiere ed è crescente** (motivazione valida). Ma la **scale-invariance del REAP-loop NON regge**: il collo del data center non è il PCIe, il **batching affossa il working-set stretto**, e il metodo è **già prior-art fino a 1026B**. Il contributo onesto è **edge/single-stream**, NON una tesi di sblocco a scala.

## Il muro a scala — CONFERMATO (la metà buona)
Footprint guidato dai **TOTAL** params (10-70× l'active): DeepSeek-V3 671B/37B, Kimi-K2 1T/32B, Llama-4 Behemoth 2T/288B. A precisione servibile alta NON sta in un nodo: V3 FP8 ~685GB > 640GB (8×H100); Kimi-K2 FP8 ~1TB satura 8×H200; Behemoth BF16 ~4TB multi-nodo intrinseco. Trend: **param 410×/2anni vs GPU-mem 2×/2anni** ([AI Memory Wall, arXiv 2403.14123]). ⚠️ CAVEAT: a **4-bit** i ~300-700B **rientrano** in un nodo H200/B200 → il muro morde solo nel regime **1T-2T ad alta precisione/KV lunga**.

## La scale-invariance — SI SMONTA (tre demolizioni)
1. **Collo DIVERSO**: 3060 = PCIe host→device ~16-32 GB/s; DC = all-to-all cross-node (InfiniBand ~400 Gb/s) o **NVLink intra-nodo 900 GB/s-1.8 TB/s** (~18× il PCIe). NVLink rende il transfer degli active-params quasi trasparente → "schivare il PCIe" ≠ "schivare l'interconnect del DC".
2. **Il BATCHING affossa il working-set** (la più letale): l'unione degli expert attivati cresce col batch fino a saturare — DeepSeek-R1 (256 exp, k=8) attiva **163/256 a batch=32, 243/256 a batch=64**; Mixtral 7.63/8 a batch~57. Il DC serve **batched** (vLLM Wide-EP: DeepSeek 671B a 2.2k tok/s/H200, tutti gli expert residenti) → il working-set stretto-per-dominio **collassa**, salvo serving **domain-shardato** (ipotesi forte, non costruita).
3. Misura utente = SOLO single-stream/single-GPU; il **batched-decode** (caveat noto, non costruito) è proprio ciò che rende il PCIe-dodging **irrilevante a scala**.

→ Sopravvive come *"stesso PRINCIPIO (working-set ≪ total), regime DIVERSO (single-stream, memory-constrained, consumer/edge)"*. NON come identità di meccanismo.

## Novelty — il metodo è AFFOLLATO (proprio al frontiere)
- **Pruning statico per-dominio**: **EASY-EP** (arXiv 2504.06792, NeurIPS'25) → subset stabile per-dominio su DeepSeek-R1 671B, metà expert, 2.99× throughput near-lossless = *esattamente* l'osservazione su cui il REAP-loop si fonda. Anche **PreMoE** (2505.17639).
- 🔴 **COLLISIONE FATALE DI NOME**: esiste GIÀ **"REAP"** (arXiv **2510.13999**, Cerebras) = "Router-weighted Expert Activation Pruning", domain-specialized, criterio router-gate × activation-norm, testato **fino a 1026B**, near-lossless a 50% su Qwen3-Coder-480B — **nome E criterio quasi identici. Il nome è bruciato.**
- **Dinamico/per-sessione**: **ExpertFlow** (2510.26730) fa dynamic per-session working-set resident + learn routing + prefetch; **LYNX** (2411.08982), **MoE-Infinity** (2401.14361), SwapMoE, ReMoE (2605.27081). Le survey chiamano "tieni il working-set residente e fetcha il resto" un *"widely used approach"*.

## Lo SLIVER novel (l'unica cosa difendibile come delta)
Il **gate-BIAS reversibile che RESTRINGE ATTIVAMENTE il routing LIVE** (non solo cache/prefetch reattivo alla ExpertFlow) + apprendimento del working-set di sessione + rilascio allo shift + integrazione ds4/DwarfStar consumer single-stream. Delta ingegneristico onesto **SOLO se posizionato come meccanismo EDGE**, non come principio a scala. **Novelty: PARZIALE, tendente a debole.**

## Azioni obbligate per il paper
1. 🔴 **RINOMINARE "REAP"** (collide con 2510.13999) — nel paper e nel branding. Il repo può restare, ma il paper serve un nome nuovo.
2. **Posizionare come EDGE/single-stream**: il muro-frontiera è la **MOTIVAZIONE** (cita AI Memory Wall), il meccanismo è consumer-specifico. NON rivendicare lo sblocco a scala.
3. **Citare (non rivendicare)**: EASY-EP, PreMoE, REAP-Cerebras (2510.13999), MoE-Infinity, ExpertFlow, LYNX, ReMoE, + le misure attivazione-per-batch.

## La frase difendibile (da mettere nel paper)

> ⚠️ **La cifra "23.6 tok/s near-lossless" della bozza originale è [SUPERSEDED — vedi docs/CLAIMS_CURRENT.md]: era un build confuso/crippled, RETRACTED come headline.** Numeri correnti difendibili: lo static keep-23 rende **11-17 t/s come SPEED DIAGNOSTIC su pod 3090** (non generalizzato); `reap/full` è **[OPEN]** a 1.009× CI[0.972,1.035] (il CI attraversa 1.0 → **non** dire "near-lossless" secco riferito al loop); il path staircase dinamico è **lento/cache-poisoned (~2.5 t/s), [ENG-BUG, path dinamico OFF]**. La versione qui sotto è riscritta senza il numero ritrattato.

> "On a single consumer GPU (RTX 3060 12GB), where expert transfer over PCIe (16-32 GB/s) dominates single-stream MoE inference [MoE-Infinity], we show that a domain-specialized working set of experts can be kept VRAM-resident via a **reversible gate-bias that actively restricts live routing** (zero-violation, reversible actuator). As a speed diagnostic on a pod 3090, a static keep-23 configuration sustains 11–17 tok/s in-domain; end-to-end `reap/full` throughput is **not yet distinguishable from full** (1.009× CI[0.972,1.035], CI crossing 1.0), and we make **no** near-lossless speed claim for the dynamic loop, whose staircase path is currently I/O/cache-bound. We frame this within the structural memory wall of frontier MoE, whose footprint is set by *total* (not active) expert parameters and grows ~410×/2yr against ~2×/2yr GPU memory [AI Memory Wall]. We explicitly do **NOT** claim this transfers unchanged to datacenter serving: batched, multi-node inference amortizes I/O across NVLink/all-to-all and re-expands the activated-expert union toward the full set [EASY-EP, ExpertFlow], so the working-set-narrowing lever is specific to the single-stream, memory-constrained (consumer/edge) regime."
