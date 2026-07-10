# 2026-07-10 Pod2 (redeploy) — RETRY build CANONICAL v2: PASS, switchover sbloccato

Retry del build canonical fallito in `runs/ds4/20260710_pod2_smokes/canonical_build/`
(errore `unknown type name 'ds4_gpu_async_read'`: sibling GPU fermi a fine-0014e),
dopo il commit dei sibling canonici **458f4b6** (`0014f-canonical-siblings-gpu-header`
+ `0014g-canonical-siblings-cuda-impl`).

## Setup

- Pod: RunPod `i7dk94f0y05iji` (SECURE RTX 3090, machine `ktxcuvw1vccq`, $0.46/h),
  redeploy fresco da ricetta R2 (`docs/POD_R2_CACHE.md`) dopo 2 tentativi falliti di
  resume del pod2 originale `o0gd30ojfacz96` ("not enough free GPUs on the host
  machine" — host community pieno; il pod resta in STOP con il suo volume).
- Base PULITA: `github.com/antirez/ds4` @ `80ebbc3` (clone fresco, `git clean -qfdx`).
- Patch: tar via `git archive HEAD patches/ds4` dal repo reap-loop @ HEAD
  (contiene 458f4b6) — blob puliti, niente CRLF da checkout Windows… quasi (v. sotto).

## Esito (catena + build)

1. **Catena canonical 21/21 APPLICA PULITA** (`canonical/` sorted, incl 0014f/0014g).
   Checkpoint: dopo le 21 canonical `ds4.c` md5 = `1db4f799` — **identico** alla
   catena livetree+0020+0021+0026 ("admit"): conferma indipendente che la serie
   canonical ricostruisce lo stato live-tree.
2. **0027/0028: git apply FALLIVA** (`patch does not apply` @ ds4.c:30415). Causa:
   i **blob committati** di 0027 (271 righe) e 0028 (67 righe) contengono CRLF misti
   (il messaggio di 458f4b6 li dichiara "committed LF blobs", ma `git show
   HEAD:patches/ds4/0027… | od -c` mostra `\r`). Fix sul pod: `sed -i 's/\r$//'`
   sui due file → entrambe applicano pulite. `canon_build_apply.err` conserva
   l'errore originale. TODO repo: normalizzare i blob a LF (+ `.gitattributes`).
3. **ds4.c md5 finale = `62ed2e71`** — ESATTO il target (= catena livetree pace0028).
4. **make cuda CUDA_ARCH=sm_86 -j32: OK** — 0 warning, 0 error (`canon_make.log`);
   prodotti `ds4` (10 868 120 B) e `ds4-server` (11 788 344 B).
5. **Micro-smoke GPU: PASS** — `ds4 -n 8 --cuda --ssd-streaming` rc=0, output
   coerente, diag `prefill/generation` presente (`canon_smoke.out/.err`; eseguito in
   contesa col job sampling → tempi non indicativi).

## Upload R2 (bucket `ds4-models`)

| oggetto | sha256 |
|---|---|
| `ds4_sm86_canonical-62ed2e71-v2` | `0f4fcafb1e64e3b6…eaa8ff` |
| `ds4-server_sm86_canonical-62ed2e71-v2` | `1a8bd0bd8787398e…51a1e` |
| `ds4_sm86_canonical-62ed2e71-v2.meta` | provenienza completa |

## Verdetto

**SWITCHOVER SBLOCCATO**: la base canonical committata (80ebbc3 + patches/ds4/
canonical/ 21 + 0027 + 0028) ricostruisce e builda l'INTERO stato live-tree
(md5 62ed2e71) senza dipendere da file non committati del live tree WSL.
Unico residuo: CR-strip di 0027/0028 prima di `git apply` (fino alla
normalizzazione dei blob nel repo).

## File

`canon_build.sh` (script eseguito), `canon_build.log` (apply 21/21 + fail 0027
pre-fix), `canon_build_apply.err` (errore CRLF originale), `canon_make.log`
(build completo), `canon_sha256.txt`, `ds4_sm86_canonical-62ed2e71-v2.meta`,
`canon_smoke.out/.err` (micro-smoke GPU).
