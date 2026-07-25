#!/bin/bash
# run_manifest.sh — MANIFEST COMPLETO di un run: cosa e' stato CHIESTO e cosa e' stato OTTENUTO.
#
# PERCHE' SERVONO ENTRAMBE LE COLONNE (lezione 2026-07-25):
#   DS4_G73_PAGEABLE_OVERFLOW_GB=14 era presente in TUTTI i run, ma l'effetto era
#   pageable=0.00 GiB, perche' la leva e' gated dentro `if (cuda_g73_open_requested())`.
#   Registrare solo l'ambiente avrebbe scritto "14" in entrambi i casi, e avremmo
#   confrontato un run con 44 GiB di arena contro uno con 30 credendoli identici.
#   L'ambiente dice cosa hai chiesto. Solo il log dice cosa hai ottenuto.
#
# Uso: run_manifest.sh <label|percorso-server.log>
set -u
G=/d/ds4_work/g73_gate
arg="${1:?uso: run_manifest.sh <label|server.log>}"
L="$arg"; [ ! -f "$L" ] && L=$(ls -t $G/*"$arg"*.server.log 2>/dev/null | head -1)
[ ! -f "$L" ] && { echo "log non trovato per '$arg'"; exit 1; }
B=$(basename "$L" .server.log)

hdr(){ echo; echo "--- $1 ---"; }
grab(){ grep -aoE "$1" "$L" 2>/dev/null | sort -u | head -"${2:-3}" | sed 's/^/    /'; }

echo "================================================================"
echo "MANIFEST: $B"
echo "================================================================"

hdr "1. BINARIO"
grep -ahoE 'EXE=[^ ]+|EXE_MD5=[a-f0-9]{32}' $G/*.txt 2>/dev/null | sort -u | head -2 | sed 's/^/    /'

hdr "2. AMBIENTE RICHIESTO (tutte le DS4_*)"
awk '/MANIFEST env DS4/{f=1;next} /EXE_MD5=/{f=0} f' $G/*_results.txt 2>/dev/null | grep 'DS4_' | sed 's/^ */    /' | head -60
echo "    (vuoto = run non passato dal runner con manifest)"

hdr "3. OTTENUTO — MEMORIA E RESIDENZA  [LA COLONNA CHE CONTA]"
grab 'dynamic arena ready pinned=[0-9.]+ GiB pageable=[0-9.]+ GiB[^|]{0,60}' 2
grab 'arena-cap\] requested_gib=[0-9.]+[^|]{0,150}' 1
grab 'resident expert cache ready: [0-9]+/[0-9]+ experts, [0-9.]+ MiB/expert, [0-9.]+ GiB total' 2
grab 'iq2-expert-cache-size\] result=[a-z]+ requested=[0-9]+ capacity=[0-9]+[^|]{0,120}' 1
grab '\[q1-0-resident-arena\] result=[a-z]+ [^|]{0,140}' 2
grab 'startup model cache prepared [0-9.]+ GiB|startup cache excluded [0-9.]+ GiB of [a-z-]+'
grab 'registered [0-9.]+ GiB contiguous host window[^|]{0,40}' 1

hdr "4. OTTENUTO — MODELLO, SIDECAR, LAYER"
grab 'Q1_0 expert sidecar source: [^ ]+' 1
grab 'sidecar validated: layers=[0-9]+ active=[0-9]+\.\.[0-9]+[^|]{0,60}' 1
grab 'installed: [0-9.]+ GiB' 1

hdr "5. OTTENUTO — CONTESTO, KV, CHUNK"
grab 'context buffers [^|]{0,120}' 1

hdr "6. OTTENUTO — TIERING E ROUTE"
grab '\[expert-tiering\] final mode=[a-z]+ policy=[a-z-]+' 1
grab 'compose_prefill_mass_tiering=[0-9] compose_router_open=[0-9]' 1
grab '\[prefill-mass\] finalize layers=[0-9]+[^|]{0,60}' 1
grab 'GPU-resident route resolver active[^|]{0,40}' 1

hdr "7. RICHIESTA (think, seed, temperatura)"
grep -ahoE '"think":[A-Za-z]+|"seed":[0-9]+|"temperature":[0-9.]+' $G/test_fullq1*.sh $G/suite10*.sh 2>/dev/null | sort -u | tr '\n' ' ' | sed 's/^/    /'; echo
echo "    THINKING nel log: $(grep -ac 'THINKING' "$L" 2>/dev/null) occorrenze  (0 = think:false applicato)"

hdr "8. TRAFFICO (da dove arrivano davvero gli esperti)"
grab 'ram_hits=[0-9]+|vram_hits=[0-9]+|cold=[0-9]+|ssd_bytes=[0-9]+|ram_h2d_bytes=[0-9]+' 6
grab 'resident_hits=[0-9]+|resident_misses=[0-9]+|direct_pread_fallbacks=[0-9]+|resident_h2d_bytes=[0-9]+' 4

hdr "9. VALIDITA' E RISULTATO"
n=$(awk '/vram-ledger] phase=decode-start/{d=1} d&&/CUDA loading model tensors/{c++} END{print c+0}' "$L" 2>/dev/null)
echo "    RELOADS_IN_DECODE=$n   $([ "${n:-0}" -eq 0 ] 2>/dev/null && echo '-> run VALIDO' || echo '-> run DA SCARTARE')"
grep -ao 'decode_ms=[0-9.]*' "$L" 2>/dev/null | sed 's/decode_ms=//' | python3 -c "
import sys,statistics
v=[float(x) for x in sys.stdin]
if not v: print('    nessun token'); raise SystemExit
print(f'    token={len(v)}')
if len(v)>15:
    m=statistics.median(v[10:]); print(f'    steady={m:.0f}ms = {1000/m:.2f} t/s')
    print(f'    primi5={[int(x) for x in v[:5]]}  ultimi5={[int(x) for x in v[-5:]]}')
" 2>/dev/null
GEN=$(ls $G/$B.gen.txt 2>/dev/null)
if [ -n "$GEN" ]; then
  echo "    testo: $(stat -c %s "$GEN") byte"
  printf "    tag HTML: "; for t in '<!DOCTYPE' '<html' '<style' '</html>'; do printf "%s=%s " "$t" "$(grep -oc -- "$t" "$GEN" 2>/dev/null)"; done; echo
  echo "    inizio: $(head -c 100 "$GEN" 2>/dev/null | tr '\n' ' ')"
else
  echo "    testo: NON catturato"
fi
echo "================================================================"
