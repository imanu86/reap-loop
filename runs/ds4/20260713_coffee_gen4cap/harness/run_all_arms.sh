#!/usr/bin/env bash
# Orchestrate coffee arms on the pod. Args: space-separated arm keys.
# Arm keys: k83_promo k83_staticpin k65_promo k100_promo
set -u
OUT=/root/out/20260713_linux12_coffee_promo
mkdir -p "$OUT"
R=/root/harness/run_coffee_arm.sh
M=/root/masks
WARM="${WARM:-48}"; MEAS="${MEAS:-1600}"; CACHE="${CACHE:-400}"

declare -A MASK=( [k83]="$M/mask_coffee_k83.txt" [k65]="$M/mask_coffee_k65.txt" [k100]="$M/mask_coffee_k100.txt" )

run_one(){
  local key="$1"; local kk="${key%%_*}"; local mode="${key#*_}"
  echo "############ ARM $key (k=$kk mode=$mode) ############"
  bash "$R" "coffee_$key" "${MASK[$kk]}" "$mode" "$CACHE" "$OUT" "$WARM" "$MEAS" 2>&1 | tail -40
}

for a in "$@"; do run_one "$a"; done

echo "======== RESULTS SUMMARY ========"
python3 - "$OUT" <<'PY'
import json,os,sys,glob
out=sys.argv[1]
rows=[]
for d in sorted(glob.glob(os.path.join(out,"coffee_*"))):
    if not os.path.isdir(d): continue
    arm=os.path.basename(d)
    m={}; g={}
    try: m=json.load(open(os.path.join(d,"metrics.json")))
    except: pass
    try: g=json.load(open(os.path.join(d,"grade.json")))
    except: pass
    tok=None
    try: tok=int(open(os.path.join(d,"tokens.txt")).read().strip())
    except: pass
    rows.append(dict(arm=arm, grade=g.get("level"),
        decode_tps=m.get("warm_decode_tps"), chunk_tps=m.get("warm_chunk_tps_median"),
        mib_per_tok=m.get("warm_total_mib_per_token"), dma_mib_per_tok=m.get("warm_dma_mib_per_token"),
        zerocopy_cover=m.get("warm_zerocopy_cover_frac"),
        vram_miss_per_tok=m.get("warm_vram_miss_copies_per_token"),
        hit_rate=(m.get("spex_wholerun") or {}).get("hit_rate"),
        cache_cap=(m.get("cache_cap") or {}).get("effective"),
        resident_gib=m.get("resident_cache_gib"), tokens=tok))
print(json.dumps(rows,indent=2))
json.dump(rows, open(os.path.join(out,"SUMMARY.json"),"w"), indent=2)
PY
echo "wrote $OUT/SUMMARY.json"
