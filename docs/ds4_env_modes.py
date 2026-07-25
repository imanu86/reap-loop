#!/usr/bin/env python3
"""DS4 runtime env constraint model -> enumeration of VALID MODES.

Pure static analysis of the gates read from:
  D:/ds4_work/wt-converge/ds4_cuda.cu
  D:/ds4_work/wt-converge/ds4.c
No runtime is executed. Every constraint carries its file:line provenance.

Backtracking search with early pruning: a constraint is evaluated as soon as
all the variables in its scope are bound.
"""
import sys
from collections import defaultdict

DOM = {
    # --- tiering core
    "TIERING":        ["off", "observe", "enforce"],   # DS4_EXPERT_TIERING
    "POLICY":         ["second-touch", "mass-lfru"],   # DS4_EXPERT_TIER_POLICY
    "COMPOSE":        [0, 1],   # DS4_CUDA_PREFILL_TIER_COMPOSE (0 == UNSET; "0" e' INVALIDO)
    "ROUTER_OPEN":    [0, 1],   # DS4_CUDA_PREFILL_TIER_ROUTER
    "RESERVE_SLOTS":  [0, 1],   # DS4_CUDA_PREFILL_TIER_RESERVE_SLOTS (1 == ">0")
    "GPU_ROUTES":     [0, 1],   # DS4_CUDA_MOE_GPU_RESIDENT_ROUTES
    "G133":           [0, 1],   # DS4_G133_TIER
    "G73_OPEN":       [0, 1],   # DS4_G73_OPEN
    # --- observers
    "PF_MASS_OBS":    [0, 1],   # DS4_CUDA_PREFILL_MASS_OBSERVE
    "PF_MASS_WRAP":   [0, 1],   # DS4_CUDA_PREFILL_MASS_WRAP
    "REAP_MASS_OBS":  [0, 1],   # DS4_CUDA_REAP_MASS_OBSERVE
    "REAP_MASS_WRAP": [0, 1],   # DS4_CUDA_REAP_MASS_WRAP
    # --- Q1_0 arena family
    "Q1_RESIDENT":    [0, 1],   # DS4_Q1_0_RESIDENT_ARENA
    "Q1_DUAL":        [0, 1],   # DS4_Q1_0_DUAL_ARENA
    "Q1_SPARSE_COMP": [0, 1],   # DS4_Q1_0_DUAL_SPARSE_COMPANION
    "Q1_COLD_ONE":    [0, 1],   # DS4_Q1_0_MIXED_COLD_ONE
    "Q1_SNAPSHOT":    [0, 1],   # DS4_Q1_0_SNAPSHOT_BACKING
    "Q1_PAGEABLE":    [0, 1],   # DS4_Q1_0_PAGEABLE_OVERFLOW
    "Q1_PROMOTION":   [0, 1],   # DS4_Q1_0_DYNAMIC_PROMOTION
    "Q1_SIDECAR":     [0, 1],   # DS4_Q1_0_EXPERT_SIDECAR (path set & bound)
    "Q1_SEL_LOAD":    [0, 1],   # DS4_Q1_0_SELECTED_LOAD == "1"
    "Q1_PROBATION":   [0, 1],   # DS4_Q1_0_PROMOTION_PROBATION_SLOTS > 0
    # --- IQ1_S family
    "IQ1_SIDECAR":    [0, 1],   # DS4_IQ1_S_EXPERT_SIDECAR
    "IQ1_COLD_K":     [0, 1],   # DS4_IQ1_S_MIXED_COLD_K == "1"
    "IQ1_PROBATION":  [0, 1],   # DS4_IQ1_PROMOTION_PROBATION_SLOTS > 0
    # --- arenas
    "ARENA_GB":       [0, 1],   # DS4_CUDA_DYNAMIC_ARENA_GB > 0
    "Q1_ARENA_GB":    [0, 1],   # DS4_Q1_0_DYNAMIC_ARENA_GB > 0
    # --- prefill VRAM seed
    "SEED_PER_LAYER": [0, 1],   # DS4_CUDA_PREFILL_VRAM_SEED_PER_LAYER > 0
    "SEED_TOTAL":     [0, 1],   # DS4_CUDA_PREFILL_VRAM_SEED_TOTAL > 0
    "SEED_FLOOR":     [0, 1],   # DS4_CUDA_PREFILL_VRAM_SEED_FLOOR_PER_LAYER > 0
    # --- split transport
    "SPLIT_HIT_MISS": [0, 1],   # DS4_CUDA_MOE_SPLIT_HIT_MISS
    "SPLIT_FUSED":    [0, 1],   # DS4_CUDA_MOE_SPLIT_FUSED
    # --- nested residual
    "NESTED_RES":     [0, 1],   # DS4_NESTED_RESIDUAL_EXACT + sidecar leggibile
}

C = []  # (id, scope, provenance, fn, note)
def rule(cid, scope, where, fn, note):
    C.append((cid, set(scope), where, fn, note))


def Q1_EXCLUSIVE(v):
    """cuda_q1_0_exclusive_arena_active() cu:2427-2431: arena Q1 legata con
    snapshot pubblicata e DUAL_ARENA non richiesto."""
    return (v["Q1_RESIDENT"] or v["Q1_SNAPSHOT"]) and not v["Q1_DUAL"]


# ================= ds4.c : gate di apertura engine (fatali) ==============
rule("C17a", ["Q1_DUAL", "Q1_RESIDENT"], "ds4.c:19544",
     lambda v: not (v["Q1_DUAL"] and not v["Q1_RESIDENT"]),
     "DS4_Q1_0_DUAL_ARENA richiede DS4_Q1_0_RESIDENT_ARENA")
rule("C17b", ["Q1_SPARSE_COMP", "Q1_DUAL", "Q1_RESIDENT", "Q1_COLD_ONE"], "ds4.c:19545-19547",
     lambda v: not (v["Q1_SPARSE_COMP"] and not (v["Q1_DUAL"] and v["Q1_RESIDENT"] and v["Q1_COLD_ONE"])),
     "DUAL_SPARSE_COMPANION richiede DUAL + RESIDENT + MIXED_COLD_ONE")
rule("C17c", ["Q1_COLD_ONE", "Q1_SPARSE_COMP"], "ds4.c:19548",
     lambda v: not (v["Q1_COLD_ONE"] and not v["Q1_SPARSE_COMP"]),
     "MIXED_COLD_ONE richiede DUAL_SPARSE_COMPANION (con C17b: biimplicazione)")
rule("C17d", ["Q1_PAGEABLE", "Q1_SNAPSHOT", "Q1_RESIDENT", "Q1_DUAL"], "ds4.c:19549-19551",
     lambda v: not (v["Q1_PAGEABLE"] and not v["Q1_SNAPSHOT"] and not (v["Q1_RESIDENT"] and v["Q1_DUAL"])),
     "PAGEABLE_OVERFLOW richiede SNAPSHOT_BACKING oppure RESIDENT+DUAL")
rule("C17e", ["Q1_SNAPSHOT", "Q1_RESIDENT", "Q1_DUAL"], "ds4.c:19552-19553",
     lambda v: not (v["Q1_SNAPSHOT"] and (v["Q1_RESIDENT"] or v["Q1_DUAL"])),
     "SNAPSHOT_BACKING esclude RESIDENT_ARENA e DUAL_ARENA")
rule("C17f", ["Q1_PROMOTION", "Q1_RESIDENT", "Q1_DUAL", "Q1_SPARSE_COMP", "Q1_COLD_ONE", "Q1_SNAPSHOT"],
     "ds4.c:19554-19557",
     lambda v: not (v["Q1_PROMOTION"] and not (v["Q1_RESIDENT"] and v["Q1_DUAL"]
                    and not v["Q1_SPARSE_COMP"] and not v["Q1_COLD_ONE"] and not v["Q1_SNAPSHOT"])),
     "DYNAMIC_PROMOTION richiede RESIDENT+DUAL puri (no sparse/cold-one/snapshot)")
rule("C17g", ["Q1_RESIDENT", "Q1_SNAPSHOT", "Q1_SIDECAR", "Q1_SEL_LOAD"], "ds4.c:19558-19560",
     lambda v: not ((v["Q1_RESIDENT"] or v["Q1_SNAPSHOT"]) and not (v["Q1_SIDECAR"] and v["Q1_SEL_LOAD"])),
     "ogni arena Q1_0 richiede sidecar valido + DS4_Q1_0_SELECTED_LOAD=1")
rule("C18a", ["Q1_RESIDENT", "Q1_ARENA_GB", "ARENA_GB"], "ds4.c:20217-20223",
     lambda v: not (v["Q1_RESIDENT"] and not (v["Q1_ARENA_GB"] or v["ARENA_GB"])),
     "RESIDENT richiede DS4_Q1_0_DYNAMIC_ARENA_GB>0 oppure DS4_CUDA_DYNAMIC_ARENA_GB>0")
rule("C18b", ["Q1_SNAPSHOT", "ARENA_GB"], "ds4.c:20226-20231",
     lambda v: not (v["Q1_SNAPSHOT"] and not v["ARENA_GB"]),
     "SNAPSHOT_BACKING richiede DS4_CUDA_DYNAMIC_ARENA_GB>0")
rule("C18c", ["Q1_DUAL", "ARENA_GB"], "ds4.c:20234-20240",
     lambda v: not (v["Q1_DUAL"] and not v["ARENA_GB"]),
     "DUAL_ARENA richiede DS4_CUDA_DYNAMIC_ARENA_GB>0 per lo storage IQ2 esatto")
rule("C19a", ["Q1_SIDECAR", "IQ1_COLD_K"], "ds4.c:19586-19592",
     lambda v: not (v["Q1_SIDECAR"] and v["IQ1_COLD_K"]),
     "sidecar Q1_0 incompatibile con DS4_IQ1_S_MIXED_COLD_K=1")
rule("C19b", ["IQ1_COLD_K", "IQ1_SIDECAR"], "ds4.c:19593-19599",
     lambda v: not (v["IQ1_COLD_K"] and not v["IQ1_SIDECAR"]),
     "DS4_IQ1_S_MIXED_COLD_K=1 richiede un sidecar IQ1_S")
rule("C20a", ["NESTED_RES", "ROUTER_OPEN", "IQ1_SIDECAR", "Q1_SIDECAR"], "ds4.c:19492-19500",
     lambda v: not (v["NESTED_RES"] and (not v["ROUTER_OPEN"] or v["IQ1_SIDECAR"] or v["Q1_SIDECAR"])),
     "nested residual esatto richiede PREFILL_TIER_ROUTER=open e nessun sidecar quant")

# ============ ds4_cuda.cu : cuda_moe_tiering_prepare (cu:26917..) ========
rule("C2", ["Q1_RESIDENT", "Q1_SNAPSHOT", "Q1_DUAL", "TIERING", "COMPOSE",
            "ROUTER_OPEN", "Q1_PROBATION", "IQ1_PROBATION", "RESERVE_SLOTS"], "cu:26935-26942",
     lambda v: not (Q1_EXCLUSIVE(v) and (v["TIERING"] != "off" or v["COMPOSE"] or v["ROUTER_OPEN"]
                    or v["Q1_PROBATION"] or v["IQ1_PROBATION"] or v["RESERVE_SLOTS"])),
     "arena Q1 ESCLUSIVA => tiering COMPLETAMENTE off (mixed-host-resolver-not-implemented)")
rule("C3", ["ROUTER_OPEN", "COMPOSE"], "cu:26957-26961",
     lambda v: not (v["ROUTER_OPEN"] and not v["COMPOSE"]),
     "router aperto richiede composed tiering")
rule("C4", ["G73_OPEN", "G133", "TIERING", "COMPOSE", "ROUTER_OPEN", "RESERVE_SLOTS",
            "GPU_ROUTES", "Q1_COLD_ONE", "IQ1_COLD_K", "Q1_SNAPSHOT"], "cu:26963-26975",
     lambda v: not (v["G73_OPEN"] and not (v["G133"] and v["TIERING"] == "enforce" and v["COMPOSE"]
                    and v["ROUTER_OPEN"] and v["RESERVE_SLOTS"] and v["GPU_ROUTES"]
                    and not v["Q1_COLD_ONE"] and not v["IQ1_COLD_K"] and not v["Q1_SNAPSHOT"])),
     "G73_OPEN=1 richiede G133 + enforce + compose + router open + reserve>0 + gpu routes, Q1/IQ1 serving off")
rule("C4b", ["G73_OPEN", "Q1_RESIDENT", "Q1_DUAL", "Q1_SPARSE_COMP", "Q1_SNAPSHOT",
             "Q1_PAGEABLE", "Q1_PROMOTION", "Q1_PROBATION", "IQ1_PROBATION",
             "IQ1_SIDECAR", "IQ1_COLD_K"], "cu:1478-1512",
     lambda v: not (v["G73_OPEN"] and (v["Q1_RESIDENT"] or v["Q1_DUAL"] or v["Q1_SPARSE_COMP"]
                    or v["Q1_SNAPSHOT"] or v["Q1_PAGEABLE"] or v["Q1_PROMOTION"]
                    or v["Q1_PROBATION"] or v["IQ1_PROBATION"] or v["IQ1_SIDECAR"] or v["IQ1_COLD_K"])),
     "ambiente ermetico G73: rifiuta ogni var Q1/IQ1 di arena/promotion/serving ereditata")
rule("C5", ["Q1_PROMOTION", "IQ1_PROBATION", "TIERING", "COMPOSE", "ROUTER_OPEN",
            "RESERVE_SLOTS", "IQ1_SIDECAR", "IQ1_COLD_K"], "cu:26977-26986",
     lambda v: not ((not v["Q1_PROMOTION"]) and v["IQ1_PROBATION"] and not (
                    v["TIERING"] == "enforce" and v["COMPOSE"] and v["ROUTER_OPEN"]
                    and v["RESERVE_SLOTS"] and v["IQ1_SIDECAR"] and v["IQ1_COLD_K"])),
     "probation IQ1 richiede mixed IQ1_S + enforce/compose/router open + reserve==probation + sidecar legato")
rule("C6", ["Q1_PROMOTION", "Q1_PROBATION", "TIERING", "COMPOSE", "ROUTER_OPEN",
            "RESERVE_SLOTS", "Q1_SIDECAR", "Q1_DUAL", "Q1_SNAPSHOT"], "cu:26987-26999",
     lambda v: not (v["Q1_PROMOTION"] and not (v["Q1_PROBATION"] and v["TIERING"] == "enforce"
                    and v["COMPOSE"] and v["ROUTER_OPEN"] and v["RESERVE_SLOTS"]
                    and v["Q1_SIDECAR"] and v["Q1_DUAL"] and not v["Q1_SNAPSHOT"])),
     "promotion Q1_0 richiede dual-arena residente + enforce/compose/router open + reserve==probation")
rule("C7a", ["Q1_PROMOTION", "Q1_PROBATION", "IQ1_PROBATION"], "cu:26155-26163",
     lambda v: not (v["Q1_PROMOTION"] and (not v["Q1_PROBATION"] or v["IQ1_PROBATION"])),
     "promotion Q1_0 richiede Q1_PROBATION>0 e IQ1_PROBATION==0")
rule("C7b", ["Q1_PROMOTION", "Q1_PROBATION"], "cu:26165-26171",
     lambda v: not ((not v["Q1_PROMOTION"]) and v["Q1_PROBATION"]),
     "DS4_Q1_0_PROMOTION_PROBATION_SLOTS richiede DS4_Q1_0_DYNAMIC_PROMOTION=1")
rule("C8a", ["TIERING", "G133"], "cu:27002-27006",
     lambda v: not (v["TIERING"] == "off" and v["G133"]),
     "DS4_G133_TIER=1 richiede DS4_EXPERT_TIERING=enforce")
rule("C8b", ["TIERING", "COMPOSE"], "cu:27007-27011",
     lambda v: not (v["TIERING"] == "off" and v["COMPOSE"]),
     "compose richiede expert tiering enforce")
rule("C9", ["G133", "TIERING", "POLICY"], "cu:27085-27091",
     lambda v: not (v["G133"] and not (v["TIERING"] == "enforce" and v["POLICY"] == "mass-lfru")),
     "G133 richiede enforce + mass-lfru")
rule("C10", ["TIERING", "ARENA_GB"], "cu:27175-27182",
     lambda v: not (v["TIERING"] == "enforce" and not v["ARENA_GB"]),
     "enforce richiede un'arena dinamica pinnata idle => DS4_CUDA_DYNAMIC_ARENA_GB>0")
rule("C11", ["TIERING", "COMPOSE", "POLICY", "PF_MASS_OBS", "PF_MASS_WRAP",
             "REAP_MASS_OBS", "REAP_MASS_WRAP"], "cu:27184-27199",
     lambda v: not (v["TIERING"] == "enforce" and v["COMPOSE"] and not (
                    v["POLICY"] == "mass-lfru" and v["PF_MASS_OBS"] and v["PF_MASS_WRAP"]
                    and not v["REAP_MASS_OBS"] and not v["REAP_MASS_WRAP"])),
     "compose richiede snapshot prefill finalizzato+pubblicato (PREFILL_MASS_OBSERVE+WRAP), mass-lfru, e NESSUN observer REAP")
rule("C12", ["TIERING", "COMPOSE", "PF_MASS_OBS", "REAP_MASS_OBS"], "cu:27200-27206",
     lambda v: not (v["TIERING"] == "enforce" and not v["COMPOSE"] and (
                    v["PF_MASS_OBS"] or v["REAP_MASS_OBS"])),
     "enforce SENZA compose richiede arena idle: esclude ogni observer prefill/REAP")
rule("C13", ["TIERING", "GPU_ROUTES"], "cu:31192-31201",
     lambda v: not (v["TIERING"] != "off" and not v["GPU_ROUTES"]),
     "qualunque tiering != off richiede DS4_CUDA_MOE_GPU_RESIDENT_ROUTES")
rule("C14", ["SEED_PER_LAYER", "SEED_TOTAL", "TIERING", "COMPOSE", "GPU_ROUTES"], "cu:31203-31210",
     lambda v: not ((v["SEED_PER_LAYER"] or v["SEED_TOTAL"]) and not (
                    v["TIERING"] == "enforce" and v["COMPOSE"] and v["GPU_ROUTES"])),
     "prefill VRAM seed richiede composed enforce tiering + GPU routes")
rule("C15a", ["SEED_PER_LAYER", "SEED_TOTAL"], "cu:31211-31213",
     lambda v: not (v["SEED_PER_LAYER"] and v["SEED_TOTAL"]),
     "SEED_PER_LAYER e SEED_TOTAL mutuamente esclusivi")
rule("C15b", ["SEED_FLOOR", "SEED_TOTAL"], "cu:31214-31215",
     lambda v: not (v["SEED_FLOOR"] and not v["SEED_TOTAL"]),
     "SEED_FLOOR_PER_LAYER richiede SEED_TOTAL>0")
rule("C16a", ["SPLIT_HIT_MISS", "SPLIT_FUSED", "G73_OPEN"], "cu:37919-37923",
     lambda v: not (v["SPLIT_HIT_MISS"] and v["SPLIT_FUSED"] and not v["G73_OPEN"]),
     "split hit/miss e split fused mutuamente esclusivi (G73 forza hit/miss a 0, cu:37912)")
rule("C20b", ["NESTED_RES", "COMPOSE", "ROUTER_OPEN"], "cu:13093-13100",
     lambda v: not (v["NESTED_RES"] and v["COMPOSE"] and not v["ROUTER_OPEN"]),
     "nested residual attivo + compose richiede router aperto")
rule("C21a", ["PF_MASS_WRAP", "PF_MASS_OBS"], "cu:12740-12758",
     lambda v: not (v["PF_MASS_WRAP"] and not v["PF_MASS_OBS"]),
     "PREFILL_MASS_WRAP senza PREFILL_MASS_OBSERVE non ha snapshot da pubblicare")

SCOPE_BY_RULE = {cid: sc for cid, sc, _, _, _ in C}


def violations(v):
    out = []
    for cid, scope, where, fn, note in C:
        if scope <= set(v) and not fn(v):
            out.append((cid, where, note))
    return out


# ---------------------------------------------------------------- search
keys = list(DOM)
# order variables so that heavily constrained ones bind early
order = sorted(keys, key=lambda k: -sum(1 for _, sc, _, _, _ in C if k in sc))
rules_ready = defaultdict(list)   # after binding var i, which rules become checkable
bound_so_far = set()
for i, k in enumerate(order):
    bound_so_far.add(k)
    for cid, sc, where, fn, note in C:
        if sc <= bound_so_far and not (sc <= (bound_so_far - {k})):
            rules_ready[i].append((cid, where, fn, note))

solutions = []
def dfs(i, v):
    if i == len(order):
        solutions.append(dict(v))
        return
    k = order[i]
    for val in DOM[k]:
        v[k] = val
        ok = True
        for cid, where, fn, note in rules_ready[i]:
            if not fn(v):
                ok = False
                break
        if ok:
            dfs(i + 1, v)
    v.pop(k, None)

sys.setrecursionlimit(10000)
dfs(0, {})

tot = 1
for k in keys:
    tot *= len(DOM[k])
print(f"spazio grezzo        : {tot:,} assegnazioni su {len(keys)} variabili di gate")
print(f"configurazioni VALIDE: {len(solutions):,}")
print(f"riduzione            : {tot/max(len(solutions),1):,.0f}x")
print(f"vincoli codificati   : {len(C)}\n")

# ------------------------------------------- collapse into named modes
DEFINING = ["G73_OPEN", "TIERING", "COMPOSE", "ROUTER_OPEN", "G133",
            "Q1_RESIDENT", "Q1_DUAL", "Q1_SNAPSHOT", "Q1_PROMOTION",
            "IQ1_COLD_K", "NESTED_RES"]

groups = defaultdict(list)
for v in solutions:
    groups[tuple(v[k] for k in DEFINING)].append(v)

print(f"MODI distinti (su {len(DEFINING)} variabili portanti): {len(groups)}\n")
print("=" * 78)

def label(d):
    t = []
    if d["G73_OPEN"]: t.append("G73-OPEN")
    if d["Q1_PROMOTION"]: t.append("Q1-promotion")
    elif d["Q1_RESIDENT"] and d["Q1_DUAL"]: t.append("Q1-dual-arena")
    elif d["Q1_RESIDENT"]: t.append("Q1-resident-ESCLUSIVA")
    elif d["Q1_SNAPSHOT"]: t.append("Q1-snapshot-ESCLUSIVA")
    if d["IQ1_COLD_K"]: t.append("IQ1-mixed")
    if d["NESTED_RES"]: t.append("nested-residual")
    if d["TIERING"] == "enforce" and d["COMPOSE"] and d["ROUTER_OPEN"]:
        t.append("enforce+compose+router-open")
    elif d["TIERING"] == "enforce" and d["COMPOSE"]:
        t.append("enforce+compose(router closed)")
    elif d["TIERING"] == "enforce": t.append("enforce-nudo")
    elif d["TIERING"] == "observe": t.append("observe")
    else: t.append("tiering-OFF")
    if d["G133"]: t.append("G133")
    return " / ".join(t)

rows = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
for i, (sig, bucket) in enumerate(rows, 1):
    d = dict(zip(DEFINING, sig))
    free = [k for k in keys if len({m[k] for m in bucket}) > 1]
    pin = {k: next(iter({m[k] for m in bucket})) for k in keys
           if len({m[k] for m in bucket}) == 1}
    print(f"\nM{i:02d}  {label(d)}   [{len(bucket)} varianti]")
    print("     portanti : " + "  ".join(f"{k}={d[k]}" for k in DEFINING if d[k] not in (0, 'off')) or "(tutto off)")
    print("     libere   : " + (", ".join(free) if free else "(nessuna)"))

# --------------------------------------------------------- anchors
print("\n" + "=" * 78)
print("VERIFICA DEGLI ANCORAGGI")
print("=" * 78)

def base(**kw):
    v = {k: (0 if DOM[k] == [0, 1] else DOM[k][0]) for k in keys}
    v.update(kw)
    return v

anchors = {
    "A  G73_OPEN costellazione completa": base(
        G73_OPEN=1, G133=1, TIERING="enforce", POLICY="mass-lfru", COMPOSE=1,
        ROUTER_OPEN=1, RESERVE_SLOTS=1, GPU_ROUTES=1, ARENA_GB=1,
        PF_MASS_OBS=1, PF_MASS_WRAP=1, SPLIT_FUSED=1),
    "B  IQ2 puro enforce+compose+router open (no G73)": base(
        TIERING="enforce", POLICY="mass-lfru", COMPOSE=1, ROUTER_OPEN=1,
        RESERVE_SLOTS=1, GPU_ROUTES=1, ARENA_GB=1, PF_MASS_OBS=1, PF_MASS_WRAP=1),
    "C  full-Q1 residente esclusivo, tiering off, seed off": base(
        Q1_RESIDENT=1, Q1_SIDECAR=1, Q1_SEL_LOAD=1, Q1_ARENA_GB=1, TIERING="off"),
}
rejected = {
    "R1 Q1_PROMOTION isolato (SSD-wrap senza costellazione)": base(Q1_PROMOTION=1),
    "R2 RESIDENT_ARENA=1 da solo con tiering enforce": base(
        Q1_RESIDENT=1, Q1_SIDECAR=1, Q1_SEL_LOAD=1, Q1_ARENA_GB=1, ARENA_GB=1,
        TIERING="enforce", GPU_ROUTES=1),
    "R3 COMPOSE senza enforce": base(COMPOSE=1, GPU_ROUTES=1),
    "R4 G133 senza enforce": base(G133=1, GPU_ROUTES=1),
    "R5 G73_OPEN con sidecar Q1 residente": base(
        G73_OPEN=1, G133=1, TIERING="enforce", POLICY="mass-lfru", COMPOSE=1,
        ROUTER_OPEN=1, RESERVE_SLOTS=1, GPU_ROUTES=1, ARENA_GB=1,
        PF_MASS_OBS=1, PF_MASS_WRAP=1, Q1_RESIDENT=1, Q1_SIDECAR=1,
        Q1_SEL_LOAD=1, Q1_ARENA_GB=1),
    "R6 compose + observer REAP insieme": base(
        TIERING="enforce", POLICY="mass-lfru", COMPOSE=1, ROUTER_OPEN=1,
        RESERVE_SLOTS=1, GPU_ROUTES=1, ARENA_GB=1, PF_MASS_OBS=1,
        PF_MASS_WRAP=1, REAP_MASS_OBS=1),
    "R7 seed VRAM con tiering off": base(SEED_TOTAL=1),
}

for name, v in anchors.items():
    vio = violations(v)
    print(f"\n[{'OK-valida' if not vio else 'ROTTA'}] {name}")
    for cid, where, note in vio:
        print(f"        viola {cid} ({where}): {note}")
for name, v in rejected.items():
    vio = violations(v)
    print(f"\n[{'OK-rifiutata' if vio else '!!! FALSO POSITIVO !!!'}] {name}")
    for cid, where, note in vio:
        print(f"        {cid} ({where}): {note}")

# ================= collasso sui MODI EFFETTIVI =========================
# Una variabile "portante" che il codice accetta ma non consuma non
# distingue un modo di servizio.  Regole di inerzia lette dal codice:
#   COMPOSE      consumato solo se TIERING==enforce   (cu:27184)
#   ROUTER_OPEN  consumato solo se COMPOSE consumato  (cu:26957 + cu:13086)
#   G133         consumato solo se TIERING==enforce   (cu:27002/27085)
print("\n" + "=" * 78)
print("MODI EFFETTIVI (collassando le variabili accettate ma NON consumate)")
print("=" * 78)

def effective(d):
    e = dict(d)
    if e["TIERING"] != "enforce":
        e["COMPOSE"] = 0
        e["ROUTER_OPEN"] = 0
        e["G133"] = 0
    elif not e["COMPOSE"]:
        e["ROUTER_OPEN"] = 0
    return tuple(e[k] for k in DEFINING)

eff = defaultdict(int)
for sig, bucket in groups.items():
    d = dict(zip(DEFINING, sig))
    eff[effective(d)] += len(bucket)

print(f"MODI EFFETTIVI: {len(eff)}  (da {len(groups)} nominali)\n")
for i, (sig, n) in enumerate(sorted(eff.items(), key=lambda kv: -kv[1]), 1):
    d = dict(zip(DEFINING, sig))
    print(f"E{i:02d}  {label(d):62s} [{n} config]")
