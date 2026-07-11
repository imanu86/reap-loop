#!/usr/bin/env python3
"""BOOT-PROBE — hardware-adaptive calibration for REAP-LOOP (P2 portability core).

At first boot on ANY host, this ~60-90 s probe MEASURES the local hardware
constants instead of hard-coding them, and writes a HW profile (JSON) that the
PACE controller consumes (``DS4_PACE_AUTO=1``). It is the concrete
implementation of ``docs/SOTA_ROADMAP.md`` §Auto-calibrazione (principle P2) and
``docs/BOOT_PROBE_DESIGN.md``.

The cardinal rule (P2): **no absolute hardware number is ever hard-coded.**
Every field is tagged with a provenance:

    measured   — read from THIS host now (nvidia-smi, dd, /proc, ds4)
    derived    — computed from measured values + geometry
    invariant  — a portable P2 constant, offline-calibrated, HW-independent
                 (LRU hit curve, cov90 K-floor, knee table) — transfers across HW
    estimated  — fallback model when a measurement could not be taken
    stub        — declared not-measured on this run

The four probes (roadmap table):
    (a) VRAM free + per-expert footprint  -> available expert-cache slots (dtype-aware)
    (b) SSD->RAM->VRAM bandwidth          -> DIMENSIONLESS regime ratio -> WRAP on/off
    (c) t/s baseline on ~32 tok           -> auto-normalized speed constant (once, at boot)
    (d) coverage floor                    -> NOT a boot measurement: read from offline E-CAL

Output: HW profile + the DERIVED control constants (cache slots, WRAP on/off,
K* initial). PACE reads them; it never reads an absolute t/s at runtime (P2).

Reproduce (on the target host, e.g. WSL Ubuntu):
    python3 scripts/boot_probe.py \
        --model /root/models/ds4-2bit.gguf \
        --ds4-bin /root/ds4/ds4 \
        --out runs/ds4/<date>_bootprobe/profile.json

    # HW-only, no ds4 decode run (baseline t/s estimated from bandwidth):
    python3 scripts/boot_probe.py --model /root/models/ds4-2bit.gguf --no-ds4

    # Emit the launch env/flags PACE would consume from a profile:
    python3 scripts/boot_probe.py --profile profile.json --emit-launch
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

# The probe lives in scripts/ next to the canonical gguf parser; reuse it so the
# per-expert footprint is dtype-aware and matches the engine's own layout math.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_inspect_ds4 as gg  # noqa: E402

GIB = 1024 ** 3
MIB = 1024 ** 2

# ===========================================================================
# INVARIANTS (P2) — offline-calibrated, portable across hardware.
# These are NOT hardware numbers: they are dimensionless / unit-invariant
# quantities that transfer to any host. Each carries its source.
# ===========================================================================

# LRU expert-cache hit-rate curve vs cache capacity (slots). Portable P2
# invariant of the ROUTING (working-set locality), independent of the GPU.
# Source: E-LAT §4 + J17/J31 LRU sim (runs/ds4/20260710_elat_tier_latency/REPORT.md).
LRU_HIT_CURVE = [
    (128, 0.20),   # below the 258 floor -> thrash (interp anchor)
    (258, 0.34),
    (407, 0.50),   # 407 = the 3060 VRAM slot maximum used as an interp knot (not a target)
    (512, 0.59),
    (1024, 0.74),
    (2048, 0.81),
]

# cov90 anti-under-provisioning floor. E-CAL verdict (NEGATIVE): coverage does
# NOT separate collapse, so it is used ONLY as a floor so we never
# under-provision K. Kmin-cov90 is task-INVARIANT (~38 across html + 11 coding
# prompts). Source: runs/ds4/20260710_ecal_coverage_threshold/REPORT.md.
COV_TARGET = 0.90
COV90_K_FLOOR = 38

# Decision-model knee table (identity-width class -> sizing target). Invariant
# ordering; K* is refined at runtime by the width sensor after ~150 tok.
# Source: docs/DECISION_MODEL.md §3.
KNEE = {"narrow": 20, "medium": 32, "wide": 48}


def hit_rate(slots: int) -> float:
    """Piecewise-linear interpolation of the invariant LRU hit curve."""
    pts = LRU_HIT_CURVE
    if slots <= pts[0][0]:
        return pts[0][1]
    if slots >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= slots <= x1:
            t = (slots - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


# ===========================================================================
# (a) GPU / VRAM + per-expert footprint
# ===========================================================================
def probe_gpu(nvidia_smi: str) -> dict:
    """Measured VRAM via nvidia-smi (works in WSL and on Windows)."""
    out = {"provenance": "measured", "tool": nvidia_smi}
    exe = shutil.which(nvidia_smi) or nvidia_smi
    try:
        raw = subprocess.check_output(
            [exe, "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=30,
        ).strip().splitlines()[0]
        name, total, free, used = [p.strip() for p in raw.split(",")]
        out.update({
            "name": name,
            "vram_total_bytes": int(float(total)) * MIB,
            "vram_free_bytes": int(float(free)) * MIB,
            "vram_used_bytes": int(float(used)) * MIB,
        })
    except Exception as exc:  # noqa: BLE001
        out.update({"provenance": "stub", "error": f"{type(exc).__name__}: {exc}",
                    "vram_total_bytes": None, "vram_free_bytes": None})
    return out


def probe_model(model_path: str) -> dict:
    """Measured per-expert footprint + geometry, straight from the GGUF header.

    Dtype-aware (roadmap: params/expert x bytes/dtype): we sum the quantized
    byte-size of one expert's {gate, up, down} tensors, so a different quant or a
    different model self-measures its own footprint.
    """
    g = gg.parse_gguf(model_path)
    layers = gg.discover_expert_layers(g)
    n_moe_layers = len(layers)
    any_layer = next(iter(layers.values()))
    footprint = sum(gg.routed_expert_nbytes(t) for t in any_layer.values())
    n_experts = min(v.dims[2] for v in any_layer.values())

    expert_tensors = [t for t in g.tensors if gg.EXPERT_RE.match(t.name)]
    expert_bytes = sum(t.nbytes for t in expert_tensors)
    total_bytes = sum(t.nbytes for t in g.tensors)
    nonexpert_bytes = total_bytes - expert_bytes

    top_k = int(g.kv.get("deepseek4.expert_used_count")
                or g.kv.get("general.expert_used_count") or 6)
    shared = int(g.kv.get("deepseek4.expert_shared_count") or 0)
    n_recalls = n_moe_layers * top_k

    return {
        "provenance": "measured",
        "path": model_path,
        "quant_gate": any_layer.get("gate").type_name if any_layer.get("gate") else None,
        "quant_down": any_layer.get("down").type_name if any_layer.get("down") else None,
        "footprint_per_expert_bytes": footprint,
        "footprint_per_expert_mib": round(footprint / MIB, 4),
        "n_moe_layers": n_moe_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "shared_experts_per_layer": shared,
        "recalls_per_token": n_recalls,           # 43 * 6 = 258 on DS4 Flash
        "decode_demand_bytes_per_token": n_recalls * footprint,
        "expert_bytes": expert_bytes,
        "nonexpert_resident_bytes": nonexpert_bytes,
        "total_bytes": total_bytes,
    }


def probe_ram() -> dict:
    """Measured host RAM (Linux/WSL /proc/meminfo, else psutil-free fallback)."""
    out = {"provenance": "measured", "source": "/proc/meminfo"}
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                info[k.strip()] = int(rest.strip().split()[0]) * 1024  # kB -> bytes
        out["ram_total_bytes"] = info.get("MemTotal")
        out["ram_available_bytes"] = info.get("MemAvailable", info.get("MemFree"))
    except Exception as exc:  # noqa: BLE001
        out.update({"provenance": "stub", "error": f"{type(exc).__name__}: {exc}",
                    "ram_total_bytes": None, "ram_available_bytes": None})
    return out


# ===========================================================================
# (b) SSD -> RAM -> VRAM bandwidth (the offload path)
# ===========================================================================
def _read_chunk(path: str, size_bytes: int, offset_bytes: int) -> float:
    """Read size_bytes from path at offset; return seconds elapsed."""
    bs = 8 * MIB
    n = max(1, size_bytes // bs)
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as fh:
        fh.seek(offset_bytes)
        got = 0
        for _ in range(n):
            b = fh.read(bs)
            if not b:
                break
            got += len(b)
    return time.perf_counter() - t0


def _drop_caches() -> bool:
    """Best-effort page-cache drop (needs root). Returns True on success."""
    try:
        subprocess.run(["sync"], timeout=30, check=False)
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("3\n")
        return True
    except Exception:  # noqa: BLE001
        return False


def probe_bandwidth(model_path: str, expert_bytes_start: int,
                    sample_gib: float = 2.0, threads: int = 4,
                    drop_caches: bool = True) -> dict:
    """Measure SSD-cold (single + threaded) and RAM-warm read bandwidth.

    The offload path the engine actually uses (WRAP) is THREADED bulk page-in, so
    we measure both single-stream (worst-case sync) and threaded (WRAP-equivalent)
    cold reads, plus the RAM page-cache read. All GiB/s; only RATIOS are used
    downstream (never an absolute MB/s), so a faster NVMe / more VRAM simply
    lands on the safe side of the dimensionless boundary.
    """
    out = {"provenance": "measured", "sample_gib": sample_gib, "threads": threads}
    size = int(sample_gib * GIB)
    # Read from inside the expert-data region so we exercise real expert bytes,
    # not the header/nonexpert prefix.
    base = max(expert_bytes_start, 4 * GIB)
    fsize = os.path.getsize(model_path)
    base = min(base, max(0, fsize - size - threads * size))

    dropped = _drop_caches() if drop_caches else False
    out["page_cache_dropped"] = dropped
    if not drop_caches:
        out["drop_caches_skipped"] = True

    # Cold single-stream (the un-prefetched, sync-in-decode worst case).
    dt_single = _read_chunk(model_path, size, base)
    b_ssd_single = (size / GIB) / dt_single if dt_single > 0 else None

    # Cold threaded (the WRAP bulk page-in path). Distinct offsets so streams do
    # not alias in cache.
    _drop_caches()
    procs = []
    t0 = time.perf_counter()
    for i in range(threads):
        off = base + i * size
        # Spawn dd-equivalent via a subprocess of THIS interpreter to stay portable.
        procs.append(subprocess.Popen(
            [sys.executable, "-c",
             "import sys;f=open(sys.argv[1],'rb',buffering=0);f.seek(int(sys.argv[2]));"
             "n=int(sys.argv[3]);\n"
             "r=0\n"
             "while r<n:\n b=f.read(8*1024*1024)\n"
             " if not b:break\n r+=len(b)",
             model_path, str(off), str(size)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    dt_threaded = time.perf_counter() - t0
    b_ssd_threaded = (threads * size / GIB) / dt_threaded if dt_threaded > 0 else None

    # RAM warm (second touch of the single-stream region -> page cache).
    dt_warm = _read_chunk(model_path, size, base)
    b_ram = (size / GIB) / dt_warm if dt_warm > 0 else None

    out.update({
        "ssd_cold_single_gibs": round(b_ssd_single, 4) if b_ssd_single else None,
        "ssd_cold_threaded_gibs": round(b_ssd_threaded, 4) if b_ssd_threaded else None,
        "ram_warm_gibs": round(b_ram, 4) if b_ram else None,
    })
    if not dropped:
        out["warning"] = ("page cache NOT dropped (no root): ssd_cold numbers may "
                          "be page-cache-warm and over-optimistic")
    return out


# ===========================================================================
# (c) baseline t/s on ~32 tokens (measured via ds4, else estimated)
# ===========================================================================
def _ds4_run(ds4_bin: str, model_path: str, cache_slots: int, ctx: int,
             tokens: int, timeout: int):
    """One ds4 -n TOKENS run; return (generation_tps, prefill_tps, wall_s, raw)."""
    import re
    ds4_dir = os.path.dirname(os.path.abspath(ds4_bin))
    prompt = "Write a short technical note about MoE expert cache locality."
    cmd = (f"cd {ds4_dir} && export DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 && "
           f"{ds4_bin} -m {model_path} --cuda --ssd-streaming "
           f"--ssd-streaming-cache-experts {cache_slots} -c {ctx} -n {tokens} "
           f"--temp 0 --seed 1 -p {json.dumps(prompt)} 2>&1")
    t0 = time.perf_counter()
    raw = subprocess.check_output(["bash", "-lc", cmd], text=True, timeout=timeout)
    wall = round(time.perf_counter() - t0, 2)
    m = re.search(r"prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s", raw)
    if not m:
        return None, None, wall, raw
    return float(m.group(2)), float(m.group(1)), wall, raw


def probe_baseline_tps(ds4_bin: str, model_path: str, cache_slots: int,
                       ctx: int, tokens: int, timeout: int, warm_first: bool = True) -> dict:
    """Run ds4 and parse generation t/s. WARM-FIRST discipline (PACE_DESIGN §7):
    a cold decode stalls while the expert cache populates, so we run once to warm
    (short) and DISCARD it, then measure the warm run. The number is minted ONCE
    here and never re-consumed by the controller at runtime (P2).
    """
    out = {"provenance": "measured", "tokens": tokens, "cache_slots": cache_slots,
           "warm_first": warm_first}
    try:
        if warm_first:
            cold_tps, _, cold_wall, _ = _ds4_run(ds4_bin, model_path, cache_slots, ctx,
                                                 max(8, tokens // 2), timeout)
            out["cold_discarded_tps"] = cold_tps
            out["cold_wall_s"] = cold_wall
        gen, pre, wall, raw = _ds4_run(ds4_bin, model_path, cache_slots, ctx, tokens, timeout)
        out["wall_s"] = wall
        if gen is not None:
            out["prefill_tps"] = pre
            out["baseline_tps"] = gen
        else:
            out.update({"provenance": "stub",
                        "error": "could not parse 'generation: N t/s' from ds4 output"})
    except Exception as exc:  # noqa: BLE001
        out.update({"provenance": "stub", "error": f"{type(exc).__name__}: {exc}"})
    return out


def estimate_baseline_tps(model: dict, bw: dict, cache_slots: int) -> dict:
    """Copy-bound UPPER-BOUND estimate when ds4 is not run: token time is at
    least the H2D copy of the miss experts. We proxy H2D bandwidth with the
    measured RAM read bandwidth (optimistic: real PCIe H2D is slower), and ignore
    pure compute, so tps here is a CEILING to be replaced by probe (c)."""
    b_ram = bw.get("ram_warm_gibs")
    miss = 1.0 - hit_rate(cache_slots)
    miss_bytes = model["recalls_per_token"] * model["footprint_per_expert_bytes"] * miss
    if not b_ram:
        return {"provenance": "stub", "error": "no ram bandwidth to estimate from"}
    t_copy_s = (miss_bytes / GIB) / b_ram
    tps_ceiling = 1.0 / t_copy_s if t_copy_s > 0 else None
    return {
        "provenance": "estimated",
        "model": "copy-bound ceiling (RAM-BW proxy for H2D, compute ignored)",
        "miss_fraction": round(miss, 4),
        "miss_bytes_per_token": int(miss_bytes),
        "baseline_tps_ceiling": round(tps_ceiling, 3) if tps_ceiling else None,
        "note": "replace with a real ds4 -n 32 run on first boot (probe c)",
    }


# ===========================================================================
# Derivation: turn measurements into the control constants PACE consumes.
# ===========================================================================
def derive(gpu: dict, model: dict, ram: dict, bw: dict, base: dict,
           ctx: int, reserve_frac: float, task_class: str) -> dict:
    d = {}
    footprint = model["footprint_per_expert_bytes"]
    vram_total = gpu.get("vram_total_bytes")
    nonexpert = model["nonexpert_resident_bytes"]

    # --- (a) expert-cache slots, dtype-aware ---------------------------------
    # slot = floor(usable_vram / footprint). usable_vram = total - non-expert
    # resident weights - a reserve headroom (KV + activations + fragmentation).
    # reserve_frac is a "do-not-OOM" safety margin, NOT a performance constant.
    if vram_total:
        reserve_bytes = int(vram_total * reserve_frac)
        usable = vram_total - nonexpert - reserve_bytes
        slots = max(0, int(usable // footprint))
        d["cache"] = {
            "provenance": "derived",
            "formula": "floor((vram_total*(1-reserve_frac) - nonexpert_resident) / footprint)",
            "reserve_frac": reserve_frac,
            "reserve_bytes": reserve_bytes,
            "usable_vram_bytes": max(0, usable),
            "cache_slots": slots,
            "cache_slots_gib": round(slots * footprint / GIB, 3),
        }
        # Cross-check against measured free VRAM if the model happened to be loaded.
        free = gpu.get("vram_free_bytes")
        if free is not None:
            d["cache"]["free_vram_now_bytes"] = free
    else:
        slots = 258  # last-resort floor only if VRAM unreadable; flagged.
        d["cache"] = {"provenance": "stub", "cache_slots": slots,
                      "error": "vram_total unreadable; slots is a stub floor"}

    # --- (d) initial K* floor (offline E-CAL invariant, refined at runtime) ---
    knee = KNEE.get(task_class, KNEE["wide"])
    k_initial = max(knee, COV90_K_FLOOR)  # anti-under-provisioning floor
    d["k_initial"] = {
        "provenance": "invariant",
        "source": "E-CAL cov90 floor + decision-model knee (docs/DECISION_MODEL.md)",
        "cov90_k_floor": COV90_K_FLOOR,
        "knee_for_class": knee,
        "task_class_assumed": task_class,
        "k_initial": k_initial,
        "note": ("HW-agnostic starting provision; the runtime width sensor (>=150 tok) "
                 "refines K*, and a proven rewind airbag can tighten toward K12-16"),
    }

    # --- (b) offload regime -> WRAP on/off (DIMENSIONLESS) --------------------
    # The discriminator is whether the ACTIVE working set fits RAM page cache.
    # If it fits, decode-path misses are served from RAM (copy-bound) -> WRAP is
    # pure overhead (measured: prefetch SLOWS the 3060 practical config). If it
    # spills, misses hit SSD sync (50-230 ms cliff) -> WRAP's threaded prefetch
    # converts them to RAM -> WRAP helps. A confirming SSD throughput ratio
    # reports how badly SSD is the bottleneck when spilling. All ratios pure.
    wss_bytes = k_initial * model["n_moe_layers"] * footprint
    ram_avail = ram.get("ram_available_bytes")
    tps = base.get("baseline_tps") or base.get("baseline_tps_ceiling")
    b_ssd_thr = bw.get("ssd_cold_threaded_gibs")
    b_ssd_single = bw.get("ssd_cold_single_gibs")

    ram_fit_ratio = (ram_avail / wss_bytes) if (ram_avail and wss_bytes) else None
    working_set_fits_ram = bool(ram_fit_ratio and ram_fit_ratio >= 1.0)

    # SSD throughput ratio: measured cold SSD bandwidth vs the decode's SSD
    # byte-demand rate (miss bytes/token * t/s). < 1 means SSD cannot feed decode.
    ssd_ratio = None
    demand_bw = None
    if tps is not None and vram_total:
        miss = 1.0 - hit_rate(slots)
        miss_bytes = model["recalls_per_token"] * footprint * miss
        demand_bw = miss_bytes * tps / GIB  # GiB/s the decode would pull from SSD
        if b_ssd_thr and demand_bw > 0:
            ssd_ratio = b_ssd_thr / demand_bw

    deeply_ssd_bound = (not working_set_fits_ram) and bool(ssd_ratio and ssd_ratio < 1.0)
    # Prefetch speedup WRAP would buy (threaded vs sync single) — informational.
    prefetch_speedup = (b_ssd_thr / b_ssd_single) if (b_ssd_thr and b_ssd_single) else None

    d["offload_regime"] = {
        "provenance": "derived",
        "working_set_bytes_at_k_initial": wss_bytes,
        "working_set_gib": round(wss_bytes / GIB, 3),
        "ram_fit_ratio": round(ram_fit_ratio, 3) if ram_fit_ratio else None,
        "working_set_fits_ram": working_set_fits_ram,
        "decode_ssd_demand_gibs": round(demand_bw, 4) if demand_bw else None,
        "ssd_throughput_ratio": round(ssd_ratio, 4) if ssd_ratio else None,
        "ssd_throughput_ratio_lt_1": bool(ssd_ratio and ssd_ratio < 1.0),
        "prefetch_speedup_threaded_over_single": round(prefetch_speedup, 2) if prefetch_speedup else None,
        "deeply_ssd_bound": deeply_ssd_bound,
        "wrap_recommended": deeply_ssd_bound,
        "rule": ("wrap=on iff working set spills RAM AND SSD cannot feed decode "
                 "(both dimensionless; never an absolute MB/s)"),
    }

    # --- (c) auto-normalized speed constant ----------------------------------
    d["speed_calibration"] = {
        "provenance": base.get("provenance", "stub"),
        "baseline_tps": base.get("baseline_tps"),
        "baseline_tps_ceiling": base.get("baseline_tps_ceiling"),
        "note": ("t/s is minted ONCE at boot and used only to auto-normalize the "
                 "objective's speed term; it is NEVER re-read by the controller at "
                 "runtime (P2). Ratios, not absolutes, drive control."),
    }
    return d


def build_launch(profile: dict) -> dict:
    """Map a profile to the DS4_PACE_AUTO launch contract (env + ds4 flags)."""
    d = profile["derived"]
    slots = d["cache"]["cache_slots"]
    k = d["k_initial"]["k_initial"]
    wrap = 1 if d["offload_regime"]["wrap_recommended"] else 0
    env = {
        "DS4_PACE": "1",
        "DS4_PACE_AUTO": "1",
        "DS4_PACE_KEEP": str(k),
        "DS4_PACE_WRAP": str(wrap),
        "DS4_PACE_WRAP_ROTATE_DELTA": str(wrap),  # cheap even when full WRAP is off
    }
    model = profile["model"]["path"]
    ds4_flags = ["--ssd-streaming", "--ssd-streaming-cache-experts", str(slots)]
    return {"env": env, "ds4_flags": ds4_flags, "model": model}


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/root/models/ds4-2bit.gguf",
                    help="GGUF path (for footprint + bandwidth target)")
    ap.add_argument("--ds4-bin", default="/root/ds4/ds4",
                    help="ds4 binary for the baseline t/s probe (c)")
    ap.add_argument("--nvidia-smi", default="nvidia-smi")
    ap.add_argument("--out", default=None, help="write profile JSON here")
    ap.add_argument("--profile", default=None,
                    help="read an existing profile (with --emit-launch), skip probing")
    ap.add_argument("--emit-launch", action="store_true",
                    help="print the DS4_PACE_AUTO launch env/flags and exit")
    ap.add_argument("--no-ds4", action="store_true",
                    help="skip probe (c); estimate baseline t/s from bandwidth")
    ap.add_argument("--no-warm-first", action="store_true",
                    help="do NOT run a discarded warm-up pass before measuring (c)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--ds4-timeout", type=int, default=420)
    ap.add_argument("--sample-gib", type=float, default=2.0,
                    help="bytes read per stream for the bandwidth probe")
    ap.add_argument("--bw-threads", type=int, default=4)
    ap.add_argument("--no-drop-caches", action="store_true",
                    help="do NOT drop the page cache before the cold bandwidth read "
                         "(courtesy on a shared host; cold numbers may read warm)")
    ap.add_argument("--reserve-frac", type=float, default=0.10,
                    help="VRAM do-not-OOM headroom (KV+activations+frag); safety, not perf")
    ap.add_argument("--task-class", default="wide", choices=list(KNEE),
                    help="assumed width class for the K* floor (refined at runtime)")
    args = ap.parse_args()

    if args.emit_launch:
        if not args.profile:
            ap.error("--emit-launch requires --profile")
        with open(args.profile) as fh:
            profile = json.load(fh)
        launch = build_launch(profile)
        exports = " ".join(f'{k}={v}' for k, v in launch["env"].items())
        print("# DS4_PACE_AUTO launch contract (source: %s)" % args.profile)
        print("export " + exports)
        print("# ds4 " + " ".join(["-m", launch["model"]] + launch["ds4_flags"]))
        print(json.dumps(launch, indent=2))
        return

    started = datetime.now(timezone.utc).isoformat()
    t_probe0 = time.perf_counter()
    print(f"[boot-probe] start {started}", file=sys.stderr)

    profile = {
        "schema": "reap-loop/boot-probe/v1",
        "started_utc": started,
        "host": {"provenance": "measured", "hostname": platform.node(),
                 "platform": platform.platform(), "python": platform.python_version()},
    }

    print("[boot-probe] (a) GPU/VRAM ...", file=sys.stderr)
    profile["gpu"] = probe_gpu(args.nvidia_smi)
    print("[boot-probe] (a) model footprint (gguf) ...", file=sys.stderr)
    profile["model"] = probe_model(args.model)
    profile["ram"] = probe_ram()

    print("[boot-probe] (b) bandwidth (SSD cold + RAM warm) ...", file=sys.stderr)
    profile["bandwidth"] = probe_bandwidth(
        args.model, profile["model"]["nonexpert_resident_bytes"],
        sample_gib=args.sample_gib, threads=args.bw_threads,
        drop_caches=not args.no_drop_caches)

    # Provisional slots for the baseline probe (needs a cache size before decode).
    vram_total = profile["gpu"].get("vram_total_bytes")
    footprint = profile["model"]["footprint_per_expert_bytes"]
    if vram_total:
        prov_slots = max(0, int((vram_total * (1 - args.reserve_frac)
                                 - profile["model"]["nonexpert_resident_bytes"]) // footprint))
    else:
        prov_slots = 256

    if args.no_ds4:
        print("[boot-probe] (c) baseline t/s ESTIMATED (--no-ds4) ...", file=sys.stderr)
        profile["baseline"] = estimate_baseline_tps(profile["model"], profile["bandwidth"], prov_slots)
    else:
        print(f"[boot-probe] (c) baseline t/s via ds4 (-n {args.tokens}) ...", file=sys.stderr)
        profile["baseline"] = probe_baseline_tps(
            args.ds4_bin, args.model, prov_slots, args.ctx, args.tokens, args.ds4_timeout,
            warm_first=not args.no_warm_first)
        if profile["baseline"].get("provenance") == "stub":
            print("[boot-probe] (c) ds4 run failed; falling back to estimate", file=sys.stderr)
            est = estimate_baseline_tps(profile["model"], profile["bandwidth"], prov_slots)
            profile["baseline"]["fallback_estimate"] = est

    print("[boot-probe] derive control constants ...", file=sys.stderr)
    profile["derived"] = derive(
        profile["gpu"], profile["model"], profile["ram"], profile["bandwidth"],
        profile["baseline"], args.ctx, args.reserve_frac, args.task_class)
    profile["launch"] = build_launch(profile)
    profile["probe_wall_s"] = round(time.perf_counter() - t_probe0, 1)

    text = json.dumps(profile, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"[boot-probe] profile -> {args.out} ({profile['probe_wall_s']} s)", file=sys.stderr)
    print(text)


if __name__ == "__main__":
    main()
