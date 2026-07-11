# BOOT-PROBE — hardware-adaptive calibration (P2 portability core)

**Status:** `[FATTO — script + reference 3060 profile]` · script `scripts/boot_probe.py`
· consumed by PACE via the `DS4_PACE_AUTO` launch contract (§4).

This is the concrete implementation of `docs/SOTA_ROADMAP.md` §Auto-calibrazione
(principle **P2 — PORTABILITÀ**). It closes the gate that the roadmap called
`[TODO]`: *"la sua esistenza è il gate P2."*

> **Why it exists.** The control laws must be portable to *any* hardware. Today
> the sizing/regime constants (cache 407 slots, WRAP on/off, K23/rotate32) are
> **3060-tuned and non-transferable** (`SOTA_ROADMAP.md` §S5 vincolo). The
> boot-probe replaces every hard-wired absolute with a value **measured at first
> boot on the local host**, or with a declared **P2 invariant** (a dimensionless
> / unit-invariant quantity that transfers across HW). After the probe, PACE has
> the cache slots, the WRAP regime, and the initial K* it needs — *derived*, not
> cabled.

---

## 0. The cardinal rule and the provenance tags

**No absolute hardware number is ever hard-coded.** Every field in the emitted
profile carries a provenance tag:

| tag | meaning |
|---|---|
| `measured` | read from THIS host now (nvidia-smi, file reads, /proc, ds4) |
| `derived` | computed from measured values + geometry |
| `invariant` | a portable **P2** constant, offline-calibrated, HW-independent (LRU hit curve, cov90 K-floor, knee table) — transfers across HW |
| `estimated` | fallback model used when a measurement could not be taken |
| `stub` | declared not-measured on this run |

A reviewer can grep the profile for `"provenance": "stub"` to see exactly what
is not yet real on a given host.

---

## 1. The four probes (roadmap table)

| Probe | Measures | Decides |
|---|---|---|
| **(a)** VRAM + per-expert footprint | free VRAM (nvidia-smi) **and** the dtype-aware per-expert footprint read from the GGUF header (`params/expert × bytes/dtype`) | **cache slots** = `floor(usable_vram / footprint)`. 407/258/512 are data points, never the constant — the slot count is always re-derived from the measured footprint |
| **(b)** SSD→RAM→VRAM bandwidth | cold single-stream + cold **threaded** (the WRAP page-in path) + RAM page-cache read | **WRAP on/off** via a **dimensionless** boundary (§3). Never an absolute MB/s: a faster NVMe / bigger VRAM simply lands on the safe side |
| **(c)** baseline t/s on ~32 tok | node speed, **warm-first** (discard the cold pass, measure the warm one — `PACE_DESIGN.md` §7) | an **auto-normalized** speed constant, minted ONCE at boot; NEVER re-read by the controller at runtime (keeps P2 and probe-c consistent) |
| **(d)** coverage floor | *not a boot measurement* — an offline **E-CAL** invariant | the **cov90 anti-under-provisioning floor** for the initial K*; refined at runtime by the width sensor |

`(a)-(b)-(c)` are the P2 gate. `(d)` is offline (E-CAL verdict was NEGATIVE:
coverage does not separate collapse, so it is used only as a floor — see
`runs/ds4/20260710_ecal_coverage_threshold/REPORT.md`).

---

## 2. Derivation formulas (all dimensionless or measured)

### (a) cache slots — dtype-aware

```
footprint_per_expert = Σ_{gate,up,down} quant_bytes(tensor)      # measured from GGUF
usable_vram          = vram_total − nonexpert_resident − reserve  # measured − measured − safety
cache_slots          = floor(usable_vram / footprint_per_expert)
```

- `nonexpert_resident` = (total tensor bytes − expert tensor bytes), measured from
  the GGUF (the attn/dense/embeddings/output weights that stay resident under
  `--ssd-streaming`).
- `reserve` = a **do-not-OOM headroom** (KV + activations + fragmentation),
  expressed as a fraction of `vram_total` (`--reserve-frac`, default 0.10). It is
  a *safety* margin, **not** a performance constant.
- Maps to ds4 `--ssd-streaming-cache-experts <cache_slots>`.

### (d) initial K* — offline invariant, refined at runtime

```
k_initial = max(knee(class), cov90_K_floor)      # anti-under-provisioning
```

- `cov90_K_floor` (≈ 38, E-CAL, **task-invariant** across html + 11 coding
  prompts) is the floor so we never under-provision.
- `knee(class)` = {narrow 20, medium 32, wide 48} from the decision model. At boot
  the class is unknown, so we assume the conservative **wide** knee; the runtime
  **width sensor** (≥150 tok, `DECISION_MODEL.md` §1) refines K*, and a *proven*
  rewind airbag can later tighten toward K12–16.

### (b) offload regime → WRAP — the dimensionless boundary

The discriminator is **whether the active working set fits RAM page cache**:

```
working_set(K) = K × n_moe_layers × footprint_per_expert
ram_fit_ratio  = ram_available / working_set(K_initial)      # PURE number
working_set_fits_ram = ram_fit_ratio ≥ 1
```

- **fits RAM** → decode-path misses are served from RAM (copy-bound). WRAP is
  pure overhead here — measured: prefetch *slows* the practical 3060 config
  (0.82 vs 1.27 t/s, `CLAIMS_CURRENT.md` §PREFETCH). → **WRAP off**.
- **spills RAM** → misses hit the SSD sync path (50–230 ms cliff, E-LAT tier-c′).
  A confirming SSD-throughput ratio says how badly SSD is the bottleneck:

```
miss_bytes/token   = recalls_per_token × footprint × (1 − hit_rate(slots))
decode_ssd_demand  = miss_bytes/token × baseline_tps          # GiB/s the decode pulls
ssd_throughput_ratio = ssd_cold_threaded / decode_ssd_demand  # PURE number
deeply_ssd_bound = (NOT working_set_fits_ram) AND (ssd_throughput_ratio < 1)
wrap_recommended = deeply_ssd_bound
```

Both ratios are pure numbers; the only threshold is **1**. A 3080-12GB / 3090 /
faster NVMe changes the *inputs* and can flip the verdict on its own — nothing
3060-specific is cabled. `hit_rate(slots)` is the **P2-invariant LRU curve**
(routing locality, HW-independent; E-LAT §4 / J17).

The pods (≈1 TB RAM) hold the whole 80.8 GiB model in page cache → `ram_fit_ratio ≫ 1`
for any K → never SSD-bound → WRAP off; they are simply compute-fast. The
boundary reproduces both ends with one formula.

---

## 3. Reference profile — RTX 3060 12GB / WSL (measured 2026-07-11)

Run: `python3 scripts/boot_probe.py --model /root/models/ds4-2bit.gguf --ds4-bin /root/ds4/ds4`.
**What is real vs stub on this host:**

| Field | Value | Provenance |
|---|---|---|
| GPU | RTX 3060, VRAM 12288 MiB total | **measured** (nvidia-smi, WSL) |
| footprint/expert | **7,077,888 B = 6.75 MiB** (gate iq2_xxs 2,162,688 + up 2,162,688 + down q2_k 2,752,512) | **measured** (GGUF header) |
| geometry | 43 MoE layers × 256 experts, top_k 6, shared 1 → **258 recalls/token**, 1.70 GiB/token demanded | **measured** (GGUF metadata) |
| model bytes | total 80.76 GiB · expert 72.56 GiB · **nonexpert resident 8.20 GiB** | **measured** (GGUF) |
| RAM | 60 GiB total, ~59 GiB available | **measured** (/proc/meminfo) |
| SSD cold single-stream | **0.314 GiB/s** (WSL vhdx, post drop_caches) | **measured** (dd-equivalent) |
| SSD cold threaded (4-way) | **2.10 GiB/s** (the WRAP page-in path; matches E-LAT tier-c 2.5–4.4) | **measured** |
| RAM warm read | **16.6 GiB/s** | **measured** |
| baseline t/s (warm) | see profile JSON `baseline.baseline_tps` | **measured** (ds4 -n 48, warm-first) |
| **cache_slots** | `floor((12288·0.9 − 8197 MiB)/6.75) ≈ 424` (cross-check: E1 max 407) | **derived** |
| **K_initial** | `max(48, 38) = 48` (wide-class assumption) | **invariant** (E-CAL + knee) |
| working_set(48) | 48·43·6.75 MiB = **13.6 GiB** → ram_fit_ratio ≈ 4.3 ≥ 1 | **derived** |
| **WRAP** | **off** (working set fits RAM → copy-bound, not SSD-bound) | **derived** |

**Honest notes on this reference run:**
- The **cold** first ds4 pass measured 1.07 t/s while the expert cache populated
  (7 GiB cached only at the end). That is *not* the steady number — probe (c)
  discards it and reports the **warm** pass (warm-first discipline). The warm t/s
  lands near the E-LAT-calibrated 3.12 t/s @ cache256; the profile JSON carries
  the actual figure from this host.
- The SSD single-stream 0.314 GiB/s is the WSL vhdx *sync* worst case; the
  threaded 2.10 GiB/s is what WRAP would actually achieve — the ~6.7× spread is
  exactly why WRAP helps *only* when the working set spills RAM.
- **What is stub / not independently measured:** the RAM→VRAM **H2D (cudaMemcpy)
  bandwidth** is not measured by an isolated CUDA micro-bench; the E-LAT `t_b`
  (0.952 ms/expert) is used only inside the *estimate* fallback (`--no-ds4`), and
  the real profile uses the measured ds4 t/s instead. A cudaMemcpy micro-bench is
  the one remaining measurement to make probe (b)'s copy-path fully first-party
  (§6).

---

## 4. How PACE consumes it — the `DS4_PACE_AUTO` contract

The profile is JSON on disk. The controller consumes it **at the launcher level**
— no engine patch is required today, because the engine already reads every
`DS4_PACE_*` env and `--ssd-streaming-cache-experts`. `boot_probe.py --emit-launch`
maps a profile to the exact env + flags:

```
python3 scripts/boot_probe.py --profile profile.json --emit-launch
# ->
export DS4_PACE=1 DS4_PACE_AUTO=1 DS4_PACE_KEEP=48 DS4_PACE_WRAP=0 DS4_PACE_WRAP_ROTATE_DELTA=0
# ds4 -m /root/models/ds4-2bit.gguf --ssd-streaming --ssd-streaming-cache-experts 424
```

| Profile field | Consumed as | Meaning |
|---|---|---|
| `derived.cache.cache_slots` | `--ssd-streaming-cache-experts N` | dtype-aware VRAM cache size |
| `derived.k_initial.k_initial` | `DS4_PACE_KEEP` | initial mask width (floor; width sensor refines) |
| `derived.offload_regime.wrap_recommended` | `DS4_PACE_WRAP` (+ `_WRAP_ROTATE_DELTA`) | WRAP bulk page-in only when deeply-SSD-bound |
| `derived.speed_calibration.baseline_tps` | auto-normalization constant | minted once; **not** a runtime controller input (P2) |

**`DS4_PACE_AUTO=1` semantics (launcher convention):** when set, the launch
wrapper reads `DS4_PACE_PROFILE=<path>` (default: the newest
`runs/ds4/*_bootprobe/profile.json`) and seeds `DS4_PACE_KEEP` / `DS4_PACE_WRAP`
/ `--ssd-streaming-cache-experts` from the derived block **unless the operator
overrode them explicitly**. This keeps P2 airtight: the controller's *inputs* are
either measured-derived constants or P2 invariants; the absolute t/s enters only
the auto-normalization, once.

**Optional future engine patch (reserved 0029, NOT authored):** folding
`DS4_PACE_AUTO` into the engine so ds4 reads the profile itself at startup is a
small, mechanical addition (parse `DS4_PACE_PROFILE`, seed the same knobs before
the PACE state machine inits). It is left as a reserved number in
`patches/README.md` rather than authored blind, because this workstation is
CPU-only and cannot compile/test a ds4.c change — and the launcher contract above
already delivers the full behavior with zero untested C. Register as
`0029-pace-auto-profile-reader` when a GPU host can compile+smoke it.

---

## 5. Portability check (how S5 uses this)

On a second HW (pod 3080-12GB / 3090-24GB, `SOTA_ROADMAP.md` §S5), the same probe
runs unchanged and self-measures:
- different **footprint** if the quant differs → different `cache_slots`, automatically;
- different **VRAM** → different `cache_slots`;
- **1 TB RAM** → `ram_fit_ratio ≫ 1` for any K → WRAP off, no manual retune;
- different **NVMe / H2D** → different `ssd_throughput_ratio`, verdict flips on its own.

The transferred config uses ONLY derived equivalents (K/floor-from-coverage,
cache-slot from probe (a), regime from probe (b)) — the forbidden 3060-tuned
absolutes (static K23 / rotate32 / cache256 / cache 407-slot / reserve=1) never
appear. That is exactly the S5 `Vincolo di trasferibilità`.

---

## 6. What stays stub / to-measure

1. **H2D (cudaMemcpy) bandwidth** — an isolated RAM-pinned→VRAM micro-bench, so
   probe (b)'s copy path is first-party rather than leaning on the E-LAT `t_b`
   for the `--no-ds4` estimate. (Needs a tiny CUDA probe or a `ds4` sub-command.)
2. **Warm-t/s in one shot** — probe (c) currently runs ds4 twice (warm-discard +
   measure). A ds4 flag that emits a last-chunk steady t/s would let (c) fit the
   60–90 s budget with a single load.
3. **The engine-side `DS4_PACE_AUTO` reader** (patch 0029) — designed here,
   authored only on a host that can compile+smoke it.
4. **Second-HW validation** — the boundary is reproduced *analytically* for the
   1 TB-RAM pod; it is confirmed empirically only when S5 runs the probe there.
