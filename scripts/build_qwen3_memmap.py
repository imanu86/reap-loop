"""build_qwen3_memmap.py — converte gli expert di Qwen3-30B-A3B in due memmap fp16 su SSD.

Gli expert (~58GB bf16) non entrano in 32GB di RAM: li riversiamo su disco come due numpy.memmap
GLOBALI fp16 (gate_up [L,E,2I,H], down [L,E,H,I]), letti poi slice-per-slice dal MemmapBacking. Il
modello quindi gira con gli expert su SSD (capacita limitata dal disco, non dalla RAM) — il cuore
della tesi "gigante su HW minimo".

RAM-safe: legge UN tensore PER-EXPERT alla volta dai safetensors (safe_open lazy), casta bf16->fp16,
scrive nel memmap, libera. Picco ~MB. NON carica il modello intero.

NB: il checkpoint Qwen3-30B-A3B salva gli expert PER-EXPERT e SEPARATI
(``...mlp.experts.{j}.{gate,up,down}_proj.weight``), NON fusi. Li impacchettiamo nel layout fuso
atteso dal MemmapBacking/cache (= ``Qwen3MoeExperts.gate_up_proj`` [E,2I,H], ``down_proj`` [E,H,I]):
  gu[li, j] = cat([gate_proj, up_proj], axis=0)  -> [2I, H]   (chunk(2) nel forward: prima gate, poi up)
  dn[li, j] = down_proj                          -> [H, I]

Uso (dopo il download):  python -u scripts/build_qwen3_memmap.py --out D:/qwen3_memmap
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--out", default="D:/qwen3_memmap", help="dir dei memmap (su SSD con spazio).")
    args = p.parse_args()

    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    L = int(cfg.num_hidden_layers)
    E = int(cfg.num_experts)
    H = int(cfg.hidden_size)
    I = int(cfg.moe_intermediate_size)
    gu_shape = (L, E, 2 * I, H)   # gate_up_proj per layer: [E, 2I, H]
    dn_shape = (L, E, H, I)       # down_proj per layer:    [E, H, I]
    print(f"[mmap] {args.model}: L={L} E={E} H={H} I={I}")
    print(f"[mmap] gate_up {gu_shape} + down {dn_shape} fp16 -> "
          f"{(np.prod(gu_shape)+np.prod(dn_shape))*2/2**30:.1f} GB su {args.out}")

    snap = snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not shards:
        print(f"[mmap] nessun safetensors in {snap} — download finito?")
        return 2

    os.makedirs(args.out, exist_ok=True)
    gu_path = os.path.join(args.out, "gate_up.f16.mmap")
    dn_path = os.path.join(args.out, "down.f16.mmap")
    gu = np.memmap(gu_path, dtype=np.float16, mode="w+", shape=gu_shape)
    dn = np.memmap(dn_path, dtype=np.float16, mode="w+", shape=dn_shape)

    import torch
    seen_gate, seen_up, seen_dn = set(), set(), set()
    for shard in shards:
        n = 0
        with safe_open(shard, framework="pt") as f:   # pt: gestisce bf16 (numpy no)
            for key in f.keys():
                if ".mlp.experts." not in key:
                    continue  # router gate / attention / norm: non-expert (li carica validate_probe su GPU)
                if key.endswith("gate_proj.weight"):
                    proj = "gate"
                elif key.endswith("up_proj.weight"):
                    proj = "up"
                elif key.endswith("down_proj.weight"):
                    proj = "down"
                else:
                    continue
                li = int(key.split(".layers.")[1].split(".")[0])
                j = int(key.split(".experts.")[1].split(".")[0])
                t = f.get_tensor(key).to(dtype=torch.float16).cpu().numpy()
                if proj == "gate":
                    gu[li, j, :I, :] = t; seen_gate.add((li, j))   # prime I righe = gate
                elif proj == "up":
                    gu[li, j, I:, :] = t; seen_up.add((li, j))     # righe I:2I = up
                else:
                    dn[li, j] = t; seen_dn.add((li, j))
                n += 1
        print(f"[mmap] {os.path.basename(shard)}: {n} tensori-expert scritti", flush=True)
    gu.flush(); dn.flush()

    meta = {"model": args.model, "L": L, "E": E, "H": H, "I": I,
            "gate_up": {"file": "gate_up.f16.mmap", "shape": list(gu_shape), "dtype": "float16"},
            "down": {"file": "down.f16.mmap", "shape": list(dn_shape), "dtype": "float16"}}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    exp_total = L * E
    print(f"[mmap] FATTO. gate={len(seen_gate)}/{exp_total} up={len(seen_up)}/{exp_total} "
          f"down={len(seen_dn)}/{exp_total} (E={E} expert su {L} layer)")
    if len(seen_gate) != exp_total or len(seen_up) != exp_total or len(seen_dn) != exp_total:
        allpairs = set((li, j) for li in range(L) for j in range(E))
        miss_g = sorted(allpairs - seen_gate)[:10]
        miss_d = sorted(allpairs - seen_dn)[:10]
        print(f"[mmap] ATTENZIONE expert mancanti (primi 10) gate={miss_g} down={miss_d}")
        return 1
    print(f"[mmap] memmap pronti in {args.out} (gate_up.f16.mmap, down.f16.mmap, meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
