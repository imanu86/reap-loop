"""build_dsv2_memmap.py — expert ROUTED di DeepSeek-V2-Lite -> due memmap fp16 su SSD.

Come build_qwen3_memmap, ma per DeepSeek-V2-Lite (modello CONCENTRATO: shared expert + fine-grained,
il banco di prova giusto per il metodo predittivo dopo il negativo su Qwen3 diffuso):
 - 27 layer, ma il layer 0 e DENSE (first_k_dense_replace=1): NON ha expert, lo si salta naturalmente
   (le sue chiavi non contengono '.mlp.experts.');
 - 64 routed expert/layer; i 2 SHARED expert NON vanno nei memmap (sono sempre attivi -> restano
   residenti su GPU, li carica validate_probe_dsv2 come backbone). Le loro chiavi sono
   '.mlp.shared_experts.*' che NON contengono '.mlp.experts.' -> escluse dal filtro;
 - layout fuso atteso da MemmapBacking: gu[li,j]=cat([gate,up]) [2I,H], dn[li,j]=down [H,I],
   con I=moe_intermediate_size=1408, H=2048, indicizzato per layer ASSOLUTO (slot layer 0 inutilizzato).

Il checkpoint salva gli expert PER-EXPERT separati (mlp.experts.{j}.{gate,up,down}_proj.weight).
RAM-safe: un tensore per-expert alla volta (~MB).

Uso (dopo il download):  python -u scripts/build_dsv2_memmap.py --out D:/dsv2_memmap
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--out", default="D:/dsv2_memmap", help="dir dei memmap (su SSD con spazio).")
    args = p.parse_args()

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    L = int(cfg.num_hidden_layers)
    E = int(cfg.n_routed_experts)
    H = int(cfg.hidden_size)
    I = int(cfg.moe_intermediate_size)
    first_moe = int(getattr(cfg, "first_k_dense_replace", 0))
    gu_shape = (L, E, 2 * I, H)   # [layer, expert, 2I, H]  (slot dei layer dense restano a zero)
    dn_shape = (L, E, H, I)       # [layer, expert, H, I]
    print(f"[mmap] {args.model}: L={L} (MoE da layer {first_moe}) E={E} H={H} I={I}")
    print(f"[mmap] gate_up {gu_shape} + down {dn_shape} fp16 -> "
          f"{(np.prod(gu_shape)+np.prod(dn_shape))*2/2**30:.1f} GB su {args.out}")

    snap = snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not shards:
        print(f"[mmap] nessun safetensors in {snap} — download finito?")
        return 2

    os.makedirs(args.out, exist_ok=True)
    gu = np.memmap(os.path.join(args.out, "gate_up.f16.mmap"), dtype=np.float16, mode="w+", shape=gu_shape)
    dn = np.memmap(os.path.join(args.out, "down.f16.mmap"), dtype=np.float16, mode="w+", shape=dn_shape)

    seen_gate, seen_up, seen_dn = set(), set(), set()
    for shard in shards:
        n = 0
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if ".mlp.experts." not in key:
                    continue  # shared_experts / layer dense / backbone: non sono routed expert
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

    meta = {"model": args.model, "L": L, "E": E, "H": H, "I": I, "first_moe_layer": first_moe,
            "gate_up": {"file": "gate_up.f16.mmap", "shape": list(gu_shape), "dtype": "float16"},
            "down": {"file": "down.f16.mmap", "shape": list(dn_shape), "dtype": "float16"}}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    exp_total = (L - first_moe) * E
    print(f"[mmap] FATTO. gate={len(seen_gate)}/{exp_total} up={len(seen_up)}/{exp_total} "
          f"down={len(seen_dn)}/{exp_total} (E={E} su {L-first_moe} layer MoE)")
    if len(seen_gate) != exp_total or len(seen_up) != exp_total or len(seen_dn) != exp_total:
        allpairs = set((li, j) for li in range(first_moe, L) for j in range(E))
        print(f"[mmap] ATTENZIONE expert mancanti (primi 10) gate={sorted(allpairs-seen_gate)[:10]} "
              f"down={sorted(allpairs-seen_dn)[:10]}")
        return 1
    print(f"[mmap] memmap pronti in {args.out} (gate_up.f16.mmap, down.f16.mmap, meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
