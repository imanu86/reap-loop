"""validate_probe_qwen3.py — predire vs reagire sul regime A TANTI EXPERT (Qwen3-30B-A3B, 128 expert).

Il banco di prova vero del metodo: 128 expert (vs 64 di OLMoE, diffuso), expert serviti da SSD
(memmap, fuori dalla RAM), cache GPU piccola (≪128). Domanda: il pre-staging PREDITTIVO (probe sul
prompt) batte la cache REATTIVA (cold + LRU, = ds4) abbattendo i miss, al netto del costo del probe?

Prerequisito: scripts/build_qwen3_memmap.py ha creato D:/qwen3_memmap/{gate_up,down}.f16.mmap + meta.json.

Loader RAM-SAFE: il modello (30.5B) NON entra in 32GB RAM. Init su `meta` (zero memoria) + carico SOLO
i pesi non-expert (~3GB) su GPU dai safetensors; gli expert restano meta, serviti dal MemmapBacking.

Uso (dopo build_qwen3_memmap):  python -u scripts/validate_probe_qwen3.py --memmap D:/qwen3_memmap
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

HF = "Qwen/Qwen3-30B-A3B"
N_EXPERTS = 128
N_PROMPTS = 6
MAX_NEW = 48
CAPS = [8, 16, 32]   # capienza cache per-layer ≪ 128


def load_nonexpert_on_gpu(model_id, device, dtype):
    """Init Qwen3 su meta + carica SOLO i pesi non-expert su GPU (RAM-safe). Expert -> restano meta."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    snap = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "merges*", "vocab*"])
    nonexpert_sd = {}
    for shard in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if ".mlp.experts." in key:
                    continue  # serviti dal memmap, non in RAM/GPU
                nonexpert_sd[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(nonexpert_sd, strict=False, assign=True)
    # missing = i pesi expert (attesi: restano meta); unexpected dovrebbe essere vuoto.
    model.tie_weights()
    # I buffer NON-persistenti del rotary embedding (inv_freq, original_inv_freq) NON stanno nei
    # safetensors: dopo l'init-su-meta restano su meta -> install_expert_cache (che sposta i buffer su
    # device) crasherebbe con "Cannot copy out of meta tensor". Li materializziamo re-istanziando il
    # rotary embedding sul device reale (il costruttore ricalcola inv_freq dal config).
    rot = model.model.rotary_emb
    model.model.rotary_emb = type(rot)(config=config, device=torch.device(device))
    model.eval()
    n_meta = sum(1 for _, p in model.named_parameters() if p.is_meta)
    print(f"[load] non-expert su GPU={len(nonexpert_sd)} | param meta rimasti (expert)={n_meta} "
          f"| unexpected={len(unexpected)}")
    return model, config


def open_memmaps(memmap_dir):
    import numpy as np
    meta = json.load(open(os.path.join(memmap_dir, "meta.json"), encoding="utf-8"))
    gu = np.memmap(os.path.join(memmap_dir, meta["gate_up"]["file"]), dtype=np.float16, mode="r",
                   shape=tuple(meta["gate_up"]["shape"]))
    dn = np.memmap(os.path.join(memmap_dir, meta["down"]["file"]), dtype=np.float16, mode="r",
                   shape=tuple(meta["down"]["shape"]))
    return gu, dn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", default="D:/qwen3_memmap", help="dir con gate_up/down .mmap + meta.json")
    p.add_argument("--n-prompts", type=int, default=N_PROMPTS, help="n. prompt dal validator (default 6)")
    p.add_argument("--max-new", type=int, default=MAX_NEW, help="token generati per prompt (default 48)")
    p.add_argument("--caps", default=",".join(str(c) for c in CAPS),
                   help="capienze cache per-layer (csv, default 8,16,32)")
    args = p.parse_args()
    n_prompts = int(args.n_prompts)
    max_new = int(args.max_new)
    caps = [int(c) for c in str(args.caps).split(",") if c.strip()]

    import torch
    from transformers import AutoTokenizer

    from msc.instrument.router_hooks import RouterHookSpec, RouterLogger
    from msc.instrument.trace import TraceWriter
    from msc.validator.python_unit_tests import PythonUnitTestValidator
    from msc.hierarchy.olmoe_cache import install_expert_cache, MemmapBacking
    from msc.hierarchy.calibrate import resident_from_trace

    if not os.path.exists(os.path.join(args.memmap, "meta.json")):
        print(f"[val] memmap mancante in {args.memmap}: esegui prima build_qwen3_memmap.py")
        return 2

    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(HF)
    print("[val] carico Qwen3-30B (non-expert su GPU, expert da memmap-SSD)...")
    model, cfg = load_nonexpert_on_gpu(HF, "cuda", dtype)
    n_layers = int(cfg.num_hidden_layers)
    layers = list(range(n_layers))
    gu, dn = open_memmaps(args.memmap)

    def backing_factory(layer_idx, experts):
        return MemmapBacking(gu, dn, layer_idx, int(experts.num_experts))

    validator = PythonUnitTestValidator("data/codegen_problems.jsonl")
    items = validator.items()[:n_prompts]

    def enc_of(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    def gen_session():
        correct, ntok, t0 = 0, 0, time.perf_counter()
        for it in items:
            enc = enc_of(it.prompt)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            new = out[0, enc["input_ids"].shape[1]:]
            ntok += int(new.shape[0])
            correct += int(validator.verify(it, tok.decode(new, skip_special_tokens=True)))
        return correct / len(items), ntok, time.perf_counter() - t0

    os.makedirs("runs", exist_ok=True)
    rows = []
    for cap in caps:
        pin = max(1, cap - 1)
        empty = {i: set() for i in layers}

        # PROBE: prefill a freddo, cattura routing -> working-set predetto
        h = install_expert_cache(model, cap, empty, backing_factory=backing_factory)
        tp = f"runs/val_qwen3_probe_c{cap}.jsonl"
        writer = TraceWriter(tp); logger = RouterLogger(model, RouterHookSpec.for_model(HF), writer)
        with logger.capture(session_id="probe", ctx_len=0):
            for it in items:
                with torch.no_grad():
                    model(**enc_of(it.prompt))
        writer.close()
        probe_miss = h.stats()["total"]["misses"]; h.remove()
        predicted = resident_from_trace(tp, pin)

        # REACTIVE (cold+LRU = ds4)
        h = install_expert_cache(model, cap, empty, backing_factory=backing_factory)
        acc_r, _, t_r = gen_session(); react = h.stats()["total"]["misses"]; h.remove()

        # PREDICTIVE (pin working-set predetto)
        h = install_expert_cache(model, cap, predicted, backing_factory=backing_factory)
        acc_p, ntok, t_p = gen_session(); pred = h.stats()["total"]["misses"]; h.remove()

        saving = react - pred
        rows.append({"cap": cap, "pin": pin, "react_miss": react, "pred_miss": pred,
                     "probe_miss": probe_miss, "saving": saving, "net_vs_probe": saving - probe_miss,
                     "react_tok_s": round(ntok / t_r, 2), "pred_tok_s": round(ntok / t_p, 2),
                     "acc_react": acc_r, "acc_pred": acc_p})
        print(f"[val] cap={cap}/{N_EXPERTS} pin={pin}: react={react} pred={pred} probe={probe_miss} "
              f"saving={saving} net={saving-probe_miss} tok/s r={ntok/t_r:.1f}/p={ntok/t_p:.1f} "
              f"acc r={acc_r:.2f}/p={acc_p:.2f}", flush=True)

    print(f"\n=== Qwen3-30B (128 expert, memmap-SSD): predire vs reagire ===")
    print(f"{'cap':>4} | {'react':>7} {'pred':>7} {'probe':>7} | {'saving':>7} {'net':>7} | tok/s r/p | acc r/p")
    for r in rows:
        print(f"{r['cap']:>4} | {r['react_miss']:>7} {r['pred_miss']:>7} {r['probe_miss']:>7} | "
              f"{r['saving']:>7} {r['net_vs_probe']:>7} | {r['react_tok_s']}/{r['pred_tok_s']} | "
              f"{r['acc_react']:.2f}/{r['acc_pred']:.2f}")
    with open("runs/validate_probe_qwen3.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[val] CSV: runs/validate_probe_qwen3.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
