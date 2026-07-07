"""Strada B — acceptance eval del drafter DSpark ufficiale su DeepSeek-V4-Flash.

Teacher-forcing: genera greedy col loop target (ground truth deterministica) e a ogni
passo chiama forward_spec; confronta i 5 draft coi 5 token greedy successivi.
Nessuna verifica/rollback da implementare: stessa matematica dell'accepted-length
(greedy match di prefisso), come il position-wise conditional acceptance del paper (Fig.2).

Output (rank 0): CSV per token con drafts, ground-truth, confidence -> acceptance per
posizione + tau medio + coppie (confidence, esito) per calibrazione STS.

Uso (dal folder inference del checkpoint):
  torchrun --nproc-per-node 2 dspark_accept_eval.py \
      --ckpt-path /root/ckpt-mp2 --config config.json --tokenizer-path /root/hf/dspark \
      --out /root/out_b/accept.csv --max-new-tokens 200
"""
import csv
import json
import os
import sys
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../encoding")))
from encoding_dsv4 import encode_messages

PROMPTS = {
    "code": "Write a Python function that parses a CSV file and returns the sum of the second column. Include error handling and a short docstring.",
    "math": "Compute step by step: what is the sum of all integers from 1 to 100 that are divisible by 3? Show your reasoning and give the final number.",
    "chat": "Give me practical, friendly advice for organizing a small team's weekly schedule. Keep it conversational.",
}


@torch.inference_mode()
def eval_prompt(model, tokenizer, prompt, max_new_tokens, rank):
    ids = tokenizer.encode(encode_messages(
        [{"role": "user", "content": prompt}], thinking_mode="chat"))
    toks = torch.tensor([ids], dtype=torch.long, device="cuda")
    L = toks.size(1)

    # Pattern identico al __main__ di model.py:
    #   prefill: out,_,mh = model(prompt,0); forward_spec(out,mh,0)  [solo cache]
    #   decode:  out,_,mh = model(cur,pos);  spec = forward_spec(out,mh,pos)
    # Al passo s: anchor = gen[s+1]; i draft predicono gen[s+2 .. s+6].
    out, _, mh = model(toks, 0)               # prefill target -> g0
    model.forward_spec(out, mh, 0)            # prefill cache drafter (ritorna None)
    gen = [out.item()]
    pending = []
    cur = out.view(1, 1)
    pos = L
    for step in range(max_new_tokens):
        out, _, mh = model(cur, pos)          # processa gen[step] -> gen[step+1]
        spec = model.forward_spec(out, mh, pos)
        gen.append(out.item())
        drafts = spec[0][0, 1:].tolist()      # output_ids = [anchor, d1..dB]
        conf = spec[2][0].float().tolist()
        pending.append({"step": step, "drafts": drafts, "conf": conf})
        cur = out.view(1, 1)
        pos += 1
        if gen[-1] == tokenizer.eos_token_id:
            break

    rows = []
    for p in pending:
        s = p["step"]
        gt = gen[s + 2:s + 2 + len(p["drafts"])]
        if len(gt) < len(p["drafts"]):
            continue  # coda troncata da EOS: scarta per pulizia statistica
        rows.append({
            "step": s,
            **{f"d{k+1}": p["drafts"][k] for k in range(len(p["drafts"]))},
            **{f"g{k+1}": gt[k] for k in range(len(gt))},
            **{f"c{k+1}": round(p["conf"][k], 5) for k in range(len(p["conf"]))},
        })
    return rows, len(gen)


def summarize(rows, block):
    n = len(rows)
    acc = []
    for k in range(1, block + 1):
        elig = [r for r in rows if all(r[f"d{j}"] == r[f"g{j}"] for j in range(1, k))]
        hit = [r for r in elig if r[f"d{k}"] == r[f"g{k}"]]
        acc.append((len(hit), len(elig)))
    # tau atteso (greedy, chain): 1 bonus + somma prodotti cumulati empirici
    tau = 1.0
    prod = 1.0
    for h, e in acc:
        if e == 0:
            break
        prod *= h / e
        tau += prod
    return acc, tau, n


def main():
    ap = ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    a = ap.parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(a.config) as f:
        args = ModelArgs(**json.load(f))
    args.temperature = 0.0                    # greedy: GT deterministica, draft greedy
    args.max_batch_size = 1
    args.max_seq_len = 8192
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer_path)
    load_model(model, os.path.join(
        a.ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")

    all_rows = {}
    for name, prompt in PROMPTS.items():
        rows, ntok = eval_prompt(model, tokenizer, prompt, a.max_new_tokens, rank)
        all_rows[name] = rows
        if rank == 0:
            acc, tau, n = summarize(rows, args.dspark_block_size)
            print(f"[{name}] tokens={ntok} cicli={n} tau_atteso={tau:.3f}")
            for k, (h, e) in enumerate(acc, 1):
                r = f"{h}/{e}={h/e:.3f}" if e else "n/a"
                print(f"  P(accept pos{k} | prefisso ok) = {r}")

    if rank == 0:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = None
            for name, rows in all_rows.items():
                for r in rows:
                    r2 = {"prompt": name, **r}
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(r2.keys()))
                        w.writeheader()
                    w.writerow(r2)
        print("CSV:", a.out)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
