"""Track REAP-ds4 — Stage A bias-mask (design doc §2, eval plan §5).

Scrive -1e9 (F32) nelle posizioni di `blk.N.exp_probs_b.bias` degli expert POTATI
per i layer non-hash, direttamente nel gguf (in-place, con backup reversibile dei
1024 byte/layer). Equivalenza esatta col pruning fisico per selezione e pesi
(design §1.1): il bias entra solo nella selezione top-6, mai nei pesi.

Uso:
  python scripts/reap_bias_mask_ds4.py --gguf PATH --maskfile reap_mask_ds4_domain.json \
      --apply reap      # keep-list della mask
  ... --apply random    # keep-list del random_control (stesso K, seed loggato)
  ... --apply random --random-seed N   # rigenera il random a pari K con seed N
      (STESSO algoritmo della mask: random.Random(N), rng.sample(range(E), keep_n)
       in sequenza sui layer potabili in ordine crescente -> seed 0 riproduce
       bit-exact il random_control della mask; il keep generato viene salvato in
       <gguf>.random_seedN.json per provenienza)
  ... --restore         # ripristina i bias originali dal backup .biasbak.json
  ... --verify          # confronta i bias su file con la config attesa (nessuna scrittura)

Il backup (offset -> base64 dei 1024 byte originali) va in <gguf>.biasbak.json alla
prima --apply e NON viene sovrascritto se esiste gia'.
"""
import argparse
import base64
import json
import os
import random
import struct

MINUS_INF = struct.pack("<f", -1e9)


def regen_random_keep(mask, seed):
    """Replica reap_saliency_ds4.py: un solo rng, sample per layer in ordine."""
    keep_n = int(mask["keep_n"])
    n_expert = int(mask["n_expert"])
    prunable = sorted(int(l) for l in mask["keep"])
    rng = random.Random(seed)
    return {str(l): sorted(rng.sample(range(n_expert), keep_n)) for l in prunable}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    S = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        read_str(f)
    elif t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        if et == 8:
            for _ in range(n):
                read_str(f)
        else:
            f.seek(S[et] * n, 1)
    else:
        f.seek(S[t], 1)


def gguf_directory(path):
    """Ritorna (tensors {name: (dims, type, abs_offset)}, data_start)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        struct.unpack("<I", f.read(4))
        nt, nk = struct.unpack("<QQ", f.read(16))
        align = 32
        for _ in range(nk):
            k = read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            if k == "general.alignment":
                align = struct.unpack("<I", f.read(4))[0] if t == 4 else align
                continue
            skip_val(f, t)
        entries = []
        for _ in range(nt):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            tt, off = struct.unpack("<IQ", f.read(12))
            entries.append((name, dims, tt, off))
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
    return {n: (d, t, data_start + o) for n, d, t, o in entries}, data_start


def bias_offsets(tensors):
    """{layer: abs_offset} dei tensori exp_probs_b (F32 [256])."""
    out = {}
    for name, (dims, tt, off) in tensors.items():
        if name.startswith("blk.") and name.endswith(".exp_probs_b.bias"):
            l = int(name.split(".")[1])
            assert tt == 0 and len(dims) == 1, (name, dims, tt)
            out[l] = (off, int(dims[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--maskfile")
    ap.add_argument("--apply", choices=["reap", "random"])
    ap.add_argument("--random-seed", type=int, default=None,
                    help="rigenera il random keep con questo seed (default: usa random_control della mask)")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    bak_path = a.gguf + ".biasbak.json"

    tensors, _ = gguf_directory(a.gguf)
    offs = bias_offsets(tensors)
    print(f"exp_probs_b trovati: {len(offs)} layer ({min(offs)}..{max(offs)})")

    if a.restore:
        bak = json.load(open(bak_path))
        with open(a.gguf, "r+b") as f:
            for l_str, ent in bak.items():
                f.seek(ent["offset"])
                f.write(base64.b64decode(ent["data"]))
        print(f"ripristinati {len(bak)} layer da {bak_path}")
        return

    mask = json.load(open(a.maskfile))

    def random_keep():
        if a.random_seed is None:
            return mask["random_control"]["keep"]
        keep = regen_random_keep(mask, a.random_seed)
        side = a.gguf + f".random_seed{a.random_seed}.json"
        json.dump({"seed": a.random_seed, "keep_n": mask["keep_n"],
                   "keep": keep}, open(side, "w"))
        print(f"random keep seed={a.random_seed} salvato in {side}")
        if a.random_seed == int(mask["random_control"]["seed"]):
            assert keep == {k: list(v) for k, v in mask["random_control"]["keep"].items()}, \
                "regen seed non riproduce il random_control della mask!"
        return keep

    keep_src = mask["keep"] if a.apply == "reap" or a.verify else random_keep()
    if a.verify and not a.apply:
        raise SystemExit("--verify richiede anche --apply reap|random per sapere la config attesa")

    if a.apply and not a.verify:
        if not os.path.exists(bak_path):
            bak = {}
            with open(a.gguf, "rb") as f:
                for l, (off, n) in offs.items():
                    f.seek(off)
                    bak[str(l)] = {"offset": off,
                                   "data": base64.b64encode(f.read(4 * n)).decode()}
            json.dump(bak, open(bak_path, "w"))
            print(f"backup bias -> {bak_path}")
        n_written = 0
        with open(a.gguf, "r+b") as f:
            for l_str, kept in keep_src.items():
                l = int(l_str)
                if l not in offs:
                    raise SystemExit(f"layer {l} nella mask ma senza exp_probs_b nel gguf")
                off, n = offs[l]
                kept_set = set(kept)
                for e in range(n):
                    if e not in kept_set:
                        f.seek(off + 4 * e)
                        f.write(MINUS_INF)
                        n_written += 1
        print(f"apply {a.apply}: scritti {n_written} bias a -1e9 "
              f"su {len(keep_src)} layer")
        return

    if a.verify:
        bad = 0
        with open(a.gguf, "rb") as f:
            for l_str, kept in keep_src.items():
                off, n = offs[int(l_str)]
                f.seek(off)
                vals = struct.unpack(f"<{n}f", f.read(4 * n))
                kept_set = set(kept)
                for e, v in enumerate(vals):
                    masked = v <= -1e8
                    if masked == (e in kept_set):
                        bad += 1
        print("VERIFY OK" if bad == 0 else f"VERIFY FAIL: {bad} posizioni incoerenti")


if __name__ == "__main__":
    main()
