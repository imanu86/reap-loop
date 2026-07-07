"""Mini parser gguf: header + KV + tensor directory (no payload)."""
import struct, sys, json

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

def read_val(f, t):
    S = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
    if t == 8:
        return read_str(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        return [read_val(f, et) for _ in range(n)] if n < 1000 else (f.seek(0,1), skip_arr(f, et, n))[1]
    return struct.unpack(S[t], f.read(struct.calcsize(S[t])))[0]

def skip_arr(f, et, n):
    S = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if et == 8:
        for _ in range(n): read_str(f)
    else:
        f.seek(S[et]*n, 1)
    return f"<array len {n}>"

def inspect(path, dump_prefixes):
    out = {'path': path, 'kv': {}, 'tensors': []}
    with open(path, 'rb') as f:
        magic, ver = f.read(4), struct.unpack('<I', f.read(4))[0]
        assert magic == b'GGUF', magic
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
        out['version'] = ver; out['n_tensors'] = n_tensors
        for _ in range(n_kv):
            k = read_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            v = read_val(f, t)
            out['kv'][k] = v
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{nd}Q', f.read(8*nd))
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            out['tensors'].append((name, list(dims), ttype, off))
    print(f"== {path}: gguf v{ver}, {n_tensors} tensors")
    for k, v in out['kv'].items():
        if 'expert' in k or 'block_count' in k or 'hash' in k or k.startswith('general.'):
            print(f"  KV {k} = {v if not isinstance(v, list) else v[:8]}")
    seen = {}
    for name, dims, ttype, off in out['tensors']:
        for p in dump_prefixes:
            if p in name:
                key = name.split('.')
                base = '.'.join(key[2:]) if key[0]=='blk' else name
                if base not in seen:
                    seen[base] = (name, dims, ttype)
    print("  -- tensori campione (primo per tipo):")
    for base, (name, dims, ttype) in sorted(seen.items()):
        print(f"  {name:42s} dims={dims} type={ttype}")
    # conta exp_probs_b e tid2eid per layer
    npb = sum(1 for n,_,_,_ in out['tensors'] if 'exp_probs_b' in n)
    ntid = sum(1 for n,_,_,_ in out['tensors'] if 'tid2eid' in n)
    ngexp = sum(1 for n,_,_,_ in out['tensors'] if 'ffn_gate_exps' in n)
    print(f"  exp_probs_b: {npb}   tid2eid: {ntid}   ffn_gate_exps: {ngexp}")
    return out

if __name__ == '__main__':
    inspect('models/ds4/DeepSeek-V4-Flash-IQ2XXS-imatrix.gguf',
            ['ffn_gate_inp','exp_probs_b','ffn_gate_exps','ffn_up_exps','ffn_down_exps','tid2eid'])
    print()
    inspect('models/ds4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf',
            ['ffn_gate_inp','exp_probs_b','ffn_gate_exps','tid2eid'])
