"""Read GGUF header/tensor directory only; report aligned storage spans, not VRAM fit."""
import json
import math
import struct
from pathlib import Path

MODEL = Path(r"D:\models\qwen36moe\Qwen3.6-35B-A3B-Q4_K_M.gguf")
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

def main():
    with MODEL.open('rb') as f:
        def read(fmt):
            size = struct.calcsize('<' + fmt)
            raw = f.read(size)
            if len(raw) != size: raise ValueError('Truncated header')
            return struct.unpack('<' + fmt, raw)[0]
        def string(keep=True):
            n = read('Q')
            if n > MODEL.stat().st_size: raise ValueError('Invalid string length')
            if not keep:
                f.seek(n, 1)
                return None
            return f.read(n).decode('utf-8')
        def value(kind, keep=False):
            if kind == 8: return string(keep)
            if kind == 9:
                sub, n = read('I'), read('Q')
                if sub in SIZES and not keep:
                    f.seek(SIZES[sub]*n, 1)
                    return None
                values = []
                for _ in range(n):
                    item = value(sub, keep)
                    if keep: values.append(item)
                return values if keep else None
            if kind not in FORMATS: raise ValueError('Unknown GGUF type')
            return read(FORMATS[kind]) if keep else f.seek(SIZES[kind], 1)
        if f.read(4) != b'GGUF': raise ValueError('Not GGUF')
        version, tensors, count = read('I'), read('Q'), read('Q')
        if version not in (2, 3): raise ValueError('Unsupported version')
        metadata = {}
        for _ in range(count):
            key, kind = string(), read('I')
            keep = key in ('general.architecture', 'general.alignment', 'qwen35moe.block_count',
                           'qwen35moe.expert_count', 'qwen35moe.expert_used_count')
            result = value(kind, keep)
            if keep: metadata[key] = result
        entries = []
        for _ in range(tensors):
            name, ndims = string(), read('I')
            dims = [read('Q') for _ in range(ndims)]
            quant, offset = read('I'), read('Q')
            entries.append({'name': name, 'dims': dims, 'quant_type': quant, 'offset': offset})
        alignment = metadata.get('general.alignment', 32)
        data_start = ((f.tell() + alignment - 1) // alignment) * alignment
    entries.sort(key=lambda row: row['offset'])
    totals = {'routed': {'aligned_bytes': 0, 'parameters': 0, 'tensors': 0},
              'nonrouted': {'aligned_bytes': 0, 'parameters': 0, 'tensors': 0}}
    types = {}
    for i, row in enumerate(entries):
        end = entries[i+1]['offset'] if i + 1 < len(entries) else MODEL.stat().st_size - data_start
        span = end - row['offset']
        routed = any(part in row['name'] for part in ('.ffn_up_exps.', '.ffn_down_exps.', '.ffn_gate_exps.', '.ffn_gate_up_exps.'))
        group = totals['routed' if routed else 'nonrouted']
        group['aligned_bytes'] += span
        group['parameters'] += math.prod(row['dims'])
        group['tensors'] += 1
        types[str(row['quant_type'])] = types.get(str(row['quant_type']), 0) + 1
    result = {'schema_version': 1, 'model_path': str(MODEL), 'file_bytes': MODEL.stat().st_size,
              'metadata': metadata, 'tensor_data_start': data_start, 'quant_type_tensor_counts': types,
              'totals': totals, 'cache32_aligned_payload_estimate': totals['routed']['aligned_bytes'] / 8,
              'note': 'Header-only read. Byte spans include alignment, not measured VRAM. Excludes KV/state/workspace/staging/OS. No new model hash computed.'}
    out = Path(__file__).with_name('gguf-memory-layout.json')
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
