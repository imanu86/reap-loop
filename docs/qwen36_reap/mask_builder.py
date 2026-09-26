"""Strict Qwen routing JSONL -> reversible keep-list; never rewrites model weights.

Trace files must contain calibration ONLY (exclude startup/warmup/held-out).
The caller supplies provenance; a split label is not independent proof of dataset isolation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid ' + label)
    return value


def aggregate(paths, model_sha256):
    if not re.fullmatch(r'[0-9a-fA-F]{64}', model_sha256):
        raise ValueError('Expected actual model SHA256')
    geometry = None
    totals = None
    counts = None
    provenance = []
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            first = stream.readline()
            digest.update(first)
            header = json.loads(first)
            if header.get('record_type') != 'header' or header.get('schema_version') != 1:
                raise ValueError('Missing schema1 trace header')
            if header.get('model_sha256', '').lower() != model_sha256.lower():
                raise ValueError('Checkpoint identity mismatch')
            shape = tuple(integer(header.get(k), k, 1) for k in ('layer_count', 'expert_count', 'top_k'))
            if shape != (40, 256, 8) or header.get('architecture') != 'qwen35moe':
                raise ValueError('Pilot restricted to verified 40-layer Qwen256/top8 without MTP')
            if geometry is not None and shape != geometry:
                raise ValueError('Mixed trace geometry')
            geometry = shape
            layers, experts, top_k = shape
            if totals is None:
                totals = [[0.0] * experts for _ in range(layers)]
                counts = [0] * layers
            batch = None
            coverage = {}
            tokens = {}
            rows = 0

            def finish_batch():
                if set(coverage) != set(range(layers)):
                    raise ValueError('Incomplete routing batch: missing layers')
                reference = coverage[0]
                if not reference or any(coverage[layer] != reference for layer in range(layers)):
                    raise ValueError('Routing token alignment differs between layers')

            for raw in stream:
                digest.update(raw)
                row = json.loads(raw)
                if row.get('record_type') != 'route' or row.get('schema_version') != 1:
                    raise ValueError('Unexpected record/error in trace')
                current = integer(row.get('batch_id'), 'batch_id')
                if batch is not None and current != batch:
                    if current < batch:
                        raise ValueError('Non-monotonic batch IDs')
                    finish_batch()
                    coverage = {}
                    tokens = {}
                batch = current
                layer = integer(row.get('layer'), 'layer')
                index = integer(row.get('token_index'), 'token_index')
                position = integer(row.get('token_position'), 'token_position')
                if layer >= layers or row.get('phase') not in ('prefill', 'decode', 'unknown'):
                    raise ValueError('Invalid layer or phase')
                seqs = row.get('seq_ids')
                if not isinstance(seqs, list) or not seqs or any(type(x) is not int or x < 0 for x in seqs) or len(set(seqs)) != len(seqs):
                    raise ValueError('Invalid sequence IDs')
                identity = (position, row.get('token_id'), tuple(seqs), row['phase'])
                if index in tokens and tokens[index] != identity:
                    raise ValueError('Token identity differs across layers')
                tokens[index] = identity
                seen = coverage.setdefault(layer, set())
                if index in seen:
                    raise ValueError('Duplicate layer/token record')
                seen.add(index)
                ids, weights = row.get('ids'), row.get('weights')
                if not isinstance(ids, list) or not isinstance(weights, list) or len(ids) != top_k or len(weights) != top_k:
                    raise ValueError('Wrong top-k shape')
                if any(type(e) is not int or not 0 <= e < experts for e in ids) or len(set(ids)) != top_k:
                    raise ValueError('Invalid or duplicate expert IDs')
                if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
                    raise ValueError('Invalid routing weights')
                if abs(math.fsum(weights) - 1.0) > 0.001:
                    raise ValueError('Expected normalized Qwen weights summing to1')
                for expert, weight in zip(ids, weights):
                    totals[layer][expert] += weight
                counts[layer] += 1
                rows += 1
            if not rows:
                raise ValueError('Empty routing trace')
            finish_batch()
        provenance.append({'file': str(path), 'sha256': digest.hexdigest(), 'rows': rows})
    if totals is None:
        raise ValueError('No traces provided')
    return totals, counts, provenance


def build(paths, model_sha256, *, split, keep=None, coverage=0.95, random_seed=None):
    if split != 'calibration':
        raise ValueError('Held-out/evaluation traces must never select experts')
    if keep is not None and (type(keep) is not int or not 8 <= keep <= 256):
        raise ValueError('Keep must preserve top8 and stay within256')
    if not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError('Invalid coverage target')
    scores, counts, provenance = aggregate(paths, model_sha256)
    rng = random.Random(random_seed)
    result = {'schema_version': 1, 'model_sha256': model_sha256.lower(), 'architecture': 'qwen35moe',
              'expert_count': 256, 'top_k': 8, 'layer_count': 40, 'layers': {},
              'method': 'mass_gate' if random_seed is None else 'random_matched_pool',
              'calibration_split': split, 'source_traces': provenance, 'diagnostics': {},
              'random_seed': random_seed, 'requested_keep': keep, 'requested_coverage': coverage}
    for layer, mass in enumerate(scores):
        total = math.fsum(mass)
        ranked = sorted(range(256), key=lambda e: (-mass[e], e))
        n = keep
        if n is None:
            running, n = 0.0, 0
            while n < 256 and (n < 8 or running < coverage * total):
                running += mass[ranked[n]]
                n += 1
        selected = ranked[:n] if random_seed is None else rng.sample(range(256), n)
        result['layers'][str(layer)] = sorted(selected)
        result['diagnostics'][str(layer)] = {'tokens': counts[layer], 'kept': n,
            'observed_experts': sum(m > 0 for m in mass),
            'retained_mass_fraction': math.fsum(mass[e] for e in selected) / total,
            'zero_observed_mass_kept': sum(mass[e] == 0 for e in selected)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, action='append', required=True)
    p.add_argument('--model-sha256', required=True)
    p.add_argument('--split', choices=['calibration'], required=True)
    p.add_argument('--keep', type=int)
    p.add_argument('--coverage', type=float, default=0.95)
    p.add_argument('--random-seed', type=int)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = build(a.trace, a.model_sha256, split=a.split, keep=a.keep,
                   coverage=a.coverage, random_seed=a.random_seed)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
