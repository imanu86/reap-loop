"""Strict Qwen routing JSONL -> reversible keep-list; never rewrites model weights.

Trace files must contain calibration ONLY (exclude startup/warmup/held-out).
The caller supplies provenance; a split label is not independent proof of dataset isolation.
Writer contract audited in isolated src/llama-reap.cpp: next_decode()/begin()/finish().
Counters are positive uint64; one ubatch has one decode_call_id and dense token indices.
Token/position/sequence IDs are nonnegative int32, but the header has no vocabulary size.
There is no total-count/end-of-stream marker: whole missing batches or a suffix missing
from ALL layers cannot be proved absent. Counter gaps are allowed (e.g. skipped warmup).
No semantic phase or cross-ubatch position ordering is inferred from batch size.
Explicit mean_selected_gate is a conditional selected-occurrence g-only ablation;
mass_gate sums gates and remains the legacy default. Neither includes output norms
or claims full REAP. Zero-weight selected occurrences still count in the denominator.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re


INT32_MAX = (1 << 31) - 1
UINT64_MAX = (1 << 64) - 1
RANKING_METHODS = ('mass_gate', 'mean_selected_gate')


def selection_recipe(ranking_method):
    """Explicit g-only recipes; neither measures expert-output norm/full REAP."""
    if ranking_method not in RANKING_METHODS:
        raise ValueError('Unknown ranking_method; choose mass_gate or mean_selected_gate')
    return {'schema_version': 1, 'ranking_method': ranking_method,
            'scope': 'g_only_ablation_not_full_REAP', 'includes_expert_output_norm': False,
            'score_formula': 'sum_selected_gate' if ranking_method == 'mass_gate' else 'sum_selected_gate / selected_count',
            'selected_count_definition': 'top8 occurrences including zero gate weights',
            'unobserved_score': 0, 'tie_break': 'score_descending_then_expert_id_ascending',
            'pool_rule': 'fixed_keep_else_actual_gate_mass_coverage_in_rank_order',
            'retained_mass_definition': 'actual_sum_selected_gate_not_sum_of_ranking_scores'}


def recipe_digest(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def integer(value, label, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError('Invalid ' + label)
    return value


def validate_sha(model_sha256):
    if not isinstance(model_sha256, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', model_sha256):
        raise ValueError('Expected actual model SHA256')


def trace_object(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate trace JSON key: ' + key)
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('Expected trace JSON object')
    return value


def aggregate(paths, model_sha256, *, include_selected_counts=False):
    """Legacy triple by default; opt-in fourth item is actual per-expert selection counts."""
    if type(include_selected_counts) is not bool:
        raise ValueError('include_selected_counts must be bool')
    validate_sha(model_sha256)
    geometry = None
    totals = None
    counts = None
    selected_counts = None
    provenance = []
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            first = stream.readline()
            digest.update(first)
            header = trace_object(first)
            if header.get('record_type') != 'header' or integer(header.get('schema_version'), 'schema_version') != 1:
                raise ValueError('Missing schema1 trace header')
            validate_sha(header.get('model_sha256'))
            if header['model_sha256'].lower() != model_sha256.lower():
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
                if include_selected_counts:
                    selected_counts = [[0] * experts for _ in range(layers)]
            batch = None
            decode_call = None
            coverage = {}
            tokens = {}
            rows = 0

            def finish_batch():
                if set(coverage) != set(range(layers)):
                    raise ValueError('Incomplete routing batch: missing layers')
                reference = coverage[0]
                if not reference or any(coverage[layer] != reference for layer in range(layers)):
                    raise ValueError('Routing token alignment differs between layers')
                if reference != set(range(len(reference))):
                    raise ValueError('Routing token indices must be dense from zero')

            for raw in stream:
                digest.update(raw)
                row = trace_object(raw)
                if row.get('record_type') != 'route' or integer(row.get('schema_version'), 'schema_version') != 1:
                    raise ValueError('Unexpected record/error in trace')
                current = integer(row.get('batch_id'), 'batch_id', 1, UINT64_MAX)
                current_call = integer(row.get('decode_call_id'), 'decode_call_id', 1, UINT64_MAX)
                if batch is not None:
                    if current == batch and current_call != decode_call:
                        raise ValueError('Decode call differs within routing batch')
                    if current != batch:
                        if current < batch:
                            raise ValueError('Non-monotonic batch IDs')
                        if current_call < decode_call:
                            raise ValueError('Non-monotonic decode call IDs')
                        finish_batch()
                        coverage = {}
                        tokens = {}
                batch, decode_call = current, current_call
                layer = integer(row.get('layer'), 'layer')
                index = integer(row.get('token_index'), 'token_index', 0, INT32_MAX)
                position = integer(row.get('token_position'), 'token_position', 0, INT32_MAX)
                token_id = integer(row.get('token_id'), 'token_id', 0, INT32_MAX)
                if layer >= layers or row.get('phase') not in ('prefill', 'decode', 'unknown'):
                    raise ValueError('Invalid layer or phase')
                seqs = row.get('seq_ids')
                if not isinstance(seqs, list) or not seqs or any(type(x) is not int or not 0 <= x <= INT32_MAX for x in seqs) or len(set(seqs)) != len(seqs):
                    raise ValueError('Invalid sequence IDs')
                identity = (position, token_id, tuple(seqs), row['phase'], current_call)
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
                    if selected_counts is not None:
                        selected_counts[layer][expert] += 1
                counts[layer] += 1
                rows += 1
            if not rows:
                raise ValueError('Empty routing trace')
            finish_batch()
        provenance.append({'file': str(path), 'sha256': digest.hexdigest(), 'rows': rows})
    if totals is None:
        raise ValueError('No traces provided')
    if include_selected_counts:
        return totals, counts, provenance, selected_counts
    return totals, counts, provenance


def validate_selection(split, keep, coverage):
    if split != 'calibration':
        raise ValueError('Held-out/evaluation traces must never select experts')
    if keep is not None and (type(keep) is not int or not 8 <= keep <= 256):
        raise ValueError('Keep must preserve top8 and stay within256')
    if type(coverage) not in (int, float) or not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError('Invalid coverage target')


def build(paths, model_sha256, *, split, keep=None, coverage=0.95, random_seed=None,
          ranking_method='mass_gate'):
    """Default mass_gate serialization remains legacy-identical; mean is explicit opt-in."""
    selection_recipe(ranking_method)
    validate_selection(split, keep, coverage)  # reject held-out before any trace I/O
    data = aggregate(paths, model_sha256, include_selected_counts=ranking_method == 'mean_selected_gate')
    return build_from_aggregate(data, model_sha256, split=split, keep=keep,
        coverage=coverage, random_seed=random_seed, ranking_method=ranking_method)


def build_from_aggregate(validated_aggregate, model_sha256, *, split, keep=None,
                         coverage=0.95, random_seed=None, ranking_method='mass_gate'):
    """Pure selection from ONE aggregate(paths, model_sha256) result; never opens files.

    mean_selected_gate is a g-only ablation, NOT full REAP. It requires the explicit
    fourth selected-count matrix from aggregate(..., include_selected_counts=True);
    no old aggregate/mask is silently upgraded or its denominator reconstructed.

    Caller must supply the unmodified validated triple/quadruple with the SAME SHA.
    Structural checks below catch corruption, not provenance fabrication: neither
    aggregate form cryptographically binds its separately supplied model_sha256.
    It can be reused for K128/K64/dynamic and seeded matched controls without rereading
    traces. Inputs are not mutated; each returned manifest owns its provenance copy.
    """
    validate_sha(model_sha256)
    recipe = selection_recipe(ranking_method)
    validate_selection(split, keep, coverage)
    if not isinstance(validated_aggregate, (tuple, list)) or len(validated_aggregate) not in (3, 4):
        raise ValueError('Expected validated aggregate triple or count-bearing quadruple')
    scores, counts, provenance = validated_aggregate[:3]
    selected_counts = validated_aggregate[3] if len(validated_aggregate) == 4 else None
    if ranking_method == 'mean_selected_gate' and selected_counts is None:
        raise ValueError('mean_selected_gate requires actual selected counts; no legacy aggregate upgrade')
    if not isinstance(scores, (tuple, list)) or len(scores) != 40 or not isinstance(counts, (tuple, list)) or len(counts) != 40:
        raise ValueError('Invalid aggregate geometry')
    for layer, mass in enumerate(scores):
        integer(counts[layer], 'aggregate token count', 1)
        if not isinstance(mass, (tuple, list)) or len(mass) != 256:
            raise ValueError('Invalid aggregate expert dimension')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in mass):
            raise ValueError('Invalid aggregate mass')
        total = math.fsum(mass)
        if not math.isfinite(total) or total <= 0:
            raise ValueError('Empty/nonfinite aggregate mass')
    if selected_counts is not None:
        if not isinstance(selected_counts, (tuple, list)) or len(selected_counts) != 40:
            raise ValueError('Invalid selected-count layer dimension')
        for layer, row in enumerate(selected_counts):
            if not isinstance(row, (tuple, list)) or len(row) != 256:
                raise ValueError('Invalid selected-count expert dimension')
            for expert, n in enumerate(row):
                integer(n, 'selected count', 0, counts[layer])
                if n == 0 and scores[layer][expert] != 0:
                    raise ValueError('Unobserved expert has nonzero gate mass')
            if sum(row) != counts[layer] * 8:
                raise ValueError('Selected-count total must equal token count times top8')
    if len(set(counts)) != 1:
        raise ValueError('Aggregate token counts differ between layers')
    if not isinstance(provenance, (tuple, list)) or not provenance:
        raise ValueError('Missing validated trace provenance')
    for source in provenance:
        if not isinstance(source, dict) or not isinstance(source.get('file'), str):
            raise ValueError('Invalid trace provenance')
        validate_sha(source.get('sha256'))
        integer(source.get('rows'), 'provenance row count', 1)
    if sum(source['rows'] for source in provenance) != sum(counts):
        raise ValueError('Aggregate/provenance row count mismatch')
    rng = random.Random(random_seed)
    result = {'schema_version': 1, 'model_sha256': model_sha256.lower(), 'architecture': 'qwen35moe',
              'expert_count': 256, 'top_k': 8, 'layer_count': 40, 'layers': {},
              'method': ranking_method if random_seed is None else 'random_matched_pool',
              'calibration_split': split, 'source_traces': copy.deepcopy(provenance), 'diagnostics': {},
              'random_seed': random_seed, 'requested_keep': keep, 'requested_coverage': coverage}
    if ranking_method != 'mass_gate':
        result.update(ranking_method=ranking_method, selection_recipe=recipe,
                      selection_recipe_sha256=recipe_digest(recipe), quality_approval=False)
    for layer, mass in enumerate(scores):
        total = math.fsum(mass)
        ranking_scores = mass if ranking_method == 'mass_gate' else [
            mass[e] / selected_counts[layer][e] if selected_counts[layer][e] else 0.0 for e in range(256)]
        ranked = sorted(range(256), key=lambda e: (-ranking_scores[e], e))
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
        if ranking_method == 'mean_selected_gate':
            result['diagnostics'][str(layer)].update(
                retained_gate_mass=math.fsum(mass[e] for e in selected), total_gate_mass=total,
                selected_count_total=sum(selected_counts[layer]),
                experts_with_selected_observations=sum(n > 0 for n in selected_counts[layer]),
                unobserved_experts_kept=sum(selected_counts[layer][e] == 0 for e in selected))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, action='append', required=True)
    p.add_argument('--model-sha256', required=True)
    p.add_argument('--split', choices=['calibration'], required=True)
    p.add_argument('--keep', type=int)
    p.add_argument('--coverage', type=float, default=0.95)
    p.add_argument('--random-seed', type=int)
    p.add_argument('--ranking-method', choices=RANKING_METHODS, default='mass_gate',
                   help='Explicit g-only recipe, NOT full REAP (default preserves mass_gate)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = build(a.trace, a.model_sha256, split=a.split, keep=a.keep,
                   coverage=a.coverage, random_seed=a.random_seed, ranking_method=a.ranking_method)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
