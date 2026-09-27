"""Offline calibration-only mask planning; no inference or model-weight writes.

Requires a completed, full50 run of the explicitly selected v2/v3 frozen protocol.
Default remains v2; v3 must be requested, never inferred from run metadata.
GPU screen readiness is recomputed from >=40 full completions and coverage of all
10 corpus families; CPU diagnostic masks with missing-family coverage remain unapproved.
The internal protocol_sha256 hashes {expected_config, recorded_source_sha256}; it is
NOT the quality policy's evaluation_protocol digest (different canonical objects).
Recorded source hashes are provenance assertions, not fresh source re-verification.
Selection is separately bound by ranking_method/selection_recipe/selection_recipe_sha256;
mean_selected_gate is an explicit NEW g-only ablation curve, not full REAP or an upgrade
of earlier mass_gate masks. Neither recipe may acquire quality approval here.
Trace validity cannot itself prove that every expected runtime call was recorded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from mask_builder import aggregate, build_from_aggregate, RANKING_METHODS, selection_recipe, recipe_digest

MODEL_SHA = '671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7'
CORE_SHA = '018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef'
POOLS = (128, 96, 64, 32)
RANDOM_SEED = 20260713
FROZEN = dict(protocol='native', policy_version='v2', final_mode='tool', sampling_profile='greedy',
              seed=0, budget=4096, max_output=1024, max_turns=13, thinking='template-default', diagnostic_raw=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def load(path):
    return parse(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen_protocol(policy_version):
    require(policy_version in ('v2', 'v3'), 'Explicit policy version must be v2 or v3')
    return dict(FROZEN, policy_version=policy_version)


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def source_digest(value, label):
    require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value), 'Missing/invalid recorded ' + label)
    return value


def validate_run(run, corpus, *, policy_version='v2'):
    frozen = frozen_protocol(policy_version)  # chosen by caller, NEVER inferred from artifacts
    run, corpus = Path(run).resolve(), Path(corpus).resolve()
    raw_corpus = corpus.read_bytes()
    require(raw_corpus.endswith(b'\n'), 'Uncommitted calibration corpus tail')
    episodes = [parse(line) for line in raw_corpus.splitlines()]
    require(all(type(e) is dict for e in episodes), 'Malformed calibration corpus')
    expected = [episode.get('id') for episode in episodes]
    require(len(expected) == 50 and all(type(i) is str and i.startswith('calibration-') for i in expected) and
            len(set(expected)) == 50, 'Require full calibration50 corpus')
    require(all(type(e.get('family')) is str and bool(e['family'].strip()) for e in episodes),
            'Missing/malformed calibration family')
    families_by_id = {e['id']: e['family'] for e in episodes}
    families = sorted(set(families_by_id.values()))
    require(len(families) == 10, 'Require exactly10 calibration families')
    family_counts = {family: {'cases': 0, 'full_completions': 0} for family in families}
    for family in families_by_id.values():
        family_counts[family]['cases'] += 1
    m = load(run / 'manifest.json')
    p = load(run / 'pilot/manifest.json')
    end = load(run / 'shutdown.json')  # absent while writer is active => reject
    summary = load(run / 'pilot/summary.json')
    require(end.get('stopped') is True and type(end.get('pid')) is int, 'Server not confirmed stopped')
    require(load(run / 'server.pid.json')['pid'] == end['pid'], 'Shutdown PID mismatch')
    require(m.get('scope') == 'calibration_integration_not_performance' and
            m.get('eligible_quality_evaluation') is True and m.get('skip_chat_parsing') is False,
            'Diagnostic/bypass run cannot select masks')
    require(m.get('model_sha256') == MODEL_SHA and m.get('runtime_arm') == 'candidate' and
            m.get('runtime_sha256', {}).get('llama.dll') == CORE_SHA, 'Unexpected model/runtime')
    require(m.get('trace_enabled') is True, 'Routing capture required')
    require(m.get('policy_version') == policy_version, 'Coordinator policy version mismatch')
    recorded_sources = {
        'pilot_runner_sha256': source_digest(p.get('runner_sha256'), 'pilot runner SHA256'),
        'coordinator_sha256': source_digest(m.get('coordinator_sha256'), 'coordinator SHA256'),
    }
    env = m['environment']
    require('QWEN36_REAP_MASK' not in env, 'Calibration must be unmasked')
    require(Path(env['QWEN36_REAP_TRACE']).resolve() == run / 'routing.jsonl', 'Trace path mismatch')
    require(p.get('split') == 'calibration' and p.get('episode_ids') == expected and
            p.get('dataset_sha256') == digest(corpus), 'Corpus identity/order mismatch')
    defaults = p.get('server_props', {}).get('default_generation_settings', {})
    require(defaults.get('n_ctx') == 4096 and defaults.get('params', {}).get('speculative.types') == 'none',
            'No-speculation context contract unverified')
    source_config = p.get('config')
    require(type(source_config) is dict and
            all(type(source_config.get(k)) is type(v) and source_config[k] == v for k, v in frozen.items()),
            'Frozen protocol mismatch')
    protocol_binding = {'expected_config': frozen, 'recorded_source_sha256': recorded_sources}
    protocol_sha256 = object_digest(protocol_binding)
    require(summary.get('cases') == 50 and summary.get('split') == 'calibration' and
            summary.get('real_actions_executed') == 0, 'Incomplete or wrong-split summary')
    ids, successes, minimum_decode_tokens = [], 0, 0
    with (run / 'pilot/transcripts.jsonl').open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'Uncommitted transcript tail')
            row = parse(line)
            ids.append(row['id'])
            require(row.get('split') == 'calibration' and row.get('protocol') == 'native' and
                    row.get('policy_version') == policy_version and row.get('final_mode') == 'tool', 'Case protocol mismatch')
            require(row['id'] in families_by_id and row.get('family') == families_by_id[row['id']],
                    'Case family differs from calibration corpus')
            require(type(row.get('full_completion')) is bool and row.get('real_actions_executed') == 0,
                    'Malformed outcome')
            require(row.get('error_class') not in ('preflight', 'transport'), 'Infrastructure error requires a clean rerun')
            turns = row.get('turns')
            require(type(turns) is list and bool(turns), 'No actual generation')
            for turn in turns:
                request = turn.get('request', {})
                required = dict(temperature=0, top_k=1, top_p=1, min_p=0, seed=0, max_tokens=1024,
                                tool_choice='auto', parallel_tool_calls=False)
                require(not any(k.startswith('speculative') for k in request), 'Unexpected speculation override')
                require(all(type(request.get(k)) is type(v) and request[k] == v for k, v in required.items()),
                        'Actual request differs from frozen sampling')
                final_defs = [t.get('function', {}) for t in request.get('tools', [])
                              if t.get('function', {}).get('name') == 'final']
                require(len(final_defs) == 1 and type(final_defs[0].get('parameters')) is dict,
                        'Missing actual public terminal tool')
                response = turn.get('response')
                require(type(response) is dict and response.get('choices'), 'Missing actual response')
                generated = response.get('usage', {}).get('completion_tokens')
                require(type(generated) is int and generated >= 0, 'Missing actual generated-token count')
                minimum_decode_tokens += max(0, generated - 1)  # noMTP/no speculation in this frozen run
                require(response.get('usage', {}).get('prompt_tokens') ==
                        turn['preflight']['prompt_tokens_preflight'], 'Unverified prompt alignment')
            if row['full_completion']:
                require(row.get('error_class') is None and turns[-1].get('selection', {}).get('kind') == 'final' and
                        turns[-1]['selection']['value'] == row['final'], 'Inconsistent success')
                successes += 1
                family_counts[row['family']]['full_completions'] += 1
    require(ids == expected, 'Missing/duplicate/out-of-order transcript cases')
    rate = summary.get('full_completion_rate')
    require(type(rate) in (float, int) and math.isfinite(rate) and abs(rate - successes / 50) < 1e-12,
            'Summary disagrees with actual records')
    require(successes >= 40, 'Baseline below calibration40/50 readiness floor')
    files = ['manifest.json', 'shutdown.json', 'server.pid.json', 'pilot/manifest.json',
             'pilot/summary.json', 'pilot/transcripts.jsonl']
    missing_families = [family for family in families if family_counts[family]['full_completions'] == 0]
    return {'run': str(run), 'successes': successes, 'cases': 50, 'minimum_decode_tokens_per_layer': minimum_decode_tokens,
            'policy_version': policy_version, 'expected_config': frozen,
            'full_source_config': source_config, 'full_source_config_sha256': object_digest(source_config),
            'recorded_source_sha256': recorded_sources, 'protocol_sha256': protocol_sha256,
            'source_hash_provenance': 'recorded_run_manifests_not_independent_source_reverification',
            'family_counts': family_counts, 'families_without_full_completion': missing_families,
            'gpu_screen_ready': successes >= 40 and not missing_families,
            'corpus_sha256': digest(corpus), 'inputs_sha256': {f: digest(run / f) for f in files}}


def plan(run, corpus, out, *, policy_version='v2', ranking_method='mass_gate'):
    # Selection recipe is separate from captured source protocol; no old masks are upgraded.
    recipe = dict(selection_recipe(ranking_method), pools=list(POOLS), random_seed=RANDOM_SEED)
    recipe_sha256 = recipe_digest(recipe)
    run, out = Path(run).resolve(), Path(out).resolve()
    require(not out.exists(), 'Output directory must be NEW')
    proof = validate_run(run, corpus, policy_version=policy_version)
    trace = run / 'routing.jsonl'
    before = trace.stat()
    data = aggregate([trace], MODEL_SHA, include_selected_counts=True)
    require(all(n >= proof['minimum_decode_tokens_per_layer'] for n in data[1]),
            'Trace smaller than no-speculation generation lower bound')
    after = trace.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Trace changed during validation')
    masks, reports = {}, []
    for keep in POOLS:
        for seed in (None, RANDOM_SEED):
            kind = 'ranked' if seed is None else 'random'
            mask = build_from_aggregate(data, MODEL_SHA, split='calibration', keep=keep,
                                        random_seed=seed, ranking_method=ranking_method)
            mask.update(quality_approval=False, gpu_screen_ready=proof['gpu_screen_ready'],
                        policy_version=policy_version, protocol_sha256=proof['protocol_sha256'],
                        ranking_method=ranking_method, selection_recipe=dict(recipe),
                        selection_recipe_sha256=recipe_sha256)
            name = f'{kind}-k{keep}.json'
            masks[name] = mask
            mass = [d['retained_mass_fraction'] for d in mask['diagnostics'].values()]
            reports.append({'file': name, 'keep': keep, 'method': mask['method'],
                            'ranking_method': ranking_method, 'selection_recipe_sha256': recipe_sha256,
                            'quality_approval': False, 'gpu_screen_ready': proof['gpu_screen_ready'],
                            'policy_version': policy_version, 'protocol_sha256': proof['protocol_sha256'],
                            'retained_mass_min': min(mass), 'retained_mass_mean': math.fsum(mass) / len(mass),
                            'retained_mass_max': max(mass)})
    out.mkdir(parents=True, exist_ok=False)
    for name, value in masks.items():
        with (out / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    report = {'schema_version': 1, 'scope': 'calibration_only_unapproved_masks',
              'quality_approval': False, 'model_weights_written': False, 'provenance': proof,
              'policy_version': policy_version, 'gpu_screen_ready': proof['gpu_screen_ready'],
              'protocol_sha256': proof['protocol_sha256'],
              'ranking_method': ranking_method, 'selection_recipe': recipe,
              'selection_recipe_sha256': recipe_sha256,
              'trace_provenance': data[2], 'planner_sha256': digest(__file__), 'candidates': reports,
              'mask_sha256': {name: digest(out / name) for name in masks}}
    with (out / 'plan.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=Path(__file__).parent / 'pilot/calibration.jsonl')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy-version', choices=('v2', 'v3'), default='v2',
                        help='Expected frozen protocol; never inferred from run artifacts (default: v2)')
    parser.add_argument('--ranking-method', choices=RANKING_METHODS, default='mass_gate',
                        help='Explicit g-only ablation recipe; neither choice is full REAP')
    args = parser.parse_args()
    report = plan(args.run, args.corpus, args.out, policy_version=args.policy_version,
                  ranking_method=args.ranking_method)
    print(json.dumps({'scope': report['scope'], 'policy_version': report['policy_version'],
                      'ranking_method': report['ranking_method'], 'selection_recipe_sha256': report['selection_recipe_sha256'],
                      'gpu_screen_ready': report['gpu_screen_ready'], 'quality_approval': False,
                      'candidates': report['candidates']}))
