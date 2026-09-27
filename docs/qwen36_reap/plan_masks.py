"""Offline calibration-only mask planning; no inference or model-weight writes.

Requires a completed, full50 run of the frozen terminal protocol. Trace validity
cannot itself prove that every expected runtime call was recorded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from mask_builder import aggregate, build_from_aggregate

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


def validate_run(run, corpus):
    run, corpus = Path(run).resolve(), Path(corpus).resolve()
    expected = [parse(line)['id'] for line in corpus.read_bytes().splitlines()]
    require(len(expected) == 50 and len(set(expected)) == 50 and
            all(type(i) is str and i.startswith('calibration-') for i in expected), 'Require full calibration50 corpus')
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
    env = m['environment']
    require('QWEN36_REAP_MASK' not in env, 'Calibration must be unmasked')
    require(Path(env['QWEN36_REAP_TRACE']).resolve() == run / 'routing.jsonl', 'Trace path mismatch')
    require(p.get('split') == 'calibration' and p.get('episode_ids') == expected and
            p.get('dataset_sha256') == digest(corpus), 'Corpus identity/order mismatch')
    defaults = p.get('server_props', {}).get('default_generation_settings', {})
    require(defaults.get('n_ctx') == 4096 and defaults.get('params', {}).get('speculative.types') == 'none',
            'No-speculation context contract unverified')
    require(all(type(p['config'].get(k)) is type(v) and p['config'][k] == v for k, v in FROZEN.items()),
            'Frozen protocol mismatch')
    require(summary.get('cases') == 50 and summary.get('split') == 'calibration' and
            summary.get('real_actions_executed') == 0, 'Incomplete or wrong-split summary')
    ids, successes, minimum_decode_tokens = [], 0, 0
    with (run / 'pilot/transcripts.jsonl').open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'Uncommitted transcript tail')
            row = parse(line)
            ids.append(row['id'])
            require(row.get('split') == 'calibration' and row.get('protocol') == 'native' and
                    row.get('policy_version') == 'v2' and row.get('final_mode') == 'tool', 'Case protocol mismatch')
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
    require(ids == expected, 'Missing/duplicate/out-of-order transcript cases')
    rate = summary.get('full_completion_rate')
    require(type(rate) in (float, int) and math.isfinite(rate) and abs(rate - successes / 50) < 1e-12,
            'Summary disagrees with actual records')
    require(successes >= 40, 'Baseline below calibration40/50 readiness floor')
    files = ['manifest.json', 'shutdown.json', 'server.pid.json', 'pilot/manifest.json',
             'pilot/summary.json', 'pilot/transcripts.jsonl']
    return {'run': str(run), 'successes': successes, 'cases': 50, 'minimum_decode_tokens_per_layer': minimum_decode_tokens,
            'corpus_sha256': digest(corpus), 'inputs_sha256': {f: digest(run / f) for f in files}}


def plan(run, corpus, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    require(not out.exists(), 'Output directory must be NEW')
    proof = validate_run(run, corpus)
    trace = run / 'routing.jsonl'
    before = trace.stat()
    data = aggregate([trace], MODEL_SHA)
    require(all(n >= proof['minimum_decode_tokens_per_layer'] for n in data[1]),
            'Trace smaller than no-speculation generation lower bound')
    after = trace.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Trace changed during validation')
    masks, reports = {}, []
    for keep in POOLS:
        for seed in (None, RANDOM_SEED):
            kind = 'ranked' if seed is None else 'random'
            mask = build_from_aggregate(data, MODEL_SHA, split='calibration', keep=keep, random_seed=seed)
            name = f'{kind}-k{keep}.json'
            masks[name] = mask
            mass = [d['retained_mass_fraction'] for d in mask['diagnostics'].values()]
            reports.append({'file': name, 'keep': keep, 'method': mask['method'],
                            'retained_mass_min': min(mass), 'retained_mass_mean': math.fsum(mass) / len(mass),
                            'retained_mass_max': max(mass)})
    out.mkdir(parents=True, exist_ok=False)
    for name, value in masks.items():
        with (out / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    report = {'schema_version': 1, 'scope': 'calibration_only_unapproved_masks',
              'quality_approval': False, 'model_weights_written': False, 'provenance': proof,
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
    args = parser.parse_args()
    report = plan(args.run, args.corpus, args.out)
    print(json.dumps({'scope': report['scope'], 'candidates': report['candidates']}))
