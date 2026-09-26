"""Coordinator-only numerical gate. Refuses running servers; never stops the daily.

The outer coordinator owns an approved temporary daily stop/restore.
No actual Cockpit/browser/phone actions. This is NOT a throughput benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'gate'))
from compare_logits import compare, load_metadata
from mask_builder import aggregate

MODEL = Path(r'D:\models\qwen36moe\Qwen3.6-35B-A3B-Q4_K_M.gguf')
MODEL_SHA = '671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7'
LAB = Path(r'D:\ds4_work\qwen36_reap_lab')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def environment():
    env = os.environ.copy()
    for k in list(env):
        if k.startswith(('LLAMA_', 'GGML_', 'QWEN', 'SPEC_')):
            del env[k]
    settings = {'GGML_SCHED_ASYNC_INPUTS': '1', 'GGML_CUDA_GRAPH_COMPAT_CACHE': '1',
                'GGML_CUDA_MMVQ_OCCUPANCY': '1', 'LLAMA_MOE_DEMAND_GPU': '1',
                'LLAMA_MOE_CACHE_BATCH': '1', 'LLAMA_MOE_ELASTIC': '0',
                'QWEN36_REAP_MODEL_SHA256': MODEL_SHA}
    env.update(settings)
    return env, settings


def check_trace_tokens(prefix, trace_path):
    metadata, _ = load_metadata(prefix)
    calls = {r['helper_call_ordinal_1based']: r for r in metadata['decode_calls']}
    seen = {ordinal: set() for ordinal in calls}
    with Path(trace_path).open(encoding='utf-8') as f:
        next(f)
        for line in f:
            row = json.loads(line)
            call = calls.get(row.get('decode_call_id'))
            if call is None or row.get('phase') != 'unknown':
                raise ValueError('Trace call identity/unknown phase inconsistent with helper')
            pos = row['token_position']
            if not call['position_start'] <= pos < call['position_start'] + call['n_tokens']:
                raise ValueError('Trace position outside actual decode call')
            expected = metadata['prompt_token_ids'][pos] if call['phase'] == 'prefill' else call['token_id']
            if row['token_id'] != expected or row['seq_ids'] != [call['seq_id']]:
                raise ValueError('Trace token/sequence differs from actual input')
            if row['layer'] == 0:
                if pos in seen[row['decode_call_id']]:
                    raise ValueError('Duplicate token in decode call trace')
                seen[row['decode_call_id']].add(pos)
    phases = {'prefill': 0, 'decode': 0}
    for ordinal, call in calls.items():
        if len(seen[ordinal]) != call['n_tokens']:
            raise ValueError('Missing actual input tokens in trace')
        phases[call['phase']] += call['n_tokens']
    return phases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-inference', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=LAB / 'build/gate-baseline')
    p.add_argument('--candidate', type=Path, default=LAB / 'build/gate-candidate')
    p.add_argument('--exe', default='reap-native-gate.exe')
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.baseline = a.baseline.resolve()
    a.candidate = a.candidate.resolve()
    if not a.allow_inference:
        raise RuntimeError('Explicit --allow-inference required')
    for process in ('llama-server.exe', a.exe):
        text = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq ' + process, '/NH'],
                              capture_output=True, text=True, check=True).stdout
        if process.lower() in text.lower():
            raise RuntimeError('Existing ' + process + '; refusing overlap or termination')
    for directory in (a.baseline, a.candidate):
        if not (directory / a.exe).is_file():
            raise FileNotFoundError(directory / a.exe)
    a.out.mkdir(parents=True, exist_ok=False)
    # Rehash original checkpoint once before any model process, not one assertion per arm.
    actual_sha = sha(MODEL)
    if actual_sha != MODEL_SHA:
        raise RuntimeError('Original checkpoint hash changed')
    env, settings = environment()
    artifacts = {}
    for label, directory in [('baseline', a.baseline), ('candidate', a.candidate)]:
        artifacts[label] = {f.name: sha(f) for f in directory.iterdir() if f.suffix.lower() in ('.dll', '.exe')}
    if artifacts['baseline'][a.exe] != artifacts['candidate'][a.exe]:
        raise RuntimeError('Helper executable differs between arms')
    if artifacts['baseline']['llama-common.dll'] != artifacts['candidate']['llama-common.dll']:
        raise RuntimeError('Common DLL differs between arms')
    invariant = [{k: v for k, v in artifacts[arm].items() if k != 'llama.dll'} for arm in ('baseline', 'candidate')]
    if invariant[0] != invariant[1]:
        raise RuntimeError('Unexpected helper/runtime differences beyond llama.dll')
    save(a.out / 'manifest.json', {'model': str(MODEL), 'model_sha256': actual_sha,
                                 'environment': settings, 'runtime_sha256': artifacts,
                                 'harness_sha256': {str(f): sha(f) for f in (Path(__file__), HERE / 'mask_builder.py',
                                     HERE / 'gate/compare_logits.py', HERE / 'gate/fixtures/web_dom_utf8.txt',
                                     HERE / 'gate/fixtures/recovery_utf8.txt')},
                                 'scope': 'numerical_only_not_performance', 'atol': 1e-5, 'rtol': 1e-5})
    all_kept = {'schema_version': 1, 'model_sha256': MODEL_SHA, 'expert_count': 256,
                'top_k': 8, 'layers': {str(layer): list(range(256)) for layer in range(40)}}
    all_path = a.out / 'all-kept.json'
    save(all_path, all_kept)
    results = []

    def run(label, directory, fixture=None, trace=False, mask=None, one_token=False, should_fail=False):
        prefix = a.out / label
        outenv = env.copy()
        outenv['PATH'] = str(directory) + os.pathsep + env.get('PATH', '')
        if trace:
            outenv['QWEN36_REAP_TRACE'] = str(prefix) + '.routing.jsonl'
        if mask:
            outenv['QWEN36_REAP_MASK'] = str(mask)
        command = [str(directory / a.exe), '-m', str(MODEL), '--no-warmup', '--no-mmap', '-ngl', '99', '--cpu-moe',
                   '--moe-expert-cache', '32', '--moe-expert-cache-inserts', '8',
                   '-c', '2048', '-b', '128', '-ub', '128', '-t', '16', '-tb', '16',
                   '-ctk', 'q8_0', '-ctv', 'q8_0', '-n', '2' if one_token else '8',
                   '--gate-out', str(prefix), '--gate-run-id', label, '--gate-model-sha256', MODEL_SHA]
        command += ['--gate-token-id', '50'] if one_token else ['-f', str(fixture)]
        save(str(prefix) + '.command.json', command)
        child = None
        start = time.monotonic()
        try:
            with open(str(prefix) + '.stdout.log', 'wb') as stdout, open(str(prefix) + '.stderr.log', 'wb') as stderr:
                child = subprocess.Popen(command, cwd=directory, env=outenv, stdout=stdout, stderr=stderr)
                save(str(prefix) + '.pid.json', {'pid': child.pid, 'executable': str(directory / a.exe)})
                code = child.wait(timeout=a.timeout)
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=20)
            save(str(prefix) + '.shutdown.json', {'pid': child.pid if child else None,
                  'returncode': child.returncode if child else None,
                  'stopped': child is None or child.poll() is not None,
                  'elapsed_seconds_diagnostic_only': time.monotonic() - start})
        if should_fail:
            if code == 0:
                raise RuntimeError('Malformed mask unexpectedly succeeded')
            # Do not confuse an unrelated OOM/load failure with a validated rejection.
            log = Path(str(prefix) + '.stderr.log').read_text(encoding='utf-8', errors='replace')
            if 'REAP keep must be in [8,256]' not in log:
                raise RuntimeError('Failure does not establish a REAP mask rejection')
            results.append({'case': label, 'expected_rejection': True, 'returncode': code})
        else:
            if code != 0:
                raise RuntimeError(label + ' failed; inspect its stderr log')
            if trace:
                _, counts, provenance = aggregate([outenv['QWEN36_REAP_TRACE']], MODEL_SHA)
                phases = check_trace_tokens(prefix, outenv['QWEN36_REAP_TRACE'])
                results.append({'case': label, 'trace_tokens_per_layer': counts, 'provenance': provenance,
                                'actual_helper_input_phases': phases})
        print(json.dumps({'case': label, 'returncode': code, 'trace': trace}), flush=True)
        return prefix

    for fixture_name in ('web_dom_utf8', 'recovery_utf8'):
        fixture = HERE / 'gate/fixtures' / (fixture_name + '.txt')
        reference = run('original_' + fixture_name, a.baseline, fixture)
        for variant, trace, mask in [('off', False, None), ('trace', True, None), ('all-kept', False, all_path)]:
            candidate = run(variant + '_' + fixture_name, a.candidate, fixture, trace, mask)
            report = compare(reference, candidate)
            save(a.out / (variant + '_' + fixture_name + '.comparison.json'), report)
            results.append({'case': variant + '_' + fixture_name, 'numerical_pass': report['pass'],
                            'bit_identical': report['bit_identical'], 'maxabs': report['maxabs']})
            if not report['pass']:
                save(a.out / 'results.json', results)
                raise RuntimeError('Numerical gate failed; thresholds are not relaxed')
    reference = run('original_single_token', a.baseline, one_token=True)
    candidate = run('trace_single_token', a.candidate, trace=True, one_token=True)
    report = compare(reference, candidate)
    save(a.out / 'single-token-comparison.json', report)
    if not report['pass']:
        raise RuntimeError('Single-token gate failed')
    bad = dict(all_kept)
    bad['layers'] = dict(all_kept['layers'])
    bad['layers']['0'] = list(range(7))
    bad_path = a.out / 'invalid-keep7.json'
    save(bad_path, bad)
    run('reject_keep7', a.candidate, one_token=True, mask=bad_path, should_fail=True)
    # Exercise the NONIDENTITY mask graph only after all numerical identity gates.
    # Arbitrary even-ID pool: validates exclusion/plumbing, NOT calibrated quality.
    subset = dict(all_kept)
    subset['layers'] = {str(layer): list(range(0, 256, 2)) for layer in range(40)}
    subset_path = a.out / 'uncalibrated-even128.json'
    save(subset_path, subset)
    run('mask_even128_smoke', a.candidate, HERE / 'gate/fixtures/web_dom_utf8.txt', trace=True, mask=subset_path)
    with (a.out / 'mask_even128_smoke.routing.jsonl').open(encoding='utf-8') as f:
        next(f)
        for line in f:
            if any(expert % 2 for expert in json.loads(line)['ids']):
                raise RuntimeError('Excluded expert present in reduced-mask trace')
    results.append({'case': 'mask_even128_smoke', 'exclusion_pass': True,
                    'calibrated': False, 'quality_tested': False})
    results.append({'case': 'single_token', 'numerical_pass': report['pass'], 'bit_identical': report['bit_identical']})
    save(a.out / 'results.json', results)
    print('NUMERICAL GATES PASSED (not a performance or web-quality result)', flush=True)


if __name__ == '__main__':
    main()
