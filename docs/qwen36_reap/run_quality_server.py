"""Opt-in, isolated UNTRACED quality comparison coordinator.

Frozen V3/native/final-tool/greedy0,4096ctx,1024out; candidate018c for BOTH arms.
Only owns the process it starts on8116. Never stops/restores another daily server.
No model writes, tracing, automatic mask approval or annotation fabrication.
Heldout needs a separately frozen hash-pinned policy and selected-mask binding.
Policy byte SHA and canonical ENTIRE evaluation_protocol SHA are separate from
planning's internal protocol hash. Corpus sample digests use canonical fixtures.
Descriptors remain PENDING annotations/index rebasing, with absolute raw refs;
calibration is NEVER export-quality eligible, including completed screening runs.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from run_native_gates import HERE, LAB, MODEL, MODEL_SHA, environment, save, sha
from plan_masks import frozen_protocol, object_digest, validate_run as validate_source_run

_spec = importlib.util.spec_from_file_location('_quality_coordinator_export', HERE / 'export/compact_gguf.py')
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)
require = gate.require
sys.path.insert(0, str(HERE / 'pilot'))
from run_pilot import NoRedirect, select_episodes

BINARY = LAB / 'bin-candidate'
CORE_SHA = '018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef'
RUNTIME = {
    'ggml-base.dll': '9d0cb021fef05cb1d53c5d22a6f30322c6b3ab966dca9489ea7e6b4cfa13ddc4',
    'ggml-cpu.dll': '1af0511165f60adaa4366186e58abbcf24c25bf9ad71e41db42ac85eda33a5a7',
    'ggml-cuda.dll': 'a8e6d2a95c1fcaa14d28914cee1657e7e492a9b788afaaf5430d8019e667886e',
    'ggml.dll': '86e77878c5ee1bcee2b11877b2a7c4f8806e2debf3a24c7883ee45cb77a436db',
    'llama-common.dll': 'aee53a4efbd7f0036be8382546922f51f8e10fbdf448e4a3c44681d7dce3fb69',
    'llama-server-impl.dll': 'c06117ce6851312cc2d09dd7cfc89468d1664bfbaa055e2638ad8b452f18c3ae',
    'llama-server.exe': '75bf4eec3d3782ed865ca04e706f259584026f86fb3223b106dd68517e7c3bae',
    'llama.dll': CORE_SHA,
    'mtmd.dll': '603a1c44bc6db6a36e4d03548d0e1ac6f3018d627362c6c25f354ddf66280d7a',
}
FROZEN_CONFIG = dict(protocol='native', model='local-pilot', max_turns=13, max_output=1024,
    budget=4096, thinking='template-default', template_supports_thinking=False,
    policy_version='v3', diagnostic_raw=False, sampling_profile='greedy', seed=0, final_mode='tool')
PROTOCOL_CORE = dict(transport='native', policy_version='v3', final_mode='tool', sampling_profile='greedy',
    seeds=[0], context=4096, budget=4096, max_output=1024, max_turns=13, thinking='template-default',
    reasoning_preserve=False, skip_chat_parsing=False)
SOURCE_FILES = {'runner_sha256': HERE / 'pilot/run_pilot.py', 'validator_sha256': HERE / 'pilot/validator.py',
                'comparison_wrapper_sha256': Path(__file__), 'capture_coordinator_sha256': HERE / 'run_server_pilot.py'}


def typed_match(actual, expected):
    return isinstance(actual, dict) and all(type(actual.get(k)) is type(v) and actual[k] == v for k, v in expected.items())


def read_json(path):
    require(Path(path).resolve() != (HERE / 'pilot/heldout.jsonl').resolve(), 'Heldout corpus is never a metadata input')
    with Path(path).open('rb') as stream:
        raw = stream.read((64 << 20) + 1)
    require(len(raw) <= 64 << 20, 'JSON exceeds64MiB')
    value = gate.decode_json(raw)
    require(type(value) is dict, 'JSON object required')
    gate.canonical(value)  # Reject overflow-to-infinity as well as NaN/constants.
    return value, hashlib.sha256(raw).hexdigest()


def mask_identity(path):
    value, raw_sha = read_json(path)
    layers = gate.validate_mask(value, gate.PRODUCTION)
    return {'path': str(Path(path).resolve()), 'file_sha256': raw_sha,
            'canonical_sha256': gate.sha(gate.canonical(value)), 'keep': len(layers['0']),
            'method': value.get('method')}


def validate_options(a):
    require(a.allow_inference, '--allow-inference required before any process/network/model access')
    require(a.split in ('calibration', 'heldout'), 'Explicit supported split required')
    require(type(a.trial) is int and a.trial == 0, 'This frozen comparison supports trial0 only')
    require(a.split != 'heldout' or (a.allow_heldout and a.episode_ids is None),
            'Heldout requires --allow-heldout and forbids any subset')
    require(a.split == 'heldout' or not a.allow_heldout, '--allow-heldout only with heldout split')
    require(a.mask is None or a.selection_plan is not None, 'Masked runs require a ready selection plan')
    out = a.out.resolve()
    require(not out.exists() and out != HERE.parents[1] and HERE.parents[1] not in out.parents,
            'Output must be a NEW directory outside repository')
    if any((a.policy, a.policy_sha256, a.protocol_sha256, a.corpus_manifest, a.corpus_manifest_sha256)):
        require(a.split == 'heldout' and all((a.policy, a.policy_sha256, a.protocol_sha256, a.corpus_manifest,
                a.corpus_manifest_sha256)), 'Frozen-policy options must be complete and heldout-only')
    if a.split == 'heldout':
        require(all((a.policy, a.policy_sha256, a.protocol_sha256, a.corpus_manifest,
                     a.corpus_manifest_sha256, a.selected_mask, a.selection_plan)),
                'Heldout requires explicit policy/protocol/corpus hashes and selected-mask/plan binding')


def ready_plan(path, selected):
    plan, raw_sha = read_json(path)
    require(plan.get('gpu_screen_ready') is True and plan.get('quality_approval') is False and
            plan.get('policy_version') == 'v3', 'Require GPU-screen-ready unapproved V3 plan, never old V2')
    proof = plan.get('provenance', {})
    require(proof.get('expected_config') == frozen_protocol('v3') and proof.get('policy_version') == 'v3',
            'Selection source must use exact frozen V3 capture configuration')
    require(typed_match(proof.get('full_source_config'), frozen_protocol('v3')) and
            object_digest(proof['full_source_config']) == proof.get('full_source_config_sha256'), 'Source config hash mismatch')
    sources = proof.get('recorded_source_sha256', {})
    require(set(sources) == {'pilot_runner_sha256', 'coordinator_sha256'} and
            all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in sources.values()), 'Source identity missing')
    require(sources['pilot_runner_sha256'] == sha(SOURCE_FILES['runner_sha256']), 'Source calibration runner differs from current frozen runner')
    require(sources['coordinator_sha256'] == sha(SOURCE_FILES['capture_coordinator_sha256']), 'Capture coordinator source changed; never compare it to the quality wrapper')
    internal = object_digest({'expected_config': proof['expected_config'], 'recorded_source_sha256': sources})
    require(plan.get('protocol_sha256') == internal == proof.get('protocol_sha256'), 'INTERNAL planning protocol hash mismatch')
    families = proof.get('family_counts', {})
    require(len(families) == 10 and all(type(v.get('cases')) is int and v['cases'] == 5 and
            type(v.get('full_completions')) is int and 1 <= v['full_completions'] <= 5 for v in families.values()) and
            sum(v['full_completions'] for v in families.values()) >= 40 and
            proof.get('families_without_full_completion') == [] and proof.get('gpu_screen_ready') is True,
            'Source family coverage/readiness gate failed')
    require(plan.get('mask_sha256', {}).get(Path(selected['path']).name) == selected['file_sha256'],
            'Selection plan does not bind exact mask file bytes')
    mask, _ = read_json(selected['path'])
    require(mask.get('policy_version') == 'v3' and mask.get('protocol_sha256') == internal and
            mask.get('gpu_screen_ready') is True and mask.get('quality_approval') is False, 'Mask planning bindings mismatch')
    require(isinstance(proof.get('run'), str), 'Source calibration run missing')
    fresh = validate_source_run(Path(proof['run']), HERE / 'pilot/calibration.jsonl', policy_version='v3')
    require(gate.canonical(fresh) == gate.canonical(proof), 'Source run/provenance changed since planning')
    return {'path': str(Path(path).resolve()), 'file_sha256': raw_sha, 'value': plan, 'internal_protocol_sha256': internal}


def frozen_policy(a, selected, plan, sources, runtime, dataset_sha, ids):
    policy, raw_sha = read_json(a.policy)
    require(raw_sha == a.policy_sha256, 'Whole policy byte hash mismatch')
    require(policy.get('frozen_before_evaluation') is True, 'Policy not frozen before evaluation')
    protocol = policy.get('evaluation_protocol')
    protocol_sha = gate.sha(gate.canonical(protocol))
    require(protocol_sha == a.protocol_sha256, 'Canonical evaluation_protocol hash mismatch')
    gate.evaluation_protocol_gate(policy, {'evaluation_protocol_sha256': protocol_sha}, {'evaluation_protocol_sha256': protocol_sha})
    require(typed_match(protocol, PROTOCOL_CORE), 'Frozen policy differs from fixed untraced V3 run configuration')
    require(protocol.get('trials') == [0] and type(protocol['trials'][0]) is int, 'Require exact frozen trial0')
    require(protocol.get('model_sha256') == MODEL_SHA, 'Policy model binding mismatch')
    require(protocol.get('runtime_sha256') == runtime, 'Policy runtime identity mismatch')
    require(typed_match(protocol, sources), 'Policy runner/validator source identity mismatch')
    require(protocol.get('selected_mask_sha256') == selected['canonical_sha256'] and
            protocol.get('selected_mask_file_sha256') == selected['file_sha256'] and
            protocol.get('selection_plan_sha256') == plan['file_sha256'], 'Policy selected mask/plan binding mismatch')
    require(isinstance(protocol.get('chat_template_sha256'), str) and len(protocol['chat_template_sha256']) == 64,
            'Frozen actual chat-template hash required')
    corpus, corpus_sha = read_json(a.corpus_manifest)
    require(set(corpus) <= {'schema_version', 'split', 'calibration_disjoint', 'dataset_sha256', 'samples'}, 'Unexpected corpus manifest fields')
    require(len(corpus.get('samples', [])) == 20 and all(set(x) == {'id', 'sha256'} and isinstance(x['sha256'], str) and
            len(x['sha256']) == 64 for x in corpus['samples']), 'Exactly20 frozen case identity hashes required')
    require(corpus_sha == a.corpus_manifest_sha256 and corpus.get('split') == 'heldout' and corpus.get('calibration_disjoint') is True,
            'Frozen heldout identity manifest mismatch')
    require(corpus.get('dataset_sha256') == dataset_sha and [x.get('id') for x in corpus.get('samples', [])] == ids,
            'Heldout dataset hash/complete ordered case IDs mismatch')
    return {'path': str(Path(a.policy).resolve()), 'file_sha256': raw_sha, 'protocol_sha256': protocol_sha,
            'protocol': protocol, 'corpus_manifest_sha256': corpus_sha}


def runtime_files():
    return [p for p in BINARY.iterdir() if p.is_file() and p.suffix.lower() in ('.dll', '.exe')]


def runtime_identity():
    actual = {p.name: sha(p) for p in runtime_files()}
    require(actual == RUNTIME, 'Candidate018c application runtime file set/hash mismatch')
    return actual


def prepare(a):
    """File-only preflight. Tests patch selector/model/runtime hashes; never executes."""
    validate_options(a)
    runtime = runtime_identity()
    sources = {key: sha(path) for key, path in SOURCE_FILES.items()}
    actual = mask_identity(a.mask) if a.mask else None
    selected = mask_identity(a.selected_mask) if a.selected_mask else actual
    require(actual is None or selected == actual, 'Actual masked arm must use exactly the frozen selected mask')
    plan = ready_plan(a.selection_plan, selected) if a.selection_plan and selected else None
    if a.selection_plan:
        require(plan is not None, 'Selection plan requires an explicit selected mask')
    policy = None
    if a.policy:
        require(all((a.policy_sha256, a.protocol_sha256, a.corpus_manifest, a.corpus_manifest_sha256, selected, plan)), 'Incomplete frozen policy bindings')
        require(a.split == 'heldout', 'Export evaluation policy used only for full heldout')
        require(selected.get('method') == 'mass_gate', 'Random control is calibration-only, not an approval candidate')
        identities, _ = read_json(a.corpus_manifest)
        policy = frozen_policy(a, selected, plan, sources, runtime, identities.get('dataset_sha256'),
                               [x.get('id') for x in identities.get('samples', [])])
    # All heldout permits and byte/protocol/mask/source bindings are verified BEFORE
    # either loading fixtures or hashing the actual heldout dataset. Live template
    # verification happens after private boot but before the first model request.
    require(a.split != 'heldout' or policy is not None, 'Heldout frozen binding missing')
    require(sha(MODEL) == MODEL_SHA, 'Immutable original model hash mismatch')
    episodes = select_episodes(a.split, None, a.episode_ids)
    ids = [e['id'] for e in episodes]
    expected_count = 20 if a.split == 'heldout' else 50
    require(ids and len(ids) == len(set(ids)) and all(x.startswith(a.split + '-') for x in ids), 'Invalid selected IDs')
    require(a.episode_ids is not None or len(ids) == expected_count, 'Default split must be complete20/50')
    dataset_sha = sha(HERE / 'pilot' / (a.split + '.jsonl'))
    if policy:
        require(dataset_sha == identities['dataset_sha256'] and ids == [x['id'] for x in identities['samples']],
                'Actual heldout dataset differs from frozen complete identity manifest')
        require(all(gate.sha(gate.canonical(e)) == record['sha256'] for e, record in zip(episodes, identities['samples'])),
                'Canonical fixture digest differs from frozen sample identity')
    env, selected_env = environment()
    require(not any(k.startswith('QWEN36_REAP_') and k.endswith(('TRACE', 'MASK')) for k in env), 'Inherited trace/mask escaped environment scrub')
    if actual:
        env['QWEN36_REAP_MASK'] = actual['path']
        selected_env['QWEN36_REAP_MASK'] = actual['path']
    env['PATH'] = str(BINARY) + ';' + env.get('PATH', '')
    return dict(args=a, ids=ids, dataset_sha256=dataset_sha, runtime_sha256=runtime, sources=sources,
                mask=actual, selected_mask=selected, plan=plan, policy=policy, env=env, recorded_env=selected_env)


def command():
    return [str(BINARY / 'llama-server.exe'), '-m', str(MODEL), '--alias', 'local-pilot',
            '--host', '127.0.0.1', '--port', '8116', '--offline', '--no-warmup', '--no-mmap',
            '-ngl', '99', '--cpu-moe', '--moe-expert-cache', '32', '--moe-expert-cache-inserts', '8',
            '-c', '4096', '-np', '1', '-b', '128', '-ub', '128', '-t', '16', '-tb', '16',
            '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '-fit', 'off', '--jinja',
            '--reasoning', 'on', '--no-reasoning-preserve', '--no-context-shift', '--cache-ram', '0']


def ensure_exclusive():
    result = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True, check=True)
    names = [r[0].strip().lower() for r in csv.reader(io.StringIO(result.stdout)) if r]
    require(not any(name.startswith('llama') for name in names), 'Existing llama process; parent must handle approved daily suspension')
    with socket.socket() as sock:
        sock.settimeout(0.5)
        require(sock.connect_ex(('127.0.0.1', 8116)) != 0, 'Port8116 occupied; never adopt another server')


def wait_server(server):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + 180
    while True:
        require(server.poll() is None, 'Owned server exited during startup')
        try:
            with opener.open('http://127.0.0.1:8116/health', timeout=2) as response:
                if json.load(response).get('status') == 'ok':
                    with opener.open('http://127.0.0.1:8116/props', timeout=5) as props:
                        return json.load(props)
        except (OSError, ValueError):
            pass
        require(time.monotonic() <= deadline, 'Private server startup timeout')
        time.sleep(0.5)


def verify_server(props, spec):
    settings = props.get('default_generation_settings', {})
    require(type(settings.get('n_ctx')) is int and settings['n_ctx'] == 4096 and
            settings.get('params', {}).get('speculative.types') == 'none', 'Unexpected server context/speculation')
    require(isinstance(props.get('model_path'), str) and Path(props['model_path']).resolve() == MODEL.resolve(), 'Server model path is not the immutable original')
    template = props.get('chat_template')
    require(isinstance(template, str) and template, 'Actual chat template missing')
    digest = hashlib.sha256(template.encode('utf-8')).hexdigest()
    if spec['policy']:
        require(digest == spec['policy']['protocol']['chat_template_sha256'], 'Actual template differs from frozen heldout protocol')
    return digest


def verify_completion(out, spec):
    """Complete raw outcomes, not all-success or quality approval. No rerating."""
    manifest, _ = read_json(out / 'pilot/manifest.json')
    summary, _ = read_json(out / 'pilot/summary.json')
    require(manifest.get('split') == spec['args'].split and manifest.get('episode_ids') == spec['ids'] and
            manifest.get('dataset_sha256') == spec['dataset_sha256'] and manifest.get('limit') is None,
            'Pilot manifest split/IDs/dataset/subset mismatch')
    require(typed_match(manifest.get('config'), FROZEN_CONFIG), 'Actual pilot configuration drift')
    require(manifest.get('runner_sha256') == spec['sources']['runner_sha256'], 'Pilot source hash drift')
    observed = verify_server(manifest.get('server_props', {}), spec)
    require('observed_template_sha256' not in spec or observed == spec['observed_template_sha256'], 'Template changed between coordinator and runner')
    ids, success = [], 0
    with (out / 'pilot/transcripts.jsonl').open('rb') as source:
        for line in source:
            require(line.endswith(b'\n'), 'Partial transcript tail')
            row = gate.decode_json(line)
            require(row.get('split') == spec['args'].split and row.get('protocol') == 'native' and
                    row.get('policy_version') == 'v3' and row.get('final_mode') == 'tool', 'Record protocol mismatch')
            require(type(row.get('full_completion')) is bool and row.get('real_actions_executed') == 0, 'Malformed outcome')
            require(row.get('error_class') not in ('preflight', 'transport'), 'Infrastructure-error record cannot qualify')
            require(isinstance(row.get('turns'), list) and 1 <= len(row['turns']) <= 13, 'Missing actual turns')
            for turn in row['turns']:
                req, pre, response = turn.get('request', {}), turn.get('preflight', {}), turn.get('response', {})
                require(typed_match(req, dict(model='local-pilot', stream=False, cache_prompt=True,
                    temperature=0, top_k=1, top_p=1, min_p=0, seed=0, max_tokens=1024,
                    tool_choice='auto', parallel_tool_calls=False)), 'Actual sampling/transport drift')
                require(not any(k in req for k in ('chat_template_kwargs', 'presence_penalty', 'repeat_penalty', 'grammar')),
                        'Unexpected thinking/sampling/grammar override')
                require('verbose' not in req and 'return_tokens' not in req, 'Diagnostic raw request not eligible')
                n = pre.get('prompt_tokens_preflight')
                require(type(n) is int and n > 0 and n + 1024 <= 4096 and pre.get('server_ctx') == 4096 and
                        pre.get('effective_budget') == 4096 and pre.get('output_reserved') == 1024 and
                        response.get('usage', {}).get('prompt_tokens') == n, 'Actual prompt/context alignment invalid')
                count = response.get('usage', {}).get('completion_tokens')
                require(type(count) is int and 0 <= count <= 1024, 'Actual completion exceeds frozen cap')
                require(type(response.get('choices')) is list and len(response['choices']) == 1, 'Missing actual response')
            ids.append(row['id'])
            if row['full_completion']:
                require(row.get('error_class') is None and row['turns'][-1].get('selection', {}).get('kind') == 'final' and
                        row['turns'][-1]['selection']['value'] == row.get('final'), 'Inconsistent successful record')
                success += 1
    require(ids == spec['ids'], 'Missing/duplicate/out-of-order outcome IDs')
    require(summary.get('cases') == len(ids) and summary.get('split') == spec['args'].split and
            summary.get('real_actions_executed') == 0, 'Incomplete/wrong summary')
    rate = summary.get('full_completion_rate')
    require(type(rate) in (int, float) and math.isfinite(rate) and abs(rate - success / len(ids)) < 1e-12, 'Summary rate mismatch')
    return {'cases': len(ids), 'successes': success, 'full_completion_rate': rate}


def execute(spec):
    a, server, completed, error = spec['args'], None, False, None
    validate_options(a)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'scope': 'untraced_quality_comparison', 'completed': False,
        'eligible_quality_evaluation': False, 'split': a.split, 'episode_ids': spec['ids'],
        'runtime_arm': 'candidate', 'trace_enabled': False, 'skip_chat_parsing': False, 'reasoning_preserve': False,
        'model_sha256': MODEL_SHA, 'mask_sha256': spec['mask']['canonical_sha256'] if spec['mask'] else None,
        'mask_file_sha256': spec['mask']['file_sha256'] if spec['mask'] else None,
        'selected_mask': spec['selected_mask'], 'selection_plan_sha256': spec['plan']['file_sha256'] if spec['plan'] else None,
        'policy_file_sha256': spec['policy']['file_sha256'] if spec['policy'] else None,
        'evaluation_protocol_sha256': spec['policy']['protocol_sha256'] if spec['policy'] else None,
        'corpus_manifest_sha256': spec['policy']['corpus_manifest_sha256'] if spec['policy'] else None,
        'dataset_sha256': spec['dataset_sha256'], 'config': FROZEN_CONFIG, 'sources': spec['sources'],
        'runtime_sha256': spec['runtime_sha256'], 'environment': spec['recorded_env'],
        'runtime_hash_scope': 'Pinned candidate application EXE/DLL files only; excludes GPU driver and system DLLs',
        'not_a_throughput_benchmark': True,
        'parser_config_sha256': gate.sha(gate.canonical({'jinja': True, 'skip_chat_parsing': False})),
        'reasoning_config_sha256': gate.sha(gate.canonical({'reasoning': 'on', 'reasoning_preserve': False, 'thinking': 'template-default'})),
        'command': command(), 'coordinator_sha256': sha(__file__)}
    save(out / 'start-manifest.json', manifest)
    try:
        with (out / 'server.stdout.log').open('xb') as stdout, (out / 'server.stderr.log').open('xb') as stderr:
            server = subprocess.Popen(command(), cwd=BINARY, env=spec['env'], stdout=stdout, stderr=stderr)
            save(out / 'server.pid.json', {'pid': server.pid, 'executable': str(BINARY / 'llama-server.exe')})
            manifest['chat_template_sha256'] = verify_server(wait_server(server), spec)
            spec['observed_template_sha256'] = manifest['chat_template_sha256']
            cmd = [sys.executable, '-B', str(HERE / 'pilot/run_pilot.py'), '--allow-inference',
                '--split', a.split, '--output-dir', str(out / 'pilot'), '--url', 'http://127.0.0.1:8116',
                '--protocol', 'native', '--policy-version', 'v3', '--final-mode', 'tool', '--sampling-profile',
                'greedy', '--seed', '0', '--budget', '4096', '--max-output', '1024', '--max-turns', '13']
            if a.episode_ids is not None:
                cmd.extend(['--episode-ids', a.episode_ids])
            save(out / 'pilot.command.json', cmd)
            subprocess.run(cmd, check=True, timeout=3600, env=spec['env'], cwd=HERE)
            require(server.poll() is None, 'Owned server exited unexpectedly before completion checks')
            manifest['outcomes'] = verify_completion(out, spec)
            require(sha(MODEL) == MODEL_SHA, 'Original model changed during run')
            require(runtime_identity() == spec['runtime_sha256'], 'Runtime changed during run')
            require(all(sha(path) == spec['sources'][key] for key, path in SOURCE_FILES.items()), 'Frozen source changed during run')
            require(sha(HERE / 'pilot' / (a.split + '.jsonl')) == spec['dataset_sha256'], 'Corpus changed during run')
            if spec['selected_mask']:
                require(mask_identity(spec['selected_mask']['path']) == spec['selected_mask'], 'Selected mask changed during run')
            if spec['plan']:
                require(sha(a.selection_plan) == spec['plan']['file_sha256'], 'Selection plan changed during run')
            if spec['policy']:
                require(sha(a.policy) == spec['policy']['file_sha256'] and sha(a.corpus_manifest) == spec['policy']['corpus_manifest_sha256'],
                        'Frozen policy/corpus identity manifest changed during run')
            completed = True
    except BaseException as exc:
        error = exc
        manifest['error_class'] = type(exc).__name__
        manifest['private_error'] = str(exc)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)
        stopped = server is None or server.poll() is not None
        save(out / 'shutdown.json', {'pid': server.pid if server else None, 'stopped': stopped,
                                    'returncode': server.returncode if server else None})
        manifest['completed'] = completed and stopped
        manifest['eligible_quality_evaluation'] = completed and stopped and a.split == 'heldout'
        save(out / 'manifest.json', manifest)
    if error:
        raise error
    require(manifest['completed'], 'Run not completed/stopped')
    descriptor = {k: manifest[k] for k in ('completed', 'eligible_quality_evaluation', 'skip_chat_parsing',
        'reasoning_preserve', 'model_sha256', 'mask_sha256', 'evaluation_protocol_sha256', 'corpus_manifest_sha256')}
    role = 'baseline' if spec['mask'] is None else ('random' if spec['mask'].get('method') == 'random_matched_pool' else 'masked')
    descriptor.update(role=role, seed=0, trial=a.trial, split=a.split,
        descriptor_status='PENDING_ANNOTATIONS_AND_INDEX_REBASE', index_ready=False,
        policy_file_sha256=manifest['policy_file_sha256'], selection_plan_sha256=manifest['selection_plan_sha256'],
        selected_mask=manifest['selected_mask'], annotations_required=True,
        transcript={'path': str(out / 'pilot/transcripts.jsonl'), 'sha256': sha(out / 'pilot/transcripts.jsonl')},
        manifest={'path': str(out / 'pilot/manifest.json'), 'sha256': sha(out / 'pilot/manifest.json')},
        coordinator_manifest={'path': str(out / 'manifest.json'), 'sha256': sha(out / 'manifest.json')},
        shutdown={'path': str(out / 'shutdown.json'), 'sha256': sha(out / 'shutdown.json')})
    save(out / 'run-descriptor.json', descriptor)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-inference', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', choices=('calibration', 'heldout'), default='calibration')
    p.add_argument('--episode-ids')
    p.add_argument('--mask', type=Path)
    p.add_argument('--selection-plan', type=Path)
    p.add_argument('--allow-heldout', action='store_true')
    p.add_argument('--selected-mask', type=Path, help='Frozen candidate identity, including for unmasked heldout baseline')
    p.add_argument('--policy', type=Path)
    p.add_argument('--policy-sha256')
    p.add_argument('--protocol-sha256')
    p.add_argument('--corpus-manifest', type=Path)
    p.add_argument('--corpus-manifest-sha256')
    p.add_argument('--trial', type=int, default=0)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    validate_options(a)
    ensure_exclusive()
    spec = prepare(a)
    ensure_exclusive()  # File/hash preflight can take time: refuse a newly started daily.
    return execute(spec)


if __name__ == '__main__':
    main()
