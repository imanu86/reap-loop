"""Opt-in private native cache-capacity NUMERICAL gate, never a benchmark/quality approval.

Nine NEW helper processes: fixtures web/recovery/single50 x capacities32/64/96.
Each64/96 compares to its fresh32 reference. Existing full-F32 comparator code is
reused with an isolated in-memory loader projection of ONLY moe_cache_slots;
no module-global monkeypatch, metadata file rewrite, hidden tolerance or retry.
Source-known startup markers MUST establish40 active GPU layers and actual slots.
Exact tensor payload bytes are DERIVED from pinned geometry, NOT exact telemetry.
Parent owns any approved daily suspension/restoration; this script never does it.
"""
import argparse
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import types
import uuid

from run_native_gates import HERE, LAB, MODEL, MODEL_SHA, environment, save, sha
import compare_logits as logits

SCOPE = 'NUMERICAL_ONLY_NOT_BENCHMARK_NOT_QUALITY_APPROVAL'
BINARY = LAB / 'build/gate-candidate'
EXE = 'reap-native-gate.exe'
CAPACITIES = (32, 64, 96)
CASES = ('web_dom_utf8', 'recovery_utf8', 'single_token')
VOCAB = 248320
PAYLOAD_BYTES = {32: 2264924160, 64: 4529848320, 96: 6794772480}
RUNTIME_PINS = {
    'reap-native-gate.exe': '97c826a2efd65628459ba8fbd65c8523e9ad2d1477ce38d4d728f0917a28d4b4',
    'llama.dll': '018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef',
    'llama-common.dll': '55a4924a199d245d9c6f99ec17899a8ddfacd740b26f637af41ae0e7d4fddaf3',
    'ggml-base.dll': '9d0cb021fef05cb1d53c5d22a6f30322c6b3ab966dca9489ea7e6b4cfa13ddc4',
    'ggml-cpu.dll': '1af0511165f60adaa4366186e58abbcf24c25bf9ad71e41db42ac85eda33a5a7',
    'ggml-cuda.dll': 'a8e6d2a95c1fcaa14d28914cee1657e7e492a9b788afaaf5430d8019e667886e',
    'ggml.dll': '86e77878c5ee1bcee2b11877b2a7c4f8806e2debf3a24c7883ee45cb77a436db',
    'llama-server-impl.dll': 'c06117ce6851312cc2d09dd7cfc89468d1664bfbaa055e2638ad8b452f18c3ae',
    'llama-server.exe': '75bf4eec3d3782ed865ca04e706f259584026f86fb3223b106dd68517e7c3bae',
    'mtmd.dll': '603a1c44bc6db6a36e4d03548d0e1ac6f3018d627362c6c25f354ddf66280d7a',
}
SOURCE_PINS = {
    HERE / 'run_native_gates.py': 'a3e6fffc5fab3e6a4e0707be16312ddcd085cfebec3a6df71b2ee049e31b1c0e',
    HERE / 'gate/compare_logits.py': 'fb94d1bf31ae407ecddbf504737f018909b7b522a66b6d5aaf7562749ec8b4d4',
    HERE / 'gate/native_gate.cpp': '70c54033444565866874fcdc9316b5c8b1bad53cddbcbb27e0bf130bad486009',
    HERE / 'gate/fixtures/web_dom_utf8.txt': '493e4d3f44347d799f8201e4244550ac770767c187f5810a37541dddc9a1e427',
    HERE / 'gate/fixtures/recovery_utf8.txt': '50c22fdd2ecab51a63570b4152848c278997596e9631ee73f82e7f7be38a121a',
    HERE / 'export/original_header_profile.json': '916b073b01ace546ad776c37332c89d07989e6390ecd271ce9c3742906f9905d',
    LAB / 'source/src/llama-moecache.cpp': '0a43eaf269f78aee556104d1e1f8ffc4b993d6536dcaa2394455aac9c3958689',
    LAB / 'source/common/arg.cpp': 'd800f5ff8c7cf33b8c04687acdd5ebae816ff44633160ecb11635034c2b7419e',
    LAB / 'source/common/log.cpp': 'c617b2d0a12fad86a96ff91a7c6557b759dbc4c05f01d3144a1ca4289ba24deb',
}
HELPER_ENV = {'QWEN36_REAP_MASK': None, 'QWEN36_REAP_TRACE': None, 'QWEN36_REAP_MODEL_SHA256': MODEL_SHA,
    'LLAMA_MOE_CACHE_BATCH': '1', 'LLAMA_MOE_ELASTIC': '0', 'LLAMA_MOE_DEMAND_GPU': '1',
    'LLAMA_MOE_PHASE_CACHE': None, 'LLAMA_MOE_MASSA': None, 'LLAMA_MOE_MASSA_EMIVITA': None,
    'LLAMA_MOE_CACHE_CONGELA': None}
SETTINGS = dict(n_ctx_requested=2048, n_ctx_actual=2048, n_batch=128, n_ubatch=128,
    n_gpu_layers=99, moe_cache_inserts=8, cache_type_k=8, cache_type_v=8,
    threads=16, threads_batch=16, flash_attn=-1)
COMPILERS = {'cl.exe', 'link.exe', 'nvcc.exe', 'cicc.exe', 'ptxas.exe', 'nvlink.exe', 'ninja.exe', 'cmake.exe',
             'msbuild.exe', 'clang.exe', 'clang++.exe', 'clang-cl.exe', 'lld-link.exe', 'gcc.exe', 'g++.exe',
             'cc1.exe', 'cc1plus.exe', 'rc.exe', 'csc.exe', 'vbc.exe', 'cudafe++.exe', 'fatbinary.exe',
             'nmake.exe', 'make.exe', 'gmake.exe', 'lld.exe', 'ld.exe', 'llvm-rc.exe', 'dxc.exe', 'fxc.exe'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def artifact(prefix, suffix):
    return Path(str(prefix) + suffix)


def fixture_path(case):
    require(case in CASES and case != 'single_token', 'Unknown UTF8 fixture')
    return HERE / 'gate/fixtures' / (case + '.txt')


def command(capacity, case, prefix, run_id):
    require(type(capacity) is int and capacity in CAPACITIES and case in CASES, 'Frozen capacity/case required')
    cmd = [str(BINARY / EXE), '-m', str(MODEL), '--no-warmup', '--no-mmap', '-ngl', '99', '--cpu-moe',
        '--moe-expert-cache', str(capacity), '--moe-expert-cache-inserts', '8', '-c', '2048', '-b', '128', '-ub', '128',
        '-t', '16', '-tb', '16', '-ctk', 'q8_0', '-ctv', 'q8_0', '-n', '2' if case == 'single_token' else '8',
        '--gate-out', str(prefix), '--gate-run-id', run_id, '--gate-model-sha256', MODEL_SHA]
    return cmd + (['--gate-token-id', '50'] if case == 'single_token' else ['-f', str(fixture_path(case))])


def ensure_exclusive():
    result = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True, check=True)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    require(rows and all(len(r) >= 2 and r[1].isdigit() for r in rows), 'Cannot verify process inventory')
    names = {r[0].lower() for r in rows}
    require(not any(n.startswith('llama') or n == EXE or n in COMPILERS for n in names),
            'Existing llama/helper/compiler: parent owns suspension; never overlap or terminate it')


def runtime_files():
    return [p for p in BINARY.iterdir() if p.is_file() and p.suffix.lower() in ('.dll', '.exe')]


def pin_inputs():
    runtime = {p.name: sha(p) for p in runtime_files()}
    require(runtime == RUNTIME_PINS, 'Native candidate helper/DLL set or hash mismatch')
    sources = {str(p): sha(p) for p in SOURCE_PINS}
    require(sources == {str(p): digest for p, digest in SOURCE_PINS.items()}, 'Frozen source/fixture hash mismatch')
    sources[str(Path(__file__).resolve())] = sha(__file__)
    ensure_exclusive()  # Recheck after file preflight, also on postflight, BEFORE model bytes.
    model_sha = sha(MODEL)
    require(model_sha == MODEL_SHA, 'Immutable original model hash mismatch')
    return {'model_sha256': model_sha, 'runtime_sha256': runtime, 'source_fixture_sha256': sources}


def controlled_environment():
    env, selected = environment()
    # Supported by frozen common/arg.cpp3991. GGML INFO is logger TRACE level4;
    # default3 suppresses the actual-capacity markers in historical gate05 logs.
    env['LLAMA_ARG_LOG_VERBOSITY'] = selected['LLAMA_ARG_LOG_VERBOSITY'] = '4'
    require('QWEN36_REAP_TRACE' not in env and 'QWEN36_REAP_MASK' not in env, 'Trace/mask must be absent')
    env['PATH'] = str(BINARY) + os.pathsep + env.get('PATH', '')
    return env, selected


def activation_proof(text, capacity):
    require(type(capacity) is int and capacity in CAPACITIES, 'Unknown capacity')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    active = re.findall(r'moe-cache: current-step GPU path active for (\d+) layers; prefill graph unchanged', text)
    enabled = re.findall(r'MoE expert cache enabled: (\d+) layers x (\d+) slots, (\d+) inserts/step, (\d+\.\d) MiB device memory', text)
    expected_mib = f'{PAYLOAD_BYTES[capacity] / (1024 * 1024):.1f}'
    require(active == ['40'] and enabled == [('40', str(capacity), '8', expected_mib)],
            'activation_unverified: missing/duplicate/wrong actual layer/slot/payload startup markers')
    require(not re.search(r'cache disabled|expert layers found - disabled|moe-elastic[:\-]|failed to allocate MoE cache', text, re.I),
            'activation_unverified: disabled/elastic/fallback cache message')
    return {'verified': True, 'evidence': 'known source startup stderr markers, not requested argv alone',
        'active_gpu_layers': 40, 'actual_slots_per_layer': capacity, 'actual_inserts_per_step': 8,
        'logged_payload_mib_rounded_1_decimal': expected_mib, 'derived_payload_bytes': PAYLOAD_BYTES[capacity],
        'payload_derivation': '40 * slots * 3 * (2048 * 512 / 256) * 144; pinned Q4_K routed geometry',
        'exact_byte_telemetry': False, 'not_total_vram': True,
        'exclusions': 'mapping tables, staging, alignment/buffer overhead, nonrouted weights, KV/workspace, driver/display'}


def validate_output(prefix, capacity, case, run_id, cmd, started_ns):
    m, data = logits.load_metadata(prefix)
    require(m.get('run_id') == run_id and m.get('argv') == cmd, 'Stale/mismatched run identity or argv')
    require(m.get('model_path') == str(MODEL) and m.get('mode') == 'greedy', 'Model path/generation recipe drift')
    expected = dict(SETTINGS, moe_cache_slots=capacity)
    require(m.get('settings') == expected and all(type(m['settings'][k]) is type(v) for k, v in expected.items()),
            'Actual helper settings differ from frozen recipe/capacity')
    require(m.get('environment') == HELPER_ENV, 'Helper environment differs or TRACE/MASK active')
    steps = 2 if case == 'single_token' else 8
    require(m['shape'] == [steps, VOCAB], 'Wrong full vocabulary/generated count')
    if case == 'single_token':
        require(m['prompt_mode'] == 'single_token' and m['prompt_token_ids'] == [50] and m['prompt_utf8'] == '', 'Not single-token50')
    else:
        require(m['prompt_mode'] == 'raw_utf8' and m['prompt_utf8'].encode('utf-8') == fixture_path(case).read_bytes(), 'Raw fixture differs')
    calls, position, prefill_calls = m.get('decode_calls'), 0, []
    require(type(calls) is list and calls, 'Actual helper decode calls missing')
    prompt_count = len(m['prompt_token_ids'])
    for i, row in enumerate(calls):
        require(row.get('helper_call_ordinal_1based') == i + 1 and row.get('seq_id') == 0 and row.get('position_start') == position,
                'Decode call identity/sequence/position mismatch')
        n = row.get('n_tokens')
        require(type(n) is int and n > 0, 'Invalid actual input token count')
        if position < prompt_count:
            require(row.get('phase') == 'prefill' and n == min(128, prompt_count - position), 'Prefill chunk mismatch')
            prefill_calls.append(i + 1)
        else:
            index = position - prompt_count
            require(row.get('phase') == 'decode' and n == 1 and index < steps - 1 and row.get('token_id') == m['token_ids'][index],
                    'Incremental input token mismatch')
        position += n
    require(position == prompt_count + steps - 1, 'Missing/extra actual decode calls')
    for i, row in enumerate(m['steps']):
        require(row.get('helper_call_ordinal_1based') == prefill_calls[-1] + i, 'Logits row not bound to actual helper call')
    for path in (artifact(prefix, '.json'), data):
        require(started_ns <= path.stat().st_mtime_ns <= time.time_ns(), 'Stale/future artifact timestamp')
    log_path = artifact(prefix, '.stderr.log')
    require(log_path.stat().st_size <= 64 << 20, 'Startup log exceeds bounded review size')
    # Preserve/hash raw log bytes; unrelated non-UTF8 diagnostics cannot create
    # any of the strictly matched ASCII activation markers.
    proof = activation_proof(log_path.read_text(encoding='utf-8', errors='replace'), capacity)
    return {'case': case, 'capacity': capacity, 'prefix': str(prefix), 'run_id': run_id, 'activation': proof,
            'actual_settings': m['settings'], 'shape': m['shape'], 'actual_prompt_tokens': prompt_count,
            'artifact_sha256': {suffix: sha(artifact(prefix, suffix)) for suffix in
                ('.json', '.logits.f32', '.stderr.log', '.stdout.log', '.command.json', '.pid.json', '.shutdown.json')}}


def compare_capacity(reference, candidate):
    a, pa = logits.load_metadata(reference)
    b, pb = logits.load_metadata(candidate)
    require(type(a['settings'].get('moe_cache_slots')) is int and a['settings']['moe_cache_slots'] == 32 and
            type(b['settings'].get('moe_cache_slots')) is int and b['settings']['moe_cache_slots'] in (64, 96), 'Only fresh32 versus64/96 allowed')
    aa, bb = deepcopy(a), deepcopy(b)
    bb['settings']['moe_cache_slots'] = 32
    require(aa['settings'] == bb['settings'], 'Non-capacity setting drift cannot be projected away')
    views = {str(reference): (aa, pa), str(candidate): (bb, pb)}
    require(len(views) == 2, 'Reference/candidate must be distinct fresh artifacts')
    before = {str(p): sha(p) for p in (artifact(reference, '.json'), pa, artifact(candidate, '.json'), pb)}
    # Reuse EXACT existing finite/F32/argmax/token/tolerance implementation with
    # PRIVATE globals. The imported module and original metadata remain unchanged.
    namespace = dict(logits.compare.__globals__, load_metadata=lambda prefix: views[str(prefix)])
    isolated = types.FunctionType(logits.compare.__code__, namespace, 'cache_capacity_projection', logits.compare.__defaults__)
    report = isolated(reference, candidate)
    require(before == {path: sha(Path(path)) for path in before}, 'Raw artifacts changed during comparison')
    report.update(scope=SCOPE, allowed_difference='moe_cache_slots', metadata_projection=True,
        actual_settings={'reference': a['settings'], 'candidate': b['settings']},
        raw_artifact_sha256=before, metadata_bit_identity_claimed=False,
        comparator_sha256=sha(HERE / 'gate/compare_logits.py'))
    return report


def run_probe(out, capacity, case, nonce, env, timeout):
    ensure_exclusive()
    label = f'cap{capacity}_{case}'
    prefix, child = out / label, None
    run_id = nonce + '_' + label
    require(not list(out.glob(label + '.*')), 'Existing/stale case prefix forbidden')
    cmd = command(capacity, case, prefix, run_id)
    save(artifact(prefix, '.command.json'), cmd)
    started = time.time_ns()
    try:
        with artifact(prefix, '.stdout.log').open('xb') as stdout, artifact(prefix, '.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(cmd, cwd=BINARY, env=env, stdout=stdout, stderr=stderr)
            save(artifact(prefix, '.pid.json'), {'pid': child.pid, 'executable': str(BINARY / EXE), 'run_id': run_id})
            code = child.wait(timeout=timeout)
    finally:
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=20)
        finally:
            save(artifact(prefix, '.shutdown.json'), {'pid': child.pid if child else None,
                'returncode': child.returncode if child else None, 'stopped': child is None or child.poll() is not None})
    require(code == 0 and child.poll() == 0, 'Native helper exit error; no retry or partial success')
    return validate_output(prefix, capacity, case, run_id, cmd, started)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-inference', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--timeout', default=600, type=int)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    require(a.allow_inference, 'Explicit --allow-inference required; parent owns GPU window')
    require(1 <= a.timeout <= 3600, 'Timeout must be1..3600')
    out = a.out.resolve()
    require(not out.exists() and out != HERE.parents[1] and HERE.parents[1] not in out.parents,
            'Require NEW exclusive private output outside repository')
    ensure_exclusive()  # BEFORE model hashing and ANY helper startup.
    before = pin_inputs()
    env, recorded_env = controlled_environment()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'scope': SCOPE, 'complete': False, 'pass': False, 'quality_approval': False, 'export_approved': False,
        'model_weights_written': False, 'trace_enabled': False, 'mask_enabled': False,
        'capacities': list(CAPACITIES), 'fixtures': list(CASES), 'atol': 1e-5, 'rtol': 1e-5,
        'environment': recorded_env, 'inputs_before': before,
        'runtime_hash_scope': 'Pinned application EXE/DLL files; external CUDA/system/driver DLLs not covered',
        'comparison_policy': 'ONLY moe_cache_slots projected in memory; original metadata/F32 artifacts immutable; full existing comparator policy',
        'failure_policy': 'Stop at first infrastructure, activation or numerical failure; no retry',
        'probes': [], 'comparisons': []}
    save(out / 'start-manifest.json', manifest)
    error = None
    try:
        nonce = uuid.uuid4().hex
        for case in CASES:
            reference = None
            for capacity in CAPACITIES:
                result = run_probe(out, capacity, case, nonce, env, a.timeout)
                manifest['probes'].append(result)
                if capacity == 32:
                    reference = Path(result['prefix'])
                else:
                    report = compare_capacity(reference, Path(result['prefix']))
                    name = f'cap{capacity}_vs32_{case}.comparison.json'
                    save(out / name, report)
                    manifest['comparisons'].append({'case': case, 'capacity': capacity, 'report': name,
                        'report_sha256': sha(out / name), 'pass': report['pass'], 'bit_identical': report['bit_identical'],
                        'maxabs': report['maxabs'], 'maxrel': report['maxrel'], 'tokens_equal': report['tokens_equal']})
                    require(report['pass'], 'Numerical mismatch; thresholds unchanged and descent stopped')
        require(len(manifest['probes']) == 9 and len(manifest['comparisons']) == 6, 'Incomplete fixed matrix')
    except BaseException as exc:
        error = exc
        manifest['error'] = {'class': type(exc).__name__, 'message': str(exc)}
    finally:
        try:
            after = pin_inputs()
            manifest['inputs_after'] = after
            require(after == before, 'Model/runtime/source/fixtures changed during numerical gate')
            for probe in manifest['probes']:
                require(all(sha(artifact(probe['prefix'], suffix)) == digest for suffix, digest in probe['artifact_sha256'].items()),
                        'Closed probe artifacts changed before publication')
            for report in manifest['comparisons']:
                require(sha(out / report['report']) == report['report_sha256'], 'Comparison report changed before publication')
        except BaseException as exc:
            manifest['postflight_error'] = {'class': type(exc).__name__, 'message': str(exc)}
            if error is None:
                error = exc
        manifest['complete'] = error is None
        manifest['pass'] = error is None
        save(out / 'manifest.json', manifest)
    if error is not None:
        raise error
    return manifest


if __name__ == '__main__':
    main()
