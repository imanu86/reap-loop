"""Isolated, instrumented calibration integration. Coordinator owns daily restore.
Never executes model-generated tools outside the offline Simulator.
Not a throughput benchmark; routing capture is ON unless explicitly disabled for controls.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from run_native_gates import LAB, MODEL, MODEL_SHA, HERE, environment, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-inference', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--limit', type=int)
    selection.add_argument('--episode-ids', help='Explicit comma-separated calibration IDs only')
    p.add_argument('--policy-version', choices=('v1', 'v2', 'v3'), default='v1')
    p.add_argument('--context', type=int, choices=(4096, 6144), default=4096)
    p.add_argument('--runtime', choices=('baseline', 'candidate'), default='candidate')
    p.add_argument('--no-trace', action='store_true', help='Diagnostic uninstrumented control, not a performance promotion')
    p.add_argument('--diagnostic-raw', action='store_true')
    p.add_argument('--skip-chat-parsing', action='store_true', help='DIAGNOSTIC ONLY: bypass parser AND generated grammar; not a pilot quality score')
    p.add_argument('--max-output', type=int, default=512)
    p.add_argument('--final-mode', choices=('content', 'tool'), default='content')
    p.add_argument('--sampling-profile', choices=('greedy', 'qwen-coding'), default='greedy')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if not 1 <= a.max_output < a.context:
        raise ValueError('Require 1 <= max_output < context before loading model')
    if not 0 <= a.seed < 4294967295:
        raise ValueError('Require deterministic seed0..4294967294, never random sentinel')
    if a.limit is None and a.episode_ids is None:
        a.limit = 2
    if not a.allow_inference or (a.limit is not None and not 1 <= a.limit <= 50):
        raise ValueError('Explicit inference opt-in and calibration limit1..50 required')
    if a.episode_ids is not None:
        sys.path.insert(0, str(HERE / 'pilot'))
        from run_pilot import select_episodes
        select_episodes('calibration', None, a.episode_ids)
    if a.runtime == 'baseline' and not a.no_trace:
        raise ValueError('Original runtime does not implement REAP capture; require --no-trace')
    if a.skip_chat_parsing and not (a.no_trace and a.diagnostic_raw):
        raise ValueError('Parser/grammar bypass requires --no-trace --diagnostic-raw and is not a quality evaluation')
    a.out = a.out.resolve()
    existing = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq llama-server.exe', '/NH'], capture_output=True, text=True, check=True)
    if 'llama-server.exe' in existing.stdout.lower():
        raise RuntimeError('Existing server: coordinator must handle approved daily suspension')
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(('127.0.0.1', 8116)) == 0:
            raise RuntimeError('Private port8116 already occupied')
    a.out.mkdir(parents=True, exist_ok=False)
    if sha(MODEL) != MODEL_SHA:
        raise RuntimeError('Original model hash changed')
    binary = LAB / ('bin-' + a.runtime)
    env, selected_env = environment()
    trace_setting = {} if a.no_trace else {'QWEN36_REAP_TRACE': str(a.out / 'routing.jsonl')}
    env.update(trace_setting)
    env['PATH'] = str(binary) + ';' + env.get('PATH', '')
    command = [str(binary / 'llama-server.exe'), '-m', str(MODEL), '--alias', 'local-pilot',
               '--host', '127.0.0.1', '--port', '8116', '--offline', '--no-warmup', '--no-mmap',
               '-ngl', '99', '--cpu-moe', '--moe-expert-cache', '32', '--moe-expert-cache-inserts', '8',
               '-c', str(a.context), '-np', '1', '-b', '128', '-ub', '128', '-t', '16', '-tb', '16',
               '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '-fit', 'off', '--jinja',
               '--reasoning', 'on', '--no-reasoning-preserve', '--no-context-shift', '--cache-ram', '0']
    if a.skip_chat_parsing:
        command.append('--skip-chat-parsing')
    scope = ('parser_grammar_bypass_DIAGNOSTIC_ONLY' if a.skip_chat_parsing else
             'context_override_DIAGNOSTIC_ONLY' if a.context != 4096 else 'calibration_integration_not_performance')
    save(a.out / 'manifest.json', {'scope': scope,
        'skip_chat_parsing': a.skip_chat_parsing,
        'eligible_quality_evaluation': not a.skip_chat_parsing and a.context == 4096,
        'runtime_arm': a.runtime, 'trace_enabled': not a.no_trace, 'diagnostic_raw': a.diagnostic_raw,
        'limit': a.limit, 'episode_ids': a.episode_ids, 'policy_version': a.policy_version, 'final_mode': a.final_mode,
        'max_output': a.max_output, 'context': a.context, 'sampling_profile': a.sampling_profile, 'seed': a.seed,
        'model_sha256': MODEL_SHA, 'command': command,
        'environment': {**selected_env, **trace_setting},
        'runtime_sha256': {f.name: sha(f) for f in binary.iterdir() if f.suffix.lower() in ('.dll', '.exe')},
        'coordinator_sha256': sha(__file__)})
    server = None
    try:
        with (a.out / 'server.stdout.log').open('xb') as stdout, (a.out / 'server.stderr.log').open('xb') as stderr:
            server = subprocess.Popen(command, cwd=binary, env=env, stdout=stdout, stderr=stderr)
            save(a.out / 'server.pid.json', {'pid': server.pid, 'executable': str(binary / 'llama-server.exe')})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Private server exited; inspect server.stderr.log')
                try:
                    with opener.open('http://127.0.0.1:8116/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Private server startup exceeded180s')
                time.sleep(0.5)
            selection_args = ['--episode-ids', a.episode_ids] if a.episode_ids is not None else ['--limit', str(a.limit)]
            cmd = [sys.executable, '-B', str(HERE / 'pilot/run_pilot.py'), '--split', 'calibration',
                   *selection_args, '--allow-inference', '--url', 'http://127.0.0.1:8116',
                   '--output-dir', str(a.out / 'pilot'), '--protocol', 'native',
                   '--policy-version', a.policy_version, '--max-output', str(a.max_output),
                   '--sampling-profile', a.sampling_profile, '--seed', str(a.seed), '--final-mode', a.final_mode,
                   '--budget', str(a.context)]
            if a.diagnostic_raw:
                cmd.append('--diagnostic-raw')
            save(a.out / 'pilot.command.json', cmd)
            subprocess.run(cmd, check=True, timeout=3600)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)
        save(a.out / 'shutdown.json', {'pid': server.pid if server else None,
                                     'returncode': server.returncode if server else None,
                                     'stopped': server is None or server.poll() is not None})


if __name__ == '__main__':
    main()
