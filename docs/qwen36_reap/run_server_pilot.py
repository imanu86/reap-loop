"""Isolated, instrumented calibration integration. Coordinator owns daily restore.
Never executes model-generated tools outside the offline Simulator.
Not a throughput benchmark; routing capture is ON.
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
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--max-output', type=int, default=512)
    a = p.parse_args()
    if not a.allow_inference or not 1 <= a.limit <= 50:
        raise ValueError('Explicit inference opt-in and calibration limit1..50 required')
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
    binary = LAB / 'bin-candidate'
    env, selected_env = environment()
    env['QWEN36_REAP_TRACE'] = str(a.out / 'routing.jsonl')
    env['PATH'] = str(binary) + ';' + env.get('PATH', '')
    command = [str(binary / 'llama-server.exe'), '-m', str(MODEL), '--alias', 'local-pilot',
               '--host', '127.0.0.1', '--port', '8116', '--offline', '--no-warmup', '--no-mmap',
               '-ngl', '99', '--cpu-moe', '--moe-expert-cache', '32', '--moe-expert-cache-inserts', '8',
               '-c', '4096', '-np', '1', '-b', '128', '-ub', '128', '-t', '16', '-tb', '16',
               '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '-fit', 'off', '--jinja',
               '--reasoning', 'on', '--no-reasoning-preserve', '--no-context-shift', '--cache-ram', '0']
    save(a.out / 'manifest.json', {'scope': 'instrumented_calibration_integration_not_performance',
        'limit': a.limit, 'model_sha256': MODEL_SHA, 'command': command,
        'environment': {**selected_env, 'QWEN36_REAP_TRACE': env['QWEN36_REAP_TRACE']},
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
            cmd = [sys.executable, '-B', str(HERE / 'pilot/run_pilot.py'), '--split', 'calibration',
                   '--limit', str(a.limit), '--allow-inference', '--url', 'http://127.0.0.1:8116',
                   '--output-dir', str(a.out / 'pilot'), '--protocol', 'native', '--max-output', str(a.max_output)]
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
