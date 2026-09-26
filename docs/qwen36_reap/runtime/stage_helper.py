"""Stage numerical gate arms with identical fresh common and unchanged original GGML.
No runtime binary, including --help, is executed.
"""
import json
from pathlib import Path
import runpy
import shutil

ROOT = Path(__file__).parent
TOOLS = runpy.run_path(str(ROOT / 'verify_abi.py'))
LAB, digest, exports, imports = (TOOLS[name] for name in ('LAB', 'digest', 'exports', 'imports'))

def main():
    exe = LAB / 'build/bin/reap-native-gate.exe'
    common = LAB / 'build/bin/llama-common.dll'
    fresh = LAB / 'build/bin/llama.dll'
    baseline = LAB / 'bin-baseline'
    arms = {name: LAB / 'build' / ('gate-' + name) for name in ('baseline', 'candidate')}
    if any(p.exists() for p in arms.values()):
        raise RuntimeError('Refuse to overwrite existing gate arms')
    frozen = json.loads((LAB / 'evidence/frozen-build-inputs.json').read_text(encoding='utf-8'))
    for record in frozen['records']:
        path = Path(record['path'])
        if path.parent == baseline and digest(path) != record['sha256']:
            raise RuntimeError('Frozen baseline changed: ' + str(path))
    abi = json.loads((LAB / 'evidence/abi-review.json').read_text(encoding='utf-8'))
    if not abi['static_symbol_gate_pass'] or digest(fresh) != abi['rebuilt_sha256']:
        raise RuntimeError('Candidate runtime not matched to ABI gate')
    runtime = {p.name.lower(): p for p in baseline.glob('*.dll')}
    runtime['llama-common.dll'] = common
    exports_by_dll = {name: exports(path, 'gate-resolved-' + name) for name, path in runtime.items()}
    candidate_exports = exports(fresh, 'gate-candidate-llama')
    missing = []
    for consumer in (exe, common, fresh):
        for dll, symbol in imports(consumer, 'gate-imports-' + consumer.name):
            if dll in exports_by_dll and symbol not in exports_by_dll[dll]:
                missing.append({'consumer': str(consumer), 'dll': dll, 'symbol': symbol})
            if dll == 'llama.dll' and symbol not in candidate_exports:
                missing.append({'consumer': str(consumer), 'dll': 'candidate-llama.dll', 'symbol': symbol})
    if missing: raise RuntimeError('Unresolved gate imports: ' + json.dumps(missing))
    records = []
    for name, dest in arms.items():
        dest.mkdir()
        sources = list(baseline.iterdir())
        for source in sources:
            if not source.is_file(): raise RuntimeError('Unexpected baseline directory')
            if source.name == 'llama-common.dll': source = common
            elif source.name == 'llama.dll' and name == 'candidate': source = fresh
            target = dest / source.name
            shutil.copy2(source, target)
            sha = digest(source)
            if digest(target) != sha: raise RuntimeError('Gate arm copy mismatch')
            records.append({'arm': name, 'path': str(target), 'source': str(source), 'sha256': sha})
        target = dest / exe.name
        shutil.copy2(exe, target)
        if digest(target) != digest(exe): raise RuntimeError('Helper copy mismatch')
        records.append({'arm': name, 'path': str(target), 'source': str(exe), 'sha256': digest(exe)})
    report = {'schema_version': 1, 'status': 'compiled_and_staged_not_executed',
              'helper_sha256': digest(exe), 'helper_source_sha256': digest(ROOT.parent / 'gate/native_gate.cpp'),
              'build_command': str(ROOT / 'build-helper.cmd'), 'build_command_sha256': digest(ROOT / 'build-helper.cmd'),
              'fresh_common_sha256_both_arms': digest(common), 'missing_named_imports': missing,
              'original_ggml_unchanged': True, 'baseline_directory_unchanged': True,
              'limitations': 'Static PE import verification only. No helper invocation, model load, GPU, inference or runtime identity verification.',
              'arms': {k: str(v / exe.name) for k, v in arms.items()}, 'files': records}
    text = json.dumps(report, indent=2) + '\n'
    (LAB / 'evidence/helper-build-manifest.json').write_text(text, encoding='utf-8')
    (ROOT / 'helper-build-manifest.json').write_text(text, encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, indent=2))

if __name__ == '__main__': main()
