"""Parent-authorized helper-only refresh; archive old binary and manifest first."""
import json
from pathlib import Path
import runpy
import shutil

ROOT = Path(__file__).parent
TOOLS = runpy.run_path(str(ROOT / 'verify_abi.py'))
LAB, digest, exports, imports = (TOOLS[name] for name in ('LAB', 'digest', 'exports', 'imports'))

def main():
    manifest_path = LAB / 'evidence/helper-build-manifest.json'
    previous_bytes = manifest_path.read_bytes()
    report = json.loads(previous_bytes)
    previous_sha = report['helper_sha256']
    exe = LAB / 'build/bin/reap-native-gate.exe'
    new_sha = digest(exe)
    if new_sha == previous_sha: raise RuntimeError('No new helper revision')
    for row in report['files']:
        if digest(Path(row['path'])) != row['sha256']:
            raise RuntimeError('Existing arm fingerprint changed: ' + row['path'])
    missing = []
    for arm, helper in report['arms'].items():
        directory = Path(helper).parent
        symbols = {p.name.lower(): exports(p, 'helper-refresh-' + arm + '-' + p.name) for p in directory.glob('*.dll')}
        for dll, name in imports(exe, 'helper-refresh-' + arm + '-exe'):
            if dll in symbols and name not in symbols[dll]:
                missing.append({'arm': arm, 'dll': dll, 'symbol': name})
    if missing: raise RuntimeError('Unresolved helper imports: ' + json.dumps(missing))
    archive = LAB / 'evidence' / ('helper-before-' + previous_sha[:16])
    archive.mkdir()
    shutil.copy2(Path(report['arms']['baseline']), archive / 'reap-native-gate.exe')
    if digest(archive / 'reap-native-gate.exe') != previous_sha:
        raise RuntimeError('Old helper archive mismatch')
    (archive / 'manifest.json').write_bytes(previous_bytes)
    for helper in report['arms'].values():
        shutil.copy2(exe, helper)
        if digest(Path(helper)) != new_sha: raise RuntimeError('Helper refresh copy mismatch')
    for row in report['files']:
        if Path(row['path']).name == exe.name:
            row['sha256'] = new_sha
        if digest(Path(row['path'])) != row['sha256']:
            raise RuntimeError('Post-refresh arm fingerprint mismatch: ' + row['path'])
    history = report.setdefault('parent_reported_gate_history', [])
    if report.get('parent_reported_previous_gate') and report['parent_reported_previous_gate'] not in history:
        history.append(report['parent_reported_previous_gate'])
    report.update({'helper_sha256': new_sha,
                   'helper_source_sha256': digest(ROOT.parent / 'gate/native_gate.cpp'),
                   'previous_helper_sha256': previous_sha, 'previous_helper_archive': str(archive),
                   'helper_revision': 'Unary --no-mmap allowed for pinned host demand-GPU weights',
                   'parent_reported_previous_gate': {'job': '106', 'outcome': 'baseline_demand_gpu_guard_before_numerical_results', 'reason': 'Original M3 requires --no-mmap for pinned host weights', 'numerical_data': False},
                   'build_job': 'pwsh-107', 'build_log': str(LAB / 'evidence/build-helper-retry3-no-mmap.log'),
                   'missing_named_imports': missing, 'status': 'compiled_and_staged_not_executed'})
    text = json.dumps(report, indent=2) + '\n'
    manifest_path.write_text(text, encoding='utf-8')
    (ROOT / 'helper-build-manifest.json').write_text(text, encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, indent=2))

if __name__ == '__main__': main()
