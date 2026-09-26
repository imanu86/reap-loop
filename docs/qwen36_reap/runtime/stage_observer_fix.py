"""Parent-authorized llama-only update; preserve all helper/common/baseline bytes."""
import json
from pathlib import Path
import runpy
import shutil

ROOT = Path(__file__).parent
TOOLS = runpy.run_path(str(ROOT / 'verify_abi.py'))
LAB, digest, imports, exports = (TOOLS[name] for name in ('LAB', 'digest', 'imports', 'exports'))

def main():
    helper_path = LAB / 'evidence/helper-build-manifest.json'
    helper = json.loads(helper_path.read_text(encoding='utf-8'))
    old_abi_path = LAB / 'evidence/abi-review.json'
    old_abi = json.loads(old_abi_path.read_text(encoding='utf-8'))
    old_sha = old_abi['rebuilt_sha256']
    candidate = LAB / 'bin-candidate/llama.dll'
    if digest(candidate) != old_sha: raise RuntimeError('Unexpected prior candidate')
    for row in helper['files']:
        if digest(Path(row['path'])) != row['sha256']:
            raise RuntimeError('Gate arm differs from frozen manifest: ' + row['path'])
    frozen = json.loads((LAB / 'evidence/frozen-build-inputs.json').read_text(encoding='utf-8'))
    for row in frozen['records']:
        path = Path(row['path'])
        if path.parent.name == 'bin-baseline' and digest(path) != row['sha256']:
            raise RuntimeError('Original baseline changed')
    archive = LAB / 'evidence/observer-fix-before'
    archive.mkdir()
    for path in (helper_path, old_abi_path, ROOT / 'qwen36-reap-runtime.patch', ROOT / 'patch-manifest.json', ROOT / 'build-summary.json'):
        shutil.copy2(path, archive / path.name)
    TOOLS['main']()  # Full static ABI check and update bin-candidate/llama.dll only.
    new_abi = json.loads(old_abi_path.read_text(encoding='utf-8'))
    new_sha = new_abi['rebuilt_sha256']
    symbols = exports(candidate, 'observer-fix-helper-resolved-llama')
    for arm, exe in helper['arms'].items():
        for dll, name in imports(Path(exe), 'observer-fix-helper-' + arm):
            if dll == 'llama.dll' and name not in symbols:
                raise RuntimeError('New runtime lacks helper import: ' + name)
    target = LAB / 'build/gate-candidate/llama.dll'
    shutil.copy2(candidate, target)
    if digest(target) != new_sha: raise RuntimeError('Gate candidate copy mismatch')
    for row in helper['files']:
        if Path(row['path']) == target: row['sha256'] = new_sha
        if digest(Path(row['path'])) != row['sha256']:
            raise RuntimeError('Non-target gate file changed: ' + row['path'])
    helper['candidate_runtime_sha256'] = new_sha
    helper['candidate_runtime_previous_sha256'] = old_sha
    helper['candidate_runtime_revision'] = 'deferred_ids_at_existing_final_weights_no_CONT'
    helper['candidate_runtime_build_job'] = 'pwsh-111'
    helper['candidate_runtime_numerical_status'] = 'UNTESTED_PARENT_GPU_GATE_REQUIRED'
    text = json.dumps(helper, indent=2) + '\n'
    helper_path.write_text(text, encoding='utf-8')
    (ROOT / 'helper-build-manifest.json').write_text(text, encoding='utf-8')
    report = {'schema_version': 1, 'build_job': 'pwsh-111', 'status': 'build_static_abi_pass_numerical_gate_pending',
              'candidate_llama_sha256': new_sha, 'previous_candidate_sha256': old_sha,
              'helper_sha256_unchanged': helper['helper_sha256'], 'common_sha256_unchanged': helper['fresh_common_sha256_both_arms'],
              'baseline_and_non_target_arm_files_unchanged': True, 'only_updated_binaries': [str(candidate), str(target)],
              'source_sha256': {name: digest(LAB / 'source/src' / name) for name in ('llama-graph.cpp', 'llama-reap.cpp')},
              'prior_gate109': 'OFF bitidentical firstfixture; trace FAIL maxabs0.8196802139; no secondfixture/allkept',
              'limitations': 'No runtime execution/GPU by worker, no claim trace drift resolved.'}
    (LAB / 'evidence/observer-fix-build.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'observer-fix-build.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
