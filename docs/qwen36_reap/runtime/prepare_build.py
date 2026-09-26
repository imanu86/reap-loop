"""Freeze verified baseline and GGML imports. Does not execute any runtime binary."""
import hashlib
import json
import shutil
from pathlib import Path

LAB = Path(r'D:\ds4_work\qwen36_reap_lab')
LIVE = Path(r'D:\ds4_work\bin_q36m3')
M3 = Path(r'D:\ds4_work\elastico_q36_m3\evidence\q36m3-build-manifest.json')

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    manifest = json.loads(M3.read_text(encoding='utf-8'))
    expected = manifest['unchanged_runtime_sha256'] | manifest['q36m3_sha256']
    if (LAB / 'bin-baseline').exists() or (LAB / 'bin-candidate').exists():
        raise RuntimeError('Refuse to overwrite runtime directories')
    for name, sha in expected.items():
        if digest(LIVE / name) != sha: raise RuntimeError('Live fingerprint mismatch: ' + name)
    libs = {Path(p).name: (Path(p), sha) for p, sha in manifest['reused_input_sha256'].items() if Path(p).suffix == '.lib'}
    if set(libs) != {'ggml.lib', 'ggml-base.lib', 'ggml-cpu.lib', 'ggml-cuda.lib'}:
        raise RuntimeError('Unexpected GGML import library set')
    for path, sha in libs.values():
        if digest(path) != sha: raise RuntimeError('Import library fingerprint mismatch: ' + str(path))
    (LAB / 'evidence').mkdir(exist_ok=True)
    records = []
    for subdir in ('bin-baseline', 'bin-candidate'):
        dest = LAB / subdir
        dest.mkdir()
        for name, sha in expected.items():
            shutil.copy2(LIVE / name, dest / name)
            if digest(dest / name) != sha: raise RuntimeError('Runtime copy mismatch')
            records.append({'path': str(dest / name), 'source': str(LIVE / name), 'sha256': sha})
    out = LAB / 'build' / 'verified-libs'
    out.mkdir(parents=True)
    for name, (path, sha) in libs.items():
        shutil.copy2(path, out / name)
        if digest(out / name) != sha: raise RuntimeError('Library copy mismatch')
        records.append({'path': str(out / name), 'source': str(path), 'sha256': sha})
    report = {'schema_version': 1, 'cuda_provenance': 'donor CMakeCache CUDA12.6.85 sm86; unchanged M3 runtime DLLs/import libs', 'records': records}
    (LAB / 'evidence' / 'frozen-build-inputs.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'frozen_files': len(records), 'report': str(LAB / 'evidence' / 'frozen-build-inputs.json')}))

if __name__ == '__main__': main()
