"""Static PE symbol and unchanged dependency checks; never loads a runtime DLL."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

LAB = Path(r'D:\ds4_work\qwen36_reap_lab')
DUMP = Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe')

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(path, option, tag):
    result = subprocess.run([str(DUMP), '/' + option, str(path)], capture_output=True, text=True, check=True)
    (LAB / 'evidence' / (tag + '-' + option + '.txt')).write_text(result.stdout, encoding='utf-8')
    return result.stdout

def exports(path, tag):
    text = dump(path, 'exports', tag)
    result = set(re.findall(r'^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)', text, re.M))
    if not result: raise RuntimeError('Export parser got no symbols: ' + str(path))
    return result

def imports(path, tag):
    text = dump(path, 'imports', tag)
    dll, result = '', []
    for line in text.splitlines():
        match = re.fullmatch(r'    (\S+\.dll)', line, re.I)
        if match: dll = match[1].lower()
        match = re.fullmatch(r'\s+([0-9A-F]+) (\S+)', line)
        if match and dll: result.append((dll, match[2]))
    return result

def main():
    baseline = LAB / 'bin-baseline'
    rebuilt = LAB / 'build' / 'bin' / 'llama.dll'
    old_exports = exports(baseline / 'llama.dll', 'baseline-llama')
    new_exports = exports(rebuilt, 'rebuilt-llama')
    # Candidate changes only llama.dll. Keep common/server/GGML exact to baseline.
    runtime = {p.name.lower(): p for p in baseline.glob('*.dll')}
    runtime['llama.dll'] = rebuilt
    symbols = {name: exports(p, 'resolved-' + name) for name, p in runtime.items()}
    missing = []
    consumers = list(baseline.glob('*.dll')) + list(baseline.glob('*.exe')) + [rebuilt, LAB / 'build/bin/llama-common.dll']
    for index, consumer in enumerate(consumers):
        for dll, name in imports(consumer, 'consumer-' + str(index) + '-' + consumer.name):
            if dll in symbols and name not in symbols[dll]: missing.append({'consumer': str(consumer), 'dll': dll, 'symbol': name})
    report = {'schema_version': 1, 'baseline_sha256': digest(baseline / 'llama.dll'), 'rebuilt_sha256': digest(rebuilt),
              'baseline_exports': len(old_exports), 'rebuilt_exports': len(new_exports),
              'removed_exports': sorted(old_exports - new_exports), 'added_exports': sorted(new_exports - old_exports),
              'missing_consumer_imports': missing, 'static_symbol_gate_pass': old_exports == new_exports and not missing,
              'limitations': 'No DLL loading or inference. Does not establish numerical identity or C++ object-layout ABI; whole llama rebuilt and public headers unchanged.'}
    if report['static_symbol_gate_pass']:
        candidate = LAB / 'bin-candidate/llama.dll'
        shutil.copy2(rebuilt, candidate)
        if digest(candidate) != report['rebuilt_sha256']: raise RuntimeError('Candidate copy mismatch')
        report['candidate_published_lab_only'] = True
    (LAB / 'evidence/abi-review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['static_symbol_gate_pass']: raise SystemExit(1)

if __name__ == '__main__': main()
