"""Summarize private pilot transcripts without publishing full prompts/reasoning.
Instrumented timing is diagnostic only, not a decode throughput benchmark.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(path):
    cases = []
    errors = Counter()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        errors[row.get('error_class') or 'success'] += 1
        turns = []
        for t in row['turns']:
            response = t.get('response', {})
            choice = response.get('choices', [{}])[0]
            turns.append({'index': t['index'], 'prompt_tokens': t.get('preflight', {}).get('prompt_tokens_preflight'),
                'usage': t.get('usage'), 'finish_reason': choice.get('finish_reason'),
                'selection': t.get('selection'), 'mock_response': t.get('mock_response'),
                'request_seconds': t.get('request_seconds'), 'timings_instrumented': t.get('timings')})
        cases.append({'id': row['id'], 'family': row['family'], 'split': row['split'],
                      'success': row['full_completion'], 'error_class': row.get('error_class'),
                      'private_error': row.get('private_error'), 'wall_seconds': row['wall_seconds'],
                      'turns': turns})
    return {'cases': cases, 'count': len(cases), 'successes': sum(c['success'] for c in cases),
            'error_counts': dict(errors), 'total_wall_seconds': sum(c['wall_seconds'] for c in cases),
            'not_a_throughput_benchmark': True, 'real_actions_executed': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('transcripts', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trace', type=Path, help='Optional actual routing trace to validate, not a mask export')
    a = p.parse_args()
    report = analyze(a.transcripts)
    if a.trace:
        from mask_builder import aggregate
        from run_native_gates import MODEL_SHA
        _, counts, provenance = aggregate([a.trace], MODEL_SHA)
        report['routing'] = {'tokens_per_layer': counts, 'provenance': provenance,
                             'mask_created': False}
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
