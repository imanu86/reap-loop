"""Offline reviewed evidence assembler. No inference, network, corpus loading or autoapproval.
Inputs use an explicit hash-pinned adapter contract; see DESIGN.md. Default is draft.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

_EXPORT = Path(__file__).resolve().parents[1] / 'export' / 'compact_gguf.py'
_spec = importlib.util.spec_from_file_location('_reap_quality_export', _EXPORT)
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)
require = gate.require
canonical = gate.canonical
sha = gate.sha
LIMIT = 64 * 1024**2


def read_raw(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(LIMIT + 1)
    require(len(raw) <= LIMIT, 'input exceeds bounded 64MiB file limit')
    return raw


def digest_value(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'invalid SHA256')
    return value


def referenced(root, ref):
    require(isinstance(ref, dict) and set(ref) == {'path','sha256'}, 'reference requires path/sha256')
    rel = Path(ref['path'])
    require(not rel.is_absolute() and '..' not in rel.parts and rel.parts, 'reference must be relative without traversal')
    target = (root / rel).resolve()
    require(target.is_relative_to(root.resolve()), 'reference escapes evidence directory')
    raw = read_raw(target)
    require(sha(raw) == digest_value(ref['sha256']), 'raw evidence hash mismatch')
    return raw


def strict_json(raw):
    value = gate.decode_json(raw)
    require(isinstance(value, dict), 'JSON object required')
    return value


def transcript_rows(raw):
    lines = raw.splitlines()
    require(len(lines) == 20 and all(line.strip() for line in lines), 'transcript requires exactly20 completed case records')
    return [strict_json(line) for line in lines]


def unique_ids(records, label):
    require(isinstance(records, list) and len(records) == 20, label + ': exactly20 records required')
    result = {}
    for record in records:
        require(isinstance(record, dict), label + ': invalid record')
        case = record.get('id')
        require(isinstance(case, str) and bool(case.strip()) and case not in result, label + ': invalid/duplicate case ID')
        result[case] = record
    return result


def events(value):
    require(isinstance(value, list) and all(isinstance(v, str) and v in gate.CRITICAL_CATEGORIES for v in value),
            'explicit critical list required, unknown category rejected')
    return value


def native_case(record, protocol, seed):
    require(record.get('split') == 'heldout', 'calibration transcript not export eligible')
    for key, expected in [('protocol','native'),('policy_version',protocol['policy_version']),('final_mode',protocol['final_mode'])]:
        require(record.get(key) == expected, 'record protocol mismatch: ' + key)
    require(record.get('real_actions_executed') == 0 and type(record.get('real_actions_executed')) is int,
            'only offline simulated transcripts supported')
    require(type(record.get('full_completion')) is bool, 'raw completion must be boolean')
    turns, actions = record.get('turns'), record.get('actions')
    require(isinstance(turns, list) and 1 <= len(turns) <= protocol['max_turns'], 'invalid/missing turns')
    require(isinstance(actions, list), 'actions required')
    selected_actions, selected_finals = [], []
    recipes = {'greedy':{'temperature':0,'top_k':1,'top_p':1,'min_p':0},
               'qwen-coding':{'temperature':0.6,'top_k':20,'top_p':0.95,'min_p':0,'presence_penalty':0,'repeat_penalty':1}}
    require(protocol['sampling_profile'] in recipes, 'unsupported native sampling recipe; explicit adapter update required')
    for i, turn in enumerate(turns):
        require(isinstance(turn, dict) and type(turn.get('index')) is int and turn['index'] == i, 'turn indices incomplete')
        request = turn.get('request')
        require(isinstance(request, dict), 'raw native request required')
        require(type(request.get('seed')) is int and request['seed'] == seed and
                type(request.get('max_tokens')) is int and request['max_tokens'] == protocol['max_output'], 'request seed/output protocol mismatch')
        for key, value in recipes[protocol['sampling_profile']].items():
            require(type(request.get(key)) in (int,float) and request[key] == value, 'request sampling recipe mismatch')
        require(request.get('tool_choice') == 'auto' and request.get('parallel_tool_calls') is False,
                'native request transport mismatch')
        tools = request.get('tools')
        require(isinstance(tools, list) and bool(tools) and all(isinstance(t,dict) and t.get('type')=='function' and
                isinstance(t.get('function'),dict) for t in tools), 'native public tool definitions required')
        final_tools = sum(t['function'].get('name') == 'final' for t in tools)
        require(final_tools == (1 if protocol['final_mode']=='tool' else 0), 'request final tool schema/protocol mismatch')
        thinking = request.get('chat_template_kwargs', {}).get('enable_thinking')
        require(thinking is (None if protocol['thinking']=='template-default' else protocol['thinking']=='on'), 'request thinking protocol mismatch')
        selection = turn.get('selection')
        if selection is not None:
            require(isinstance(selection, dict) and selection.get('kind') in ('action','final'), 'invalid selection')
            if selection['kind']=='action': selected_actions.append(selection.get('value'))
            else: selected_finals.append(selection.get('value'))
    require(selected_actions == actions, 'raw actions differ from turn selections')
    final = record.get('final')
    require(selected_finals == ([] if final is None else [final]), 'raw final differs from turn selections')
    if record['full_completion']:
        require(isinstance(final, dict) and record.get('error_class') is None and
                turns[-1].get('selection', {}).get('kind') == 'final', 'raw success incomplete/inconsistent')
    else:
        require(isinstance(record.get('error_class'), str) and bool(record['error_class']), 'failed raw record requires error class')
    return actions


def reviewed_case(record, annotation, protocol, seed):
    actions = native_case(record, protocol, seed)
    require(annotation.get('reviewed') is True and isinstance(annotation.get('reviewer'), str) and
            bool(annotation['reviewer'].strip()), 'explicit truthful case reviewer required')
    require(annotation.get('transcript_record_sha256') == sha(canonical(record)), 'annotation case content binding mismatch')
    reviewed = annotation.get('reviewed_full_completion')
    require(type(reviewed) is bool, 'explicit reviewed completion required')
    require(not reviewed or record['full_completion'], 'historical raw failure cannot be upgraded/rescored')
    critical = list(events(annotation.get('case_critical_violations')))
    reviews = annotation.get('action_reviews')
    require(isinstance(reviews, list) and len(reviews) == len(actions), 'all action indices must be explicitly reviewed')
    for index, (action, review) in enumerate(zip(actions, reviews)):
        require(isinstance(review, dict) and type(review.get('action_index')) is int and review['action_index'] == index,
                'annotation action index missing/duplicated/out of order')
        require(review.get('action_sha256') == sha(canonical(action)), 'annotation action content binding mismatch')
        critical.extend(events(review.get('critical_violations')))
    critical.extend(events(annotation.get('final_critical_violations')))
    require(not reviewed or not critical, 'reviewed success inconsistent with critical violations')
    return {'full_completion':record['full_completion'] and reviewed and not critical, 'critical_violations':critical}


def validate_manifest(manifest, run, corpus, protocol, ids):
    require(manifest.get('split') == 'heldout' and manifest.get('protocol') == 'native', 'runner manifest wrong split/transport')
    require(manifest.get('dataset_sha256') == corpus['dataset_sha256'], 'runner dataset hash mismatch')
    digest_value(manifest.get('runner_sha256'))
    listed = manifest.get('episode_ids')
    require(isinstance(listed, list) and len(listed) == 20 and all(isinstance(x,str) for x in listed) and
            set(listed) == ids, 'runner manifest case IDs mismatch')
    require(manifest.get('limit') is None, 'subset smoke cannot qualify for export')
    cfg = manifest.get('config')
    require(isinstance(cfg, dict), 'runner config missing')
    for key in ('policy_version','final_mode','sampling_profile','max_output','max_turns','thinking'):
        require(type(cfg.get(key)) is type(protocol[key]) and cfg[key] == protocol[key], 'runner config mismatch: ' + key)
    require(cfg.get('protocol') == 'native' and cfg.get('diagnostic_raw') is False,
            'runner diagnostic/non-native config not eligible')
    require(type(cfg.get('seed')) is int and cfg['seed'] == run['seed'], 'runner seed mismatch')
    observed_context = manifest.get('server_props', {}).get('default_generation_settings', {}).get('n_ctx')
    require(type(observed_context) is int and observed_context == protocol['context'], 'runner server context mismatch')
    if 'runner_sha256' in protocol:
        require(manifest['runner_sha256'] == protocol['runner_sha256'], 'frozen runner implementation mismatch')


def prepare(policy_path, corpus_path, mask_path, index_path, *, allow_approve=False, reviewer=None):
    policy_raw, corpus_raw, mask_raw, index_raw = [read_raw(p) for p in (policy_path,corpus_path,mask_path,index_path)]
    policy, corpus, mask, index = map(strict_json, (policy_raw,corpus_raw,mask_raw,index_raw))
    gate.validate_mask(mask, gate.PRODUCTION)
    model_sha, mask_sha, corpus_sha = mask['model_sha256'], sha(canonical(mask)), sha(corpus_raw)
    require(corpus.get('split') == 'heldout' and corpus.get('calibration_disjoint') is True, 'only heldout disjoint corpus can assemble export evidence')
    require(set(corpus) <= {'schema_version','split','calibration_disjoint','dataset_sha256','samples'},
            'corpus manifest must contain identity metadata only, never private fixture text')
    samples = unique_ids(corpus.get('samples'), 'corpus')
    for sample in samples.values():
        require(set(sample) == {'id','sha256'}, 'sample manifest must contain only id/sha256, never private content')
        digest_value(sample.get('sha256'))
    digest_value(corpus.get('dataset_sha256'))
    protocol = policy.get('evaluation_protocol')
    require(isinstance(protocol, dict) and policy.get('frozen_before_evaluation') is True, 'frozen evaluation protocol required')
    protocol_sha = sha(canonical(protocol))
    gate.evaluation_protocol_gate(policy, {'evaluation_protocol_sha256':protocol_sha}, {'evaluation_protocol_sha256':protocol_sha})
    trials = protocol.get('trials')
    require(isinstance(trials, list) and bool(trials) and all(type(t) is int and t >= 0 for t in trials) and len(set(trials)) == len(trials),
            'explicit unique nonnegative integer protocol.trials required')
    require(2 * len(protocol['seeds']) * len(trials) <= 256, 'bounded assembler supports at most256 runs')
    expected = {(role,seed,trial) for role in ('baseline','masked') for seed in protocol['seeds'] for trial in trials}
    require(type(index.get('schema_version')) is int and index['schema_version'] == 1 and
            isinstance(index.get('runs'), list) and len(index['runs']) == len(expected), 'all requested seed/trial runs required')
    require(len(expected) <= 256, 'bounded assembler supports at most256 runs')
    accum = {role:{case:[] for case in samples} for role in ('baseline','masked')}
    replica_events = {'baseline':Counter(), 'masked':Counter()}
    provenance, seen = [], set()
    root = Path(index_path).parent
    for run in index['runs']:
        require(isinstance(run,dict) and type(run.get('seed')) is int and type(run.get('trial')) is int, 'run seed/trial missing')
        key = (run.get('role'),run['seed'],run['trial'])
        require(key in expected and key not in seen, 'unexpected/duplicate seed/trial run')
        seen.add(key)
        role, seed, trial = key
        require(run.get('completed') is True and run.get('eligible_quality_evaluation') is True and
                run.get('skip_chat_parsing') is False, 'run incomplete/ineligible/parser bypass')
        require(type(run.get('reasoning_preserve')) is bool and run['reasoning_preserve'] == protocol['reasoning_preserve'], 'run reasoning mode mismatch')
        for name, expected_value in [('model_sha256',model_sha),('corpus_manifest_sha256',corpus_sha),
                                     ('evaluation_protocol_sha256',protocol_sha),('mask_sha256',mask_sha if role=='masked' else None)]:
            require(name in run and run[name] == expected_value, 'run evidence binding mismatch: ' + name)
        raw = referenced(root, run['transcript'])
        manifest_raw = referenced(root, run['manifest'])
        annotation_raw = referenced(root, run['annotations'])
        records = unique_ids(transcript_rows(raw), 'transcript')
        require(set(records) == set(samples), 'transcript case IDs mismatch')
        validate_manifest(strict_json(manifest_raw), run, corpus, protocol, set(samples))
        annotations = strict_json(annotation_raw)
        require(type(annotations.get('schema_version')) is int and annotations['schema_version'] == 1 and
                annotations.get('transcript_sha256') == sha(raw), 'review transcript hash mismatch')
        reviews = unique_ids(annotations.get('cases'), 'annotations')
        require(set(reviews) == set(samples), 'review case IDs mismatch')
        for case in samples:
            item = reviewed_case(records[case], reviews[case], protocol, seed)
            accum[role][case].append(item)
            replica_events[role].update((seed,trial,case,cat) for cat in item['critical_violations'])
        provenance.append({'role':role,'seed':seed,'trial':trial,'transcript_sha256':sha(raw),
                           'manifest_sha256':sha(manifest_raw),'annotations_sha256':sha(annotation_raw)})
    require(seen == expected, 'incomplete requested replicas')
    reports = {}
    for role in ('baseline','masked'):
        cases = []
        for case, replicas in accum[role].items():
            counts = Counter(cat for replica in replicas for cat in replica['critical_violations'])
            critical = [cat for cat in sorted(counts) for _ in range(counts[cat])]
            cases.append({'id':case,'full_completion':all(r['full_completion'] for r in replicas),
                          'critical_violations':critical})
        completions = sum(r['full_completion'] for r in cases)
        reports[role] = {'model_sha256':model_sha,'mask_sha256':mask_sha if role=='masked' else None,
            'corpus_manifest_sha256':corpus_sha,'evaluation_protocol_sha256':protocol_sha,'complete':True,
            'cases':cases,'replicas_per_case':len(protocol['seeds'])*len(trials),
            'metrics':{'full_completion_count':completions,'full_completion_rate':completions/20,
                       'critical_violations':sum(len(r['critical_violations']) for r in cases)}}
    failures = []
    if replica_events['masked'] - replica_events['baseline']:
        failures.append('NEW critical occurrences within matched seed/trial/case/category; replica cancellation forbidden')
    try:
        gate.decode_screening_gate(corpus,reports['baseline'],reports['masked'],policy)
    except ValueError as exc:
        failures.append(str(exc))
    if allow_approve:
        require(isinstance(reviewer,str) and bool(reviewer.strip()), 'explicit truthful approving reviewer required')
        require(not failures, '; '.join(failures))
    bundle = {'corpus.json':corpus_raw, 'policy.json':policy_raw,
              'baseline.json':canonical(reports['baseline']), 'masked.json':canonical(reports['masked'])}
    refs = {role:{'path':file,'sha256':sha(bundle[file])} for role,file in
            [('corpus_manifest','corpus.json'),('policy','policy.json'),('baseline','baseline.json'),('masked','masked.json')]}
    proof = {'schema_version':1,'decision':'draft','evaluation_mode':'heldout-original-vs-masked',
             'model_sha256':model_sha,'mask_sha256':mask_sha,'calibration_disjoint':True,
             'reviewer':reviewer,'artifacts':refs,'approval_requested':allow_approve,'screening_failures':failures,
             'provenance':{'assembler_sha256':sha(read_raw(__file__)),'exporter_sha256':sha(read_raw(_EXPORT)),
                           'mask_file_sha256':sha(mask_raw),'input_index_sha256':sha(index_raw),'runs':provenance,
                           'no_raw_transcript_content_published':True}}
    return bundle, proof, mask


def assemble(policy_path, corpus_path, mask_path, index_path, output, *, allow_approve=False, reviewer=None):
    """Publish summary files exclusively; approval marker is last, after real exporter verification."""
    output = Path(output)
    require(output.parent.is_dir() and not output.exists(), 'output must be a NEW directory with existing parent')
    bundle, proof, mask = prepare(policy_path,corpus_path,mask_path,index_path,allow_approve=allow_approve,reviewer=reviewer)
    created, output_created = [], False
    with tempfile.TemporaryDirectory(prefix='__quality_', dir=output.parent) as tmp:
        stage = Path(tmp)
        for filename, raw in bundle.items():
            with (stage/filename).open('xb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
        if allow_approve:
            # Private candidate only. No published approval before mandatory exporter gates pass.
            proof['decision'] = 'pass'
            candidate = canonical(proof)
            with (stage/'quality.json').open('xb') as f:
                f.write(candidate); f.flush(); os.fsync(f.fileno())
            gate.quality_gate(stage/'quality.json',sha(candidate),mask,mask['model_sha256'])
        else:
            with (stage/'quality.json').open('xb') as f:
                f.write(canonical(proof)); f.flush(); os.fsync(f.fileno())
        try:
            output.mkdir()  # exclusive reservation; never replaces another directory
            output_created = True
            for filename in [*bundle,'quality.json']:
                os.link(stage/filename, output/filename)  # no-clobber, approval last
                created.append(output/filename)
        except BaseException:
            for p in reversed(created): p.unlink()
            if output_created:
                try: output.rmdir()
                except OSError: pass  # foreign files are never removed
            raise
    return {'decision':proof['decision'],'quality_proof_sha256':sha(canonical(proof)),
            'output':str(output),'raw_content_published':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('policy','corpus','mask','index','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reviewer')
    p.add_argument('--allow-approve',action='store_true')
    a = p.parse_args()
    result = assemble(a.policy,a.corpus,a.mask,a.index,a.output,allow_approve=a.allow_approve,reviewer=a.reviewer)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
