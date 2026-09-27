"""Invented20-case offline evidence only. Never load project corpus or inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import assemble_evidence as a


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='__synthetic_',dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.out = self.root/'result'
        self.protocol = {'transport':'native','policy_version':'v2','final_mode':'tool',
            'sampling_profile':'greedy','seeds':[0,42],'trials':[0], 'context':8192,
            'max_output':512,'max_turns':12,'thinking':'off','reasoning_preserve':False,
            'skip_chat_parsing':False,'runner_sha256':'a'*64,'template_sha256':'b'*64}
        self.policy = {'frozen_before_evaluation':True,'evaluation_protocol':self.protocol,
            'decode_gates':{'protocol':'qwen36-reap-decode-screening-v1','heldout_cases':20,
                'min_baseline_full_completion':16,'max_new_noncritical_failures':1,
                'max_new_critical_violations':0,'critical_categories':sorted(a.gate.CRITICAL_CATEGORIES)},'checks':[]}
        self.corpus = {'split':'heldout','calibration_disjoint':True,'dataset_sha256':'c'*64,
            'samples':[{'id':f'invented-{i}','sha256':a.sha(f'invented-{i}'.encode())} for i in range(20)]}
        self.mask = {'schema_version':1,'architecture':'qwen35moe','layer_count':40,'expert_count':256,
            'top_k':8,'model_sha256':a.gate.MODEL_SHA,'layers':{str(i):list(range(128)) for i in range(40)}}
        self.save('policy.json',self.policy); self.save('corpus.json',self.corpus); self.save('mask.json',self.mask)
        self.index = {'schema_version':1,'runs':[]}
        self.records, self.annotations, self.manifests = {}, {}, {}
        for role in ('baseline','masked'):
            for seed in self.protocol['seeds']:
                key = (role,seed,0)
                request = {'seed':seed,'max_tokens':512,'temperature':0,'top_k':1,'top_p':1,'min_p':0,
                    'tool_choice':'auto','parallel_tool_calls':False,'chat_template_kwargs':{'enable_thinking':False},
                    'tools':[{'type':'function','function':{'name':'sim_read'}},{'type':'function','function':{'name':'final'}}],
                    'messages':[{'role':'user','content':'PRIVATE_INPUT_SENTINEL'}]}
                rows, reviews = [], []
                for i in range(20):
                    action = {'op':'sim.read','args':{'target':i}}
                    final = {'done':True}
                    row = {'id':f'invented-{i}','split':'heldout','protocol':'native','policy_version':'v2','final_mode':'tool',
                        'full_completion':True,'error_class':None,'real_actions_executed':0,'actions':[action],'final':final,
                        'turns':[{'index':0,'request':copy.deepcopy(request),'selection':{'kind':'action','value':action},
                                  'reasoning':'PRIVATE_REASONING_SENTINEL'},
                                 {'index':1,'request':copy.deepcopy(request),'selection':{'kind':'final','value':final}}]}
                    rows.append(row)
                    reviews.append({'id':row['id'],'transcript_record_sha256':a.sha(a.canonical(row)),
                        'reviewed':True,'reviewer':'TEST_ONLY authorized reviewer','reviewed_full_completion':True,
                        'case_critical_violations':[],'final_critical_violations':[],
                        'action_reviews':[{'action_index':0,'action_sha256':a.sha(a.canonical(action)),'critical_violations':[]}]})
                self.records[key], self.annotations[key] = rows, {'schema_version':1,'cases':reviews}
                cfg = {k:self.protocol[k] for k in ('policy_version','final_mode','sampling_profile','max_output','max_turns','thinking')}
                cfg.update(protocol='native',diagnostic_raw=False,seed=seed)
                self.manifests[key] = {'split':'heldout','protocol':'native','dataset_sha256':self.corpus['dataset_sha256'],
                    'runner_sha256':'a'*64,'episode_ids':[r['id'] for r in rows],'limit':None,'config':cfg,
                    'server_props':{'default_generation_settings':{'n_ctx':8192}}}
                self.index['runs'].append({'role':role,'seed':seed,'trial':0,'completed':True,
                    'eligible_quality_evaluation':True,'skip_chat_parsing':False,'reasoning_preserve':False,
                    'model_sha256':a.gate.MODEL_SHA,'mask_sha256':a.sha(a.canonical(self.mask)) if role=='masked' else None,
                    'corpus_manifest_sha256':a.sha(a.canonical(self.corpus)),
                    'evaluation_protocol_sha256':a.sha(a.canonical(self.protocol))})
        self.sync()

    def tearDown(self): self.tmp.cleanup()

    def save(self,name,value):
        raw = a.canonical(value)
        (self.root/name).write_bytes(raw)
        return {'path':name,'sha256':a.sha(raw)}

    def sync(self):
        for run in self.index['runs']:
            key = (run['role'],run['seed'],run['trial'])
            if key not in self.records: continue
            stem = f'{key[0]}-{key[1]}-{key[2]}'
            raw = b'\n'.join(a.canonical(r) for r in self.records[key])+b'\n'
            name = stem+'.jsonl'
            (self.root/name).write_bytes(raw)
            run['transcript'] = {'path':name,'sha256':a.sha(raw)}
            self.annotations[key]['transcript_sha256'] = a.sha(raw)
            run['annotations'] = self.save(stem+'-review.json',self.annotations[key])
            run['manifest'] = self.save(stem+'-manifest.json',self.manifests[key])
        self.save('index.json',self.index)

    def assemble(self,approve=False,reviewer=None):
        return a.assemble(self.root/'policy.json',self.root/'corpus.json',self.root/'mask.json',self.root/'index.json',
                          self.out,allow_approve=approve,reviewer=reviewer)

    def fail(self,case,role='masked',seed=0,critical=None):
        key = (role,seed,0)
        row = self.records[key][case]
        row['full_completion'] = False; row['error_class'] = 'final_validation'
        annotation = self.annotations[key]['cases'][case]
        annotation['reviewed_full_completion'] = False
        annotation['transcript_record_sha256'] = a.sha(a.canonical(row))
        if critical is not None: annotation['action_reviews'][0]['critical_violations'] = critical

    def test_default_draft_never_passes_export_gate(self):
        result = self.assemble()
        self.assertEqual(result['decision'],'draft')
        proof = a.strict_json((self.out/'quality.json').read_bytes())
        self.assertIsNone(proof['reviewer'])
        with self.assertRaises(ValueError):
            a.gate.quality_gate(self.out/'quality.json',result['quality_proof_sha256'],self.mask,a.gate.MODEL_SHA)

    def test_explicit_approved_bundle_passes_real_export_gate_without_private_content(self):
        result = self.assemble(True,'TEST_ONLY authorized agent')
        self.assertEqual(result['decision'],'pass')
        a.gate.quality_gate(self.out/'quality.json',result['quality_proof_sha256'],self.mask,a.gate.MODEL_SHA)
        proof = a.strict_json((self.out/'quality.json').read_bytes())
        self.assertEqual(len(proof['provenance']['runs']),4)
        for path in self.out.iterdir():
            self.assertNotIn(b'PRIVATE_REASONING_SENTINEL',path.read_bytes())
            self.assertNotIn(b'PRIVATE_INPUT_SENTINEL',path.read_bytes())
        self.assertEqual(a.strict_json((self.out/'masked.json').read_bytes())['replicas_per_case'],2)

    def test_no_auto_approval_or_missing_reviewer(self):
        with self.assertRaisesRegex(ValueError,'reviewer'): self.assemble(True)
        self.assertFalse(self.out.exists())

    def test_all_requested_seed_trial_case_runs_required(self):
        self.index['runs'].pop(); self.sync()
        with self.assertRaises(ValueError): self.assemble(True,'reviewer')
        self.assertFalse(self.out.exists())

    def test_trial_schedule_is_explicit_and_frozen(self):
        self.protocol.pop('trials'); self.save('policy.json',self.policy)
        with self.assertRaisesRegex(ValueError,'trials'): self.assemble(True,'reviewer')
        self.protocol['trials'] = [0,1]; self.save('policy.json',self.policy)
        with self.assertRaisesRegex(ValueError,'all requested'): self.assemble(True,'reviewer')

    def test_duplicate_run_and_case_rejected(self):
        self.index['runs'][-1] = copy.deepcopy(self.index['runs'][0]); self.sync()
        with self.assertRaises(ValueError): self.assemble(True,'reviewer')

    def test_ineligible_parser_bypass_and_incomplete_runs_rejected(self):
        run = self.index['runs'][0]
        for key,value in [('eligible_quality_evaluation',False),('skip_chat_parsing',True),('completed',False),
                          ('evaluation_protocol_sha256','0'*64),('model_sha256','0'*64),('corpus_manifest_sha256','0'*64)]:
            with self.subTest(key=key):
                old = run[key]; run[key] = value; self.sync()
                with self.assertRaises(ValueError): self.assemble(True,'reviewer')
                run[key] = old

    def test_calibration_never_eligible_even_with_approval_flags(self):
        self.corpus['split'] = 'calibration'; self.save('corpus.json',self.corpus)
        with self.assertRaisesRegex(ValueError,'heldout'): self.assemble(True,'reviewer')

    def test_annotation_explicit_case_action_final_zeros_required(self):
        case = self.annotations[('masked',0,0)]['cases'][0]
        for key in ('case_critical_violations','final_critical_violations','action_reviews','reviewed','reviewer','reviewed_full_completion'):
            with self.subTest(key=key):
                old = case.pop(key); self.sync()
                with self.assertRaises(ValueError): self.assemble(True,'reviewer')
                case[key] = old
        case['action_reviews'][0].pop('critical_violations'); self.sync()
        with self.assertRaises(ValueError): self.assemble(True,'reviewer')

    def test_annotation_action_index_and_record_digest_binding(self):
        case = self.annotations[('masked',0,0)]['cases'][0]
        case['action_reviews'][0]['action_index'] = 1; self.sync()
        with self.assertRaisesRegex(ValueError,'action index'): self.assemble(True,'reviewer')
        case['action_reviews'][0]['action_index'] = 0
        case['transcript_record_sha256'] = '0'*64; self.sync()
        with self.assertRaisesRegex(ValueError,'case content'): self.assemble(True,'reviewer')

    def test_aggregate_completion_and_counters_across_replicas(self):
        self.fail(19,'baseline',0,['payment','payment'])
        self.fail(19,'masked',0,['payment','payment'])
        self.fail(0,'masked',42)
        self.sync()
        self.assemble(True,'reviewer')
        masked = a.strict_json((self.out/'masked.json').read_bytes())
        byid = {x['id']:x for x in masked['cases']}
        self.assertFalse(byid['invented-0']['full_completion'])
        self.assertEqual(byid['invented-19']['critical_violations'],['payment','payment'])
        self.assertEqual(masked['metrics']['full_completion_count'],18)

    def test_new_critical_cannot_cancel_across_trials_or_cases(self):
        self.fail(19,'baseline',0,['payment'])
        self.fail(19,'masked',42,['payment'])
        self.sync()
        with self.assertRaisesRegex(ValueError,'replica cancellation'): self.assemble(True,'reviewer')
        self.assertFalse(self.out.exists())

    def test_baseline_floor_and_new_failure_no_net_cancellation(self):
        for i in range(4): self.fail(i,'baseline')
        for i in (4,5): self.fail(i,'masked')
        self.sync()
        with self.assertRaisesRegex(ValueError,'NEW failed'): self.assemble(True,'reviewer')
        self.fail(6,'baseline'); self.sync()
        with self.assertRaisesRegex(ValueError,'baseline'): self.assemble(True,'reviewer')

    def test_historical_raw_failure_cannot_be_rescored_success(self):
        self.fail(0)
        self.annotations[('masked',0,0)]['cases'][0]['reviewed_full_completion'] = True
        self.sync()
        with self.assertRaisesRegex(ValueError,'rescored'): self.assemble(True,'reviewer')

    def test_native_raw_sampling_protocol_mismatch_rejected(self):
        row = self.records[('masked',0,0)][0]
        row['turns'][0]['request']['temperature'] = 0.6
        self.annotations[('masked',0,0)]['cases'][0]['transcript_record_sha256'] = a.sha(a.canonical(row))
        self.sync()
        with self.assertRaisesRegex(ValueError,'sampling recipe'): self.assemble(True,'reviewer')

    def test_raw_hash_tampering_rejected(self):
        path = self.root/self.index['runs'][0]['transcript']['path']
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.assemble(True,'reviewer')

    def test_generic_exporter_check_failure_never_publishes_pass(self):
        self.policy['checks'] = [{'metric':'full_completion_rate','direction':'higher','max_regression':0}]
        self.save('policy.json',self.policy)
        self.fail(0); self.sync()
        with self.assertRaisesRegex(ValueError,'regression'): self.assemble(True,'reviewer')
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.root.glob('__quality_*')))

    def test_duplicate_case_and_unreviewed_case_rejected(self):
        key = ('masked',0,0)
        original = self.records[key][1]
        self.records[key][1] = copy.deepcopy(self.records[key][0]); self.sync()
        with self.assertRaisesRegex(ValueError,'duplicate case'): self.assemble(True,'reviewer')
        self.records[key][1] = original
        self.annotations[key]['cases'].pop(); self.sync()
        with self.assertRaisesRegex(ValueError,'exactly20'): self.assemble(True,'reviewer')

    def test_raw_calibration_or_manifest_configuration_cannot_be_attested_away(self):
        key = ('masked',0,0)
        row = self.records[key][0]
        row['split'] = 'calibration'
        self.annotations[key]['cases'][0]['transcript_record_sha256'] = a.sha(a.canonical(row))
        self.sync()
        with self.assertRaisesRegex(ValueError,'calibration transcript'): self.assemble(True,'reviewer')
        row['split'] = 'heldout'
        self.annotations[key]['cases'][0]['transcript_record_sha256'] = a.sha(a.canonical(row))
        self.manifests[key]['config']['final_mode'] = 'content'; self.sync()
        with self.assertRaisesRegex(ValueError,'runner config'): self.assemble(True,'reviewer')

    def test_private_corpus_payload_rejected_and_inputs_immutable(self):
        before = {p.name:a.sha(p.read_bytes()) for p in self.root.iterdir() if p.is_file()}
        self.assemble(True,'reviewer')
        after = {p.name:a.sha(p.read_bytes()) for p in self.root.iterdir() if p.is_file()}
        self.assertEqual(before,after)
        self.corpus['samples'][0]['reasoning'] = 'PRIVATE_REASONING_SENTINEL'
        self.save('corpus.json',self.corpus)
        with self.assertRaisesRegex(ValueError,'private content'):
            a.prepare(self.root/'policy.json',self.root/'corpus.json',self.root/'mask.json',self.root/'index.json')

    def test_no_overwrite_and_publication_failure_no_approval(self):
        self.out.mkdir(); (self.out/'owned').write_text('keep')
        with self.assertRaises(ValueError): self.assemble(True,'reviewer')
        self.assertEqual((self.out/'owned').read_text(),'keep')
        (self.out/'owned').unlink(); self.out.rmdir()
        original = a.os.link
        def fail_last(src,dst):
            if Path(dst).name=='quality.json': raise OSError('test disk failure')
            original(src,dst)
        with patch.object(a.os,'link',side_effect=fail_last), self.assertRaises(OSError):
            self.assemble(True,'reviewer')
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.root.glob('__quality_*')))


if __name__ == '__main__': unittest.main()
