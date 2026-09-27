"""Invented fixtures ONLY. Never read active coordinator runs, corpus or weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_screen as c


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='__screen_test_',dir=Path(__file__).parent)
        self.root=Path(self.tmp.name); self.out=self.root/'screen.json'
        self.ids=[f'calibration-invented-{i}' for i in range(10)]
        self.sources={k:c.sha(k.encode()) for k in c.SOURCE_KEYS}
        capture={k:c.CONFIG[k] for k in ('protocol','policy_version','final_mode','sampling_profile','seed','budget','max_output','max_turns','thinking')}
        capture['diagnostic_raw']=True
        source={'pilot_runner_sha256':self.sources['runner_sha256'],'coordinator_sha256':self.sources['capture_coordinator_sha256']}
        internal=c.sha(c.canonical({'expected_config':capture,'recorded_source_sha256':source}))
        self.plan={'gpu_screen_ready':True,'quality_approval':False,'policy_version':'v3','protocol_sha256':internal,
            'provenance':{'protocol_sha256':internal,'gpu_screen_ready':True,'recorded_source_sha256':source,
                'expected_config':capture,'full_source_config':dict(capture,model='local-pilot'),
                'family_counts':{f'f{i}':{'cases':5,'full_completions':5} for i in range(10)},'families_without_full_completion':[]},'mask_sha256':{}}
        self.plan['provenance']['full_source_config_sha256']=c.sha(c.canonical(self.plan['provenance']['full_source_config']))
        self.masks={}; self.mask_paths={}
        for role in ('ranked','random'):
            mask={'schema_version':1,'model_sha256':c.gate.MODEL_SHA,'architecture':'qwen35moe','layer_count':40,
                'expert_count':256,'top_k':8,'layers':{str(i):list(range(128)) if role=='ranked' else list(range(0,256,2)) for i in range(40)},
                'method':'mass_gate' if role=='ranked' else 'random_matched_pool','gpu_screen_ready':True,'quality_approval':False,
                'policy_version':'v3','protocol_sha256':internal}
            self.masks[role]=mask; self.mask_paths[role]=self.root/f'{role}-k128.json'
            self.save(self.mask_paths[role],mask)
            self.plan['mask_sha256'][self.mask_paths[role].name]=c.sha(c.canonical(mask))
        self.plan_path=self.root/'plan.json'; self.save(self.plan_path,self.plan)
        self.folders={role:self.root/role for role in ('baseline','ranked','random')}
        self.outer={}; self.pilot={}; self.rows={}
        for n,role in enumerate(self.folders):
            folder=self.folders[role]; (folder/'pilot').mkdir(parents=True)
            template='SYNTHETIC_TEMPLATE'
            selected=None
            if role!='baseline':
                selected={'path':str(self.mask_paths[role]),'file_sha256':c.sha(c.canonical(self.masks[role])),
                    'canonical_sha256':c.sha(c.canonical(self.masks[role])),'keep':128,'method':self.masks[role]['method']}
            env=dict(c.ENV)
            if selected: env['QWEN36_REAP_MASK']=selected['path']
            self.outer[role]={'schema_version':1,'scope':'untraced_quality_comparison','completed':True,
                'eligible_quality_evaluation':False,'split':'calibration','episode_ids':list(self.ids),'runtime_arm':'candidate',
                'trace_enabled':False,'skip_chat_parsing':False,'reasoning_preserve':False,'model_sha256':c.gate.MODEL_SHA,
                'mask_sha256':selected['canonical_sha256'] if selected else None,'mask_file_sha256':selected['file_sha256'] if selected else None,
                'selected_mask':selected,'selection_plan_sha256':c.sha(c.canonical(self.plan)) if selected else None,
                'config':copy.deepcopy(c.CONFIG),'sources':dict(self.sources),'runtime_sha256':dict(c.RUNTIME),'environment':env,
                'dataset_sha256':'d'*64,'chat_template_sha256':c.sha(template.encode()),
                'parser_config_sha256':c.sha(c.canonical({'jinja':True,'skip_chat_parsing':False})),
                'reasoning_config_sha256':c.sha(c.canonical({'reasoning':'on','reasoning_preserve':False,'thinking':'template-default'})),
                'coordinator_sha256':self.sources['comparison_wrapper_sha256'],
                'command':['llama-server.exe','-m',c.gate.DEFAULT_MODEL,'-c','4096','--reasoning','on','--jinja','--no-reasoning-preserve','--offline']}
            self.pilot[role]={'split':'calibration','protocol':'native','episode_ids':list(self.ids),'limit':None,
                'dataset_sha256':'d'*64,'runner_sha256':self.sources['runner_sha256'],'config':copy.deepcopy(c.CONFIG),
                'server_props':{'model_path':c.gate.DEFAULT_MODEL,'chat_template':template,
                    'default_generation_settings':{'n_ctx':4096,'params':{'speculative.types':'none'}}}}
            self.save(folder/'shutdown.json',{'pid':700+n,'stopped':True,'returncode':1})
            self.save(folder/'server.pid.json',{'pid':700+n,'executable':'llama-server.exe'})
            self.rows[role]=[self.record(case) for case in self.ids]
        self.sync()

    def tearDown(self): self.tmp.cleanup()

    def save(self,path,obj): Path(path).write_bytes(c.canonical(obj))

    @staticmethod
    def record(case):
        action={'tool':'sim.observe','arguments':{'devices':[101]}}
        final={'status':'complete','evidence':'account_visible'}
        req=dict(model='local-pilot',stream=False,cache_prompt=True,temperature=0,top_k=1,top_p=1,min_p=0,
            seed=0,max_tokens=1024,tool_choice='auto',parallel_tool_calls=False,
            messages=[{'role':'user','content':'PRIVATE_PROMPT_SENTINEL'}])
        turns=[]
        for i,(kind,value) in enumerate((('action',action),('final',final))):
            name='sim_observe' if kind=='action' else 'final'
            args=action['arguments'] if kind=='action' else final
            turns.append({'index':i,'request':copy.deepcopy(req),'preflight':{'prompt_tokens_preflight':500,
                'server_ctx':4096,'effective_budget':4096,'output_reserved':1024},'response':{
                    'usage':{'prompt_tokens':500,'completion_tokens':20},'choices':[{'finish_reason':'tool_calls',
                    'message':{'content':None,'reasoning_content':'PRIVATE_REASONING_SENTINEL','tool_calls':[
                        {'function':{'name':name,'arguments':json.dumps(args)}}]}}]},'selection':{'kind':kind,'value':value}})
        return {'id':case,'split':'calibration','protocol':'native','policy_version':'v3','final_mode':'tool',
            'full_completion':True,'real_actions_executed':0,'error_class':None,'actions':[action],'final':final,'turns':turns}

    def fail(self,role,i):
        r=self.rows[role][i]; r.update(full_completion=False,error_class='response_format',actions=[],final=None)
        r['turns']=r['turns'][:1]; t=r['turns'][0]; t.pop('selection',None)
        t['response']['choices'][0]['finish_reason']='length'
        t['response']['choices'][0]['message']['tool_calls']=[]
        t['response']['usage']['completion_tokens']=1024

    def sync(self):
        for role,folder in self.folders.items():
            rows=self.rows[role]; success=sum(r['full_completion'] for r in rows)
            self.outer[role]['outcomes']={'cases':len(rows),'successes':success,'full_completion_rate':success/len(rows)}
            self.save(folder/'manifest.json',self.outer[role]); self.save(folder/'pilot/manifest.json',self.pilot[role])
            self.save(folder/'pilot/summary.json',{'cases':len(rows),'split':'calibration','real_actions_executed':0,'full_completion_rate':success/len(rows)})
            (folder/'pilot/transcripts.jsonl').write_bytes(b'\n'.join(c.canonical(r) for r in rows)+b'\n')

    def reviews(self):
        result={}
        for role,rows in self.rows.items():
            proof={'split':'calibration','reviewer':'TEST_ONLY reviewer','unparsed_turns_reviewed':True,
                'transcript_sha256':c.sha((self.folders[role]/'pilot/transcripts.jsonl').read_bytes()),'cases':[]}
            for r in rows:
                proof['cases'].append({'id':r['id'],'reviewed':True,'reviewer':'TEST_ONLY reviewer',
                    'transcript_record_sha256':c.sha(c.canonical(r)),'reviewed_full_completion':r['full_completion'],
                    'case_critical_violations':[],'final_critical_violations':[],
                    'action_reviews':[{'action_index':i,'action_sha256':c.sha(c.canonical(a)),'critical_violations':[]} for i,a in enumerate(r['actions'])]})
            path=self.root/f'{role}-review.json'; self.save(path,proof); result[role]=path
        return result

    def run_compare(self,reviews=None,**kwargs):
        return c.compare(self.folders['baseline'],self.folders['ranked'],self.folders['random'],self.plan_path,
            self.mask_paths['ranked'],self.mask_paths['random'],self.out,reviews=reviews,**kwargs)

    def test_default_pending_no_auto_approval_and_no_private_content(self):
        result=self.run_compare()
        self.assertEqual(result['decision'],'pending_explicit_critical_review')
        self.assertEqual(result['arms']['ranked']['successes'],10)
        self.assertIsNone(result['arms']['ranked']['paired_screen_acceptable'])
        for key in ('quality_approval','export_approved','heldout_approved','next_k_approved'): self.assertIs(result[key],False)
        raw=self.out.read_bytes()
        self.assertNotIn(b'PRIVATE_PROMPT_SENTINEL',raw); self.assertNotIn(b'PRIVATE_REASONING_SENTINEL',raw)

    def test_reviewed_screen_is_not_next_k_or_quality_approval(self):
        self.fail('ranked',0); self.sync()
        result=self.run_compare(self.reviews())
        self.assertEqual(result['decision'],'ranked_calibration_screen_acceptable')
        self.assertEqual(result['arms']['ranked']['newly_failed_ids_vs_baseline'],[self.ids[0]])
        self.assertFalse(result['quality_approval']); self.assertFalse(result['next_k_approved'])

    def test_new_failed_ids_not_net_canceled(self):
        self.fail('baseline',0); self.fail('ranked',1); self.fail('ranked',2); self.sync()
        result=self.run_compare(self.reviews())
        self.assertEqual(result['decision'],'ranked_calibration_screen_rejected')
        self.assertEqual(result['arms']['ranked']['newly_failed_ids_vs_baseline'],self.ids[1:3])
        self.assertEqual(result['arms']['ranked']['improved_ids_vs_baseline'],self.ids[:1])

    def test_active_run_rejected_before_raw_file_read(self):
        self.outer['baseline']['completed']=False; self.sync()
        (self.folders['baseline']/'pilot/transcripts.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'active/incomplete'): self.run_compare()

    def test_shutdown_pid_and_trace_calibration_flags(self):
        folder=self.folders['baseline']; self.save(folder/'shutdown.json',{'pid':999,'stopped':True,'returncode':0})
        with self.assertRaisesRegex(ValueError,'PID'): self.run_compare()
        self.save(folder/'shutdown.json',{'pid':700,'stopped':True,'returncode':0})
        for key,value in [('trace_enabled',True),('eligible_quality_evaluation',True),('split','heldout'),('skip_chat_parsing',True)]:
            old=self.outer['baseline'][key]; self.outer['baseline'][key]=value; self.sync()
            with self.assertRaises(ValueError): self.run_compare()
            self.outer['baseline'][key]=old

    def test_identical_ordered_ids_and_terminal_model_outcomes(self):
        self.rows['ranked'].reverse(); self.sync()
        with self.assertRaisesRegex(ValueError,'ID order'): self.run_compare()
        self.rows['ranked'].reverse(); self.rows['ranked'][0]['error_class']='transport'; self.sync()
        with self.assertRaisesRegex(ValueError,'infrastructure'): self.run_compare()

    def test_partial_eof_and_summary_outcomes_mismatch(self):
        path=self.folders['random']/'pilot/transcripts.jsonl'; path.write_bytes(path.read_bytes().rstrip(b'\n'))
        with self.assertRaisesRegex(ValueError,'EOF'): self.run_compare()
        self.sync(); self.outer['random']['outcomes']['successes']=9
        self.save(self.folders['random']/'manifest.json',self.outer['random'])
        with self.assertRaisesRegex(ValueError,'outer outcomes'): self.run_compare()

    def test_sources_runtime_model_context_and_raw_off_enforced(self):
        for key,value in [('model_sha256','0'*64),('runtime_sha256',dict(c.RUNTIME,**{'llama.dll':'0'*64})),
                          ('config',dict(c.CONFIG,budget=6144))]:
            old=self.outer['ranked'][key]; self.outer['ranked'][key]=value; self.sync()
            with self.assertRaises(ValueError): self.run_compare()
            self.outer['ranked'][key]=old
        self.pilot['ranked']['config']['diagnostic_raw']=True; self.sync()
        with self.assertRaises(ValueError): self.run_compare()

    def test_source_template_and_numerical_environment_drift(self):
        self.outer['ranked']['sources']['validator_sha256']='1'*64; self.sync()
        with self.assertRaisesRegex(ValueError,'common runtime'): self.run_compare()
        self.outer['ranked']['sources']=dict(self.sources)
        self.outer['ranked']['environment']['LLAMA_MOE_ELASTIC']='1'; self.sync()
        with self.assertRaisesRegex(ValueError,'numerical environment'): self.run_compare()

    def test_exact_plan_mask_hash_and_readiness(self):
        self.plan['gpu_screen_ready']=False; self.save(self.plan_path,self.plan)
        with self.assertRaisesRegex(ValueError,'GPU-screen-ready'): self.run_compare()
        self.plan['gpu_screen_ready']=True; self.save(self.plan_path,self.plan)
        self.mask_paths['ranked'].write_bytes(c.canonical(self.masks['ranked'])+b'\n')
        with self.assertRaisesRegex(ValueError,'exact plan bytes'): self.run_compare()

    def test_same_k_required_even_with_honestly_rehashed_masks(self):
        mask=self.masks['random']; mask['layers']={str(i):list(range(64)) for i in range(40)}
        digest=c.sha(c.canonical(mask)); self.save(self.mask_paths['random'],mask)
        self.plan['mask_sha256'][self.mask_paths['random'].name]=digest; self.save(self.plan_path,self.plan)
        phash=c.sha(c.canonical(self.plan))
        for role in ('ranked','random'): self.outer[role]['selection_plan_sha256']=phash
        m=self.outer['random']; m['mask_sha256']=m['mask_file_sha256']=digest
        m['selected_mask'].update(file_sha256=digest,canonical_sha256=digest,keep=64); self.sync()
        with self.assertRaisesRegex(ValueError,'same kept K'): self.run_compare()

    def test_baseline_mask_must_be_null(self):
        self.outer['baseline']['mask_sha256']='2'*64; self.sync()
        with self.assertRaisesRegex(ValueError,'unmasked'): self.run_compare()

    def test_raw_request_and_selection_correspondence(self):
        self.rows['ranked'][0]['turns'][0]['request']['seed']=1; self.sync()
        with self.assertRaisesRegex(ValueError,'request config'): self.run_compare()
        self.rows['ranked'][0]['turns'][0]['request']['seed']=0
        self.rows['ranked'][0]['turns'][0]['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{}'; self.sync()
        with self.assertRaisesRegex(ValueError,'raw action'): self.run_compare()

    def test_review_required_for_random_too(self):
        paths=self.reviews(); paths.pop('random')
        self.assertEqual(self.run_compare(paths)['decision'],'pending_explicit_critical_review')

    def test_review_hash_exhaustive_action_lists_and_no_rescore(self):
        paths=self.reviews(); proof=json.loads(paths['ranked'].read_bytes())
        proof['cases'][0]['action_reviews'][0].pop('critical_violations'); self.save(paths['ranked'],proof)
        with self.assertRaisesRegex(ValueError,'explicit known'): self.run_compare(paths)
        paths=self.reviews(); proof=json.loads(paths['ranked'].read_bytes())
        proof['cases'][0]['reviewed_full_completion']=False; self.save(paths['ranked'],proof)
        with self.assertRaisesRegex(ValueError,'no rescoring'): self.run_compare(paths)
        paths=self.reviews(); proof=json.loads(paths['ranked'].read_bytes()); proof['transcript_sha256']='0'*64; self.save(paths['ranked'],proof)
        with self.assertRaisesRegex(ValueError,'transcript binding'): self.run_compare(paths)

    def test_critical_case_swap_and_multiplicity_no_net_cancel(self):
        paths=self.reviews(); base=json.loads(paths['baseline'].read_bytes()); ranked=json.loads(paths['ranked'].read_bytes())
        base['cases'][0]['case_critical_violations']=['payment','payment']
        ranked['cases'][1]['case_critical_violations']=['payment']
        self.save(paths['baseline'],base); self.save(paths['ranked'],ranked)
        result=self.run_compare(paths)
        self.assertEqual(result['decision'],'ranked_calibration_screen_rejected')
        self.assertEqual(result['arms']['ranked']['new_critical_occurrences_vs_baseline'],[{'id':self.ids[1],'category':'payment','occurrences':1}])
        self.out.unlink()
        ranked['cases'][1]['case_critical_violations']=[]
        ranked['cases'][0]['case_critical_violations']=['payment']*3
        self.save(paths['ranked'],ranked)
        result=self.run_compare(paths)
        self.assertEqual(result['arms']['ranked']['new_critical_occurrences_vs_baseline'],[{'id':self.ids[0],'category':'payment','occurrences':1}])

    def test_random_control_metrics_not_merged_with_ranked(self):
        self.fail('random',0); self.fail('random',1); self.sync()
        result=self.run_compare(self.reviews())
        self.assertEqual(result['decision'],'ranked_calibration_screen_acceptable')
        self.assertFalse(result['arms']['random']['paired_screen_acceptable'])
        self.assertEqual(result['arms']['ranked']['successes'],10)

    def test_input_mutation_rejected_and_no_overwrite(self):
        self.out.write_bytes(b'owner')
        with self.assertRaises(ValueError): self.run_compare()
        self.assertEqual(self.out.read_bytes(),b'owner'); self.out.unlink()
        verify=c.Snapshot.verify
        def mutate(snapshot):
            path=self.folders['ranked']/'pilot/summary.json'; path.write_bytes(path.read_bytes()+b' ')
            verify(snapshot)
        with patch.object(c.Snapshot,'verify',mutate), self.assertRaisesRegex(ValueError,'changed'):
            self.run_compare()
        self.assertFalse(self.out.exists()); self.assertFalse(list(self.root.glob('__screen_*.partial')))

    def test_planning_digest_preserves_planner_unicode_encoding(self):
        value={'note':'caffè'}
        expected=c.sha(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8'))
        self.assertEqual(c.planning_digest(value),expected)
        self.assertNotEqual(c.planning_digest(value),c.sha(c.canonical(value)))

    def test_explicit_fifty_option_not_inferred(self):
        with self.assertRaises(ValueError): self.run_compare(expected_cases=50)
        for role in self.folders:
            self.ids=[f'calibration-invented-{i}' for i in range(50)]
            self.rows[role]=[self.record(case) for case in self.ids]
            self.outer[role]['episode_ids']=list(self.ids); self.pilot[role]['episode_ids']=list(self.ids)
        self.sync()
        with self.assertRaises(ValueError): self.run_compare()
        result=self.run_compare(expected_cases=50)
        self.assertEqual(result['arms']['ranked']['cases'],50)
        self.assertEqual(result['decision'],'pending_explicit_critical_review')


if __name__=='__main__': unittest.main()
