"""Invented fixtures only. Never opens actual heldout/model, launches or contacts anything."""
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_quality_server as q


class OwnedServer:
    pid = 987654
    returncode = None
    terminated = False
    killed = False
    def poll(self):
        return self.returncode
    def terminate(self):
        self.terminated = True
        self.returncode = 0
    def wait(self, timeout):
        return self.returncode
    def kill(self):
        self.killed = True
        self.returncode = 1


class QualityServerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sources = {key: str(i + 1) * 64 for i, key in enumerate(q.SOURCE_FILES)}
        self.source_paths = {path.resolve(): self.sources[key] for key, path in q.SOURCE_FILES.items()}
        self.dataset_sha = 'd' * 64
        real_sha = q.sha
        def fake_sha(path):
            path = Path(path).resolve()
            if path == q.MODEL.resolve():
                return q.MODEL_SHA
            if path.parent == q.BINARY.resolve():
                return q.RUNTIME[path.name]
            if path in self.source_paths:
                return self.source_paths[path]
            if path.parent == (q.HERE / 'pilot').resolve() and path.name in ('calibration.jsonl', 'heldout.jsonl'):
                return self.dataset_sha
            return real_sha(path)
        self.fake_sha = fake_sha
        def selector(split, limit, ids):
            assert limit is None
            names = [f'{split}-invented-{i}' for i in range(20 if split == 'heldout' else 50)]
            if ids is not None:
                chosen = ids.split(',')
                if split != 'calibration' or not chosen or len(set(chosen)) != len(chosen) or any(x not in names for x in chosen):
                    raise ValueError('Invalid invented subset')
                names = chosen
            return [{'id': name} for name in names]
        self.selector = selector
        for target, kwargs in [
            ('run_quality_server.sha', {'side_effect': fake_sha}),
            ('run_quality_server.runtime_files', {'return_value': [q.BINARY / name for name in q.RUNTIME]}),
            ('run_quality_server.select_episodes', {'side_effect': selector}),
            ('run_quality_server.subprocess.Popen', {'side_effect': AssertionError('No process launch')}),
            ('run_quality_server.subprocess.run', {'side_effect': AssertionError('No process inspection/execution')}),
            ('run_quality_server.socket.socket', {'side_effect': AssertionError('No socket')}),
            ('run_quality_server.urllib.request.build_opener', {'side_effect': AssertionError('No network')}),
        ]:
            context = patch(target, **kwargs)
            context.start()
            self.addCleanup(context.stop)
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            absolute = path.resolve()
            if absolute == q.MODEL.resolve() or (absolute.parent == (q.HERE / 'pilot').resolve() and absolute.name.endswith('.jsonl')):
                raise AssertionError('Actual corpus/model read forbidden')
            return original_open(path, *args, **kwargs)
        guard = patch.object(Path, 'open', guarded_open)
        guard.start()
        self.addCleanup(guard.stop)

    def args(self, *extra):
        return q.parser().parse_args(['--allow-inference', '--out', str(self.root / 'out'), *map(str, extra)])

    def put(self, path, value):
        path.write_bytes(q.gate.canonical(value))
        return path

    def bundle(self, method='mass_gate'):
        config = q.frozen_protocol('v3')
        recorded = {'pilot_runner_sha256': self.sources['runner_sha256'],
                    'coordinator_sha256': self.sources['capture_coordinator_sha256']}
        internal = q.object_digest({'expected_config': config, 'recorded_source_sha256': recorded})
        self.proof = dict(run=str(self.root / 'invented-source'), policy_version='v3', expected_config=config,
            full_source_config=dict(config, model='local-pilot', template_supports_thinking=False),
            recorded_source_sha256=recorded, protocol_sha256=internal,
            family_counts={f'invented-family-{i}': {'cases':5, 'full_completions':4} for i in range(10)},
            families_without_full_completion=[], gpu_screen_ready=True, cases=50, successes=40)
        self.proof['full_source_config_sha256'] = q.object_digest(self.proof['full_source_config'])
        self.mask = dict(schema_version=1, architecture='qwen35moe', layer_count=40, expert_count=256,
            top_k=8, model_sha256=q.MODEL_SHA, layers={str(i):list(range(128)) for i in range(40)},
            policy_version='v3', protocol_sha256=internal, gpu_screen_ready=True, quality_approval=False, method=method)
        self.mask_path = self.put(self.root / 'mask.json', self.mask)
        self.plan = dict(gpu_screen_ready=True, quality_approval=False, policy_version='v3', protocol_sha256=internal,
            provenance=deepcopy(self.proof), mask_sha256={self.mask_path.name: q.sha(self.mask_path)})
        self.plan_path = self.put(self.root / 'plan.json', self.plan)
        check = patch('run_quality_server.validate_source_run', side_effect=lambda *a, **kw: deepcopy(self.proof))
        self.source_validator = check.start()
        self.addCleanup(check.stop)
        return self.mask_path, self.plan_path

    def recipe_bundle(self, method='mean_selected_gate', random=False):
        self.bundle(method='random_matched_pool' if random else method)
        recipe=q.expected_selection_recipe(method)
        digest=q.object_digest(recipe)
        self.mask.update(ranking_method=method,selection_recipe=recipe,selection_recipe_sha256=digest)
        self.put(self.mask_path,self.mask)
        self.plan.update(ranking_method=method,selection_recipe=deepcopy(recipe),selection_recipe_sha256=digest)
        self.plan['mask_sha256'][self.mask_path.name]=q.sha(self.mask_path)
        self.plan['candidates']=[{'file':self.mask_path.name,'keep':128,'method':self.mask['method'],
                                 'ranking_method':method,'selection_recipe_sha256':digest}]
        self.put(self.plan_path,self.plan)
        return self.args('--mask',self.mask_path,'--selection-plan',self.plan_path,'--ranking-method',method)

    def heldout_args(self, masked=False):
        self.bundle()
        selected = q.mask_identity(self.mask_path)
        episodes = self.selector('heldout', None, None)
        corpus = dict(schema_version=1, split='heldout', calibration_disjoint=True, dataset_sha256=self.dataset_sha,
            samples=[{'id':e['id'], 'sha256':q.gate.sha(q.gate.canonical(e))} for e in episodes])
        self.corpus_path = self.put(self.root / 'corpus.json', corpus)
        self.protocol = dict(q.PROTOCOL_CORE, trials=[0], model_sha256=q.MODEL_SHA, runtime_sha256=q.RUNTIME,
            **self.sources, chat_template_sha256=hashlib.sha256(b'invented template').hexdigest(),
            selected_mask_sha256=selected['canonical_sha256'], selected_mask_file_sha256=selected['file_sha256'],
            selection_plan_sha256=q.sha(self.plan_path))
        self.policy = dict(frozen_before_evaluation=True, evaluation_protocol=self.protocol)
        self.policy_path = self.put(self.root / 'policy.json', self.policy)
        extras = ['--split','heldout','--allow-heldout','--selected-mask', self.mask_path,
            '--selection-plan',self.plan_path,'--policy',self.policy_path,'--policy-sha256',q.sha(self.policy_path),
            '--protocol-sha256',q.gate.sha(q.gate.canonical(self.protocol)), '--corpus-manifest',self.corpus_path,
            '--corpus-manifest-sha256',q.sha(self.corpus_path)]
        if masked:
            extras += ['--mask',self.mask_path]
        return self.args(*extras)

    def props(self):
        return {'default_generation_settings':{'n_ctx':4096,'params':{'speculative.types':'none'}},
                'chat_template':'invented template','model_path':str(q.MODEL)}

    def results(self, spec, failures=0):
        out = spec['args'].out / 'pilot'
        out.mkdir()
        self.put(out / 'manifest.json', dict(split=spec['args'].split, episode_ids=spec['ids'],
            dataset_sha256=spec['dataset_sha256'], limit=None, config=q.FROZEN_CONFIG,
            runner_sha256=spec['sources']['runner_sha256'], server_props=self.props()))
        rows = []
        for i, case in enumerate(spec['ids']):
            success = i >= failures
            final = {'status':'invented-complete'}
            req = dict(model='local-pilot',stream=False,cache_prompt=True,temperature=0,top_k=1,top_p=1,min_p=0,
                seed=0,max_tokens=1024,tool_choice='auto',parallel_tool_calls=False)
            turn = dict(request=req, preflight={'prompt_tokens_preflight':100,'server_ctx':4096,'effective_budget':4096,'output_reserved':1024},
                response={'usage':{'prompt_tokens':100,'completion_tokens':20},'choices':[{'finish_reason':'tool_calls'}]},
                selection={'kind':'final','value':final})
            rows.append(dict(id=case,split=spec['args'].split,protocol='native',policy_version='v3',final_mode='tool',
                full_completion=success,real_actions_executed=0,turns=[turn],actions=[],final=final,
                error_class=None if success else 'final_validation'))
        (out / 'transcripts.jsonl').write_bytes(b'\n'.join(q.gate.canonical(r) for r in rows) + b'\n')
        self.put(out / 'summary.json', dict(cases=len(rows),split=spec['args'].split,real_actions_executed=0,
            full_completion_rate=(len(rows)-failures)/len(rows)))
        return rows

    def test_optins_and_heldout_subset_reject_before_any_selector(self):
        with patch('run_quality_server.select_episodes', side_effect=AssertionError('Must not load')):
            for args in [q.parser().parse_args(['--out',str(self.root/'out')]), self.args('--split','heldout'),
                         self.args('--split','heldout','--allow-heldout','--episode-ids','anything'),
                         self.args('--split','heldout','--allow-heldout'), self.args('--allow-heldout')]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    q.prepare(args)

    def test_ranking_flag_unmasked_control_and_frozen_inference_contract(self):
        baseline=q.prepare(self.args())
        mean=q.prepare(self.args('--ranking-method','mean_selected_gate'))
        self.assertEqual(baseline['args'].ranking_method,'mass_gate')
        self.assertEqual(mean['args'].ranking_method,'mean_selected_gate')
        self.assertIsNone(mean['mask']);self.assertIsNone(mean['selected_mask']);self.assertIsNone(mean['plan'])
        self.assertEqual(baseline['recorded_env'],mean['recorded_env'])
        self.assertEqual(q.gate.sha(q.gate.canonical(q.command())),
                         '987ba8a7aafdf6eaa7ebf3ae64ab8076a07eec6bc8e1aaa2828a0beee0e74e5b')
        self.assertEqual(q.gate.sha(q.gate.canonical(q.FROZEN_CONFIG)),
                         '03f37bef233ec5ae75ea72c9eecde52657cf155248daa223d2cec223cf51e9e4')
        with patch('run_quality_server.select_episodes',side_effect=AssertionError('No heldout load')):
            with self.assertRaises(ValueError):
                q.prepare(self.args('--split','heldout','--allow-heldout','--ranking-method','mean_selected_gate'))

    def test_new_recipe_ranked_and_random_require_explicit_matching_method(self):
        for random in (False,True):
            args=self.recipe_bundle(random=random)
            spec=q.prepare(args)
            self.assertEqual(spec['mask']['method'],'random_matched_pool' if random else 'mean_selected_gate')
            self.assertEqual(set(spec['mask']),{'path','file_sha256','canonical_sha256','keep','method'})
            self.assertEqual(set(spec['plan']),{'path','file_sha256','value','internal_protocol_sha256'})
            args.ranking_method='mass_gate'
            with self.assertRaises(ValueError):q.prepare(args)
        args=self.recipe_bundle(method='mass_gate')
        self.assertEqual(q.prepare(args)['mask']['method'],'mass_gate')
        self.bundle(method='random_matched_pool')
        with self.assertRaises(ValueError):
            q.prepare(self.args('--mask',self.mask_path,'--selection-plan',self.plan_path,'--ranking-method','mean_selected_gate'))

    def test_partial_recipe_and_candidate_binding_cannot_fallback_to_legacy(self):
        self.recipe_bundle()
        selected=q.mask_identity(self.mask_path)
        mutations=[lambda p,m:p.pop('selection_recipe_sha256'),lambda p,m:m.pop('selection_recipe'),
            lambda p,m:p['candidates'][0].pop('ranking_method'),
            lambda p,m:p['candidates'][0].update(selection_recipe_sha256='0'*64),
            lambda p,m:p['candidates'][0].update(method='mass_gate'),
            lambda p,m:p['candidates'].append(deepcopy(p['candidates'][0])),
            lambda p,m:p['candidates'][0].update(keep=128.0),
            lambda p,m:m.update(selection_recipe_sha256='0'*64)]
        for mutate in mutations:
            plan,mask=deepcopy(self.plan),deepcopy(self.mask);mutate(plan,mask)
            with self.assertRaises(ValueError):q.verify_ranking_binding(plan,mask,selected,'mean_selected_gate')
        self.bundle()
        self.plan['ranking_method']='mass_gate'
        with self.assertRaises(ValueError):
            q.verify_ranking_binding(self.plan,self.mask,q.mask_identity(self.mask_path),'mass_gate')

    def test_recipe_exact_types_formula_and_canonical_digest(self):
        self.recipe_bundle()
        self.assertEqual(q.object_digest(q.expected_selection_recipe('mean_selected_gate')),
                         '7e404ac0d429abaac9a6dabe1df75348ea7089edc972f1597ddea19de25d9132')
        selected=q.mask_identity(self.mask_path)
        for key,value in [('includes_expert_output_norm',True),('includes_expert_output_norm',0),
                          ('schema_version',True),('unobserved_score',0.0),('random_seed',20260713.0),
                          ('score_formula','sum_selected_gate / all_tokens'),('pools',[128,96,64,8])]:
            plan,mask=deepcopy(self.plan),deepcopy(self.mask)
            recipe=deepcopy(plan['selection_recipe']);recipe[key]=value
            digest=q.object_digest(recipe)
            for obj in (plan,mask):obj.update(selection_recipe=deepcopy(recipe),selection_recipe_sha256=digest)
            plan['candidates'][0]['selection_recipe_sha256']=digest
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                q.verify_ranking_binding(plan,mask,selected,'mean_selected_gate')
        pretty=q.json.dumps(self.mask,indent=2,sort_keys=True).encode('utf-8')
        self.mask_path.write_bytes(pretty)
        reformatted=q.mask_identity(self.mask_path)
        self.assertEqual(selected['canonical_sha256'],reformatted['canonical_sha256'])
        self.assertNotEqual(selected['file_sha256'],reformatted['file_sha256'])
        with self.assertRaises(ValueError):q.ready_plan(self.plan_path,reformatted,'mean_selected_gate')

    def test_unmasked_mean_manifest_records_reference_not_fake_active_mask(self):
        spec=q.prepare(self.args('--ranking-method','mean_selected_gate','--episode-ids','calibration-invented-0'))
        server=OwnedServer()
        with patch('run_quality_server.subprocess.Popen',return_value=server),patch('run_quality_server.wait_server',return_value=self.props()), \
             patch('run_quality_server.subprocess.run',side_effect=lambda *a,**kw:self.results(spec)):
            manifest=q.execute(spec)
        descriptor,_=q.read_json(spec['args'].out/'run-descriptor.json')
        self.assertEqual(manifest['ranking_method'],'mean_selected_gate')
        self.assertEqual(descriptor['ranking_method'],'mean_selected_gate')
        self.assertIsNone(manifest['mask_sha256']);self.assertIsNone(manifest['selected_mask'])
        self.assertIsNone(manifest['selection_plan_sha256'])
        self.assertFalse(manifest['eligible_quality_evaluation'])

    def test_calibration_default_full50_and_ordered_subset_only(self):
        self.assertEqual(len(q.prepare(self.args())['ids']), 50)
        ids = 'calibration-invented-4,calibration-invented-0'
        self.assertEqual(q.prepare(self.args('--episode-ids',ids))['ids'], ids.split(','))
        with self.assertRaises(ValueError):
            q.prepare(self.args('--episode-ids','calibration-invented-0,calibration-invented-0'))
        for option in ('--limit','--runtime','--context','--diagnostic-raw','--trace','--max-output'):
            with self.subTest(option=option), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                q.parser().parse_args(['--out',str(self.root/'out'),option,'1'])

    def test_mask_and_model_production_gates(self):
        path, plan = self.bundle()
        self.assertEqual(q.prepare(self.args('--mask',path,'--selection-plan',plan))['mask']['keep'], 128)
        for field, value in [('top_k',7),('model_sha256','0'*64),('expert_count',128),('layer_count',39)]:
            bad = deepcopy(self.mask); bad[field] = value; self.put(path,bad)
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.mask_identity(path)
        bad = deepcopy(self.mask); bad['layers']['0'] = list(range(64)); self.put(path,bad)
        with self.assertRaises(ValueError): q.mask_identity(path)
        path.write_text('{"schema_version":1,"schema_version":1}',encoding='utf-8')
        with self.assertRaises(ValueError): q.mask_identity(path)
        path.write_text('{"x":1e999}',encoding='utf-8')
        with self.assertRaises(ValueError): q.read_json(path)
        with patch('run_quality_server.sha', side_effect=lambda p: '0'*64 if Path(p)==q.MODEL else self.fake_sha(p)):
            with self.assertRaises(ValueError): q.prepare(self.args())
        with patch('run_quality_server.sha', side_effect=lambda p: '0'*64 if Path(p).name=='llama.dll' else self.fake_sha(p)):
            with self.assertRaises(ValueError): q.prepare(self.args())

    def test_ready_plan_exact_file_v3_source_family_and_internal_hash(self):
        path, plan = self.bundle()
        args = self.args('--mask',path,'--selection-plan',plan)
        q.prepare(args)
        self.source_validator.assert_called_with(Path(self.proof['run']),q.HERE/'pilot/calibration.jsonl',policy_version='v3')
        mutations = [lambda p:p.update(gpu_screen_ready=False), lambda p:p.update(policy_version='v2'),
            lambda p:p['mask_sha256'].update({path.name:'0'*64}),
            lambda p:p['provenance']['expected_config'].update(policy_version='v2'),
            lambda p:p['provenance'].update(full_source_config_sha256='0'*64),
            lambda p:p['provenance'].update(protocol_sha256='0'*64),
            lambda p:p['provenance']['family_counts']['invented-family-0'].update(full_completions=0),
            lambda p:p['provenance']['recorded_source_sha256'].update(coordinator_sha256='0'*64)]
        for mutate in mutations:
            bad=deepcopy(self.plan); mutate(bad); self.put(plan,bad)
            with self.assertRaises(ValueError): q.prepare(args)
        self.put(plan,self.plan)
        with patch('run_quality_server.validate_source_run', return_value={}):
            with self.assertRaises(ValueError): q.prepare(args)
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaises(ValueError): q.prepare(args)

    def test_environment_trace_always_absent_and_mask_only_when_provided(self):
        self.bundle()
        with patch.dict(os.environ, {'QWEN36_REAP_TRACE':'secret','QWEN36_REAP_MASK':'old','LLAMA_SPEC':'on','GGML_UNKNOWN':'1'}):
            baseline=q.prepare(self.args()); masked=q.prepare(self.args('--mask',self.mask_path,'--selection-plan',self.plan_path))
        for spec in (baseline,masked):
            self.assertNotIn('QWEN36_REAP_TRACE',spec['env'])
            self.assertNotIn('QWEN36_REAP_TRACE',spec['recorded_env'])
            self.assertNotIn('LLAMA_SPEC',spec['env'])
        self.assertNotIn('QWEN36_REAP_MASK',baseline['env'])
        self.assertEqual(masked['env']['QWEN36_REAP_MASK'],str(self.mask_path.resolve()))
        cmd=q.command()
        self.assertEqual(cmd[cmd.index('-m')+1],str(q.MODEL))
        self.assertEqual(cmd[0],str(q.BINARY/'llama-server.exe'))
        for flag,value in [('-c','4096'),('-np','1'),('--moe-expert-cache','32'),('--port','8116')]:
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertNotIn('--skip-chat-parsing',cmd)
        self.assertIn('--no-reasoning-preserve',cmd)

    def test_heldout_frozen_baseline_and_masked_bind_same_candidate(self):
        args=self.heldout_args()
        baseline=q.prepare(args)
        args.mask=self.mask_path
        masked=q.prepare(args)
        self.assertIsNone(baseline['mask'])
        self.assertEqual(baseline['selected_mask'],masked['selected_mask'])
        self.assertEqual(baseline['policy']['protocol_sha256'],masked['policy']['protocol_sha256'])
        self.assertEqual(len(baseline['ids']),20)
        self.assertNotEqual(baseline['policy']['protocol_sha256'],baseline['plan']['internal_protocol_sha256'])

    def test_bad_policy_rejected_before_any_heldout_load(self):
        args=self.heldout_args()
        with patch('run_quality_server.select_episodes',side_effect=AssertionError('Must not load heldout')):
            args.policy_sha256='0'*64
            with self.assertRaises(ValueError): q.prepare(args)
            args.policy_sha256=q.sha(self.policy_path)
            for key, value in [('context',6144),('runner_sha256','0'*64),('capture_coordinator_sha256','0'*64),
                               ('comparison_wrapper_sha256','0'*64),('selected_mask_sha256','0'*64),
                               ('selection_plan_sha256','0'*64),('reasoning_preserve',True)]:
                protocol=deepcopy(self.protocol);protocol[key]=value
                self.put(self.policy_path,dict(self.policy,evaluation_protocol=protocol))
                args.policy_sha256=q.sha(self.policy_path);args.protocol_sha256=q.gate.sha(q.gate.canonical(protocol))
                with self.subTest(key=key),self.assertRaises(ValueError): q.prepare(args)

    def test_live_props_template_context_model_checks_before_inference(self):
        spec=q.prepare(self.heldout_args())
        q.verify_server(self.props(),spec)
        for key,value in [('chat_template','different'),('model_path','other.gguf')]:
            props=self.props();props[key]=value
            with self.assertRaises(ValueError): q.verify_server(props,spec)
        props=self.props();props['default_generation_settings']['n_ctx']=6144
        with self.assertRaises(ValueError): q.verify_server(props,spec)

    def test_exclusive_refuses_any_llama_and_occupied_port(self):
        with patch('run_quality_server.subprocess.run',return_value=SimpleNamespace(stdout='"llama-cli.exe","123"\n')):
            with self.assertRaises(ValueError): q.ensure_exclusive()
        with patch('run_quality_server.subprocess.run',return_value=SimpleNamespace(stdout='"python.exe","123"\n')):
            with patch('run_quality_server.socket.socket') as socket:
                socket.return_value.__enter__.return_value.connect_ex.return_value=0
                with self.assertRaises(ValueError): q.ensure_exclusive()
                socket.return_value.__enter__.return_value.connect_ex.return_value=1
                q.ensure_exclusive()

    def test_complete_failures_count_but_partial_or_drift_do_not(self):
        spec=q.prepare(self.args('--episode-ids','calibration-invented-0,calibration-invented-1'))
        spec['args'].out.mkdir();self.results(spec,failures=1)
        self.assertEqual(q.verify_completion(spec['args'].out,spec)['successes'],1)
        summary=spec['args'].out/'pilot/summary.json'
        data,_=q.read_json(summary);data['cases']=1;self.put(summary,data)
        with self.assertRaises(ValueError): q.verify_completion(spec['args'].out,spec)
        data['cases']=2;self.put(summary,data)
        records=spec['args'].out/'pilot/transcripts.jsonl'
        raw=records.read_bytes();records.write_bytes(raw.rstrip(b'\n'))
        with self.assertRaises(ValueError): q.verify_completion(spec['args'].out,spec)
        rows=[q.gate.decode_json(l) for l in raw.splitlines()]
        for mutate in [lambda r:r.update(id='wrong'),lambda r:r.update(error_class='transport'),
                       lambda r:r['turns'][0]['request'].update(verbose=True),
                       lambda r:r['turns'][0]['preflight'].update(prompt_tokens_preflight=4000)]:
            bad=deepcopy(rows);mutate(bad[0]);records.write_bytes(b'\n'.join(q.gate.canonical(r) for r in bad)+b'\n')
            with self.assertRaises(ValueError): q.verify_completion(spec['args'].out,spec)

    def test_nonpositive_prompt_counts_rejected_even_when_usage_matches(self):
        spec=q.prepare(self.args('--episode-ids','calibration-invented-0'))
        spec['args'].out.mkdir()
        rows=self.results(spec)
        records=spec['args'].out/'pilot/transcripts.jsonl'
        for n in (-1, 0):
            rows[0]['turns'][0]['preflight']['prompt_tokens_preflight']=n
            rows[0]['turns'][0]['response']['usage']['prompt_tokens']=n
            records.write_bytes(b'\n'.join(q.gate.canonical(r) for r in rows)+b'\n')
            with self.assertRaisesRegex(ValueError, 'prompt/context'):
                q.verify_completion(spec['args'].out,spec)

    def test_owned_lifecycle_only_and_calibration_always_ineligible(self):
        spec=q.prepare(self.args('--episode-ids','calibration-invented-0'))
        server=OwnedServer()
        with patch('run_quality_server.subprocess.Popen',return_value=server),patch('run_quality_server.wait_server',return_value=self.props()), \
             patch('run_quality_server.subprocess.run',side_effect=lambda *a,**kw:self.results(spec,failures=1)):
            report=q.execute(spec)
        self.assertTrue(server.terminated)
        self.assertTrue(report['completed'])
        self.assertFalse(report['eligible_quality_evaluation'])
        descriptor,_=q.read_json(spec['args'].out/'run-descriptor.json')
        self.assertFalse(descriptor['index_ready'])
        self.assertNotIn('annotations',descriptor)
        self.assertEqual(Path(descriptor['manifest']['path']),spec['args'].out/'pilot/manifest.json')
        self.assertTrue(Path(descriptor['shutdown']['path']).is_file())

    def test_invented_full20_heldout_failures_are_completed_not_approved(self):
        spec=q.prepare(self.heldout_args())
        server=OwnedServer()
        with patch('run_quality_server.subprocess.Popen',return_value=server),patch('run_quality_server.wait_server',return_value=self.props()), \
             patch('run_quality_server.subprocess.run',side_effect=lambda *a,**kw:self.results(spec,failures=2)):
            report=q.execute(spec)
        self.assertTrue(report['completed']);self.assertTrue(report['eligible_quality_evaluation'])
        self.assertEqual(report['outcomes']['cases'],20);self.assertEqual(report['outcomes']['successes'],18)
        descriptor,_=q.read_json(spec['args'].out/'run-descriptor.json')
        self.assertEqual(descriptor['role'],'baseline');self.assertIsNone(descriptor['mask_sha256'])
        self.assertIsNotNone(descriptor['selected_mask']);self.assertFalse(descriptor['index_ready'])
        self.assertNotIn('annotations',descriptor)

    def test_random_control_is_separate_calibration_role_never_approval_pair(self):
        self.bundle(method='random_matched_pool')
        spec=q.prepare(self.args('--mask',self.mask_path,'--selection-plan',self.plan_path,
                                '--episode-ids','calibration-invented-0'))
        server=OwnedServer()
        with patch('run_quality_server.subprocess.Popen',return_value=server),patch('run_quality_server.wait_server',return_value=self.props()), \
             patch('run_quality_server.subprocess.run',side_effect=lambda *a,**kw:self.results(spec)):
            q.execute(spec)
        descriptor,_=q.read_json(spec['args'].out/'run-descriptor.json')
        self.assertEqual(descriptor['role'],'random')
        self.assertFalse(descriptor['eligible_quality_evaluation'])

    def test_server_exit_and_startup_template_failure_never_complete(self):
        spec=q.prepare(self.args('--episode-ids','calibration-invented-0'))
        server=OwnedServer()
        def premature_exit(*args,**kwargs):
            self.results(spec);server.returncode=7
        with patch('run_quality_server.subprocess.Popen',return_value=server),patch('run_quality_server.wait_server',return_value=self.props()), \
             patch('run_quality_server.subprocess.run',side_effect=premature_exit):
            with self.assertRaises(ValueError):q.execute(spec)
        report,_=q.read_json(spec['args'].out/'manifest.json')
        self.assertFalse(report['completed']);self.assertFalse(server.terminated)
        self.assertFalse((spec['args'].out/'run-descriptor.json').exists())

    def test_owned_timeout_escalation_does_not_touch_other_processes(self):
        spec=q.prepare(self.args('--episode-ids','calibration-invented-0'))
        server=OwnedServer()
        def terminate(): server.terminated=True
        def wait(timeout):
            if not server.killed: raise subprocess.TimeoutExpired('owned',timeout)
            return server.returncode
        server.terminate=terminate;server.wait=wait
        with patch('run_quality_server.subprocess.Popen',return_value=server),patch('run_quality_server.wait_server',side_effect=ValueError('bad actual props')):
            with self.assertRaises(ValueError):q.execute(spec)
        self.assertTrue(server.terminated);self.assertTrue(server.killed)
        report,_=q.read_json(spec['args'].out/'manifest.json')
        self.assertFalse(report['completed']);self.assertFalse(report['eligible_quality_evaluation'])

    def test_inference_subprocess_failure_stops_only_owned_server_and_never_eligible(self):
        spec=q.prepare(self.args())
        server=OwnedServer()
        with patch('run_quality_server.subprocess.Popen',return_value=server),patch('run_quality_server.wait_server',return_value=self.props()), \
             patch('run_quality_server.subprocess.run',side_effect=subprocess.CalledProcessError(1,'invented')):
            with self.assertRaises(subprocess.CalledProcessError): q.execute(spec)
        self.assertTrue(server.terminated)
        report,_=q.read_json(spec['args'].out/'manifest.json')
        self.assertFalse(report['completed']);self.assertFalse(report['eligible_quality_evaluation'])
        self.assertFalse((spec['args'].out/'run-descriptor.json').exists())


if __name__=='__main__':
    unittest.main()
