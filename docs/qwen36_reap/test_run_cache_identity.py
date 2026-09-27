"""Invented small logits and mocked lifecycle ONLY; no actual model/GPU/process inspection."""
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_cache_identity as q


class Child:
    pid = 987654
    def __init__(self, timeout=False):
        self.returncode = None
        self.timeout = timeout
        self.terminated = self.killed = False
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired('invented-owned-helper', timeout)
        self.returncode = 1 if self.killed else 0
        return self.returncode
    def terminate(self):
        self.terminated = True
    def kill(self):
        self.killed = True


class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixtures = {}
        for case in q.CASES[:-1]:
            path = self.root / (case + '.txt')
            path.write_text('Invented raw UTF8 è ' + case, encoding='utf-8')
            self.fixtures[case] = path
        for target, value in [('VOCAB',64),('fixture_path',lambda case:self.fixtures[case])]:
            p = patch.object(q,target,value);p.start();self.addCleanup(p.stop)
        self.children = []
        self.model_hash_calls = 0
        self.inventory = SimpleNamespace(stdout='"python.exe","1234","Console","1","10 K"\n')
        p=patch.object(q.subprocess,'run',return_value=self.inventory);self.inventory_mock=p.start();self.addCleanup(p.stop)
        self.real_sha=q.sha
        def fake_sha(path):
            path=Path(path)
            if path==q.MODEL:
                self.model_hash_calls+=1
                return q.MODEL_SHA
            if path.parent==q.BINARY:return q.RUNTIME_PINS[path.name]
            if path in q.SOURCE_PINS:return q.SOURCE_PINS[path]
            return self.real_sha(path)
        p=patch.object(q,'sha',side_effect=fake_sha);self.sha_mock=p.start();self.addCleanup(p.stop)
        p=patch.object(q,'runtime_files',return_value=[q.BINARY/name for name in q.RUNTIME_PINS]);p.start();self.addCleanup(p.stop)
        p=patch.object(q.subprocess,'Popen',side_effect=self.launch);self.popen=p.start();self.addCleanup(p.stop)

    def markers(self, capacity):
        return ('I moe-cache: current-step GPU path active for 40 layers; prefill graph unchanged\n'
                f'I operator(): MoE expert cache enabled: 40 layers x {capacity} slots, 8 inserts/step, '
                f'{q.PAYLOAD_BYTES[capacity]/1048576:.1f} MiB device memory\n')

    def metadata(self, prefix, capacity, case='web_dom_utf8', cmd=None, run_id='invented'):
        count=2 if case=='single_token' else 8
        ids=[50] if case=='single_token' else [3,4]
        prompt='' if case=='single_token' else self.fixtures[case].read_text(encoding='utf-8')
        calls=[dict(helper_call_ordinal_1based=1,phase='prefill',position_start=0,n_tokens=len(ids),seq_id=0)]
        calls += [dict(helper_call_ordinal_1based=i+2,phase='decode',position_start=len(ids)+i,n_tokens=1,seq_id=0,token_id=2)
                  for i in range(count-1)]
        m=dict(schema_version=1,complete=True,run_id=run_id,model_path=str(q.MODEL),model_sha256=q.MODEL_SHA,
            architecture='qwen35moe',layer_count=40,expert_count=256,top_k=8,dtype='float32',endianness='little',shape=[count,64],
            token_ids=[2]*count,greedy_token_ids=[2]*count,prompt_token_ids=ids,mode='greedy',warmup=False,target_only=True,
            fixed_count_including_eog=True,parse_special=False,prompt_mode='single_token' if case=='single_token' else 'raw_utf8',
            prompt_utf8=prompt,prompt_bytes=len(prompt.encode()),add_special=case!='single_token',
            settings=dict(q.SETTINGS,moe_cache_slots=capacity),environment=deepcopy(q.HELPER_ENV),decode_calls=calls,argv=cmd or [],
            steps=[dict(step=i,phase='last_prefill' if i==0 else 'incremental_decode',prefix_length=len(ids)+i,
                        logits_offset_bytes=i*64*4,token_id=2,greedy_token_id=2,helper_call_ordinal_1based=i+1) for i in range(count)])
        q.save(q.artifact(prefix,'.json'),m)
        row=[0.0]*64;row[2]=1.0
        q.artifact(prefix,'.logits.f32').write_bytes(struct.pack('<64f',*row)*count)
        return m

    def launch(self, cmd, **kw):
        capacity=int(cmd[cmd.index('--moe-expert-cache')+1])
        prefix=Path(cmd[cmd.index('--gate-out')+1]);run_id=cmd[cmd.index('--gate-run-id')+1]
        case='single_token' if '--gate-token-id' in cmd else Path(cmd[cmd.index('-f')+1]).stem
        self.metadata(prefix,capacity,case,cmd,run_id)
        kw['stderr'].write(self.markers(capacity).encode())
        child=Child();self.children.append(child)
        return child

    def test_optin_new_private_output_and_no_read_on_busy(self):
        with self.assertRaises(ValueError):q.main(['--out',str(self.root/'out')])
        self.inventory_mock.assert_not_called();self.assertEqual(self.model_hash_calls,0)
        for name in ['llama-server.exe','reap-native-gate.exe','cl.exe','nvcc.exe','MSBuild.exe']:
            self.inventory.stdout=f'"{name}","42"\n'
            with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(self.root/'out')])
        self.assertEqual(self.model_hash_calls,0);self.popen.assert_not_called()
        self.inventory.stdout='"python.exe","1"\n'
        with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(self.root)])
        with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(q.HERE/'new-output')])

    def test_unverifiable_inventory_and_runtime_drift_stop_before_model(self):
        self.inventory.stdout='INFO: invalid/no process inventory\n'
        with self.assertRaises(ValueError):q.ensure_exclusive()
        self.inventory.stdout='"python.exe","1"\n'
        original=self.sha_mock.side_effect
        self.sha_mock.side_effect=lambda p:'0'*64 if Path(p).name==q.EXE else original(p)
        with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(self.root/'out')])
        self.assertEqual(self.model_hash_calls,0);self.popen.assert_not_called()

    def test_known_markers_prove_capacity_not_requested_argument(self):
        for capacity in q.CAPACITIES:
            proof=q.activation_proof(self.markers(capacity),capacity)
            self.assertTrue(proof['verified']);self.assertEqual(proof['derived_payload_bytes'],q.PAYLOAD_BYTES[capacity])
            self.assertFalse(proof['exact_byte_telemetry']);self.assertTrue(proof['not_total_vram'])
        bad=['',self.markers(32).splitlines()[0],self.markers(32)*2,self.markers(64),
             self.markers(32).replace('40 layers','39 layers'),self.markers(32).replace('2160.0','2160.1'),
             self.markers(32)+'\nmoe-elastic: enabled',self.markers(32)+'\ncache disabled']
        for text in bad:
            with self.assertRaises(ValueError):q.activation_proof(text,32)

    def test_env_scrub_logging_only_override_and_frozen_command(self):
        with patch.dict(os.environ,{'LLAMA_MOE_ELASTIC':'1','LLAMA_ARG_LOG_VERBOSITY':'0','QWEN36_REAP_TRACE':'unsafe',
                                    'QWEN36_REAP_MASK':'unsafe','LLAMA_MOE_CACHE_CONGELA':'3'}):
            env,recorded=q.controlled_environment()
        self.assertEqual(env['LLAMA_MOE_CACHE_BATCH'],'1');self.assertEqual(env['LLAMA_MOE_ELASTIC'],'0')
        self.assertEqual(env['LLAMA_MOE_DEMAND_GPU'],'1');self.assertEqual(recorded['LLAMA_ARG_LOG_VERBOSITY'],'4')
        for key in ['QWEN36_REAP_TRACE','QWEN36_REAP_MASK','LLAMA_MOE_CACHE_CONGELA']:self.assertNotIn(key,env)
        cmd=q.command(96,'single_token',self.root/'x','fresh')
        for key,value in [('--moe-expert-cache','96'),('-c','2048'),('-b','128'),('-ub','128'),('-t','16'),('-ctk','q8_0'),('-n','2')]:
            self.assertEqual(cmd[cmd.index(key)+1],value)
        self.assertEqual(cmd[-2:],['--gate-token-id','50'])
        self.assertNotIn('--gate-teacher',cmd)

    def test_projection_reuses_full_math_without_global_patch_or_file_rewrite(self):
        a,b=self.root/'a',self.root/'b'
        self.metadata(a,32);self.metadata(b,64)
        original_loader=q.logits.load_metadata
        hashes=[q.sha(q.artifact(p,'.json')) for p in (a,b)]
        with self.assertRaises(ValueError):q.logits.compare(a,b) # original remains strict
        report=q.compare_capacity(a,b)
        self.assertTrue(report['pass']);self.assertTrue(report['bit_identical']);self.assertEqual(report['maxabs'],0)
        self.assertTrue(report['metadata_projection']);self.assertFalse(report['metadata_bit_identity_claimed'])
        self.assertEqual(report['actual_settings']['candidate']['moe_cache_slots'],64)
        self.assertIs(original_loader,q.logits.load_metadata)
        self.assertEqual(hashes,[q.sha(q.artifact(p,'.json')) for p in (a,b)])
        with self.assertRaises(ValueError):q.logits.compare(a,b)
        path=q.artifact(b,'.json');m=json.loads(path.read_text());m['settings']['n_ctx_actual']=1024
        path.write_text(json.dumps(m))
        with self.assertRaises(ValueError):q.compare_capacity(a,b)

    def test_nonfinite_numeric_errors_top1_metadata_and_partial_logits_fail(self):
        a=self.root/'a';self.metadata(a,32)
        for i,value in enumerate((float('nan'),float('inf'),0.01,2.0)):
            b=self.root/f'b{i}';self.metadata(b,64)
            path=q.artifact(b,'.logits.f32');raw=bytearray(path.read_bytes());raw[:4]=struct.pack('<f',value);path.write_bytes(raw)
            report=q.compare_capacity(a,b)
            self.assertFalse(report['pass'])
            if value==0.01:self.assertGreater(report['maxabs'],0.009)
            if value==2.0:self.assertFalse(report['steps'][0]['top1_metadata_valid'])
        b=self.root/'short';self.metadata(b,96)
        q.artifact(b,'.logits.f32').write_bytes(b'\0')
        with self.assertRaises(ValueError):q.compare_capacity(a,b)

    def test_tolerance_is_not_bit_identity_and_generated_divergence_fails(self):
        a,b=self.root/'a',self.root/'b';self.metadata(a,32);self.metadata(b,96)
        data=q.artifact(b,'.logits.f32');raw=bytearray(data.read_bytes());raw[:4]=struct.pack('<f',1e-6);data.write_bytes(raw)
        report=q.compare_capacity(a,b)
        self.assertTrue(report['pass']);self.assertFalse(report['bit_identical'])
        self.assertEqual(report['policy']['atol'],1e-5);self.assertEqual(report['policy']['rtol'],1e-5)
        mpath=q.artifact(b,'.json');m=json.loads(mpath.read_text())
        m['token_ids']=m['greedy_token_ids']=[3]*8
        for row in m['steps']:row['token_id']=row['greedy_token_id']=3
        for row in m['decode_calls'][1:]:row['token_id']=3
        mpath.write_text(json.dumps(m));row=[0.0]*64;row[3]=1.0;data.write_bytes(struct.pack('<64f',*row)*8)
        report=q.compare_capacity(a,b)
        self.assertFalse(report['pass']);self.assertFalse(report['tokens_equal'])
        self.assertEqual(report['first_token_divergence'],0);self.assertEqual(report['compared_steps'],1)

    def test_source_fixture_pin_mismatch_precedes_model_read(self):
        original=self.sha_mock.side_effect
        target=next(p for p in q.SOURCE_PINS if p.name=='web_dom_utf8.txt')
        self.sha_mock.side_effect=lambda path:'0'*64 if Path(path)==target else original(path)
        with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(self.root/'out')])
        self.assertEqual(self.model_hash_calls,0);self.popen.assert_not_called()

    def test_spawn_error_and_nonzero_exit_never_complete(self):
        self.popen.side_effect=OSError('invented spawn failure')
        with self.assertRaises(OSError):q.main(['--allow-inference','--out',str(self.root/'spawn')])
        shutdown=json.loads((self.root/'spawn/cap32_web_dom_utf8.shutdown.json').read_text())
        self.assertIsNone(shutdown['pid']);self.assertTrue(shutdown['stopped'])
        child=Child();child.returncode=7
        child.wait=lambda timeout:7
        self.popen.side_effect=None;self.popen.return_value=child
        with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(self.root/'exit')])
        self.assertFalse(json.loads((self.root/'exit/manifest.json').read_text())['pass'])
        self.assertFalse(child.terminated)

    def test_postflight_model_hash_mismatch_is_not_numerical_pass(self):
        original=self.sha_mock.side_effect
        def changed(path):
            value=original(path)
            return '0'*64 if Path(path)==q.MODEL and self.model_hash_calls==2 else value
        self.sha_mock.side_effect=changed
        with self.assertRaisesRegex(ValueError,'model hash mismatch'):
            q.main(['--allow-inference','--out',str(self.root/'out')])
        m=json.loads((self.root/'out/manifest.json').read_text())
        self.assertFalse(m['pass']);self.assertFalse(m['complete']);self.assertIn('postflight_error',m)

    def test_complete_nine_probe_six_comparison_matrix_and_post_hashes(self):
        out=self.root/'out'
        result=q.main(['--allow-inference','--out',str(out)])
        self.assertTrue(result['complete']);self.assertTrue(result['pass'])
        self.assertEqual(result['scope'],q.SCOPE);self.assertFalse(result['quality_approval'])
        self.assertEqual(len(result['probes']),9);self.assertEqual(len(result['comparisons']),6)
        self.assertEqual(self.model_hash_calls,2)
        self.assertEqual(result['inputs_before'],result['inputs_after'])
        self.assertEqual([p['capacity'] for p in result['probes']],[32,64,96]*3)
        self.assertTrue(all(not c.terminated and c.poll()==0 for c in self.children))
        self.assertEqual(len({p['run_id'] for p in result['probes']}),9)
        with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(out)])
        self.assertEqual(self.popen.call_count,9)

    def test_missing_activation_is_failure_not_argv_confirmation(self):
        original=self.launch
        def no_marker(cmd,**kw):
            child=original(cmd,**kw);kw['stderr'].seek(0);kw['stderr'].truncate();return child
        self.popen.side_effect=no_marker
        out=self.root/'out'
        with self.assertRaisesRegex(ValueError,'activation_unverified'):q.main(['--allow-inference','--out',str(out)])
        m=json.loads((out/'manifest.json').read_text())
        self.assertFalse(m['pass']);self.assertFalse(m['complete']);self.assertEqual(m['probes'],[])
        self.assertEqual(self.popen.call_count,1);self.assertEqual(self.model_hash_calls,2)

    def test_timeout_escalates_only_owned_handle_and_records_failure(self):
        child=Child(timeout=True)
        self.popen.side_effect=None;self.popen.return_value=child
        out=self.root/'out'
        with self.assertRaises(subprocess.TimeoutExpired):q.main(['--allow-inference','--out',str(out)])
        self.assertTrue(child.terminated and child.killed)
        m=json.loads((out/'manifest.json').read_text());self.assertFalse(m['pass'])
        shutdown=json.loads((out/'cap32_web_dom_utf8.shutdown.json').read_text())
        self.assertTrue(shutdown['stopped']);self.assertEqual(shutdown['pid'],child.pid)
        self.assertEqual(self.model_hash_calls,2)

    def test_stale_run_metadata_incomplete_and_timestamp_are_rejected(self):
        for index,kind in enumerate(('run_id','complete','timestamp')):
            original=self.launch
            def stale(cmd,kind=kind,**kw):
                child=original(cmd,**kw);prefix=Path(cmd[cmd.index('--gate-out')+1]);path=q.artifact(prefix,'.json')
                if kind=='timestamp':os.utime(path,ns=(1,1))
                else:
                    m=json.loads(path.read_text());m[kind]='old-run' if kind=='run_id' else False;path.write_text(json.dumps(m))
                return child
            self.popen.side_effect=stale
            with self.assertRaises(ValueError):q.main(['--allow-inference','--out',str(self.root/f'bad{index}')])

    def test_postflight_model_or_source_drift_prevents_pass(self):
        original=q.pin_inputs
        calls=0
        def drift():
            nonlocal calls
            calls+=1;result=original()
            if calls==2:result['source_fixture_sha256']['changed']='wrong'
            return result
        out=self.root/'out'
        with patch.object(q,'pin_inputs',side_effect=drift),self.assertRaises(ValueError):
            q.main(['--allow-inference','--out',str(out)])
        m=json.loads((out/'manifest.json').read_text());self.assertFalse(m['pass']);self.assertIn('postflight_error',m)


if __name__=='__main__':unittest.main()
