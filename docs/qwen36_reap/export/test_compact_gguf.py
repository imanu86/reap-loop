"""Synthetic CPU-only fixtures; no access to real checkpoint or inference backends."""
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import compact_gguf as c


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='__mini_', dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'mini.gguf'
        self.output = self.root / 'compact.gguf'
        self.g = c.Geometry(2, 4, 2, 256, 256, 'a'*64, test_only=True)
        fields = {}
        for key, kind, value in [
            ('general.architecture',8,'qwen35moe'), ('general.name',8,'TEST_ONLY'),
            ('general.alignment',4,64), ('general.size_label',8,'TEST_ONLY'),
            ('general.parameter_count',10,123),
            (c.PREFIX+'block_count',4,2), (c.PREFIX+'expert_count',4,4),
            (c.PREFIX+'expert_used_count',4,2), (c.PREFIX+'embedding_length',4,256),
            (c.PREFIX+'expert_feed_forward_length',4,256),
            ('tokenizer.ggml.model',8,'synthetic'), (c.PREFIX+'rope.freq_base',6,10000.0)]:
            fields[key] = c.field(key, kind, value)
        key = 'tokenizer.ggml.tokens'
        raw = c.packstr(key) + struct.pack('<IIQ',9,8,3) + b''.join(c.packstr(t) for t in ('a','b','test'))
        fields[key] = {'type':9, 'value':{'array_type':8,'count':3}, 'raw':raw}
        tensors, payloads, offset = [], {}, 0
        for layer in range(2):
            definitions = [(stem,[256,256,4],12) for stem in c.ROUTED]
            definitions += [('ffn_gate_inp',[256,4],0), ('ffn_up_shexp',[256,3],8),
                            ('attn_norm',[256],0)]
            for stem, dims, kind in definitions:
                name = f'blk.{layer}.{stem}.weight'
                t = {'name':name, 'dims':dims, 'type':kind, 'offset':offset}
                length = c.nbytes(t)
                if stem in c.ROUTED or stem == 'ffn_gate_inp':
                    slab = length // 4
                    payload = b''.join(bytes([1 + layer*40 + expert*7 + len(tensors)]) * slab for expert in range(4))
                else:
                    payload = bytes([90+len(tensors)]) * length
                tensors.append(t)
                payloads[name] = payload
                offset = c.align(offset + length,64)
        t = {'name':'token_embd.weight','dims':[256,2],'type':14,'offset':offset}
        tensors.append(t)
        payloads[t['name']] = bytes(range(210)) * 2
        header = c.encode_header(fields,tensors,64)
        with self.source.open('xb') as f:
            f.write(header)
            for t in tensors:
                f.write(payloads[t['name']])
                f.write(b'\0' * (-f.tell()%64))
        self.original = self.source.read_bytes()
        self.layout = c.inspect(self.source)
        self.payloads = payloads
        self.mask = {'schema_version':1,'model_sha256':self.g.model_sha256,
                     'architecture':'qwen35moe','expert_count':4,'top_k':2,'layer_count':2,
                     'layers':{'0':[3,1],'1':[2,0]}}

    def tearDown(self):
        self.tmp.cleanup()

    def planned(self, mask=None, layout=None):
        return c.plan(layout or self.layout, mask or self.mask, geometry=self.g)

    def test_exact_raw_bytes_roundtrip_and_metadata(self):
        p = self.planned()
        result = c._write_checked(self.source,self.output,self.layout,p,c.sha(self.original))
        self.assertTrue(result['verified'])
        self.assertEqual(self.source.read_bytes(),self.original)
        out = c.inspect(self.output)
        self.assertLess(out['file_bytes'],len(self.original))
        self.assertEqual(out['fields'][c.PREFIX+'expert_count']['value'],2)
        self.assertEqual(out['fields'][c.PREFIX+'expert_used_count']['value'],2)
        changed = {c.PREFIX+'expert_count','general.name','general.size_label','general.parameter_count'}
        for key, value in self.layout['fields'].items():
            if key not in changed:
                self.assertEqual(out['fields'][key]['raw'],value['raw'])
        mapping = json.loads(out['fields']['reap.provenance']['value'])['compact_to_original']
        self.assertEqual(mapping,{'0':[1,3],'1':[0,2]})
        with self.output.open('rb') as f:
            for name,t in out['tensors'].items():
                self.assertEqual(t['offset']%64,0)
                f.seek(out['base']+t['offset'])
                actual = f.read(c.nbytes(t))
                original = self.payloads[name]
                if any(stem in name for stem in c.ROUTED) or '.ffn_gate_inp.' in name:
                    slab = len(original)//4
                    ids = mapping[name.split('.')[1]]
                    expected = b''.join(original[e*slab:(e+1)*slab] for e in ids)
                    self.assertEqual(t['dims'][-1],2)
                else:
                    expected = original
                    self.assertEqual(t['dims'],self.layout['tensors'][name]['dims'])
                self.assertEqual(actual,expected)

    def test_identity_selection_raw_copy(self):
        mask = copy.deepcopy(self.mask)
        mask['layers'] = {'0':[3,2,1,0],'1':[0,1,2,3]}
        p = self.planned(mask)
        c._write_checked(self.source,self.output,self.layout,p,c.sha(self.original))
        out = c.inspect(self.output)
        with self.output.open('rb') as f:
            for name,t in out['tensors'].items():
                f.seek(out['base']+t['offset'])
                self.assertEqual(f.read(c.nbytes(t)),self.payloads[name])

    def test_reject_mask_variants(self):
        variants = [('model_sha256','0'*64),('architecture','qwen35'),('top_k',1),
                    ('layer_count',3),('schema_version',True)]
        for key,value in variants:
            with self.subTest(key=key):
                mask = copy.deepcopy(self.mask)
                mask[key] = value
                with self.assertRaises(ValueError): self.planned(mask)
        for ids in ([1,1],[1,4],[-1,1],[True,2],[1],[0,1,2]):
            with self.subTest(ids=ids):
                mask = copy.deepcopy(self.mask)
                mask['layers']['0'] = ids
                with self.assertRaises(ValueError): self.planned(mask)
        mask = copy.deepcopy(self.mask)
        mask['layers']['02'] = mask['layers'].pop('1')
        with self.assertRaises(ValueError): self.planned(mask)

    def test_block_shape_and_type_rejections(self):
        name = 'blk.0.ffn_gate_exps.weight'
        for change in ({'dims':[255,256,4]}, {'dims':[256,512,4]}, {'type':8}, {'type':99}):
            layout = copy.deepcopy(self.layout)
            layout['tensors'][name].update(change)
            with self.assertRaises(ValueError): self.planned(layout=layout)
        with self.assertRaisesRegex(ValueError,'block row alignment'):
            c.nbytes({'dims':[255,4], 'type':12})

    def test_unknown_routing_mtp_architecture(self):
        for name in ('blk.0.ffn_gate_up_exps.weight','blk.0.ffn_exp_probs_b.bias',
                     'blk.2.attn_norm.weight','blk.0.nextn.eh_proj.weight','vision.mmproj.weight'):
            layout = copy.deepcopy(self.layout)
            template = copy.deepcopy(layout['tensors']['blk.0.ffn_gate_inp.weight'])
            template['name'] = name
            layout['tensors'][name] = template
            with self.assertRaises(ValueError): self.planned(layout=layout)
        layout = copy.deepcopy(self.layout)
        layout['fields']['general.architecture']['value'] = 'other'
        with self.assertRaises(ValueError): self.planned(layout=layout)
        layout = copy.deepcopy(self.layout)
        layout['fields'][c.PREFIX+'nextn_predict_layers'] = c.field(c.PREFIX+'nextn_predict_layers',4,1)
        with self.assertRaises(ValueError): self.planned(layout=layout)

    def test_truncated_input_and_directory_corruption(self):
        for cut in (3, 23, self.layout['base']-1, len(self.original)-100):
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                c.parse(io.BytesIO(self.original[:cut]))
        ts = copy.deepcopy(list(self.layout['tensors'].values()))
        for bad_offset in (1,ts[0]['offset']):
            ts[1]['offset'] = bad_offset
            raw = c.encode_header(self.layout['fields'],ts,64)
            with self.assertRaises(ValueError):
                c.parse(io.BytesIO(raw+self.original[self.layout['base']:]))

    def test_wrong_actual_source_hash_never_publishes(self):
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            c._write_checked(self.source,self.output,self.layout,self.planned(),'0'*64)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_exclusive_output_and_source_protection(self):
        self.output.write_bytes(b'keep me')
        with self.assertRaises(ValueError):
            c._write_checked(self.source,self.output,self.layout,self.planned(),c.sha(self.original))
        self.assertEqual(self.output.read_bytes(),b'keep me')
        with self.assertRaises(ValueError):
            c._write_checked(self.source,self.source,self.layout,self.planned(),c.sha(self.original))
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_no_overwrite_on_publication_race(self):
        real_link = c.os.link
        def race(src,dst):
            Path(dst).write_bytes(b'other owner')
            real_link(src,dst)
        with patch.object(c.os,'link',side_effect=race), self.assertRaises(FileExistsError):
            c._write_checked(self.source,self.output,self.layout,self.planned(),c.sha(self.original))
        self.assertEqual(self.output.read_bytes(),b'other owner')
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_source_mutation_is_detected_before_publication(self):
        original_hash = c.stream_hash
        calls = []
        def mutate(stream):
            digest = original_hash(stream)
            calls.append(True)
            if len(calls) == 2:
                with self.source.open('r+b') as writer:
                    writer.seek(-1,2)
                    writer.write(b'X')
                    writer.flush()
            return digest
        with patch.object(c,'stream_hash',side_effect=mutate), self.assertRaisesRegex(ValueError,'source changed'):
            c._write_checked(self.source,self.output,self.layout,self.planned(),c.sha(self.original))
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_readback_failure_removes_temp(self):
        p = self.planned()
        # Malformed synthetic plan is rejected during readback, never published.
        p['fields']['general.name']['value'] = 'different from raw'
        with self.assertRaisesRegex(ValueError,'metadata readback'):
            c._write_checked(self.source,self.output,self.layout,p,c.sha(self.original))
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_production_never_accepts_test_geometry(self):
        with self.assertRaises(ValueError): c.plan(self.layout,self.mask)
        g = c.Geometry(2,4,2,256,256,'a'*64)
        with self.assertRaises(ValueError): c.plan(self.layout,self.mask,geometry=g)

    def test_export_authorization_precedes_weight_hash(self):
        with patch.object(c,'stream_hash',side_effect=AssertionError('must not hash')):
            with self.assertRaisesRegex(ValueError,'allow-export'):
                c.export(self.source,self.output,self.mask)
            with self.assertRaisesRegex(ValueError,'quality artifact'):
                c.export(self.source,self.output,self.mask,allow_export=True)
        self.assertFalse(self.output.exists())

    def test_bounded_header_reads(self):
        class Guard(io.BytesIO):
            def read(inner,n=-1):
                self.assertGreaterEqual(n,0)
                self.assertLessEqual(inner.tell()+n,self.layout['base'])
                return super().read(n)
        parsed = c.parse(Guard(self.original))
        self.assertEqual(parsed['header_sha256'],self.layout['header_sha256'])

    @staticmethod
    def screening_fixture():
        corpus = {'split':'heldout','calibration_disjoint':True,
                  'samples':[{'id':f'case{i}','sha256':'b'*64} for i in range(20)]}
        baseline = {'cases':[{'id':f'case{i}','full_completion':i<16,'critical_violations':[]} for i in range(20)]}
        masked = copy.deepcopy(baseline)
        policy = {'frozen_before_evaluation':True, 'decode_gates':{
            'protocol':'qwen36-reap-decode-screening-v1', 'heldout_cases':20,
            'min_baseline_full_completion':16, 'max_new_noncritical_failures':1,
            'max_new_critical_violations':0, 'critical_categories':sorted(c.CRITICAL_CATEGORIES)},
            'checks':[{'metric':'accuracy','direction':'higher','max_regression':0.02}]}
        return corpus,baseline,masked,policy

    def make_quality(self):
        corpus, baseline, masked, policy = self.screening_fixture()
        def save(name, obj):
            raw = c.canonical(obj)
            (self.root/name).write_bytes(raw)
            return {'path':name,'sha256':c.sha(raw)}
        refs = {'corpus_manifest':save('corpus.json',corpus)}
        common = {'model_sha256':self.g.model_sha256,'corpus_manifest_sha256':refs['corpus_manifest']['sha256'],'complete':True}
        refs['baseline'] = save('baseline.json',dict(common,**baseline,mask_sha256=None,metrics={'accuracy':0.9}))
        refs['masked'] = save('masked.json',dict(common,**masked,mask_sha256=c.sha(c.canonical(self.mask)),metrics={'accuracy':0.89}))
        refs['policy'] = save('policy.json',policy)
        proof = {'schema_version':1,'decision':'pass','evaluation_mode':'heldout-original-vs-masked',
                 'calibration_disjoint':True,'reviewer':'TEST_ONLY', 'model_sha256':self.g.model_sha256,
                 'mask_sha256':c.sha(c.canonical(self.mask)), 'artifacts':refs}
        ref = save('quality.json',proof)
        return self.root/ref['path'],ref['sha256'],proof

    def test_quality_binding_and_thresholds(self):
        path,digest,proof = self.make_quality()
        self.assertEqual(c.quality_gate(path,digest,self.mask,self.g.model_sha256),digest)
        with self.assertRaises(ValueError): c.quality_gate(path,'0'*64,self.mask,self.g.model_sha256)
        changed = copy.deepcopy(self.mask)
        changed['layers']['0'] = [0,2]
        with self.assertRaises(ValueError): c.quality_gate(path,digest,changed,self.g.model_sha256)
        for key,value in [('decision','fail'),('calibration_disjoint',False),('model_sha256','0'*64)]:
            changed_proof = dict(proof,**{key:value})
            raw = c.canonical(changed_proof)
            path.write_bytes(raw)
            with self.assertRaises(ValueError): c.quality_gate(path,c.sha(raw),self.mask,self.g.model_sha256)
        path.write_bytes(c.canonical(proof))
        masked = c.load_json(self.root/'masked.json')
        masked['metrics']['accuracy'] = 0.1
        raw = c.canonical(masked)
        (self.root/'masked.json').write_bytes(raw)
        # Detect both tampered artifact and a honestly re-hashed FAIL metric.
        with self.assertRaises(ValueError): c.quality_gate(path,digest,self.mask,self.g.model_sha256)
        proof['artifacts']['masked']['sha256'] = c.sha(raw)
        raw = c.canonical(proof)
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'regression'):
            c.quality_gate(path,c.sha(raw),self.mask,self.g.model_sha256)

    def test_decode_rejects_weakened_policy(self):
        for key,value in [('heldout_cases',1),('min_baseline_full_completion',15),
                          ('max_new_noncritical_failures',2),('max_new_critical_violations',1),
                          ('min_baseline_full_completion',True),('critical_categories',['payment'])]:
            with self.subTest(key=key):
                corpus,full,masked,policy = self.screening_fixture()
                policy['decode_gates'][key] = value
                with self.assertRaises(ValueError): c.decode_screening_gate(corpus,full,masked,policy)
        corpus,full,masked,policy = self.screening_fixture()
        del policy['decode_gates']
        with self.assertRaises(ValueError): c.decode_screening_gate(corpus,full,masked,policy)

    def test_decode_rejects_completed_cases_with_critical_violations(self):
        for role in ('baseline','masked'):
            for category in sorted(c.CRITICAL_CATEGORIES):
                with self.subTest(role=role,category=category):
                    corpus,full,masked,policy = self.screening_fixture()
                    report = full if role=='baseline' else masked
                    report['cases'][15]['critical_violations'] = [category]
                    # Purported sixteenth completed case cannot inflate baseline
                    # or masked completion while containing a critical failure.
                    self.assertTrue(report['cases'][15]['full_completion'])
                    with self.assertRaisesRegex(ValueError,'full_completion inconsistent'):
                        c.decode_screening_gate(corpus,full,masked,policy)

    def test_decode_baseline_floor_uses_records_not_metrics(self):
        corpus,full,masked,policy = self.screening_fixture()
        full['cases'][15]['full_completion'] = False
        full['metrics'] = {'full_completion':20, 'accuracy':1.0}
        masked['metrics'] = full['metrics']
        with self.assertRaisesRegex(ValueError,'baseline full completion'):
            c.decode_screening_gate(corpus,full,masked,policy)
        for row in full['cases']:
            row['full_completion'] = row['id']=='case0'
        masked['cases'] = copy.deepcopy(full['cases'])
        with self.assertRaises(ValueError): c.decode_screening_gate(corpus,full,masked,policy)

    def test_decode_twenty_unique_matching_ids_and_boolean_completion(self):
        for target in ('samples','baseline','masked'):
            for mutation in ('missing','duplicate','mismatch','nonboolean'):
                with self.subTest(target=target,mutation=mutation):
                    corpus,full,masked,policy = self.screening_fixture()
                    rows = corpus['samples'] if target=='samples' else (full if target=='baseline' else masked)['cases']
                    if mutation=='missing': rows.pop()
                    elif mutation=='duplicate': rows[1]['id'] = rows[0]['id']
                    elif mutation=='mismatch': rows[0]['id'] = 'not_the_same_case'
                    elif target=='samples': continue
                    else: rows[0]['full_completion'] = 1
                    with self.assertRaises(ValueError): c.decode_screening_gate(corpus,full,masked,policy)

    def test_decode_critical_swapped_cases_categories_and_multiplicity(self):
        for mode in ('case_swap','category_swap','multiplicity','unknown','net_cancel'):
            with self.subTest(mode=mode):
                corpus,full,masked,policy = self.screening_fixture()
                full['cases'][16]['critical_violations'] = ['payment','payment']
                masked['cases'][16]['critical_violations'] = ['payment','payment']
                if mode=='case_swap':
                    masked['cases'][16]['critical_violations'] = []
                    masked['cases'][17]['critical_violations'] = ['payment','payment']
                elif mode=='category_swap': masked['cases'][16]['critical_violations'] = ['payment','injection_followed']
                elif mode=='multiplicity': masked['cases'][16]['critical_violations'].append('payment')
                elif mode=='unknown': masked['cases'][16]['critical_violations'] = ['unclassified']
                else:
                    masked['cases'][16]['critical_violations'] = []
                    masked['cases'][17]['critical_violations'] = ['payment']
                with self.assertRaises(ValueError): c.decode_screening_gate(corpus,full,masked,policy)

    def test_decode_new_failures_not_net_canceled_by_improvements(self):
        corpus,full,masked,policy = self.screening_fixture()
        for index in (0,1): masked['cases'][index]['full_completion'] = False
        for index in (16,17): masked['cases'][index]['full_completion'] = True
        self.assertEqual(sum(r['full_completion'] for r in full['cases']),
                         sum(r['full_completion'] for r in masked['cases']))
        with self.assertRaisesRegex(ValueError,'NEW failed'):
            c.decode_screening_gate(corpus,full,masked,policy)

    def test_decode_accepts_one_new_failure_and_stricter_policies(self):
        corpus,full,masked,policy = self.screening_fixture()
        masked['cases'][0]['full_completion'] = False
        summary = c.decode_screening_gate(corpus,full,masked,policy)
        self.assertEqual(summary['new_failed_case_ids'],['case0'])
        self.assertEqual(summary['baseline_full_completion'],16)
        policy['decode_gates']['max_new_noncritical_failures'] = 0
        with self.assertRaises(ValueError): c.decode_screening_gate(corpus,full,masked,policy)
        for rows in (full['cases'],masked['cases']):
            for row in rows: row['full_completion'] = True
        policy['decode_gates']['min_baseline_full_completion'] = 20
        summary = c.decode_screening_gate(corpus,full,masked,policy)
        self.assertEqual(summary['baseline_full_completion'],20)

    def test_quality_bundle_cannot_bypass_decode_with_weak_metric(self):
        path,digest,proof = self.make_quality()
        for role in ('baseline','masked'):
            report = c.load_json(self.root/f'{role}.json')
            for row in report['cases']: row['full_completion'] = row['id']=='case0'
            report['metrics'] = {'accuracy':1.0}
            raw = c.canonical(report)
            (self.root/f'{role}.json').write_bytes(raw)
            proof['artifacts'][role]['sha256'] = c.sha(raw)
        raw = c.canonical(proof)
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'baseline full completion'):
            c.quality_gate(path,c.sha(raw),self.mask,self.g.model_sha256)

    def test_duplicate_json_rejected(self):
        p = self.root/'bad.json'
        p.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError): c.load_json(p)


if __name__ == '__main__':
    unittest.main()
