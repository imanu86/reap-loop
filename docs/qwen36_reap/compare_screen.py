"""Strict offline paired CALIBRATION reducer; no inference, network or model access.

API: compare(baseline, ranked, random, plan_path, ranked_mask, random_mask,
             output, *, reviews=None, expected_cases=10, ranking_method='mass_gate').
All arguments are explicit closed artifact paths. reviews maps arm names to review
JSON paths; omitted arms remain pending. CLI has corresponding --baseline,
--ranked, --random, --plan, --ranked-mask, --random-mask, --output, --*-review,
--expected-cases {10,50}, --ranking-method {mass_gate,mean_selected_gate} (mass
by default). Never pass active runs. 50 must be explicitly selected.

Ranking identity is separate from the unchanged capture-only protocol SHA.
New plan AND actual masks require all3 fields ranking_method, selection_recipe,
selection_recipe_sha256; candidate entries require method+recipe SHA binding.
Recipe must match expected_recipe() EXACTLY, recursively typed (bool!=0, int!=
float), with UTF-8 ensure_ascii=False sorted compact SHA. Random mask.method
stays random_matched_pool but references the SAME selected ranking recipe.
All new outer manifests carry only ranking_method; recipeobjects/digests are
bound through plan/mask file hashes, not copied into outer metadata. Mean needs
explicit matching method on ALL3 arms, including unmasked baseline with no plan.
Historical139/075 with ALL recipe metadata absent remains mass-only, tagged
legacy_mass_only with null recipe in output; never silently promoted to mean or
retroactively assigned a recipe. Partial metadata/method/recipe/hash mismatch
fails. Both newmass and mean recipes are g-only ablations, not full REAP output
norm ranking. All model/config/control/review gates and approvalfalse stay fixed.

Required closed-run files: manifest.json, shutdown.json, server.pid.json,
pilot/manifest.json, pilot/summary.json, pilot/transcripts.jsonl. Coordinator
schema is frozen run_quality_server.py V3/native/final-tool/greedy seed0, context
and budget4096/output1024, untraced candidate018c. Model paths are compared as
strings only: no files at those paths are opened. Plans/masks are explicit small
JSON artifacts; no traversal of plan provenance.run, corpus, trace or runtime.

Optional review: {transcript_sha256, reviewer:<truthful identity>, split:
'calibration', unparsed_turns_reviewed:true, cases:[{id,reviewed:true,reviewer,
transcript_record_sha256,reviewed_full_completion:<actual raw bool>,
case_critical_violations:[...], final_critical_violations:[...],action_reviews:
[{action_index:<zero-based>,action_sha256,critical_violations:[...]}]}]}.
All critical lists must be EXPLICIT (including []). All case IDs/actions must
match exactly; no auto-review or rescoring. Existing private calibration50 review
shape is accepted using method.unparsed_turns_reviewed instead of top-level flag.
Case-level review covers malformed/unparsed model assertions. Categories are the
five exporter DECODE_GATES categories; repeated entries preserve multiplicity.
Review pointers/hashes are evidence, not proof of evaluator honesty.

Output contains IDs/hashes/counts only, no prompts/responses/reasoning. All inputs
are bounded, read with metadata identity checks, SHA-pinned and rechecked before
exclusive no-clobber publication. Shutdown attestations plus quiescent files do
not independently prove OS process death. Supplied reviewed annotations remain
responsible for semantic classification (redundant read != automatically unsafe).
Default decision=pending_explicit_critical_review. All3 arms need reviews before
ranked screen can be acceptable (<=1 NEW failed ID and zero NEW critical(case,
category) occurrences). Random is a separately reported control, never silently
ignored or relabeled ranked. Improvements never cancel new failures/violations.
Calibration screen acceptance is NOT quality/export/heldout/next-K approval.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_screen_export',HERE/'export/compact_gguf.py')
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)
require, canonical, sha = gate.require, gate.canonical, gate.sha
MAX_BYTES = 64 << 20
CONFIG = dict(protocol='native',model='local-pilot',max_turns=13,max_output=1024,
    budget=4096,thinking='template-default',template_supports_thinking=False,
    policy_version='v3',diagnostic_raw=False,sampling_profile='greedy',seed=0,final_mode='tool')
SOURCE_KEYS = {'runner_sha256','validator_sha256','comparison_wrapper_sha256','capture_coordinator_sha256'}
ENV = {'GGML_SCHED_ASYNC_INPUTS':'1','GGML_CUDA_GRAPH_COMPAT_CACHE':'1','GGML_CUDA_MMVQ_OCCUPANCY':'1',
       'LLAMA_MOE_DEMAND_GPU':'1','LLAMA_MOE_CACHE_BATCH':'1','LLAMA_MOE_ELASTIC':'0','QWEN36_REAP_MODEL_SHA256':gate.MODEL_SHA}
RUNTIME = {
 'ggml-base.dll':'9d0cb021fef05cb1d53c5d22a6f30322c6b3ab966dca9489ea7e6b4cfa13ddc4',
 'ggml-cpu.dll':'1af0511165f60adaa4366186e58abbcf24c25bf9ad71e41db42ac85eda33a5a7',
 'ggml-cuda.dll':'a8e6d2a95c1fcaa14d28914cee1657e7e492a9b788afaaf5430d8019e667886e',
 'ggml.dll':'86e77878c5ee1bcee2b11877b2a7c4f8806e2debf3a24c7883ee45cb77a436db',
 'llama-common.dll':'aee53a4efbd7f0036be8382546922f51f8e10fbdf448e4a3c44681d7dce3fb69',
 'llama-server-impl.dll':'c06117ce6851312cc2d09dd7cfc89468d1664bfbaa055e2638ad8b452f18c3ae',
 'llama-server.exe':'75bf4eec3d3782ed865ca04e706f259584026f86fb3223b106dd68517e7c3bae',
 'llama.dll':'018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef',
 'mtmd.dll':'603a1c44bc6db6a36e4d03548d0e1ac6f3018d627362c6c25f354ddf66280d7a'}


def match(actual, expected):
    return isinstance(actual,dict) and all(type(actual.get(k)) is type(v) and actual[k]==v for k,v in expected.items())


def planning_digest(value):
    # plan_masks.object_digest uses UTF-8 literals, unlike exporter canonical().
    return sha(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8'))


RANKING_METHODS = ('mass_gate','mean_selected_gate')
RECIPE_FIELDS = {'ranking_method','selection_recipe','selection_recipe_sha256'}


def typed_exact(actual, expected):
    """JSON equality without Python's True==1 / 0.0==0 equivalence."""
    if type(actual) is not type(expected): return False
    if isinstance(expected,dict):
        return set(actual)==set(expected) and all(typed_exact(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return len(actual)==len(expected) and all(typed_exact(a,b) for a,b in zip(actual,expected))
    return actual==expected


def expected_recipe(ranking_method):
    require(type(ranking_method) is str and ranking_method in RANKING_METHODS,'unsupported explicit ranking method')
    return {'schema_version':1,'ranking_method':ranking_method,'scope':'g_only_ablation_not_full_REAP',
        'includes_expert_output_norm':False,
        'score_formula':'sum_selected_gate' if ranking_method=='mass_gate' else 'sum_selected_gate / selected_count',
        'selected_count_definition':'top8 occurrences including zero gate weights','unobserved_score':0,
        'tie_break':'score_descending_then_expert_id_ascending',
        'pool_rule':'fixed_keep_else_actual_gate_mass_coverage_in_rank_order',
        'retained_mass_definition':'actual_sum_selected_gate_not_sum_of_ranking_scores',
        'pools':[128,96,64,32],'random_seed':20260713}


def selection_metadata(value,ranking_method,*,legacy_allowed):
    present=RECIPE_FIELDS & set(value)
    if not present:
        require(legacy_allowed and ranking_method=='mass_gate','missing ranking recipe; legacy supports mass_gate only')
        return None
    require(present==RECIPE_FIELDS,'partial ranking/recipe metadata rejected')
    recipe=expected_recipe(ranking_method)
    require(typed_exact(value['ranking_method'],ranking_method) and typed_exact(value['selection_recipe'],recipe),
            'ranking method/typed exact selection recipe mismatch')
    require(value['selection_recipe_sha256']==planning_digest(value['selection_recipe']), 'selection recipe hash mismatch')
    return value['selection_recipe_sha256']


def hash_value(value):
    require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'invalid SHA256')
    return value


class Snapshot:
    def __init__(self): self.files = {}

    @staticmethod
    def stamp(path):
        s = path.stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

    def raw(self,path):
        p = Path(path).resolve()
        require(p.suffix.lower() in ('.json','.jsonl'),'only explicit JSON/JSONL evidence may be read')
        before = self.stamp(p)
        require(before[2] <= MAX_BYTES,'evidence exceeds64MiB bound')
        with p.open('rb') as f: raw = f.read(MAX_BYTES+1)
        require(len(raw) == before[2] and self.stamp(p)==before,'input changed during read')
        digest = sha(raw)
        if p in self.files: require(self.files[p]==(before,digest),'input changed between reads')
        self.files[p] = (before,digest)
        return raw

    def obj(self,path):
        raw = self.raw(path)
        value = gate.decode_json(raw)
        require(isinstance(value,dict),'JSON object required')
        canonical(value)
        return value,sha(raw)

    def verify(self):
        for path,(stamp,digest) in list(self.files.items()):
            require(self.stamp(path)==stamp and sha(self.raw(path))==digest,'source evidence changed before publication')


def closed_run(snapshot,folder,expected_cases):
    root = Path(folder)
    m,mhash = snapshot.obj(root/'manifest.json')
    require(match(m,dict(completed=True,eligible_quality_evaluation=False,split='calibration',
            scope='untraced_quality_comparison',runtime_arm='candidate',trace_enabled=False,
            skip_chat_parsing=False,reasoning_preserve=False,model_sha256=gate.MODEL_SHA)),
            'run active/incomplete/ineligible configuration: closed calibration only')
    require('error_class' not in m and 'private_error' not in m,'coordinator error cannot qualify')
    end,_ = snapshot.obj(root/'shutdown.json'); pid,_ = snapshot.obj(root/'server.pid.json')
    require(end.get('stopped') is True and type(end.get('pid')) is int and end['pid']>0 and
            type(pid.get('pid')) is int and pid['pid']==end['pid'] and type(end.get('returncode')) is int,
            'shutdown/PID completion mismatch')
    require(match(m.get('config'),CONFIG),'coordinator frozen config mismatch')
    require(m.get('runtime_sha256')==RUNTIME,'candidate runtime fingerprint mismatch')
    sources=m.get('sources'); require(isinstance(sources,dict) and set(sources)==SOURCE_KEYS,'source identities required')
    for digest in sources.values(): hash_value(digest)
    require(m.get('coordinator_sha256')==sources['comparison_wrapper_sha256'],'wrapper source mismatch')
    for key in ('dataset_sha256','chat_template_sha256','parser_config_sha256','reasoning_config_sha256'): hash_value(m.get(key))
    require(m['parser_config_sha256']==sha(canonical({'jinja':True,'skip_chat_parsing':False})) and
            m['reasoning_config_sha256']==sha(canonical({'reasoning':'on','reasoning_preserve':False,'thinking':'template-default'})),
            'parser/reasoning controls mismatch')
    env=m.get('environment'); require(isinstance(env,dict),'recorded environment required')
    require(not any('TRACE' in k.upper() for k in env),'trace environment must be absent')
    require({k:v for k,v in env.items() if k!='QWEN36_REAP_MASK'}==ENV,'frozen numerical environment mismatch')
    command=m.get('command')
    require(isinstance(command,list) and all(isinstance(x,str) for x in command) and len(command)>1 and
            PureWindowsPath(command[0]).name=='llama-server.exe','recorded server command required')
    for flag,value in [('-m',gate.DEFAULT_MODEL),('-c','4096'),('--reasoning','on')]:
        require(command.count(flag)==1 and command.index(flag)+1<len(command) and command[command.index(flag)+1]==value,'server command drift')
    require(all(flag in command for flag in ('--jinja','--no-reasoning-preserve','--offline')) and
            '--skip-chat-parsing' not in command,'server parser/offline flags mismatch')
    ids=m.get('episode_ids')
    require(isinstance(ids,list) and len(ids)==expected_cases and all(isinstance(x,str) and x.startswith('calibration-') for x in ids)
            and len(set(ids))==expected_cases,'ordered unique calibration IDs required')
    # Only after final completion/shutdown checks may raw records be opened.
    p,_=snapshot.obj(root/'pilot/manifest.json'); summary,_=snapshot.obj(root/'pilot/summary.json')
    require(p.get('split')=='calibration' and p.get('protocol')=='native' and p.get('episode_ids')==ids and p.get('limit') is None
            and p.get('dataset_sha256')==m['dataset_sha256'] and p.get('runner_sha256')==sources['runner_sha256']
            and match(p.get('config'),CONFIG),'nested pilot manifest mismatch')
    props=p.get('server_props',{}); settings=props.get('default_generation_settings',{})
    require(type(settings.get('n_ctx')) is int and settings['n_ctx']==4096 and settings.get('params',{}).get('speculative.types')=='none',
            'server context/speculation mismatch')
    require(isinstance(props.get('model_path'),str) and PureWindowsPath(props['model_path'])==PureWindowsPath(gate.DEFAULT_MODEL),
            'server original model path mismatch')
    template=props.get('chat_template'); require(isinstance(template,str) and bool(template) and sha(template.encode())==m['chat_template_sha256'],
            'actual chat template hash mismatch')
    raw=snapshot.raw(root/'pilot/transcripts.jsonl')
    require(raw.endswith(b'\n'),'partial transcript EOF')
    lines=raw.splitlines(); require(len(lines)==expected_cases and all(lines),'incomplete transcript records')
    rows=[gate.decode_json(line) for line in lines]
    require([r.get('id') for r in rows]==ids,'raw record ID order/membership mismatch')
    for r in rows: validate_record(r)
    successes=sum(r['full_completion'] for r in rows)
    require(match(summary,dict(cases=expected_cases,split='calibration',real_actions_executed=0)),'nested summary mismatch')
    rate=summary.get('full_completion_rate')
    require(type(rate) in (int,float) and math.isfinite(rate) and abs(rate-successes/expected_cases)<1e-12,'summary success rate mismatch')
    require(match(m.get('outcomes'),dict(cases=expected_cases,successes=successes)) and
            type(m['outcomes'].get('full_completion_rate')) in (int,float) and m['outcomes']['full_completion_rate']==rate,
            'outer outcomes mismatch')
    return {'outer':m,'records':rows,'transcript_sha256':sha(raw),'manifest_sha256':mhash,'ids':ids,
            'successes':successes,'failed_ids':[r['id'] for r in rows if not r['full_completion']]}


def validate_record(r):
    require(match(r,dict(split='calibration',protocol='native',policy_version='v3',final_mode='tool',real_actions_executed=0))
            and type(r.get('full_completion')) is bool,'raw outcome protocol/type mismatch')
    require(r.get('error_class') in (None,'response_format','action_validation','final_validation','turn_limit'),
            'infrastructure/nonterminal outcome rejected')
    turns=r.get('turns'); actions=r.get('actions')
    require(isinstance(turns,list) and 1<=len(turns)<=13 and isinstance(actions,list),'missing actual turns/actions')
    selected_actions=[]; selected_finals=[]
    for i,t in enumerate(turns):
        require(type(t.get('index')) is int and t['index']==i,'turn order invalid')
        req=t.get('request',{}); pre=t.get('preflight',{}); resp=t.get('response',{})
        require(match(req,dict(model='local-pilot',stream=False,cache_prompt=True,temperature=0,top_k=1,top_p=1,min_p=0,
                seed=0,max_tokens=1024,tool_choice='auto',parallel_tool_calls=False)),'actual request config drift')
        require(not any(k in req for k in ('chat_template_kwargs','presence_penalty','repeat_penalty','grammar','verbose','return_tokens')),
                'raw/grammar/thinking override')
        n=pre.get('prompt_tokens_preflight'); usage=resp.get('usage',{})
        require(type(n) is int and 0<n<=3072 and match(pre,dict(server_ctx=4096,effective_budget=4096,output_reserved=1024))
                and type(usage.get('prompt_tokens')) is int and usage['prompt_tokens']==n,'preflight/context alignment mismatch')
        require(type(usage.get('completion_tokens')) is int and 0<=usage['completion_tokens']<=1024,'invalid completion count')
        choices=resp.get('choices'); require(isinstance(choices,list) and len(choices)==1,'missing model response')
        selection=t.get('selection')
        if selection:
            require(selection.get('kind') in ('action','final'),'selection kind invalid')
            require(choices[0].get('finish_reason') not in ('length','content_filter'),'truncated selection cannot be executed')
            calls=choices[0].get('message',{}).get('tool_calls')
            require(isinstance(calls,list) and len(calls)==1,'selected turn must have one native call')
            function=calls[0].get('function',{}); args=gate.decode_json(function.get('arguments',''))
            if selection['kind']=='action':
                v=selection.get('value',{}); require(function.get('name')==v.get('tool','').replace('.','_') and args==v.get('arguments'),'raw action/selection mismatch')
                selected_actions.append(v)
            else:
                require(function.get('name')=='final' and args==selection.get('value'),'raw final/selection mismatch')
                selected_finals.append(selection['value'])
    require(actions==selected_actions and selected_finals==([] if r.get('final') is None else [r['final']]),'raw record action/final mismatch')
    if r['full_completion']:
        require(r.get('error_class') is None and isinstance(r.get('final'),dict) and turns[-1].get('selection',{}).get('kind')=='final','inconsistent success')
    else: require(r.get('error_class') is not None,'failed record missing model-stage error')


def validate_plan(snapshot,path,masks,arms,ranking_method='mass_gate'):
    plan,phash=snapshot.obj(path)
    recipe_sha=selection_metadata(plan,ranking_method,legacy_allowed=True)
    candidates={}
    entries=plan.get('candidates',[])
    require(isinstance(entries,list),'candidate entries must be a list')
    if recipe_sha is not None:
        require(bool(entries),'recipe-bound plan requires candidate entries')
    for entry in entries:
        require(isinstance(entry,dict),'candidate entry must be an object')
        fields={'ranking_method','selection_recipe_sha256'} & set(entry)
        if recipe_sha is None:
            require(not fields and 'selection_recipe' not in entry,'legacy plan cannot silently acquire candidate recipe metadata')
            require(entry.get('method') in ('mass_gate','random_matched_pool'),'legacy candidate cannot declare a new ranking method')
        else:
            require(fields=={'ranking_method','selection_recipe_sha256'} and
                    typed_exact(entry['ranking_method'],ranking_method) and entry['selection_recipe_sha256']==recipe_sha,
                    'candidate ranking/recipe hash missing or mismatched')
            if 'selection_recipe' in entry:
                require(typed_exact(entry['selection_recipe'],plan['selection_recipe']),'candidate recipe content mismatch')
            require(type(entry.get('keep')) is int and entry['keep'] in plan['selection_recipe']['pools'] and
                    entry.get('method') in (ranking_method,'random_matched_pool'),'candidate keep/method mismatch')
        name=entry.get('file')
        require(isinstance(name,str) and name and Path(name).name==name and PureWindowsPath(name).name==name and name not in candidates,
                'candidate filenames invalid/duplicate')
        candidates[name]=entry
    if recipe_sha is not None:
        require(set(candidates)==set(plan.get('mask_sha256',{})),'recipe candidate inventory differs from plan mask hashes')
        for digest in plan['mask_sha256'].values(): hash_value(digest)
    for arm in arms.values():
        outer=arm['outer']
        require(not ({'selection_recipe','selection_recipe_sha256','legacy_selection_recipe'} & set(outer)),
                'unsupported outer recipe metadata: wrapper contract adds only ranking_method')
        if 'ranking_method' not in outer:
            require(ranking_method=='mass_gate' and recipe_sha is None,'recipe-bound experiment requires explicit arm ranking_method')
        else:
            require(typed_exact(outer['ranking_method'],ranking_method),'arm ranking_method differs from explicit experiment')
    require(match(plan,dict(gpu_screen_ready=True,quality_approval=False,policy_version='v3')),'plan not GPU-screen-ready V3/unapproved')
    internal=hash_value(plan.get('protocol_sha256')); proof=plan.get('provenance',{})
    require(proof.get('protocol_sha256')==internal and proof.get('gpu_screen_ready') is True,'plan provenance readiness mismatch')
    families=proof.get('family_counts',{})
    require(isinstance(families,dict) and len(families)==10 and all(isinstance(v,dict) and type(v.get('cases')) is int and
            v['cases']==5 and type(v.get('full_completions')) is int and 1<=v['full_completions']<=5 for v in families.values())
            and sum(v['full_completions'] for v in families.values())>=40 and proof.get('families_without_full_completion')==[],
            'plan family coverage/readiness proof mismatch')
    source=proof.get('recorded_source_sha256',{})
    require(set(source)=={'pilot_runner_sha256','coordinator_sha256'},'plan source provenance absent')
    baseline=arms['baseline']['outer']
    require(source['pilot_runner_sha256']==baseline['sources']['runner_sha256'] and
            source['coordinator_sha256']==baseline['sources']['capture_coordinator_sha256'],'selection source identity mismatch')
    expected_config=proof.get('expected_config'); actual_config=proof.get('full_source_config')
    expected_capture={k:CONFIG[k] for k in ('protocol','policy_version','final_mode','sampling_profile','seed','budget','max_output','max_turns','thinking')}
    expected_capture['diagnostic_raw']=True
    require(match(expected_config,expected_capture) and match(actual_config,expected_capture) and
            planning_digest(actual_config)==proof.get('full_source_config_sha256') and
            planning_digest({'expected_config':expected_config,'recorded_source_sha256':source})==internal,'planning config/digest mismatch')
    result={}
    for role,mask_path in masks.items():
        mask,digest=snapshot.obj(mask_path); selected=gate.validate_mask(mask,gate.PRODUCTION); keep=len(selected['0'])
        observed_recipe=selection_metadata(mask,ranking_method,legacy_allowed=recipe_sha is None)
        require(observed_recipe==recipe_sha,'mask recipe status/hash differs from plan; no silent upgrade')
        method=ranking_method if role=='ranked' else 'random_matched_pool'
        if recipe_sha is not None:
            entry=candidates.get(Path(mask_path).name,{})
            require(typed_exact(entry.get('keep'),keep) and typed_exact(entry.get('method'),method),
                    'selected mask does not match candidate entry')
        require(mask.get('method')==method and match(mask,dict(gpu_screen_ready=True,quality_approval=False,policy_version='v3',protocol_sha256=internal)),
                'mask method/planning provenance mismatch')
        require(plan.get('mask_sha256',{}).get(Path(mask_path).name)==digest,'mask exact plan bytes mismatch')
        m=arms[role]['outer']; ident=m.get('selected_mask')
        require(isinstance(ident,dict) and match(ident,dict(file_sha256=digest,canonical_sha256=sha(canonical(mask)),keep=keep,method=method)),
                'arm selected mask identity mismatch')
        require(PureWindowsPath(ident.get('path','')).name==Path(mask_path).name and m.get('mask_file_sha256')==digest and
                m.get('mask_sha256')==sha(canonical(mask)) and m.get('selection_plan_sha256')==phash,'arm mask/plan binding mismatch')
        require(m.get('environment',{}).get('QWEN36_REAP_MASK')==ident['path'],'actual recorded mask environment mismatch')
        result[role]={'keep':keep,'file_sha256':digest,'canonical_sha256':sha(canonical(mask))}
    require(result['ranked']['keep']==result['random']['keep'],'ranked/random must have same kept K')
    require(baseline.get('mask_sha256') is None and baseline.get('mask_file_sha256') is None and
            not any('MASK' in k.upper() for k in baseline['environment']),'baseline must be unmasked')
    require(baseline.get('selection_plan_sha256') in (None,phash),'baseline chosen plan mismatch')
    return phash,result,{'ranking_method':ranking_method,'selection_recipe':plan.get('selection_recipe'),
        'selection_recipe_sha256':recipe_sha,'ranking_metadata_status':'recipe_bound' if recipe_sha else 'legacy_mass_only',
        'capture_protocol_sha256':internal}


def explicit_review(snapshot,path,arm):
    review,digest=snapshot.obj(path)
    require(review.get('split')=='calibration' and review.get('transcript_sha256')==arm['transcript_sha256'], 'review raw transcript binding mismatch')
    require(isinstance(review.get('reviewer'),str) and bool(review['reviewer'].strip()),'truthful review identity required')
    require(review.get('unparsed_turns_reviewed') is True or review.get('method',{}).get('unparsed_turns_reviewed') is True,
            'explicit unparsed/case-level review required')
    cases=review.get('cases'); require(isinstance(cases,list) and [r.get('id') for r in cases]==arm['ids'],'all ordered cases must be reviewed')
    counts=Counter()
    def add(case,values):
        require(isinstance(values,list) and all(isinstance(v,str) and v in gate.CRITICAL_CATEGORIES for v in values),'explicit known critical list required')
        counts.update((case,v) for v in values)
    for c,r in zip(cases,arm['records']):
        require(c.get('reviewed') is True and isinstance(c.get('reviewer'),str) and bool(c['reviewer'].strip()),'case reviewer missing')
        require(c.get('transcript_record_sha256')==sha(canonical(r)) and type(c.get('reviewed_full_completion')) is bool
                and c['reviewed_full_completion']==r['full_completion'],'review record/completion mismatch; no rescoring')
        add(r['id'],c.get('case_critical_violations')); add(r['id'],c.get('final_critical_violations'))
        actions=c.get('action_reviews'); require(isinstance(actions,list) and len(actions)==len(r['actions']),'exhaustive action reviews required')
        for i,(a,v) in enumerate(zip(actions,r['actions'])):
            require(type(a.get('action_index')) is int and a['action_index']==i and a.get('action_sha256')==sha(canonical(v)),'review action index/hash mismatch')
            add(r['id'],a.get('critical_violations'))
    return counts,digest


def event_list(counts):
    return [{'id':case,'category':category,'occurrences':count} for (case,category),count in sorted(counts.items())]


def compare(baseline,ranked,random,plan_path,ranked_mask,random_mask,output,*,reviews=None,expected_cases=10,ranking_method='mass_gate'):
    expected_recipe(ranking_method)  # explicit supported selector, before input reads
    require(type(expected_cases) is int and expected_cases in (10,50),'explicit calibration screen size10 or50 only')
    output=Path(output); require(not output.exists() and output.parent.is_dir(),'output must be NEW with existing parent')
    snap=Snapshot(); arms={role:closed_run(snap,path,expected_cases) for role,path in [('baseline',baseline),('ranked',ranked),('random',random)]}
    base=arms['baseline']; common=('config','sources','runtime_sha256','dataset_sha256','chat_template_sha256','parser_config_sha256','reasoning_config_sha256','command')
    for role,arm in arms.items():
        require(arm['ids']==base['ids'],'arms must use identical ordered IDs')
        require(all(arm['outer'].get(k)==base['outer'].get(k) for k in common),'common runtime/source/protocol identity drift')
        env={k:v for k,v in arm['outer']['environment'].items() if k!='QWEN36_REAP_MASK'}
        require(env==base['outer']['environment'],'common numerical environment drift')
    plan_hash,mask_ids,recipe_identity=validate_plan(snap,plan_path,{'ranked':ranked_mask,'random':random_mask},arms,ranking_method)
    review_paths=reviews or {}; require(set(review_paths)<=set(arms),'unknown review role')
    counters={}; review_hashes={}
    for role,path in review_paths.items(): counters[role],review_hashes[role]=explicit_review(snap,path,arms[role])
    results={}; baseline_fail=set(base['failed_ids'])
    for role,arm in arms.items():
        failed=set(arm['failed_ids'])
        new=[case for case in base['ids'] if case in failed-baseline_fail]
        improved=[case for case in base['ids'] if case in baseline_fail-failed]
        reviewed=role in counters and 'baseline' in counters
        added=counters[role]-counters['baseline'] if reviewed else None
        results[role]={'cases':expected_cases,'successes':arm['successes'],'failed_ids':arm['failed_ids'],
            'newly_failed_ids_vs_baseline':new,'improved_ids_vs_baseline':improved,
            'new_failed_case_count':len(new),'new_failure_gate_passed':len(new)<=1,
            'critical_review_complete':role in counters,'critical_occurrences':event_list(counters[role]) if role in counters else None,
            'new_critical_occurrences_vs_baseline':event_list(added) if added is not None else None,
            'paired_screen_acceptable':(len(new)<=1 and not added) if reviewed else None,
            'transcript_sha256':arm['transcript_sha256'],'coordinator_manifest_sha256':arm['manifest_sha256']}
    all_reviewed=set(counters)==set(arms)
    decision='pending_explicit_critical_review' if not all_reviewed else (
        'ranked_calibration_screen_acceptable' if results['ranked']['paired_screen_acceptable'] else 'ranked_calibration_screen_rejected')
    report={'schema_version':1,'scope':'paired_calibration_only','decision':decision,'quality_approval':False,'export_approved':False,
        'heldout_approved':False,'next_k_approved':False,'no_automatic_next_k_selection':True,'not_a_throughput_benchmark':True,
        'model_sha256':gate.MODEL_SHA,'ordered_ids':base['ids'],'selection_plan_sha256':plan_hash,'masks':mask_ids,'arms':results,
        'review_sha256':review_hashes,'notes':['New failures and improvements are separate; never net-canceled.',
            'Random is a reviewed control, not a substitute for ranked evidence; control failures remain explicit.',
            'This10/50 calibration screen does not satisfy heldout20 gate or approve export/next K.'],
        'input_sha256':{str(path):digest for path,(_,digest) in snap.files.items()}}
    report.update(recipe_identity)
    snap.verify()
    data=canonical(report)+b'\n'; temp=None
    try:
        fd,name=tempfile.mkstemp(prefix='__screen_',suffix='.partial',dir=output.parent); temp=Path(name)
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        snap.verify(); os.link(temp,output)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','ranked','random','plan','ranked-mask','random-mask','output'): p.add_argument('--'+key,type=Path,required=True)
    for role in ('baseline','ranked','random'): p.add_argument('--'+role+'-review',type=Path)
    p.add_argument('--expected-cases',type=int,choices=(10,50),default=10)
    p.add_argument('--ranking-method',choices=RANKING_METHODS,default='mass_gate',help='Explicit calibration ranking experiment; never inferred from scores or files')
    a=p.parse_args(); reviews={role:getattr(a,role+'_review') for role in ('baseline','ranked','random') if getattr(a,role+'_review')}
    result=compare(a.baseline,a.ranked,a.random,a.plan,a.ranked_mask,a.random_mask,a.output,reviews=reviews,expected_cases=a.expected_cases,ranking_method=a.ranking_method)
    print(json.dumps({'decision':result['decision'],'quality_approval':False,'output':str(a.output)}))


if __name__=='__main__': main()
