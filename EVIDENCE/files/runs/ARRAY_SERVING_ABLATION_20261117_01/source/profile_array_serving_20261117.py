"""Frozen eight-policy serving ablation; authorized, hash-gated project models only."""
from pathlib import Path
import sys,json,argparse,time,gc,threading,subprocess,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import psutil,joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
import profile_exact_runtime as old
from wb_profile import checked_file,read_frame
from wb_research import WBPredictor,cube,mixture
from exact_runtime import ExactRuntime,FrozenEncoding
SOURCE=ROOT/'runs/EXACT_RUNTIME_20261020_01'
MANIFEST_SHA='b935207d33fb17b3fb74dbb977683f9c1e3444997c07b7fea955ff8aa7ce03f2'
POLICIES=['WV_native_full','WV_native_certified','WV_compact_full','WV_compact_certified','RF_native','RF_compact','LR_native','LR_compact']
INPUT_SOURCE=['scripts/profile_array_serving_20261117.py','src/exact_runtime.py','src/wb_scheduler.py','src/wb_research.py','scripts/profile_exact_runtime.py','src/multiscale_context.py','scripts/replay_multiscale_repair.py','docs/ARRAY_SERVING_ENHANCEMENT_PROTOCOL_20261117.md']

class CompactSingleton:
    def __init__(self,bundle,name,encoding):self.estimator=bundle['models'][name]['estimator'];self.name=name;self.encoding=encoding
    def predict_proba(self,X):
        Z=self.encoding.tree(X)
        if self.name=='logistic_regression':Z=self.encoding.linear(Z)
        return self.estimator.predict_proba(Z)
    def predict(self,X):return self.predict_proba(X).argmax(1)

class NativeFull:
    def __init__(self,bundle):self.bundle=bundle
    def predict_proba(self,X):return mixture(cube(self.bundle['models'],X),self.bundle['weights'])
    def predict(self,X):return self.predict_proba(X).argmax(1)

def load():
    assert e.sha256(SOURCE/'inputs_manifest.json')==MANIFEST_SHA
    freeze=json.loads((SOURCE/'inputs_manifest.json').read_text());complete=json.loads((SOURCE/'complete.json').read_text())
    assert complete['status']=='PASS_EXACT_RUNTIME' and complete['inputs_manifest_sha256']==MANIFEST_SHA
    for rec in freeze['configuration']['source_manifest']:assert e.sha256(ROOT/rec['path'])==rec['sha256']
    for rec in freeze['files']:checked_file(SOURCE,rec)
    spec=freeze['policies']['baseline25__native_shared'];model=checked_file(SOURCE,spec['model'])
    # Only this owned bundle is deserialized, after the full manifest gate above.
    bundle=joblib.load(model)
    assert bundle['view']=='legacy25_multiscale60' and bundle['case']=='balanced'
    files={r['path']:r for r in freeze['files']};schema=freeze['schemas']['baseline25']
    frames={s:read_frame(checked_file(SOURCE,files[f'inputs/baseline25_{s}.csv']),schema) for s in ['1','1000','replay']}
    with np.load(checked_file(SOURCE,files['inputs/expected.npz']),allow_pickle=False) as z:stored=dict(z)
    np.testing.assert_array_equal(frames['1000'].index,stored['row_ids']);np.testing.assert_array_equal(frames['replay'].index,stored['replay_row_ids'])
    enc=FrozenEncoding.from_models(bundle['models'])
    objects={'WV_native_full':NativeFull(bundle),'WV_native_certified':WBPredictor(bundle['models'],bundle['weights'],bundle['order'],'certified'),
       'WV_compact_full':ExactRuntime.from_bundle(bundle,'shared'),'WV_compact_certified':ExactRuntime.from_bundle(bundle,'certified'),
       'RF_native':bundle['models']['random_forest'],'RF_compact':CompactSingleton(bundle,'random_forest',enc),
       'LR_native':bundle['models']['logistic_regression'],'LR_compact':CompactSingleton(bundle,'logistic_regression',enc)}
    expected={}
    for policy in POLICIES:
        mode='random_forest' if policy.startswith('RF') else ('logistic_regression' if policy.startswith('LR') else 'native_shared')
        expected[policy]={s:stored[f'baseline25__{mode}__{s}'] for s in frames}
    return freeze,bundle,objects,frames,expected

def replay(freeze,obj,frames,expected,verify=False):
    return old.replay(SOURCE,freeze,'baseline25__native_shared',obj,frames,expected,verify=verify)

def worker(out,policy,index,workload):
    cfg=json.loads((out/'configuration_freeze.json').read_text());assert cfg['source_sha256']==e.sha256(Path(__file__))
    with threadpool_limits(limits=1):
        f,b,o,x,exp=load();obj=o[policy];gc.collect();proc=psutil.Process();base=proc.memory_info().rss;peaks=[base];stop=threading.Event();ready=threading.Event()
        def sample():
            peaks.append(proc.memory_info().rss);ready.set()
            while not stop.wait(.002):peaks.append(proc.memory_info().rss)
        t=threading.Thread(target=sample,daemon=True);t.start();ready.wait()
        try:
            if workload=='replay':replay(f,obj,x,exp[policy])
            else:np.testing.assert_array_equal(obj.predict(x['1000']),exp[policy]['1000'])
        finally:peaks.append(proc.memory_info().rss);stop.set();t.join()
    e.write_json(out/f'rss_{policy}_{workload}_{index}.json',dict(policy=policy,workload=workload,repeat=index,baseline_mib=base/2**20,peak_mib=max(peaks)/2**20,delta_mib=(max(peaks)-base)/2**20,samples=len(peaks),interval_seconds=.002,labels_match=True))

def main(out,smoke):
    assert not out.exists() and out.parent==(ROOT/'runs').resolve()
    assert e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv')=='53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a'
    out.mkdir();(out/'source').mkdir()
    sources=[]
    for rel in INPUT_SOURCE:
        p=ROOT/rel;shutil.copy2(p,out/'source'/p.name);sources.append(dict(path=rel,sha256=e.sha256(p)))
    cfg=dict(date='2026-11-17',seed=20261117,smoke=smoke,repeats=2 if smoke else 30,warmups=1 if smoke else 5,
       cold_rss_repeats=0 if smoke else 5,interval_seconds=.002,bootstrap=10000,policies=POLICIES,batches=[1,1000],
       source_manifest_sha256=MANIFEST_SHA,source_sha256=e.sha256(Path(__file__)),source_files=sources,
       threads=1,new_classifier_fits=0,new_data=False,stage='post-observation serving comparison; not independent predictive validation')
    e.write_json(out/'configuration_freeze.json',cfg);e.write_json(out/'environment.json',e.environment())
    rng=np.random.default_rng(cfg['seed']);timing=[];replicas=[];equivalence=[];posteriors={}
    with threadpool_limits(limits=1):
        f,b,o,x,exp=load()
        for key,X in x.items():
            refs={m:o[m+'_native'].predict_proba(X) for m in ['RF','LR']};refs['WV']=o['WV_native_full'].predict_proba(X)
            for policy,obj in o.items():
                np.testing.assert_array_equal(obj.predict(X),exp[policy][key]);name=policy.split('_')[0]
                if not policy.endswith('certified'):
                    p=obj.predict_proba(X);np.testing.assert_allclose(p,refs[name],atol=1e-12,rtol=0)
                    equivalence.append(dict(policy=policy,workload=key,rows=len(X),max_posterior_difference=float(np.max(np.abs(p-refs[name]))),labels_match=True))
                else:equivalence.append(dict(policy=policy,workload=key,rows=len(X),max_posterior_difference=None,labels_match=True))
                posteriors[f'{policy}__{key}__labels']=exp[policy][key]
            for name,p in refs.items():posteriors[f'{name}__{key}__probabilities']=p
            posteriors[key+'__row_ids']=X.index.to_numpy()
        np.savez_compressed(out/'expected_and_native_posteriors.npz',**posteriors)
        e.write_json(out/'equivalence.json',equivalence)
        for policy,obj in o.items():
            for key in ['1','1000']:
                for _ in range(cfg['warmups']):np.testing.assert_array_equal(obj.predict(x[key]),exp[policy][key])
        for rep in range(cfg['repeats']):
            for position,policy in enumerate(rng.permutation(POLICIES)):
                for key in ['1','1000']:
                    start=time.perf_counter();p=o[policy].predict(x[key]);elapsed=time.perf_counter()-start
                    np.testing.assert_array_equal(p,exp[policy][key]);timing.append(dict(policy=policy,batch=int(key),repeat=rep,position=position,seconds=elapsed))
        pd.DataFrame(timing).to_csv(out/'latency_raw.csv',index=False)
        for policy,obj in o.items():
            replay(f,obj,x,exp[policy],verify=True)
            for _ in range(cfg['warmups']):replay(f,obj,x,exp[policy])
        for rep in range(cfg['repeats']):
            for position,policy in enumerate(rng.permutation(POLICIES)):
                cpu=time.process_time();row=replay(f,o[policy],x,exp[policy]);row['policy']=policy
                replicas.append(dict(**row,repeat=rep,position=position,cpu_seconds=time.process_time()-cpu))
            pd.DataFrame(replicas).to_csv(out/'replay_raw.csv',index=False)
            print('PAIRED_EIGHT_POLICY_REPLAY',rep+1,'/',cfg['repeats'],flush=True)
    # Cold workers execute sequentially, never competing with timed policies.
    for policy in POLICIES:
        for workload in ['feature','replay']:
            for i in range(cfg['cold_rss_repeats']):
                with (out/f'rss_{policy}_{workload}_{i}.log').open('x') as log:
                    subprocess.run([sys.executable,'-B',str(Path(__file__)),'--output',str(out),'--worker',policy,'--index',str(i),'--workload',workload],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COLD_RSS_COMPLETE',policy,flush=True)
    if not smoke:pd.DataFrame([json.loads(p.read_text()) for p in out.glob('rss_*.json')]).to_csv(out/'memory_raw.csv',index=False)
    for rec in sources:assert e.sha256(ROOT/rec['path'])==rec['sha256']
    records=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    e.write_json(out/'catalog.json',records);e.write_json(out/'complete.json',dict(status='PASS_SERVING_SMOKE' if smoke else 'PASS_SERVING_ABLATION',new_classifier_fits=0,policies=8,feature_timings=len(timing),replays=len(replicas),cold_memory_runs=16*cfg['cold_rss_repeats'],all_labels_match=True,catalog_sha256=e.sha256(out/'catalog.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--worker');p.add_argument('--index',type=int);p.add_argument('--workload');a=p.parse_args()
    if a.worker:worker(a.output.resolve(),a.worker,a.index,a.workload)
    else:main(a.output.resolve(),a.smoke)
