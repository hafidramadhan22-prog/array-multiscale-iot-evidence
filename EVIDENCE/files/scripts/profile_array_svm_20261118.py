"""One-session, matched label-serving measurements for standalone SVM and references."""
from pathlib import Path
import sys,argparse,time,gc,threading,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_svm_20261118 import read,npz,stage_inputs
from evaluate_array_svm_20261118 import load_svm
from stress_array_svm_20261118 import read_features,TRAIN,EVAL
from profile_array_serving_20261117 import load as load_reference

POLICIES=['SVM_default','SVM_G_selected','WV_native_certified','WV_compact_certified','RF_native','RF_compact','LR_native','LR_compact']

def load_objects(out):
    cfg=read(out/'configuration_freeze.json')
    assert cfg['training_catalog_sha256']==e.sha256(TRAIN/'catalog.json') and cfg['source_sha256']==e.sha256(Path(__file__))
    row=next(r for r in read(TRAIN/'catalog.json') if r['path']=='stages/temporal_seed42/models_manifest.json')
    assert e.sha256(TRAIN/row['path'])==row['sha256']
    freeze=read(TRAIN/'stages/temporal_seed42/selection_before_V.json')
    _,_,objects,_,_=load_reference()
    objects={k:objects[k] for k in POLICIES if k in objects}
    objects.update(SVM_default=load_svm(TRAIN,'temporal_seed42',freeze['default']),SVM_G_selected=load_svm(TRAIN,'temporal_seed42',freeze['selected']))
    schema=read(out/'schema.json');frames={}
    for size in [1,1000]:
        f=out/'inputs'/f'batch{size}.csv';rec=next(r for r in cfg['inputs'] if r['path']==f.relative_to(out).as_posix());assert e.sha256(f)==rec['sha256']
        frames[size]=read_features(f,schema)
    expected=npz(out/'expected.npz')
    assert e.sha256(out/'expected.npz')==cfg['expected_sha256']
    return objects,frames,expected

def worker(out,policy,rep):
    with threadpool_limits(limits=1):
        objects,frames,expected=load_objects(out);gc.collect();process=psutil.Process();base=process.memory_info().rss;peaks=[base];stop=threading.Event();ready=threading.Event()
        def sample():
            peaks.append(process.memory_info().rss);ready.set()
            while not stop.wait(.002):peaks.append(process.memory_info().rss)
        t=threading.Thread(target=sample,daemon=True);t.start();ready.wait()
        try:np.testing.assert_array_equal(objects[policy].predict(frames[1000]),expected[policy+'__1000'])
        finally:peaks.append(process.memory_info().rss);stop.set();t.join()
    e.write_json(out/f'rss_{policy}_{rep}.json',dict(policy=policy,repeat=rep,batch=1000,baseline_mib=base/2**20,peak_mib=max(peaks)/2**20,delta_mib=(max(peaks)-base)/2**20,samples=len(peaks),interval_seconds=.002,labels_match=True))

def main(out):
    assert not out.exists() and read(EVAL/'complete.json')['status']=='PASS_SVM_MATCHED_EVALUATION'
    out.mkdir();(out/'inputs').mkdir();(out/'source').mkdir()
    sources=['scripts/profile_array_svm_20261118.py','scripts/stress_array_svm_20261118.py','scripts/evaluate_array_svm_20261118.py','scripts/profile_array_serving_20261117.py','src/exact_runtime.py','src/wb_scheduler.py','src/wb_research.py','docs/ARRAY_SVM_COMPARATOR_PROTOCOL_20261118.md']
    for rel in sources:shutil.copy2(ROOT/rel,out/'source'/Path(rel).name)
    with threadpool_limits(limits=1):
        X,y,raw,_,parts,_,_,_,_=stage_inputs('temporal_seed42');v=parts[3]
        positions=v[e.sample_positions(y[v],1000,20261118)];frames={1000:X.iloc[positions].copy(),1:X.iloc[positions[:1]].copy()}
        plan=read(TRAIN/'stages/temporal_seed42/training_plan.json')
        schema={k:plan[k] for k in ['numeric','categorical','feature_order']};e.write_json(out/'schema.json',schema)
        inputs=[]
        for size,x in frames.items():
            f=out/'inputs'/f'batch{size}.csv';x.to_csv(f,index_label='row_id');inputs.append(dict(path=f.relative_to(out).as_posix(),sha256=e.sha256(f),bytes=f.stat().st_size))
        cfg=dict(date='2026-11-18',seed=20261118,training_catalog_sha256=e.sha256(TRAIN/'catalog.json'),evaluation_catalog_sha256=e.sha256(EVAL/'catalog.json'),source_sha256=e.sha256(Path(__file__)),sources=[dict(path=s,sha256=e.sha256(ROOT/s)) for s in sources],inputs=inputs,policies=POLICIES,warmups=5,repeats=30,cold_rss_repeats=5,interval_seconds=.002,threads=1,new_fits=0,workload='prepared V rows; labels only; no raw feature extraction or replay',timing_not_pooled_with_previous_sessions=True)
        # Load and replay the already frozen models before any measured calls.
        _,_,objects,_,_=load_reference();freeze=read(TRAIN/'stages/temporal_seed42/selection_before_V.json')
        objects.update(SVM_default=load_svm(TRAIN,'temporal_seed42',freeze['default']),SVM_G_selected=load_svm(TRAIN,'temporal_seed42',freeze['selected']))
        gold=npz(EVAL/'stages/temporal_seed42/evaluation.npz');lookup={int(r):j for j,r in enumerate(gold['row_ids'])};expected={}
        for size in [1,1000]:
            reloaded=read_features(out/'inputs'/f'batch{size}.csv',schema);frames[size]=reloaded;ids=[lookup[int(r)] for r in reloaded.index];expected['row_ids_'+str(size)]=reloaded.index.to_numpy()
            for policy in POLICIES:
                key='WV' if policy.startswith('WV') else 'RF' if policy.startswith('RF') else 'LR' if policy.startswith('LR') else policy
                pred=objects[policy].predict(reloaded);np.testing.assert_array_equal(pred,gold[key][ids]);expected[policy+'__'+str(size)]=pred
        np.savez_compressed(out/'expected.npz',**expected);cfg['expected_sha256']=e.sha256(out/'expected.npz')
        e.write_json(out/'configuration_freeze.json',cfg);e.write_json(out/'environment.json',e.environment())
        rng=np.random.default_rng(20261118);rows=[]
        for policy in POLICIES:
            for size in [1,1000]:
                for _ in range(5):np.testing.assert_array_equal(objects[policy].predict(frames[size]),expected[policy+'__'+str(size)])
        for rep in range(30):
            for order,policy in enumerate(rng.permutation(POLICIES)):
                for size in [1,1000]:
                    cpu=time.process_time();start=time.perf_counter();pred=objects[policy].predict(frames[size]);seconds=time.perf_counter()-start;cpu=time.process_time()-cpu
                    np.testing.assert_array_equal(pred,expected[policy+'__'+str(size)]);rows.append(dict(policy=policy,batch=size,repeat=rep,position=order,seconds=seconds,cpu_seconds=cpu))
            pd.DataFrame(rows).to_csv(out/'latency_raw.csv',index=False)
        print('MATCHED_LABEL_LATENCY_COMPLETE',len(rows),flush=True)
    # No concurrent training or timing while these cold subprocesses run.
    for policy in POLICIES:
        for rep in range(5):
            with (out/f'rss_{policy}_{rep}.log').open('x') as f:subprocess.run([sys.executable,'-B',str(Path(__file__)),'--output',str(out),'--worker',policy,'--repeat',str(rep)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('SVM_COLD_RSS_COMPLETE',policy,flush=True)
    pd.DataFrame([read(p) for p in out.glob('rss_*.json')]).to_csv(out/'memory_raw.csv',index=False)
    for r in cfg['sources']:assert e.sha256(ROOT/r['path'])==r['sha256']
    records=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=e.sha256(f)) for f in sorted(out.rglob('*')) if f.is_file()]
    e.write_json(out/'catalog.json',records);e.write_json(out/'complete.json',dict(status='PASS_SVM_MATCHED_LABEL_SERVING',latency_calls=len(rows),cold_RSS_calls=40,new_fits=0,all_labels_match_saved_evaluation=True,catalog_sha256=e.sha256(out/'catalog.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker');p.add_argument('--repeat',type=int);a=p.parse_args()
    if a.worker:worker(a.output.resolve(),a.worker,a.repeat)
    else:main(a.output.resolve())
