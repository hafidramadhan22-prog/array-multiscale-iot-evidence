"""Same-session feature-input latency and cold inference delta RSS, never fitting."""
from pathlib import Path
import argparse,gc,json,os,shutil,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from multiscale_context import repair_frame
from run_wb_research import load_data
from run_context_repair import check_record
from wb_research import view_data,WBPredictor
from wb_profile import file_record,read_frame,checked_file
from analyze_context_repair import evaluation,labels

MODES=['shared','certified','logistic_regression','random_forest']

def bundle_record(run,cfg,view):
    folder=ROOT/cfg['source_run'] if view.startswith('baseline') else run
    done=json.loads((folder/'quality_complete.json').read_text())
    assert e.sha256(folder/'models_manifest.json')==done['models_manifest_sha256']
    actual=('legacy25_symmetric' if view=='baseline25' else 'legacy29_symmetric') if view.startswith('baseline') else view
    matches=[r for r in json.loads((folder/'models_manifest.json').read_text()) if r['stage']=='temporal_seed42' and r['view']==actual and r['case']=='balanced']
    assert len(matches)==1;check_record(folder,matches[0])
    return folder,matches[0],actual

def prepare(run):
    cfg=json.loads((run/'configuration_freeze.json').read_text())
    assert json.loads((run/'verification_quality.json').read_text())['status']=='PASS_CONTEXT_REPAIR_QUALITY'
    dest=run/'profiling'
    if dest.exists():raise ValueError('Profiling folder already exists; inspect rather than overwrite')
    dest.mkdir();(dest/'inputs').mkdir();(dest/'models').mkdir();(dest/'source').mkdir()
    configuration=dict(seed=20261019,batches=[1,1000],warmups=5,blocks=3,repeats=15,memory_runs=3,
        memory_interval_seconds=.002,views=['baseline25','baseline29']+cfg['views'],modes=MODES,
        treatment='balanced',fit_seed=42,scope='Precomputed feature-input prediction only; no raw decode/context/legacy/ratio extraction or SHAP included',
        authorization='Current user request to repair experiments; only these project-owned checksum-verified bundles are deserialized',
        baseline_models_retrained=False)
    e.write_json(dest/'configuration_freeze.json',configuration)
    for rel in ['scripts/profile_multiscale_repair.py','src/multiscale_context.py','src/wb_research.py','src/wb_scheduler.py','src/wb_profile.py']:
        shutil.copy2(ROOT/rel,dest/'source'/Path(rel).name)
    raw,y,holdout,data=load_data(cfg);first,_=evaluation(run,cfg,'temporal_seed42','baseline25','balanced')
    ids=first['row_ids'];chosen=ids[np.random.default_rng(20261019).permutation(len(ids))[:1000]]
    assert len(chosen)==1000 and not np.intersect1d(chosen,holdout).size
    expected={};records=[];schemas={};policies={}
    for view in configuration['views']:
        X,num,cat=(view_data(raw,data,'legacy25_symmetric' if view=='baseline25' else 'legacy29_symmetric')
                   if view.startswith('baseline') else repair_frame(raw,data,view))
        schemas[view]=dict(numeric=num,categorical=cat,columns=list(X.columns))
        for batch in configuration['batches']:
            path=dest/'inputs'/f'{view}_{batch}.csv';frame=X.loc[chosen[:batch]]
            frame.to_csv(path,index=True,index_label='row_id',float_format='%.17g')
            loaded=read_frame(path,schemas[view]);pd.testing.assert_frame_equal(loaded,frame,check_dtype=False,
                check_names=False,check_exact=False,rtol=1e-14,atol=0)
            records.append(file_record(run,path))
        owner,rec,actual=bundle_record(run,cfg,view)
        # Provenance checked before every deserialization, with no unknown joblib accepted.
        bundle=joblib.load(owner/rec['path'])
        assert bundle['view']==actual and bundle['case']=='balanced' and set(bundle['models'])==set(e.NAMES)
        ev,sel=evaluation(run,cfg,'temporal_seed42',view,'balanced')
        np.testing.assert_allclose(bundle['weights'],sel['weights'],rtol=0,atol=0)
        lookup={int(row):i for i,row in enumerate(ev['row_ids'])}
        for mode in MODES:
            policy=f'{view}__{mode}';path=dest/'models'/f'{policy}.joblib'
            obj=bundle if mode in ['shared','certified'] else bundle['models'][mode]
            joblib.dump(obj,path,compress=1);model_record=file_record(run,path);records.append(model_record)
            policies[policy]=dict(view=view,mode=mode,model=model_record,origin_folder=str(owner.relative_to(ROOT)),origin_model=rec)
            prediction=labels(ev,sel,'weighted_voting' if mode in ['shared','certified'] else mode)
            for batch in configuration['batches']:expected[f'{policy}__{batch}']=prediction[[lookup[int(row)] for row in chosen[:batch]]]
    np.savez_compressed(dest/'inputs/expected.npz',row_ids=chosen,**expected);records.append(file_record(run,dest/'inputs/expected.npz'))
    freeze=dict(configuration=configuration,schemas=schemas,policies=policies,files=records,
                source_sha256=e.sha256(Path(__file__)),quality_manifest_sha256=e.sha256(run/'models_manifest.json'))
    e.write_json(dest/'inputs_manifest.json',freeze);e.write_json(dest/'environment.json',e.environment());return freeze

def load_policy(run,freeze,policy):
    spec=freeze['policies'][policy];mode=spec['mode']
    model=joblib.load(checked_file(run,spec['model']))
    predictor=WBPredictor(model['models'],model['weights'],model['order'],mode) if mode in ['shared','certified'] else model
    matrices={}
    records={z['path']:z for z in freeze['files']}
    for batch in freeze['configuration']['batches']:
        rel=f"profiling/inputs/{spec['view']}_{batch}.csv"
        matrices[batch]=read_frame(checked_file(run,records[rel]),freeze['schemas'][spec['view']])
    rel='profiling/inputs/expected.npz';path=checked_file(run,records[rel])
    with np.load(path,allow_pickle=False) as z:expected={batch:z[f'{policy}__{batch}'] for batch in matrices}
    return predictor,matrices,expected

def worker(run,policy,index):
    freeze=json.loads((run/'profiling/inputs_manifest.json').read_text())
    assert freeze['source_sha256']==e.sha256(Path(__file__))
    with threadpool_limits(limits=1):
        predictor,matrices,expected=load_policy(run,freeze,policy);X=matrices[1000]
        gc.collect();process=psutil.Process();baseline=process.memory_info().rss;peaks=[baseline]
        stop=threading.Event();ready=threading.Event()
        def sample():
            peaks.append(process.memory_info().rss);ready.set()
            while not stop.wait(.002):peaks.append(process.memory_info().rss)
        thread=threading.Thread(target=sample,daemon=True);thread.start();ready.wait()
        pred=predictor.predict(X);peaks.append(process.memory_info().rss);stop.set();thread.join()
        np.testing.assert_array_equal(pred,expected[1000]);peak=max(peaks)
    e.write_json(run/f'profiling/rss_{policy}_{index}.json',dict(policy=policy,run=index,
        baseline_rss_mib=baseline/2**20,peak_rss_mib=peak/2**20,delta_peak_inference_mib=(peak-baseline)/2**20,
        samples=len(peaks),batch=1000,labels_match=True,interval_seconds=.002,cold_first_prediction=True))

def run_profile(run):
    run=run.resolve()
    freeze=prepare(run);cfg=freeze['configuration'];rows=[]
    with threadpool_limits(limits=1):
        for policy in freeze['policies']:
            predictor,matrices,expected=load_policy(run,freeze,policy)
            for batch,X in matrices.items():
                for _ in range(cfg['warmups']):np.testing.assert_array_equal(predictor.predict(X),expected[batch])
                for block in range(cfg['blocks']):
                    for repeat in range(cfg['repeats']):
                        t=time.perf_counter();pred=predictor.predict(X);elapsed=time.perf_counter()-t
                        np.testing.assert_array_equal(pred,expected[batch])
                        rows.append(dict(policy=policy,batch=batch,block=block,repeat=repeat,seconds=elapsed,
                                         rows_per_second=batch/elapsed,labels_match=True))
            print('TIMED',policy,flush=True)
    pd.DataFrame(rows).to_csv(run/'profiling/latency_raw.csv',index=False)
    memory=[]
    for policy in freeze['policies']:
        for index in range(cfg['memory_runs']):
            with (run/f'profiling/rss_{policy}_{index}.log').open('x',encoding='utf8') as stream:
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run',str(run),'--worker',policy,'--index',str(index)],
                                 cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if p.returncode:raise RuntimeError(f'Memory gate failed {policy}/{index}')
            memory.append(json.loads((run/f'profiling/rss_{policy}_{index}.json').read_text()))
        print('RSS',policy,flush=True)
    memory=pd.DataFrame(memory);memory.to_csv(run/'profiling/memory_raw.csv',index=False)
    np.testing.assert_allclose(memory.peak_rss_mib-memory.baseline_rss_mib,memory.delta_peak_inference_mib,atol=1e-12,rtol=0)
    latency=pd.DataFrame(rows).groupby(['policy','batch']).agg(seconds_mean=('seconds','mean'),seconds_median=('seconds','median'),
        seconds_std=('seconds','std'),timings=('seconds','size')).reset_index()
    latency.to_csv(run/'profiling/latency_summary.csv',index=False)
    summary=memory.groupby('policy')[['baseline_rss_mib','peak_rss_mib','delta_peak_inference_mib']].mean().reset_index()
    summary['artifact_bytes']=[freeze['policies'][p]['model']['bytes'] for p in summary.policy]
    summary.to_csv(run/'profiling/memory_summary.csv',index=False)
    for rec in freeze['files']:checked_file(run,rec)
    e.write_json(run/'profiling/complete.json',dict(status='PASS_REPAIR_FEATURE_INPUT_PROFILE',policies=len(freeze['policies']),
        timings=len(rows),cold_memory_runs=len(memory),labels_match=True,feature_extraction_included=False,
        inputs_manifest_sha256=e.sha256(run/'profiling/inputs_manifest.json'),source_sha256=freeze['source_sha256']))
    print('PROFILING COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--worker');p.add_argument('--index',type=int)
    args=p.parse_args()
    if args.worker:worker(args.run,args.worker,args.index)
    else:run_profile(args.run)
