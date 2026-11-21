"""Finite paired native/compact runtime audit; owned bundles only, no fitting."""
from pathlib import Path
import argparse,gc,json,shutil,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from exact_runtime import ExactRuntime
from wb_research import cube,mixture,WBPredictor
from multiscale_context import repair_frame as old_frame,CONTEXT_NAMES,MultiscaleRing
from time_context import repair_frame as time_frame,TIME_COLUMNS,AGE_INDEX,HostTimeRing
from run_wb_research import load_data
from run_time_repair import check_record,read_npz
from analyze_time_repair import evaluation,labels
from wb_profile import file_record,checked_file,read_frame,LocalSource
from replay_multiscale_repair import multiscale60_query
from sbfe import iter_pcap,role_for

MODES=['native_shared','native_certified','compact_shared','compact_certified','logistic_regression','random_forest']


def frame_for(raw,data,view):
    return old_frame(raw,data,'legacy25_multiscale60') if view=='baseline25' else time_frame(raw,data,view)


def prepare(output,quality):
    if output.exists() or output.parent!=(ROOT/'runs').resolve():raise ValueError('New run required')
    assert json.loads((quality/'verification_quality.json').read_text())['status']=='PASS_CONTEXT_REPAIR_QUALITY'
    cfg=json.loads((quality/'configuration_freeze.json').read_text())
    chosen=json.loads((quality/'stages/temporal_seed42/candidate_selection_before_V.json').read_text())['selected']['balanced']['wb_only']['view']
    views=['baseline25']+([] if chosen=='baseline25' else [chosen])
    output.mkdir();(output/'source').mkdir();(output/'inputs').mkdir();(output/'models').mkdir();(output/'replay').mkdir()
    source=[]
    for rel in ['scripts/profile_exact_runtime.py','src/exact_runtime.py','src/time_context.py','src/multiscale_context.py',
                'src/wb_research.py','src/wb_scheduler.py','docs/TIME_RUNTIME_PROTOCOL_20261020.md','docs/TIME_RUNTIME_LITERATURE_20261020.md']:
        shutil.copy2(ROOT/rel,output/'source'/Path(rel).name);source.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    configuration=dict(quality_run=str(quality.relative_to(ROOT)),views=views,modes=MODES,seed=20261020,
        batches=[1,1000],warmups=5,blocks=3,repeats=15,memory_runs=3,memory_interval_seconds=.002,
        replay_timed_repeats=5,posterior_tolerance=1e-12,no_fitting=True,source_manifest=source,
        limitation='Laptop; supplied legacy features. Replay adds decode, retained context, framing and prediction, not raw legacy extraction or SHAP.')
    e.write_json(output/'configuration_freeze.json',configuration);e.write_json(output/'environment.json',e.environment())
    raw,y,holdout,data=load_data(cfg);records=[];schemas={};policies={};expected={};equivalence=[]
    old=ROOT/'runs/WB_RESEARCH_FULL_20261019_01'
    cat={r['path']:r for r in json.loads((old/'artifacts_sha256.json').read_text())}
    check_record(old,cat['profiling/workload.json']);work=json.loads((old/'profiling/workload.json').read_text())
    for item in work['files']:
        rec=cat[item['file']];check_record(old,rec)
        assert rec['sha256']==item['sha256'] and rec['bytes']==item['bytes']
        shutil.copy2(old/item['file'],output/'replay'/Path(item['file']).name)
        records.append(file_record(output,output/'replay'/Path(item['file']).name))
    e.write_json(output/'replay/workload.json',work);records.append(file_record(output,output/'replay/workload.json'))
    replay_ids=np.array([int(item['targets'][str(i)]['row_id']) for item in work['files'] for i in sorted(map(int,item['targets']))])
    first,_=evaluation(quality,cfg,'temporal_seed42','baseline25','balanced')
    chosen_ids=first['row_ids'][np.random.default_rng(20261020).permutation(len(first['row_ids']))[:1000]]
    assert len(chosen_ids)==1000 and not np.intersect1d(np.r_[chosen_ids,replay_ids],holdout).size
    for view in views:
        owner=ROOT/cfg['source_run'] if view=='baseline25' else quality
        actual='legacy25_multiscale60' if view=='baseline25' else view
        done=json.loads((owner/'quality_complete.json').read_text())
        assert done['models_manifest_sha256']==e.sha256(owner/'models_manifest.json')
        matches=[r for r in json.loads((owner/'models_manifest.json').read_text()) if (r['stage'],r['view'],r['case'])==('temporal_seed42',actual,'balanced')]
        assert len(matches)==1;rec=matches[0];check_record(owner,rec)
        bundle=joblib.load(owner/rec['path']) # authorized owned SHA-verified model
        assert bundle['view']==actual and bundle['case']=='balanced'
        fast=ExactRuntime.from_bundle(bundle);X,num,categorical=frame_for(raw,data,view)
        ev,sel=evaluation(quality,cfg,'temporal_seed42',view,'balanced');np.testing.assert_array_equal(bundle['weights'],sel['weights'])
        maxdiff=0.;seen=0
        for start in range(0,len(X),1000):
            chunk=X.iloc[start:start+1000];Z=fast.encoding.tree(chunk);L=fast.encoding.linear(Z)
            for got,wanted in [(Z,bundle['models']['decision_tree']['preprocessor'].transform(chunk)),
                               (L,bundle['models']['logistic_regression']['preprocessor'].transform(chunk))]:
                assert sparse.issparse(got)==sparse.issparse(wanted)
                if sparse.issparse(got):
                    np.testing.assert_array_equal(got.indices,wanted.indices);np.testing.assert_array_equal(got.indptr,wanted.indptr)
                    np.testing.assert_array_equal(got.data,wanted.data)
                else:np.testing.assert_array_equal(got,wanted)
            a=cube(bundle['models'],chunk);b=fast.expert_cube(chunk);delta=float(np.max(np.abs(a-b)));maxdiff=max(maxdiff,delta)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
            native=mixture(a,bundle['weights']).argmax(1)
            np.testing.assert_array_equal(native,fast.predict(chunk))
            np.testing.assert_array_equal(native,WBPredictor(bundle['models'],bundle['weights'],bundle['order'],'certified').predict(chunk));seen+=len(chunk)
        equivalence.append(dict(view=view,rows=seen,encoded_values_bit_identical=True,csr_structure_identical=True,
            expert_probability_max_abs_delta=maxdiff,full_and_certified_labels_identical=True,origin_folder=str(owner.relative_to(ROOT)),origin_model=rec))
        schemas[view]=dict(numeric=num,categorical=categorical,columns=list(X.columns))
        for suffix,ids in [('1',chosen_ids[:1]),('1000',chosen_ids),('replay',replay_ids)]:
            path=output/'inputs'/f'{view}_{suffix}.csv';X.loc[ids].to_csv(path,index_label='row_id',float_format='%.17g')
            pd.testing.assert_frame_equal(X.loc[ids],read_frame(path,schemas[view]),check_names=False,check_dtype=False,rtol=1e-14,atol=0)
            records.append(file_record(output,path))
        lookup={int(r):i for i,r in enumerate(ev['row_ids'])}
        for mode in MODES:
            policy=f'{view}__{mode}';path=output/'models'/f'{policy}.joblib'
            if mode.startswith('compact'):
                obj=ExactRuntime.from_bundle(bundle,'certified' if mode.endswith('certified') else 'shared')
            elif mode.startswith('native'):obj=bundle
            else:obj=bundle['models'][mode]
            joblib.dump(obj,path,compress=1);model_record=file_record(output,path);records.append(model_record)
            policies[policy]=dict(view=view,mode=mode,model=model_record,origin_folder=str(owner.relative_to(ROOT)),origin_model=rec)
            pred=labels(ev,sel,mode if mode in e.NAMES else 'weighted_voting')
            for suffix,ids in [('1',chosen_ids[:1]),('1000',chosen_ids),('replay',replay_ids)]:
                expected[f'{policy}__{suffix}']=pred[[lookup[int(r)] for r in ids]]
    np.savez_compressed(output/'inputs/expected.npz',row_ids=chosen_ids,replay_row_ids=replay_ids,**expected)
    records.append(file_record(output,output/'inputs/expected.npz'))
    e.write_json(output/'equivalence.json',dict(status='PASS_EXACT_RUNTIME_INPUTS',checks=equivalence,labels_tolerance=0,posterior_tolerance=1e-12))
    freeze=dict(configuration=configuration,schemas=schemas,policies=policies,files=records,source_sha256=e.sha256(Path(__file__)))
    e.write_json(output/'inputs_manifest.json',freeze);return freeze


def load_policy(run,freeze,policy):
    spec=freeze['policies'][policy];mode=spec['mode'];model=joblib.load(checked_file(run,spec['model']))
    predictor=(WBPredictor(model['models'],model['weights'],model['order'],mode.split('_')[1])
               if mode.startswith('native') else model)
    files={r['path']:r for r in freeze['files']};view=spec['view']
    frames={suffix:read_frame(checked_file(run,files[f'inputs/{view}_{suffix}.csv']),freeze['schemas'][view]) for suffix in ['1','1000','replay']}
    exp=read_npz(checked_file(run,files['inputs/expected.npz']))
    expected={suffix:exp[f'{policy}__{suffix}'] for suffix in frames}
    return predictor,frames,expected


def replay(run,freeze,policy,predictor,frames,expected,verify=False):
    view=freeze['policies'][policy]['view'];work=json.loads((run/'replay/workload.json').read_text())
    history=list(CONTEXT_NAMES)
    if view!='baseline25':
        history+=list(TIME_COLUMNS if not view.endswith('age_context66') else np.asarray(TIME_COLUMNS)[AGE_INDEX])
    X=frames['replay'];base=X.drop(columns=history);pred=[];got_ids=[];seen=0;update=query=model=0.;capacity=0
    start=time.perf_counter()
    for item in work['files']:
        ring=MultiscaleRing() if view=='baseline25' else HostTimeRing()
        for index,stamp,_,_,packet in iter_pcap(LocalSource(),run/'replay'/Path(item['file']).name):
            seen+=1;target=item['targets'].get(str(index))
            if target is not None:
                rid=int(target['row_id']);tick=time.perf_counter()
                if view=='baseline25':context=multiscale60_query(ring,stamp,packet.src,packet.dst)
                else:
                    host,_,timing=ring.query_time(stamp,packet.src,packet.dst)
                    context=np.r_[host,timing[AGE_INDEX] if view.endswith('age_context66') else timing]
                query+=time.perf_counter()-tick
                frame=pd.concat([base.loc[[rid]],pd.DataFrame(context[None],index=[rid],columns=history)],axis=1)[X.columns]
                if verify:pd.testing.assert_frame_equal(frame,X.loc[[rid]],check_dtype=False,check_names=False,atol=1e-12,rtol=0)
                tick=time.perf_counter();pred.append(int(predictor.predict(frame)[0]));model+=time.perf_counter()-tick;got_ids.append(rid)
            tick=time.perf_counter();ring.push(stamp,packet);update+=time.perf_counter()-tick
        capacity+=ring.evicted_capacity
    elapsed=time.perf_counter()-start
    np.testing.assert_array_equal(got_ids,X.index.to_numpy());np.testing.assert_array_equal(pred,expected['replay']);assert seen==work['packet_rows']
    return dict(policy=policy,total_seconds=elapsed,update_seconds=update,query_seconds=query,model_seconds=model,
        decisions=len(pred),packets=seen,labels_match=True,capacity_evictions=capacity,ring_allocated_bytes=ring.data.nbytes)


def memory_worker(run,policy,index,workload):
    freeze=json.loads((run/'inputs_manifest.json').read_text());assert freeze['source_sha256']==e.sha256(Path(__file__))
    with threadpool_limits(limits=1):
        predictor,frames,expected=load_policy(run,freeze,policy);gc.collect()
        process=psutil.Process();baseline=process.memory_info().rss;peaks=[baseline];stop=threading.Event();ready=threading.Event()
        def sample():
            peaks.append(process.memory_info().rss);ready.set()
            while not stop.wait(.002):peaks.append(process.memory_info().rss)
        thread=threading.Thread(target=sample,daemon=True);thread.start();ready.wait()
        if workload=='replay':replay(run,freeze,policy,predictor,frames,expected)
        else:np.testing.assert_array_equal(predictor.predict(frames['1000']),expected['1000'])
        peaks.append(process.memory_info().rss);stop.set();thread.join();peak=max(peaks)
    e.write_json(run/f'rss_{workload}_{policy}_{index}.json',dict(policy=policy,workload=workload,run=index,
        baseline_rss_mib=baseline/2**20,peak_rss_mib=peak/2**20,delta_peak_inference_mib=(peak-baseline)/2**20,
        samples=len(peaks),interval_seconds=.002,labels_match=True,cold_first_prediction=True,ring_included=workload=='replay'))


def profile(run):
    freeze=json.loads((run/'inputs_manifest.json').read_text());cfg=freeze['configuration'];rows=[];replays=[]
    assert freeze['source_sha256']==e.sha256(Path(__file__))
    for record in freeze['files']:checked_file(run,record)
    # Freeze CPU-intensive work before entering this command; only one policy executes at a time.
    with threadpool_limits(limits=1):
        for policy in freeze['policies']:
            predictor,frames,expected=load_policy(run,freeze,policy)
            for batch in cfg['batches']:
                X=frames[str(batch)]
                for _ in range(cfg['warmups']):np.testing.assert_array_equal(predictor.predict(X),expected[str(batch)])
                for block in range(cfg['blocks']):
                    for repeat in range(cfg['repeats']):
                        tick=time.perf_counter();pred=predictor.predict(X);elapsed=time.perf_counter()-tick
                        np.testing.assert_array_equal(pred,expected[str(batch)])
                        rows.append(dict(policy=policy,batch=batch,block=block,repeat=repeat,seconds=elapsed,rows_per_second=batch/elapsed))
            replay(run,freeze,policy,predictor,frames,expected,verify=True)
            print('LATENCY/EQUALITY',policy,flush=True)
        rng=np.random.default_rng(cfg['seed'])
        for repeat in range(cfg['replay_timed_repeats']):
            for policy in rng.permutation(list(freeze['policies'])):
                predictor,frames,expected=load_policy(run,freeze,policy)
                replays.append(dict(repeat=repeat,**replay(run,freeze,policy,predictor,frames,expected)))
            print('RANDOMIZED REPLAY',repeat+1,flush=True)
    pd.DataFrame(rows).to_csv(run/'latency_raw.csv',index=False)
    pd.DataFrame(replays).to_csv(run/'replay_raw.csv',index=False)
    pd.DataFrame(rows).groupby(['policy','batch']).seconds.agg(['mean','median','std','count']).reset_index().to_csv(run/'latency_summary.csv',index=False)
    pd.DataFrame(replays).groupby('policy')[['total_seconds','update_seconds','query_seconds','model_seconds']].agg(['mean','std']).to_csv(run/'replay_summary.csv')
    memory=[]
    for policy in freeze['policies']:
        for workload in ['feature_input','replay']:
            for index in range(cfg['memory_runs']):
                with (run/f'rss_{workload}_{policy}_{index}.log').open('x',encoding='utf8') as stream:
                    p=subprocess.run([sys.executable,str(Path(__file__)),'--output',str(run),'--worker',policy,'--index',str(index),'--workload',workload],
                                     cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                if p.returncode:raise RuntimeError('Memory gate failed '+policy+'/'+workload)
                memory.append(json.loads((run/f'rss_{workload}_{policy}_{index}.json').read_text()))
        print('COLD RSS',policy,flush=True)
    memory=pd.DataFrame(memory);memory.to_csv(run/'memory_raw.csv',index=False)
    np.testing.assert_allclose(memory.peak_rss_mib-memory.baseline_rss_mib,memory.delta_peak_inference_mib,rtol=0,atol=1e-12)
    memory.groupby(['policy','workload'])[['baseline_rss_mib','peak_rss_mib','delta_peak_inference_mib']].mean().reset_index().to_csv(run/'memory_summary.csv',index=False)
    for record in freeze['files']:checked_file(run,record)
    e.write_json(run/'complete.json',dict(status='PASS_EXACT_RUNTIME',policies=len(freeze['policies']),timings=len(rows),
        randomized_replays=len(replays),cold_memory_runs=len(memory),all_labels_and_replayed_features_match=True,
        new_classifier_fits=0,source_sha256=e.sha256(Path(__file__)),inputs_manifest_sha256=e.sha256(run/'inputs_manifest.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--quality-run',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--profile-only',action='store_true')
    p.add_argument('--worker');p.add_argument('--index',type=int);p.add_argument('--workload',choices=['feature_input','replay'])
    a=p.parse_args();run=a.output.resolve()
    if a.worker:memory_worker(run,a.worker,a.index,a.workload)
    elif a.profile_only:profile(run)
    else:
        with threadpool_limits(limits=1):prepare(run,a.quality_run.resolve())
        if not a.prepare_only:profile(run)
