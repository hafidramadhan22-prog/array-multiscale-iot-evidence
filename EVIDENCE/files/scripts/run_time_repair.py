"""Frozen, finite feature repair with G selection before V; no historic holdout."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse,json,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from e4 import metrics
from sbfe import guard_roles
from time_context import repair_frame
from context_repair import select_on_g
from run_wb_research import load_data,parts_for
from wb_research import fit_experts,cube,mixture,learn_weights,profile_components,bijective_ports
from wb_scheduler import best_order,replay_scheduler

SOURCE=['scripts/run_time_repair.py','scripts/verify_time_repair.py','src/time_context.py','configs/time_repair.json',
 'docs/TIME_RUNTIME_PROTOCOL_20261020.md','tests/test_time_context.py',
 'src/wb_research.py','src/wb_scheduler.py','src/wb_behavior.py','src/e4.py','src/sbfe.py',
 'src/ta_hafid_engine.py','scripts/run_wb_research.py','scripts/prepare_wb_behavior.py',
 'scripts/prepare_sbfe.py','configs/target.json','requirements.lock.txt','src/multiscale_context.py','docs/TIME_RUNTIME_LITERATURE_20261020.md','src/context_repair.py','scripts/run_context_repair.py']

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def check_record(folder,record):
    p=(folder/record['path']).resolve()
    if not p.is_relative_to(folder.resolve()) or not p.is_file() or p.stat().st_size!=record['bytes'] or e.sha256(p)!=record['sha256']:
        raise ValueError('Artifact checksum mismatch: '+str(p))

def source_file(cfg,relative):
    folder=ROOT/cfg['source_run']
    if 'legacy29_symmetric' in relative:folder=ROOT/cfg['port_reference_run']
    relative=relative.replace('legacy25_symmetric','legacy25_multiscale60')
    items=json.loads((folder/'artifacts_sha256.json').read_text())
    matches=[z for z in items if z['path']==relative]
    if len(matches)!=1:raise ValueError('Uncatalogued baseline cache: '+relative)
    check_record(folder,matches[0]);return folder/relative

def baseline(cfg,stage,case,view,ids):
    prefix=f'stages/{stage}/{view}__{case}/'
    calibration=read_npz(source_file(cfg,prefix+'calibration.npz'))
    selection=json.loads(source_file(cfg,prefix+'selection_before_V.json').read_text())
    # Smoke only checks aligned processing; the old model had a larger A.
    result=[]
    requests=[(ids[0],calibration,'G'),(ids[1],calibration,'C')]
    if len(ids)==3:
        requests.append((ids[2],read_npz(source_file(cfg,prefix+'evaluation.npz')),'V'))
    elif len(ids)!=2:raise ValueError('Expected G/C or post-freeze G/C/V requests')
    for target,cache,key in requests:
        old_ids=cache['row_ids' if key=='V' else key+'_row_ids']
        lookup={int(v):i for i,v in enumerate(old_ids)}
        if len(lookup)!=len(old_ids) or any(int(v) not in lookup for v in target):raise ValueError('Cache row identity')
        ix=np.array([lookup[int(v)] for v in target])
        result.append(cache['p_experts' if key=='V' else key+'_experts'][ix])
    return result,selection

def metric_rows(stage,seed,view,case,y,p,weights,reference):
    out=[]
    for condition in ['original','port_bijection']:
        methods={n:p[:,j] for j,n in enumerate(e.NAMES)}
        methods.update(weighted_voting=mixture(p,weights),soft_voting=p.mean(1),
                       weighted_reference=mixture(p,reference))
        for method,post in methods.items():
            out.append(dict(stage=stage,protocol='temporal' if stage.startswith('temporal') else 'signature',
                seed=seed,view=view,case=case,condition=condition,method=method,**metrics(y,post.argmax(1))))
    return out

def stage_run(run,stage):
    cfg=json.loads((run/'configuration_freeze.json').read_text())
    for item in json.loads((run/'source_manifest.json').read_text()):
        if e.sha256(ROOT/item['path'])!=item['sha256']:raise ValueError('Source changed after freeze')
    raw,y,holdout,data=load_data(cfg)
    parts,groups=parts_for(stage,raw,y,holdout,data,cfg['smoke_test']);a,g,c,v=parts
    guard_roles(raw.index.to_numpy(),parts,holdout)
    for part in parts:
        if set(y[part])!=set(range(5)):raise ValueError('Missing class')
    seed=int(stage.split('seed')[1]) if stage.startswith('temporal') else cfg['signature_seed']
    local={**e.DEFAULT_CONFIG,**json.loads((ROOT/'configs/target.json').read_text()),**cfg}
    local.update(seed=seed,_training_ids=raw.index.to_numpy(),_holdout_ids=holdout,train_rows=len(raw))
    if cfg['smoke_test']:local['full_balance_target']=60
    folder=run/'stages'/stage;folder.mkdir()
    np.savez_compressed(folder/'split.npz',**{key:raw.index[pos].to_numpy() for key,pos in zip('AGCV',parts)})
    manifest=[];fitted={};baselines={};references={};g_records=[]
    shifted=bijective_ports(raw,cfg['port_bijection_offset'])
    for case in cfg['cases']:
        for view in ['baseline25','baseline29']:
            arrays,sel=baseline(cfg,stage,case,'legacy25_symmetric' if view=='baseline25' else 'legacy29_symmetric',
                                [raw.index[g],raw.index[c]])
            baselines[(view,case)]=(arrays,sel)
            if view=='baseline25':references[case]=sel['weights']
            if view=='baseline25':
                methods={n:arrays[0][:,j] for j,n in enumerate(e.NAMES)}
                methods.update(weighted_voting=mixture(arrays[0],sel['weights']),soft_voting=arrays[0].mean(1))
                for name,post in methods.items():
                    m=metrics(y[g],post.argmax(1));j=e.NAMES.index(name) if name in e.NAMES else None
                    sizes=sel['costs']['serialized_pipeline_bytes']
                    g_records.append(dict(role='G',view=view,case=case,method=name,score=.5*(m['macro_f1']+m['ddos_f2']),
                        bytes=sizes[j] if j is not None else sum(sizes),metrics=m))
        for view in cfg['views']:
            X,num,cat=repair_frame(raw,data,view);Xs,_,_=repair_frame(shifted,data,view)
            pd.testing.assert_frame_equal(X,Xs)
            dest=folder/f'{view}__{case}';dest.mkdir()
            print(f'FIT {stage}/{view}/{case} A={len(a)} G={len(g)} C={len(c)} V={len(v)}',flush=True)
            with threadpool_limits(limits=1):
                models,plans,fit_records=fit_experts(X,y,a,np.r_[g,c,v],case,num,cat,cfg['params'],local)
                pg=cube(models,X.iloc[g]);pc=cube(models,X.iloc[c])
                sample=a[e.sample_positions(y[a],min(1000,len(a)),seed)]
                costs=profile_components(models,X.iloc[sample],X.iloc[c])
                best,candidates=learn_weights(y[g],pg,costs['training_pipeline_latency_seconds'],costs['serialized_pipeline_bytes'],local)
                w=best['weights'];schedule=best_order(pc,w,costs['expert_costs'],
                    preprocessing_groups=costs['preprocessing_groups'],preprocessing_costs=costs['preprocessing_costs'])
                e.write_json(dest/'selection_before_V.json',dict(weights=w,alpha=best['alpha'],weight_candidates=candidates,
                    order=schedule['order'],schedule=schedule,costs=costs,plans=plans,fit_records=fit_records,
                    weight_role='G',order_role='C',feature_columns=list(X.columns),reference_weights=references[case]))
                bundle=dict(models=models,weights=w,order=list(schedule['order']),view=view,case=case,numeric=num,categorical=cat)
                path=dest/'model.joblib';joblib.dump(bundle,path,compress=1)
                record=dict(stage=stage,view=view,case=case,path=path.relative_to(run).as_posix(),
                    sha256=e.sha256(path),bytes=path.stat().st_size)
                manifest.append(record);e.write_json(dest/'model_frozen_before_V.json',record)
                np.savez_compressed(dest/'calibration.npz',G_row_ids=raw.index[g].to_numpy(),G_y=y[g],G_experts=pg,
                                    C_row_ids=raw.index[c].to_numpy(),C_y=y[c],C_experts=pc)
                methods={n:pg[:,j] for j,n in enumerate(e.NAMES)}
                methods.update(weighted_voting=mixture(pg,w),soft_voting=pg.mean(1))
                for name,post in methods.items():
                    m=metrics(y[g],post.argmax(1));j=e.NAMES.index(name) if name in e.NAMES else None
                    sizes=costs['serialized_pipeline_bytes']
                    g_records.append(dict(role='G',view=view,case=case,method=name,score=.5*(m['macro_f1']+m['ddos_f2']),
                        bytes=sizes[j] if j is not None else sum(sizes),metrics=m))
                fitted[(view,case)]=(models,X,w,schedule['order'])
            print(f'FROZEN {stage}/{view}/{case}',flush=True)
    selected={}
    for case in cfg['cases']:
        records=[z for z in g_records if z['case']==case]
        selected[case]=dict(wb_only=select_on_g([z for z in records if z['method']=='weighted_voting']),
                            all_models=select_on_g(records))
    e.write_json(folder/'candidate_selection_before_V.json',dict(candidates=g_records,selected=selected,role='G'))
    e.write_json(folder/'models_manifest.json',manifest)
    e.write_json(folder/'stage_frozen_before_V.json',dict(timestamp=datetime.now().astimezone().isoformat(),
        models_manifest_sha256=e.sha256(folder/'models_manifest.json'),selection_sha256=e.sha256(folder/'candidate_selection_before_V.json')))
    rows=[]
    for (view,case),(models,X,w,order) in fitted.items():
        with threadpool_limits(limits=1):pv=cube(models,X.iloc[v])
        pw=mixture(pv,w);scheduled=replay_scheduler(pv,w,order)
        np.testing.assert_array_equal(scheduled.labels,pw.argmax(1))
        dest=folder/f'{view}__{case}'
        np.savez_compressed(dest/'evaluation.npz',row_ids=raw.index[v].to_numpy(),y=y[v],p_experts=pv,
            p_weighted=pw,p_reference=mixture(pv,references[case]),p_soft=pv.mean(1),
            certified_labels=scheduled.labels,certified_counts=scheduled.counts,
            capture=data['capture'][v],logical_ns=data['logical_ns'][v],record=data['record'][v])
        rows+=metric_rows(stage,seed,view,case,y[v],pv,w,references[case])
    for (view,case),(arrays,sel) in baselines.items():
        arrays,_=baseline(cfg,stage,case,'legacy25_symmetric' if view=='baseline25' else 'legacy29_symmetric',
                          [raw.index[g],raw.index[c],raw.index[v]])
        rows+=metric_rows(stage,seed,view,case,y[v],arrays[2],sel['weights'],references[case])
        # baseline29 is not invariant: replace its port-shift metrics with cached stress.
        if view=='baseline29':
            old=read_npz(source_file(cfg,f'stages/{stage}/legacy29_symmetric__{case}/evaluation.npz'))
            lookup={int(row):i for i,row in enumerate(old['row_ids'])};ix=np.array([lookup[int(row)] for row in raw.index[v]])
            ps=old['p_experts_port_shift'][ix]
            replacement=metric_rows(stage,seed,view,case,y[v],ps,sel['weights'],references[case])
            rows=[z for z in rows if not(z['view']==view and z['case']==case and z['condition']=='port_bijection')]
            rows += [z for z in replacement if z['condition']=='port_bijection']
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    e.write_json(folder/'complete.json',dict(status='CONTEXT_REPAIR_STAGE_COMPLETE',expert_fits=5*len(manifest),
        bundles=len(manifest),metrics_sha256=e.sha256(folder/'metrics.csv'),models_manifest_sha256=e.sha256(folder/'models_manifest.json')))
    print(f'DONE {stage}: {len(manifest)} bundles/{len(rows)} metric rows',flush=True)

def run(output,smoke):
    output=output.resolve()
    if output.parent!=(ROOT/'runs').resolve() or output.exists():raise ValueError('New direct child of runs required')
    cfg=json.loads((ROOT/'configs/time_repair.json').read_text())
    if cfg['historical_holdout_used'] or cfg['new_data_or_captures']:raise ValueError('Scope gate')
    raw,y,holdout,data=load_data(cfg)
    old=ROOT/cfg['source_run']
    for item in json.loads((old/'artifacts_sha256.json').read_text()):check_record(old,item)
    params=json.loads((old/'configuration_freeze.json').read_text())['params']
    cfg.update(params=params,smoke_test=smoke,model_order=e.NAMES,class_order=e.CLASSES,
        dataset_sha256=e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv'),
        data_catalog_sha256=e.sha256(ROOT/cfg['data_folder']/'catalog.json'),
        source_catalog_sha256=e.sha256(old/'artifacts_sha256.json'),created_local=datetime.now().astimezone().isoformat())
    if smoke:
        cfg.update(seeds=[42],runtime_workers=1)
        for name in ['random_forest','xgboost','lightgbm']:cfg['params'][name]['n_estimators']=8
    output.mkdir()
    for name in ['source','stages','tables']:(output/name).mkdir()
    e.write_json(output/'configuration_freeze.json',cfg);e.write_json(output/'environment.json',e.environment())
    e.write_json(output/'effective_operational_config.json',{**e.DEFAULT_CONFIG,**json.loads((ROOT/'configs/target.json').read_text()),**cfg})
    snapshot=[]
    for rel in SOURCE:
        shutil.copy2(ROOT/rel,output/'source'/Path(rel).name);snapshot.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    e.write_json(output/'source_manifest.json',snapshot)
    np.savez_compressed(output/'canonical_indices.npz',development_row_ids=raw.index.to_numpy(),historical_holdout_row_ids=holdout)
    stages=[f'temporal_seed{s}' for s in cfg['seeds']]+([] if smoke else ['signature_fold1','signature_fold2'])
    e.write_json(output/'planned_stages.json',stages)
    def launch(stage):
        with (output/f'{stage}.log').open('x',encoding='utf8') as stream:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--stage',stage],cwd=ROOT,
                             stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Stage failed: '+stage)
    with ThreadPoolExecutor(max_workers=cfg['runtime_workers']) as pool:list(pool.map(launch,stages))
    frames=[];manifest=[]
    for stage in stages:
        dest=output/'stages'/stage;done=json.loads((dest/'complete.json').read_text())
        if e.sha256(dest/'metrics.csv')!=done['metrics_sha256']:raise ValueError('Stage metrics changed')
        frames.append(pd.read_csv(dest/'metrics.csv'));manifest+=json.loads((dest/'models_manifest.json').read_text())
    pd.concat(frames,ignore_index=True).to_csv(output/'tables/metrics.csv',index=False)
    e.write_json(output/'models_manifest.json',manifest)
    e.write_json(output/'quality_complete.json',dict(status='SMOKE_TEST_ONLY' if smoke else 'CONTEXT_REPAIR_QUALITY_COMPLETE',
        stages=len(stages),bundles=len(manifest),expert_fits=5*len(manifest),historical_holdout_used=False,
        metrics_sha256=e.sha256(output/'tables/metrics.csv'),models_manifest_sha256=e.sha256(output/'models_manifest.json')))
    print(f'Completed {output}: {5*len(manifest)} expert fits',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--stage')
    args=p.parse_args()
    try:
        if args.stage:stage_run(args.output,args.stage)
        else:run(args.output,args.smoke)
    except Exception:
        traceback.print_exc();raise
