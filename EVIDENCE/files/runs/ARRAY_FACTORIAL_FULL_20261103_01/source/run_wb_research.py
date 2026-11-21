"""Authorized WB-centered representation, resampling, port-stress and scheduling study."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse,json,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from e4 import grouped_splits,metrics,verified_historical_file
from prepare_sbfe import load_raw
from prepare_wb_behavior import verify_catalog
from sbfe import guard_roles
from wb_research import view_data,fit_experts,cube,mixture,learn_weights,profile_components,bijective_ports
from wb_scheduler import best_order,replay_scheduler


SOURCE=['scripts/run_wb_research.py','src/wb_research.py','src/wb_behavior.py','src/wb_scheduler.py',
        'src/ta_hafid_engine.py','src/e4.py','src/sbfe.py','scripts/prepare_wb_behavior.py',
        'scripts/prepare_sbfe.py','configs/target.json',
        'configs/wb_research.json','docs/WB_RESEARCH_PROTOCOL_20261019.md',
        'docs/WB_BEHAVIOR_FEATURES.md','docs/WB_SCHEDULER_METHOD.md','requirements.lock.txt']


def load_data(cfg):
    folder=ROOT/cfg['data_folder'];verify_catalog(folder)
    complete=json.loads((folder/'complete.json').read_text())
    if complete['status']!='WB_BEHAVIOR_DATA_COMPLETE':raise ValueError('Unverified behavior data')
    with np.load(folder/'packet_features.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    raw,y,holdout=load_raw();np.testing.assert_array_equal(raw.index,data['row_ids'])
    with np.load(folder/'target_indices.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(y,z['y']);np.testing.assert_array_equal(holdout,z['historical_holdout_ids'])
    return raw,y,holdout,data


def parts_for(name,raw,y,holdout,data,smoke):
    valid=data['valid_after_purge'].copy()
    if smoke:
        valid[:]=False
        for role in range(4):
            for c in range(5):
                inds=np.flatnonzero(data['valid_after_purge']&(data['role']==role)&(y==c))[:15]
                valid[inds]=True
    if name.startswith('temporal'):
        return [np.flatnonzero(valid&(data['role']==r)) for r in range(4)],None
    groups,splits,_=grouped_splits(raw,y,holdout);fold=int(name.split('fold')[1]);train,test=splits[fold-1]
    train=train[valid[train]];test=test[valid[test]]
    a,gc=train_test_split(train,test_size=.25,stratify=y[train],random_state=42+fold)
    g,c=train_test_split(gc,test_size=.5,stratify=y[gc],random_state=142+fold)
    parts=[np.sort(a),np.sort(g),np.sort(c),np.sort(test)]
    if set(groups[np.r_[a,g,c]])&set(groups[test]):raise ValueError('Signature overlap')
    return parts,groups


def verify_frozen_source(run):
    for item in json.loads((run/'source_manifest.json').read_text()):
        if e.sha256(ROOT/item['path'])!=item['sha256']:raise ValueError('Source changed after freeze: '+item['path'])


def stage_run(run,name):
    verify_frozen_source(run)
    cfg=json.loads((run/'configuration_freeze.json').read_text());raw,y,holdout,data=load_data(cfg)
    seed=int(name.split('seed')[1]) if name.startswith('temporal') else cfg['signature_seed']
    local={**e.DEFAULT_CONFIG,**json.loads((ROOT/'configs/target.json').read_text()),**cfg}
    local.update(seed=seed,_training_ids=raw.index.to_numpy(),_holdout_ids=holdout,train_rows=len(raw))
    if cfg['smoke_test']:local['full_balance_target']=max(10,300//5)
    parts,groups=parts_for(name,raw,y,holdout,data,cfg['smoke_test']);a,g,c,v=parts
    guard_roles(raw.index.to_numpy(),parts,holdout)
    for part in parts:
        if set(y[part])!=set(range(5)):raise ValueError('Missing class in role')
    folder=run/'stages'/name;folder.mkdir()
    np.savez_compressed(folder/'split.npz',**{key:raw.index[pos].to_numpy() for key,pos in zip(['A','G','C','V'],parts)},
        **({} if groups is None else {'signature_ids':groups,'all_row_ids':raw.index.to_numpy()}))
    rows=[];manifest=[];fits=0
    shifted=bijective_ports(raw,cfg['port_bijection_offset'])
    for view in cfg['views']:
        X,num,cat=view_data(raw,data,view);Xs,_,_=view_data(shifted,data,view)
        if view.startswith('legacy25'):pd.testing.assert_frame_equal(X,Xs)
        for case in cfg['cases']:
            dest=folder/f'{view}__{case}';dest.mkdir()
            print(f'FIT {name}/{view}/{case} A={len(a)} G={len(g)} C={len(c)} V={len(v)}',flush=True)
            with threadpool_limits(limits=1):
                models,plans,fit_records=fit_experts(X,y,a,np.r_[g,c,v],case,num,cat,cfg['params'],local)
                fits+=5
                pg=cube(models,X.iloc[g]);pc=cube(models,X.iloc[c])
                sample=a[e.sample_positions(y[a],min(1000,len(a)),seed)]
                costs=profile_components(models,X.iloc[sample],X.iloc[c])
                weight,candidates=learn_weights(y[g],pg,costs['training_pipeline_latency_seconds'],costs['serialized_pipeline_bytes'],local)
                weights=weight['weights']
                schedule=best_order(pc,weights,costs['expert_costs'],
                    preprocessing_groups=costs['preprocessing_groups'],preprocessing_costs=costs['preprocessing_costs'])
                e.write_json(dest/'selection_before_V.json',dict(weights=weights,alpha=weight['alpha'],
                    weight_candidates=candidates,weight_role='G',order_role='C',order=schedule['order'],
                    schedule=schedule,costs=costs,plans=plans,fit_records=fit_records))
                bundle=dict(models=models,weights=weights,order=list(schedule['order']),view=view,case=case,numeric=num,categorical=cat)
                path=dest/'model.joblib';joblib.dump(bundle,path,compress=1)
                record=dict(stage=name,view=view,case=case,path=path.relative_to(run).as_posix(),sha256=e.sha256(path),bytes=path.stat().st_size)
                manifest.append(record);e.write_json(dest/'model_frozen_before_V.json',record)
                np.savez_compressed(dest/'calibration.npz',G_row_ids=raw.index[g].to_numpy(),G_y=y[g],G_experts=pg,
                    C_row_ids=raw.index[c].to_numpy(),C_y=y[c],C_experts=pc)
                pv=cube(models,X.iloc[v]);pw=mixture(pv,weights);soft=pv.mean(1)
                fixed=replay_scheduler(pv,weights,list(range(5)));selected=replay_scheduler(pv,weights,schedule['order'])
                np.testing.assert_array_equal(fixed.labels,pw.argmax(1));np.testing.assert_array_equal(selected.labels,pw.argmax(1))
                pshift=cube(models,Xs.iloc[v]);ws=mixture(pshift,weights)
                np.savez_compressed(dest/'evaluation.npz',row_ids=raw.index[v].to_numpy(),y=y[v],p_experts=pv,p_weighted=pw,p_soft=soft,
                    p_experts_port_shift=pshift,p_weighted_port_shift=ws,certified_counts=selected.counts,fixed_counts=fixed.counts,
                    certified_labels=selected.labels,capture=data['capture'][v],record=data['record'][v])
                for condition,posterior in [('original',pv),('port_bijection',pshift)]:
                    preds={n:posterior[:,j] for j,n in enumerate(e.NAMES)}
                    preds.update(weighted_voting=mixture(posterior,weights),soft_voting=posterior.mean(1))
                    for method,p in preds.items():
                        rows.append(dict(stage=name,protocol='temporal' if name.startswith('temporal') else 'signature',seed=seed,
                            view=view,case=case,condition=condition,method=method,**metrics(y[v],p.argmax(1))))
                e.write_json(dest/'scheduling_evaluation.json',dict(identical_labels=True,mean_experts_selected=float(selected.counts.mean()),
                    mean_experts_fixed=float(fixed.counts.mean()),fraction_early_selected=float((selected.counts<5).mean()),
                    full_experts=5,not_full_probabilities=True))
            pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
            e.write_json(folder/'models_manifest.json',manifest)
            print(f'DONE {name}/{view}/{case}; fitting cumulative={fits}',flush=True)
    e.write_json(folder/'complete.json',dict(status='WB_STAGE_COMPLETE',expert_fits=fits,stage=name,
        model_manifest_sha256=e.sha256(folder/'models_manifest.json'),metrics_sha256=e.sha256(folder/'metrics.csv')))


def run(output,smoke=False):
    output=output.resolve()
    if output.parent!=(ROOT/'runs').resolve() or output.exists():raise ValueError('New direct child of runs required')
    cfg=json.loads((ROOT/'configs/wb_research.json').read_text())
    if cfg['historical_holdout_used']:raise ValueError('Holdout gate')
    raw,y,holdout,data=load_data(cfg)
    parameter_file=verified_historical_file(ROOT/'runs/FULL_20261013_01','dstport29/tables/selection_full_training_balanced.json')
    cfg['params']={n:r['params'] for n,r in json.loads(parameter_file.read_text())['selected']['ddos_tuned'].items()}
    cfg.update(smoke_test=smoke,parameter_source_sha256=e.sha256(parameter_file),
        dataset_sha256=e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv'),data_catalog_sha256=e.sha256(ROOT/cfg['data_folder']/'catalog.json'),
        model_order=e.NAMES,class_order=e.CLASSES,created_local=datetime.now().astimezone().isoformat())
    if smoke:
        cfg.update(views=['legacy29','legacy29_symmetric'],cases=['balanced'],seeds=[42],runtime_workers=1)
        for name in ['random_forest','xgboost','lightgbm']:cfg['params'][name]['n_estimators']=8
    output.mkdir()
    for d in ['source','stages','tables','profiling']:(output/d).mkdir()
    e.write_json(output/'configuration_freeze.json',cfg);e.write_json(output/'environment.json',e.environment())
    e.write_json(output/'effective_operational_config.json',{**e.DEFAULT_CONFIG,**json.loads((ROOT/'configs/target.json').read_text()),**cfg})
    snapshot=[]
    for rel in SOURCE:
        if not (ROOT/rel).exists():raise ValueError('Missing source: '+rel)
        shutil.copy2(ROOT/rel,output/'source'/Path(rel).name);snapshot.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    e.write_json(output/'source_manifest.json',snapshot)
    np.savez_compressed(output/'canonical_indices.npz',development_row_ids=raw.index.to_numpy(),historical_holdout_row_ids=holdout)
    stages=[f'temporal_seed{s}' for s in cfg['seeds']]+([] if smoke else ['signature_fold1','signature_fold2'])
    e.write_json(output/'planned_stages.json',stages)
    def launch(name):
        logfile=output/f'{name}.log'
        with logfile.open('x',encoding='utf8') as stream:
            command=[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--stage',name]
            proc=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f'Stage failed {name}; inspect {logfile}')
        return name
    with ThreadPoolExecutor(max_workers=cfg['runtime_workers']) as pool:
        list(pool.map(launch,stages))
    frames=[];manifest=[];total=0
    for name in stages:
        folder=output/'stages'/name;done=json.loads((folder/'complete.json').read_text());total+=done['expert_fits']
        if e.sha256(folder/'models_manifest.json')!=done['model_manifest_sha256']:raise ValueError('Stage manifest changed')
        frames.append(pd.read_csv(folder/'metrics.csv'));manifest+=json.loads((folder/'models_manifest.json').read_text())
    pd.concat(frames,ignore_index=True).to_csv(output/'tables/metrics.csv',index=False)
    e.write_json(output/'models_manifest.json',manifest)
    e.write_json(output/'quality_complete.json',dict(status='SMOKE_TEST_ONLY' if smoke else 'WB_QUALITY_COMPLETE',
        stages=len(stages),bundles=len(manifest),expert_fits=total,historical_holdout_used=False,
        models_manifest_sha256=e.sha256(output/'models_manifest.json'),metrics_sha256=e.sha256(output/'tables/metrics.csv')))
    print(f'WB finished: {output}; expert fits={total}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--stage')
    args=parser.parse_args()
    if args.stage:stage_run(args.output,args.stage)
    else:run(args.output,args.smoke)
