"""Frozen 2x2 context/ordering experiment with cached matched corners."""
from pathlib import Path
import argparse,json,sys,shutil,subprocess,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from prepare_wb_behavior import verify_catalog
from run_wb_research import load_data,parts_for
from wb_research import fit_experts,cube,mixture
from sbfe import guard_roles
from array_revision import VIEWS,frame,reorder_context,scored

OLD=ROOT/'runs/MULTISCALE_REPAIR_FULL_20261020_01'
SHORT=ROOT/'runs/WB_RESEARCH_FULL_20261019_01'
def read_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def checked(folder,rel):
    items=json.loads((folder/'artifacts_sha256.json').read_text())
    r=next(x for x in items if x['path'].replace('\\','/')==rel)
    p=folder/rel
    assert e.sha256(p)==r['sha256'] and p.stat().st_size==r['bytes'],str(p)
    return p
def frozen(run):
    for row in json.loads((run/'source_manifest.json').read_text()):assert e.sha256(ROOT/row['path'])==row['sha256']
def methods(p,w):return {'weighted_voting':mixture(p,w),'uniform_voting':p.mean(1),'random_forest':p[:,1],'logistic_regression':p[:,4]}

def stage(run,name):
    frozen(run);cfg=json.loads((run/'configuration_freeze.json').read_text())
    raw,y,holdout,data=load_data(cfg)
    parts,_=parts_for(name,raw,y,holdout,data,cfg['smoke_test']);a,g,c,v=parts
    guard_roles(raw.index.to_numpy(),parts,holdout)
    seed=int(name.split('seed')[1]) if name.startswith('temporal') else 42
    folder=run/'stages'/name;folder.mkdir()
    np.savez_compressed(folder/'split.npz',**{k:raw.index[pos].to_numpy() for k,pos in zip('AGCV',parts)})
    prefix=f'stages/{name}/legacy25_multiscale60__balanced/'
    selection=json.loads(checked(OLD,prefix+'selection_before_V.json').read_text());weights=selection['weights']
    e.write_json(folder/'fixed_weights.json',dict(weights=weights,source=str(checked(OLD,prefix+'selection_before_V.json').relative_to(ROOT)),role='Previously frozen G; identical weights in all four factorial cells'))
    cfg.update(seed=seed,_training_ids=raw.index.to_numpy(),_holdout_ids=holdout,train_rows=len(raw))
    local={**e.DEFAULT_CONFIG,**json.loads((ROOT/'configs/target.json').read_text()),**cfg}
    if cfg['smoke_test']:local['full_balance_target']=60
    rows=[]
    for view in VIEWS:
        dest=folder/view;dest.mkdir();X,num,cat=frame(raw,data['prior_multiscale'],view)
        if view in ('c40_o1','c60_o10') and not cfg['smoke_test']:
            owner=SHORT if view=='c40_o1' else OLD
            oldview='legacy25_symmetric' if view=='c40_o1' else 'legacy25_multiscale60'
            oldsplit=read_npz(checked(owner,f'stages/{name}/split.npz'))
            for k,pos in zip('AGCV',parts):np.testing.assert_array_equal(raw.index[pos],oldsplit[k])
            if view=='c40_o1':
                verify_catalog(ROOT/'runs/WB_BEHAVIOR_DATA_20261019_01')
                olddata=read_npz(ROOT/'runs/WB_BEHAVIOR_DATA_20261019_01/packet_features.npz')
                np.testing.assert_array_equal(reorder_context(data['prior_multiscale'],view),olddata['prior_symmetric'])
            else:np.testing.assert_array_equal(reorder_context(data['prior_multiscale'],view),data['prior_multiscale'])
            cache=read_npz(checked(owner,f'stages/{name}/{oldview}__balanced/evaluation.npz'))
            np.testing.assert_array_equal(cache['row_ids'],raw.index[v]);np.testing.assert_array_equal(cache['y'],y[v]);pv=cache['p_experts']
            e.write_json(dest/'provenance.json',dict(reused=True,owner=owner.relative_to(ROOT).as_posix(),view=oldview,split_identical=True,features_identical=True))
            print('CACHE_VERIFIED',name,view,flush=True)
        else:
            print('FIT',name,view,'A',len(a),'V',len(v),flush=True)
            with threadpool_limits(limits=1):
                models,plans,records=fit_experts(X,y,a,np.r_[g,c,v],'balanced',num,cat,cfg['params'],local)
                pg=cube(models,X.iloc[g]);pc=cube(models,X.iloc[c])
                model=dest/'model.joblib';joblib.dump(dict(models=models,weights=weights,order=selection['order'],numeric=num,categorical=cat,view=view,case='balanced'),model,compress=1)
                e.write_json(dest/'frozen_before_V.json',dict(sha256=e.sha256(model),bytes=model.stat().st_size,plans=plans,fit_records=records,weights=weights,selection='No selection on new evaluation; fixed stage reference weights',time=time.time()))
                np.savez_compressed(dest/'calibration.npz',G_row_ids=raw.index[g],G_y=y[g],G_experts=pg,C_row_ids=raw.index[c],C_y=y[c],C_experts=pc)
                pv=cube(models,X.iloc[v])
            print('FROZEN_AND_EVALUATED',name,view,flush=True)
        np.savez_compressed(dest/'evaluation.npz',row_ids=raw.index[v],y=y[v],p_experts=pv,weights=np.asarray(weights))
        for method,p in methods(pv,weights).items():rows.append(dict(stage=name,view=view,method=method,**scored(y[v],p.argmax(1))))
        pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    e.write_json(folder/'complete.json',dict(status='PASS',stage=name,smoke=cfg['smoke_test']))

def run(output,smoke):
    assert output.parent.resolve()==(ROOT/'runs').resolve() and not output.exists()
    cfg=json.loads((OLD/'configuration_freeze.json').read_text())
    cfg.update(experiment='ARRAY_FACTORIAL_20261103',smoke_test=smoke,views=list(VIEWS),cases=['balanced'],weight_policy='fixed_per_stage_C60_O10',new_selection_on_V=False)
    if smoke:
        for n in ['random_forest','xgboost','lightgbm']:cfg['params'][n]['n_estimators']=8
    raw,y,holdout,data=load_data(cfg)
    verify_catalog(ROOT/'runs/WB_BEHAVIOR_DATA_20261019_01')
    short=read_npz(ROOT/'runs/WB_BEHAVIOR_DATA_20261019_01/packet_features.npz')
    np.testing.assert_array_equal(reorder_context(data['prior_multiscale'],'c40_o1'),short['prior_symmetric'])
    np.testing.assert_array_equal(reorder_context(data['prior_multiscale'],'c60_o10'),data['prior_multiscale'])
    output.mkdir();(output/'stages').mkdir();(output/'source').mkdir()
    e.write_json(output/'configuration_freeze.json',cfg);e.write_json(output/'environment.json',e.environment())
    sources=['scripts/run_array_factorial_20261103.py','src/array_revision.py','src/wb_research.py','src/sbfe.py','src/e4.py','src/ta_hafid_engine.py','src/multiscale_context.py','src/wb_behavior.py','scripts/run_wb_research.py','scripts/prepare_wb_behavior.py','scripts/prepare_sbfe.py','docs/ARRAY_MAJOR_REVISION_PROTOCOL_20261103.md','requirements.lock.txt','configs/target.json']
    manifest=[]
    for rel in sources:
        shutil.copy2(ROOT/rel,output/'source'/Path(rel).name);manifest.append(dict(path=rel,sha256=e.sha256(ROOT/rel)))
    e.write_json(output/'source_manifest.json',manifest)
    names=['temporal_seed42'] if smoke else ['temporal_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']
    def launch(name):
        with (output/(name+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--output',str(output),'--stage',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(name+' failed; see log')
        print('DONE',name,flush=True)
    with ThreadPoolExecutor(max_workers=1 if smoke else 2) as pool:list(pool.map(launch,names))
    pd.concat([pd.read_csv(output/'stages'/name/'metrics.csv') for name in names]).to_csv(output/'metrics.csv',index=False)
    e.write_json(output/'artifacts_sha256.json',[dict(path=p.relative_to(output).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'])
    e.write_json(output/'complete.json',dict(status='PASS_FACTORIAL_SMOKE' if smoke else 'PASS_FACTORIAL_FULL',stages=names,new_expert_fits=20 if smoke else 50,all_old_data_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--stage');a=p.parse_args()
    if a.stage:stage(a.output.resolve(),a.stage)
    else:run(a.output.resolve(),a.smoke)
