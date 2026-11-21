"""Corrected reproduction runner; numerical results remain the original 63 fits plus audited recovery."""
from pathlib import Path
import sys,json,argparse,time,shutil,subprocess,warnings,gc
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from wb_research import prepare,mixture
from run_wb_research import load_data,parts_for
from array_revision import frame
from array_svm_20261118 import CLASSES,label_metrics,scale_gamma,candidate_specs,select_on_G
from sbfe import guard_roles

PARENTS={
 'data':('MULTISCALE_DATA_20261020_01','catalog.json','2597a4f4e825cde4d615a6bbde352085a96a2282b392e2c050859d837cc567fa'),
 'history_data':('ARRAY_HISTORY_DATA_20261103_01','catalog.json','89dae99bcbc8776023756e22ee6c8e96be50355b0fb7b2e231e194f1751daa0f'),
 'history_models':('ARRAY_HISTORY_FULL_20261103_01','catalog.json','58cb8e1a93eead3fa4dcf04c3366cc82f43b8b2021e233569a109f64dbd108dd'),
 'reference':('MULTISCALE_REPAIR_FULL_20261020_01','artifacts_sha256.json','d89f9b971d2889e11a0e76345352b08abb93d31d4cced1536fa88ddf41112d2f'),
 'aligned':('ARRAY_FLAG_FORMAT_20261104_01','catalog.json','a68acf541a06c44455a96f77c14f11d8f25e796783a12f30f37e5c92c1e5f309'),
 'guard':('FPR_SQL_GUARD_20261109_01','artifacts_sha256.json','f30f4247e55d77b42c0701eceec9017a3f38de605b732e0b8815512f723ee652'),
 'external':('INDEPENDENT_CAPTURE_20261109_01','artifacts_sha256.json','9b25f63c75777d7c1aad4d04c75405832062d994c2410985e58fc53f6513d537')}
SOURCES=['scripts/run_array_svm_v2_20261118.py','src/array_svm_20261118.py','src/wb_research.py','src/array_revision.py','src/ta_hafid_engine.py','src/sbfe.py','src/e4.py','src/multiscale_context.py','scripts/run_wb_research.py','scripts/prepare_wb_behavior.py','scripts/prepare_sbfe.py','configs/target.json','docs/ARRAY_SVM_COMPARATOR_PROTOCOL_20261118.md','requirements.lock.txt']
STAGES=['private_fold1','private_fold2','temporal_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def npz(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def gated(parent,rel):
    folder,name,digest=PARENTS[parent];base=ROOT/'runs'/folder;catalog=base/name
    if e.sha256(catalog)!=digest:raise ValueError('Parent catalog changed: '+parent)
    rows=read(catalog);row=next(r for r in rows if r['path'].replace('\\','/')==rel)
    p=base/rel
    if not p.resolve().is_relative_to(base.resolve()) or p.stat().st_size!=row['bytes'] or e.sha256(p)!=row['sha256']:raise ValueError('Input hash gate: '+str(p))
    return p

def verify_sources(run):
    for r in read(run/'source_manifest.json'):
        if e.sha256(ROOT/r['path'])!=r['sha256']:raise ValueError('Source changed after freeze: '+r['path'])

def stage_inputs(name,smoke=False):
    cfg=read(gated('reference','configuration_freeze.json'));raw,y,holdout,data=load_data(cfg)
    original='signature_fold'+name[-1] if name.startswith('private') else name
    parts,_=parts_for(original,raw,y,holdout,data,False)
    if name.startswith('private'):
        h=npz(gated('history_data','histories.npz'));np.testing.assert_array_equal(h['row_ids'],raw.index)
        fold=int(name[-1]);values=h[f'fold{fold}_exclusive'].copy();values[~data['valid_after_purge']]=0
        for k,p in enumerate(parts):assert (h[f'fold{fold}_roles'][p]==k).all()
        base=f'{original}_exclusive';ref=npz(gated('history_models',base+'/evaluation.npz'));split=npz(gated('history_models',base+'/split.npz'))
    else:
        values=data['prior_multiscale'];base=f'stages/{original}/legacy25_multiscale60__balanced'
        ref=npz(gated('reference',base+'/evaluation.npz'));split=npz(gated('reference',f'stages/{original}/split.npz'))
    for key,p in zip('AGCV',parts):np.testing.assert_array_equal(raw.index[p],split[key])
    np.testing.assert_array_equal(raw.index[parts[3]],ref['row_ids']);np.testing.assert_array_equal(y[parts[3]],ref['y'])
    if smoke:parts=[np.concatenate([p[y[p]==cl][:12] for cl in range(5)]) for p in parts]
    guard_roles(raw.index.to_numpy(),parts,holdout)
    X,num,cat=frame(raw,values,'c60_o10')
    seed=int(name.split('seed')[1]) if 'seed' in name else 42
    local={**e.DEFAULT_CONFIG,**read(ROOT/'configs/target.json'),**cfg,'seed':seed,'_training_ids':raw.index.to_numpy(),'_holdout_ids':holdout,'train_rows':len(raw)}
    if smoke:local['full_balance_target']=60
    return X,y,raw,holdout,parts,num,cat,local,ref

def fit_stage(run,name):
    verify_sources(run);settings=read(run/'configuration_freeze.json');dest=run/'stages'/name;dest.mkdir()
    X,y,raw,holdout,parts,num,cat,local,ref=stage_inputs(name,settings['smoke'])
    a,g,c,v=parts
    np.savez_compressed(dest/'split.npz',**{k:raw.index[p].to_numpy() for k,p in zip('AGCV',parts)},historical_holdout_ids=holdout)
    with threadpool_limits(limits=1):
        prep,Z,yy,plan=prepare(X.iloc[a],y[a],'balanced','linear',num,cat,local,raw.index[np.r_[g,c,v]])
        Z=Z.tocsr();Zg=prep.transform(X.iloc[g]).tocsr();basegamma=scale_gamma(Z)
        e.write_json(dest/'training_plan.json',dict(plan=plan,numeric=num,categorical=cat,feature_order=list(X.columns),gamma_scale=basegamma,source_ids=raw.index.to_numpy(),preprocessing_fit_role='A',resampling_role='A'))
        specs=[s for s in candidate_specs() if s['id']=='C1_G1'] if settings['smoke'] else candidate_specs()
        rows=[];gp={};models={};manifest=[]
        for spec in specs:
            est=SVC(C=spec['C'],gamma=basegamma*spec['gamma_multiplier'],kernel='rbf',probability=False,tol=.001,max_iter=-1,cache_size=512,shrinking=True,class_weight=None,random_state=local['seed'])
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);est.fit(Z,yy)
            seconds=time.perf_counter()-start
            if any(issubclass(w.category,ConvergenceWarning) for w in caught) or est.fit_status_!=0:raise ValueError('SVM convergence failed: '+spec['id'])
            np.testing.assert_array_equal(est.classes_,np.arange(5))
            pred=est.predict(Zg);m,_=label_metrics(y[g],pred);gp[spec['id']]=pred
            row=dict(**spec,role='G',score=.5*(m['macro_f1']+m['ddos_f2']),macro_f1=m['macro_f1'],ddos_f2=m['ddos_f2'],normal_fpr=m['normal_fpr'],fit_seconds=seconds,n_support=int(est.n_support_.sum()),gamma=est._gamma,fit_status=int(est.fit_status_),iterations=est.n_iter_.tolist())
            pipe=Pipeline([('preprocessor',prep),('estimator',est)]);model=dest/(spec['id']+'.joblib');joblib.dump(pipe,model,compress=1)
            manifest.append(dict(candidate=spec['id'],path=model.relative_to(run).as_posix(),sha256=e.sha256(model),bytes=model.stat().st_size));rows.append(row);models[spec['id']]=pipe
            pd.DataFrame(rows).to_csv(dest/'G_candidates.csv',index=False)
            print('CANDIDATE',name,spec['id'],'fit_s',round(seconds,3),'G_F1',round(100*m['macro_f1'],4),flush=True)
        selected=select_on_G(rows);default='C1_G1'
        np.savez_compressed(dest/'G_predictions.npz',row_ids=raw.index[g],y=y[g],**gp)
        e.write_json(dest/'models_manifest.json',manifest)
        frozen=dict(selected=selected,default=default,selection_role='G',evaluation_role='V',models=manifest,selected_G_score=next(r['score'] for r in rows if r['id']==selected),frozen_unix=time.time(),V_scored=False)
        e.write_json(dest/'selection_before_V.json',frozen)
        evals={'SVM_default':models[default].predict(X.iloc[v]),'SVM_G_selected':models[selected].predict(X.iloc[v])}
    lookup={int(i):j for j,i in enumerate(ref['row_ids'])};positions=np.array([lookup[int(i)] for i in raw.index[v]])
    p=ref['p_experts'][positions]
    if 'weights' in ref:w=ref['weights']
    else:w=read(gated('reference',f'stages/{name}/legacy25_multiscale60__balanced/selection_before_V.json'))['weights']
    evals.update(WV=mixture(p,w).argmax(1),RF=p[:,1].argmax(1),LR=p[:,4].argmax(1),Uniform=p.mean(1).argmax(1))
    group='packet_private' if name.startswith('private') else 'temporal' if name.startswith('temporal') else 'signature_shared'
    result=[];arrays=dict(row_ids=raw.index[v].to_numpy(),y=y[v])
    for method,pred in evals.items():
        m,cm=label_metrics(y[v],pred);result.append(dict(stage=name,protocol=group,method=method,**m));arrays[method]=pred;arrays['CM_'+method]=cm
    np.savez_compressed(dest/'evaluation.npz',**arrays)
    pd.DataFrame(result).to_csv(dest/'metrics.csv',index=False)
    e.write_json(dest/'complete.json',dict(status='PASS_SVM_STAGE',selected=selected,scientific_fit=not settings['smoke'],SVC_fits=len(rows),V_scored_after_freeze=True,completed_unix=time.time()))
    verify_sources(run);print('STAGE_COMPLETE',name,'selected',selected,flush=True)
    del models,X;gc.collect()

def seal(run,status):
    verify_sources(run)
    records=[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ['catalog.json','complete.json']]
    e.write_json(run/'catalog.json',records)
    stages=read(run/'configuration_freeze.json')['stages']
    e.write_json(run/'complete.json',dict(status=status,stages=stages,new_SVC_fits=sum(read(run/'stages'/s/'complete.json')['SVC_fits'] for s in stages),new_WV_fits=0,WV_unchanged=True,RF_retained=True,catalog_sha256=e.sha256(run/'catalog.json')))

def main(run,smoke):
    assert run.parent==(ROOT/'runs').resolve() and not run.exists()
    stages=['private_fold1'] if smoke else STAGES
    # Resolve parent hash gates before any fit; canonical identity is checked by load_data.
    for parent in PARENTS:
        folder,name,digest=PARENTS[parent]
        assert e.sha256(ROOT/'runs'/folder/name)==digest,parent
    assert e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv')==e.DEFAULT_CONFIG['dataset_sha256']
    run.mkdir();(run/'source').mkdir();(run/'stages').mkdir()
    manifest=[]
    for rel in SOURCES:
        path=ROOT/rel;shutil.copy2(path,run/'source'/path.name);manifest.append(dict(path=rel,sha256=e.sha256(path)))
    e.write_json(run/'source_manifest.json',manifest);e.write_json(run/'parents.json',PARENTS);e.write_json(run/'environment.json',e.environment())
    e.write_json(run/'configuration_freeze.json',dict(date='2026-11-18',experiment='standalone SVM comparator; WV composition unchanged',smoke=smoke,stages=stages,candidates=candidate_specs(),selection='G only',default='C1_G1',expected_new_SVC_fits=len(stages)*(1 if smoke else 9),kernel='rbf',probability=False,max_iter=-1,threads_per_process=1,max_training_workers=2,new_data=False,RF_retained=True,protocol_sha256=e.sha256(ROOT/'docs/ARRAY_SVM_COMPARATOR_PROTOCOL_20261118.md'),frozen_unix=time.time()))
    def launch(name):
        with (run/(name+'.log')).open('x',encoding='utf-8') as f:
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--output',str(run),'--stage',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError('Stage failed; see '+name+'.log')
        print('DONE',name,flush=True)
    with ThreadPoolExecutor(max_workers=1 if smoke else 2) as pool:list(pool.map(launch,stages))
    pd.concat([pd.read_csv(run/'stages'/s/'metrics.csv') for s in stages],ignore_index=True).to_csv(run/'metrics.csv',index=False)
    seal(run,'PASS_SVM_SMOKE' if smoke else 'PASS_SVM_PREDICTIVE_EXPERIMENT')
    print('SVM_PREDICTIVE_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--stage');a=p.parse_args()
    if a.stage:fit_stage(a.output.resolve(),a.stage)
    else:main(a.output.resolve(),a.smoke)
