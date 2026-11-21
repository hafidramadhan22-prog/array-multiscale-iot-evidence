"""Independent reconstruction of every G candidate and frozen V prediction."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_array_svm_20261118 import score,compare,sha,load

STAGES=['private_fold1','private_fold2','temporal_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']
NEW=['WV_G_constrained','WV_G_unconstrained','WV_G_heuristic']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def mix(p,w):return sum(float(w[j])*p[:,j] for j in range(5))
def replay_G(p,w):
    result=np.empty((len(w),len(p)),dtype=np.uint8)
    for start in range(0,len(w),48):
        chunk=w[start:start+48];s=np.zeros((len(chunk),len(p),5),dtype=np.float64)
        for j in range(5):s=s+chunk[:,j,None,None]*p[None,:,j,:]
        result[start:start+len(chunk)]=s.argmax(2)
    return result
def best(scores,w,ref,ok):
    ids=np.flatnonzero(ok);return int(min(ids,key=lambda i:(-scores[i],float(np.abs(w[i]-ref).sum()),int(i))))

def main(root,out):
    run=root/'runs/ARRAY_WEIGHT_RETUNE_FULL_20261118_01';report=root/'reports/ARRAY_WEIGHT_RETUNE_20261118'
    done=read(run/'complete.json');assert done['status']=='PASS_WEIGHT_RETUNE_EXPERIMENT' and done['catalog_sha256']==sha(run/'catalog.json')
    cat=read(run/'catalog.json')
    for r in cat:
        p=run/r['path'];assert p.resolve().is_relative_to(run.resolve()) and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
    cfg=read(run/'configuration_freeze.json');assert cfg['grid_step']==cfg['grid_floor']==.05 and cfg['weight_shrinkage']==.5
    assert sha(run/'source/ARRAY_WEIGHT_RETUNE_PROTOCOL_20261118.md')==cfg['source_protocol_sha256']
    parents=read(run/'parent_catalogs.json');original={}
    for key,(_,name,digest) in parents.items():
        file=run/'inputs'/(key+'_'+name);assert sha(file)==digest
        original[key]={r['path'].replace('\\','/'):r for r in read(file)}
    parent_count=0
    for key,catalog in original.items():
        folder=run/'inputs'/key
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(folder).as_posix();r=catalog[rel];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'];parent_count+=1
    svmcatalog=report/'parent_SVM_catalog.json';assert sha(svmcatalog)=='fb15166077926690c253a0c995f8a62d620df5824201e2ff5c7b9fd18182ddc1'
    svmlist={r['path']:r for r in read(svmcatalog)}
    for p in (run/'inputs/SVM').rglob('*.npz'):
        r=svmlist[p.relative_to(run/'inputs/SVM').as_posix()];assert sha(p)==r['sha256']
    grid=np.array([[a,b,c,d,15-a-b-c-d] for a in range(16) for b in range(16-a) for c in range(16-a-b) for d in range(16-a-b-c)],dtype=float)*.05+.05
    assert len(grid)==3876
    metrics=pd.read_csv(run/'metrics.csv');summary=pd.read_csv(report/'protocol_summary.csv');vectors=0;probabilities=0;outputs={}
    for name in STAGES:
        private=name.startswith('private');original_name='signature_fold'+name[-1] if private else name
        parent='history_models' if private else 'reference';base=original_name+'_exclusive' if private else 'stages/'+name+'/legacy25_multiscale60__balanced'
        inputs=run/'inputs'/parent/base;cal=load(inputs/'calibration.npz');v=load(inputs/'evaluation.npz')
        selmeta=read(inputs/('frozen_before_V.json' if private else 'selection_before_V.json'));reference=np.asarray(selmeta['weights'])
        stage=run/'stages'/name;z=load(stage/'G_search.npz');split=load(stage/'split.npz');frozen=read(stage/'selection_before_V.json');g=pd.read_csv(stage/'G_candidates.csv');weights=z['weights'];assert len(g)==len(weights)==3881
        assert len(np.unique(np.concatenate([split[k] for k in 'AGCV'])))==sum(len(split[k]) for k in 'AGCV')
        assert not set(np.concatenate([split[k] for k in 'AGCV']))&set(split['historical_holdout_ids'])
        np.testing.assert_array_equal(z['row_ids'],split['G']);np.testing.assert_array_equal(z['row_ids'],cal['G_row_ids']);np.testing.assert_array_equal(z['y'],cal['G_y'])
        np.testing.assert_array_equal(weights[0],reference);np.testing.assert_allclose(weights[1],.2,atol=0,rtol=0);np.testing.assert_array_equal(weights[5:],grid)
        assert (weights>=.05-1e-15).all();np.testing.assert_allclose(weights.sum(1),1,atol=1e-12,rtol=0)
        pG=cal['G_experts'];quality=np.array([score(cal['G_y'],pG[:,j].argmax(1))[0]['macro_f1'] for j in range(5)]);quality/=quality.max()
        costs=read(run/'inputs/reference/stages'/original_name/'legacy25_multiscale60__balanced/selection_before_V.json')['costs']
        lat=np.asarray(costs['training_pipeline_latency_seconds']);size=np.asarray(costs['serialized_pipeline_bytes']);assert (lat>0).all() and (size>0).all();eff=.5*(lat.min()/lat+size.min()/size)
        for i,a in enumerate([.5,.6,.7]):
            raw=a*quality+(1-a)*eff;w=raw/raw.sum();w=.1+.5*w;w=np.maximum(w,.05);w/=w.sum();np.testing.assert_allclose(weights[2+i],w,rtol=0,atol=1e-14)
        labels=replay_G(pG,weights);np.testing.assert_array_equal(labels,z['labels'])
        cm=np.bincount((25*np.arange(len(labels))[:,None]+5*z['y'][None,:]+labels).ravel(),minlength=25*len(labels)).reshape(-1,5,5);np.testing.assert_array_equal(cm,z['CM'])
        tp=np.diagonal(cm,axis1=1,axis2=2);sup=cm.sum(2);fp=cm.sum(1)-tp;fn=sup-tp
        f1=(2*tp/(2*tp+fp+fn)).mean(1);f2=5*tp[:,1]/(5*tp[:,1]+4*fn[:,1]+fp[:,1]);scores=.5*(f1+f2)
        np.testing.assert_allclose(g.G_macro_f1,f1,rtol=0,atol=1e-12);np.testing.assert_allclose(g.G_score,scores,rtol=0,atol=1e-12)
        ok=(fn[:,0]<=fn[0,0])&(tp[:,1:]>=tp[0,1:]).all(1);np.testing.assert_array_equal(ok,z['G_feasible']);np.testing.assert_array_equal(ok,g.G_feasible)
        assert frozen['selected']['WV_G_constrained']==best(scores,weights,reference,ok)
        assert frozen['selected']['WV_G_unconstrained']==best(scores,weights,reference,np.ones(len(weights),dtype=bool))
        # Legacy heuristic uses its own inherited alpha tie rule.
        hids=[2,3,4];hbest=max(hids,key=lambda i:(scores[i],-abs([.5,.6,.7][i-2]-.6)))
        assert frozen['selected']['WV_G_heuristic']==hbest
        assert ok[frozen['selected']['WV_G_constrained']] and frozen['V_scored'] is False
        assert read(stage/'complete.json')['completed_unix']>=frozen['frozen_unix']
        ev=load(stage/'evaluation.npz');np.testing.assert_array_equal(ev['row_ids'],v['row_ids']);np.testing.assert_array_equal(ev['row_ids'],split['V']);np.testing.assert_array_equal(ev['y'],v['y'])
        for method,i in frozen['selected'].items():
            p=mix(v['p_experts'],weights[i]);np.testing.assert_allclose(p,ev['p_'+method],rtol=0,atol=1e-15);np.testing.assert_array_equal(p.argmax(1),ev[method]);probabilities+=1
        svm=load(run/'inputs/SVM/stages'/name/'evaluation.npz');np.testing.assert_array_equal(ev['row_ids'],svm['row_ids']);np.testing.assert_array_equal(ev['WV_reference'],svm['WV'])
        for method in ['RF','LR','SVM_default','SVM_G_selected']:np.testing.assert_array_equal(ev[method],svm[method])
        methods=[k for k in ev if not k.startswith(('CM_','p_')) and k not in ['row_ids','y']]
        for method in methods:
            m,matrix=score(ev['y'],ev[method]);np.testing.assert_array_equal(matrix,ev['CM_'+method]);compare(metrics[(metrics.stage==name)&(metrics.method==method)].iloc[0],m);vectors+=1
        outputs[name]=ev
    for prefix,protocol in [('private','packet_private'),('signature','signature_shared')]:
        zs=[outputs[f'{prefix}_fold{k}'] for k in [1,2]];assert len(set(zs[0]['row_ids'])&set(zs[1]['row_ids']))==0
        for method in metrics.method.unique():
            m,cm=score(np.concatenate([z['y'] for z in zs]),np.concatenate([z[method] for z in zs]));compare(summary[(summary.protocol==protocol)&(summary.method==method)].iloc[0],m)
    for method in metrics.method.unique():
        m=metrics[(metrics.protocol=='temporal')&(metrics.method==method)];s=summary[(summary.protocol=='temporal')&(summary.method==method)].iloc[0]
        for key in ['macro_f1','normal_fpr','balanced_accuracy','errors']:assert np.isclose(m[key].mean(),s[key],rtol=0,atol=1e-12)
    sm=pd.read_csv(run/'stress_metrics.csv');freeze=read(run/'stages/temporal_seed42/selection_before_V.json');stress_vectors=0
    for cohort in sm.cohort.unique():
        z=load(run/'stress'/(cohort+'.npz'));assert z['row_ids'].dtype.kind in 'US'
        if cohort.startswith('aligned_'):
            source=cohort[len('aligned_'):];p=load(run/'inputs/aligned'/(source+'_predictions.npz'));cube=p['p_after'];truth=np.zeros(len(cube),dtype=int)
        else:
            p=load(run/'inputs/guard/posteriors'/(cohort[len('block3_'):]+'_later.npz')) if cohort.startswith('block3_') else load(run/'inputs/external'/cohort/'probabilities.npz')
            cube=np.stack([p[n] for n in ['decision_tree','random_forest','xgboost','lightgbm','logistic_regression']],axis=1);truth=p['y_scenario'] if 'y_scenario' in p else p['y']
        np.testing.assert_array_equal(z['y'],truth)
        for method,w in freeze['weights'].items():np.testing.assert_allclose(z['p_'+method],mix(cube,w),rtol=0,atol=1e-15);np.testing.assert_array_equal(z[method],mix(cube,w).argmax(1))
        for method in sm.method.unique():
            m,cm=score(z['y'],z[method]);compare(sm[(sm.cohort==cohort)&(sm.method==method)].iloc[0],m);np.testing.assert_array_equal(cm,z['CM_'+method]);stress_vectors+=1
    analysis=read(report/'analysis.json');assert not analysis['primary_strict_success_all_protocols'] and analysis['RF_retained'] and not analysis['old_runtime_SHAP_contracts_inherited']
    result=dict(status='PASS_INDEPENDENT_WEIGHT_SEARCH_AND_PREDICTIONS',sealed_files=len(cat),parent_artifacts_verified=parent_count,G_candidate_evaluations=7*3881,G_all_candidate_labels_recomputed=True,stage_label_vectors=vectors,new_probability_vectors=probabilities,stress_label_vectors=stress_vectors,new_classifier_fits=0,models_deserialized=False,RF_and_both_SVMs_retained=True,primary_consistent_success=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args();main(a.root.resolve(),(a.output or a.root/'reports/ARRAY_WEIGHT_RETUNE_20261118/verification.json').resolve())
