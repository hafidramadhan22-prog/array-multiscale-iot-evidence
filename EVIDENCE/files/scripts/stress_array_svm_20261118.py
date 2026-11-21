"""Post-freeze SVM stress checks on existing aligned benign and external inputs."""
from pathlib import Path
import sys,argparse,time,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_svm_20261118 import read,npz,gated
from evaluate_array_svm_20261118 import load_svm
from array_svm_20261118 import label_metrics
from wb_research import mixture
from profile_array_serving_20261117 import load as load_reference

TRAIN=ROOT/'runs/ARRAY_SVM_FULL_20261118_01'
EVAL=ROOT/'runs/ARRAY_SVM_EVALUATION_20261118_01'
EXTERNAL=['CIC_Normal0','CIC_Normal3','CIC_DDoS_SYN','CIC_PortScan','CIC_SQL','TON_Ransom1','TON_Ransom2']

def read_features(path,plan,index_col='row_id'):
    X=pd.read_csv(path,index_col=index_col,dtype={c:str for c in plan['categorical']},keep_default_na=False,float_precision='round_trip')
    assert set(X.columns)==set(plan['feature_order']) and X.index.is_unique
    X=X[plan['feature_order']]
    for c in plan['numeric']:X[c]=pd.to_numeric(X[c],errors='coerce')
    return X

def main(out):
    assert not out.exists() and read(EVAL/'complete.json')['status']=='PASS_SVM_MATCHED_EVALUATION'
    freeze=read(TRAIN/'stages/temporal_seed42/selection_before_V.json');plan=read(TRAIN/'stages/temporal_seed42/training_plan.json')
    out.mkdir();(out/'source').mkdir();(out/'inputs').mkdir();(out/'predictions').mkdir()
    sources=['scripts/stress_array_svm_20261118.py','scripts/evaluate_array_svm_20261118.py','scripts/run_array_svm_20261118.py','scripts/profile_array_serving_20261117.py','src/array_svm_20261118.py','docs/ARRAY_SVM_CACHE_ADAPTER_ADDENDUM_20261118.md']
    for rel in sources:shutil.copy2(ROOT/rel,out/'source'/Path(rel).name)
    e.write_json(out/'configuration_freeze.json',dict(date='2026-11-18',training_catalog_sha256=e.sha256(TRAIN/'catalog.json'),evaluation_catalog_sha256=e.sha256(EVAL/'catalog.json'),SVM_default=freeze['default'],SVM_G_selected=freeze['selected'],new_fits=0,retrospective_existing_cohorts=True,no_tuning_from_stress=True,frozen_unix=time.time(),sources=[dict(path=r,sha256=e.sha256(ROOT/r)) for r in sources]))
    rows=[];input_records=[];modelchecks=[]
    with threadpool_limits(limits=1):
        svms={name:load_svm(TRAIN,'temporal_seed42',freeze[k]) for name,k in [('SVM_default','default'),('SVM_G_selected','selected')]}
        _,bundle,objects,_,_=load_reference()
        def evaluate(name,family,scope,X,z,y):
            np.testing.assert_array_equal(X.index.astype(str),z['row_ids'].astype(str))
            # A successful old-model replay gates feature alignment before new scoring.
            p=objects['WV_native_full'].predict_proba(X)
            np.testing.assert_allclose(p,z['weighted_reference'],rtol=0,atol=1e-12)
            rf=bundle['models']['random_forest'].predict_proba(X)
            np.testing.assert_allclose(rf,z['random_forest'],rtol=0,atol=1e-12)
            modelchecks.append(dict(cohort=name,max_WV_probability_difference=float(np.max(np.abs(p-z['weighted_reference']))),max_RF_probability_difference=float(np.max(np.abs(rf-z['random_forest'])))))
            labels={key:model.predict(X) for key,model in svms.items()}
            labels.update(WV=z['weighted_reference'].argmax(1),RF=z['random_forest'].argmax(1),LR=z['logistic_regression'].argmax(1),Uniform=z['uniform_reference'].argmax(1))
            arrays=dict(row_ids=X.index.astype(str).to_numpy(),y=np.asarray(y,dtype=int))
            for method,pred in labels.items():
                m,cm=label_metrics(y,pred);rows.append(dict(cohort=name,family=family,label_scope=scope,method=method,**m));arrays[method]=pred;arrays['CM_'+method]=cm
            X.to_csv(out/'inputs'/(name+'.csv'),index_label='row_id')
            np.savez_compressed(out/'predictions'/(name+'.npz'),**arrays)
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print('SVM_STRESS_SCORED',name,len(X),flush=True)
        base=ROOT/'runs/ARRAY_FLAG_FORMAT_20261104_01'
        for f in sorted(base.glob('*_features.csv')):
            name=f.name[:-13];fp=gated('aligned',f.name);z0=npz(gated('aligned',name+'_predictions.npz'))
            X=read_features(fp,plan,'record_index');ids=X.index.astype(int).to_numpy();np.testing.assert_array_equal(ids,z0['record_index'])
            p=z0['p_after'];z=dict(row_ids=np.array([name+':'+str(i) for i in ids]),weighted_reference=mixture(p,z0['weights']),random_forest=p[:,1],logistic_regression=p[:,4],uniform_reference=p.mean(1))
            X.index=z['row_ids'];evaluate('aligned_'+name,'additional_aligned','Normal capture label',X,z,np.zeros(len(X),dtype=int))
            input_records.append(dict(cohort='aligned_'+name,source=fp.relative_to(ROOT).as_posix(),sha256=e.sha256(fp)))
        base=ROOT/'runs/FPR_SQL_GUARD_20261109_01'
        for f in sorted(base.glob('*_later_features.csv')):
            if f.name=='all_later_features.csv':continue
            name=f.name[:-19];fp=gated('guard',f.name);z=npz(gated('guard','posteriors/'+name+'_later.npz'))
            X=read_features(fp,plan);evaluate('block3_'+name,'later_block3','Normal capture label; existing sources',X,z,z['y'])
            input_records.append(dict(cohort='block3_'+name,source=fp.relative_to(ROOT).as_posix(),sha256=e.sha256(fp)))
        for name in EXTERNAL:
            fp=gated('external',name+'/features.csv');z=npz(gated('external',name+'/probabilities.npz'));X=read_features(fp,plan)
            y=z['y_scenario'] if 'y_scenario' in z else z['y']
            scope='CIC file-scenario agreement' if name.startswith('CIC') else 'TON author-flow-derived type labels'
            evaluate(name,'CIC' if name.startswith('CIC') else 'TON',scope,X,z,y)
            input_records.append(dict(cohort=name,source=fp.relative_to(ROOT).as_posix(),sha256=e.sha256(fp)))
    e.write_json(out/'input_provenance.json',input_records);e.write_json(out/'old_model_alignment_checks.json',modelchecks)
    # No fits, threshold updates or reinterpretation of external label scope.
    for record in read(out/'configuration_freeze.json')['sources']:assert e.sha256(ROOT/record['path'])==record['sha256']
    records=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=e.sha256(f)) for f in sorted(out.rglob('*')) if f.is_file()]
    e.write_json(out/'catalog.json',records);e.write_json(out/'complete.json',dict(status='PASS_SVM_POST_FREEZE_STRESS',cohorts=len(input_records),new_fits=0,independent_new_test=False,old_model_replay_alignment_passed=True,catalog_sha256=e.sha256(out/'catalog.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output.resolve())
