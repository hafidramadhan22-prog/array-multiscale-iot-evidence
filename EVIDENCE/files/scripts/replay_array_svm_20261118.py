"""Explicit opt-in replay of this project's hash-verified SVM models; no training."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def main(root,out):
    train=root/'runs/ARRAY_SVM_FULL_20261118_01';stress=root/'runs/ARRAY_SVM_STRESS_PORTABLE_20261118_01';serve=root/'runs/ARRAY_SVM_SERVING_20261118_01'
    done=read(train/'complete.json');assert done['catalog_sha256']==sha(train/'catalog.json')
    for r in read(train/'catalog.json'):
        p=train/r['path'];assert p.resolve().is_relative_to(train.resolve()) and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    d=train/'stages/temporal_seed42';freeze=read(d/'selection_before_V.json');manifest=read(d/'models_manifest.json');plan=read(d/'training_plan.json')
    checks=[];total=0
    def X(path):
        a=pd.read_csv(path,index_col='row_id',dtype={c:str for c in plan['categorical']},keep_default_na=False,float_precision='round_trip');a=a[plan['feature_order']]
        for c in plan['numeric']:a[c]=pd.to_numeric(a[c],errors='coerce')
        return a
    with threadpool_limits(limits=1):
        for method,key in [('SVM_default','default'),('SVM_G_selected','selected')]:
            rec=next(r for r in manifest if r['candidate']==freeze[key]);p=train/rec['path'];assert sha(p)==rec['sha256']
            # These are the project-owned candidates just trained under the frozen protocol.
            model=joblib.load(p)
            for inputfile in sorted((stress/'inputs').glob('*.csv')):
                with np.load(stress/'predictions'/(inputfile.stem+'.npz'),allow_pickle=False) as z:gold=z[method]
                observed=model.predict(X(inputfile));np.testing.assert_array_equal(observed,gold);total+=len(gold);checks.append(dict(method=method,cohort=inputfile.stem,rows=len(gold),labels_identical=True))
            gold=loadnp(serve/'expected.npz')
            for batch in [1,1000]:
                observed=model.predict(X(serve/'inputs'/f'batch{batch}.csv'));np.testing.assert_array_equal(observed,gold[method+'__'+str(batch)]);total+=batch
    result=dict(status='PASS_EXPLICIT_SVM_MODEL_REPLAY',label_checks=total,stress_vectors=len(checks),prepared_batches=4,models_loaded=2,new_fits=0,checks=checks)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('PASS_EXPLICIT_SVM_MODEL_REPLAY',total,'labels')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args();main(a.root.resolve(),(a.output or a.root/'reports/ARRAY_SVM_20261118/model_replay_verification.json').resolve())
