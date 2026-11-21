"""Predict with verified project-owned pipelines; accepts prepared context85 CSV.

No fitting or threshold tuning. This is a separate candidate, not the dashboard.
"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sql_guard_rf_rescue_20261109 import infer,combine
from rf_normal_voting_type_v2_20261109 import full_voting
from fpr_target8_20261109 import NAMES,CLASSES
OWNED_REFERENCE='77aa1beb6980f4eb9db9838af6505949899c5aa574f3be49d55adfdc1efcf31d'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text('utf8'))

def verify_run(run):
    complete=read(run/'complete.json');assert complete['status']=='FPR_SQL_GUARD_COMPLETE'
    assert sha(run/'artifacts_sha256.json')==complete['catalog_sha256']
    for row in read(run/'artifacts_sha256.json'):
        path=(run/row['path']).resolve();assert path.is_relative_to(run)
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    for manifest in ['source_manifest.json']:
        for row in read(run/manifest):assert sha(ROOT/row['path'])==row['sha256'],row['path']
    row=next(x for x in read(run/'models_manifest.json') if x['key']=='sql_guard_rf_rescue')
    path=run/row['path'];assert row['sha256']==OWNED_REFERENCE and sha(path)==OWNED_REFERENCE
    return path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',type=Path,default=ROOT/'runs/FPR_SQL_GUARD_20261109_01');args=parser.parse_args()
    assert not args.output.exists(),'Output already exists; choose a new path'
    schema=read(ROOT/'configs/fpr_normal_voting_schema_20261109.json');assert schema['owned_reference_sha256']==OWNED_REFERENCE
    model_path=verify_run(args.run.resolve())
    frame=pd.read_csv(args.features,index_col='row_id',dtype={c:str for c in schema['categorical']},keep_default_na=False,float_precision='round_trip')
    expected=schema['numeric']+schema['categorical']
    assert len(frame)>0 and frame.index.is_unique and not frame.index.isna().any()
    assert set(frame.columns)==set(expected) and len(frame.columns)==85,'Exactly the prepared context85 features are required, without labels/IDs/payload fields'
    frame=frame[expected]
    for col in schema['numeric']:frame[col]=pd.to_numeric(frame[col],errors='raise')
    assert np.isfinite(frame[schema['numeric']].to_numpy(float)).all(),'Nonfinite numeric input'
    import joblib
    bundle=joblib.load(model_path)
    with threadpool_limits(limits=1):
        p=infer(bundle,frame,NAMES);rf=bundle['models']['random_forest'].predict_proba(frame);wv=full_voting(bundle,frame,NAMES);gold,reason=combine(wv,rf)
    np.testing.assert_allclose(p,gold,rtol=0,atol=1e-12)
    out=pd.DataFrame(dict(row_id=frame.index,weighted_prediction=[CLASSES[i] for i in wv.argmax(1)],RF_prediction=[CLASSES[i] for i in rf.argmax(1)],candidate_prediction=[CLASSES[i] for i in p.argmax(1)],reason=reason))
    for j,c in enumerate(CLASSES):out['p_'+c]=p[:,j]
    args.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)
    print(json.dumps(dict(status='PASS',n=len(out),candidate_counts=out.candidate_prediction.value_counts().to_dict(),model_sha256=OWNED_REFERENCE,max_probability_difference=float(np.max(abs(p-gold))),new_fits=0)))

if __name__=='__main__':main()
