"""Use the verified compact model on already extracted domain features, without fit."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from wb_profile import checked_file,read_frame


def predict(run,input_path,output_path,posterior=False):
    run=run.resolve();input_path=input_path.resolve();output_path=output_path.resolve()
    if not output_path.is_relative_to(ROOT) or output_path.exists():raise ValueError('New output file inside workspace required')
    assert json.loads((run/'verification_independent.json').read_text())['status']=='PASS_RUNTIME_EVIDENCE_RECONSTRUCTION'
    done=json.loads((run/'complete.json').read_text());assert done['inputs_manifest_sha256']==e.sha256(run/'inputs_manifest.json')
    freeze=json.loads((run/'inputs_manifest.json').read_text());spec=freeze['policies']['baseline25__compact_certified']
    model=joblib.load(checked_file(run,spec['model'])) # only this run-owned, verified artifact
    X=read_frame(input_path,freeze['schemas'][spec['view']])
    with threadpool_limits(limits=1):
        if posterior:
            p=model.predict_proba(X);pred=p.argmax(1)
        else:pred=model.predict(X)
    out=pd.DataFrame({'row_id':X.index,'class_index':pred,'label':[e.CLASSES[int(v)] for v in pred]})
    if posterior:
        for j,label in enumerate(e.CLASSES):out['prob_'+label]=p[:,j]
    out.to_csv(output_path,index=False)
    print(json.dumps(dict(rows=len(out),output=str(output_path),full_posterior=posterior,
        new_training=False,raw_feature_extraction_included=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'runs/EXACT_RUNTIME_20261020_01')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-posterior',action='store_true',help='Calls all experts; use for probability/SHAP targets, not partial scheduler scores')
    a=p.parse_args();predict(a.run,a.input,a.output,a.full_posterior)
