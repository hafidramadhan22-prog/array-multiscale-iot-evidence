"""Frozen-model flag-format diagnostic; no fit/selection, original targets unchanged."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_factorial_20261103 import read_npz,checked,OLD,methods
from run_array_history_models_20261103 import verify_data
from wb_research import cube
from e4 import columns

FIELDS=['tcp.flags','mqtt.conflags','mqtt.hdrflags']
def semantic_flag(text):
    text=str(text).strip()
    return e.normalize_absent(str(float(int(text,16)))) if text.lower().startswith('0x') else text

def main():
    source=ROOT/'runs/ARRAY_BENIGN_CAPTURE_20261103_04';verify_data(source)
    out=ROOT/'runs/ARRAY_FLAG_FORMAT_20261104_01';assert not out.exists()
    path=checked(OLD,'stages/temporal_seed42/legacy25_multiscale60__balanced/model.joblib')
    freeze=json.loads((source/'configuration_freeze.json').read_text());assert e.sha256(path)==freeze['model_sha256']
    out.mkdir();(out/'source').mkdir()
    for rel in ['scripts/audit_array_flag_format_20261104.py','docs/ARRAY_FLAG_FORMAT_DIAGNOSTIC_20261104.md']:shutil.copy2(ROOT/rel,out/'source'/Path(rel).name)
    e.write_json(out/'configuration_freeze.json',dict(model=str(path.relative_to(ROOT)),model_sha256=e.sha256(path),source_catalog_sha256=e.sha256(source/'catalog.json'),changed_fields=FIELDS,normalization='hex to decimal-float string then unchanged missing handling',new_fits=0,targets=760,unscorable_excluded=107,selection=False))
    bundle=joblib.load(path);_,cat=columns('noport25');metrics=[];audit=[];decisions=[]
    with threadpool_limits(limits=1):
        for rec in json.loads((source/'source_audit.json').read_text()):
            name=rec['source'];old=read_npz(source/(name+'_predictions.npz'));ids=old['record_index'].astype(int)
            if not len(ids):continue
            X=pd.read_csv(source/(name+'_features.csv'),index_col='record_index',dtype={c:str for c in cat})
            p0=cube(bundle['models'],X);np.testing.assert_allclose(p0,old['p_experts'],rtol=0,atol=1e-12)
            fixed=X.copy();changes={}
            for c in FIELDS:
                fixed[c]=fixed[c].map(semantic_flag);changes[c]=int((X[c]!=fixed[c]).sum())
            assert fixed.drop(columns=FIELDS).equals(X.drop(columns=FIELDS))
            p1=cube(bundle['models'],fixed)
            fixed.to_csv(out/(name+'_features.csv'),index_label='record_index')
            np.savez_compressed(out/(name+'_predictions.npz'),record_index=ids,p_before=p0,p_after=p1,weights=old['weights'])
            for variant,p in [('historical_format',p0),('semantic_format',p1)]:
                for method,prob in methods(p,old['weights']).items():
                    pred=prob.argmax(1);metrics.append(dict(source=name,format=variant,method=method,n=len(pred),false_alarms=int((pred!=0).sum()),**{f'predicted_{i}':int((pred==i).sum()) for i in range(5)}))
                    decisions.extend(dict(source=name,record_index=int(i),format=variant,method=method,predicted=int(lab),true_class=0) for i,lab in zip(ids,pred))
            audit.append(dict(source=name,n=len(ids),changed_cells=changes,max_probability_error_original=float(np.max(abs(p0-old['p_experts'])))))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(decisions).to_csv(out/'predictions.csv',index=False);e.write_json(out/'audit.json',audit)
    sums=pd.DataFrame(metrics).groupby(['format','method'])[['n','false_alarms']].sum();sums['fpr_percent']=100*sums.false_alarms/sums.n;sums.to_csv(out/'summary.csv')
    e.write_json(out/'catalog.json',[dict(path=p.relative_to(out).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
    e.write_json(out/'complete.json',dict(status='PASS_FROZEN_FLAG_FORMAT_DIAGNOSTIC',new_fits=0,new_model_loads=1,targets=760,changed_fields=FIELDS,original_predictions_reproduced=True,all_other_inputs_equal=True,catalog_sha256=e.sha256(out/'catalog.json')))
    print(sums.to_string())
if __name__=='__main__':main()
