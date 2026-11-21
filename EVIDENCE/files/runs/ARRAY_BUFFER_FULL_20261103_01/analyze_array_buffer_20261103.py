"""Fixed fitted-predictor buffer/warm-up controls on identical targets."""
from pathlib import Path
import json,sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_factorial_20261103 import checked,OLD,read_npz,methods
from run_array_history_models_20261103 import verify_data
from run_wb_research import load_data,parts_for
from array_revision import frame,scored
from wb_research import cube

def main():
    out=ROOT/'runs/ARRAY_BUFFER_FULL_20261103_01';assert not out.exists()
    source=ROOT/'runs/ARRAY_HISTORY_DATA_20261103_01';verify_data(source);h=read_npz(source/'histories.npz')
    raw,y,holdout,data=load_data(dict(data_folder='runs/MULTISCALE_DATA_20261020_01'));np.testing.assert_array_equal(h['row_ids'],raw.index)
    out.mkdir();shutil.copy2(__file__,out/Path(__file__).name)
    e.write_json(out/'configuration_freeze.json',dict(source_catalog_sha256=e.sha256(source/'catalog.json'),capacities=[4096,16384,65536],refit=False,warmup='1s reference; 10s removes early decisions without changing retained history; common-target predictions identical by definition'))
    rows=[];effects=[]
    with threadpool_limits(limits=1):
        for name in ['temporal_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']:
            parts,_=parts_for(name,raw,y,holdout,data,False);v=parts[3];prefix=f'stages/{name}/legacy25_multiscale60__balanced/'
            bundle=joblib.load(checked(OLD,prefix+'model.joblib'))
            for cap in [4096,16384,65536]:
                values=data['prior_multiscale'] if cap==16384 else h[f'buffer{cap}']
                if cap==16384:
                    cached=read_npz(checked(OLD,prefix+'evaluation.npz'));np.testing.assert_array_equal(cached['row_ids'],raw.index[v]);pv=cached['p_experts']
                else:
                    X,_,_=frame(raw,values,'c60_o10');pv=cube(bundle['models'],X.iloc[v])
                np.savez_compressed(out/f'{name}_{cap}.npz',row_ids=raw.index[v],y=y[v],p_experts=pv,weights=bundle['weights'],warm10=h['warm10'][v],truncated=h[f'truncated_{cap}'][v])
                for method,p in methods(pv,bundle['weights']).items():
                    for cohort,mask in [('warm1',np.ones(len(v),bool)),('common10',h['warm10'][v]),('early_excluded',~h['warm10'][v])]:
                        if len(np.unique(y[v][mask]))!=5:
                            effects.append(dict(stage=name,capacity=cap,cohort=cohort,n=int(mask.sum()),five_classes=False));continue
                        rows.append(dict(stage=name,capacity=cap,cohort=cohort,method=method,truncated_fraction=float(h[f'truncated_{cap}'][v][mask].mean()),**scored(y[v][mask],p[mask].argmax(1))))
            print('BUFFER_SCORED',name,flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);e.write_json(out/'noncomparable_cohorts.json',effects)
    e.write_json(out/'catalog.json',[dict(path=p.relative_to(out).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
    e.write_json(out/'complete.json',dict(status='PASS_BUFFER',new_fits=0,catalog_sha256=e.sha256(out/'catalog.json')))
if __name__=='__main__':main()
