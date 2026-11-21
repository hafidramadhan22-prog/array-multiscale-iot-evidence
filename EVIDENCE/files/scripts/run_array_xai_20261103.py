"""Nine-group exact donor-game audit, same owned predictor/targets/backgrounds."""
from pathlib import Path
import sys,json,argparse,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_factorial_20261103 import checked,OLD,read_npz
from run_wb_research import load_data
from multiscale_context import repair_frame
from xai_audit import semantic_groups,attribution_distance,METHODS,exact_shapley
from run_xai_audit import group_evaluate,save_group,metric_rows,model_output_indices,verify_catalog

def main(out,smoke):
    assert not out.exists()
    src=ROOT/'runs/XAI_BACKGROUND_FULL_20261020_01';verify_catalog(src)
    idx=read_npz(src/'indices.npz')
    raw,y,holdout,data=load_data(dict(data_folder='runs/MULTISCALE_DATA_20261020_01'))
    split=read_npz(checked(OLD,'stages/temporal_seed42/split.npz'))
    ids=idx['row_ids'][:2] if smoke else idx['row_ids']
    assert set(ids)<=set(split['V']) and not set(ids)&set(holdout)
    assert set(idx['background_A'])<=set(split['A']) and set(idx['background_B'])<=set(split['A'])
    assert not set(idx['background_A'])&set(idx['background_B'])
    model=checked(OLD,'stages/temporal_seed42/legacy25_multiscale60__balanced/model.joblib')
    bundle=joblib.load(model)
    X,_,_=repair_frame(raw,data,'legacy25_multiscale60');pos=raw.index.get_indexer(ids)
    protocol=semantic_groups([c for c in X.columns if not c.startswith('multiscale_')]);groups=dict(protocol)
    for ms in [100,1000,10000]:groups['history_'+str(ms)+'ms']=[c for c in X.columns if c.startswith('multiscale_') and '_'+str(ms)+'ms_' in c]
    assert len(groups)==9 and all(len(groups[k])==20 for k in groups if k.startswith('history'))
    out.mkdir();(out/'source').mkdir()
    for rel in ['scripts/run_array_xai_20261103.py','src/xai_audit.py','scripts/run_xai_audit.py','docs/ARRAY_MAJOR_REVISION_PROTOCOL_20261103.md']:
        shutil.copy2(ROOT/rel,out/'source'/Path(rel).name)
    e.write_json(out/'configuration_freeze.json',dict(groups=groups,model_sha256=e.sha256(model),model=model.relative_to(ROOT).as_posix(),
        indices_sha256=e.sha256(src/'indices.npz'),rows=len(ids),background_size=32,smoke=smoke,new_classifier_fits=0))
    results={};allrows=[];dist=[];importance=[]
    with threadpool_limits(limits=1):
        for name,key in [('A32','background_A'),('B32','background_B')]:
            result=group_evaluate(bundle,X.loc[ids],X.loc[idx[key]],groups,ids,'ARRAY9_'+name)
            save_group(out/'groups'/name,result,ids,y[pos],groups,idx[key]);results[name]=result
        targets=results['A32']['values'][:,-1].argmax(2)
        for donor,rank in [('A32','A32'),('B32','B32'),('B32','A32')]:
            rows,_=metric_rows(results[donor],ids,y[pos],targets,ranking_result=results[rank],donor=donor,ranking_donor=rank);allrows+=rows
        for i,rid in enumerate(ids):
            for method,j in zip(METHODS,model_output_indices()):
                target=targets[i,j]
                dist.append(dict(row_id=int(rid),method=method,**attribution_distance(results['A32']['phi'][i,:,j,target],results['B32']['phi'][i,:,j,target])))
        for name,result in results.items():
            for i,rid in enumerate(ids):
                for method,j in zip(METHODS,model_output_indices()):
                    for k,group in enumerate(groups):importance.append(dict(background=name,row_id=int(rid),method=method,group=group,phi=float(result['phi'][i,k,j,targets[i,j]])))
            for i in range(len(ids)):np.testing.assert_allclose(exact_shapley(result['values'][i]),result['phi'][i],rtol=1e-9,atol=1e-11)
    pd.DataFrame(allrows).to_csv(out/'faithfulness.csv',index=False);pd.DataFrame(dist).to_csv(out/'background_sensitivity.csv',index=False)
    pd.DataFrame(importance).to_csv(out/'importance.csv',index=False)
    e.write_json(out/'catalog.json',[dict(path=p.relative_to(out).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
    e.write_json(out/'complete.json',dict(status='PASS_XAI_SMOKE' if smoke else 'PASS_XAI_FULL',games=len(ids)*2,coalitions=512,training=False,catalog_sha256=e.sha256(out/'catalog.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.output.resolve(),a.smoke)
