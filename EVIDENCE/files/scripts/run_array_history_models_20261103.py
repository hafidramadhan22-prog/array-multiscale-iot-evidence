"""Fixed-budget refits for packet-private histories; no evaluation selection."""
from pathlib import Path
import argparse,json,sys,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_factorial_20261103 import checked,OLD,read_npz,methods
from run_wb_research import load_data,parts_for
from wb_research import fit_experts,cube
from sbfe import guard_roles
from array_revision import frame,scored

def verify_data(folder):
    done=json.loads((folder/'complete.json').read_text());assert done['catalog_sha256']==e.sha256(folder/'catalog.json')
    for r in json.loads((folder/'catalog.json').read_text()):
        p=folder/r['path'];assert p.resolve().is_relative_to(folder.resolve()) and p.stat().st_size==r['bytes'] and e.sha256(p)==r['sha256']

def main(out,smoke):
    assert not out.exists()
    source=ROOT/'runs/ARRAY_HISTORY_DATA_20261103_01';verify_data(source);h=read_npz(source/'histories.npz')
    cfg=json.loads((OLD/'configuration_freeze.json').read_text());raw,y,holdout,data=load_data(cfg)
    np.testing.assert_array_equal(h['row_ids'],raw.index)
    if smoke:
        for n in ['random_forest','xgboost','lightgbm']:cfg['params'][n]['n_estimators']=8
    out.mkdir();(out/'source').mkdir()
    for rel in ['scripts/run_array_history_models_20261103.py','src/array_revision.py','src/wb_research.py','docs/ARRAY_MAJOR_REVISION_PROTOCOL_20261103.md']:shutil.copy2(ROOT/rel,out/'source'/Path(rel).name)
    e.write_json(out/'configuration_freeze.json',dict(**cfg,smoke=smoke,history_catalog_sha256=e.sha256(source/'catalog.json'),weight_policy='fixed original C60/O10 per fold',new_selection_on_V=False));e.write_json(out/'environment.json',e.environment())
    rows=[]
    for fold in ([1] if smoke else [1,2]):
        name=f'signature_fold{fold}';parts,_=parts_for(name,raw,y,holdout,data,False)
        for k,pos in enumerate(parts):assert (h[f'fold{fold}_roles'][pos]==k).all()
        if smoke:parts=[np.concatenate([p[y[p]==cl][:15] for cl in range(5)]) for p in parts]
        a,g,c,v=parts;guard_roles(raw.index.to_numpy(),parts,holdout)
        selection=json.loads(checked(OLD,f'stages/{name}/legacy25_multiscale60__balanced/selection_before_V.json').read_text());weights=selection['weights']
        local={**e.DEFAULT_CONFIG,**json.loads((ROOT/'configs/target.json').read_text()),**cfg,'seed':42,'_training_ids':raw.index.to_numpy(),'_holdout_ids':holdout,'train_rows':len(raw)}
        if smoke:local['full_balance_target']=60
        for mode in ['exclusive','shared_density']:
            dest=out/f'{name}_{mode}';dest.mkdir()
            # Invalid one-second warm-up targets are never used in any supervised role.
            values=h[f'fold{fold}_{mode}'].copy();values[~data['valid_after_purge']]=0
            X,num,cat=frame(raw,values,'c60_o10')
            np.savez_compressed(dest/'split.npz',**{k:raw.index[pos].to_numpy() for k,pos in zip('AGCV',parts)})
            print('FIT',name,mode,len(a),len(v),flush=True)
            with threadpool_limits(limits=1):
                models,plans,records=fit_experts(X,y,a,np.r_[g,c,v],'balanced',num,cat,cfg['params'],local)
                pg=cube(models,X.iloc[g]);pc=cube(models,X.iloc[c])
                model=dest/'model.joblib';joblib.dump(dict(models=models,weights=weights,order=selection['order'],numeric=num,categorical=cat),model,compress=1)
                e.write_json(dest/'frozen_before_V.json',dict(sha256=e.sha256(model),plans=plans,records=records,weights=weights,time=time.time(),selection='none'))
                np.savez_compressed(dest/'calibration.npz',G_row_ids=raw.index[g],G_y=y[g],G_experts=pg,C_row_ids=raw.index[c],C_y=y[c],C_experts=pc)
                pv=cube(models,X.iloc[v])
            np.savez_compressed(dest/'evaluation.npz',row_ids=raw.index[v],y=y[v],p_experts=pv,weights=np.asarray(weights))
            for method,p in methods(pv,weights).items():rows.append(dict(stage=name,history=mode,method=method,**scored(y[v],p.argmax(1))))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);print('DONE',name,mode,flush=True)
    e.write_json(out/'catalog.json',[dict(path=p.relative_to(out).as_posix(),sha256=e.sha256(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
    e.write_json(out/'complete.json',dict(status='PASS_HISTORY_SMOKE' if smoke else 'PASS_HISTORY_FULL',new_expert_fits=10 if smoke else 20,catalog_sha256=e.sha256(out/'catalog.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.output.resolve(),a.smoke)
