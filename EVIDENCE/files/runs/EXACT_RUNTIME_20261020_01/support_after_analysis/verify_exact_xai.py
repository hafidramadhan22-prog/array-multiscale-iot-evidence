"""Recompute existing donor coalition games through the compact runtime. No fit."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_wb_research import load_data
from run_time_repair import read_npz
from profile_exact_runtime import load_policy
from run_xai_audit import verify_catalog
from multiscale_context import repair_frame
from xai_audit import subset_masks,exact_shapley


def verify(run):
    freeze=json.loads((run/'inputs_manifest.json').read_text())
    assert 'baseline25__compact_shared' in freeze['policies']
    fast,_,_=load_policy(run,freeze,'baseline25__compact_shared')
    old=ROOT/'runs/XAI_MULTISCALE_FULL_20261020_01';verify_catalog(old)
    cfg=json.loads((ROOT/'runs/MULTISCALE_REPAIR_FULL_20261020_01/configuration_freeze.json').read_text())
    raw,y,holdout,data=load_data(cfg);X,_,_=repair_frame(raw,data,'legacy25_multiscale60')
    ids=read_npz(old/'indices.npz');pos=raw.index.get_indexer(ids['row_ids']);assert (pos>=0).all()
    assert not set(ids['row_ids'])&set(holdout)
    delayed_data={**data,'prior_multiscale':data['prior_multiscale'].copy()}
    jitter=read_npz(ROOT/'runs/MULTISCALE_JITTER_DATA_20261020_01/jitter.npz')
    np.testing.assert_array_equal(jitter['row_ids'],ids['row_ids'])
    delayed_data['prior_multiscale'][pos]=jitter['prior_multiscale']
    delayed,_,_=repair_frame(raw,delayed_data,'legacy25_multiscale60')
    rows=[]
    with threadpool_limits(limits=1):
        for name in ['A32','B32','query_delay_5ms']:
            saved=read_npz(old/'groups'/name/'coalitions.npz')
            meta=json.loads((old/'groups'/name/'metadata.json').read_text());groups=meta['groups'];masks=subset_masks(len(groups))
            background=X.loc[ids['background_B' if name=='B32' else 'background_A']]
            frame=(delayed if name=='query_delay_5ms' else X).iloc[pos]
            for i,(_,sample) in enumerate(frame.iterrows()):
                hybrids=[]
                for mask in masks:
                    hybrid=background.copy()
                    for active,columns in zip(mask,groups.values()):
                        if active:
                            for col in columns:hybrid[col]=sample[col]
                    hybrids.append(hybrid)
                batch=pd.concat(hybrids,ignore_index=True);ps=fast.expert_cube(batch)
                wb=sum(float(fast.weights[j])*ps[:,j] for j in range(5))
                values=np.concatenate([wb[:,None],ps],axis=1).reshape(len(masks),len(background),6,5).mean(1)
                phi=exact_shapley(values)
                np.testing.assert_allclose(values,saved['values'][i],atol=1e-12,rtol=0)
                np.testing.assert_allclose(phi,saved['phi'][i],atol=1e-12,rtol=0)
                rows.append(dict(game=name,row_id=int(ids['row_ids'][i]),coalitions=len(masks),donors=len(background),
                    max_probability_difference=float(np.max(np.abs(values-saved['values'][i]))),
                    max_shap_difference=float(np.max(np.abs(phi-saved['phi'][i])))))
            print('COMPACT XAI',name,len(frame),flush=True)
    pd.DataFrame(rows).to_csv(run/'xai_equivalence_raw.csv',index=False)
    e.write_json(run/'xai_equivalence.json',dict(status='PASS_COMPACT_XAI_GAMES',games=len(rows),
        coalitions=sum(r['coalitions'] for r in rows),donor_rows=sum(r['coalitions']*r['donors'] for r in rows),
        max_probability_difference=max(r['max_probability_difference'] for r in rows),
        max_shap_difference=max(r['max_shap_difference'] for r in rows),
        source_xai_catalog_sha256=e.sha256(old/'artifacts_sha256.json'),new_classifier_fits=0,
        limitation='Same classifier and donor intervention definition; this does not establish causal validity or new faithfulness scores.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run.resolve())
