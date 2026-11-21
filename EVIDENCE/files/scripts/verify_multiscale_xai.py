"""Independently reconstruct saved XAI metrics and groups without loading models."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from run_context_repair import read_npz,check_record
from run_wb_research import load_data
from multiscale_context import repair_frame
from xai_audit import exact_shapley,attribution_distance,METHODS
from run_xai_audit import metric_rows,model_output_indices
from run_xai_background_extension import robust_rows
from run_multiscale_xai import groups_for

def verify(run):
    cfg=json.loads((run/'configuration_freeze.json').read_text());owner=ROOT/cfg['quality_run'];check_record(owner,cfg['owned_model'])
    wc=json.loads((owner/'configuration_freeze.json').read_text());raw,y,holdout,data=load_data(wc)
    idx=read_npz(run/'indices.npz');pos=raw.index.get_indexer(idx['row_ids']);assert (pos>=0).all()
    split=read_npz(owner/'stages/temporal_seed42/split.npz')
    assert set(idx['row_ids']).issubset(split['V']) and not set(idx['row_ids'])&set(holdout)
    assert len(idx['row_ids'])==100 and np.array_equal(np.bincount(y[pos],minlength=5),np.full(5,20))
    assert set(np.r_[idx['background_A'],idx['background_B']]).issubset(split['A']) and not set(idx['background_A'])&set(idx['background_B'])
    selection=json.loads((owner/'stages/temporal_seed42/candidate_selection_before_V.json').read_text())
    assert cfg['view']==selection['selected']['balanced']['wb_only']['view']
    X,_,_=repair_frame(raw,data,cfg['view']);frame=X.iloc[pos];groups=groups_for(frame.columns)
    sel=json.loads((owner/'stages/temporal_seed42'/f"{cfg['view']}__balanced"/'selection_before_V.json').read_text());weights=np.array(sel['weights'])
    games={}
    for name in ['A32','B32','query_delay_5ms']:
        folder=run/'groups'/name;game=read_npz(folder/'coalitions.npz');meta=json.loads((folder/'metadata.json').read_text())
        assert meta['groups']==groups and all(z['max_difference']<=1e-10 for z in meta['official_checks'])
        np.testing.assert_array_equal(game['row_ids'],idx['row_ids']);np.testing.assert_array_equal(game['y'],y[pos])
        for i in range(100):np.testing.assert_allclose(exact_shapley(game['values'][i]),game['phi'][i],atol=1e-11,rtol=1e-9)
        np.testing.assert_allclose(game['phi'][:,:,0],np.einsum('m,njmc->njc',weights,game['phi'][:,:,1:]),rtol=1e-8,atol=1e-10)
        np.testing.assert_array_equal(game['background_ids'],idx['background_B' if name=='B32' else 'background_A'])
        games[name]=game
    targets=games['A32']['values'][:,-1].argmax(2);faith=[]
    for donor,rank in [('A32','A32'),('B32','B32'),('B32','A32')]:
        rows,_=metric_rows(games[donor],idx['row_ids'],y[pos],targets,ranking_result=games[rank],donor=donor,ranking_donor=rank);faith+=rows
    pd.testing.assert_frame_equal(pd.read_csv(run/'tables/faithfulness.csv'),pd.DataFrame(faith),check_dtype=False,atol=1e-12,rtol=0)
    jitter_run=ROOT/'runs/MULTISCALE_JITTER_DATA_20261020_01';jitter=read_npz(jitter_run/'jitter.npz')
    assert cfg['jitter_sha256']==e.sha256(jitter_run/'jitter.npz');np.testing.assert_array_equal(jitter['row_ids'],idx['row_ids'])
    modified_data={**data,'prior_multiscale':data['prior_multiscale'].copy(),'prior_directional':data['prior_directional'].copy()}
    modified_data['prior_multiscale'][pos]=jitter['prior_multiscale'];modified_data['prior_directional'][pos]=jitter['prior_directional']
    modified,_,_=repair_frame(raw,modified_data,cfg['view']);modified=modified.iloc[pos]
    robust=robust_rows(games['A32'],games['query_delay_5ms'],frame,modified,idx['row_ids'],y[pos],'query_delay_5ms')
    robust+=robust_rows(games['A32'],games['A32'],frame,frame,idx['row_ids'],y[pos],'port_bijection')
    pd.testing.assert_frame_equal(pd.read_csv(run/'tables/robustness.csv'),pd.DataFrame(robust),check_dtype=False,atol=1e-12,rtol=0)
    distance=[]
    for i,row_id in enumerate(idx['row_ids']):
        for method,j in zip(METHODS,model_output_indices()):
            target=int(targets[i,j]);distance.append(dict(row_id=int(row_id),method=method,
                **attribution_distance(games['A32']['phi'][i,:,j,target],games['B32']['phi'][i,:,j,target])))
    pd.testing.assert_frame_equal(pd.read_csv(run/'tables/background_sensitivity.csv'),pd.DataFrame(distance),check_dtype=False,atol=1e-12,rtol=0)
    result=dict(status='PASS_INDEPENDENT_XAI_RECONSTRUCTION',faithfulness_rows=len(faith),robustness_rows=len(robust),
        background_rows=len(distance),all_coalition_attributions_reconstructed=True,official_shap_checks=True,new_model_loads=0,new_classifier_fits=0)
    e.write_json(run/'verification_independent.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
