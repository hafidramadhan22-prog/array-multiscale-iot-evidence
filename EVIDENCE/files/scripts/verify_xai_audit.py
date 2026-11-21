"""Reconstruct XAI scores from arrays and verify source provenance. Never loads a model."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from xai_audit import exact_shapley,attribution_distance,curve_metrics
from run_xai_audit import verify_catalog,metric_rows,model_output_indices


def verify(run):
    run=run.resolve();done=json.loads((run/'execution_complete.json').read_text())
    if done['status']!='XAI_AUDIT_EXECUTION_COMPLETE':raise ValueError('Full complete run required')
    cfg=json.loads((run/'configuration_freeze.json').read_text());inputs=json.loads((run/'inputs_manifest.json').read_text())
    assert done['new_classifier_fits']==0 and not cfg['historical_holdout_used']
    for row in json.loads((run/'source_manifest.json').read_text()):
        assert e.sha256(ROOT/row['path'])==row['sha256']==e.sha256(run/'source'/row['path'])
    old=verify_catalog(ROOT/cfg['source_run'])
    assert old==inputs['source_run']
    old_runs=[old]
    for name in ['FULL_20261013_01','LIFT_ID_E1_20261018_02','LIFT_ID_FULL_20261018_01','E4_FULL_20261019_01','MPED_FULL_20261019_04','SBFE_FULL_20261019_01']:
        old_runs.append(verify_catalog(ROOT/'runs'/name))
    with np.load(run/'indices.npz',allow_pickle=False) as z:
        ids=z['row_ids'];labels=z['y'];a=z['A'];v=z['V'];h=z['historical_holdout_ids'];ba=z['background_A'];bb=z['background_B']
    assert len(ids)==100 and len(set(ids))==100 and set(ids)<=set(v) and set(ba)<=set(a) and set(bb)<=set(a)
    assert not set(ids)&set(h) and not set(ba)&set(bb) and not set(a)&set(v) and not (set(ba)|set(bb))&set(h)
    assert dict(zip(*np.unique(labels,return_counts=True)))=={k:20 for k in range(5)}
    faithful=pd.read_csv(run/'tables/faithfulness.csv');reconstructed=[];group_checks=[];robust_rows=[];background_rows=[]
    for folder in sorted((run/'groups').iterdir()):
        games={}
        for variant in sorted(folder.iterdir()):
            with np.load(variant/'coalitions.npz',allow_pickle=False) as z: result={k:z[k] for k in z.files}
            meta=json.loads((variant/'metadata.json').read_text());groups=meta['groups'];assert result['phi'].shape==(100,len(groups),6,5)
            np.testing.assert_array_equal(result['row_ids'],ids);np.testing.assert_array_equal(result['y'],labels)
            for val,phi in zip(result['values'],result['phi']):
                assert np.isfinite(val).all() and (val>=0).all() and (val<=1+1e-10).all()
                np.testing.assert_allclose(val.sum(-1),1,atol=1e-6)
                np.testing.assert_allclose(exact_shapley(val),phi,atol=1e-11,rtol=1e-9)
            assert len(meta['official_checks'])==2 and max(r['max_difference'] for r in meta['official_checks'])<1e-10
            games[variant.name]=result
            group_checks.append(dict(bundle=folder.name,variant=variant.name,rows=100,groups=len(groups),all_coalitions=result['values'].shape[1]))
        original=games['donor_A'];targets=original['values'][:,-1].argmax(2)
        rows,_=metric_rows(original,ids,labels,targets);reconstructed.extend([dict(bundle=folder.name,**r) for r in rows])
        for variant,result in games.items():
            if variant=='donor_A':continue
            if variant.startswith('donor_'):
                donor=variant.removeprefix('donor_')
                np.testing.assert_allclose(result['values'][:,-1],original['values'][:,-1],atol=1e-12)
                rows,_=metric_rows(result,ids,labels,targets,donor=donor,ranking_donor=donor)
                cross,_=metric_rows(result,ids,labels,targets,ranking_result=original,donor=donor,ranking_donor='A')
                reconstructed.extend([dict(bundle=folder.name,**r) for r in rows+cross])
                for i,rid in enumerate(ids):
                    for method,j in zip(['weighted_voting','random_forest','logistic_regression'],model_output_indices()):
                        target=targets[i,j]
                        background_rows.append(dict(bundle=folder.name,donor=donor,row_id=int(rid),true_class=int(labels[i]),method=method,
                            **attribution_distance(original['phi'][i,:,j,target],result['phi'][i,:,j,target])))
            else:
                for i,rid in enumerate(ids):
                    for method,j in zip(['weighted_voting','random_forest','logistic_regression'],model_output_indices()):
                        target=targets[i,j];new=result['values'][i,-1,j];oldp=original['values'][i,-1,j]
                        robust_rows.append(dict(bundle=folder.name,variant=variant,row_id=int(rid),true_class=int(labels[i]),method=method,
                            original_prediction=int(target),perturbed_prediction=int(new.argmax()),prediction_unchanged=bool(new.argmax()==target),
                            probability_l1=float(np.abs(new-oldp).sum()),target_probability_change=float(new[target]-oldp[target]),
                            **attribution_distance(original['phi'][i,:,j,target],result['phi'][i,:,j,target])))
    def compare(actual,expected,keys,ignore=()):
        actual=actual.sort_values(keys).reset_index(drop=True);expected=pd.DataFrame(expected).sort_values(keys).reset_index(drop=True)
        assert len(actual)==len(expected)
        for col in expected.columns:
            if col in ignore:continue
            if expected[col].dtype.kind in 'biuf': np.testing.assert_allclose(actual[col].astype(float),expected[col].astype(float),atol=1e-11,rtol=1e-9)
            else:np.testing.assert_array_equal(actual[col],expected[col])
    compare(faithful,reconstructed,['bundle','row_id','method','donor','ranking_donor','ranking'])
    compare(pd.read_csv(run/'tables/robustness.csv'),robust_rows,['bundle','variant','row_id','method'])
    compare(pd.read_csv(run/'tables/background_sensitivity.csv'),background_rows,['bundle','donor','row_id','method'])
    atom_rows=[];noise_rows=[];atomic_checks=[]
    for folder in sorted((run/'atomic').iterdir()):
        with np.load(folder/'attributions.npz',allow_pickle=False) as z: vals={k:z[k] for k in z.files}
        assert len(vals['row_ids'])==25 and vals['phi'].shape[:2]==(25,2)
        curves=json.loads((folder/'curves.json').read_text())
        for i,entry in enumerate(curves):
            assert entry['row_id']==vals['row_ids'][i]
            for line in entry['curves']:
                atom_rows.append(dict(bundle=folder.name,row_id=entry['row_id'],method=line['method'],seed=line['seed'],
                    **curve_metrics(*[np.array(line[k]) for k in ['deletion','insertion','random_deletion','random_insertion']])))
            for method,j in zip(['weighted_voting','random_forest','logistic_regression'],model_output_indices()):
                # Target stored in the metric file; fixing it avoids any new class choice after a perturbation.
                actual=pd.read_csv(run/'tables/atomic_faithfulness.csv')
                target=int(actual[(actual.bundle==folder.name)&(actual.row_id==entry['row_id'])&(actual.method==method)].target.iloc[0])
                noise_rows.append(dict(bundle=folder.name,row_id=int(entry['row_id']),method=method,
                    **attribution_distance(vals['phi'][i,0,:,j,target],vals['phi'][i,1,:,j,target])))
        atomic_checks.append(dict(bundle=folder.name,rows=25,seeds=len(vals['seeds'])))
    compare(pd.read_csv(run/'tables/atomic_faithfulness.csv'),atom_rows,['bundle','row_id','method','seed'])
    compare(pd.read_csv(run/'tables/atomic_seed_sensitivity.csv'),noise_rows,['bundle','row_id','method'])
    result=dict(status='PASS_XAI_RECONSTRUCTION_AND_PROVENANCE',group_checks=group_checks,atomic_checks=atomic_checks,
        metric_rows_reconstructed=len(reconstructed),robustness_rows_reconstructed=len(robust_rows),atomic_metric_rows_reconstructed=len(atom_rows),
        old_runs=old_runs,old_artifacts_unchanged=sum(r['artifacts'] for r in old_runs),new_classifier_fits=0,historical_holdout_used=False)
    e.write_json(run/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();verify(a.run)
