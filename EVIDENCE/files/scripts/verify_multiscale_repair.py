"""Recompute quality, G selection, role/cap guards and checksums without joblib."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import ta_hafid_engine as e
from e4 import metrics
from sbfe import guard_roles
from wb_research import learn_weights,mixture
from wb_scheduler import replay_scheduler,best_order
from run_wb_research import load_data,parts_for
from run_context_repair import check_record,read_npz,baseline,metric_rows
from context_repair import select_on_g

def verify(run):
    cfg=json.loads((run/'configuration_freeze.json').read_text())
    done=json.loads((run/'quality_complete.json').read_text());smoke=cfg['smoke_test']
    assert done['status']==('SMOKE_TEST_ONLY' if smoke else 'CONTEXT_REPAIR_QUALITY_COMPLETE')
    assert done['expert_fits']==(30 if smoke else 150)
    assert not cfg['historical_holdout_used'] and not cfg['new_data_or_captures']
    assert e.sha256(run/'tables/metrics.csv')==done['metrics_sha256']
    assert e.sha256(run/'models_manifest.json')==done['models_manifest_sha256']
    for rec in json.loads((run/'models_manifest.json').read_text()):check_record(run,rec)
    for rec in json.loads((run/'source_manifest.json').read_text()):
        assert e.sha256(run/'source'/Path(rec['path']).name)==rec['sha256']
    raw,y,holdout,data=load_data(cfg)
    assert e.sha256(ROOT/cfg['data_folder']/'catalog.json')==cfg['data_catalog_sha256']
    assert e.sha256(ROOT/cfg['source_run']/'artifacts_sha256.json')==cfg['source_catalog_sha256']
    actual_table=pd.read_csv(run/'tables/metrics.csv');expected=[];choices=[]
    for stage in json.loads((run/'planned_stages.json').read_text()):
        parts,groups=parts_for(stage,raw,y,holdout,data,smoke);a,g,c,v=parts
        guard_roles(raw.index.to_numpy(),parts,holdout)
        folder=run/'stages'/stage;sp=read_npz(folder/'split.npz')
        for key,pos in zip('AGCV',parts):np.testing.assert_array_equal(sp[key],raw.index[pos])
        freeze=json.loads((folder/'stage_frozen_before_V.json').read_text())
        assert freeze['models_manifest_sha256']==e.sha256(folder/'models_manifest.json')
        assert freeze['selection_sha256']==e.sha256(folder/'candidate_selection_before_V.json')
        selected=json.loads((folder/'candidate_selection_before_V.json').read_text());g_records=[]
        seed=int(stage.split('seed')[1]) if stage.startswith('temporal') else cfg['signature_seed']
        for case in cfg['cases']:
            base,base_sel=baseline(cfg,stage,case,'legacy25_symmetric',[raw.index[g],raw.index[c],raw.index[v]])
            reference=base_sel['weights']
            for view in ['baseline25','baseline29']+cfg['views']:
                if view.startswith('baseline'):
                    arrays,sel=baseline(cfg,stage,case,'legacy25_symmetric' if view=='baseline25' else 'legacy29_symmetric',
                                       [raw.index[g],raw.index[c],raw.index[v]])
                    pg,pc,pv=arrays
                else:
                    dest=folder/f'{view}__{case}';sel=json.loads((dest/'selection_before_V.json').read_text())
                    cal=read_npz(dest/'calibration.npz');ev=read_npz(dest/'evaluation.npz')
                    for key,pos in [('G',g),('C',c)]:
                        np.testing.assert_array_equal(cal[key+'_row_ids'],raw.index[pos])
                        np.testing.assert_array_equal(cal[key+'_y'],y[pos])
                    np.testing.assert_array_equal(ev['row_ids'],raw.index[v]);np.testing.assert_array_equal(ev['y'],y[v])
                    pg,pc,pv=cal['G_experts'],cal['C_experts'],ev['p_experts']
                    for plan in sel['plans']:
                        np.testing.assert_array_equal(plan['preprocessing_fit_row_ids'],raw.index[a])
                        assert set(plan['validation_ids'])==set(raw.index[np.r_[g,c,v]])
                        assert plan['before']=={str(j):int((y[a]==j).sum()) for j in range(5)}
                        assert all(plan['synthetic'][k]<=np.floor(.5*n) for k,n in plan['before'].items())
                        assert plan['synthetic_fraction_of_final_training']<=.5
                        if case=='original':assert plan['before']==plan['after']
                    cost=sel['costs']
                    picked,_=learn_weights(y[g],pg,cost['training_pipeline_latency_seconds'],cost['serialized_pipeline_bytes'],
                                           {**e.DEFAULT_CONFIG,**cfg})
                    np.testing.assert_allclose(picked['weights'],sel['weights'],atol=1e-14,rtol=0)
                    order=best_order(pc,sel['weights'],cost['expert_costs'],preprocessing_groups=cost['preprocessing_groups'],
                                     preprocessing_costs=cost['preprocessing_costs'])
                    assert order['order']==sel['order']
                    sch=replay_scheduler(pv,sel['weights'],sel['order'])
                    np.testing.assert_array_equal(sch.labels,ev['certified_labels'])
                    np.testing.assert_array_equal(sch.counts,ev['certified_counts'])
                    np.testing.assert_array_equal(mixture(pv,sel['weights']),ev['p_weighted'])
                    np.testing.assert_array_equal(sch.labels,ev['p_weighted'].argmax(1))
                    np.testing.assert_array_equal(mixture(pv,reference),ev['p_reference'])
                for pp in [pg,pc,pv]:
                    assert pp.shape[1:]==(5,5) and np.isfinite(pp).all() and (pp>=0).all()
                    np.testing.assert_allclose(pp.sum(2),1,atol=1e-6,rtol=0)
                rows=metric_rows(stage,seed,view,case,y[v],pv,sel['weights'],reference)
                if view=='baseline29':
                    from run_context_repair import source_file
                    cache=read_npz(source_file(cfg,f'stages/{stage}/legacy29_symmetric__{case}/evaluation.npz'))
                    lookup={int(row):i for i,row in enumerate(cache['row_ids'])};ix=np.array([lookup[int(row)] for row in raw.index[v]])
                    stress=metric_rows(stage,seed,view,case,y[v],cache['p_experts_port_shift'][ix],sel['weights'],reference)
                    rows=[z for z in rows if z['condition']=='original']+[z for z in stress if z['condition']=='port_bijection']
                expected+=rows
                if view!='baseline29':
                    methods={n:pg[:,j] for j,n in enumerate(e.NAMES)}
                    methods.update(weighted_voting=mixture(pg,sel['weights']),soft_voting=pg.mean(1))
                    sizes=sel['costs']['serialized_pipeline_bytes']
                    for method,post in methods.items():
                        m=metrics(y[g],post.argmax(1));j=e.NAMES.index(method) if method in e.NAMES else None
                        g_records.append(dict(role='G',view=view,case=case,method=method,score=.5*(m['macro_f1']+m['ddos_f2']),
                                              bytes=sizes[j] if j is not None else sum(sizes),metrics=m))
            for policy in ['wb_only','all_models']:
                records=[z for z in g_records if z['case']==case and (policy=='all_models' or z['method']=='weighted_voting')]
                picked=select_on_g(records)
                assert picked==selected['selected'][case][policy]
                choices.append(dict(stage=stage,case=case,policy=policy,view=picked['view'],method=picked['method']))
    sort=['stage','case','view','condition','method']
    wanted=pd.DataFrame(expected).sort_values(sort).reset_index(drop=True)
    got=actual_table.sort_values(sort).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[wanted.columns],wanted,check_dtype=False,atol=1e-12,rtol=0)
    old_count=0
    for old in [ROOT/cfg['source_run'],ROOT/'runs/WB_CONTEXT_FULL_20261020_01',ROOT/'runs/XAI_BACKGROUND_FULL_20261020_01']:
        for rec in json.loads((old/'artifacts_sha256.json').read_text()):check_record(old,rec);old_count+=1
    result=dict(status='PASS_REPAIR_SMOKE_ONLY' if smoke else 'PASS_CONTEXT_REPAIR_QUALITY',
        expert_fits=done['expert_fits'],metric_rows=len(got),g_selection_reconstructed=True,
        roles_and_training_only_caps_pass=True,scheduler_label_identity=True,old_artifacts_unchanged=old_count,
        historical_holdout_used=False,independent_capture_evaluation=False,choices=choices)
    e.write_json(run/'verification_quality.json',result);print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();verify(args.run)
