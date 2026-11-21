"""Frozen G-only reweighting of cached experts; no model loading or refitting."""
from pathlib import Path
import sys,json,argparse,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import ta_hafid_engine as e
from run_array_svm_20261118 import PARENTS,gated,read,npz,STAGES
from array_svm_20261118 import label_metrics,CLASSES
from array_weight_retune_20261118 import simplex_grid,canonical_mixture,search_labels,counts_and_scores,feasible_G,select_G
from wb_research import learn_weights

SVM=ROOT/'runs/ARRAY_SVM_EVALUATION_20261118_01'
SVM_SHA='fb15166077926690c253a0c995f8a62d620df5824201e2ff5c7b9fd18182ddc1'
POLICIES=['WV_reference','WV_G_constrained','WV_G_unconstrained','WV_G_heuristic']
SOURCES=['scripts/run_array_weight_retune_20261118.py','src/array_weight_retune_20261118.py','src/array_svm_20261118.py','src/wb_research.py','src/e4.py','src/ta_hafid_engine.py','scripts/run_array_svm_20261118.py','docs/ARRAY_WEIGHT_RETUNE_PROTOCOL_20261118.md','configs/target.json','requirements.lock.txt']

def copy_input(run,parent,rel):
    p=gated(parent,rel);dest=run/'inputs'/parent/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(p,dest)
    assert e.sha256(dest)==e.sha256(p)
    return dest

def svm_input(run,name):
    assert e.sha256(SVM/'catalog.json')==SVM_SHA
    rel='stages/'+name+'/evaluation.npz';row=next(r for r in read(SVM/'catalog.json') if r['path']==rel)
    p=SVM/rel;assert p.stat().st_size==row['bytes'] and e.sha256(p)==row['sha256']
    q=run/'inputs/SVM'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);return q

def stage(run,name,cfg):
    dest=run/'stages'/name;dest.mkdir();private=name.startswith('private')
    original='signature_fold'+name[-1] if private else name
    parent='history_models' if private else 'reference'
    base=original+'_exclusive' if private else 'stages/'+name+'/legacy25_multiscale60__balanced'
    splitfile=copy_input(run,parent,base+'/split.npz' if private else 'stages/'+name+'/split.npz')
    calfile=copy_input(run,parent,base+'/calibration.npz')
    selectfile=copy_input(run,parent,base+'/frozen_before_V.json' if private else base+'/selection_before_V.json')
    meta=read(selectfile);reference=np.asarray(meta['weights']);costfile=copy_input(run,'reference','stages/'+original+'/legacy25_multiscale60__balanced/selection_before_V.json')
    costs=read(costfile)['costs'];split=npz(splitfile)
    # Read only G values before freezing the chosen policies; C is not selected on.
    with np.load(calfile,allow_pickle=False) as z:G={k:z[k] for k in ['G_row_ids','G_y','G_experts']}
    np.testing.assert_array_equal(G['G_row_ids'],split['G'])
    assert len(set(np.concatenate([split[k] for k in 'AGCV'])))==sum(len(split[k]) for k in 'AGCV')
    # The original SVM split lives in its fit run; its holdout IDs gate all roles.
    owned_split=ROOT/'runs/ARRAY_SVM_FULL_20261118_01/stages'/name/'split.npz'
    assert e.sha256(owned_split)==next(r['sha256'] for r in read(ROOT/'runs/ARRAY_SVM_FULL_20261118_01/catalog.json') if r['path']==owned_split.relative_to(ROOT/'runs/ARRAY_SVM_FULL_20261118_01').as_posix())
    ids=npz(owned_split)
    for k in 'AGCV':np.testing.assert_array_equal(ids[k],split[k])
    assert not set(np.concatenate([split[k] for k in 'AGCV']))&set(ids['historical_holdout_ids'])
    np.savez_compressed(dest/'split.npz',**{k:split[k] for k in 'AGCV'},historical_holdout_ids=ids['historical_holdout_ids'])
    best,heuristics=learn_weights(G['G_y'],G['G_experts'],costs['training_pipeline_latency_seconds'],costs['serialized_pipeline_bytes'],cfg)
    grid=simplex_grid();grid=grid[::250] if cfg['smoke'] else grid
    weights=np.vstack([reference,np.full(5,.2),*[x['weights'] for x in heuristics],grid])
    labels=search_labels(G['G_experts'],weights);cm,f1,f2,score=counts_and_scores(G['G_y'],labels);ok=feasible_G(cm)
    ref_cm,ref_f1,ref_f2,_=counts_and_scores(G['G_y'],search_labels(G['G_experts'],reference[None,:]))
    selected=dict(WV_reference=0,WV_G_constrained=select_G(score,weights,reference,ok),WV_G_unconstrained=select_G(score,weights,reference),WV_G_heuristic=2+int(np.argmin([abs(x['alpha']-best['alpha']) for x in heuristics])))
    tp=np.diagonal(cm,axis1=1,axis2=2);support=cm.sum(2);normal_fp=support[:,0]-tp[:,0]
    candidates=pd.DataFrame(dict(candidate=np.arange(len(weights)),kind=['reference','uniform','heuristic_a0.5','heuristic_a0.6','heuristic_a0.7']+['simplex_grid']*len(grid),G_macro_f1=f1,G_DDoS_F2=f2,G_score=score,G_normal_FP=normal_fp,G_feasible=ok,L1_to_reference=np.abs(weights-reference).sum(1),**{f'w_{n}':weights[:,j] for j,n in enumerate(['DT','RF','XGB','LGB','LR'])},**{f'G_correct_{n}':tp[:,j] for j,n in enumerate(CLASSES)}))
    candidates.to_csv(dest/'G_candidates.csv',index=False)
    np.savez_compressed(dest/'G_search.npz',weights=weights,labels=labels,CM=cm,row_ids=G['G_row_ids'],y=G['G_y'],G_feasible=ok)
    frozen=dict(stage=name,selected=selected,weights={k:weights[i] for k,i in selected.items()},G_scores={k:float(score[i]) for k,i in selected.items()},G_constraints_reference=ref_cm[0],primary='WV_G_constrained',costs_fixed_from=costfile.relative_to(run).as_posix(),G_rows=len(labels[0]),candidates=len(weights),V_scored=False,frozen_unix=time.time())
    e.write_json(dest/'selection_before_V.json',frozen)
    # All new policies are fixed before accessing the historical V cache.
    path=copy_input(run,parent,base+'/evaluation.npz');V=npz(path)
    np.testing.assert_array_equal(V['row_ids'],split['V'])
    svm=npz(svm_input(run,name));np.testing.assert_array_equal(V['row_ids'],svm['row_ids']);np.testing.assert_array_equal(V['y'],svm['y'])
    vp=V['p_experts'];result=[];output=dict(row_ids=V['row_ids'],y=V['y'])
    methods={k:canonical_mixture(vp,weights[i]) for k,i in selected.items()}
    methods.update(Uniform=vp.mean(1),**{n:vp[:,j] for j,n in enumerate(['DT','RF','XGB','LGB','LR'])})
    labels_V={k:p.argmax(1) for k,p in methods.items()}
    labels_V.update(SVM_default=svm['SVM_default'],SVM_G_selected=svm['SVM_G_selected'])
    np.testing.assert_array_equal(labels_V['WV_reference'],svm['WV']);np.testing.assert_array_equal(labels_V['RF'],svm['RF'])
    protocol='packet_private' if private else 'temporal' if name.startswith('temporal') else 'signature_shared'
    for method,pred in labels_V.items():
        m,matrix=label_metrics(V['y'],pred);result.append(dict(stage=name,protocol=protocol,method=method,**m));output[method]=pred;output['CM_'+method]=matrix
        if method in POLICIES:output['p_'+method]=methods[method]
    np.savez_compressed(dest/'evaluation.npz',**output);pd.DataFrame(result).to_csv(dest/'metrics.csv',index=False)
    e.write_json(dest/'complete.json',dict(status='PASS_WEIGHT_RETUNE_STAGE',G_only=True,new_classifier_fits=0,candidates=len(weights),V_scored_after_freeze=True,completed_unix=time.time()))
    print('WEIGHTS_FROZEN_AND_EVALUATED',name,selected,'G_feasible',int(ok.sum()),flush=True)
    return result

def stress(run):
    d=run/'stress';d.mkdir();frozen=read(run/'stages/temporal_seed42/selection_before_V.json');rows=[]
    def evaluate(name,family,z,y,ids,experts,reference):
        output=dict(row_ids=np.asarray(ids,dtype=str),y=np.asarray(y,dtype=int))
        mix={k:canonical_mixture(experts,w) for k,w in frozen['weights'].items()}
        np.testing.assert_allclose(mix['WV_reference'],reference,rtol=0,atol=1e-12)
        mix.update(Uniform=experts.mean(1),**{n:experts[:,j] for j,n in enumerate(['DT','RF','XGB','LGB','LR'])})
        for method,p in mix.items():
            pred=p.argmax(1);m,cm=label_metrics(y,pred);rows.append(dict(cohort=name,family=family,method=method,**m));output[method]=pred;output['CM_'+method]=cm
            if method in POLICIES:output['p_'+method]=p
        np.savez_compressed(d/(name+'.npz'),**output)
    for source in ['Distance','Flame_Sensor','Heart_Rate','IR_Receiver','phValue','Soil_Moisture','Sound_Sensor','Water_Level']:
        z=npz(copy_input(run,'aligned',source+'_predictions.npz'));p=z['p_after'];ids=[source+':'+str(i) for i in z['record_index']]
        evaluate('aligned_'+source,'additional_aligned',z,np.zeros(len(p),dtype=int),ids,p,canonical_mixture(p,z['weights']))
    guard=ROOT/'runs/FPR_SQL_GUARD_20261109_01/posteriors'
    for path in sorted(guard.glob('*_later.npz')):
        z=npz(copy_input(run,'guard','posteriors/'+path.name));p=np.stack([z[n] for n in e.NAMES],axis=1)
        evaluate('block3_'+path.stem[:-6],'later_block3',z,z['y'],z['row_ids'],p,z['weighted_reference'])
    for source in ['CIC_Normal0','CIC_Normal3','CIC_DDoS_SYN','CIC_PortScan','CIC_SQL','TON_Ransom1','TON_Ransom2']:
        z=npz(copy_input(run,'external',source+'/probabilities.npz'));p=np.stack([z[n] for n in e.NAMES],axis=1);y=z['y_scenario'] if 'y_scenario' in z else z['y']
        evaluate(source,'CIC' if source.startswith('CIC') else 'TON',z,y,z['row_ids'],p,z['weighted_reference'])
    pd.DataFrame(rows).to_csv(run/'stress_metrics.csv',index=False)

def main(run,smoke):
    assert not run.exists() and run.parent==(ROOT/'runs').resolve()
    assert e.sha256(ROOT/'ML-EdgeIIoT-dataset.csv')==e.DEFAULT_CONFIG['dataset_sha256']
    cfg={**e.DEFAULT_CONFIG,**read(ROOT/'configs/target.json'),**read(gated('reference','configuration_freeze.json'))}
    assert cfg['weight_shrinkage']==.5 and cfg['min_base_weight']==.05 and cfg['weight_alphas']==[.5,.6,.7]
    run.mkdir();(run/'stages').mkdir();(run/'source').mkdir();(run/'inputs').mkdir()
    stages=['private_fold1','temporal_seed42'] if smoke else STAGES
    cfg.update(date='2026-11-18',experiment='G-only history-matched weight retuning',smoke=smoke,stages=stages,grid_step=.05,grid_floor=.05,primary='WV_G_constrained',source_protocol_sha256=e.sha256(ROOT/'docs/ARRAY_WEIGHT_RETUNE_PROTOCOL_20261118.md'))
    e.write_json(run/'configuration_freeze.json',cfg);e.write_json(run/'environment.json',e.environment())
    source=[]
    for rel in SOURCES:
        p=ROOT/rel;shutil.copy2(p,run/'source'/p.name);source.append(dict(path=rel,sha256=e.sha256(p)))
    e.write_json(run/'source_manifest.json',source);e.write_json(run/'parent_catalogs.json',PARENTS)
    for key,(folder,name,digest) in PARENTS.items():
        p=ROOT/'runs'/folder/name;assert e.sha256(p)==digest;shutil.copy2(p,run/'inputs'/(key+'_'+name))
    allrows=[]
    with threadpool_limits(limits=1):
        for name in stages:allrows.extend(stage(run,name,cfg));pd.DataFrame(allrows).to_csv(run/'metrics.csv',index=False)
        if not smoke:stress(run)
    for r in source:assert e.sha256(ROOT/r['path'])==r['sha256']
    records=[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=e.sha256(p)) for p in sorted(run.rglob('*')) if p.is_file()]
    e.write_json(run/'catalog.json',records);e.write_json(run/'complete.json',dict(status='PASS_WEIGHT_RETUNE_SMOKE' if smoke else 'PASS_WEIGHT_RETUNE_EXPERIMENT',stages=len(stages),G_candidate_evaluations=sum(read(run/'stages'/s/'complete.json')['candidates'] for s in stages),new_classifier_fits=0,models_deserialized=False,new_captures=False,RF_retained=True,old_predictor_and_paper_unchanged=True,catalog_sha256=e.sha256(run/'catalog.json')))
    print('WEIGHT_RETUNING_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.output.resolve(),a.smoke)
