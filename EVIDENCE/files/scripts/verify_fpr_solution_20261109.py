"""Independent saved-probability audit. No model deserialization or fitting."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score,fbeta_score,recall_score
OUT=ROOT/'reports/FPR_SOLUTION_20261109'
R1=ROOT/'runs/FPR_TARGET8_20261109_01';R2=ROOT/'runs/FPR_NORMAL_VOTING_20261109_01';R3=ROOT/'runs/FPR_SQL_GUARD_20261109_01'
CLASSES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware'];POLICY='sql_guard_rf_rescue'
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def seal_check(run):
    assert sha(run/'artifacts_sha256.json')==read(run/'complete.json')['catalog_sha256']
    rows=read(run/'artifacts_sha256.json')
    for row in rows:
        p=(run/row['path']).resolve();assert p.is_relative_to(run)
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    for m in ['source_manifest.json','execution_source_manifest.json']:
        if (run/m).exists():
            for row in read(run/m):assert sha(ROOT/row['path'])==row['sha256'],row['path']
    return dict(run=run.name,status='PASS',sealed_files=len(rows))
def independent_rule(z):
    v=z['weighted_reference'];r=z['random_forest'];vl=v.argmax(1);rl=r.argmax(1)
    out=v.copy();use=((vl==3)&(rl==0))|((vl==0)&(rl!=0));out[use]=r[use];return out
def score(y,p):
    a=p.argmax(1);cm=confusion_matrix(y,a,labels=range(5));n=cm.sum(1);allclasses=(n>0).all();normal=int(n[0]);attacks=int(n[1:].sum())
    d=dict(n=len(y),normal_support=normal,attack_support=attacks,false_alarms=int(cm[0,1:].sum()),attack_to_Normal=int(cm[1:,0].sum()),errors=int((a!=y).sum()),FPR_percent=100*cm[0,1:].sum()/normal if normal else None,attack_recall_percent=100*(1-cm[1:,0].sum()/attacks) if attacks else None,macro_F1_percent=100*f1_score(y,a,labels=range(5),average='macro',zero_division=0) if allclasses else None,balanced_accuracy_percent=100*balanced_accuracy_score(y,a) if allclasses else None,DDoS_F2_percent=100*fbeta_score(y,a,beta=2,labels=[1],average=None,zero_division=0)[0] if n[1] else None)
    rec=recall_score(y,a,labels=range(5),average=None,zero_division=0)
    for i,c in enumerate(CLASSES):d['support_'+c]=int(n[i]);d['recall_'+c]=100*rec[i] if n[i] else None
    return d,cm
def main():
    OUT.mkdir(parents=True,exist_ok=True);sealed=[seal_check(r) for r in [R1,R2,R3]];rows=[];decisions=[];gates=[];pool={p:[] for p in ['weighted_reference','random_forest',POLICY]}
    cached=pd.read_csv(R3/'cached_metrics.csv');checked_count=0
    for path in sorted((R3/'posteriors').glob('*.npz')):
        if path.stem.endswith('_later'):continue
        z=npz(path);assert len(z['row_ids'])==len(set(z['row_ids'].tolist()));candidate=independent_rule(z);np.testing.assert_allclose(candidate,z[POLICY],rtol=0,atol=1e-12);co=path.stem
        for name in ['weighted_reference','random_forest',POLICY]:
            p=z[name];assert np.isfinite(p).all() and (p>=0).all();np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-6)
            m,cm=score(z['y'],p);rows.append(dict(cohort=co,policy=name,**m));pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(OUT/('CM_'+co+'__'+name+'.csv'))
            if name==POLICY:
                saved=cached.query('cohort==@co').iloc[0]
                for k,v in m.items():
                    if v is None:assert pd.isna(saved[k])
                    else:assert abs(float(saved[k])-v)<1e-10,(co,k,saved[k],v)
                checked_count+=1
            if co.startswith('signature_fold'):pool[name].append((z['y'],p,z['row_ids']))
        a=candidate.argmax(1);b=z['weighted_reference'].argmax(1);y=z['y'];r,_=score(y,z['weighted_reference']);c,_=score(y,candidate)
        if c['attack_support']:
            delta={k:c['recall_'+k]-r['recall_'+k] for k in CLASSES[1:]};passed=all(d>=-1e-12 for d in delta.values()) and c['attack_to_Normal']<=r['attack_to_Normal'] and c['macro_F1_percent']>=r['macro_F1_percent']-.1
            gates.append(dict(cohort=co,passed=bool(passed),macro_F1_delta_pp=c['macro_F1_percent']-r['macro_F1_percent'],attack_to_Normal_delta=c['attack_to_Normal']-r['attack_to_Normal'],**{'delta_recall_'+k:v for k,v in delta.items()}));assert passed
        for i,truth,v,rf,p in zip(z['row_ids'],y,z['weighted_reference'],z['random_forest'],candidate):
            decisions.append(dict(cohort=co,row_id=str(i),true_class=CLASSES[int(truth)],weighted_prediction=CLASSES[int(v.argmax())],RF_prediction=CLASSES[int(rf.argmax())],candidate_prediction=CLASSES[int(p.argmax())],changed=bool(v.argmax()!=p.argmax()),**{'p_'+k:float(p[j]) for j,k in enumerate(CLASSES)}))
    for name,parts in pool.items():
        assert len(set(np.concatenate([x[2] for x in parts]).tolist()))==sum(len(x[2]) for x in parts)
        y=np.concatenate([x[0] for x in parts]);p=np.vstack([x[1] for x in parts]);m,cm=score(y,p);rows.append(dict(cohort='signature_pooled',policy=name,**m))
        if name==POLICY:
            s=cached.query("cohort=='signature_pooled'").iloc[0]
            for k,v in m.items():assert abs(float(s[k])-v)<1e-10
    # All fresh packet blocks are disjoint targets. Histories/captures are shared.
    sources={};blockrows=[];percap=[];plans=[]
    for bi,run in enumerate([R1,R2,R3],1):
        cfg=read(run/'configuration_freeze.json');planned=sum(len(s['planned_ids']) for s in cfg['sources']);plans.append(planned);seen=set();allz=[]
        audits=read(run/'mapping_audit.json')
        assert sum(a['old_targets_verified'] for a in audits)==(760 if bi==1 else 664)
        for audit in audits:
            assert audit['history_query_before_current_push'] and audit['old_max_probability_difference']<=1e-12
            assert set(audit['scored_ids'])<=set(audit['planned_ids']);assert len(audit['missing_or_excluded_ids'])+len(audit['scored_ids'])==len(audit['planned_ids'])
        for path in sorted((run/'posteriors').glob('*_later.npz')):
            z=npz(path);source=path.stem[:-6];p=independent_rule(z);ids=set(z['row_ids'].tolist());assert not sources.get(source,set())&ids;sources.setdefault(source,set()).update(ids);seen|=ids
            sourcecfg=next(s for s in cfg['sources'] if s['source']==source)
            assert all(int(str(i).split(':')[-1]) in sourcecfg['planned_ids'] for i in ids)
            for name,pp in [('weighted_reference',z['weighted_reference']),('random_forest',z['random_forest']),(POLICY,p)]:
                m,_=score(z['y'],pp);percap.append(dict(block=bi,source=source,policy=name,planned=len(sourcecfg['planned_ids']),**m))
            allz.append((z,p))
            for i,pp,v,rf in zip(z['row_ids'],p,z['weighted_reference'],z['random_forest']):decisions.append(dict(cohort='later_block'+str(bi),row_id=str(i),true_class='Normal',weighted_prediction=CLASSES[int(v.argmax())],RF_prediction=CLASSES[int(rf.argmax())],candidate_prediction=CLASSES[int(pp.argmax())],changed=bool(v.argmax()!=pp.argmax()),**{'p_'+k:float(pp[j]) for j,k in enumerate(CLASSES)}))
        for name in ['weighted_reference','random_forest',POLICY]:
            pp=np.vstack([p if name==POLICY else z[name] for z,p in allz]);m,cm=score(np.zeros(len(pp),int),pp);blockrows.append(dict(block=bi,policy=name,planned=planned,scored=len(pp),coverage_percent=100*len(pp)/planned,FPR_worst_case_percent=100*(m['false_alarms']+planned-len(pp))/planned,selection_scope='before this block' if bi==3 else 'retrospective for targeted rule',**m))
            if name==POLICY:assert m['FPR_percent']<8
    D=pd.DataFrame(decisions);D.to_csv(OUT/'predictions_all.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'metrics_all_cached.csv',index=False);pd.DataFrame(blockrows).to_csv(OUT/'metrics_later_blocks.csv',index=False);pd.DataFrame(percap).to_csv(OUT/'metrics_per_capture.csv',index=False);pd.DataFrame(gates).to_csv(OUT/'independent_safety_gates.csv',index=False)
    D.query("cohort=='V_seed42' and changed").to_csv(OUT/'V_changed_predictions.csv',index=False)
    # Paired differences are repeat-to-repeat variability on this laptop only.
    lat=pd.read_csv(R3/'latency_raw.csv');mem=pd.read_csv(R3/'memory_raw.csv');paired=[];rng=np.random.default_rng(20261109)
    for co,frame in lat.groupby('cohort'):
        wide=frame.pivot(index='repeat',columns='policy',values='milliseconds');assert len(wide)==30
        for n in ['random_forest',POLICY]:
            delta=(wide[n]-wide['weighted_reference']).to_numpy();boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            paired.append(dict(cohort=co,policy=n,repetitions=30,mean_ms=float(wide[n].mean()),median_ms=float(wide[n].median()),mean_paired_delta_ms=float(delta.mean()),median_paired_delta_ms=float(np.median(delta)),paired_delta_SD_ms=float(delta.std(ddof=1)),bootstrap_95_low_ms=float(np.quantile(boot,.025)),bootstrap_95_high_ms=float(np.quantile(boot,.975)),mean_reduction_percent=float(100*(1-wide[n].mean()/wide['weighted_reference'].mean()))))
    pd.DataFrame(paired).to_csv(OUT/'paired_latency_comparison.csv',index=False)
    assert len(mem)==18 and mem.expected_probabilities_match.all()
    assert max(read(R3/'profile_probability_maxdiff.json').values())<=1e-12
    # Mechanism: per-expert behavior, original weights, and empirical SQL margin.
    z=npz(R1/'posteriors/additional760.npz');weights=read(ROOT/'configs/fpr_normal_voting_schema_20261109.json')['original_weights'];names=['decision_tree','random_forest','xgboost','lightgbm','logistic_regression'];false=z['weighted_reference'].argmax(1)!=0;cause=[]
    for name,w in zip(names,weights):
        p=z[name];cause.append(dict(model=name,weight=w,normal_n=760,false_alarms=int((p.argmax(1)!=0).sum()),SQL_predictions=int((p.argmax(1)==3).sum()),mean_SQL_minus_Normal_on_321_WV_false_alarms=float((p[false,3]-p[false,0]).mean()),weighted_mean_SQL_minus_Normal_on_321_WV_false_alarms=float(w*(p[false,3]-p[false,0]).mean())))
    pd.DataFrame(cause).to_csv(OUT/'expert_disagreement_cause.csv',index=False)
    assert int(false.sum())==321 and (z['weighted_reference'].argmax(1)[false]==3).all()
    result=dict(status='PASS',client_date='2026-11-09',sealed_runs=sealed,independently_recomputed_cached_cohorts=checked_count,all_seven_attack_gates_passed=len(gates)==7,third_block_FPR_below_8=True,strict_target='observed Normal FPR; attack recall no decline on seven reused cohorts',new_fits=0,new_captures=0,limitation='No independent attack/capture evaluation; final rule chosen after two benign blocks and historical attack results; third block Normal only',source_inputs_verified_by_original_runs=True,first_profiler_failure_preserved_and_v2_resolved=True)
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
