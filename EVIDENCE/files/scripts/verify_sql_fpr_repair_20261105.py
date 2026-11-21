"""Recompute sealed experiment evidence without fitting any classifier.

Only the project's catalog-verified joblib files are deserialized. Vocabulary
refits below are training-only audit checks, not new model experiments.
"""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits
from payload_tfidf_pilot import augment
from payload_target_repair import predict
from payload_pilot import BASE_NAMES
from sql_disagreement_guard import apply_guard,eligible_rows

P=ROOT/'runs/PAYLOAD_TARGET_REPAIR_20261105_01'
G=ROOT/'runs/SQL_DISAGREEMENT_REPAIR_20261105_01'
OUT=ROOT/'reports/SQL_FPR_REPAIR_20261105'
checks={}
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def owned(folder,r):
    p=folder/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    return p
def seal(folder):
    done=read(folder/'complete.json');assert sha(folder/'artifacts_sha256.json')==done['catalog_sha256']
    rows=read(folder/'artifacts_sha256.json')
    for r in rows:owned(folder,r)
    for r in read(folder/'source_manifest.json'):assert sha(ROOT/r['path'])==r['sha256'],r['path']
    checks[folder.name+'_sealed_files']=len(rows)
def metrics(y,p,nc):
    # Independent implementation; no sklearn metric helper from the runner.
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    cm=np.bincount(nc*y+p,minlength=nc*nc).reshape(nc,nc)
    n=cm.sum(1);predn=cm.sum(0);tp=cm.diagonal()
    f1=np.divide(2*tp,n+predn,out=np.zeros(nc,dtype=float),where=(n+predn)>0)
    r={'n':len(y),'normal_support':int(n[0]),'attack_support':int(n[1:].sum()),
       'false_alarms':int(cm[0,1:].sum()),'false_negatives':int(cm[1:,0].sum()),
       'errors':int((y!=p).sum()),'fpr_percent':100*cm[0,1:].sum()/n[0] if n[0] else np.nan,
       'attack_recall_percent':100*(n[1:].sum()-cm[1:,0].sum())/n[1:].sum() if n[1:].sum() else np.nan,
       'macro_f1_percent':100*f1.mean() if (n>0).all() else np.nan,
       'balanced_accuracy_percent':100*(tp/n).mean() if (n>0).all() else np.nan}
    names=['Normal','SQL_injection'] if nc==2 else ['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']
    for i,name in enumerate(names):
        r['support_'+name]=int(n[i]);r['recall_'+name]=100*tp[i]/n[i] if n[i] else np.nan
    return r,cm
def compare(row,m):
    for k,v in m.items():
        if np.isnan(v):assert pd.isna(row[k]),(k,row[k],v)
        else:np.testing.assert_allclose(row[k],v,rtol=0,atol=1e-9,err_msg=k)
def validp(p):
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(-1),1,rtol=0,atol=1e-6)
def hd(t):return hashlib.sha256(t.lower().encode('utf8')).hexdigest() if t else ''
def verify():
    OUT.mkdir(parents=True,exist_ok=True)
    seal(P);seal(G);print('SEALED_CATALOGS_OK',flush=True)
    cfg=read(P/'configuration_freeze.json');gc=read(G/'configuration_freeze.json')
    assert not cfg['smoke_test'] and cfg['reference_V_verified'] and cfg['reference_E_verified']
    assert not gc['threshold_fitted'] and gc['design_after_observation'] and gc['RF_confidence_threshold']==.5
    assert gc['new_classifier_fits']==0 and not gc['main_WV_replaced']
    manifest=read(P/'models_manifest.json')
    models={r['key']:joblib.load(owned(P,r)) for r in manifest}
    d=joblib.load(owned(P,read(P/'input_manifest.json')))
    X=augment(d['base'],d['raw']);EX=augment(d['additional_base'],d['additional_raw']);Y=d['y'];parts=d['parts']
    idx=npz(P/'indices.npz');h=set(idx['holdout_ids'])
    for k,ix in parts.items():
        np.testing.assert_array_equal(idx[k],X.index[ix]);assert not set(idx[k])&h
        for j in parts:
            if k!=j:assert not set(idx[k])&set(idx[j])
    assert [len(parts[k]) for k in 'AGCV']==[31676,5360,5190,10358]
    assert len(X)==52684 and len(EX)==760 and len(h)==13171
    b=npz(P/'binary_indices.npz');ids=b['training_ids'];pos=X.index.get_indexer(ids);BX=X.loc[ids]
    assert (pos>=0).all() and len(ids)==len(set(ids))==200 and set(ids)<=set(idx['A'])
    np.testing.assert_array_equal(np.where(Y[pos]==0,0,1),b['y']);assert np.bincount(b['y']).tolist()==[100,100]
    trainhash={hd(t) for t in BX.payload_text};assert '' not in trainhash and len(trainhash)==200
    for f in (1,2):
        tr=b[f'fold{f}_train'];va=b[f'fold{f}_val'];assert not set(tr)&set(va) and set(tr)|set(va)==set(ids)
        assert not {hd(t) for t in X.loc[tr].payload_text}&{hd(t) for t in X.loc[va].payload_text}
    assert set(b['fold1_val'])|set(b['fold2_val'])==set(ids) and not set(b['fold1_val'])&set(b['fold2_val'])
    coverage=read(P/'binary_coverage.json');assert coverage['training_kind_counts']=={'Normal':{'MQTT body':50,'Other TCP bytes':50},'SQL_injection':{'HTTP request values':100}}
    bv=parts['V'][np.isin(Y[parts['V']],[0,3])];testids=X.index[bv][[bool(t) and hd(t) not in trainhash for t in X.iloc[bv].payload_text]]
    assert len(testids)==1012 and np.bincount(np.where(Y[X.index.get_indexer(testids)]==0,0,1)).tolist()==[895,117]
    checks['split_and_binary_content_controls']=True
    # Every fit plan must be training-only, including vocabulary fitting and cap.
    plan_count=0
    for key,bundle in models.items():
        if key=='reference':continue
        fs=read(P/(key+'_fit.json'))
        for plan in fs['plans']:
            fit=set(plan['fit_row_ids']);val=set(plan['validation_row_ids']);assert not fit&val and not fit&h
            if key.startswith('binary_'):
                stage=plan['stage'];wanted=set(ids) if stage=='final' else set(b['fold'+stage[3:]+'_train'])
                assert fit==wanted and plan['nclasses']==2
                if stage!='final':assert val==set(b['fold'+stage[3:]+'_val'])
            else:assert fit==set(idx['A']) and val==set(idx['G'])|set(idx['C'])|set(idx['V']) and plan['nclasses']==5
            before={int(k):v for k,v in plan['before'].items()};after={int(k):v for k,v in plan['after'].items()};syn={int(k):v for k,v in plan['synthetic'].items()}
            yy=Y[X.index.get_indexer(list(fit))]
            if key.startswith('binary_'):yy=np.where(yy==0,0,1)
            assert before==dict(zip(*np.unique(yy,return_counts=True)))
            assert all(syn[c]==max(0,after[c]-before[c]) and syn[c]<=int(.5*before[c]) for c in before)
            assert sum(syn.values())/sum(after.values())<=.5
            assert set(plan['vocabulary_fit_row_ids'])==(fit if plan['vocabulary_size'] else set())
            plan_count+=1
        if bundle['nclasses']==2:
            assert bundle['models']['xgboost'].named_steps['estimator'].get_params()['objective']=='binary:logistic'
            assert bundle['models']['lightgbm'].named_steps['estimator'].get_params()['objective']=='binary'
            z=npz(P/'posteriors'/(key+'_OOF.npz'));assert z['probabilities'].shape==(200,5,2);validp(z['probabilities']);np.testing.assert_array_equal(z['row_ids'],ids)
        # Independently reconstruct final training-only vocabulary/IDF.
        if 'tfidf' in bundle['models']['decision_tree'].named_steps['preprocessor'].named_transformers_:
            train=BX if bundle['nclasses']==2 else X.iloc[parts['A']]
            v=TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=2048,min_df=1,lowercase=True,sublinear_tf=True,norm='l2',dtype=np.float64).fit(train.payload_text)
            for model in bundle['models'].values():
                actual=model.named_steps['preprocessor'].named_transformers_['tfidf'];assert v.vocabulary_==actual.vocabulary_
                np.testing.assert_allclose(v.idf_,actual.idf_,rtol=0,atol=1e-12)
    checks['verified_fit_plans']=plan_count;checks['training_only_vocabulary_reconstruction']=True
    print('SPLIT_FIT_VOCABULARY_OK',flush=True)
    table=pd.read_csv(P/'metrics.csv');nmetrics=0
    for file in (P/'posteriors').glob('*__*.npz'):
        key,cohort=file.stem.split('__',1);z=npz(file);nc=2 if key.startswith('binary_') else 5
        assert len(z['row_ids'])==len(z['y']) and len(set(z['row_ids']))==len(z['y'])
        if cohort=='binary_content_disjoint':np.testing.assert_array_equal(z['row_ids'],testids)
        for method in set(z)-{'row_ids','y'}:
            p=z[method];validp(p);r=table[(table.key==key)&(table.cohort==cohort)&(table.method==method)];assert len(r)==1
            m,_=metrics(z['y'],p.argmax(1),nc);compare(r.iloc[0],m);nmetrics+=1
    assert nmetrics==len(table);checks['parent_metric_rows_recomputed']=nmetrics
    # Verify the first-stage content-only policy against the saved G/C trial rule.
    for policy in read(P/'correction_policies_frozen.json'):
        bundle=models[policy['binary_key']];method=policy['method'];results={}
        for role in ('G','C'):
            xx=X.iloc[parts[role]];yy=Y[parts[role]];ref=npz(P/'posteriors'/('reference__'+role+'.npz'))['weighted_voting'].argmax(1)
            pp=predict(bundle,xx,method)[:,0];gate=(ref==3)&xx.payload_text.str.len().gt(0).to_numpy();results[role]=(pp,gate,yy)
        pg,gate,yy=results['G'];safe=[]
        for trial in policy['G_trials']:
            t=trial['threshold'];flip=gate&(pg>=t)&(pg>.5);harm=int((flip&(yy>0)).sum())
            assert harm==trial['new_missed_attacks_G'] and int((flip&(yy==0)).sum())==trial['normal_recovered_G']
            if harm==0:safe.append(t)
        assert policy['threshold']==(safe[0] if safe else None)
        if safe:
            pc,gate,yy=results['C'];harm=int((gate&(pc>=safe[0])&(pc>.5)&(yy>0)).sum())
            assert harm==policy['new_missed_attacks_C'] and policy['active']==(harm==0)
    correction=pd.read_csv(P/'correction_results.csv')
    for _,row in correction.iterrows():
        f=P/'posteriors'/f'correction_{row.binary_key}_{row.method}_{row.cohort}.npz';z=npz(f);m,_=metrics(z['y'],z['new_labels'],5);compare(row,m)
    checks['content_policy_G_C_selection_and_final_metrics']=True
    # Reconstruct each new guard from verified old probabilities + current content.
    gt=pd.read_csv(G/'metrics.csv');activation=pd.read_csv(G/'activation_coverage.csv');ng=0
    for cohort in ('G','C','V','additional'):
        xx=EX if cohort=='additional' else X.iloc[parts[cohort]];gold=npz(P/'posteriors'/('reference__'+cohort+'.npz'))
        w=gold['weighted_voting'];rf=gold['random_forest'];has=xx.payload_text.str.len().gt(0).to_numpy();elig=eligible_rows(w,rf,.5)
        ar=activation[activation.cohort==cohort].iloc[0]
        assert ar.eligible==int(elig.sum()) and ar.eligible_attack==int((elig&(gold['y']>0)).sum()) and ar.eligible_nonempty==int((elig&has).sum())
        for key in gt[gt.cohort==cohort].key:
            if key=='reference':p=w;changed=np.zeros(len(xx),dtype=bool)
            elif key=='RF_only':p,changed,_=apply_guard(w,rf,has,use_payload=False)
            else:
                if key in ('syntax_WV','tfidf_WV','hybrid_WV'):bk,method={'syntax_WV':('binary_syntax_only','weighted_voting'),'tfidf_WV':('binary_tfidf_only','weighted_voting'),'hybrid_WV':('binary_syntax_tfidf','weighted_voting')}[key]
                else:bk,method=key.rsplit('__',1)
                bp=np.full((len(xx),2),np.nan);ix=np.flatnonzero(elig&has)
                if len(ix):bp[ix]=predict(models[bk],xx.iloc[ix],method)
                p,changed,_=apply_guard(w,rf,has,bp,.5,True)
            z=npz(G/'posteriors'/f'{key}_{cohort}.npz');np.testing.assert_array_equal(z['row_ids'],xx.index);np.testing.assert_array_equal(z['y'],gold['y']);np.testing.assert_array_equal(z['new_labels'],p.argmax(1));np.testing.assert_array_equal(z['old_labels'],w.argmax(1));np.testing.assert_array_equal(z['changed'],changed)
            row=gt[(gt.key==key)&(gt.cohort==cohort)].iloc[0];m,_=metrics(z['y'],z['new_labels'],5);compare(row,m);assert row.changed==int(changed.sum());ng+=1
            if key=='hybrid_WV':
                if cohort=='additional':assert changed.sum()==321 and (z['new_labels']==0).all()
                else:assert not changed.any()
                if cohort=='V':assert m['false_negatives']==2 and m['false_alarms']==36
    assert ng==len(gt);checks['guard_metric_rows_recomputed']=ng
    assert activation.eligible_attack.sum()==0 and activation[activation.cohort!='additional'].eligible.sum()==0
    checks['attack_conflict_coverage']=0
    checks['attack_conflict_limitation_required']=True
    # Exported CSV CLI must give the same identities and normalized posteriors.
    cli=pd.read_csv(OUT/'example_predictions760.csv');z=npz(G/'posteriors'/'hybrid_WV_additional.npz')
    np.testing.assert_array_equal(cli.row_id,z['row_ids']);assert cli.changed.sum()==321 and cli.repaired_prediction.eq('Normal').all()
    names=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware'];cp=cli[['p_'+n for n in names]].to_numpy();validp(cp);np.testing.assert_array_equal(cp.argmax(1),z['new_labels'])
    gold=npz(P/'posteriors'/'reference__additional.npz');expected=gold['weighted_voting'].copy();expected[z['changed']]=gold['random_forest'][z['changed']]
    np.testing.assert_allclose(cp,expected,rtol=0,atol=1e-12);checks['runnable_CSV_prototype_verified']=True
    # Profiling uses the same batch/machine/pipeline for all candidates.
    times=pd.read_csv(G/'latency_raw.csv');mem=pd.read_csv(G/'memory_raw.csv');eff=pd.read_csv(G/'efficiency_summary.csv')
    assert len(times)==100 and len(mem)==12 and (times.pipeline_ms>0).all() and (mem.delta_peak_rss_mib>=0).all() and mem.predictions_match.all()
    for row in eff.itertuples():
        q=times[(times.cohort==row.cohort)&(times.key==row.key)];assert len(q)==10 and q.n.nunique()==1 and q.n.iloc[0]==row.n
        for name,actual in [('median_ms',q.pipeline_ms.median()),('mean_ms',q.pipeline_ms.mean()),('sd_ms',q.pipeline_ms.std())]:np.testing.assert_allclose(getattr(row,name),actual,rtol=0,atol=1e-9)
        mm=mem[(mem.cohort==row.cohort)&(mem.key==row.key)]
        if len(mm):assert len(mm)==2;np.testing.assert_allclose(row.delta_peak_rss_mean_mib,mm.delta_peak_rss_mib.mean(),rtol=0,atol=1e-9)
        else:assert pd.isna(row.delta_peak_rss_mean_mib)
    checks['latency_measurements']=100;checks['cold_memory_measurements']=12
    # Read-only checks of the unchanged manuscript when present in the workspace.
    oldpdf={'ARRAY_HAFID_REVISED_20261105.pdf':'46e5a5ff056f0a0a9d70044643937e41cf240b85d38ba85a2761873dcd0fbea4','ARRAY_HAFID_SUPPLEMENT_S1_S45_20261105.pdf':'305e5bbf61b2ad26ca2e5e4bf4d08e03d21c896489a529425af48fa7b64ddef2'}
    for name,wanted in oldpdf.items():
        f=ROOT/'output/pdf'/name
        if f.exists():assert sha(f)==wanted;checks[name+'_unchanged']=True
    # Compact tables for review, derived exclusively from sealed evidence.
    table[(table.cohort=='binary_content_disjoint')].to_csv(OUT/'binary_test_all_models.csv',index=False)
    table[(table.cohort.isin(['V','additional']))&(table.method=='weighted_voting')].to_csv(OUT/'matched_main_comparison.csv',index=False)
    gt[(gt.key.isin(['reference','RF_only','syntax_WV','tfidf_WV','hybrid_WV']))].to_csv(OUT/'guard_comparison.csv',index=False)
    z=npz(P/'posteriors'/'reference__additional.npz');bad=z['weighted_voting'].argmax(1)!=0
    has=EX.payload_text.str.len().gt(0).to_numpy();assert int((bad&has).sum())==64 and int((bad&~has).sum())==257
    residual=d['additional_base'].iloc[np.flatnonzero(bad&~has)]
    assert residual['tcp.len'].eq(0).all()
    flags=pd.to_numeric(residual['tcp.flags']).value_counts().sort_index()
    assert flags.to_dict()=={2:16,4:61,16:140,17:16,18:24}
    flags.rename_axis('tcp.flags').rename('packets_without_payload').to_csv(OUT/'residual_control_packets.csv')
    experts=[]
    for i,name in enumerate(BASE_NAMES):
        experts.append(dict(model=name,weight=cfg['weights'][i],additional_false_alarms=int((z[name].argmax(1)!=0).sum()),weighted_SQL_minus_Normal_on_321=float((cfg['weights'][i]*(z[name][bad,3]-z[name][bad,0])).mean())))
    pd.DataFrame(experts).to_csv(OUT/'expert_margin_diagnosis.csv',index=False)
    assert [r['additional_false_alarms'] for r in experts]==[611,0,2,509,420]
    _,cm=metrics(npz(P/'posteriors'/'reference__V.npz')['y'],npz(P/'posteriors'/'reference__V.npz')['weighted_voting'].argmax(1),5)
    pd.DataFrame(cm,index=names,columns=names).rename_axis('true_class').to_csv(OUT/'reference_and_guard_V_confusion.csv')
    checks['payload_residual_and_expert_margin_diagnosis']=True
    report={'status':'PASS','classifier_fits_during_verification':0,'checks':checks,'limitations':['Retrospective, within-trace/capture evaluation; observed data reused.','Normal and SQL content have protocol/source confounding.','No G/C/V attack triggered the RF/WV conflict guard. Zero observed extra-Normal alarms is not an independent sensitivity guarantee.','RF-only produces the same observed FPR reduction; unique TF-IDF benefit is unproven.']}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):verify()
