"""Independent count-based saved evidence audit. No joblib loading or inference."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd

METHODS=['WV','RF','LR','Uniform','SVM_default','SVM_G_selected']
STAGES=['private_fold1','private_fold2','temporal_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']
NAMES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def seal_check(run):
    done=read(run/'complete.json');assert sha(run/'catalog.json')==done['catalog_sha256']
    records=read(run/'catalog.json')
    for r in records:
        f=run/r['path'];assert f.resolve().is_relative_to(run.resolve()) and f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],f
    return len(records)
def score(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int);assert y.shape==p.shape and np.isin(y,range(5)).all() and np.isin(p,range(5)).all()
    cm=np.bincount(5*y+p,minlength=25).reshape(5,5);support=cm.sum(1);tp=cm.diagonal();fp=cm.sum(0)-tp;fn=support-tp
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(5,dtype=float),where=(2*tp+fp+fn)>0)
    complete=bool((support>0).all());normal=int(support[0]);attacks=int(support[1:].sum())
    m=dict(n=len(y),macro_f1=float(f1.mean()) if complete else None,balanced_accuracy=float(np.mean(tp/support)) if complete else None,ddos_f2=float(5*tp[1]/(5*tp[1]+4*fn[1]+fp[1])) if support[1] else None,errors=int(fn.sum()),false_alarms=int(fn[0]),normal_support=normal,normal_fpr=float(fn[0]/normal) if normal else None,attack_support=attacks,attack_to_normal=int(cm[1:,0].sum()),binary_attack_recall=float(1-cm[1:,0].sum()/attacks) if attacks else None,label_agreement=float(tp.sum()/len(y)),all_five_true_classes_present=complete)
    for j,n in enumerate(NAMES):m.update({f'support_{n}':int(support[j]),f'recall_{n}':float(tp[j]/support[j]) if support[j] else None,f'errors_{n}':int(fn[j])})
    return m,cm
def compare(record,m):
    for key,value in m.items():
        if value is None:assert pd.isna(record[key]),key
        elif isinstance(value,bool):assert bool(record[key])==value,key
        else:assert np.isclose(float(record[key]),value,atol=1e-12,rtol=1e-12),(key,record[key],value)

def main(root,out):
    train=root/'runs/ARRAY_SVM_FULL_20261118_01';ev=root/'runs/ARRAY_SVM_EVALUATION_20261118_01';stress=root/'runs/ARRAY_SVM_STRESS_PORTABLE_20261118_01';serve=root/'runs/ARRAY_SVM_SERVING_20261118_01';report=root/'reports/ARRAY_SVM_20261118'
    sealed={r.name:seal_check(r) for r in [train,ev,stress,serve]}
    assert read(train/'complete.json')['new_SVC_fits']==63 and read(ev/'complete.json')['new_SVC_fits']==0
    assert read(stress/'portable_encoding_correction.json')['no_pickle_loading']
    assert read(ev/'configuration_freeze.json')['training_catalog_sha256']==sha(train/'catalog.json')
    parentdir=report/'parent_inputs';parentmanifest=read(parentdir/'manifest.json');catalogs={}
    expected_catalogs={'reference':'d89f9b971d2889e11a0e76345352b08abb93d31d4cced1536fa88ddf41112d2f','history_models':'58cb8e1a93eead3fa4dcf04c3366cc82f43b8b2021e233569a109f64dbd108dd','aligned':'a68acf541a06c44455a96f77c14f11d8f25e796783a12f30f37e5c92c1e5f309','guard':'f30f4247e55d77b42c0701eceec9017a3f38de605b732e0b8815512f723ee652','external':'9b25f63c75777d7c1aad4d04c75405832062d994c2410985e58fc53f6513d537'}
    for rec in parentmanifest['catalogs']:
        assert sha(parentdir/rec['path'])==rec['sha256']==expected_catalogs[rec['parent']]
        catalogs[rec['parent']]={r['path'].replace('\\','/'):r for r in read(parentdir/rec['path'])}
    parentfiles={}
    for rec in parentmanifest['files']:
        original=catalogs[rec['parent']][rec['original_rel']];p=parentdir/rec['path']
        assert sha(p)==original['sha256']==rec['sha256'] and p.stat().st_size==original['bytes']==rec['bytes']
        parentfiles[(rec['parent'],rec['original_rel'])]=p
    allstage=pd.read_csv(ev/'metrics.csv');summary=pd.read_csv(report/'protocol_summary.csv');stagevalues={};modelcount=0
    for name in STAGES:
        d=train/'stages'/name;z=load(ev/'stages'/name/'evaluation.npz');split=load(d/'split.npz');g=load(d/'G_predictions.npz');plan=read(d/'training_plan.json')['plan']
        ids=[split[k] for k in 'AGCV'];assert len(np.unique(np.concatenate(ids)))==sum(map(len,ids))
        assert not set(np.concatenate(ids))&set(split['historical_holdout_ids'])
        np.testing.assert_array_equal(split['V'],z['row_ids']);np.testing.assert_array_equal(split['G'],g['row_ids'])
        np.testing.assert_array_equal(plan['preprocessing_fit_row_ids'],split['A'])
        assert sum(plan['before'].values())==len(split['A']) and plan['denominator']=='original_per_class_before_RUS'
        for c,n in plan['before'].items():
            assert plan['synthetic'][c]==max(0,plan['after'][c]-n) and plan['synthetic'][c]<=np.floor(.5*n)
        candidates=pd.read_csv(d/'G_candidates.csv');assert len(candidates)==9 and (candidates.fit_status==0).all()
        manifest=read(d/'models_manifest.json');modelcount+=len(manifest)
        for rec in manifest:
            p=train/rec['path'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256']
        for row in candidates.to_dict('records'):
            m,_=score(g['y'],g[row['id']]);assert np.isclose(row['macro_f1'],m['macro_f1'],rtol=0,atol=1e-12)
            assert np.isclose(row['score'],.5*(m['macro_f1']+m['ddos_f2']),rtol=0,atol=1e-12)
        freeze=read(d/'selection_before_V.json');best=max(candidates.to_dict('records'),key=lambda r:(r['score'],r['macro_f1'],-abs(np.log10(r['C']))-abs(np.log10(r['gamma_multiplier'])),r['id']))
        assert freeze['selected']==best['id'] and freeze['default']=='C1_G1' and freeze['V_scored'] is False
        check=read(ev/'stages'/name/'checks.json');assert check['original_freeze_unix']==freeze['frozen_unix'] and freeze['frozen_unix']<check['new_timestamp']
        for method in METHODS:
            m,cm=score(z['y'],z[method]);np.testing.assert_array_equal(cm,z['CM_'+method]);row=allstage[(allstage.stage==name)&(allstage.method==method)].iloc[0];compare(row,m)
        if name.startswith('private'):
            parent='history_models';base='signature_fold'+name[-1]+'_exclusive'
            old=load(parentfiles[parent,base+'/evaluation.npz']);cal=load(parentfiles[parent,base+'/calibration.npz']);oldsplit=load(parentfiles[parent,base+'/split.npz']);weights=old['weights']
        else:
            parent='reference';base='stages/'+name+'/legacy25_multiscale60__balanced'
            old=load(parentfiles[parent,base+'/evaluation.npz']);cal=load(parentfiles[parent,base+'/calibration.npz']);oldsplit=load(parentfiles[parent,'stages/'+name+'/split.npz']);weights=read(parentfiles[parent,base+'/selection_before_V.json'])['weights']
        for key in 'AGCV':np.testing.assert_array_equal(oldsplit[key],split[key])
        np.testing.assert_array_equal(z['row_ids'],old['row_ids']);np.testing.assert_array_equal(z['y'],old['y'])
        np.testing.assert_array_equal(g['row_ids'],cal['G_row_ids']);np.testing.assert_array_equal(g['y'],cal['G_y'])
        p=old['p_experts'];expected=sum(float(weights[j])*p[:,j] for j in range(5))
        np.testing.assert_array_equal(z['WV'],expected.argmax(1));np.testing.assert_array_equal(z['RF'],p[:,1].argmax(1));np.testing.assert_array_equal(z['LR'],p[:,4].argmax(1))
        stagevalues[name]=z
    assert modelcount==63
    pooled={}
    for prefix,protocol in [('private','packet_private'),('signature','signature_shared')]:
        zs=[stagevalues[f'{prefix}_fold{k}'] for k in [1,2]];ids=np.concatenate([z['row_ids'] for z in zs]);assert len(np.unique(ids))==52584
        pooled[protocol]={}
        for method in METHODS:
            m,cm=score(np.concatenate([z['y'] for z in zs]),np.concatenate([z[method] for z in zs]));pooled[protocol][method]=m
            compare(summary[(summary.protocol==protocol)&(summary.method==method)].iloc[0],m)
            np.testing.assert_array_equal(cm,pd.read_csv(report/f'CM_{protocol}_{method}.csv',index_col=0).to_numpy())
    assert np.isclose(pooled['packet_private']['WV']['macro_f1'],.9376935651371078,rtol=0,atol=1e-12)
    assert np.isclose(pooled['packet_private']['RF']['macro_f1'],.9699661927981221,rtol=0,atol=1e-12)
    for method in METHODS:
        rows=allstage[(allstage.protocol=='temporal')&(allstage.method==method)];s=summary[(summary.protocol=='temporal')&(summary.method==method)].iloc[0]
        for key in ['macro_f1','normal_fpr','balanced_accuracy','errors']:assert np.isclose(rows[key].mean(),s[key],rtol=0,atol=1e-12)
        assert np.isclose(rows.macro_f1.std(ddof=1),s.macro_f1_sd,rtol=0,atol=1e-12)
    rawstress=pd.read_csv(stress/'metrics.csv');stresssummary=pd.read_csv(report/'stress_summary.csv');stressvectors=0
    for cohort in rawstress.cohort.unique():
        z=load(stress/'predictions'/(cohort+'.npz'));assert z['row_ids'].dtype.kind in 'US'
        X=pd.read_csv(stress/'inputs'/(cohort+'.csv'),usecols=['row_id'],dtype=str,keep_default_na=False)
        np.testing.assert_array_equal(z['row_ids'],X.row_id.to_numpy(dtype=str))
        for method in METHODS:
            m,cm=score(z['y'],z[method]);compare(rawstress[(rawstress.cohort==cohort)&(rawstress.method==method)].iloc[0],m);np.testing.assert_array_equal(cm,z['CM_'+method]);stressvectors+=1
        if cohort.startswith('aligned_'):
            source=cohort[len('aligned_'):];old=load(parentfiles['aligned',source+'_predictions.npz']);p=old['p_after'];ids=np.array([source+':'+str(i) for i in old['record_index']]);weights=old['weights'];wp=sum(float(weights[j])*p[:,j] for j in range(5));rf=p[:,1]
            np.testing.assert_array_equal(z['y'],np.zeros(len(ids),dtype=int))
        else:
            if cohort.startswith('block3_'):source=cohort[len('block3_'):];old=load(parentfiles['guard','posteriors/'+source+'_later.npz'])
            else:old=load(parentfiles['external',cohort+'/probabilities.npz'])
            ids=old['row_ids'];wp=old['weighted_reference'];rf=old['random_forest'];yy=old['y_scenario'] if 'y_scenario' in old else old['y'];np.testing.assert_array_equal(z['y'],yy)
        np.testing.assert_array_equal(z['row_ids'],ids);np.testing.assert_array_equal(z['WV'],wp.argmax(1));np.testing.assert_array_equal(z['RF'],rf.argmax(1))
    for family in rawstress.family.unique():
        zs=[load(stress/'predictions'/(c+'.npz')) for c in rawstress[rawstress.family==family].cohort.unique()]
        for method in METHODS:
            m,_=score(np.concatenate([z['y'] for z in zs]),np.concatenate([z[method] for z in zs]));compare(stresssummary[(stresssummary.family==family)&(stresssummary.method==method)].iloc[0],m)
    for row in read(stress/'old_model_alignment_checks.json'):assert row['max_WV_probability_difference']<=1e-12 and row['max_RF_probability_difference']<=1e-12
    t=pd.read_csv(serve/'latency_raw.csv');mem=pd.read_csv(serve/'memory_raw.csv');cost=pd.read_csv(report/'serving_summary.csv');assert len(t)==480 and len(mem)==40
    expected=load(serve/'expected.npz');v=stagevalues['temporal_seed42'];lookup={int(r):i for i,r in enumerate(v['row_ids'])}
    for policy in t.policy.unique():
        for batch in [1,1000]:
            d=t[(t.policy==policy)&(t.batch==batch)];assert len(d)==30 and set(d['repeat'])==set(range(30)) and (d.seconds>0).all()
            m=cost[(cost.policy==policy)&(cost.batch==batch)].iloc[0]
            assert np.isclose(m.median_ms,1000*d.seconds.median(),rtol=0,atol=1e-9)
            key='WV' if policy.startswith('WV') else 'RF' if policy.startswith('RF') else 'LR' if policy.startswith('LR') else policy
            ids=[lookup[int(r)] for r in expected['row_ids_'+str(batch)]];np.testing.assert_array_equal(expected[policy+'__'+str(batch)],v[key][ids])
        rss=mem[mem.policy==policy];assert len(rss)==5 and (rss.samples>=3).all()
        np.testing.assert_allclose(rss.peak_mib-rss.baseline_mib,rss.delta_mib,rtol=0,atol=1e-9)
    analysis=read(report/'analysis.json');assert analysis['RF_retained'] and not analysis['WV_FPR_changed'] and not analysis['requested_WV_higher_than_both_SVMs_in_every_protocol']
    assert pooled['packet_private']['SVM_default']['macro_f1']>pooled['packet_private']['WV']['macro_f1']
    result=dict(status='PASS_INDEPENDENT_SVM_SAVED_EVIDENCE_AUDIT',sealed_files=sealed,original_parent_artifacts_verified=len(parentfiles),converged_SVC_fits=63,stage_label_vectors=42,stress_label_vectors=stressvectors,latency_calls=480,cold_RSS_calls=40,all_split_holdout_preprocessing_cap_G_selection_and_CM_checks=True,RF_retained=True,requested_universal_WV_win=False,models_deserialized=False,new_fits=0,new_inference=0)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args();main(a.root.resolve(),(a.output or a.root/'reports/ARRAY_SVM_20261118/verification.json').resolve())
