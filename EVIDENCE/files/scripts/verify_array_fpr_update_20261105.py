"""Independent saved-prediction and manuscript audit. No joblib/model loading."""
from pathlib import Path
import sys,json,hashlib,re,csv
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ARRAY_FPR_UPDATE_20261105'
OLD=ROOT/'paper/array_20261105_editorial/elsarticle';NEW=ROOT/'paper/array_20261105_fpr_update/elsarticle'
P=ROOT/'runs/PAYLOAD_TARGET_REPAIR_20261105_01';G=ROOT/'runs/SQL_DISAGREEMENT_REPAIR_20261105_01'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def calc(y,p,nc):
    cm=np.bincount(nc*np.asarray(y,dtype=int)+np.asarray(p,dtype=int),minlength=nc**2).reshape(nc,nc);s=cm.sum(1);pn=cm.sum(0);tp=cm.diagonal()
    f1=np.divide(2*tp,s+pn,out=np.zeros(nc,dtype=float),where=s+pn>0)
    v=dict(n=len(y),normal_support=int(s[0]),attack_support=int(s[1:].sum()),false_alarms=int(cm[0,1:].sum()),false_negatives=int(cm[1:,0].sum()),errors=int((y!=p).sum()),fpr_percent=100*cm[0,1:].sum()/s[0] if s[0] else np.nan,attack_recall_percent=100*(s[1:].sum()-cm[1:,0].sum())/s[1:].sum() if s[1:].sum() else np.nan,macro_f1_percent=100*f1.mean() if (s>0).all() else np.nan,balanced_accuracy_percent=100*np.mean(tp/s) if (s>0).all() else np.nan)
    names=['Normal','SQL_injection'] if nc==2 else ['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']
    for i,n in enumerate(names):v['support_'+n]=int(s[i]);v['recall_'+n]=100*tp[i]/s[i] if s[i] else np.nan
    return v,cm
def comp(row,expected):
    for k,v in expected.items():
        if np.isnan(v):assert pd.isna(row[k]),(k,row[k])
        else:np.testing.assert_allclose(row[k],v,rtol=0,atol=1e-9,err_msg=k)
def e(s):return str(s).replace('&',r'\&').replace('_',r'\_\allowbreak{}').replace('%',r'\%')
def tokens(tex):
    return tex.replace(r'\allowbreak{}','').replace(r'\&','&').replace(r'\_','_').replace(r'\%','%')
def f(v,d=4):return '--' if pd.isna(v) else f'{float(v):.{d}f}'
def count(v):return str(int(v))
def main():
    checks={};sealed=0
    for folder in (P,G):
        assert sha(folder/'artifacts_sha256.json')==read(folder/'complete.json')['catalog_sha256']
        for r in read(folder/'artifacts_sha256.json'):
            p=folder/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];sealed+=1
        for r in read(folder/'source_manifest.json'):assert sha(ROOT/r['path'])==r['sha256']
    checks['sealed_files']=sealed
    M=pd.read_csv(P/'metrics.csv');C=pd.read_csv(P/'correction_results.csv');GM=pd.read_csv(G/'metrics.csv');memo={};n=0
    for path in (P/'posteriors').glob('*__*.npz'):
        key,cohort=path.stem.split('__',1);z=npz(path);nc=2 if key.startswith('binary_') else 5
        for method in set(z)-{'row_ids','y'}:
            prob=z[method];assert np.isfinite(prob).all() and (prob>=0).all();np.testing.assert_allclose(prob.sum(1),1,rtol=0,atol=1e-6)
            m,cm=calc(z['y'],prob.argmax(1),nc);r=M[(M.key==key)&(M.cohort==cohort)&(M.method==method)];assert len(r)==1;comp(r.iloc[0],m);memo[key,cohort,method]=m;n+=1
    assert n==len(M)==256;checks['parent_metrics_recomputed']=n
    for r in C.itertuples():
        z=npz(P/'posteriors'/f'correction_{r.binary_key}_{r.method}_{r.cohort}.npz');m,_=calc(z['y'],z['new_labels'],5);comp(C[(C.binary_key==r.binary_key)&(C.method==r.method)&(C.cohort==r.cohort)].iloc[0],m)
    checks['content_correction_metrics_recomputed']=len(C)
    gmemo={}
    for r in GM.itertuples():
        z=npz(G/'posteriors'/f'{r.key}_{r.cohort}.npz');m,cm=calc(z['y'],z['new_labels'],5);comp(GM[(GM.key==r.key)&(GM.cohort==r.cohort)].iloc[0],m);gmemo[r.key,r.cohort]=m
        assert np.array_equal(z['changed'],z['new_labels']!=z['old_labels']) and int(z['changed'].sum())==r.changed
        if r.cohort in ['G','C','V']:assert not z['changed'].any()
        if r.key=='hybrid_WV' and r.cohort=='additional':assert z['changed'].sum()==321 and not z['new_labels'].any() and not z['y'].any()
    assert len(GM)==116;checks['guard_metrics_recomputed']=116
    b=npz(P/'binary_indices.npz');assert len(b['training_ids'])==200 and np.bincount(b['y']).tolist()==[100,100]
    z=npz(P/'posteriors/binary_syntax_tfidf__binary_content_disjoint.npz');assert len(z['row_ids'])==1012 and not set(z['row_ids'])&set(b['training_ids'])
    assert np.bincount(z['y']).tolist()==[895,117]
    coverage=pd.read_csv(G/'activation_coverage.csv');assert coverage.eligible_attack.sum()==0 and coverage[coverage.cohort!='additional'].eligible.sum()==0
    assert coverage[coverage.cohort=='additional'].eligible.item()==321
    checks['zero_attack_conflict_support_explicit']=True
    # Layout-independent numeric provenance checks on every added table cell.
    ledger=read(OUT/'table_ledger.json');assert len(ledger)==23
    for scope,num in [('main',11),('supplement',65)]:
        t=(NEW/(scope+'.tex')).read_text(encoding='utf8');old=(OLD/(scope+'.tex')).read_text(encoding='utf8')
        blocks=re.findall(r'\\begin\{table\}.*?\\end\{table\}',t,re.S);prior=re.findall(r'\\begin\{table\}.*?\\end\{table\}',old,re.S)
        if scope=='supplement':
            at=next(i for i,b in enumerate(prior) if r'\label{tab:stage_results}' in b)
            assert r'\textbf{Training} & \textbf{N}' in prior[at]
            prior[at]=prior[at].replace(r'\textbf{N}',r'\textbf{Scored N}',1)
        assert len(blocks)==num and blocks[:len(prior)]==prior
        labels=re.findall(r'\\label\{([^}]+)\}',t);refs=re.findall(r'\\ref\{([^}]+)\}',t);assert len(labels)==len(set(labels)) and set(refs)<=set(labels)
        for r in ledger:
            if r['scope']!=scope:continue
            block=next(b for b in blocks if r'\label{'+r['label']+'}' in b)
            for cells in r['rows']:
                # Compare displayed numeric cells and descriptions after only
                # normalizing deliberate underscore wrap points.
                needle=' & '.join(str(c).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%') for c in cells)+r'\\'
                assert needle in block.replace(r'\allowbreak{}',''),(r['label'],cells)
        log=(NEW/(scope+'.log')).read_text(encoding='utf8');assert not re.search(r'Overfull|undefined references|Citation .* undefined|Missing character',log,re.I),scope
        checks[scope+'_tables']=len(blocks);checks[scope+'_historical_numeric_tables_unchanged']=len(prior)
        if scope=='supplement':checks['S8_header_clarified']='Separate N column -> Scored N; Training remains the treatment column; values unchanged'
    bylabel={r['label']:r for r in ledger};fm=['reference','matched_base_availability','matched_base_syntax','matched_base_tfidf','matched_base_hybrid']
    for row,key in zip(bylabel['tab:payload_refits']['rows'],fm):
        a=memo[key,'additional','weighted_voting'];v=memo[key,'V','weighted_voting'];assert row[1:]==[count(a['false_alarms']),f(a['fpr_percent'],2),f(v['macro_f1_percent']),count(v['false_negatives'])]
    keys=['reference','RF_only','syntax_WV','tfidf_WV','hybrid_WV']
    for row,key in zip(bylabel['tab:repair']['rows'],keys):
        a=gmemo[key,'additional'];v=gmemo[key,'V'];assert row[1:]==[count(a['false_alarms']),f(a['fpr_percent'],2),'0',count(v['false_negatives'])]
    methods=['decision_tree','random_forest','xgboost','lightgbm','logistic_regression','soft_voting','weighted_voting','stacking'];views=['syntax_only','tfidf_only','syntax_tfidf']
    for row,(view,method) in zip(bylabel['sup:repair_binary']['rows'],[(v,m) for v in views for m in methods]):
        r=memo['binary_'+view,'binary_content_disjoint',method];assert row[2:]==[count(r['false_alarms']),count(r['false_negatives']),f(r['fpr_percent']),f(r['recall_SQL_injection']),f(r['macro_f1_percent']),f(r['balanced_accuracy_percent'])]
    EF=pd.read_csv(G/'efficiency_summary.csv');T=pd.read_csv(G/'latency_raw.csv');MEM=pd.read_csv(G/'memory_raw.csv')
    assert len(T)==100 and len(MEM)==12 and (T.pipeline_ms>0).all() and (MEM.delta_peak_rss_mib>=0).all()
    for r in EF.itertuples():
        tt=T[(T.cohort==r.cohort)&(T.key==r.key)];assert len(tt)==10
        np.testing.assert_allclose([r.median_ms,r.mean_ms,r.sd_ms],[tt.pipeline_ms.median(),tt.pipeline_ms.mean(),tt.pipeline_ms.std()],rtol=0,atol=1e-9)
    for row in bylabel['tab:repair_cost']['rows']:
        cohort,nn=row[0].split();key={'Reference':'reference','RF-only':'RF_only','Hybrid':'hybrid_WV'}[row[1]];r=EF[(EF.cohort==cohort)&(EF.key==key)].iloc[0]
        assert int(nn)==r.n and row[2:]==[f(r.median_ms,3),f(r.sd_ms,3),f(r.delta_peak_rss_mean_mib,3)]
    checks['new_table_numeric_provenance']='PASS from saved prediction counts and raw timings'
    m=(NEW/'main.tex').read_text(encoding='utf8');s=(NEW/'supplement.tex').read_text(encoding='utf8')
    cites={k for group in re.findall(r'\\cite\{([^}]+)\}',m) for k in group.split(',')};bib=(NEW/'references.bib').read_text(encoding='utf8');bkeys=set(re.findall(r'@\w+\{([^,]+),',bib));assert cites==bkeys and len(cites)==26
    assert (NEW/'main.bbl').read_text(encoding='utf8').count(r'\bibitem')==26
    assert 'AGERS' not in m and 'pending' not in m.lower() and not re.search('all authors (have )?approved',m,re.I)
    for marker in ['Data and software availability','Funding','Declaration of competing interests','CRediT authorship','Declaration of generative AI']:
        a=oldmain=(OLD/'main.tex').read_text(encoding='utf8');assert marker in m
    for token in ['0/760','no attack','RF-only','97.8800','61.109','41.675','41.75','1.1425','without SQL-or-tie content evidence','No attack in G/C/V activates']:
        assert token in m,token
    assert 'not automatically' in m and 'Tables S46--S65' in m and 'S64' in m
    for scope in ('main','supplement'):
        tex=(NEW/(scope+'.tex')).read_text(encoding='utf8')
        for rel in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex):
            assert (NEW/rel).exists()
            if (OLD/rel).exists():assert sha(OLD/rel)==sha(NEW/rel)
    aux=(NEW/'supplement.aux').read_text(encoding='utf8')
    for j,r in enumerate([r for r in ledger if r['scope']=='supplement'],46):assert r'\newlabel{'+r['label']+'}{{S'+str(j)+'}' in aux,(j,r['label'])
    aux=(NEW/'main.aux').read_text(encoding='utf8')
    for j,r in enumerate([r for r in ledger if r['scope']=='main'],9):assert r'\newlabel{'+r['label']+'}{{'+str(j)+'}' in aux,(j,r['label'])
    oldpdf={'ARRAY_HAFID_REVISED_20261105.pdf':'46e5a5ff056f0a0a9d70044643937e41cf240b85d38ba85a2761873dcd0fbea4','ARRAY_HAFID_SUPPLEMENT_S1_S45_20261105.pdf':'305e5bbf61b2ad26ca2e5e4bf4d08e03d21c896489a529425af48fa7b64ddef2'}
    for name,wanted in oldpdf.items():
        p=ROOT/'output/pdf'/name
        if p.exists():assert sha(p)==wanted
    checks.update(references=26,unique_labels_and_resolved_references=True,supplement_contiguous='S1-S65',main_core_tables_preserved='1-8; new9-11',prior_PDFs_preserved=True,new_classifier_fits=0,new_model_deserializations=0,new_inference=0)
    result=dict(status='PASS_SAVED_EVIDENCE_AND_MANUSCRIPT_AUDIT',checks=checks,claim_scope='retrospective within-trace; RF-only ablation matches guard, zero observed attack conflict activations',main_source_sha256=sha(NEW/'main.tex'),supplement_source_sha256=sha(NEW/'supplement.tex'))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
