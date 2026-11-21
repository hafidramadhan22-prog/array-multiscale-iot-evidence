"""Saved-evidence and manuscript verification only; never loads fitted models."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score,fbeta_score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ARRAY_FINAL_INTEGRATION_20261110'
BASE=ROOT/'paper/array_20261110_integrated/elsarticle';OLD=ROOT/'paper/array_20261109_payload_audit/elsarticle'
R=ROOT/'runs/FPR_SQL_GUARD_20261109_01';EXT=ROOT/'runs/INDEPENDENT_CAPTURE_20261109_01'
CLASSES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware'];K='sql_guard_rf_rescue'

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tables(s):return re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S)
def body(s):return re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',s,re.S)[0]
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def score(y,p):
    pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(5));n=cm.sum(1)
    result=dict(n=len(y),normal_support=int(n[0]),attack_support=int(n[1:].sum()),false_alarms=int(cm[0,1:].sum()),attack_to_Normal=int(cm[1:,0].sum()),errors=int((y!=pred).sum()),FPR_percent=100*cm[0,1:].sum()/n[0] if n[0] else None,macro_F1_percent=100*f1_score(y,pred,labels=range(5),average='macro',zero_division=0) if (n>0).all() else None,balanced_accuracy_percent=100*balanced_accuracy_score(y,pred) if (n>0).all() else None)
    for i,c in enumerate(CLASSES):result['recall_'+c]=100*cm[i,i]/n[i] if n[i] else None
    return result,cm
def compare(saved,actual):
    for k,v in actual.items():
        if v is None:assert pd.isna(saved[k]),k
        else:assert abs(float(saved[k])-v)<1e-9,(k,saved[k],v)
def gv(z):
    v=z['weighted_reference'];r=z['random_forest'];a=v.argmax(1);b=r.argmax(1);p=v.copy();use=((a==3)&(b==0))|((a==0)&(b!=0));p[use]=r[use];return p

def main():
    main=(BASE/'main.tex').read_text('utf8');sup=(BASE/'supplement.tex').read_text('utf8')
    expected_counts=read(OUT/'build_summary.json')
    assert len(tables(main))==15 and len(tables(sup))==95
    for name,source,count in [('main',main,12),('supplement',sup,73)]:
        prior=tables((OLD/(name+'.tex')).read_text('utf8'))
        now=tables(source)
        assert len(prior)==count
        for old,new in zip(prior,now):assert body(old)==body(new),'Previous numeric table changed'
        for match in re.finditer(r'\\begin\{table\}',source):
            # No new nested/unclosed table or minipage allowed.
            assert source[:match.start()].count(r'\begin{table}')==source[:match.start()].count(r'\end{table}')
    labels=re.findall(r'\\label\{([^}]+)',main);assert len(labels)==len(set(labels))
    assert set(re.findall(r'\\ref\{([^}]+)',main))<=set(labels)
    for label in ['tab:factorial','tab:history','tab:ensemble','tab:benign','tab:buffer','tab:runtime','tab:faith','tab:guard_fpr','tab:external_normal','tab:external_attack']:
        assert r'\ref{'+label+'}' in main
    for label in ['fig:flow','fig:factorial','fig:history','fig:fpr_transfer']:
        assert r'\ref{'+label+'}' in main
    assert 'author approval pending' not in (main+sup).lower()
    assert 'AGERS' not in main+sup
    bibliography=(BASE/'references.bib').read_text('utf8');oldbib=(OLD/'references.bib').read_text('utf8');assert bibliography.startswith(oldbib)
    keys=set(re.findall(r'@\w+\{([^,]+),',bibliography));used=set()
    for match in re.findall(r'\\cite\{([^}]+)',main):used.update(match.split(','))
    assert keys==used and len(keys)==30,(keys-used,used-keys)
    for key in ['r27','r28','r29']:
        m=read(OUT/'references'/(key+'.json'))['message'];assert m['DOI'].lower() in bibliography.lower() and m['title'][0].replace('_',r'\_') in bibliography

    cached=pd.read_csv(ROOT/'reports/FPR_SOLUTION_20261109/metrics_all_cached.csv').set_index(['cohort','policy'])
    checked=0;outputs=[]
    for cohort in ['additional760','G','C','V_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']:
        z=load(R/'posteriors'/(cohort+'.npz'));np.testing.assert_allclose(gv(z),z[K],rtol=0,atol=1e-12)
        for policy in ['weighted_reference','random_forest',K]:
            m,cm=score(z['y'],z[policy]);compare(cached.loc[(cohort,policy)],m);checked+=1
            outputs.append(dict(cohort=cohort,policy=policy,**m))
    later=pd.read_csv(ROOT/'reports/FPR_SOLUTION_20261109/metrics_later_blocks.csv').set_index(['block','policy']);later_ids=set()
    for b,run in [(1,'FPR_TARGET8_20261109_01'),(2,'FPR_NORMAL_VOTING_20261109_01'),(3,'FPR_SQL_GUARD_20261109_01')]:
        blocks=[load(p) for p in sorted((ROOT/'runs'/run/'posteriors').glob('*_later.npz'))]
        ids=np.concatenate([z['row_ids'] for z in blocks]);assert len(ids)==len(set(ids)) and not set(ids)&later_ids;later_ids.update(ids)
        y=np.concatenate([z['y'] for z in blocks]);assert not y.any()
        for policy in ['weighted_reference','random_forest',K]:
            p=np.concatenate([gv(z) if policy==K else z[policy] for z in blocks]);m,_=score(y,p);compare(later.loc[(b,policy)],m);outputs.append(dict(cohort='later'+str(b),policy=policy,**m))
    external=pd.read_csv(ROOT/'reports/INDEPENDENT_CAPTURE_20261109/primary_policy_metrics_verified.csv').set_index(['source','policy']);normal_totals={};ton_missed=0;ton_typed=0
    for source in ['CIC_Normal0','CIC_Normal3','CIC_SQL','CIC_PortScan','CIC_DDoS_SYN','TON_Ransom1','TON_Ransom2']:
        z=load(EXT/source/'probabilities.npz');y=z.get('y_scenario',z.get('y'));np.testing.assert_allclose(gv(z),z[K],rtol=0,atol=1e-12)
        for policy in ['weighted_reference','random_forest',K]:
            m,cm=score(y,z[policy]);compare(external.loc[(source,policy)],m)
            np.testing.assert_array_equal(cm,pd.read_csv(EXT/source/('CM_'+policy+'.csv'),index_col=0).to_numpy())
            outputs.append(dict(cohort=source,policy=policy,**m))
            family=source.split('_')[0];key=(family,policy);v=normal_totals.setdefault(key,[0,0]);v[0]+=m['normal_support'];v[1]+=m['false_alarms']
            if source.startswith('TON') and policy==K:ton_missed+=m['attack_to_Normal'];ton_typed+=cm[4,4]
    assert normal_totals[('CIC',K)]==[2000,638] and normal_totals[('TON',K)]==[2000,377]
    assert ton_missed==1657 and ton_typed==31
    pd.DataFrame(outputs).to_csv(OUT/'independently_recomputed_metrics.csv',index=False)
    assert '1.0104' in main and '0.1435' in main and '31.90' in main and '18.85' in main and '1,657/2,000' in main
    assert r'\section*{Declaration of generative AI' in main and r'\section*{Funding}' in main
    # Every newly generated table row must occur in the actual source.
    from build_array_final_integration_20261110 import esc
    ledger=read(OUT/'new_table_ledger.json')
    for t in ledger:
        source=main if t['scope']=='main' else sup
        block=next(b for b in tables(source) if r'\label{'+t['label']+'}' in b)
        for row in t['rows']:
            cells=' & '.join(esc(v) for v in row)+r'\\';assert cells in block,(t['label'],row)
    result=dict(status='PASS_TABLES_COUNTS_REFERENCES_AND_RECOMPUTED_EVIDENCE',new_fits=0,new_inference=0,old_main_table_bodies_preserved=12,old_supplement_table_bodies_preserved=73,new_tables_verified=len(ledger),cached_policy_metrics_recomputed=checked,later_blocks_recomputed=3,external_sources_recomputed=7,external_primary_decisions=27000,main_source_sha256=sha(BASE/'main.tex'),supplement_source_sha256=sha(BASE/'supplement.tex'),reference_count=30,FPR_below1_all_main_populations=False,external_target8_met=False,author_approval_not_fabricated=True,AI_disclosure_retained=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
