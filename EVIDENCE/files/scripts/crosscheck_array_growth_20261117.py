"""Additional descriptive review from saved results; no paper/model mutation."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ARRAY_GROWTH_CROSSCHECK_20261117'
INPUTS=[
'paper/array_20261117_revision/latex/main.tex',
'reports/ARRAY_MAJOR_REVISION_20261103/factorial_summary.csv',
'runs/ARRAY_RUNTIME_FULL_20261103_01/latency_raw.csv',
'runs/ARRAY_RUNTIME_FULL_20261103_01/memory_raw.csv',
'runs/ARRAY_XAI_FULL_20261103_01/faithfulness.csv',
'runs/ARRAY_XAI_FULL_20261103_01/groups/A32/coalitions.npz',
'runs/ARRAY_XAI_FULL_20261103_01/groups/B32/coalitions.npz']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before={f:sha(ROOT/f) for f in INPUTS}
    s=(ROOT/INPUTS[0]).read_text('utf8');related=s.split(r'\section{Related work}')[1].split(r'\section{Study design}')[0]
    # This is a whitespace-token count, not a publishing requirement.
    words=len(related.split())
    l=pd.read_csv(ROOT/INPUTS[2]);summary=l.groupby(['policy','batch']).seconds.agg(n='size',median='median',mean='mean',sd='std',q25=lambda a:a.quantile(.25),q75=lambda a:a.quantile(.75)).reset_index()
    for col in ['median','mean','sd','q25','q75']:summary[col+'_ms']=1000*summary.pop(col)
    summary.to_csv(OUT/'latency_descriptive.csv',index=False)
    pivot=summary.pivot(index='policy',columns='batch',values='median_ms')
    contrasts=[]
    for batch in [1,1000]:
        for singleton in ['random_forest','logistic_regression']:
            a=float(pivot.loc['compact_certified',batch]);b=float(pivot.loc[singleton,batch])
            contrasts.append(dict(batch=batch,reference='compact_certified_WV',comparator=singleton,compact_median_ms=a,comparator_median_ms=b,
                compact_latency_ratio=a/b,compact_relative_latency_reduction_percent=100*(1-a/b)))
    pd.DataFrame(contrasts).to_csv(OUT/'batch_tradeoff.csv',index=False)
    f=pd.read_csv(ROOT/INPUTS[1]);f=f[(f.protocol=='temporal')&(f.method=='weighted_voting')].copy()
    f[['view','seeds','macro_f1','macro_f1_sd','errors','errors_sd']].to_csv(OUT/'temporal_WV_seed_variation.csv',index=False)
    faith=pd.read_csv(ROOT/INPUTS[4]);q=faith[(faith.donor=='B32')&(faith.ranking_donor=='A32')&(faith.ranking=='signed')]
    g=q.groupby(['method','true_class']).agg(n=('row_id','size'),deletion_gain_mean=('deletion_gain','mean'),deletion_gain_sd=('deletion_gain','std'),
        insertion_gain_mean=('insertion_gain','mean'),insertion_gain_sd=('insertion_gain','std'),deletion_gain_nonpositive=('deletion_gain',lambda a:int((a<=0).sum())),correct=('correct','sum')).reset_index()
    g.to_csv(OUT/'XAI_per_class_descriptive.csv',index=False)
    gains=q.groupby(['method','correct']).agg(n=('row_id','size'),deletion_gain_mean=('deletion_gain','mean'),insertion_gain_mean=('insertion_gain','mean')).reset_index()
    gains.to_csv(OUT/'XAI_correctness_descriptive.csv',index=False)
    a=dict(np.load(ROOT/INPUTS[5],allow_pickle=False));b=dict(np.load(ROOT/INPUTS[6],allow_pickle=False));rank=[]
    for method,j in [('weighted_voting',0),('random_forest',2),('logistic_regression',5)]:
        for i,row in enumerate(a['row_ids']):
            c=int(a['values'][i,-1,j].argmax());pa=np.abs(a['phi'][i,:,j,c]);pb=np.abs(b['phi'][i,:,j,c])
            ia=np.argsort(-pa,kind='stable');ib=np.argsort(-pb,kind='stable')
            rank.append(dict(row_id=int(row),method=method,true_class=int(a['y'][i]),top1_same=bool(ia[0]==ib[0]),
                ordered_top3_same=bool(np.array_equal(ia[:3],ib[:3])),full_rank_kendall_tau=float(kendalltau(pa,pb).statistic)))
    pd.DataFrame(rank).to_csv(OUT/'XAI_ordered_background_descriptive.csv',index=False)
    typo='RF'+chr(92)+"'s"
    result=dict(status='COMPLETE_READ_ONLY_GROWTH_CROSSCHECK',date='2026-11-17',related_work_whitespace_tokens=words,
        related_work_cited_reference_count=len(set(k for x in re.findall(r'\\cite\{([^}]+)',related) for k in x.split(','))),
        latex_possessive_typo=typo if typo in s else None,
        six_decimal_percentage_examples=re.findall(r'\d+\.\d{6}\\%',s),
        batch_tradeoffs=contrasts,XAI_WV_audit_targets=100,XAI_WV_wrong_targets=int((~q[q.method=='weighted_voting'].correct).sum()),
        interpretation='Descriptive post-hoc summaries of existing saved results; no independent episodes, hardware or confidence claims added.',
        input_hashes=before,papers_changed=False,archived_releases_changed=False,new_fits=0,new_inference=0,models_deserialized=False)
    assert before=={f:sha(ROOT/f) for f in INPUTS}
    (OUT/'crosscheck.json').write_text(json.dumps(result,indent=2)+'\n','utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'},indent=2))
if __name__=='__main__':main()
