"""Derive publication numbers directly from guarded saved predictions."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from run_array_factorial_20261103 import read_npz,methods
from array_revision import scored,VIEWS
OUT=ROOT/'reports/ARRAY_MAJOR_REVISION_20261103';OUT.mkdir(exist_ok=True)
F=ROOT/'runs/ARRAY_FACTORIAL_FULL_20261103_01';H=ROOT/'runs/ARRAY_HISTORY_FULL_20261103_01';B=ROOT/'runs/ARRAY_BUFFER_FULL_20261103_01'

def pooled(paths,mask=None):
    zs=[read_npz(p) for p in paths];ys=[];ps={};ids=[]
    for z in zs:
        m=np.ones(len(z['y']),bool) if mask is None else mask(z)
        ys.append(z['y'][m]);ids.extend(z['row_ids'][m]);pr=methods(z['p_experts'],z['weights'])
        for k,p in pr.items():ps.setdefault(k,[]).append(p[m].argmax(1))
    assert len(ids)==len(set(ids))
    return np.concatenate(ys),{k:np.concatenate(p) for k,p in ps.items()},np.asarray(ids)

def main():
    summary=[];fm=pd.read_csv(F/'metrics.csv')
    for (view,method),g in fm[fm.stage.str.startswith('temporal')].groupby(['view','method']):
        row=dict(protocol='temporal',view=view,method=method,n=10358,seeds=3)
        for k in g.select_dtypes('number').columns:
            if k!='n':row[k]=float(g[k].mean());row[k+'_sd']=float(g[k].std())
        summary.append(row)
    for view in VIEWS:
        y,p,ids=pooled([F/'stages'/f'signature_fold{k}'/view/'evaluation.npz' for k in [1,2]])
        for method,pred in p.items():summary.append(dict(protocol='signature',view=view,method=method,**scored(y,pred)))
    pd.DataFrame(summary).to_csv(OUT/'factorial_summary.csv',index=False)
    hr=[]
    for history in ['exclusive','shared_density']:
        y,p,ids=pooled([H/f'signature_fold{k}_{history}'/'evaluation.npz' for k in [1,2]])
        for method,pred in p.items():hr.append(dict(history=history,method=method,**scored(y,pred)))
    pd.DataFrame(hr).to_csv(OUT/'history_summary.csv',index=False)
    buf=pd.read_csv(B/'metrics.csv');br=[]
    for (cap,cohort,method),g in buf[buf.stage.str.startswith('temporal')].groupby(['capacity','cohort','method']):
        br.append(dict(protocol='temporal',capacity=cap,cohort=cohort,method=method,**{k:float(g[k].mean()) for k in g.select_dtypes('number').columns if k!='capacity'}))
    for cap in [4096,16384,65536]:
        for cohort,mask in [('warm1',None),('common10',lambda z:z['warm10']),('early_excluded',lambda z:~z['warm10'])]:
            y,p,ids=pooled([B/f'signature_fold{k}_{cap}.npz' for k in [1,2]],mask)
            for method,pred in p.items():br.append(dict(protocol='signature',capacity=cap,cohort=cohort,method=method,**scored(y,pred)))
    pd.DataFrame(br).to_csv(OUT/'buffer_summary.csv',index=False)
    dis=[];examples=[]
    for stage in ['temporal_seed42','temporal_seed43','temporal_seed44','signature_fold1','signature_fold2']:
        z=read_npz(F/'stages'/stage/'c60_o10'/'evaluation.npz');pr={k:v.argmax(1) for k,v in methods(z['p_experts'],z['weights']).items()};y=z['y'];w=pr['weighted_voting']
        for method in ['uniform_voting','random_forest','logistic_regression']:
            p=pr[method]
            for cl in [-1,0,1,2,3,4]:
                m=np.ones(len(y),bool) if cl==-1 else y==cl
                dis.append(dict(stage=stage,comparator=method,true_class=cl,n=int(m.sum()),ensemble_only_correct=int(((w==y)&(p!=y)&m).sum()),comparator_only_correct=int(((p==y)&(w!=y)&m).sum()),both_wrong=int(((p!=y)&(w!=y)&m).sum()),label_disagreements=int(((p!=w)&m).sum())))
            for i in np.flatnonzero((w==y)!=(p==y)):
                examples.append(dict(stage=stage,row_id=int(z['row_ids'][i]),true_class=int(y[i]),comparator=method,weighted_label=int(w[i]),comparator_label=int(p[i]),winner='weighted' if w[i]==y[i] else method))
    pd.DataFrame(dis).to_csv(OUT/'correctness_disagreements.csv',index=False);pd.DataFrame(examples).to_csv(OUT/'disagreement_examples.csv',index=False)
    x=ROOT/'runs/ARRAY_XAI_FULL_20261103_01'
    if (x/'complete.json').exists():
        faith=pd.read_csv(x/'faithfulness.csv');faith[(faith.donor=='B32')&(faith.ranking_donor=='A32')&(faith.ranking=='signed')].groupby('method')[['deletion_auc','insertion_auc','deletion_gain','insertion_gain']].mean().to_csv(OUT/'xai_faithfulness.csv')
        pd.read_csv(x/'background_sensitivity.csv').groupby('method')[['top3_jaccard','normalized_l1','cosine']].mean().to_csv(OUT/'xai_sensitivity.csv')
        imp=pd.read_csv(x/'importance.csv');imp['abs_phi']=imp.phi.abs();s=imp.groupby(['background','method','group']).abs_phi.mean().reset_index();s['share']=s.abs_phi/s.groupby(['background','method']).abs_phi.transform('sum');s.to_csv(OUT/'xai_shares.csv',index=False)
    print(pd.DataFrame(summary).query("method=='weighted_voting'")[['protocol','view','macro_f1','balanced_accuracy','errors']].to_string(index=False))
    print(pd.DataFrame(hr).query("method=='weighted_voting'").to_string(index=False))
    print(pd.DataFrame(br).query("method=='weighted_voting' and cohort!='early_excluded'")[['protocol','capacity','cohort','n','macro_f1','recall_Ransomware','errors']].to_string(index=False))
if __name__=='__main__':main()
