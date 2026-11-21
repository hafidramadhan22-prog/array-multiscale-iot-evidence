"""Summarize actual frozen SVM labels and cost records; no estimator loading."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array_svm_20261118 import label_metrics,CLASSES

EVAL='ARRAY_SVM_EVALUATION_20261118_01';STRESS='ARRAY_SVM_STRESS_PORTABLE_20261118_01';SERVE='ARRAY_SVM_SERVING_20261118_01';TRAIN='ARRAY_SVM_FULL_20261118_01'
METHODS=['WV','RF','LR','Uniform','SVM_default','SVM_G_selected']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def savefig(fig,path):
    for suffix in ['.png','.pdf','.svg']:fig.savefig(path.with_suffix(suffix),bbox_inches='tight',dpi=180)
    plt.close(fig)

def main(root,out):
    ev=root/'runs'/EVAL;stress=root/'runs'/STRESS;serve=root/'runs'/SERVE;train=root/'runs'/TRAIN
    assert read(ev/'complete.json')['status']=='PASS_SVM_MATCHED_EVALUATION'
    assert read(stress/'complete.json')['status']=='PASS_SVM_POST_FREEZE_STRESS'
    assert read(serve/'complete.json')['status']=='PASS_SVM_MATCHED_LABEL_SERVING'
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    stage=pd.read_csv(ev/'metrics.csv');stage.to_csv(out/'stage_metrics.csv',index=False)
    protocols={};rows=[];pooled={};contrasts=[]
    for prefix,protocol in [('private','packet_private'),('signature','signature_shared')]:
        zs=[load(ev/'stages'/f'{prefix}_fold{k}'/'evaluation.npz') for k in [1,2]]
        ids=np.concatenate([z['row_ids'] for z in zs]);assert len(np.unique(ids))==len(ids)==52584
        z={key:np.concatenate([x[key] for x in zs]) for key in ['row_ids','y']+METHODS};pooled[protocol]=z
        protocols[protocol]={}
        for method in METHODS:
            m,cm=label_metrics(z['y'],z[method]);protocols[protocol][method]=m
            rows.append(dict(protocol=protocol,method=method,aggregation='pooled OOF; not mean fold F1',macro_f1_sd=None,n_repeats=2,**m))
            pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/f'CM_{protocol}_{method}.csv')
        np.savez_compressed(out/(protocol+'_pooled_predictions.npz'),**z)
    protocols['temporal']={}
    for method in METHODS:
        d=stage[(stage.protocol=='temporal')&(stage.method==method)];assert len(d)==3 and d.n.nunique()==1
        m={}
        for col in d.columns:
            if col in ['stage','protocol','method']:continue
            m[col]=bool(d[col].all()) if col=='all_five_true_classes_present' else int(d[col].iloc[0]) if col in ['n','normal_support','attack_support'] or col.startswith('support_') else float(d[col].mean())
        protocols['temporal'][method]=m
        rows.append(dict(protocol='temporal',method=method,aggregation='mean of three estimator seeds on fixed targets',macro_f1_sd=float(d.macro_f1.std(ddof=1)),n_repeats=3,**m))
    summary=pd.DataFrame(rows);summary.to_csv(out/'protocol_summary.csv',index=False)
    for protocol,values in protocols.items():
        for method in ['SVM_default','SVM_G_selected','RF']:
            contrasts.append(dict(protocol=protocol,comparator=method,WV_macro_f1_percent=100*values['WV']['macro_f1'],comparator_macro_f1_percent=100*values[method]['macro_f1'],WV_minus_comparator_pp=100*(values['WV']['macro_f1']-values[method]['macro_f1']),WV_higher=bool(values['WV']['macro_f1']>values[method]['macro_f1']),WV_FPR_percent=100*values['WV']['normal_fpr'],comparator_FPR_percent=100*values[method]['normal_fpr']))
    pd.DataFrame(contrasts).to_csv(out/'goal_comparisons.csv',index=False)
    stressrows=[];stressdata={};rawstress=pd.read_csv(stress/'metrics.csv');rawstress.to_csv(out/'stress_per_capture.csv',index=False)
    for family in ['additional_aligned','later_block3','CIC','TON']:
        groups=rawstress[rawstress.family==family].cohort.unique();zs=[load(stress/'predictions'/(n+'.npz')) for n in groups]
        z={key:np.concatenate([x[key] for x in zs]) for key in ['row_ids','y']+METHODS};assert len(np.unique(z['row_ids']))==len(z['y'])
        stressdata[family]={}
        for method in METHODS:
            m,cm=label_metrics(z['y'],z[method]);stressdata[family][method]=m
            stressrows.append(dict(family=family,method=method,scope='retrospective source-specific diagnostic; not new final test',**m))
            pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/f'CM_{family}_{method}.csv')
    pd.DataFrame(stressrows).to_csv(out/'stress_summary.csv',index=False)
    selection=[]
    for d in sorted((train/'stages').iterdir()):
        f=read(d/'selection_before_V.json');g=pd.read_csv(d/'G_candidates.csv');s=g[g.id==f['selected']].iloc[0]
        plan=read(d/'training_plan.json')['plan']
        selection.append(dict(stage=d.name,selected=f['selected'],default=f['default'],G_macro_f1_percent=100*float(s.macro_f1),G_DDoS_F2_percent=100*float(s.ddos_f2),selected_support_vectors=int(s.n_support),candidate_fits=len(g),training_before=sum(plan['before'].values()),training_after=sum(plan['after'].values()),encoded_features=plan['encoded_features'],total_fit_seconds=float(g.fit_seconds.sum()),concurrent_fit_times_not_isolated=True))
    pd.DataFrame(selection).to_csv(out/'selection_and_training.csv',index=False)
    timing=pd.read_csv(serve/'latency_raw.csv');memory=pd.read_csv(serve/'memory_raw.csv');cost=[]
    for (policy,batch),d in timing.groupby(['policy','batch'],sort=False):
        v=d.seconds.to_numpy()*1000;rss=memory[memory.policy==policy].delta_mib.to_numpy()
        cost.append(dict(policy=policy,batch=int(batch),calls=len(v),median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),sd_ms=float(np.std(v,ddof=1)),q25_ms=float(np.quantile(v,.25)),q75_ms=float(np.quantile(v,.75)),delta_RSS_mean_mib=float(np.mean(rss)),delta_RSS_sd_mib=float(np.std(rss,ddof=1)),RSS_repeats=len(rss)))
    pd.DataFrame(cost).to_csv(out/'serving_summary.csv',index=False)
    # Plots reflect raw saved results and retain the stronger RF comparator.
    fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(3);width=.19;colors=['#195D8A','#328969','#BD7B19','#8455A8']
    for j,method in enumerate(['WV','RF','SVM_default','SVM_G_selected']):
        vals=[100*protocols[p][method]['macro_f1'] for p in ['temporal','packet_private','signature_shared']]
        ax.bar(x+(j-1.5)*width,vals,width,label=method.replace('_',' '),color=colors[j])
    ax.set_xticks(x,['Temporal\n3-seed mean','Packet-private\npooled OOF','Signature/shared\npooled OOF']);ax.set_ylabel('Macro-F1 (%)');ax.set_ylim(0,105);ax.legend(ncol=2,loc='lower left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();savefig(fig,out/'figures'/'macro_f1_protocols')
    fig,axes=plt.subplots(2,2,figsize=(9,8));z=pooled['packet_private']
    for ax,method in zip(axes.flat,['WV','RF','SVM_default','SVM_G_selected']):
        m,cm=label_metrics(z['y'],z[method]);den=cm.sum(1,keepdims=True);ax.imshow(cm/den,cmap='Blues',vmin=0,vmax=1)
        for i in range(5):
            for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]/den[i,0]>.55 else '#202020')
        ax.set_xticks(range(5),['Normal','DDoS','Scan','SQL','Ransom'],rotation=35,ha='right');ax.set_yticks(range(5),['Normal','DDoS','Scan','SQL','Ransom']);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(method.replace('_',' ')+f" | F1 {100*m['macro_f1']:.2f}%",fontsize=11)
    fig.suptitle('Packet-private history | 52,584 out-of-fold targets\nColour: row proportion; text: observed count',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));savefig(fig,out/'figures'/'packet_private_confusions')
    report=dict(status='PASS_SVM_SUMMARIES_FROM_ACTUAL_LABELS_AND_RAW_COSTS',date='2026-11-18',new_SVC_fits=63,smoke_fits_excluded=1,protocols=protocols,comparisons=contrasts,stress=stressdata,selection=selection,serving=cost,
        requested_WV_higher_than_both_SVMs_in_every_protocol=all(c['WV_higher'] for c in contrasts if c['comparator'].startswith('SVM')),
        conclusion='WV does not consistently outperform the SVM comparators. Default SVM exceeds WV under packet-private history; G-selected SVM slightly exceeds WV on temporal evaluation. Shared-signature results favour WV. Report all protocols and both SVM configurations.',
        WV_composition_changed=False,WV_FPR_changed=False,RF_retained=True,new_independent_capture=False,manuscript_changed=False)
    write(out/'analysis.json',report);print(report['conclusion'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);a=p.parse_args();main(a.root.resolve(),(a.output or a.root/'reports/ARRAY_SVM_20261118').resolve())
