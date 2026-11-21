"""Read-only summaries of frozen G-reweighted WV outcomes."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array_svm_20261118 import label_metrics,CLASSES

RUN='ARRAY_WEIGHT_RETUNE_FULL_20261118_01'
METHODS=['WV_reference','WV_G_constrained','WV_G_unconstrained','WV_G_heuristic','Uniform','DT','RF','XGB','LGB','LR','SVM_default','SVM_G_selected']
NEW=['WV_G_constrained','WV_G_unconstrained','WV_G_heuristic']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def strict(old,new):
    return dict(macro_f1_improved=bool(new['macro_f1']>old['macro_f1']),Normal_FPR_not_worse=bool(new['normal_fpr']<=old['normal_fpr']),attack_recalls_not_worse=bool(all(new['recall_'+c]>=old['recall_'+c] for c in CLASSES[1:])),strict_success=bool(new['macro_f1']>old['macro_f1'] and new['normal_fpr']<=old['normal_fpr'] and all(new['recall_'+c]>=old['recall_'+c] for c in CLASSES[1:])))

def main(root,out):
    run=root/'runs'/RUN;assert read(run/'complete.json')['status']=='PASS_WEIGHT_RETUNE_EXPERIMENT'
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    stages=pd.read_csv(run/'metrics.csv');stages.to_csv(out/'stage_metrics.csv',index=False)
    protocols={};rows=[];arrays={};paired=[]
    for prefix,protocol in [('private','packet_private'),('signature','signature_shared')]:
        zs=[load(run/'stages'/f'{prefix}_fold{k}'/'evaluation.npz') for k in [1,2]]
        z={k:np.concatenate([a[k] for a in zs]) for k in ['row_ids','y']+METHODS};assert len(np.unique(z['row_ids']))==52584
        arrays[protocol]=z;protocols[protocol]={}
        for method in METHODS:
            m,cm=label_metrics(z['y'],z[method]);protocols[protocol][method]=m;rows.append(dict(protocol=protocol,method=method,aggregation='pooled out-of-fold labels',macro_f1_sd=None,**m))
            pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/f'CM_{protocol}_{method}.csv')
        np.savez_compressed(out/(protocol+'_pooled_predictions.npz'),**z)
        oldwrong=z['WV_reference']!=z['y']
        for method in NEW:
            wrong=z[method]!=z['y'];corrected=int((oldwrong&~wrong).sum());lost=int((~oldwrong&wrong).sum())
            assert int(wrong.sum())==int(oldwrong.sum())-corrected+lost
            paired.append(dict(protocol=protocol,method=method,old_errors=int(oldwrong.sum()),new_errors=int(wrong.sum()),old_errors_corrected=corrected,previously_correct_lost=lost,changed_labels=int((z[method]!=z['WV_reference']).sum())))
    protocols['temporal']={}
    for method in METHODS:
        df=stages[(stages.protocol=='temporal')&(stages.method==method)];assert len(df)==3
        m={}
        for c in df.columns:
            if c in ['stage','protocol','method']:continue
            m[c]=bool(df[c].all()) if c=='all_five_true_classes_present' else int(df[c].iloc[0]) if c in ['n','normal_support','attack_support'] or c.startswith('support_') else float(df[c].mean())
        protocols['temporal'][method]=m;rows.append(dict(protocol='temporal',method=method,aggregation='three-seed mean on fixed targets',macro_f1_sd=float(df.macro_f1.std(ddof=1)),**m))
    pd.DataFrame(rows).to_csv(out/'protocol_summary.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_error_changes.csv',index=False)
    goals=[]
    for protocol,values in protocols.items():
        old=values['WV_reference']
        for method in NEW:
            new=values[method];goals.append(dict(protocol=protocol,method=method,macro_F1_change_pp=100*(new['macro_f1']-old['macro_f1']),Normal_FPR_change_pp=100*(new['normal_fpr']-old['normal_fpr']),errors_change=new['errors']-old['errors'],binary_attack_recall_change_pp=100*(new['binary_attack_recall']-old['binary_attack_recall']),binary_attack_recall_not_worse=bool(new['binary_attack_recall']>=old['binary_attack_recall']),**strict(old,new),exceeds_RF=bool(new['macro_f1']>values['RF']['macro_f1']),exceeds_default_SVM=bool(new['macro_f1']>values['SVM_default']['macro_f1']),exceeds_G_selected_SVM=bool(new['macro_f1']>values['SVM_G_selected']['macro_f1'])))
    pd.DataFrame(goals).to_csv(out/'goal_assessment.csv',index=False)
    stressmeta=pd.read_csv(run/'stress_metrics.csv');stressmeta.to_csv(out/'stress_per_capture.csv',index=False);stress={};stressrows=[]
    for family in ['additional_aligned','later_block3','CIC','TON']:
        names=stressmeta[stressmeta.family==family].cohort.unique();zs=[load(run/'stress'/(n+'.npz')) for n in names]
        stress[family]={}
        for method in [m for m in METHODS if not m.startswith('SVM')]:
            y=np.concatenate([z['y'] for z in zs]);pred=np.concatenate([z[method] for z in zs]);m,cm=label_metrics(y,pred);stress[family][method]=m
            stressrows.append(dict(family=family,method=method,**m));pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/f'CM_{family}_{method}.csv')
    pd.DataFrame(stressrows).to_csv(out/'stress_summary.csv',index=False)
    weights=[];Gchecks=[]
    for d in sorted((run/'stages').iterdir()):
        f=read(d/'selection_before_V.json');g=pd.read_csv(d/'G_candidates.csv')
        for policy in ['WV_reference']+NEW:
            i=f['selected'][policy];r=g[g.candidate==i].iloc[0]
            weights.append(dict(stage=d.name,policy=policy,candidate=i,G_score=float(r.G_score),G_macro_F1_percent=100*float(r.G_macro_f1),G_normal_FP=int(r.G_normal_FP),G_feasible=bool(r.G_feasible),**{name:float(v) for name,v in zip(['DT','RF','XGB','LGB','LR'],f['weights'][policy])}))
        Gchecks.append(dict(stage=d.name,candidates=len(g),feasible=int(g.G_feasible.sum()),primary_candidate=f['selected']['WV_G_constrained'],primary_G_score_gain=float(g.loc[g.candidate==f['selected']['WV_G_constrained'],'G_score'].iloc[0]-g.loc[g.candidate==0,'G_score'].iloc[0])))
    pd.DataFrame(weights).to_csv(out/'selected_weights.csv',index=False);pd.DataFrame(Gchecks).to_csv(out/'G_search_summary.csv',index=False)
    z=arrays['packet_private'];fig,axes=plt.subplots(1,2,figsize=(9,4.8));names=['WV_reference','WV_G_constrained']
    for ax,method in zip(axes,names):
        m,cm=label_metrics(z['y'],z[method]);den=cm.sum(1,keepdims=True);ax.imshow(cm/den,vmin=0,vmax=1,cmap='Blues')
        for i in range(5):
            for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,color='white' if cm[i,j]/den[i,0]>.55 else '#222')
        ax.set_xticks(range(5),['Normal','DDoS','Scan','SQL','Ransom'],rotation=40,ha='right');ax.set_yticks(range(5),['Normal','DDoS','Scan','SQL','Ransom']);ax.set_title(method.replace('_',' ')+f"\nF1 {100*m['macro_f1']:.2f}%");ax.set_xlabel('Predicted');ax.set_ylabel('True')
    fig.suptitle('Packet-private | 52,584 targets | counts and row-normalized colour');fig.tight_layout(rect=(0,0,1,.9))
    for ext in ['png','pdf','svg']:fig.savefig(out/'figures'/('private_confusions.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    conclusion='G reweighting improves packet-private macro-F1 modestly but does not surpass RF/default SVM; temporal scores and false alarms worsen for the primary grid policy. It is not a consistent detection-preserving improvement.'
    result=dict(status='PASS_WEIGHT_RETUNE_SUMMARIES',date='2026-11-18',primary='WV_G_constrained',protocols=protocols,goals=goals,stress=stress,weights=weights,G_search=Gchecks,paired_errors=paired,new_classifier_fits=0,models_deserialized=False,G_candidate_evaluations=27167,conclusion=conclusion,primary_strict_success_all_protocols=all(g['strict_success'] for g in goals if g['method']=='WV_G_constrained'),RF_retained=True,new_WV_weights_change_function=True,old_runtime_SHAP_contracts_inherited=False,journal_changed=False,new_independent_capture=False)
    write(out/'analysis.json',result);print(conclusion)
    print(pd.DataFrame(goals).query("method=='WV_G_constrained'").to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);a=p.parse_args();main(a.root.resolve(),(a.output or a.root/'reports/ARRAY_WEIGHT_RETUNE_20261118').resolve())
