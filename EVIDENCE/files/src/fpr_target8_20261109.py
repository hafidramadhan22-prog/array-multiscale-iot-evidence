"""Fixed existing-model comparisons and explicit empirical safety checks."""
import numpy as np
NAMES=['decision_tree','random_forest','xgboost','lightgbm','logistic_regression']
CLASSES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']

def alternatives(cube,weights):
    p=np.asarray(cube,dtype=float);w=np.asarray(weights,dtype=float)
    if p.ndim!=3 or p.shape[1:]!=(5,5) or not np.isfinite(p).all():raise ValueError('Invalid five-class expert cube')
    if (p<0).any() or (p>1).any() or not np.allclose(p.sum(2),1,atol=1e-6):raise ValueError('Invalid probabilities')
    if w.shape!=(5,) or (w<0).any() or not np.isclose(w.sum(),1):raise ValueError('Invalid original weights')
    out={n:p[:,i] for i,n in enumerate(NAMES)}
    out.update(weighted_reference=np.einsum('nmc,m->nc',p,w),uniform_reference=p.mean(1),RF_XGB_50=.5*p[:,1]+.5*p[:,2],RF_XGB_75=.75*p[:,1]+.25*p[:,2])
    return out

def metrics(y,pred):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int)
    if y.shape!=pred.shape or y.ndim!=1 or not np.isin(y,range(5)).all() or not np.isin(pred,range(5)).all():raise ValueError('Invalid label vectors')
    cm=np.bincount(5*y+pred,minlength=25).reshape(5,5);n=cm.sum(1);tp=cm.diagonal();pn=cm.sum(0)
    f1=np.divide(2*tp,n+pn,out=np.zeros(5,float),where=n+pn>0)
    precision=np.divide(tp,pn,out=np.zeros(5,float),where=pn>0);recall=np.divide(tp,n,out=np.zeros(5,float),where=n>0)
    ddosden=4*precision[1]+recall[1]
    result=dict(n=len(y),normal_support=int(n[0]),attack_support=int(n[1:].sum()),false_alarms=int(cm[0,1:].sum()),attack_to_Normal=int(cm[1:,0].sum()),errors=int(cm.sum()-tp.sum()),FPR_percent=100*cm[0,1:].sum()/n[0] if n[0] else None,attack_recall_percent=100*(1-cm[1:,0].sum()/n[1:].sum()) if n[1:].sum() else None,macro_F1_percent=float(100*f1.mean()) if (n>0).all() else None,balanced_accuracy_percent=float(100*recall.mean()) if (n>0).all() else None,DDoS_F2_percent=float(100*5*precision[1]*recall[1]/ddosden) if n[1] and ddosden else (0. if n[1] else None))
    for i,c in enumerate(CLASSES):result['support_'+c]=int(n[i]);result['recall_'+c]=float(100*recall[i]) if n[i] else None
    return result,cm

def safety_gate(y,reference,candidate,macro_drop_pp=.1):
    r,_=metrics(y,reference);c,_=metrics(y,candidate)
    if any(c['support_'+name]==0 for name in CLASSES[1:]):return dict(evaluable=False,passed=None,reason='All four attack classes are required; benign-only data cannot validate attack sensitivity.')
    attack_ok=c['attack_to_Normal']<=r['attack_to_Normal'];recalls_ok=all(c['recall_'+name]>=r['recall_'+name]-1e-12 for name in CLASSES[1:])
    f1_ok=c['macro_F1_percent']>=r['macro_F1_percent']-macro_drop_pp
    return dict(evaluable=True,passed=bool(attack_ok and recalls_ok and f1_ok),attack_to_Normal_not_increased=bool(attack_ok),no_attack_class_recall_drop=bool(recalls_ok),macro_F1_drop_within_limit=bool(f1_ok),macro_F1_delta_pp=c['macro_F1_percent']-r['macro_F1_percent'],attack_to_Normal_delta=c['attack_to_Normal']-r['attack_to_Normal'],recall_deltas_pp={name:c['recall_'+name]-r['recall_'+name] for name in CLASSES[1:]},scope='empirical comparison on these correlated targets; no independent non-inferiority claim')
