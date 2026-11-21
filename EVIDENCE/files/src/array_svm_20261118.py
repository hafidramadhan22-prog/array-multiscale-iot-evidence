"""Standalone SVM experiment helpers; label metrics and training-only selection."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score

CLASSES=['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']

def label_metrics(y,pred):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int)
    if y.shape!=pred.shape or y.ndim!=1 or len(y)==0 or not np.isin(y,range(5)).all() or not np.isin(pred,range(5)).all():
        raise ValueError('Invalid five-class labels')
    cm=confusion_matrix(y,pred,labels=range(5));support=cm.sum(1)
    complete=bool((support>0).all());normal=int(support[0]);attacks=int(support[1:].sum())
    result=dict(n=len(y),macro_f1=float(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)) if complete else None,
        balanced_accuracy=float(np.mean(cm.diagonal()/support)) if complete else None,
        ddos_f2=float(fbeta_score(y==1,pred==1,beta=2,zero_division=0)) if support[1] else None,
        errors=int((y!=pred).sum()),false_alarms=int(cm[0,1:].sum()),normal_support=normal,
        normal_fpr=float(cm[0,1:].sum()/normal) if normal else None,
        attack_support=attacks,attack_to_normal=int(cm[1:,0].sum()),
        binary_attack_recall=float(1-cm[1:,0].sum()/attacks) if attacks else None,
        label_agreement=float((y==pred).mean()),all_five_true_classes_present=complete)
    for j,name in enumerate(CLASSES):
        result['support_'+name]=int(support[j]);result['recall_'+name]=float(cm[j,j]/support[j]) if support[j] else None
        result['errors_'+name]=int(support[j]-cm[j,j])
    return result,cm

def scale_gamma(Z):
    mean=float(Z.mean());var=float(Z.multiply(Z).mean())-mean*mean
    if not np.isfinite(var) or var<=0:raise ValueError('SVM training variance must be positive')
    return 1./(Z.shape[1]*var)

def candidate_specs():
    return [dict(id=f'C{c:g}_G{g:g}',C=c,gamma_multiplier=g) for c in [.1,1.,10.] for g in [.1,1.,10.]]

def select_on_G(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unique G candidates required')
    for r in rows:
        if r.get('role')!='G' or not np.isfinite([r['score'],r['macro_f1'],r['ddos_f2']]).all():raise ValueError('Selection requires valid G-only metrics')
    def key(r):return (r['score'],r['macro_f1'],-abs(np.log10(r['C']))-abs(np.log10(r['gamma_multiplier'])),r['id'])
    return max(rows,key=key)['id']
