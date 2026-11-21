"""Factorial context definitions and metric guards for the frozen Array revision."""
import numpy as np
import pandas as pd
from e4 import DomainFeatures, columns, metrics
from multiscale_context import CONTEXT_NAMES

VIEWS=('c40_o1','c40_o10','c60_o1','c60_o10')

def reorder_context(values,view):
    if view not in VIEWS:raise ValueError('Unknown frozen factorial view')
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[1]!=60 or not np.isfinite(x).all():raise ValueError('Invalid context')
    h=x.reshape(-1,2,30).copy()
    for i in range(len(h)):
        key=lambda v: tuple(v[10:20])+tuple(v[:20]) if view.endswith('_o1') else tuple(v[20:30])+tuple(v)
        if key(h[i,1])>key(h[i,0]):h[i]=h[i,::-1]
    return h[:,:,:20].reshape(-1,40) if view.startswith('c40') else h.reshape(-1,60)

def frame(raw,values,view):
    context=reorder_context(values,view)
    names=[c for c in CONTEXT_NAMES if '_10000ms_' not in c] if view.startswith('c40') else list(CONTEXT_NAMES)
    num,cat=columns('noport25')
    X=pd.concat([DomainFeatures('noport25').transform(raw),pd.DataFrame(context,index=raw.index,columns=names)],axis=1)
    return X,list(num)+names,list(cat)

def scored(y,pred):
    result=metrics(y,pred)
    result['balanced_accuracy']=float(np.mean([result[k] for k in result if k.startswith('recall_')]))
    result['errors']=int(np.sum(np.asarray(y)!=pred))
    for c in range(5):result['errors_class_'+str(c)]=int(np.sum((np.asarray(y)==c)&(np.asarray(y)!=pred)))
    return result

def allocate_history(u,fractions,mode,target_role=None):
    f=np.asarray(fractions,dtype=float)
    if len(f)!=4 or (f<=0).any() or not np.isclose(f.sum(),1):raise ValueError('Role fractions')
    if target_role is not None:
        if target_role not in range(4):raise ValueError('Target role')
        return [target_role]
    if mode=='exclusive':return [min(int(np.searchsorted(np.cumsum(f),u,side='right')),3)]
    if mode=='shared_density':return np.flatnonzero(u<f).tolist()
    raise ValueError('Unreviewed history mode')
