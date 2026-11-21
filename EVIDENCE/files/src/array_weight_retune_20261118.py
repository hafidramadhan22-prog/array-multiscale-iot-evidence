"""Canonical finite WV weight search; selection inputs are G labels only."""
import numpy as np

def compositions(total,n):
    if n==1:
        yield (total,);return
    for first in range(total+1):
        for tail in compositions(total-first,n-1):yield (first,)+tail

def simplex_grid():
    w=.05+.05*np.asarray(list(compositions(15,5)),dtype=np.float64)
    assert w.shape==(3876,5) and np.all(w>=.05) and np.allclose(w.sum(1),1,atol=1e-12)
    return w

def canonical_mixture(p,w):
    p=np.asarray(p,dtype=np.float64);w=np.asarray(w,dtype=np.float64)
    if p.ndim!=3 or p.shape[1:]!=(5,5) or w.shape!=(5,):raise ValueError('Five expert/class dimensions required')
    return sum(float(w[j])*p[:,j] for j in range(5))

def search_labels(p,weights,chunk=64):
    p=np.asarray(p,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if p.ndim!=3 or p.shape[1:]!=(5,5) or weights.ndim!=2 or weights.shape[1]!=5:raise ValueError('Probability/weight dimensions')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(2),1,atol=1e-6):raise ValueError('Invalid expert probabilities')
    if not np.isfinite(weights).all() or (weights<.05-1e-14).any() or not np.allclose(weights.sum(1),1,atol=1e-12):raise ValueError('Weight floor/sum')
    labels=np.empty((len(weights),len(p)),dtype=np.uint8)
    for start in range(0,len(weights),chunk):
        w=weights[start:start+chunk];s=np.zeros((len(w),len(p),5),dtype=np.float64)
        for j in range(5):s+=w[:,j,None,None]*p[None,:,j,:]
        labels[start:start+len(w)]=s.argmax(2)
    return labels

def counts_and_scores(y,labels):
    y=np.asarray(y,dtype=int);labels=np.asarray(labels,dtype=int)
    if labels.ndim!=2 or labels.shape[1]!=len(y) or set(y)!=set(range(5)) or not np.isin(labels,range(5)).all():raise ValueError('G requires all five classes')
    cm=np.bincount((25*np.arange(len(labels))[:,None]+5*y[None,:]+labels).ravel(),minlength=25*len(labels)).reshape(-1,5,5)
    tp=np.diagonal(cm,axis1=1,axis2=2);support=cm.sum(2);fp=cm.sum(1)-tp;fn=support-tp
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0).mean(1)
    f2=5*tp[:,1]/(5*tp[:,1]+4*fn[:,1]+fp[:,1]);score=.5*(f1+f2)
    return cm,f1,f2,score

def select_G(scores,weights,reference,feasible=None):
    scores=np.asarray(scores,dtype=float);weights=np.asarray(weights);reference=np.asarray(reference)
    ok=np.ones(len(scores),dtype=bool) if feasible is None else np.asarray(feasible,dtype=bool)
    if len(scores)!=len(weights) or not np.isfinite(scores).all() or not ok.any():raise ValueError('Invalid G candidate scores')
    eligible=np.flatnonzero(ok);best=max(scores[eligible]);tied=eligible[scores[eligible]==best]
    distance=np.abs(weights[tied]-reference).sum(1)
    return int(tied[np.lexsort((tied,distance))[0]])

def feasible_G(cm,reference_index=0):
    tp=np.diagonal(cm,axis1=1,axis2=2);base=tp[reference_index]
    normal_fp=cm[:,0,:].sum(1)-tp[:,0];old_fp=normal_fp[reference_index]
    result=(normal_fp<=old_fp)&(tp[:,1:]>=base[None,1:]).all(1)
    assert result[reference_index]
    return result
