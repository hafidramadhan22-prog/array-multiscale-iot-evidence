"""A transparent changed predictor; RF determines Normal membership, voting types attacks."""
import numpy as np

def combine(voting,rf):
    v=np.asarray(voting,dtype=float);r=np.asarray(rf,dtype=float)
    if v.shape!=r.shape or v.ndim!=2 or v.shape[1]!=5 or not np.isfinite(v).all() or not np.isfinite(r).all():raise ValueError('Aligned finite five-class posteriors required')
    if (v<0).any() or (r<0).any() or not np.allclose(v.sum(1),1,atol=1e-6) or not np.allclose(r.sum(1),1,atol=1e-6):raise ValueError('Normalized nonnegative posteriors required')
    vl=v.argmax(1);rl=r.argmax(1);use_rf=(rl==0)|(vl==0)
    result=v.copy();result[use_rf]=r[use_rf]
    reason=np.where(rl==0,'RF_Normal',np.where(vl==0,'RF_rescues_voting_Normal','retain_voting_attack_type'))
    assert np.array_equal(result.argmax(1)==0,rl==0)
    return result,reason

def infer(bundle,X,names):
    rf=bundle['models']['random_forest'].predict_proba(X);out=rf.copy();active=np.flatnonzero(rf.argmax(1)!=0)
    if len(active):
        xx=X.iloc[active];weighted=np.zeros((len(active),5),float)
        for n,w in zip(names,bundle['weights']):weighted+=float(w)*(rf[active] if n=='random_forest' else bundle['models'][n].predict_proba(xx))
        out[active]=combine(weighted,rf[active])[0]
    return out
