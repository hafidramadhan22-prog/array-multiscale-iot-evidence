"""Explicit new predictor: reconcile SQL/Normal disagreement, never mutate WV."""
import numpy as np

def eligible_rows(weighted,rf,threshold=.5):
    w=np.asarray(weighted,dtype=float);r=np.asarray(rf,dtype=float)
    if w.ndim!=2 or w.shape!=r.shape or w.shape[1]!=5:raise ValueError('Expected same five-class posteriors')
    for p in (w,r):
        if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid posterior')
        np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-6)
    return (w.argmax(1)==3)&(r.argmax(1)==0)&(r[:,0]>=threshold)

def apply_guard(weighted,rf,has_content,binary=None,threshold=.5,use_payload=True):
    w=np.asarray(weighted,dtype=float);r=np.asarray(rf,dtype=float);has=np.asarray(has_content,dtype=bool)
    eligible=eligible_rows(w,r,threshold)
    if has.shape!=(len(w),):raise ValueError('Content identity/shape mismatch')
    keep_sql=np.zeros(len(w),dtype=bool)
    if use_payload:
        b=np.asarray(binary,dtype=float)
        if b.shape!=(len(w),2):raise ValueError('Expected aligned binary probabilities')
        check=eligible&has
        if check.any():
            if not np.isfinite(b[check]).all() or (b[check]<0).any():raise ValueError('Missing binary evidence for nonempty content')
            np.testing.assert_allclose(b[check].sum(1),1,rtol=0,atol=1e-6)
            # Ambiguous0.5/0.5 also keeps the SQL alert.
            keep_sql=check&(b[:,1]>=b[:,0])
    change=eligible&~keep_sql
    out=w.copy();out[change]=r[change]
    reason=np.full(len(w),'keep_original',dtype='<U64')
    reason[keep_sql]='keep_SQL_payload_evidence'
    reason[change&has]='RF_Normal_payload_Normal' if use_payload else 'RF_Normal_without_payload_guard'
    reason[change&~has]='RF_Normal_no_usable_payload'
    assert not ((out.argmax(1)!=w.argmax(1))&~change).any()
    assert (out[change].argmax(1)==0).all()
    return out,change,reason
