"""Conditional two-view WB fusion with training-only local minimax Brier weights."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from e4 import metrics
from wb_research import cube,mixture,WBPredictor
from wb_behavior import PRIOR_SYMMETRIC


def brier_rows(y,p):
    target=np.eye(5)[np.asarray(y,dtype=int)]
    return np.square(np.asarray(p)-target).sum(-1)


def gate_features(p0,context):
    p0=np.asarray(p0,dtype=float);context=np.asarray(context,dtype=float)
    if p0.ndim!=2 or p0.shape[1]!=5 or context.shape!=(len(p0),40) or not np.isfinite(context).all():raise ValueError('Gate input schema')
    ordered=np.sort(p0,axis=1)
    return np.column_stack([p0,ordered[:,-1],ordered[:,-1]-ordered[:,-2],-np.sum(p0*np.log(np.maximum(p0,1e-15)),axis=1),context])


def probabilities(p0,p1,alpha):
    p0=np.asarray(p0);p1=np.asarray(p1);a=np.asarray(alpha)
    if a.ndim==0:a=np.full(len(p0),a)
    if p0.shape!=p1.shape or a.shape!=(len(p0),) or ((a<0)|(a>1)).any():raise ValueError('Invalid mixture')
    return (1-a[:,None])*p0+a[:,None]*p1


def alpha_for(policy,features,p0):
    n=len(p0)
    if policy['kind']=='constant':return np.full(n,policy['alpha'],dtype=float)
    if policy['kind']=='confidence':return np.where(p0.max(1)<policy['threshold'],policy['alpha'],0.)
    if policy['kind']=='leaf':
        leaves=policy['tree'].apply(features)
        return np.array([policy['leaf_alpha'].get(int(leaf),0.) for leaf in leaves])
    raise ValueError('Unknown fusion policy')


def fit_partition(features,p0,p1s,y,depth,min_leaf,seed,robust,ddos_weight):
    losses=np.stack([brier_rows(y,p) for p in p1s]);gain=brier_rows(y,p0)-losses
    target=gain.min(0) if robust else gain[0]
    tree=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=min_leaf,random_state=seed)
    tree.fit(features,target,sample_weight=np.where(y==1,ddos_weight,1.))
    return tree


def fit_leaf_policy(tree,features,p0,p1s,y,grid,regularization,branch_penalty,ddos_weight,robust):
    leaves=tree.apply(features);mapping={};audit=[]
    scenarios=p1s if robust else p1s[:1]
    for leaf in np.unique(leaves):
        ix=leaves==leaf;w=np.where(y[ix]==1,ddos_weight,1.);candidates=[]
        for a in grid:
            losses=[float(np.average(brier_rows(y[ix],probabilities(p0[ix],p[ix],a)),weights=w)) for p in scenarios]
            score=max(losses)+regularization*a*a+branch_penalty*int(a>0)
            candidates.append(dict(alpha=float(a),objective=score,domain_losses=losses))
        best=min(candidates,key=lambda r:(r['objective'],r['alpha']));mapping[int(leaf)]=best['alpha']
        audit.append(dict(leaf=int(leaf),rows=int(ix.sum()),selected=best,candidates=candidates))
    return dict(kind='leaf',tree=tree,leaf_alpha=mapping),audit


def candidate_metrics(y,p0,p1s,alpha):
    base=metrics(y,p0.argmax(1));base_fn=int(np.sum((y==1)&(p0.argmax(1)!=1)));rows=[]
    for p1 in p1s:
        p=probabilities(p0,p1,alpha);pred=p.argmax(1);m=metrics(y,pred)
        fn=int(np.sum((y==1)&(pred!=1)))
        rows.append(dict(**m,ddos_false_negatives=fn,eligible=bool(m['macro_f1']>=base['macro_f1']-1e-15 and fn<=base_fn)))
    return dict(eligible=all(r['eligible'] for r in rows),worst_score=min(.5*(r['macro_f1']+r['ddos_f2']) for r in rows),
        escalation=float(np.mean(alpha>0)),mean_alpha=float(np.mean(alpha)),domains=rows)


def choose_candidate(records):
    eligible=[r for r in records if r['eligible']]
    if not eligible:raise ValueError('Explicit alpha-zero fallback must be eligible')
    return max(eligible,key=lambda r:(r['worst_score'],-r['escalation'],-r['mean_alpha'],-r['complexity'],-r['candidate_index']))


class FusionPredictor:
    """Full p0 for routing; p1 callback is genuinely skipped on alpha-zero rows."""
    def __init__(self,bundle25,bundle29,policy,baseline=None):
        self.b25=bundle25;self.b29=bundle29;self.policy=policy;self.baseline=baseline
        self.last_escalation=0
    @staticmethod
    def full(bundle,X):return mixture(cube(bundle['models'],X),bundle['weights'])
    def predict(self,frames):
        X25,X29=frames
        if not X25.index.equals(X29.index):raise ValueError('Two views must align')
        if self.baseline is not None:
            which,mode=self.baseline;bundle=self.b25 if which=='25' else self.b29;frame=X25 if which=='25' else X29
            return WBPredictor(bundle['models'],bundle['weights'],bundle['order'],mode).predict(frame)
        if self.policy['kind']=='constant' and self.policy['alpha']==1:
            self.last_escalation=len(X29);return self.full(self.b29,X29).argmax(1)
        p0=self.full(self.b25,X25);features=gate_features(p0,X25[PRIOR_SYMMETRIC].to_numpy())
        alpha=alpha_for(self.policy,features,p0);active=np.flatnonzero(alpha>0);self.last_escalation=len(active)
        if len(active):p0[active]=probabilities(p0[active],self.full(self.b29,X29.iloc[active]),alpha[active])
        return p0.argmax(1)
