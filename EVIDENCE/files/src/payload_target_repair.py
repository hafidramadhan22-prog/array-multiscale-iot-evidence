"""Binary content diagnosis and matched main-model payload intervention.

All previous protocols/modules stay frozen. Binary label 1 is SQL, not DDoS.
"""
import time,warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import ta_hafid_engine as e
from payload_tfidf_pilot import preprocessor
from payload_pilot import BASE_NAMES

def make_estimator(name,params,seed,nclasses):
    if nclasses==5:out=e.estimator(name,params,seed)
    elif nclasses==2:
        p={k:v for k,v in params.items() if k!='ddos_loss_weight'}
        if name=='xgboost':out=XGBClassifier(objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=seed,n_jobs=1,verbosity=0,**p)
        elif name=='lightgbm':out=LGBMClassifier(objective='binary',random_state=seed,n_jobs=1,verbosity=-1,**p)
        else:out=e.estimator(name,params,seed)
    else:raise ValueError('Unsupported label schema')
    if name=='logistic_regression':out.set_params(max_iter=10000,tol=1e-6)
    return out

def fit_group(X,y,fit,other,view,num,cat,cfg,nclasses):
    y=np.asarray(y);modes={};plans=[]
    for mode in ('tree','linear'):
        xt=X.iloc[fit];e.guard_training(xt,cfg,X.index[other],'training')
        prep=preprocessor(view,mode,num,cat,cfg);Z=prep.fit_transform(xt);yy=y[fit].copy()
        before={int(k):int(n) for k,n in zip(*np.unique(yy,return_counts=True))}
        if set(before)!=set(range(nclasses)):raise ValueError('Training class missing')
        target=min(before[0],max(round(cfg['full_balance_target']*len(yy)/cfg['train_rows']),max(before[c] for c in range(1,nclasses))))
        if cfg['synthetic_cap']!=.5 or cfg['synthetic_cap_denominator']!='per_class_original_before_resampling':raise ValueError('Changed synthetic cap')
        targets={c:min(target,n+int(np.floor(.5*n))) if n<target else target for c,n in before.items()}
        rus={c:targets[c] for c,n in before.items() if n>targets[c]};smote={c:targets[c] for c,n in before.items() if n<targets[c]}
        if rus:Z,yy=RandomUnderSampler(sampling_strategy=rus,random_state=cfg['seed']).fit_resample(Z,yy)
        if smote:Z,yy=SMOTE(sampling_strategy=smote,random_state=cfg['seed'],k_neighbors=min(5,min(before[c] for c in smote)-1)).fit_resample(Z,yy)
        after={int(k):int(n) for k,n in zip(*np.unique(yy,return_counts=True))};synthetic={c:max(0,after[c]-before[c]) for c in before}
        if any(synthetic[c]>int(np.floor(.5*before[c])) for c in before) or sum(synthetic.values())/len(yy)>.5:raise ValueError('Synthetic cap exceeded')
        vocab=prep.named_transformers_['tfidf'].vocabulary_ if 'tfidf' in prep.named_transformers_ else {}
        plans.append(dict(mode=mode,fit_row_ids=xt.index.tolist(),validation_row_ids=X.index[other].tolist(),before=before,after=after,synthetic=synthetic,synthetic_fraction=sum(synthetic.values())/len(yy),encoded_features=int(Z.shape[1]),vocabulary_size=len(vocab),vocabulary_fit_row_ids=xt.index.tolist() if vocab else [],nclasses=nclasses))
        modes[mode]=(prep,Z,yy)
    models={};records=[]
    for name in BASE_NAMES:
        prep,Z,yy=modes['linear' if name=='logistic_regression' else 'tree'];est=make_estimator(name,cfg['params'][name],cfg['seed'],nclasses)
        # DDoS weighting is inapplicable to binary Normal/SQL.
        sw=np.ones(len(yy)) if nclasses==2 else np.where(yy==1,cfg['params'][name]['ddos_loss_weight'],1.)
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning);est.fit(Z,yy,sample_weight=sw)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise ValueError('Convergence gate: '+view+'/'+name)
        np.testing.assert_array_equal(est.classes_,np.arange(nclasses));models[name]=Pipeline([('preprocessor',prep),('estimator',est)])
        records.append(dict(model=name,fit_seconds=time.perf_counter()-start))
    return models,plans,records

def predict(bundle,X,method):
    if method in BASE_NAMES:p=bundle['models'][method].predict_proba(X)
    else:
        ps=[np.asarray(bundle['models'][n].predict_proba(X),dtype=float) for n in BASE_NAMES]
        if method=='soft_voting':p=np.mean(ps,axis=0)
        elif method=='weighted_voting':p=sum(float(w)*q for w,q in zip(bundle['weights'],ps))
        elif method=='stacking' and 'meta' in bundle:p=bundle['meta'].predict_proba(np.hstack(ps))
        else:raise ValueError('Predictor unavailable')
    p=np.asarray(p,dtype=float)
    if p.shape!=(len(X),bundle['nclasses']) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Posterior schema mismatch')
    np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-6)
    return p
