"""Development-only WB experiment. Explicit columns, training guards, no import side effects."""
from pathlib import Path
import json, pickle, time, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import ta_hafid_engine as e
from e4 import DomainFeatures, columns, metrics


def view_data(raw,data,view):
    mode='noport25' if view.startswith('legacy25') else 'dstport29'
    x=DomainFeatures(mode).transform(raw);num,cat=columns(mode)
    if view.endswith('_destination'):
        from wb_behavior import PRIOR_DESTINATION
        names=list(PRIOR_DESTINATION);key='prior_destination'
    elif view.endswith('_symmetric'):
        from wb_behavior import PRIOR_SYMMETRIC
        names=list(PRIOR_SYMMETRIC);key='prior_symmetric'
    else:names=[];key=None
    if names:
        x=pd.concat([x,pd.DataFrame(data[key],index=raw.index,columns=names)],axis=1)
    return x,list(num)+names,list(cat)


def prepare(X,y,case,mode,numeric,categorical,cfg,validation_ids):
    e.guard_training(X,cfg,validation_ids,'training')
    if case not in ('original','balanced') or len(X)!=len(y):raise ValueError('Invalid training')
    steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if mode=='linear':steps.append(('scaler',StandardScaler(with_mean=False)))
    prep=ColumnTransformer([('numeric',Pipeline(steps),numeric),('categorical',Pipeline([
        ('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__')),
        ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),categorical)],
        sparse_threshold=1.,remainder='drop',verbose_feature_names_out=True)
    Z=prep.fit_transform(X);yy=np.asarray(y).copy()
    before={int(c):int(v) for c,v in zip(*np.unique(yy,return_counts=True))}
    if set(before)!=set(range(5)):raise ValueError('Five classes required')
    target=None;targets=dict(before)
    if case=='balanced':
        target=round(cfg['full_balance_target']*len(y)/cfg['train_rows'])
        target=min(before[0],max(target,max(before[c] for c in range(1,5))))
        cap=cfg['synthetic_cap']
        if not 0<=cap<=.5 or cfg['synthetic_cap_denominator']!='per_class_original_before_resampling':raise ValueError('Invalid cap')
        targets={c:min(target,n+int(np.floor(cap*n))) if n<target else target for c,n in before.items()}
        rus={c:targets[c] for c,n in before.items() if n>targets[c]}
        smote={c:targets[c] for c,n in before.items() if n<targets[c]}
        if rus:Z,yy=RandomUnderSampler(sampling_strategy=rus,random_state=cfg['seed']).fit_resample(Z,yy)
        if smote:
            k=min(5,min(before[c] for c in smote)-1)
            if k<1:raise ValueError('SMOTE class too small')
            Z,yy=SMOTE(sampling_strategy=smote,random_state=cfg['seed'],k_neighbors=k).fit_resample(Z,yy)
    after={int(c):int(v) for c,v in zip(*np.unique(yy,return_counts=True))}
    synthetic={c:max(0,after[c]-before[c]) for c in before}
    if any(synthetic[c]>np.floor(cfg['synthetic_cap']*before[c]) for c in before):raise ValueError('Cap exceeded')
    fraction=sum(synthetic.values())/sum(after.values())
    if fraction>.5:raise ValueError('Total cap exceeded')
    plan=dict(case=case,mode=mode,before=before,after=after,synthetic=synthetic,
        requested_target=target,actual_targets=targets,encoded_features=int(Z.shape[1]),
        synthetic_per_original_class={c:synthetic[c]/before[c] for c in before},
        synthetic_fraction_of_final_training=fraction,denominator='original_per_class_before_RUS',
        preprocessing_fit_row_ids=X.index.to_numpy(),validation_ids=list(validation_ids))
    return prep,Z,yy,plan


def fit_experts(X,y,fit,other,case,num,cat,params,cfg):
    modes={};plans=[]
    for mode in ['tree','linear']:
        prep,Z,yy,plan=prepare(X.iloc[fit],y[fit],case,mode,num,cat,cfg,X.index[other])
        modes[mode]=(prep,Z,yy);plans.append(plan)
    models={};records=[]
    for name in e.NAMES:
        prep,Z,yy=modes['linear' if name=='logistic_regression' else 'tree']
        est=e.estimator(name,params[name],cfg['seed'])
        if name=='logistic_regression':est.set_params(max_iter=cfg['lr_max_iter'],tol=cfg['lr_tol'])
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            est.fit(Z,yy,sample_weight=np.where(yy==1,params[name]['ddos_loss_weight'],1.))
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise ValueError('Convergence gate: '+name)
        if not np.array_equal(est.classes_,np.arange(5)):raise ValueError('Class order')
        models[name]=Pipeline([('preprocessor',prep),('estimator',est)])
        records.append(dict(model=name,fit_seconds=time.perf_counter()-t,params=est.get_params()))
    return models,plans,records


def cube(models,X,shared=True):
    if shared:
        Z=models['decision_tree']['preprocessor'].transform(X)
        L=models['logistic_regression']['preprocessor'].transform(X)
        ps=[models[n]['estimator'].predict_proba(L if n=='logistic_regression' else Z) for n in e.NAMES]
    else:ps=[models[n].predict_proba(X) for n in e.NAMES]
    p=np.stack([np.asarray(q,dtype=np.float64) for q in ps],axis=1)
    if p.shape!=(len(X),5,5) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(2),1,atol=1e-6):raise ValueError('Invalid posterior cube')
    return p


def mixture(p,w):
    return sum(float(w[j])*np.asarray(p[:,j],dtype=np.float64) for j in range(5))


def learn_weights(y,p,latency,size,cfg):
    quality=np.array([metrics(y,p[:,i].argmax(1))['macro_f1'] for i in range(5)])
    quality=quality/max(quality.max(),1e-15)
    latency=np.maximum(np.asarray(latency),1e-12);size=np.maximum(np.asarray(size),1)
    efficiency=(latency.min()/latency+size.min()/size)/2
    candidates=[]
    for a in cfg['weight_alphas']:
        raw=a*quality+(1-a)*efficiency;w=raw/raw.sum()
        w=(1-cfg['weight_shrinkage'])/5+cfg['weight_shrinkage']*w
        w=np.maximum(w,cfg['min_base_weight']);w/=w.sum()
        m=metrics(y,mixture(p,w).argmax(1));score=.5*(m['macro_f1']+m['ddos_f2'])
        candidates.append(dict(alpha=a,weights=w,score=score,metrics=m))
    best=max(candidates,key=lambda z:(z['score'],-abs(z['alpha']-.6)))
    return best,candidates


def profile_components(models,training_sample,calibration_sample):
    latency=[];sizes=[]
    for name in e.NAMES:
        model=models[name];model.predict_proba(training_sample);times=[]
        for _ in range(3):
            t=time.perf_counter();model.predict_proba(training_sample);times.append(time.perf_counter()-t)
        latency.append(float(np.median(times)/len(training_sample)))
        sizes.append(len(pickle.dumps(model,protocol=5)))
    # Scalar cost controls match scheduler's per-decision optimization objective.
    prep_cost={'tree':[],'linear':[]};expert_cost=[[] for _ in range(5)]
    for row in range(min(24,len(calibration_sample))):
        frame=calibration_sample.iloc[[row]];matrices={}
        for mode,name in [('tree','decision_tree'),('linear','logistic_regression')]:
            t=time.perf_counter();matrices[mode]=models[name]['preprocessor'].transform(frame)
            prep_cost[mode].append(time.perf_counter()-t)
        for j,name in enumerate(e.NAMES):
            z=matrices['linear' if name=='logistic_regression' else 'tree']
            t=time.perf_counter();models[name]['estimator'].predict_proba(z);expert_cost[j].append(time.perf_counter()-t)
    return dict(training_pipeline_latency_seconds=latency,serialized_pipeline_bytes=sizes,
        expert_costs=[float(np.median(x)) for x in expert_cost],
        preprocessing_costs={k:float(np.median(v)) for k,v in prep_cost.items()},
        preprocessing_groups=['linear' if n=='logistic_regression' else 'tree' for n in e.NAMES])


def bijective_ports(raw,offset=32768):
    out=raw.copy()
    for name in e.PORT_FEATURES:
        p=pd.to_numeric(raw[name],errors='coerce');valid=p.between(1,65535)&p.eq(np.floor(p))
        out[name]=out[name].astype(object);out.loc[valid,name]=1+((p[valid]-1+offset)%65535)
    pd.testing.assert_frame_equal(DomainFeatures('noport25').transform(raw),DomainFeatures('noport25').transform(out))
    return out


class WBPredictor:
    """Inference over supplied domain features; no feature-extraction cost is hidden in this class."""
    def __init__(self,models,weights,order=None,policy='shared'):
        self.models=models;self.weights=np.asarray(weights);self.order=list(range(5)) if order is None else list(order);self.policy=policy
    def predict(self,X):
        if self.policy in ['shared','independent']:
            return mixture(cube(self.models,X,self.policy=='shared'),self.weights).argmax(1)
        from wb_scheduler import predict_lazy
        transformed={}
        def callback(j):
            name=e.NAMES[j];mode='linear' if name=='logistic_regression' else 'tree'
            def call(active):
                # Transform the current batch once per preprocessing family; estimator only unresolved rows.
                if mode not in transformed:transformed[mode]=self.models[name]['preprocessor'].transform(X)
                return np.asarray(self.models[name]['estimator'].predict_proba(transformed[mode][active]),dtype=np.float64)
            return call
        result=predict_lazy(len(X),5,self.weights,self.order,[callback(j) for j in range(5)])
        return result.labels
