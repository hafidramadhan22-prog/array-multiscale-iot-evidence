"""Salinan kerja V4.3, diselaraskan dengan arahan dospem 13 Oktober 2026.
Rekomendasi dari holdout; angka operasional adalah desain penelitian eksplisit.
Historical holdout: comparative/exploratory, bukan independent final test.
"""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
import sys, json, gc, time, hashlib, platform, subprocess, threading, shutil, warnings
from pathlib import Path
from itertools import combinations
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib, psutil
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore',message='X does not have valid feature names')

TARGET = 'Attack_type'
DERIVED_BINARY_LABEL = 'Attack_label'
CLASSES = ['Normal','DDoS_TCP','Port_Scanning','SQL_injection','Ransomware']
NAMES = ['decision_tree','random_forest','xgboost','lightgbm','logistic_regression']
METHODS = ['soft_voting','weighted_voting','stacking']
CASES = ['original','balanced']
VARIANTS = ['baseline_search','ddos_tuned']
DEFAULT_CONFIG = dict(seed=42, outer_splits=5, inner_splits=2, feature_mode='bands28', permutation_repeats=3,
    dataset_sha256='53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a',
    raw_shape=[157800,63], five_rows=65855, train_rows=52684, holdout_rows=13171,
    test_size=0.20, full_balance_target=10000,
    tuning_macro_f1_weight=0.5, tuning_ddos_f2_weight=0.5,
    max_macro_f1_drop_from_legacy=0.01,
    final_tradeoff_performance_weight=0.6, weight_alphas=[0.5,0.6,0.7],
    weight_shrinkage=0.5, min_base_weight=0.05,
    profile_rows=1000, memory_runs=5, memory_interval_seconds=0.002,
    latency_blocks=3, latency_repeats=15,
    shap_background_rows=20, shap_global_rows=50, shap_nsamples=256,
    shap_local_max_rows=16,
    historical_holdout_previously_observed=True,
    historical_holdout_note='V2/V4.1/V4.2 holdout already observed. V4.3 informed by previous error analysis and TRAINING-only feature audit. Reused holdout is comparative/exploratory, not a new untouched test.',
    smoke_test=False)

def jsonable(x):
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    return x

def atomic_replace(source,target):
    """Windows dapat mengunci file sementara (AV/indexer); retry, tetap atomik."""
    delays=[0.01,0.02,0.04,0.08,0.16,0.32,0.5,0.5,0.5]
    for attempt in range(len(delays)+1):
        try:
            Path(source).replace(target);return
        except PermissionError:
            if attempt==len(delays):raise
            time.sleep(delays[attempt])


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(jsonable(value),indent=2),encoding='utf-8');atomic_replace(tmp,path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_dump(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp');joblib.dump(value,tmp,compress=1);atomic_replace(tmp,path)

def sample_positions(y,n,seed=42,equal=False):
    y=np.asarray(y); rng=np.random.default_rng(seed); labels,counts=np.unique(y,return_counts=True)
    n=min(int(n),len(y)); quota=np.zeros(len(labels),dtype=int)
    if n>=len(labels): quota[:]=1
    weights=np.ones(len(labels)) if equal else counts.astype(float)
    while quota.sum()<n:
        score=weights/(quota+1);score[quota>=counts]=-1
        quota[np.argmax(score)]+=1
    pos=np.concatenate([rng.choice(np.flatnonzero(y==c),q,replace=False) for c,q in zip(labels,quota)])
    rng.shuffle(pos);return pos

def environment():
    from importlib.metadata import distributions
    packages={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','imbalanced-learn','xgboost','lightgbm','shap','joblib','psutil']}
    return dict(python=sys.version,platform=platform.platform(),logical_cpus=os.cpu_count(),
        cpu=platform.processor(),hostname=platform.node(),physical_ram_mib=psutil.virtual_memory().total/2**20,
        packages=packages,all_packages={d.metadata['Name']:d.version for d in distributions() if d.metadata['Name']},thread_limit=1)

def metrics(y,p):
    p=np.asarray(p);pred=p.argmax(1)
    assert p.shape==(len(y),5) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    out=dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',zero_division=0),
        recall_macro=recall_score(y,pred,average='macro',zero_division=0),
        precision_macro=precision_score(y,pred,average='macro',zero_division=0),
        ddos_recall=recall_score(y,pred,labels=[1],average='macro',zero_division=0),
        ddos_precision=precision_score(y,pred,labels=[1],average='macro',zero_division=0),
        ddos_f2=fbeta_score(y,pred,beta=2,labels=[1],average='macro',zero_division=0),
        port_scanning_recall=recall_score(y,pred,labels=[2],average='macro',zero_division=0))
    out['auc_ovr_macro']=roc_auc_score(y,p,multi_class='ovr',average='macro',labels=np.arange(5))
    return {k:float(v) for k,v in out.items()}

def tuning_score(m,cfg):
    return cfg['tuning_macro_f1_weight']*m['macro_f1']+cfg['tuning_ddos_f2_weight']*m['ddos_f2']

def predict_bundle(bundle,X):
    ps=[np.asarray(bundle['models'][n].predict_proba(X)) for n in NAMES]
    if bundle['method']=='soft_voting':return np.mean(ps,axis=0)
    if bundle['method']=='weighted_voting':return sum(bundle['weights'][n]*p for n,p in zip(NAMES,ps))
    return bundle['meta'].predict_proba(np.hstack(ps))


NUMERIC_KEEP = [
    "http.response","tcp.connection.fin","tcp.connection.rst","tcp.connection.syn",
    "tcp.connection.synack","tcp.flags.ack","tcp.len","udp.time_delta","mqtt.len",
]

CATEGORICAL_KEEP = [
    "arp.opcode","arp.hw.size","http.request.method","http.request.version",
    "tcp.flags","dns.qry.qu","dns.retransmission","mqtt.conflag.cleansess",
    "mqtt.conflags","mqtt.hdrflags","mqtt.msgtype","mqtt.proto_len",
    "mqtt.protoname","mqtt.topic_len","mqtt.ver",
]

PORT_FEATURES=["tcp.srcport","tcp.dstport","udp.port"]

FORBIDDEN_RAW={
    TARGET,DERIVED_BINARY_LABEL,"frame.time","ip.src_host","ip.dst_host",
    "arp.dst.proto_ipv4","arp.src.proto_ipv4","dns.qry.name",
    "http.request.full_uri","http.request.uri.query","http.file_data","http.referer",
    "mqtt.msg","mqtt.topic","tcp.ack","tcp.ack_raw","tcp.checksum","tcp.options",
    "tcp.payload","tcp.seq","udp.stream",
}

def normalize_absent(value):
    text=str(value).strip()
    return "__ABSENT__" if text in {"","0","0.0","nan","None","<NA>"} else text

def port_band(value):
    try: number=float(str(value).strip())
    except Exception: return "INVALID_TEXT"
    if not np.isfinite(number): return "MISSING_OR_NONFINITE"
    port=int(number)
    if port==0: return "INACTIVE_0"
    if 1<=port<=1023: return "WELL_KNOWN_1_1023"
    if 1024<=port<=49151: return "REGISTERED_1024_49151"
    if 49152<=port<=65535: return "DYNAMIC_49152_65535"
    return "OUT_OF_RANGE"

def deterministic_domain_transform(frame: pd.DataFrame) -> pd.DataFrame:
    out={}
    for c in NUMERIC_KEEP: out[c]=pd.to_numeric(frame[c],errors="coerce")
    for c in CATEGORICAL_KEEP: out[c]=frame[c].map(normalize_absent).astype(str)
    for c in PORT_FEATURES: out[f"{c}__band"]=frame[c].map(port_band).astype(str)
    dns_raw=frame["dns.qry.name.len"].astype(str)
    out["dns.qry.name.len__numeric"]=pd.to_numeric(dns_raw,errors="coerce")
    if CURRENT_FEATURE_MODE=='dstport29':
        ports=pd.to_numeric(frame['tcp.dstport'],errors='coerce')
        valid=ports.between(0,65535)&ports.eq(np.floor(ports))
        out['tcp.dstport__numeric']=ports.where(valid,np.nan)
    return pd.DataFrame(out,index=frame.index)

def build_preprocessor(mode="tree"):
    numeric_steps=[("imputer",SimpleImputer(strategy="median"))]
    if mode=="linear": numeric_steps.append(("scaler",StandardScaler(with_mean=False)))
    cat_steps=[
        ("imputer",SimpleImputer(strategy="constant",fill_value="__MISSING__")),
        ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=True)),
    ]
    return ColumnTransformer(
        [("numeric",Pipeline(numeric_steps),NUMERIC_FEATURES),
         ("categorical",Pipeline(cat_steps),CATEGORICAL_FEATURES)],
        remainder="drop",sparse_threshold=1.0,verbose_feature_names_out=True,
    )

LEGACY_CANDIDATES={
    "decision_tree":[
        dict(max_depth=10,min_samples_leaf=5,min_samples_split=10,criterion="gini"),
        dict(max_depth=12,min_samples_leaf=5,min_samples_split=20,criterion="log_loss"),
    ],
    "random_forest":[
        dict(n_estimators=75,max_depth=10,min_samples_leaf=2,min_samples_split=4,max_features="sqrt"),
        dict(n_estimators=100,max_depth=12,min_samples_leaf=2,min_samples_split=4,max_features="sqrt"),
    ],
    "xgboost":[
        dict(n_estimators=75,max_depth=4,learning_rate=0.10,subsample=0.90,colsample_bytree=0.90,reg_alpha=0.0,reg_lambda=1.0),
        dict(n_estimators=100,max_depth=5,learning_rate=0.05,subsample=0.90,colsample_bytree=0.90,reg_alpha=0.05,reg_lambda=1.0),
    ],
    "lightgbm":[
        dict(n_estimators=75,max_depth=6,num_leaves=31,learning_rate=0.10,min_child_samples=20,reg_alpha=0.0,reg_lambda=1.0),
        dict(n_estimators=100,max_depth=6,num_leaves=31,learning_rate=0.05,min_child_samples=20,reg_alpha=0.05,reg_lambda=1.0),
    ],
    "logistic_regression":[dict(C=0.5),dict(C=1.0)],
}

CURRENT_FEATURE_MODE='bands28'
NUMERIC_FEATURES=NUMERIC_KEEP+['dns.qry.name.len__numeric']
CATEGORICAL_FEATURES=CATEGORICAL_KEEP+[f'{c}__band' for c in PORT_FEATURES]
FEATURE_COLUMNS=NUMERIC_FEATURES+CATEGORICAL_FEATURES

def configure_features(mode):
    global CURRENT_FEATURE_MODE,NUMERIC_FEATURES,FEATURE_COLUMNS
    assert mode in ['bands28','dstport29']
    CURRENT_FEATURE_MODE=mode
    NUMERIC_FEATURES=NUMERIC_KEEP+['dns.qry.name.len__numeric']+(['tcp.dstport__numeric'] if mode=='dstport29' else [])
    FEATURE_COLUMNS=NUMERIC_FEATURES+CATEGORICAL_FEATURES


def candidates(name,smoke=False):
    legacy=[dict(p,ddos_loss_weight=1.0) for p in LEGACY_CANDIDATES[name]]
    if smoke:return legacy[:1]+[dict(legacy[0],ddos_loss_weight=2.0)]
    best=dict(legacy[1]);out=legacy+[dict(best,ddos_loss_weight=w) for w in [1.1,1.25,1.5]]
    deeper=dict(best)
    if name=='decision_tree':deeper.update(max_depth=16,min_samples_leaf=2,min_samples_split=4)
    elif name=='random_forest':deeper.update(n_estimators=150,max_depth=16,min_samples_leaf=1)
    elif name=='xgboost':deeper.update(n_estimators=150,max_depth=6,learning_rate=0.05)
    elif name=='lightgbm':deeper.update(n_estimators=150,max_depth=8,num_leaves=63,min_child_samples=10)
    else:deeper.update(C=5.0)
    return out+[dict(deeper,ddos_loss_weight=w) for w in [1.0,1.1,1.25]]

def estimator(name,params,seed):
    p={k:v for k,v in params.items() if k!='ddos_loss_weight'}
    if name=='decision_tree':return DecisionTreeClassifier(random_state=seed,**p)
    if name=='random_forest':return RandomForestClassifier(random_state=seed,n_jobs=1,**p)
    if name=='xgboost':return XGBClassifier(objective='multi:softprob',num_class=5,eval_metric='mlogloss',tree_method='hist',random_state=seed,n_jobs=1,verbosity=0,**p)
    if name=='lightgbm':return LGBMClassifier(objective='multiclass',num_class=5,random_state=seed,n_jobs=1,verbosity=-1,**p)
    return LogisticRegression(solver='lbfgs',max_iter=1000,tol=1e-3,random_state=seed,**p)

def guard_training(X, cfg, validation_ids=(), partition=None):
    if partition != 'training':
        raise ValueError('Preprocessing fit/resampling hanya boleh pada training.')
    ids=set(X.index)
    if len(ids)!=len(X) or not ids or not ids.issubset(set(cfg['_training_ids'])):
        raise ValueError('Training indices harus unik dan bagian development 80%.')
    if ids.intersection(cfg['_holdout_ids']) or ids.intersection(validation_ids):
        raise ValueError('Kebocoran validation/holdout ke training.')


def prepared(X,y,case,mode,cfg,*,partition=None,validation_ids=()):
    guard_training(X,cfg,validation_ids,partition)
    if case not in CASES or len(X)!=len(y):raise ValueError('Training case/length invalid')
    prep=build_preprocessor(mode);Z=prep.fit_transform(X); yy=np.array(y,copy=True)
    before={int(c):int(v) for c,v in zip(*np.unique(yy,return_counts=True))};after=dict(before)
    if set(before)!=set(range(5)):raise ValueError('Training partition harus berisi lima kelas')
    target=None;targets=dict(before)
    if case=='balanced':
        target=round(cfg['full_balance_target']*len(y)/cfg['train_rows'])
        target=min(before[0],max(target,max(before[c] for c in range(1,5))))
        cap=cfg['synthetic_cap']
        if not 0<=cap<=0.5 or cfg['synthetic_cap_denominator']!='per_class_original_before_resampling':
            raise ValueError('Cap/penyebut sintetis tidak sesuai protokol kandidat')
        targets={c:min(target,n+int(np.floor(cap*n))) if n<target else target for c,n in before.items()}
        rus={c:targets[c] for c,n in before.items() if n>targets[c]}
        smote={c:targets[c] for c,n in before.items() if n<targets[c]}
        if rus:Z,yy=RandomUnderSampler(sampling_strategy=rus,random_state=cfg['seed']).fit_resample(Z,yy)
        if smote:
            k=min(5,min(before[c] for c in smote)-1)
            if k<1:raise ValueError('SMOTE perlu minimal dua sampel asli per kelas yang ditambah')
            Z,yy=SMOTE(sampling_strategy=smote,random_state=cfg['seed'],k_neighbors=k).fit_resample(Z,yy)
        after={int(c):int(v) for c,v in zip(*np.unique(yy,return_counts=True))}
    synthetic={c:max(0,after[c]-before[c]) for c in before}
    if any(synthetic[c]>np.floor(cfg['synthetic_cap']*before[c]) for c in before):
        raise ValueError('Batas sintetis terlampaui')
    fraction=sum(synthetic.values())/sum(after.values())
    if fraction>0.5:raise ValueError('Sintetis melebihi 50% total training akhir')
    return prep,Z,yy,dict(before=before,after=after,synthetic=synthetic,
        encoded_features=int(Z.shape[1]),synthetic_fraction_of_final_training=fraction,
        synthetic_per_original_class={c:synthetic[c]/before[c] for c in before},
        synthetic_denominator='original_per_class_before_RUS',requested_target=target,
        actual_targets=targets,exactly_balanced=len(set(after.values()))==1,
        partition='training',mode=mode,case=case)

def fit_est(name,params,Z,y,cfg):
    est=estimator(name,params,cfg['seed'])
    weight=np.where(np.asarray(y)==1,float(params['ddos_loss_weight']),1.0)
    with threadpool_limits(limits=1):est.fit(Z,y,sample_weight=weight)
    return est

def fit_case(X,y,Xval,case,cfg,tag,outdir,with_baseline=True):
    """No holdout argument. Inner folds select params; outer Xval only evaluates."""
    validation_ids=() if Xval is None else Xval.index
    guard_training(X,cfg,validation_ids,'training')
    if Xval is not None and set(Xval.index).intersection(cfg['_holdout_ids']):
        raise ValueError('fit_case tidak boleh membaca holdout')
    cv=StratifiedKFold(cfg['inner_splits'],shuffle=True,random_state=cfg['seed']+17)
    splits=list(cv.split(X,y)); caches={};full={};plans=[]
    split_arrays={}
    for i,(tr,va) in enumerate(splits):
        split_arrays[f'inner{i+1}_train_row_ids']=X.index.to_numpy()[tr]
        split_arrays[f'inner{i+1}_validation_row_ids']=X.index.to_numpy()[va]
    split_arrays['fit_train_row_ids']=X.index.to_numpy()
    split_arrays['outer_validation_row_ids']=np.asarray(validation_ids)
    np.savez_compressed(Path(outdir)/f'indices_{tag}_{case}.npz',**split_arrays)
    for mode in ['tree','linear']:
        fs=[]
        for i,(tr,va) in enumerate(splits):
            prep,Z,yy,plan=prepared(X.iloc[tr],y[tr],case,mode,cfg,
                partition='training',validation_ids=X.index[va].union(validation_ids))
            plans.append(dict(stage=tag,inner_fold=i+1,**plan))
            fs.append((tr,va,Z,yy,prep.transform(X.iloc[va])))
        caches[mode]=fs
        full[mode]=prepared(X,y,case,mode,cfg,partition='training',validation_ids=validation_ids)
        plans.append(dict(stage=tag,inner_fold=None,**full[mode][3]))
    variants=VARIANTS if with_baseline else ['ddos_tuned']
    fitted={v:{} for v in variants};oo={v:{} for v in variants};vp={v:{} for v in variants}
    profiles={v:{} for v in variants};selected={v:{} for v in variants};rows=[]
    profilepos=sample_positions(y,min(cfg['profile_rows'],len(y)),cfg['seed'])
    for name in NAMES:
        print(f'  {tag}/{case}: tuning {name}',flush=True)
        mode='linear' if name=='logistic_regression' else 'tree';cand=candidates(name,cfg['smoke_test'])
        cp=[];cm=[]
        for ci,pa in enumerate(cand):
            if not cfg['smoke_test']:
                print(f'    {tag}/{case}/{name}: candidate {ci+1}/{len(cand)}',flush=True)
            write_json(Path(outdir)/'training_progress.json',dict(stage=tag,case=case,model=name,
                candidate=ci+1,total_candidates=len(cand),status='FITTING_INNER_FOLDS'))
            p=np.zeros((len(y),5));seen=np.zeros(len(y),int);fold_ms=[]
            for tr,va,Z,yy,Zv in caches[mode]:
                est=fit_est(name,pa,Z,yy,cfg)
                p[va]=est.predict_proba(Zv);seen[va]+=1
                fold_ms.append(metrics(y[va],p[va]));del est
            assert np.all(seen==1)
            m={k:float(np.mean([r[k] for r in fold_ms])) for k in fold_ms[0]}
            m['tuning_score']=tuning_score(m,cfg);cp.append(p);cm.append(m)
            rows.append(dict(stage=tag,case=case,model=name,candidate=ci,legacy=ci<(1 if cfg['smoke_test'] else 2),params=json.dumps(pa,sort_keys=True),**m))
            write_json(Path(outdir)/'training_progress.json',dict(stage=tag,case=case,model=name,
                candidate=ci+1,total_candidates=len(cand),status='INNER_SCORED',metrics=m))
        legacy_count=1 if cfg['smoke_test'] else 2
        bi=max(range(legacy_count),key=lambda i:cm[i]['macro_f1'])
        eligible=[i for i,m in enumerate(cm) if m['macro_f1']>=cm[bi]['macro_f1']-cfg['max_macro_f1_drop_from_legacy']]
        ti=max(eligible,key=lambda i:(cm[i]['tuning_score'],cm[i]['macro_f1'],-i))
        choices={'baseline_search':bi,'ddos_tuned':ti};fit_cache={}
        prep,Z,yy,_=full[mode]
        for v in variants:
            ci=choices[v];pa=cand[ci]
            if ci not in fit_cache:
                est=fit_est(name,pa,Z,yy,cfg);pipe=Pipeline([('preprocessor',prep),('estimator',est)])
                pv=pipe.predict_proba(Xval) if Xval is not None else None
                xp=X.iloc[profilepos];pipe.predict_proba(xp)
                timings=[]
                for _ in range(3):
                    st=time.perf_counter();pipe.predict_proba(xp);timings.append(time.perf_counter()-st)
                size=len(__import__('pickle').dumps(pipe,protocol=5))/2**20
                fit_cache[ci]=(pipe,pv,dict(latency_ms_per_sample=1000*np.mean(timings)/len(xp),artifact_mib=size))
            pipe,pv,prof=fit_cache[ci];fitted[v][name]=pipe;oo[v][name]=cp[ci];vp[v][name]=pv
            profiles[v][name]=prof;selected[v][name]=dict(candidate=ci,params=pa,inner_metrics=cm[ci])
        gc.collect()
    bundles={};preds={};base_rows=[]
    for v in variants:
        quality=np.array([metrics(y,oo[v][n])['macro_f1'] for n in NAMES]);quality=quality/quality.max()
        lat=np.array([profiles[v][n]['latency_ms_per_sample'] for n in NAMES]);sz=np.array([profiles[v][n]['artifact_mib'] for n in NAMES])
        efficiency=(lat.min()/lat+sz.min()/sz)/2
        alpha_results=[]
        for a in cfg['weight_alphas']:
            raw=a*quality+(1-a)*efficiency;w=raw/raw.sum()
            w=(1-cfg['weight_shrinkage'])/5+cfg['weight_shrinkage']*w
            w=np.maximum(w,cfg['min_base_weight']);w=w/w.sum()
            mi=metrics(y,sum(wi*oo[v][n] for wi,n in zip(w,NAMES)))
            sc=mi['macro_f1'] if v=='baseline_search' else tuning_score(mi,cfg)
            alpha_results.append((sc,-abs(a-.6),a,w))
        _,_,a,w=max(alpha_results,key=lambda z:z[:2]);weights=dict(zip(NAMES,w))
        meta=LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,random_state=cfg['seed'])
        with threadpool_limits(limits=1):meta.fit(np.hstack([oo[v][n] for n in NAMES]),y)
        for method in METHODS:
            b=dict(method=method,case=case,variant=v,feature_mode=cfg['feature_mode'],models=fitted[v],model_order=NAMES,
                selected_params=selected[v],weight_alpha=a,feature_columns=list(X.columns),seed=cfg['seed'])
            if method=='weighted_voting':b['weights']=weights
            if method=='stacking':b['meta']=meta
            key=f'{v}__{case}__{method}';bundles[key]=b
            if Xval is not None:
                ps=[vp[v][n] for n in NAMES]
                preds[key]=np.mean(ps,0) if method=='soft_voting' else (sum(weights[n]*vp[v][n] for n in NAMES) if method=='weighted_voting' else meta.predict_proba(np.hstack(ps)))
        for n in NAMES:
            if Xval is not None:base_rows.append(dict(variant=v,case=case,model=n,probabilities=vp[v][n]))
    # Shuffle only the newly added numeric port, within existing port bands.
    # This is a TRAINING outer-validation sensitivity diagnostic, not a new test set.
    if Xval is not None and cfg['feature_mode']=='dstport29':
        diagnostic=[]
        for rep in range(cfg['permutation_repeats']):
            shuffled=Xval.copy();rng=np.random.default_rng(cfg['seed']+100+rep)
            for band,indices in Xval.groupby('tcp.dstport__band',sort=True).indices.items():
                col=shuffled.columns.get_loc('tcp.dstport__numeric')
                shuffled.iloc[indices,col]=rng.permutation(Xval.iloc[indices,col].to_numpy())
            for key,b in bundles.items():
                if b['variant']!='ddos_tuned':continue
                # Outer labels are available for evaluation only, never parameter selection.
                probability=predict_bundle(b,shuffled)
                diagnostic.append(dict(stage=tag,case=case,method=b['method'],repeat=rep,probabilities=probability))
        # Labels are passed separately in cross_validate, keeping fit_case label-blind to outer validation.
        atomic_dump(diagnostic,Path(outdir)/f'port_permutation_{tag}_{case}.joblib')
    pd.DataFrame(rows).to_csv(Path(outdir)/f'tuning_{tag}_{case}.csv',index=False)
    write_json(Path(outdir)/f'selection_{tag}_{case}.json',dict(selected=selected,plans=plans))
    return bundles,preds,base_rows

class Experiment:
    def __init__(self,dataset,output,config=None):
        self.cfg=dict(DEFAULT_CONFIG);self.cfg.update(config or {})
        configure_features(self.cfg['feature_mode'])
        self.dataset=Path(dataset);self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        for d in ['models','tables','manifests','cv_cache','profiling','shap','figures','source']: (self.out/d).mkdir(exist_ok=True)
        code=Path(__file__).read_bytes()+(Path(__file__).parent/'recommendation.py').read_bytes()
        self.signature=hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()+code).hexdigest()
        pp=self.out/'manifests/protocol.json'
        if pp.exists():
            assert json.loads(pp.read_text())['signature']==self.signature,'Output contains a different protocol/source. Use a new output folder.'
        else:write_json(pp,dict(signature=self.signature,config=self.cfg,status=self.cfg['protocol_status']))
        write_json(self.out/'manifests/environment.json',environment())
        shutil.copy2(__file__,self.out/'source/ta_hafid_engine.py')

    def prepare(self):
        cfg=self.cfg;actual=sha256(self.dataset)
        assert actual==cfg['dataset_sha256'],'Dataset SHA-256 mismatch; do not bypass.'
        raw=pd.read_csv(self.dataset,low_memory=False)
        assert list(raw.shape)==cfg['raw_shape']
        raw['__row_id__']=np.arange(len(raw));df=raw[raw[TARGET].isin(CLASSES)].copy()
        assert len(df)==cfg['five_rows'] and df.drop(columns='__row_id__').duplicated().sum()==0
        X=deterministic_domain_transform(df);y=df[TARGET].map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
        rid=df['__row_id__'].to_numpy();tr,ho=train_test_split(np.arange(len(y)),test_size=cfg['test_size'],stratify=y,random_state=cfg['seed'])
        tr=np.sort(tr);ho=np.sort(ho)
        if cfg['smoke_test']:
            # Smoke mempertahankan ketimpangan training agar RUS+SMOTE ikut diuji.
            tr=tr[sample_positions(y[tr],250,cfg['seed'],False)];ho=ho[sample_positions(y[ho],100,cfg['seed'],True)]
            cfg['train_rows']=len(tr);cfg['full_balance_target']=max(10,len(tr)//5)
        else:assert (len(tr),len(ho))==(cfg['train_rows'],cfg['holdout_rows'])
        self.X=X.iloc[tr].copy();self.y=y[tr];self.ids=rid[tr]
        self._Xh=X.iloc[ho].copy();self._yh=y[ho];self._idh=rid[ho]
        self.X.index=self.ids;self._Xh.index=self._idh
        cfg['_training_ids']=self.ids;cfg['_holdout_ids']=self._idh
        write_json(self.out/'manifests/effective_config.json',{k:v for k,v in cfg.items() if not k.startswith('_')})
        assert not set(self.ids).intersection(self._idh)
        assert not FORBIDDEN_RAW.intersection(self.X.columns)
        self.splits=list(StratifiedKFold(cfg['outer_splits'],shuffle=True,random_state=cfg['seed']).split(self.X,self.y))
        assignment=np.full(len(tr),-1)
        for f,(_,va) in enumerate(self.splits):assignment[va]=f+1
        pd.DataFrame(dict(original_row_id=self.ids,outer_validation_fold=assignment,label=self.y)).to_csv(self.out/'tables/training_fold_assignment.csv',index=False)
        pd.DataFrame(dict(original_row_id=self._idh)).to_csv(self.out/'tables/holdout_row_ids.csv',index=False)
        self.data_hash=actual
        write_json(self.out/'manifests/split.json',dict(dataset_sha256=actual,train_rows=len(tr),holdout_rows=len(ho),holdout_sampling='NONE',inner_cv=cfg['inner_splits'],outer_cv=cfg['outer_splits'],historical_holdout_previously_observed=True,holdout_claim=cfg['historical_holdout_note'],smoke_test=cfg['smoke_test']))
        # Conflicting feature patterns are examined only in training.
        sig=pd.util.hash_pandas_object(self.X,index=False)
        counts=pd.crosstab(sig,self.y);conflict=counts[counts.gt(0).sum(axis=1)>1]
        conflict.to_csv(self.out/'tables/training_conflicting_feature_patterns.csv')
        feature_info=dict(feature_mode=cfg['feature_mode'],input_features=len(self.X.columns),feature_columns=list(self.X.columns),conflicting_training_patterns=len(conflict),rows_in_conflicting_patterns=int(conflict.sum().sum()),diagnostic_source='TRAINING ONLY',raw_destination_port_added=cfg['feature_mode']=='dstport29')
        if cfg['feature_mode']=='dstport29':
            feature_info['numeric_port_valid_rows']=int(self.X['tcp.dstport__numeric'].notna().sum())
            feature_info['numeric_port_distinct_training_values']=int(self.X['tcp.dstport__numeric'].nunique())
        write_json(self.out/'manifests/features.json',feature_info)
        print(f'Prepared {len(self.X)} training, {len(self._Xh)} holdout; conflicting training feature patterns: {len(conflict)}',flush=True)
        return self

    def cross_validate(self):
        allrows=[];baserows=[];permutation_rows=[];oof={};seen=np.zeros(len(self.X),int)
        for f,(tr,va) in enumerate(self.splits):
            seen[va]+=1
            for case in CASES:
                path=self.out/f'cv_cache/fold{f+1}_{case}.joblib'
                if path.exists():r=joblib.load(path);assert r['signature']==self.signature
                else:
                    print(f'OUTER {f+1}/{len(self.splits)}, {case}',flush=True)
                    bundles,ps,bs=fit_case(self.X.iloc[tr],self.y[tr],self.X.iloc[va],case,self.cfg,f'outer{f+1}',self.out/'tables')
                    r=dict(signature=self.signature,preds=ps,base=bs);atomic_dump(r,path);del bundles;gc.collect()
                for key,p in r['preds'].items():
                    variant,c,method=key.split('__');allrows.append(dict(fold=f+1,variant=variant,case=c,method=method,**metrics(self.y[va],p)))
                    if key not in oof:oof[key]=np.zeros((len(self.X),5))
                    oof[key][va]=p
                for b in r['base']:baserows.append(dict(fold=f+1,variant=b['variant'],case=b['case'],model=b['model'],**metrics(self.y[va],b['probabilities'])))
                permutation_path=self.out/f'tables/port_permutation_outer{f+1}_{case}.joblib'
                if self.cfg['feature_mode']=='dstport29':
                    assert permutation_path.exists()
                    for row in joblib.load(permutation_path):
                        permutation_rows.append(dict(fold=f+1,case=row['case'],method=row['method'],repeat=row['repeat'],**metrics(self.y[va],row['probabilities'])))
        assert np.all(seen==1);self.oof=oof
        np.savez_compressed(self.out/'cv_cache/oof_predictions.npz',**oof)
        df=pd.DataFrame(allrows);df.to_csv(self.out/'tables/cv_ensemble_folds.csv',index=False)
        pd.DataFrame(baserows).to_csv(self.out/'tables/cv_base_folds.csv',index=False)
        if permutation_rows:pd.DataFrame(permutation_rows).to_csv(self.out/'tables/cv_port_permutation.csv',index=False)
        cols=list(metrics(self.y,next(iter(oof.values()))));summary=df.groupby(['variant','case','method'])[cols].agg(['mean','std'])
        summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(self.out/'tables/cv_ensemble_summary.csv',index=False)
        pooled=[]
        for key,p in oof.items():
            v,c,m=key.split('__');pooled.append(dict(variant=v,case=c,method=m,**metrics(self.y,p)))
        pd.DataFrame(pooled).to_csv(self.out/'tables/cv_pooled_oof_metrics.csv',index=False)
        # Fold comparisons are descriptive; five folds are dependent, not five independent datasets.
        stat=[];pairs=[]
        tuned=df[df.variant=='ddos_tuned'];groups=[g.sort_values('fold') for _,g in tuned.groupby(['case','method'])]
        for metric in ['macro_f1','ddos_recall']:
            arrays=[g[metric].to_numpy() for g in groups]
            if all(np.array_equal(arrays[0],x) for x in arrays[1:]):s,p=0.,1.
            else:s,p=friedmanchisquare(*arrays)
            stat.append(dict(metric=metric,test='Friedman',statistic=s,p_value=p,scope='six tuned configs; exploratory fold comparison'))
            for (k1,g1),(k2,g2) in combinations(list(tuned.groupby(['case','method'])),2):
                a=g1.sort_values('fold')[metric].to_numpy();b=g2.sort_values('fold')[metric].to_numpy()
                if np.array_equal(a,b):s,p=0.,1.
                else:s,p=wilcoxon(a,b,zero_method='wilcox')
                pairs.append(dict(metric=metric,a=str(k1),b=str(k2),statistic=s,p_raw=p))
        pairdf=pd.DataFrame(pairs)
        for metric,ix in pairdf.groupby('metric').groups.items():
            ids=list(ix);order=sorted(ids,key=lambda i:pairdf.loc[i,'p_raw']);prev=0.
            for rank,i in enumerate(order):prev=max(prev,min(1.,(len(order)-rank)*pairdf.loc[i,'p_raw']));pairdf.loc[i,'p_holm']=prev
        pd.DataFrame(stat).to_csv(self.out/'tables/cv_friedman.csv',index=False);pairdf.to_csv(self.out/'tables/cv_wilcoxon_holm.csv',index=False)
        write_json(self.out/'manifests/cv_complete.json',dict(signature=self.signature,oof_once=bool(np.all(seen==1)),rows=len(self.X)))
        print(summary.to_string(),flush=True);return df

    def fit_all(self):
        for case in CASES:
            marker=self.out/f'manifests/fit_{case}.json'
            if marker.exists():continue
            bundles,_,_=fit_case(self.X,self.y,None,case,self.cfg,'full_training',self.out/'tables',False)
            info={}
            for key,b in bundles.items():
                short=f'{case}__{b["method"]}';path=self.out/f'models/{short}.joblib';atomic_dump(b,path)
                info[short]=dict(path=path.name,sha256=sha256(path),params=b['selected_params'],weights=b.get('weights'),alpha=b['weight_alpha'])
            write_json(marker,info);del bundles;gc.collect()
        paths={f'{c}__{m}':self.out/f'models/{c}__{m}.joblib' for c in CASES for m in METHODS}
        self.model_paths=paths
        atomic_dump(dict(bundles={k:joblib.load(p) for k,p in paths.items()},protocol_signature=self.signature,dataset_sha256=self.data_hash),self.out/'models/checkpoint_training.joblib')
        write_json(self.out/'manifests/models_frozen.json',dict(signature=self.signature,model_hashes={k:sha256(p) for k,p in paths.items()},status='six models frozen before holdout'))
        print('Six final models saved and frozen.',flush=True)
        return paths

    def profile(self):
        self.model_paths={f'{c}__{m}':self.out/f'models/{c}__{m}.joblib' for c in CASES for m in METHODS}
        frozen=json.loads((self.out/'manifests/configuration_freeze.json').read_text())
        for key,path in self.model_paths.items():
            if sha256(path)!=frozen['model_hashes'][key]:raise ValueError('Model berubah setelah freeze')
        pos=sample_positions(self._yh,self.cfg['profile_rows'],self.cfg['seed']);batch=self.out/'profiling/batch.joblib'
        atomic_dump(self._Xh.iloc[pos],batch)
        pd.DataFrame(dict(holdout_position=pos,original_row_id=self._idh[pos])).to_csv(self.out/'profiling/batch_rows.csv',index=False)
        lat=[];mem=[];rng=np.random.default_rng(self.cfg['seed'])
        for block in range(self.cfg['latency_blocks']):
            for key in rng.permutation(list(self.model_paths)):
                print('Latency',block+1,key,flush=True)
                r=launch_profile(self.model_paths[key],batch,self.out/'profiling',f'latency_{block}_{key}','latency',self.cfg)
                for n,values in r['latency'].items():
                    for i,ms in enumerate(values):lat.append(dict(config=key,block=block,repeat=i,batch_rows=int(n),batch_ms=ms,amortized_ms_per_sample=ms/int(n),throughput_per_second=1000*int(n)/ms))
        for rep in range(self.cfg['memory_runs']):
            for key in rng.permutation(list(self.model_paths)):
                print('Memory',rep+1,key,flush=True)
                r=launch_profile(self.model_paths[key],batch,self.out/'profiling',f'memory_{rep}_{key}','memory',self.cfg)
                for phase in ['cold','warm']:
                    vals={k:v for k,v in r[phase].items() if not k.startswith('trace_')}
                    mem.append(dict(config=key,repeat=rep,phase=phase,**r['load'],**vals))
        lat=pd.DataFrame(lat);mem=pd.DataFrame(mem)
        lat.to_csv(self.out/'profiling/latency_raw.csv',index=False);mem.to_csv(self.out/'profiling/memory_raw.csv',index=False)
        for frame,groups,cols,filename in [(lat,['config','batch_rows'],['batch_ms','amortized_ms_per_sample','throughput_per_second'],'latency_summary.csv'),(mem,['config','phase'],['baseline_rss_mib','sampled_peak_rss_mib','peak_minus_baseline_mib','load_delta_rss_mib'],'memory_summary.csv')]:
            s=frame.groupby(groups)[cols].agg(['count','mean','std','median','min','max']);s.columns=['_'.join(c) for c in s.columns];s.reset_index().to_csv(self.out/'profiling'/filename,index=False)
        write_json(self.out/'manifests/profiling_complete.json',dict(profile_input='fixed stratified ORIGINAL HOLDOUT batch; labels only determine shared batch indices',batch_rows=len(pos),memory_definition='cold first predict in fresh process per repetition; sampled peak RSS minus baseline after load; mean across repetitions; recommendation uses delta MiB',machine=environment(),batch_sha256=sha256(batch),rss_scope='whole isolated process; model and inputs loaded before baseline',input_feature_count=len(self.X.columns),latency_definition='Separate process without RSS sampler; warm batch and single request separately; preprocessing through ensemble probabilities included; raw feature construction and disk/network excluded',caution='Sampled peak can miss spikes; zero delta makes draft recommendation incomplete; MiB=2**20 bytes; MB=MiB*1.048576',signature=self.signature))
        return lat,mem

    def freeze_configuration(self):
        """Freeze model/aturan SEBELUM holdout; belum ada rekomendasi."""
        hashes={k:sha256(p) for k,p in self.model_paths.items()}
        audit=dict(train_holdout_index_overlap=len(set(self.ids)&set(self._idh)),
            features_exclude_targets=not bool(FORBIDDEN_RAW&set(self.X.columns)),
            outer_oof_once=json.loads((self.out/'manifests/cv_complete.json').read_text())['oof_once'],
            holdout_resampled=False,features_hyperparameters_weights_fit_use_holdout=False,
            final_model_recommendation_uses_holdout=True,session_capture_independence='NOT_PROVEN',
            historical_holdout_previously_observed=True)
        if audit['train_holdout_index_overlap'] or not audit['features_exclude_targets'] or not audit['outer_oof_once']:
            raise ValueError('Leakage audit gagal')
        write_json(self.out/'manifests/audit.json',audit)
        freeze=dict(signature=self.signature,model_hashes=hashes,
            features_hyperparameters_weights_fit_use_holdout=False,
            final_model_recommendation_uses_holdout=True,recommendation_policy=self.cfg['recommendation_policy'],
            protocol_status=self.cfg['protocol_status'],score_tie_atol=1e-12,
            performance_weight=self.cfg['final_tradeoff_performance_weight'],memory_metric='delta_peak_inference_cold_mib',
            latency_metric='batch_ms',profile_rows=self.cfg['profile_rows'],
            shap_reference='original__soft_voting',prediction_threshold='argmax, unchanged',
            class_order=CLASSES,averaging='macro; AUROC OvR macro',
            holdout_status='historical comparative/exploratory; not independent post-selection evaluation')
        write_json(self.out/'manifests/configuration_freeze.json',freeze)
        self.freeze=freeze

    def recommend_holdout(self):
        from recommendation import rank_holdout,sensitivity_holdout
        hold=pd.read_csv(self.out/'tables/holdout_all_six.csv')
        lat=pd.read_csv(self.out/'profiling/latency_summary.csv')
        lat=lat[lat.batch_rows==self.cfg['profile_rows']]
        mem=pd.read_csv(self.out/'profiling/memory_summary.csv');mem=mem[mem.phase=='cold']
        t=hold.merge(lat[['config','batch_ms_mean']],on='config',how='left',validate='one_to_one')
        t=t.merge(mem[['config','peak_minus_baseline_mib_mean']],on='config',how='left',validate='one_to_one')
        t=t.rename(columns={'batch_ms_mean':'latency_batch_ms','peak_minus_baseline_mib_mean':'delta_peak_mib'})
        if self.cfg['recommendation_policy'] != 'holdout_all_six':
            raise ValueError('Engine aktif hanya mendukung ranking seluruh enam tanpa prioritas treatment')
        all_six,eligible,info=rank_holdout(t,self.cfg['final_tradeoff_performance_weight'],self.cfg['protocol_status'])
        all_six.to_csv(self.out/'tables/holdout_ranking_all_six.csv',index=False)
        eligible.to_csv(self.out/'tables/holdout_ranking_eligible.csv',index=False)
        info['signature']=self.signature
        info['operational_numbers_are_research_design']=True
        info['main_rule_frozen_before_holdout']=True
        write_json(self.out/'manifests/recommendation.json',info)
        summary,rankings=sensitivity_holdout(t,protocol_status=self.cfg['protocol_status'])
        summary.to_csv(self.out/'tables/recommendation_sensitivity.csv',index=False)
        rankings.to_csv(self.out/'tables/recommendation_sensitivity_all_six.csv',index=False)
        print('REKOMENDASI HOLDOUT (angka operasional: desain penelitian):',info,flush=True)
        return info

    def evaluate_holdout(self):
        self.freeze=json.loads((self.out/'manifests/configuration_freeze.json').read_text())
        assert self.freeze['signature']==self.signature
        rows=[]
        for key,expected in self.freeze['model_hashes'].items():
            path=self.out/f'models/{key}.joblib';assert sha256(path)==expected
            b=joblib.load(path);p=predict_bundle(b,self._Xh);m=metrics(self._yh,p);pred=p.argmax(1)
            rows.append(dict(config=key,feature_mode=b['feature_mode'],case=b['case'],method=b['method'],**m))
            pd.DataFrame(classification_report(self._yh,pred,labels=np.arange(5),target_names=CLASSES,output_dict=True,zero_division=0)).T.to_csv(self.out/f'tables/holdout_per_class_{key}.csv')
            cm=confusion_matrix(self._yh,pred,labels=np.arange(5));pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(self.out/f'tables/holdout_confusion_{key}.csv')
            pd.DataFrame(dict(original_row_id=self._idh,true_class=self._yh,predicted_class=pred,**{f'prob_{c}':p[:,i] for i,c in enumerate(CLASSES)})).to_csv(self.out/f'tables/holdout_predictions_{key}.csv',index=False)
            fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(5),CLASSES,rotation=35,ha='right');ax.set_yticks(range(5),CLASSES)
            for i in range(5):
                for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
            ax.set(xlabel='Predicted',ylabel='Actual',title=key);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(self.out/f'figures/confusion_{key}.png',dpi=150);plt.close(fig)
            print('Holdout',key,m,flush=True)
        df=pd.DataFrame(rows);assert len(df)==6;df.to_csv(self.out/'tables/holdout_all_six.csv',index=False)
        write_json(self.out/'manifests/holdout_complete.json',dict(signature=self.signature,freeze_sha256=sha256(self.out/'manifests/configuration_freeze.json'),rows=len(self._yh),configs=6,repeated_holdout_disclosure=self.cfg['historical_holdout_note'],holdout_did_not_change_fitted_models=True,final_model_recommendation_uses_holdout=True))
        return df

    def explain(self):
        import shap
        self.freeze=json.loads((self.out/'manifests/configuration_freeze.json').read_text())
        # Same background, global rows, local rows and masks across six configurations.
        bgpos=sample_positions(self.y,self.cfg['shap_background_rows'],self.cfg['seed'],True)
        glpos=sample_positions(self.y,self.cfg['shap_global_rows'],self.cfg['seed']+1,True)
        predframes={k:pd.read_csv(self.out/f'tables/holdout_predictions_{k}.csv') for k in self.freeze['model_hashes']}
        ref=predframes[self.freeze['shap_reference']].predicted_class.to_numpy()
        local=[];coverage=[]
        rng=np.random.default_rng(self.cfg['seed'])
        for c in range(5):
            for correct in [True,False]:
                pool=np.flatnonzero((self._yh==c)&((ref==self._yh)==correct));pick=int(rng.choice(pool)) if len(pool) else None
                if pick is not None:local.append(pick)
                coverage.append(dict(class_id=c,reference_correct=correct,available=len(pool),selected_position=pick))
        for k,pdfr in predframes.items():
            pool=np.flatnonzero((self._yh==1)&(pdfr.predicted_class.to_numpy()==2))
            if len(pool):local.append(int(pool[0]))
        local=np.array(list(dict.fromkeys(local))[:self.cfg['shap_local_max_rows']],dtype=int)
        pd.DataFrame(coverage).to_csv(self.out/'shap/case_coverage.csv',index=False)
        pd.DataFrame(dict(holdout_position=local,original_row_id=self._idh[local],true_class=self._yh[local])).to_csv(self.out/'shap/shared_local_cases.csv',index=False)
        pd.DataFrame(dict(training_position=bgpos,original_row_id=self.ids[bgpos])).to_csv(self.out/'shap/background_rows.csv',index=False)
        pd.DataFrame(dict(training_position=glpos,original_row_id=self.ids[glpos])).to_csv(self.out/'shap/global_rows.csv',index=False)
        levels={c:sorted(self.X[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}
        medians={c:float(pd.to_numeric(self.X[c],errors='coerce').median()) for c in NUMERIC_FEATURES}
        medians={k:(v if np.isfinite(v) else 0.) for k,v in medians.items()}
        columns=list(self.X.columns)
        def encode(frame):
            a=np.empty((len(frame),len(columns)))
            for j,c in enumerate(columns):
                if c in levels:a[:,j]=frame[c].astype(str).map({s:i for i,s in enumerate(levels[c])}).fillna(-1)
                else:a[:,j]=pd.to_numeric(frame[c],errors='coerce').fillna(medians[c])
            return a
        def decode(a):
            a=np.asarray(a);out={}
            for j,c in enumerate(columns):
                if c in levels:
                    lev=levels[c];out[c]=[lev[int(round(v))] if 0<=int(round(v))<len(lev) else '__UNSEEN__' for v in a[:,j]]
                else:out[c]=a[:,j]
            return pd.DataFrame(out)[columns]
        bg=encode(self.X.iloc[bgpos]);inputs=pd.concat([self.X.iloc[glpos],self._Xh.iloc[local]],ignore_index=True);enc=encode(inputs)
        globalrows=[];localrows=[];checks=[]
        for key,expected in self.freeze['model_hashes'].items():
            assert sha256(self.out/f'models/{key}.joblib')==expected
            artifact=self.out/f'shap/values_{key}.npz';b=joblib.load(self.out/f'models/{key}.joblib')
            fn=lambda a:predict_bundle(b,decode(a))
            # Numeric fill and categorical coding must not silently alter predictions.
            assert np.allclose(fn(enc),predict_bundle(b,inputs),atol=1e-7),'SHAP wrapper changed predictions'
            if artifact.exists():
                saved=np.load(artifact);values=saved['values'];base=saved['base']
            else:
                print('SHAP',key,'rows',len(enc),flush=True);np.random.seed(self.cfg['seed'])
                explainer=shap.KernelExplainer(fn,bg,link='identity')
                values=explainer.shap_values(enc,nsamples=self.cfg['shap_nsamples'],l1_reg=0.0,silent=True)
                if isinstance(values,list):values=np.stack(values,axis=-1)
                values=np.asarray(values);base=np.asarray(explainer.expected_value)
                assert values.shape==(len(enc),len(columns),5)
                np.savez_compressed(artifact,values=values,base=base,encoded_inputs=enc)
            prob=fn(enc);resid=np.abs(base+values.sum(axis=1)-prob)
            assert np.max(resid)<1e-4,'SHAP additivity failed; inspect before reporting'
            vg=values[:len(glpos)];importance=np.abs(vg).mean(axis=(0,2))
            for j,c in enumerate(columns):
                globalrows.append(dict(config=key,feature=c,output_class='ALL_MEAN_ABS',mean_abs_shap=importance[j]))
                for cl in range(5):globalrows.append(dict(config=key,feature=c,output_class=CLASSES[cl],mean_abs_shap=float(np.abs(vg[:,:,cl]).mean(0)[j])))
            order=np.argsort(importance)[-12:];fig,ax=plt.subplots(figsize=(9,6));ax.barh(np.array(columns)[order],importance[order]);ax.set(title=key+' | global SHAP',xlabel='mean |SHAP| across rows and outputs');fig.tight_layout();fig.savefig(self.out/f'figures/shap_global_{key}.png',dpi=150);plt.close(fig)
            for i,pos in enumerate(local):
                ix=len(glpos)+i;true=int(self._yh[pos]);pred=int(prob[ix].argmax())
                checks.append(dict(config=key,holdout_position=int(pos),original_row_id=int(self._idh[pos]),true_class=CLASSES[true],predicted_class=CLASSES[pred],correct=true==pred,max_additivity_residual=float(resid[ix].max())))
                for cl in range(5):
                    for j,c in enumerate(columns):localrows.append(dict(config=key,holdout_position=int(pos),original_row_id=int(self._idh[pos]),output_class=CLASSES[cl],feature=c,feature_value=str(inputs.iloc[ix,j]),shap_value=float(values[ix,j,cl]),base_value=float(base[cl]),output_probability=float(prob[ix,cl])))
                for cl in sorted({true,pred}):
                    exp=shap.Explanation(values=values[ix,:,cl],base_values=float(base[cl]),data=inputs.iloc[ix].to_numpy(),feature_names=columns)
                    shap.plots.waterfall(exp,max_display=12,show=False);plt.gcf().set_size_inches(14,8);plt.title(f'{key} | row {int(self._idh[pos])} | {CLASSES[cl]}');plt.tight_layout();plt.savefig(self.out/f'figures/shap_local_{key}_row{int(self._idh[pos])}_{CLASSES[cl]}.png',dpi=120,bbox_inches='tight');plt.close('all')
        pd.DataFrame(globalrows).to_csv(self.out/'shap/global_and_per_output_class.csv',index=False)
        pd.DataFrame(localrows).to_csv(self.out/'shap/local_all_features_all_outputs.csv',index=False)
        pd.DataFrame(checks).to_csv(self.out/'shap/local_predictions_additivity.csv',index=False)
        write_json(self.out/'manifests/shap_complete.json',dict(configs=6,local_rows=len(local),global_rows=len(glpos),background_rows=len(bgpos),nsamples=self.cfg['shap_nsamples'],global_sample='class-stratified TRAINING sample; output-class SHAP is not a causal explanation',local_selection='one correct/error per true class under fixed Original Soft Voting reference frozen before holdout plus DDoS->Port errors across configs; illustrative, not random population estimate',shared_cases=True,background_source='training only',additivity_max=float(pd.DataFrame(checks).max_additivity_residual.max()),signature=self.signature))
        print('SHAP complete for all six configurations.',flush=True)

    def finish(self):
        required=['cv_complete.json','models_frozen.json','profiling_complete.json','configuration_freeze.json','audit.json','holdout_complete.json','shap_complete.json','recommendation.json']
        if not all((self.out/'manifests'/n).exists() for n in required):raise ValueError('Tahap belum lengkap')
        info=json.loads((self.out/'manifests/recommendation.json').read_text())
        hold=pd.read_csv(self.out/'tables/holdout_all_six.csv')
        if len(hold)!=6:raise ValueError('Harus enam hasil holdout')
        status='SMOKE_TEST_ONLY' if self.cfg['smoke_test'] else ('EXPERIMENT_COMPLETE' if info['status']=='COMPLETE' else 'METRICS_COMPLETE_RECOMMENDATION_INCOMPLETE')
        lines=['# Analisis kandidat lokal TA Hafid',status,
            'Arahan dospem: rekomendasi holdout, validation/holdout original; angka operasional desain penelitian. Holdout historis comparative/exploratory; bukan evaluasi independen pascaseleksi.',
            'Rekomendasi: '+json.dumps(info,ensure_ascii=False),hold.to_csv(index=False),
            'Profil mesin run ini saja; jangan digabung dengan Colab lama. Lihat ranking semua enam dan eligible.']
        (self.out/'RINGKASAN_HASIL.md').write_text('\n\n'.join(lines),encoding='utf8')
        write_json(self.out/'manifests/run_complete.json',dict(status=status,signature=self.signature,
            model_count=6,recommendation_status=info['status'],all_shap_cases=True))
        return str(self.out)


def resident_bytes(path=None):
    if sys.platform.startswith('linux'):
        return int(Path(path or '/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    return psutil.Process().memory_info().rss

def profile_worker(config):
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    with threadpool_limits(limits=1):
        if sha256(config['batch'])!=config['batch_sha256'] or sha256(config['model'])!=config['model_sha256']:
            raise ValueError('Profiler artifact checksum mismatch')
        X=joblib.load(config['batch']);gc.collect();before=resident_bytes()
        bundle=joblib.load(config['model']);gc.collect();after=resident_bytes()
        load=dict(before_model_load_rss_mib=before/2**20,after_model_load_rss_mib=after/2**20,load_delta_rss_mib=(after-before)/2**20,artifact_mib=Path(config['model']).stat().st_size/2**20)
        if config['mode']=='latency':
            data={};cold={}
            for n in sorted({1,len(X)}):
                xx=X.iloc[:n];t=time.perf_counter_ns();p=predict_bundle(bundle,xx);cold[str(n)]=(time.perf_counter_ns()-t)/1e6;del p
                for _ in range(2):predict_bundle(bundle,xx)
                vals=[]
                for _ in range(config['latency_repeats']):
                    t=time.perf_counter_ns();p=predict_bundle(bundle,xx);vals.append((time.perf_counter_ns()-t)/1e6);del p
                data[str(n)]=vals
            print(json.dumps(dict(event='result',load=load,latency=data,cold_latency_ms=cold)),flush=True)
        else:
            for phase,calls in [('cold',1),('warm',5)]:
                if phase=='warm':
                    for _ in range(2):predict_bundle(bundle,X)
                gc.collect();base=resident_bytes()
                procfs_pid=Path('/proc/self/stat').read_text().split()[0] if sys.platform.startswith('linux') else str(os.getpid())
                print(json.dumps(dict(event='ready',phase=phase,baseline=base,load=load,calls=calls,rss_path=f'/proc/{procfs_pid}/statm')),flush=True)
                assert sys.stdin.readline().strip()=='START'
                for _ in range(calls):p=predict_bundle(bundle,X)
                end=resident_bytes()
                print(json.dumps(dict(event='done',phase=phase,end=end)),flush=True)
                assert sys.stdin.readline().strip()=='ACK'
                del p

def launch_profile(model,batch,directory,name,mode,cfg):
    directory=Path(directory);result=directory/f'{name}.json'
    if result.exists():return json.loads(result.read_text())
    conf=dict(model=str(Path(model).resolve()),batch=str(Path(batch).resolve()),mode=mode,latency_repeats=cfg['latency_repeats'],model_sha256=sha256(model),batch_sha256=sha256(batch))
    inp=directory/f'{name}_input.json';write_json(inp,conf)
    env=os.environ.copy();env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
    err=open(directory/f'{name}.log','w')
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(inp)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env=env,bufsize=1)
    def message():
        while True:
            line=p.stdout.readline()
            if not line:raise RuntimeError(f'Profiler failed: {name}.log')
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'event' in r:return r
    try:
        if mode=='latency':out=message()
        else:
            out={}
            for phase in ['cold','warm']:
                ready=message();assert ready['event']=='ready' and ready['phase']==phase
                out['load']=ready['load'];samples=[(0.,ready['baseline'])];stop=threading.Event();start=time.perf_counter()
                def monitor():
                    while not stop.wait(cfg['memory_interval_seconds']):
                        try:
                            rss=resident_bytes(ready['rss_path']) if sys.platform.startswith('linux') else psutil.Process(p.pid).memory_info().rss
                            samples.append((time.perf_counter()-start,rss))
                        except (psutil.NoSuchProcess,FileNotFoundError):break
                thread=threading.Thread(target=monitor,daemon=True);thread.start()
                p.stdin.write('START\n');p.stdin.flush();done=message();assert done['event']=='done'
                samples.append((time.perf_counter()-start,done['end']));stop.set();thread.join()
                peak=max(v for _,v in samples);intervals=np.diff([t for t,_ in samples])
                out[phase]=dict(baseline_rss_mib=ready['baseline']/2**20,sampled_peak_rss_mib=peak/2**20,peak_minus_baseline_mib=max(0,peak-ready['baseline'])/2**20,ending_rss_mib=done['end']/2**20,samples=len(samples),requested_interval_ms=cfg['memory_interval_seconds']*1000,max_actual_gap_ms=float(intervals.max()*1000),calls=ready['calls'],trace_seconds=[t for t,_ in samples],trace_rss_mib=[v/2**20 for _,v in samples])
                p.stdin.write('ACK\n');p.stdin.flush()
        code=p.wait(timeout=30);assert code==0
    finally:
        if p.poll() is None:p.kill();p.wait()
        err.close()
    write_json(result,out);return out


FEATURE_MODES=['bands28','dstport29']

def config_for(smoke=False,mode='bands28'):
    cfg=dict(feature_mode=mode)
    if smoke:cfg.update(smoke_test=True,outer_splits=2,inner_splits=2,profile_rows=20,memory_runs=1,latency_blocks=1,latency_repeats=2,shap_background_rows=5,shap_global_rows=5,shap_local_max_rows=3,shap_nsamples=64,permutation_repeats=1)
    return cfg

def comparison_protocol(root,smoke):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    sig=hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'recommendation.py').read_bytes()+json.dumps(dict(DEFAULT_CONFIG,smoke_requested=smoke),sort_keys=True).encode()).hexdigest()
    path=root/'comparison_protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==sig,'Different V4.3 source/protocol. Use a new folder.'
    else:write_json(path,dict(signature=sig,declared_before_cv=True,feature_modes=FEATURE_MODES,
        description='Same split/folds, cases and base learners for 28/29 features; full: 2 baseline / 8 tuned candidates; smoke: 1 baseline / 2 tuned, reduced folds and real-data subsets.',
        tuning='0.5 Macro-F1 + 0.5 F2 DDoS; base-candidate Macro-F1 floor = best 2-candidate baseline -0.01',
        feature_selection='Mean tuning score over all six ddos_tuned case/ensemble CV means; overall mean Macro-F1 floor = bands28 mean -0.01; ties within 1e-6 prefer bands28.',
        diagnostic='Outer-validation within-band port permutation; not used for feature or model selection; not an independent capture test.',
        reused_holdout=True,features_hyperparameters_weights_fit_use_holdout=False,final_model_recommendation_uses_holdout=True,protocol_status=DEFAULT_CONFIG['protocol_status'],smoke_test=smoke))
    return sig

def run_comparison(dataset,root,smoke=False):
    root=Path(root);sig=comparison_protocol(root,smoke);summaries=[];folds=[]
    baseline_assignment=None;baseline_holdout=None
    for mode in FEATURE_MODES:
        exp=Experiment(dataset,root/mode,config_for(smoke,mode)).prepare()
        if baseline_assignment is None:baseline_assignment=exp.ids.copy();baseline_holdout=exp._idh.copy()
        else:assert np.array_equal(baseline_assignment,exp.ids) and np.array_equal(baseline_holdout,exp._idh)
        print('FEATURE ABLATION:',mode,flush=True);exp.cross_validate()
        for filename,collection in [('cv_ensemble_summary.csv',summaries),('cv_ensemble_folds.csv',folds)]:
            df=pd.read_csv(exp.out/'tables'/filename);df.insert(0,'feature_mode',mode);collection.append(df)
        del exp;gc.collect()
    cv=pd.concat(summaries,ignore_index=True);cv.to_csv(root/'cv_feature_comparison.csv',index=False)
    allfolds=pd.concat(folds,ignore_index=True);allfolds.to_csv(root/'cv_feature_folds.csv',index=False)
    baseline=cv[(cv.feature_mode=='bands28')&(cv.variant=='ddos_tuned')]
    improved=cv[(cv.feature_mode=='dstport29')&(cv.variant=='ddos_tuned')]
    paired=baseline.merge(improved,on=['variant','case','method'],suffixes=('_bands28','_dstport29'))
    for metric in ['macro_f1','ddos_recall','ddos_precision','ddos_f2','port_scanning_recall','accuracy']:
        paired[metric+'_change_pp']=100*(paired[metric+'_mean_dstport29']-paired[metric+'_mean_bands28'])
    paired.to_csv(root/'paired_feature_effects.csv',index=False)
    results=[]
    for mode,g in cv[cv.variant=='ddos_tuned'].groupby('feature_mode'):
        macro=float(g.macro_f1_mean.mean());f2=float(g.ddos_f2_mean.mean())
        results.append(dict(feature_mode=mode,macro_f1_mean_across_six=macro,ddos_f2_mean_across_six=f2,
            ddos_recall_mean_across_six=float(g.ddos_recall_mean.mean()),ddos_precision_mean_across_six=float(g.ddos_precision_mean.mean()),
            port_scanning_recall_mean_across_six=float(g.port_scanning_recall_mean.mean()),score=.5*macro+.5*f2))
    rank=pd.DataFrame(results);floor=float(rank.loc[rank.feature_mode=='bands28','macro_f1_mean_across_six'].iloc[0])-.01
    rank['eligible']=rank.macro_f1_mean_across_six>=floor
    eligible=rank[rank.eligible];best=eligible[eligible.score>=eligible.score.max()-1e-6]
    chosen=('bands28' if 'bands28' in set(best.feature_mode) else best.sort_values('score',ascending=False).iloc[0].feature_mode)
    rank['selected']=rank.feature_mode==chosen;rank.to_csv(root/'feature_mode_selection.csv',index=False)
    write_json(root/'feature_mode_freeze.json',dict(signature=sig,selected_feature_mode=chosen,
        selection_used_holdout=False,selection_inputs='fresh outer CV for both feature modes',
        macro_floor=floor,shared_split_ids=True,shared_outer_fold_assignment=True,results=rank.to_dict('records'),
        caveat='Feature-mode selection using outer CV means makes selected CV score a model-selection estimate, not a fresh independent test.'))
    print('FEATURE MODE FROZEN:',chosen,rank.to_string(index=False),flush=True)
    return chosen

def selected_experiment(dataset,root,smoke):
    root=Path(root);sig=comparison_protocol(root,smoke)
    freeze=json.loads((root/'feature_mode_freeze.json').read_text());assert freeze['signature']==sig
    return Experiment(dataset,root/freeze['selected_feature_mode'],config_for(smoke,freeze['selected_feature_mode'])).prepare()

def finish_comparison(exp,root,smoke):
    root=Path(root);exp.finish()
    write_json(root/'run_complete.json',dict(status='SMOKE_TEST_ONLY' if smoke else 'COMPLETE_CANONICAL_DATASET',
        selected_feature_mode=exp.cfg['feature_mode'],six_holdout_configs=True,six_shap_configs=True,
        fresh_cv_feature_comparison=True,holdout_reused=True,features_hyperparameters_weights_fit_use_holdout=False,final_model_recommendation_uses_holdout=True,protocol_status=exp.cfg['protocol_status']))
    print('Analisis selesai:',root,';',exp.cfg['protocol_status'],flush=True)

def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        profile_worker(json.loads(Path(sys.argv[2]).read_text()))
    else:
        raise SystemExit('Gunakan runner lokal: python scripts/run_local.py --help')

if __name__=='__main__':main()
