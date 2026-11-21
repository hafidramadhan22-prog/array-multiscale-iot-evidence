"""Frozen syntax/character TF-IDF pilot; source features are never modified."""
import re,time,warnings
from urllib.parse import parse_qsl,urlsplit
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import ta_hafid_engine as e
from payload_pilot import bytes_from_hex,extract_sql_features,BASE_NAMES,METHODS,predict

LIMIT=4096
SYNTAX_NAMES=['syntax_quote','syntax_or','syntax_equals','syntax_tautology','syntax_union_select','syntax_isdecimal','syntax_double_quote','syntax_and','syntax_line_comment','syntax_block_comment','syntax_semicolon','syntax_sql_keyword']
META_NAMES=['content_raw_present','content_hex_valid','content_log_bytes','content_truncated']
VIEWS=('baseline85','base_availability','syntax_only','tfidf_only','syntax_tfidf','base_syntax','base_tfidf','base_hybrid')

def content_text(tcp_value,mqtt_value):
    tcp=bytes_from_hex(tcp_value);mqtt=bytes_from_hex(mqtt_value)
    if tcp is not None:
        text=tcp[:LIMIT].decode('latin1')
        first=text.split('\r\n',1)[0].split('\n',1)[0]
        request=re.match(r'^(GET|POST|HEAD|PUT|DELETE|OPTIONS|PATCH|CONNECT|TRACE)\s+(\S+)\s+HTTP/\d',first)
        if request:
            try:q=urlsplit(request.group(2)).query
            except ValueError:q=''
            values=[v for _,v in parse_qsl(q,keep_blank_values=True,encoding='utf8',errors='replace')]
            head,sep,body=text.partition('\r\n\r\n')
            if sep and re.search(r'(?im)^content-type:\s*application/x-www-form-urlencoded(?:\s*;|\s*$)',head):
                values.extend(v for _,v in parse_qsl(body,keep_blank_values=True,encoding='utf8',errors='replace'))
            return ' '.join(values)
        if text.startswith('HTTP/'):
            # Server/date/cookies are deliberately excluded; keep current body.
            return text.partition('\r\n\r\n')[2]
    if mqtt is not None:return mqtt[:LIMIT].decode('latin1').strip()
    return tcp[:LIMIT].decode('latin1') if tcp is not None else ''

def syntax12(text):
    t=text.lower()
    extra=[t.count('"'),len(re.findall(r'\band\b',t)),t.count('--')+t.count('#'),t.count('/*'),t.count(';'),len(re.findall(r'\b(?:select|union|insert|update|delete|drop|sleep|benchmark|information_schema|case|when)\b',t))]
    return np.r_[extract_sql_features(t),np.asarray(extra,dtype=float)]

def augment(base,raw):
    if not base.index.equals(raw.index):raise ValueError('Payload/base identity mismatch')
    if not {'tcp.payload','mqtt.msg'}.issubset(raw):raise ValueError('Missing raw payload columns')
    text=[];features=[];meta=[]
    for a,b in raw[['tcp.payload','mqtt.msg']].itertuples(index=False,name=None):
        tcp=bytes_from_hex(a);mqtt=bytes_from_hex(b);t=content_text(a,b)
        text.append(t);features.append(syntax12(t))
        n=len(tcp) if tcp is not None else len(mqtt) if mqtt is not None else 0
        meta.append([float(tcp is not None or mqtt is not None),float(tcp is not None),np.log1p(n),float(n>LIMIT)])
    out=base.copy();out['payload_text']=text
    out[META_NAMES]=np.asarray(meta,dtype=float).reshape(len(raw),4)
    out[SYNTAX_NAMES]=np.asarray(features,dtype=float).reshape(len(raw),12)
    if not np.isfinite(out[META_NAMES+SYNTAX_NAMES].to_numpy()).all():raise ValueError('Invalid content features')
    return out

def uses_base(view):return view.startswith('base')
def uses_syntax(view):return view in ('syntax_only','syntax_tfidf','base_syntax','base_hybrid')
def uses_tfidf(view):return view in ('tfidf_only','syntax_tfidf','base_tfidf','base_hybrid')

def preprocessor(view,mode,num,cat,cfg):
    if view not in VIEWS or mode not in ('tree','linear'):raise ValueError('Unknown feature recipe')
    branches=[]
    numeric=(list(num) if uses_base(view) else [])+(META_NAMES if view=='base_availability' else [])+(SYNTAX_NAMES if uses_syntax(view) else [])
    if numeric:
        steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
        if mode=='linear':steps.append(('scale',StandardScaler(with_mean=False)))
        branches.append(('numeric',Pipeline(steps),numeric))
    if uses_base(view):
        branches.append(('categorical',Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),list(cat)))
    if uses_tfidf(view):
        branches.append(('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=cfg['tfidf_max_features'],min_df=1,lowercase=True,sublinear_tf=True,norm='l2',dtype=np.float64),'payload_text'))
    return ColumnTransformer(branches,sparse_threshold=1.,remainder='drop',verbose_feature_names_out=True)

def fit_models(X,y,fit,other,view,num,cat,cfg):
    modes={};plans=[];y=np.asarray(y)
    for mode in ('tree','linear'):
        xt=X.iloc[fit];e.guard_training(xt,cfg,X.index[other],'training')
        prep=preprocessor(view,mode,num,cat,cfg);Z=prep.fit_transform(xt);yy=y[fit].copy()
        before={int(c):int(n) for c,n in zip(*np.unique(yy,return_counts=True))}
        if set(before)!=set(range(5)):raise ValueError('Missing training class')
        target=min(before[0],max(round(cfg['full_balance_target']*len(yy)/cfg['train_rows']),max(before[c] for c in range(1,5))))
        if cfg['synthetic_cap']!=.5 or cfg['synthetic_cap_denominator']!='per_class_original_before_resampling':raise ValueError('Synthetic cap contract changed')
        targets={c:min(target,n+int(np.floor(.5*n))) if n<target else target for c,n in before.items()}
        rus={c:targets[c] for c,n in before.items() if n>targets[c]};smote={c:targets[c] for c,n in before.items() if n<targets[c]}
        if rus:Z,yy=RandomUnderSampler(sampling_strategy=rus,random_state=cfg['seed']).fit_resample(Z,yy)
        if smote:Z,yy=SMOTE(sampling_strategy=smote,random_state=cfg['seed'],k_neighbors=min(5,min(before[c] for c in smote)-1)).fit_resample(Z,yy)
        after={int(c):int(n) for c,n in zip(*np.unique(yy,return_counts=True))}
        if any(max(0,after[c]-before[c])>int(np.floor(.5*before[c])) for c in before):raise ValueError('Synthetic cap exceeded')
        # Balanced pilot/OOF inputs must remain unchanged. Do not fake six runs.
        if before!=after or rus or smote:raise ValueError('Equal-class pilot unexpectedly resampled')
        vocab=prep.named_transformers_['tfidf'].vocabulary_ if uses_tfidf(view) else {}
        plans.append(dict(mode=mode,fit_row_ids=xt.index.tolist(),validation_row_ids=X.index[other].tolist(),before=before,after=after,synthetic=0,encoded_features=int(Z.shape[1]),vocabulary_size=len(vocab),vocabulary_fit_rows=xt.index.tolist() if vocab else [],resampling_equivalent_to_original=True))
        modes[mode]=(prep,Z,yy)
    models={};records=[]
    for name in BASE_NAMES:
        prep,Z,yy=modes['linear' if name=='logistic_regression' else 'tree'];est=e.estimator(name,cfg['params'][name],cfg['seed'])
        if name=='logistic_regression':est.set_params(max_iter=cfg['lr_max_iter'],tol=cfg['lr_tol'])
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning);est.fit(Z,yy,sample_weight=np.where(yy==1,cfg['params'][name]['ddos_loss_weight'],1.))
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise ValueError('Convergence gate: '+view+'/'+name)
        np.testing.assert_array_equal(est.classes_,np.arange(5));models[name]=Pipeline([('preprocessor',prep),('estimator',est)])
        records.append(dict(model=name,estimator_fit_seconds=time.perf_counter()-start))
    return models,plans,records
