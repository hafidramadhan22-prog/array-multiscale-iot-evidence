"""Isolated SQL-content pilot; no changes to the production feature contract."""
import re
from urllib.parse import parse_qsl,urlsplit
import numpy as np
import pandas as pd

ABSENT={'','0','0.0','nan','none','<na>','null'}
PREFIX_BYTES=4096
META_NAMES=['payload_present','payload_hex_valid','payload_log_bytes','payload_prefix_truncated']
SQL_NAMES=['payload_quote_count','payload_or_count','payload_equals_count','payload_numeric_tautology','payload_union_select','payload_isdecimal']
VIEWS=('baseline85','payload_meta89','payload_sql95')
BASE_NAMES=['decision_tree','random_forest','xgboost','lightgbm','logistic_regression']
METHODS=BASE_NAMES+['soft_voting','weighted_voting','stacking']

def bytes_from_hex(value):
    s=str(value).strip()
    if s.lower() in ABSENT:return None
    if re.fullmatch(r'(?:[0-9A-Fa-f]{2})+',s):return bytes.fromhex(s)
    if re.fullmatch(r'(?:[0-9A-Fa-f]{2}:)+[0-9A-Fa-f]{2}',s):return bytes.fromhex(s.replace(':',''))
    return None

def extract_sql_features(parameter):
    text=str(parameter).lower()
    # This refines the screenshot's toy regex: accept quoted/unquoted integer
    # equalities after OR/AND, and require the two numeric values to agree.
    # The supervisor's parameter ends with an open quote ('1); the surrounding
    # application's query can supply its closing quote. Accept that example.
    matches=re.finditer(r'''\b(?:or|and)\b\s*(['"]?)(\d+)\1\s*=\s*(['"]?)(\d+)(?:\3)?(?!\d)''',text)
    tautology=any(m.group(2)==m.group(4) for m in matches)
    return np.array([text.count("'"),len(re.findall(r'\bor\b',text)),text.count('='),int(tautology),int(bool(re.search(r'\bunion\s+select\b',text))),int(text.isdecimal())],dtype=float)

def parameter_text(tcp,mqtt):
    """Current-packet parameter values only; never header/cookie/host features.

    parse_qsl performs one URL decode. HTTP responses and binary TCP frames
    contribute no request-parameter text. MQTT falls back to its exported body.
    No reassembly, future packet, Attack_type or label is inspected.
    """
    if tcp is not None:
        text=tcp[:PREFIX_BYTES].decode('latin1')
        first=text.split('\r\n',1)[0].split('\n',1)[0]
        request=re.match(r'^(GET|POST|HEAD|PUT|DELETE|OPTIONS|PATCH|CONNECT|TRACE)\s+(\S+)\s+HTTP/\d',first)
        if request:
            try:query=urlsplit(request.group(2)).query
            except ValueError:query=''
            pairs=parse_qsl(query,keep_blank_values=True,encoding='utf8',errors='replace')
            headers,sep,body=text.partition('\r\n\r\n')
            if sep and re.search(r'(?im)^content-type:\s*application/x-www-form-urlencoded(?:\s*;|\s*$)',headers):
                pairs+=parse_qsl(body,keep_blank_values=True,encoding='utf8',errors='replace')
            return ' '.join(value for _,value in pairs)
    b=bytes_from_hex(mqtt)
    return b[:PREFIX_BYTES].decode('utf8',errors='replace').strip() if b is not None else ''

def extract_frame(raw):
    required={'tcp.payload','mqtt.msg'}
    if not required.issubset(raw.columns):raise ValueError('Payload columns missing')
    rows=[]
    for tcp,mqtt in raw[['tcp.payload','mqtt.msg']].itertuples(index=False,name=None):
        b=bytes_from_hex(tcp)
        meta=[float(str(tcp).strip().lower() not in ABSENT),float(b is not None),float(np.log1p(len(b))) if b is not None else 0.,float(b is not None and len(b)>PREFIX_BYTES)]
        rows.append(meta+extract_sql_features(parameter_text(b,mqtt)).tolist())
    values=np.asarray(rows,dtype=float).reshape(len(raw),10)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid payload features')
    return pd.DataFrame(values,index=raw.index,columns=META_NAMES+SQL_NAMES)

def attach(base,extra,view):
    if view not in VIEWS:raise ValueError('Unknown pilot view')
    if not base.index.equals(extra.index):raise ValueError('Payload/base row alignment changed')
    if view=='baseline85':return base
    names=META_NAMES if view=='payload_meta89' else META_NAMES+SQL_NAMES
    return pd.concat([base,extra[names]],axis=1)

def predict(bundle,X,method):
    if method not in METHODS:raise ValueError('Unknown pilot model')
    if method in BASE_NAMES:p=bundle['models'][method].predict_proba(X)
    else:
        ps=[np.asarray(bundle['models'][n].predict_proba(X),dtype=float) for n in BASE_NAMES]
        if method=='soft_voting':p=np.mean(ps,axis=0)
        elif method=='weighted_voting':p=sum(float(w)*q for w,q in zip(bundle['weights'],ps))
        else:p=bundle['meta'].predict_proba(np.hstack(ps))
    p=np.asarray(p,dtype=float)
    if p.shape!=(len(X),5) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid probabilities')
    np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-6)
    return p
