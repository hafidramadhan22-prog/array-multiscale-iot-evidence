"""Frozen feature repair: remove packet signatures and expose scale-free ratios."""
import numpy as np
import pandas as pd
from wb_behavior import PRIOR_SYMMETRIC, HOST_NAMES, WINDOWS
from e4 import DomainFeatures, columns

RATIO_NAMES = ('peer_per_packet','local_tcp_port_per_packet','peer_tcp_port_per_packet',
               'byte_coefficient_variation','syn_among_syn_ack','rst_among_syn_rst')
RATIO_COLUMNS = [f'context_{slot}_{int(window*1000)}ms_{name}'
                 for slot in ('higher','lower') for window in WINDOWS for name in RATIO_NAMES]
VIEWS = ('context40','context64','legacy25_context64')

def divide(a,b):
    return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)

def context_ratios(symmetric):
    x=np.asarray(symmetric,dtype=float)
    if x.ndim!=2 or x.shape[1]!=40 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Expected finite nonnegative symmetric40')
    shaped=x.reshape(len(x),2,2,len(HOST_NAMES))
    counts=np.expm1(shaped[...,:5])
    if not np.isfinite(counts).all() or (counts[...,2:5]>counts[...,:1]+1e-8).any():
        raise ValueError('Invalid diversity/count values')
    flags=shaped[...,7:10]
    if (flags>1).any():raise ValueError('Invalid flag fraction')
    ratios=np.stack([divide(counts[...,2],counts[...,0]),divide(counts[...,3],counts[...,0]),
        divide(counts[...,4],counts[...,0]),divide(shaped[...,6],shaped[...,5]),
        divide(flags[...,0],flags[...,0]+flags[...,1]),
        divide(flags[...,2],flags[...,0]+flags[...,2])],axis=-1)
    if not np.isfinite(ratios).all():raise ValueError('Nonfinite derived features')
    return ratios.reshape(len(x),24)

def repair_frame(raw,data,view):
    if view not in VIEWS:raise ValueError('Unreviewed repair view')
    if not np.array_equal(raw.index.to_numpy(),data['row_ids']):
        raise ValueError('Feature/row identity mismatch')
    context=np.asarray(data['prior_symmetric'],dtype=float)
    if context.shape!=(len(raw),40):raise ValueError('Context shape mismatch')
    # Validate even the context-only control, without using labels/IDs as values.
    ratios=context_ratios(context)
    names=list(PRIOR_SYMMETRIC)
    x=pd.DataFrame(context,index=raw.index,columns=names)
    if view!='context40':
        x=pd.concat([x,pd.DataFrame(ratios,index=raw.index,columns=RATIO_COLUMNS)],axis=1)
        names+=RATIO_COLUMNS
    categorical=[]
    if view.startswith('legacy25'):
        num,cat=columns('noport25')
        x=pd.concat([DomainFeatures('noport25').transform(raw),x],axis=1)
        names=list(num)+names;categorical=list(cat)
    return x,names,categorical

def select_on_g(records):
    """Pure calibration selection; caller must freeze this before touching V."""
    if not records or any(z.get('role')!='G' for z in records):
        raise ValueError('Selection is restricted to G')
    return min(records,key=lambda z:(-z['score'],z['bytes'],z['view'],z['method']))
