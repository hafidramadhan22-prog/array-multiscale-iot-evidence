"""Frozen, targeted rule: RF checks SQL false alarms and rescues voting Normal.

Existing labels remain unchanged in every other disagreement. Full five-class
posteriors are selected, not synthesized confidence values. No payload model.
"""
import numpy as np
from rf_normal_voting_type_20261109 import combine as validate_pair
from rf_normal_voting_type_v2_20261109 import full_voting


def combine(voting,rf):
    validate_pair(voting,rf)
    v=np.asarray(voting,dtype=np.float64);r=np.asarray(rf,dtype=np.float64)
    vl=v.argmax(1);rl=r.argmax(1)
    sql_normal=(vl==3)&(rl==0);rescue=(vl==0)&(rl!=0)
    out=v.copy();out[sql_normal|rescue]=r[sql_normal|rescue]
    reason=np.where(sql_normal,'RF_checks_voting_SQL_as_Normal',np.where(rescue,'RF_rescues_voting_Normal','retain_voting'))
    return out,reason


def infer(bundle,X,names):
    # All five experts are needed: this targeted rule cannot skip voting for
    # RF-Normal rows because non-SQL attack predictions must remain intact.
    rf=np.asarray(bundle['models']['random_forest'].predict_proba(X),dtype=np.float64)
    voting=np.zeros_like(rf,dtype=np.float64)
    for name,weight in zip(names,bundle['weights']):
        p=rf if name=='random_forest' else np.asarray(bundle['models'][name].predict_proba(X),dtype=np.float64)
        voting+=float(weight)*p
    return combine(voting,rf)[0]
