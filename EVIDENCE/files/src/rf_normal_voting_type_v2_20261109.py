"""Same frozen label rule with explicit float64 probability accumulation.

The first profiler exposed float32 expert-product rounding (~1e-8). The
posteriors used for scientific decisions were already float64. This version
matches those posteriors without relaxing the 1e-12 verification tolerance.
"""
import numpy as np
from rf_normal_voting_type_20261109 import combine


def full_voting(bundle, X, names):
    out = np.zeros((len(X), 5), dtype=np.float64)
    for name, weight in zip(names, bundle['weights']):
        out += float(weight) * np.asarray(bundle['models'][name].predict_proba(X), dtype=np.float64)
    return out


def infer(bundle, X, names):
    rf = np.asarray(bundle['models']['random_forest'].predict_proba(X), dtype=np.float64)
    out = rf.copy()
    active = np.flatnonzero(rf.argmax(1) != 0)
    if len(active):
        xx = X.iloc[active]
        weighted = np.zeros((len(active), 5), dtype=np.float64)
        for name, weight in zip(names, bundle['weights']):
            p = rf[active] if name == 'random_forest' else np.asarray(bundle['models'][name].predict_proba(xx), dtype=np.float64)
            weighted += float(weight) * p
        out[active] = combine(weighted, rf[active])[0]
    return out
