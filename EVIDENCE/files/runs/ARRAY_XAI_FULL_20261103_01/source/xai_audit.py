"""Model-agnostic SHAP audit on explicit donor intervention games. No fitting."""
from math import comb
import numpy as np
import pandas as pd
from wb_research import cube, mixture
import ta_hafid_engine as e

METHODS = ['weighted_voting', 'random_forest', 'logistic_regression']


def semantic_groups(columns):
    out = {}
    for col in columns:
        if col.startswith('prior_'): key = 'endpoint_history'
        elif col.endswith('__band') or col == 'tcp.dstport__numeric': key = 'explicit_ports'
        else: key = col.split('.')[0]
        if key not in ['arp', 'http', 'tcp', 'udp', 'dns', 'mqtt', 'explicit_ports', 'endpoint_history']:
            raise ValueError('Unmapped domain feature '+col)
        out.setdefault(key, []).append(col)
    flat = [c for cols in out.values() for c in cols]
    if len(flat) != len(set(flat)) or set(flat) != set(columns): raise ValueError('Group partition')
    return out


def subset_masks(k):
    if not 1 <= k <= 12: raise ValueError('Bounded exact enumeration required')
    return ((np.arange(2**k)[:, None] >> np.arange(k)) & 1).astype(bool)


def exact_shapley(values):
    n = len(values); k = n.bit_length()-1
    if 2**k != n: raise ValueError('All coalitions required')
    masks = subset_masks(k); counts = masks.sum(1)
    phi = np.zeros((k,)+values.shape[1:], dtype=float)
    for j in range(k):
        ix = np.flatnonzero(~masks[:, j])
        weights = np.array([1/(k*comb(k-1, int(counts[i]))) for i in ix])
        delta = values[ix | (1 << j)]-values[ix]
        phi[j] = np.tensordot(weights, delta, axes=(0, 0))
    np.testing.assert_allclose(phi.sum(0), values[-1]-values[0], rtol=1e-9, atol=1e-11)
    return phi


class HybridGame:
    """Binary coalition input; hidden columns all use the same empirical donor row."""
    def __init__(self, bundle, sample, background, groups, chunk_rows=4096):
        self.bundle = bundle; self.sample = sample.copy(); self.background = background.copy()
        self.groups = list(groups.values()); self.names = list(groups); self.chunk_rows = chunk_rows
        flat = [c for g in self.groups for c in g]
        if len(flat) != len(set(flat)) or set(flat) != set(sample.index) or list(background.columns) != list(sample.index):
            raise ValueError('Hybrid game schema mismatch')
        self.cache = {}; self.rows_evaluated = 0

    def __call__(self, masks):
        masks = np.asarray(masks)
        if masks.ndim != 2 or masks.shape[1] != len(self.groups) or not np.isin(masks, [0,1]).all():
            raise ValueError('Binary coalition masks required')
        keys = [tuple(map(bool, row)) for row in masks]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        per_chunk = max(1, self.chunk_rows//len(self.background))
        for start in range(0, len(missing), per_chunk):
            frames = []
            current = missing[start:start+per_chunk]
            for mask in current:
                frame = self.background.copy()
                for active, columns in zip(mask, self.groups):
                    if active:
                        for c in columns: frame[c] = self.sample[c]
                frames.append(frame)
            X = pd.concat(frames, ignore_index=True)
            ps = cube(self.bundle['models'], X)
            wb = mixture(ps, self.bundle['weights'])
            # Also preserve all five experts to verify linearity of Shapley aggregation.
            all_outputs = np.concatenate([wb[:, None, :], ps], axis=1)
            means = all_outputs.reshape(len(current), len(self.background), 6, 5).mean(1)
            self.rows_evaluated += len(X)
            self.cache.update(zip(current, means))
        return np.stack([self.cache[key].reshape(-1) for key in keys])


def path_masks(order):
    k = len(order)
    if sorted(order) != list(range(k)): raise ValueError('Permutation required')
    ins = np.zeros((k+1, k), dtype=bool)
    for i,j in enumerate(order): ins[i+1] = ins[i]; ins[i+1,j] = True
    return ~ins, ins


def auc(curve):
    return float(np.trapezoid(curve, dx=1/(len(curve)-1)))


def curve_metrics(deletion, insertion, random_deletion, random_insertion):
    return dict(deletion_auc=auc(deletion), insertion_auc=auc(insertion),
        random_deletion_auc=auc(random_deletion), random_insertion_auc=auc(random_insertion),
        deletion_gain=auc(random_deletion)-auc(deletion), insertion_gain=auc(insertion)-auc(random_insertion),
        comprehensiveness_one=float(deletion[0]-deletion[1]),
        sufficiency_gap_one=float(insertion[-1]-insertion[1]))


def grouped_curves(values, attribution, target, model_index, ranking='signed'):
    k = len(attribution); masks = subset_masks(k); score = attribution[:, model_index, target]
    order = np.argsort(-score if ranking == 'signed' else (-np.abs(score) if ranking == 'absolute' else score), kind='stable')
    dm, im = path_masks(order)
    ix = 2**np.arange(k)
    deletion = values[dm@ix, model_index, target]; insertion = values[im@ix, model_index, target]
    counts = masks.sum(1)
    random_insertion = np.array([values[counts==j, model_index, target].mean() for j in range(k+1)])
    random_deletion = random_insertion[::-1].copy()
    return order, deletion, insertion, random_deletion, random_insertion


def attribution_distance(a, b, top=3):
    a,b=np.asarray(a),np.asarray(b); na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if a.shape != b.shape: raise ValueError('Attribution shape')
    k=min(top,len(a)); sa=set(np.argsort(-np.abs(a),kind='stable')[:k]); sb=set(np.argsort(-np.abs(b),kind='stable')[:k])
    return dict(normalized_l1=float(np.abs(a-b).sum()/(np.abs(a).sum()+np.abs(b).sum()+1e-15)),
        top3_jaccard=len(sa&sb)/len(sa|sb),
        cosine=float(np.dot(a,b)/(na*nb)) if na*nb>1e-15 else (1. if max(na,nb)<1e-12 else 0.),
        attribution_degenerate=bool(max(na,nb)<1e-12))


def mask_violations(sample, background, masks, groups):
    """Count checkable hybrid violations, not a complete network semantic validator."""
    atomic = all(len(v)==1 for v in groups.values())
    if not atomic: return dict(context_donor_mixed=0, port_raw_band_disagreement=0, masks=len(masks), full_network_validity_checked=False)
    names=list(groups); history=[i for i,c in enumerate(names) if c.startswith('prior_')]
    mixed=int(np.sum(np.any(masks[:,history],axis=1)&~np.all(masks[:,history],axis=1))) if history else 0
    ports=0
    if 'tcp.dstport__numeric' in names and 'tcp.dstport__band' in names:
        a=names.index('tcp.dstport__numeric');b=names.index('tcp.dstport__band')
        for mask in masks:
            for _,donor in background.iterrows():
                raw=sample[names[a]] if mask[a] else donor[names[a]]
                band=sample[names[b]] if mask[b] else donor[names[b]]
                ports += int(e.port_band(raw) != band)
    return dict(context_donor_mixed=mixed, port_raw_band_disagreement=ports, masks=len(masks), full_network_validity_checked=False)
