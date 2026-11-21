"""E4: fixed development-only port ablation with common signature groups.

No model deserialization, holdout evaluation, hyperparameter search or resampling.
The common signature is a conservative exact-feature pattern, not a capture ID.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, fbeta_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import ta_hafid_engine as e

BASE_NUMERIC = e.NUMERIC_KEEP + ['dns.qry.name.len__numeric']
BASE_CATEGORICAL = list(e.CATEGORICAL_KEEP)
PORT_BANDS = [f'{c}__band' for c in e.PORT_FEATURES]
RAW_COLUMNS = e.NUMERIC_KEEP + e.CATEGORICAL_KEEP + ['dns.qry.name.len'] + e.PORT_FEATURES
MODES = ['noport25', 'bands28', 'dstport29']
SCENARIOS = ['original', 'within_band', 'cross_band']


def columns(mode):
    if mode not in MODES:
        raise ValueError('Unknown E4 feature mode')
    numeric = BASE_NUMERIC + (['tcp.dstport__numeric'] if mode == 'dstport29' else [])
    categorical = BASE_CATEGORICAL + (PORT_BANDS if mode != 'noport25' else [])
    return numeric, categorical


class DomainFeatures(TransformerMixin, BaseEstimator):
    """Stateless source-compatible domain mapping; independent of engine globals."""
    def __init__(self, mode='noport25'):
        self.mode = mode

    def fit(self, X, y=None):
        columns(self.mode)
        return self

    def transform(self, X):
        numeric, categorical = columns(self.mode)
        out = {c: pd.to_numeric(X[c], errors='coerce') for c in e.NUMERIC_KEEP}
        out['dns.qry.name.len__numeric'] = pd.to_numeric(X['dns.qry.name.len'].astype(str), errors='coerce')
        for c in BASE_CATEGORICAL:
            out[c] = X[c].map(e.normalize_absent).astype(str)
        if self.mode != 'noport25':
            for c in e.PORT_FEATURES:
                out[f'{c}__band'] = X[c].map(e.port_band).astype(str)
        if self.mode == 'dstport29':
            p = pd.to_numeric(X['tcp.dstport'], errors='coerce')
            out['tcp.dstport__numeric'] = p.where(p.between(0, 65535) & p.eq(np.floor(p)), np.nan)
        return pd.DataFrame(out, index=X.index)[numeric + categorical]


def signatures(raw):
    X = DomainFeatures('noport25').transform(raw)
    g = pd.util.hash_pandas_object(X, index=False).to_numpy()
    # Collision check against exact feature equality on the actual input.
    if len(np.unique(g)) != len(X.drop_duplicates()):
        raise ValueError('Signature hash collision; stop rather than merge patterns')
    return g


def guard_split(ids, y, groups, train, validation, holdout_ids):
    ids, y, groups = np.asarray(ids), np.asarray(y), np.asarray(groups)
    train, validation = np.asarray(train), np.asarray(validation)
    if len(np.unique(ids)) != len(ids) or set(ids) & set(holdout_ids):
        raise ValueError('Duplicate development IDs or historical holdout leakage')
    if set(train) & set(validation) or set(np.r_[train, validation]) != set(range(len(ids))):
        raise ValueError('Train/validation must partition development')
    if set(groups[train]) & set(groups[validation]):
        raise ValueError('Signature overlap between training and validation')
    for part in [train, validation]:
        if set(y[part]) != set(range(5)):
            raise ValueError('Each training and validation partition must contain all five classes')


def grouped_splits(raw, y, holdout_ids):
    groups = signatures(raw)
    counts = {e.CLASSES[c]: int(len(np.unique(groups[y == c]))) for c in range(5)}
    if min(counts.values()) < 2:
        raise ValueError(f'Too few signatures for two-fold class coverage: {counts}')
    splits = list(StratifiedGroupKFold(2, shuffle=False).split(raw, y, groups))
    for tr, va in splits:
        guard_split(raw.index.to_numpy(), y, groups, tr, va, holdout_ids)
    return groups, splits, counts


def make_pipeline(name, mode, params, seed):
    if name not in ['logistic_regression', 'random_forest']:
        raise ValueError('E4 only permits the requested LR/RF')
    num, cat = columns(mode)
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if name == 'logistic_regression':
        num_steps.append(('scaler', StandardScaler(with_mean=False)))
    prep = ColumnTransformer([
        ('numeric', Pipeline(num_steps), num),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), cat)
    ], remainder='drop', sparse_threshold=1.0, verbose_feature_names_out=True)
    return Pipeline([('features', DomainFeatures(mode)), ('preprocessor', prep),
                     ('model', e.estimator(name, params, seed))])


def predict(model, raw):
    p = np.asarray(model.predict_proba(raw))
    if not np.array_equal(model.classes_, np.arange(5)):
        raise ValueError('Wrong probability class order')
    if p.shape != (len(raw), 5) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probabilities')
    return p, p.argmax(axis=1)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if set(y) != set(range(5)):
        raise ValueError('Do not report comparable five-class metrics without class coverage')
    out = dict(n=len(y), macro_f1=float(f1_score(y, pred, labels=range(5), average='macro', zero_division=0)),
               ddos_f2=float(fbeta_score(y == 1, pred == 1, beta=2, zero_division=0)))
    recalls = recall_score(y, pred, labels=range(5), average=None, zero_division=0)
    for c, label in enumerate(e.CLASSES):
        out[f'recall_{label}'] = float(recalls[c])
        out[f'support_{label}'] = int((y == c).sum())
    return out


def shifted_ports(raw, scenario, seed):
    """Change port values, not labels, using a frozen label-independent rule.

    Within each band use a nonzero cyclic offset (a bijection). Cross-band maps
    W->R->D->W; cardinality differs so collisions are possible and are disclosed.
    All three explicit raw port columns are stressed consistently.
    """
    if scenario not in SCENARIOS:
        raise ValueError('Unknown stress scenario')
    out = raw.copy()
    mappings = []
    if scenario == 'original':
        return out, mappings
    bounds = [(1, 1023), (1024, 49151), (49152, 65535)]
    rng = np.random.default_rng(seed)
    for column in e.PORT_FEATURES:
        p = pd.to_numeric(raw[column], errors='coerce')
        valid = p.between(1, 65535) & p.eq(np.floor(p))
        frequencies = p[valid].value_counts()
        for band, (lo, hi) in enumerate(bounds):
            mask = valid & p.between(lo, hi)
            target_lo, target_hi = bounds[band if scenario == 'within_band' else (band + 1) % 3]
            size = target_hi - target_lo + 1
            offset = int(rng.integers(1, size))
            values = np.sort(p[mask].unique()).astype(np.int64)
            mapped = target_lo + ((values - lo + offset) % size)
            mapping = dict(zip(values, mapped))
            # object dtype avoids float/string assignment surprises; invalid tokens stay intact.
            out[column] = out[column].astype(object)
            out.loc[mask, column] = p[mask].map(mapping).to_numpy()
            for old, new in zip(values, mapped):
                mappings.append(dict(column=column, old_port=int(old), new_port=int(new),
                                     n_rows=int(frequencies[old]), source_band=band,
                                     target_band=band if scenario == 'within_band' else (band + 1) % 3))
    # A label-free invariant: projections excluding ports are exactly unchanged.
    pd.testing.assert_frame_equal(DomainFeatures('noport25').transform(raw), DomainFeatures('noport25').transform(out))
    if scenario == 'within_band':
        pd.testing.assert_frame_equal(DomainFeatures('bands28').transform(raw), DomainFeatures('bands28').transform(out))
    return out, mappings


def latency_block(model, raw, warmups, repeats):
    with threadpool_limits(limits=1):
        for _ in range(warmups):
            predict(model, raw)
        times = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model.predict_proba(raw).argmax(axis=1)
            times.append((time.perf_counter_ns() - start) / 1e6)
    return times


def verified_historical_file(root, relative):
    root = Path(root)
    matches = [r for r in json.loads((root/'artifacts_sha256.json').read_text()) if r['path'] == relative]
    if len(matches) != 1:
        raise ValueError(f'Historical artifact is not catalogued: {relative}')
    path = root / relative
    if e.sha256(path) != matches[0]['sha256'] or path.stat().st_size != matches[0]['bytes']:
        raise ValueError(f'Historical checksum mismatch: {relative}')
    return path
