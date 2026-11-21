"""Conservative missing-feature bounds for a fixed heterogeneous soft vote.

Research feasibility wrapper, not a new classifier or a formally verified
floating-point implementation. Unsupported estimators return [0, 1]. Unknown
coordinates range over the complete real line; no empirical feature ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import warnings

import numpy as np
from scipy import sparse
from scipy.special import logsumexp


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _softmax_intervals(lower, upper, guard=0.):
    """Coordinate bounds, allowing infinite logit endpoints without inf-inf."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    classes = len(lower)
    lo, hi = np.zeros(classes), np.ones(classes)
    for c in range(classes):
        other = np.arange(classes) != c
        if np.isfinite(lower[c]) and not np.isposinf(upper[other]).any():
            lo[c] = np.exp(-np.logaddexp(0., logsumexp(upper[other] - lower[c])))
        if np.isfinite(upper[c]) and not np.isneginf(lower[other]).any():
            hi[c] = np.exp(-np.logaddexp(0., logsumexp(lower[other] - upper[c])))
    return np.maximum(0., lo-guard), np.minimum(1., hi+guard)


@dataclass
class BoundResult:
    lower: np.ndarray
    upper: np.ndarray
    certified_labels: np.ndarray
    expert_lower: np.ndarray
    expert_upper: np.ndarray


class EstimatorBounds:
    """Interval relaxation. Independent tree extrema can be conservative.

    Numeric trees only; scipy CSR absent entries are missing for XGBoost and
    zero for sklearn/LightGBM. Known sklearn/XGBoost inputs are cast float32
    before their native split comparison. Missing unknown inputs remain
    possible for XGBoost; all branches are conservatively reachable.
    """
    def __init__(self, estimator, calibration_matrix):
        self.estimator = estimator
        self.classes = len(estimator.classes_)
        self.kind = 'unsupported'
        self.reason = None
        self.trees = []
        self.offset = np.zeros(self.classes)
        self.offset_guard = 0.
        self.probability_guard = 128*np.finfo(float).eps
        self.descriptor = {'estimator': type(estimator).__name__, 'classes': self.classes}
        try:
            self._build()
            self._check_reference(calibration_matrix)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as ex:
            self.kind = 'unsupported'
            self.reason = f'{type(ex).__name__}: {ex}'
        self.descriptor.update(kind=self.kind, unsupported_reason=self.reason,
                               probability_guard=self.probability_guard,
                               offset=self.offset.tolist(), offset_guard=self.offset_guard,
                               numeric_assumption='IEEE arithmetic; conservative operation-count guards; not a machine-checked proof')

    def _build(self):
        name = type(self.estimator).__name__
        if self.classes < 3:
            raise ValueError('Only multiclass probability objectives are supported')
        if name in ('DecisionTreeClassifier', 'RandomForestClassifier'):
            self.kind = 'sklearn_tree'
            estimators = [self.estimator] if name == 'DecisionTreeClassifier' else self.estimator.estimators_
            for model in estimators:
                t = model.tree_
                values = t.value[:, 0, :].astype(float)
                totals = values.sum(1, keepdims=True)
                values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
                self.trees.append(dict(feature=t.feature, threshold=t.threshold,
                    left=t.children_left, right=t.children_right, values=values,
                    missing_left=getattr(t, 'missing_go_to_left', np.zeros(t.node_count))))
            self.probability_guard = 64*np.finfo(float).eps*(len(self.trees)+self.classes+1)
        elif name == 'LogisticRegression':
            if (self.estimator.coef_.shape[0] != self.classes or self.estimator.solver == 'liblinear'
                    or getattr(self.estimator, 'multi_class', None) == 'ovr'):
                raise ValueError('Only multinomial logistic regression is supported')
            self.kind = 'linear'
        elif name == 'XGBClassifier':
            self._xgboost()
        elif name == 'LGBMClassifier':
            self._lightgbm()
        else:
            raise ValueError('No audited bounds adapter')

    def _xgboost(self):
        b = self.estimator.get_booster()
        document = json.loads(bytes(b.save_raw(raw_format='json')).decode())
        learner = document['learner']
        if learner['objective']['name'] != 'multi:softprob':
            raise ValueError('Unsupported XGBoost objective')
        gradient = learner['gradient_booster']
        if gradient['name'] != 'gbtree':
            raise ValueError('Only gbtree booster is supported')
        model = gradient['model']
        base = np.asarray(json.loads(learner['learner_model_param']['base_score']), float).reshape(-1)
        if len(base) == 1: base = np.repeat(base, self.classes)
        if len(base) != self.classes or not np.isfinite(base).all():
            raise ValueError('Unsupported XGBoost base score')
        self.offset = base
        for raw, cls in zip(model['trees'], model['tree_info']):
            if any(raw['split_type']) or raw.get('categories'):
                raise ValueError('Categorical XGBoost split unsupported')
            # JSON retains the native binary32 thresholds and leaf values.
            self.trees.append(dict(feature=np.asarray(raw['split_indices'], int),
                threshold=np.asarray(raw['split_conditions'], dtype=np.float32).astype(float),
                left=np.asarray(raw['left_children'], int), right=np.asarray(raw['right_children'], int),
                missing_left=np.asarray(raw['default_left'], bool),
                values=np.asarray(raw['split_conditions'], dtype=np.float32).astype(float), cls=int(cls)))
        self.kind = 'xgboost'
        self.probability_guard = 64*(self.classes+1)*np.finfo(np.float32).eps

    def _lightgbm(self):
        model = self.estimator.booster_.dump_model()
        if not str(model['objective']).startswith('multiclass ') and model['objective'] != 'multiclass':
            raise ValueError('Unsupported LightGBM objective')
        if model.get('average_output') or model['num_tree_per_iteration'] != self.classes:
            raise ValueError('Unsupported LightGBM aggregation')
        for index, raw in enumerate(model['tree_info']):
            nodes = []
            def visit(node):
                at = len(nodes); nodes.append(None)
                if 'leaf_value' in node:
                    nodes[at] = dict(feature=-1, threshold=0., left=-1, right=-1,
                                     value=float(node['leaf_value']), default_left=False, missing_type='None')
                else:
                    if node['decision_type'] != '<=':
                        raise ValueError('Categorical/nonstandard LightGBM split unsupported')
                    left, right = visit(node['left_child']), visit(node['right_child'])
                    nodes[at] = dict(feature=int(node['split_feature']), threshold=float(node['threshold']),
                        left=left, right=right, value=0., default_left=bool(node.get('default_left', False)),
                        missing_type=node.get('missing_type', 'None'))
                return at
            visit(raw['tree_structure'])
            self.trees.append(dict(feature=np.array([n['feature'] for n in nodes]),
                threshold=np.array([n['threshold'] for n in nodes]),
                left=np.array([n['left'] for n in nodes]), right=np.array([n['right'] for n in nodes]),
                values=np.array([n['value'] for n in nodes]),
                missing_left=np.array([n['default_left'] for n in nodes]),
                missing_type=[n['missing_type'] for n in nodes], cls=index % self.classes))
        # LightGBM dump leaf values already include shrinkage and initial scores.
        self.kind = 'lightgbm'
        self.probability_guard = 256*(self.classes+1)*np.finfo(float).eps

    def _known_row(self, matrix, row):
        if sparse.issparse(matrix):
            r = matrix.getrow(row).tocsr()
            values = np.zeros(matrix.shape[1], dtype=float)
            values[r.indices] = r.data
            absent = np.ones(matrix.shape[1], dtype=bool); absent[r.indices] = False
        else:
            values = np.asarray(matrix[row], dtype=float).reshape(-1).copy()
            absent = np.zeros(len(values), dtype=bool)
        missing = np.isnan(values) | (absent if self.kind == 'xgboost' else False)
        if self.kind == 'xgboost':
            sentinel = self.estimator.missing
            if sentinel is not None and np.isfinite(sentinel): missing |= values == sentinel
        if self.kind in ('xgboost', 'sklearn_tree'):
            values = values.astype(np.float32).astype(float)
        return values, missing

    def _tree_range(self, tree, values, missing, unknown):
        low = np.full(self.classes, np.inf) if self.kind == 'sklearn_tree' else np.inf
        high = -low
        stack = [0]
        while stack:
            node = stack.pop()
            if tree['left'][node] < 0:
                value = tree['values'][node]
                low, high = np.minimum(low, value), np.maximum(high, value)
                continue
            feature = tree['feature'][node]
            if unknown[feature]:
                stack.extend((int(tree['left'][node]), int(tree['right'][node])))
                continue
            value = values[feature]
            is_missing = missing[feature]
            if self.kind == 'lightgbm':
                missing_type = tree['missing_type'][node]
                if missing_type == 'Zero' and abs(value) <= 1e-35: is_missing = True
                if missing_type == 'None' and np.isnan(value):
                    value = 0.; is_missing = False
            if is_missing:
                left = tree['missing_left'][node]
            else:
                left = value < tree['threshold'][node] if self.kind == 'xgboost' else value <= tree['threshold'][node]
            stack.append(int(tree['left'][node] if left else tree['right'][node]))
        return low, high

    def _raw_bounds(self, matrix, row, unknown):
        values, missing = self._known_row(matrix, row)
        if self.kind == 'linear':
            coef = self.estimator.coef_
            finite_values = values.copy(); finite_values[unknown] = 0.
            if not np.isfinite(finite_values).all(): return np.full(self.classes, -np.inf), np.full(self.classes, np.inf)
            score = coef @ finite_values + self.estimator.intercept_
            eps = np.finfo(float).eps
            magnitude = np.abs(coef) @ np.abs(finite_values) + np.abs(self.estimator.intercept_)
            guard = 64*eps*(len(values)+1)*np.maximum(1., magnitude)
            lo, hi = score-guard, score+guard
            affected = np.any(coef[:, unknown] != 0, axis=1)
            lo[affected], hi[affected] = -np.inf, np.inf
            return lo, hi
        if self.kind == 'sklearn_tree':
            ranges = [self._tree_range(t, values, missing, unknown) for t in self.trees]
            return np.mean([r[0] for r in ranges], axis=0), np.mean([r[1] for r in ranges], axis=0)
        lo, hi, magnitude, counts = self.offset.copy(), self.offset.copy(), np.abs(self.offset), np.zeros(self.classes)
        for tree in self.trees:
            a, b = self._tree_range(tree, values, missing, unknown); c = tree['cls']
            lo[c] += a; hi[c] += b; magnitude[c] += max(abs(a), abs(b)); counts[c] += 1
        eps = np.finfo(np.float32 if self.kind == 'xgboost' else float).eps
        # Conservative summation envelope: products already reside in stored
        # leaf values; guards cover native summation and Python accumulation.
        terms = 8*(counts+2)*eps
        if (terms >= .5).any(): raise ValueError('Too many stages for numerical guard')
        guard = terms/(1-terms)*np.maximum(1., magnitude)+self.offset_guard
        return lo-guard, hi+guard

    def probabilities(self, matrix, unknown):
        unknown = np.asarray(unknown, dtype=bool)
        if unknown.shape != (matrix.shape[1],): raise ValueError('Unknown feature mask dimension')
        lower, upper = np.zeros((len_or_rows(matrix), self.classes)), np.ones((len_or_rows(matrix), self.classes))
        if self.kind == 'unsupported': return lower, upper
        for row in range(matrix.shape[0]):
            lo, hi = self._raw_bounds(matrix, row, unknown)
            if self.kind == 'sklearn_tree':
                lower[row], upper[row] = np.maximum(0, lo-self.probability_guard), np.minimum(1, hi+self.probability_guard)
            else:
                lower[row], upper[row] = _softmax_intervals(lo, hi, self.probability_guard)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
            raise ValueError('Invalid probability bounds')
        return lower, upper

    def _check_reference(self, matrix):
        if matrix.shape[0] < 2: raise ValueError('At least two provenance-approved calibration rows required')
        if self.kind in ('xgboost', 'lightgbm'):
            exact = np.asarray(self.estimator.predict(matrix, output_margin=True) if self.kind == 'xgboost'
                               else self.estimator.predict(matrix, raw_score=True), float)
            raw = []
            for row in range(matrix.shape[0]):
                values, missing = self._known_row(matrix, row); summed = np.zeros(self.classes)
                for tree in self.trees:
                    a, b = self._tree_range(tree, values, missing, np.zeros(matrix.shape[1], bool))
                    if a != b: raise ValueError('All-known tree has multiple leaves')
                    summed[tree['cls']] += a
                raw.append(summed)
            residual = exact - np.asarray(raw)
            # Base margins must match the serialized model contract. Residuals
            # are recorded for audit, not used to fit an empirical correction.
            self.descriptor['raw_constant_observed_min'] = residual.min(0).tolist()
            self.descriptor['raw_constant_observed_max'] = residual.max(0).tolist()
            self.descriptor['raw_constant_declared'] = self.offset.tolist()
            for row in range(matrix.shape[0]):
                lo, hi = self._raw_bounds(matrix, row, np.zeros(matrix.shape[1], bool))
                if (exact[row] < lo).any() or (exact[row] > hi).any():
                    raise ValueError('Serialized raw-margin constant failed native validation')
        lower, upper = self.probabilities(matrix, np.zeros(matrix.shape[1], bool))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            exact = np.asarray(self.estimator.predict_proba(matrix), float)
        if (exact < lower).any() or (exact > upper).any():
            raise ValueError('All-known probability enclosure failed')
        self.descriptor['all_known_validation_rows'] = int(matrix.shape[0])
        self.descriptor['all_known_max_interval_width'] = float(np.max(upper-lower))


def len_or_rows(matrix):
    return matrix.shape[0]


class FeatureCertificate:
    """Mask prior_* features *before* transform; actual context is inaccessible."""
    def __init__(self, models, calibration_frame, names, *, prefix='prior_'):
        self.models, self.names, self.prefix = models, tuple(names), prefix
        self.adapters, self.unknown = {}, {}
        for name in self.names:
            model = models[name]; prep = model['preprocessor']
            features = np.asarray(prep.get_feature_names_out(), str)
            mask = np.array([part.split('__', 1)[-1].startswith(prefix) for part in features])
            if not mask.any(): raise ValueError('No unknown prior behavior feature found')
            self.unknown[name] = mask
            self.adapters[name] = EstimatorBounds(model['estimator'], prep.transform(calibration_frame))

    def descriptor(self):
        return dict(domain='all real finite transformed prior_* values; missing also admitted where native format permits',
            empirical_feature_range_used=False, model_names=list(self.names),
            models={n:dict(self.adapters[n].descriptor, unknown_features=int(self.unknown[n].sum())) for n in self.names})

    def bounds(self, frame, weights):
        weights = np.asarray(weights, float)
        if weights.shape != (len(self.names),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
            raise ValueError('Invalid canonical probability weights')
        cheap = frame.copy()
        for column in cheap.columns:
            if str(column).startswith(self.prefix): cheap[column] = 0.
        lower, upper = [], []
        cache = {}
        for name in self.names:
            prep = self.models[name]['preprocessor']; key = id(prep)
            if key not in cache: cache[key] = prep.transform(cheap)
            lo, hi = self.adapters[name].probabilities(cache[key], self.unknown[name])
            lower.append(lo); upper.append(hi)
        expert_lower, expert_upper = np.stack(lower, axis=1), np.stack(upper, axis=1)
        lo = sum(weights[j]*expert_lower[:, j] for j in range(len(weights)))
        hi = sum(weights[j]*expert_upper[:, j] for j in range(len(weights)))
        guard = 128*np.finfo(float).eps*len(weights)
        lo, hi = np.maximum(0, lo-guard), np.minimum(1, hi+guard)
        labels = lo.argmax(1); rivals = hi.copy(); rivals[np.arange(len(frame)), labels] = -np.inf
        labels[lo[np.arange(len(frame)), labels] <= rivals.max(1)+guard] = -1
        return BoundResult(lo, hi, labels, expert_lower, expert_upper)
