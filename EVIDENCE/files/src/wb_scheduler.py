"""Exact class-label short circuit for a fixed nonnegative weighted ensemble.

This is an inference optimization, not a new classifier or a novelty claim.
Uncomputed expert probabilities are never synthesized.  The full reference is
the float64 weighted sum in canonical expert-index order, followed by argmax.
"""
from dataclasses import dataclass
from itertools import permutations
import math

import numpy as np


@dataclass
class SchedulerResult:
    labels: np.ndarray
    counts: np.ndarray
    evaluated: np.ndarray
    certified: np.ndarray
    partial_scores: np.ndarray


def _weights_order(weights, order):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.ndim != 1 or len(weights) == 0 or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-10)):
        raise ValueError('Weights must be finite, nonnegative, and already sum to one')
    order = tuple(order)
    if (any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in order)
            or len(order) != len(weights) or set(order) != set(range(len(weights)))):
        raise ValueError('Order must be a permutation of all canonical expert indices')
    return weights, tuple(int(v) for v in order)


def _probabilities(values, rows, classes):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (rows, classes) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError('Expert output must be aligned finite probabilities in [0,1] summing to one')
    return values


def _canonical_sum(cache, weights):
    scores = np.zeros((len(cache), cache.shape[-1]), dtype=np.float64)
    for expert, weight in enumerate(weights):
        scores += weight * cache[:, expert, :]
    return scores


def full_reference(probabilities, weights):
    """Return reference probabilities using the same float64/canonical contract.

    The probability tensor has shape (samples, experts, classes).  Inputs and
    weights are checked but never renormalized or clipped.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] < 2:
        raise ValueError('Expected probabilities shaped (samples, experts, classes>=2)')
    weights, _ = _weights_order(weights, range(p.shape[1]))
    for expert in range(p.shape[1]):
        _probabilities(p[:, expert], len(p), p.shape[2])
    return _canonical_sum(p, weights)


def predict_lazy(n_samples, n_classes, weights, order, callbacks, *, tolerance=1e-12):
    """Predict labels while calling each expert only for its unresolved rows.

    callbacks[i](row_indices) must return probabilities for precisely those
    input rows, in the ensemble's common class order.  The caller owns model
    loading, row-to-input mapping and shared preprocessing.  A skipped callback
    is genuinely not invoked.  Zero-weight experts are never invoked.

    Stopping: partial leader minus every rival is strictly greater than the
    total unevaluated weight plus tolerance and a float64 roundoff guard.  Rows
    not certified early use a full canonical-order sum, including canonical
    argmax tie behavior.  partial_scores are NOT complete probabilities and
    must not be used for AUROC, calibration, or probability-based explanation.
    """
    weights, order = _weights_order(weights, order)
    if (not isinstance(n_samples, (int, np.integer)) or n_samples < 0
            or not isinstance(n_classes, (int, np.integer)) or n_classes < 2):
        raise ValueError('Invalid number of samples or classes')
    if len(callbacks) != len(weights) or any(not callable(fn) for fn in callbacks):
        raise ValueError('Provide one lazy callback for each expert')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    models = len(weights)
    cache = np.zeros((n_samples, models, n_classes), dtype=np.float64)
    scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    labels = np.full(n_samples, -1, dtype=np.int64)
    evaluated = np.zeros((n_samples, models), dtype=bool)
    certified = np.zeros(n_samples, dtype=bool)
    active = np.arange(n_samples, dtype=np.int64)
    # This exceeds the usual gamma_k bound for <=m products/additions on
    # nonnegative, bounded scores. It also protects subtraction and differences
    # between execution-order and canonical-order accumulation.
    numeric_guard = 64 * np.finfo(np.float64).eps * max(1, models) * max(1., float(weights.sum()))
    positive_order = tuple(i for i in order if weights[i] > 0)
    for step, expert in enumerate(positive_order):
        if not len(active):
            break
        values = _probabilities(callbacks[expert](active.copy()), len(active), n_classes)
        cache[active, expert] = values
        evaluated[active, expert] = True
        scores[active] += weights[expert] * values
        remaining = math.fsum(float(weights[i]) for i in positive_order[step+1:])
        if step == len(positive_order)-1:
            canonical = _canonical_sum(cache[active], weights)
            scores[active] = canonical
            labels[active] = canonical.argmax(axis=1)
            active = active[:0]
            break
        active_scores = scores[active]
        leader = active_scores.argmax(axis=1)
        rival = active_scores.copy()
        rival[np.arange(len(active)), leader] = -np.inf
        margin = active_scores[np.arange(len(active)), leader] - rival.max(axis=1)
        finish = margin > remaining + tolerance + numeric_guard
        done = active[finish]
        labels[done] = leader[finish]
        certified[done] = True
        active = active[~finish]
    if len(active) or (labels < 0).any():
        raise RuntimeError('Scheduler failed to resolve all rows')
    return SchedulerResult(labels, evaluated.sum(axis=1), evaluated, certified, scores)


def replay_scheduler(probabilities, weights, order, *, tolerance=1e-12):
    """Simulate decisions on cached posteriors; this does not measure latency."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] < 2:
        raise ValueError('Expected probabilities shaped (samples, experts, classes>=2)')
    weights, order = _weights_order(weights, order)
    if p.shape[1] != len(weights):
        raise ValueError('Expert dimension does not match weights')
    # Unlike lazy prediction all cached values are available, so reject invalid
    # values even in an expert that the replay would not happen to evaluate.
    for expert in range(p.shape[1]):
        _probabilities(p[:, expert], len(p), p.shape[2])
    callbacks = [lambda rows, i=i: p[rows, i] for i in range(p.shape[1])]
    return predict_lazy(len(p), p.shape[2], weights, order, callbacks, tolerance=tolerance)


def estimated_row_cost(result, expert_costs, *, shared_preprocessing_cost=0.,
                       preprocessing_groups=None, preprocessing_costs=None):
    """Additive per-decision calibration proxy; actual runtime still needs testing.

    Expert costs exclude preprocessing. A common cost is charged to every row.
    A group-specific preprocessing cost is charged once iff any expert from the
    group was evaluated for that row. Costs must use one consistent time unit.
    This is not a model of nonlinear batched throughput or scheduling overhead.
    """
    expert_costs = np.asarray(expert_costs, dtype=np.float64)
    if (expert_costs.shape != (result.evaluated.shape[1],)
            or not np.isfinite(expert_costs).all() or (expert_costs < 0).any()
            or not np.isfinite(shared_preprocessing_cost) or shared_preprocessing_cost < 0):
        raise ValueError('Costs must be aligned, finite and nonnegative')
    cost = result.evaluated @ expert_costs + float(shared_preprocessing_cost)
    if (preprocessing_groups is None) != (preprocessing_costs is None):
        raise ValueError('Supply both preprocessing_groups and preprocessing_costs')
    if preprocessing_groups is not None:
        groups = tuple(preprocessing_groups)
        if len(groups) != len(expert_costs) or set(groups) != set(preprocessing_costs):
            raise ValueError('Preprocessing group cost keys must match the expert groups')
        for group in dict.fromkeys(groups):
            value = float(preprocessing_costs[group])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Preprocessing costs must be finite and nonnegative')
            used = result.evaluated[:, [i for i, g in enumerate(groups) if g == group]].any(axis=1)
            cost = cost + value * used
    return cost


def best_order(calibration_probabilities, weights, expert_costs, *,
               shared_preprocessing_cost=0., preprocessing_groups=None,
               preprocessing_costs=None, tolerance=1e-12):
    """Enumerate fixed orders using calibration posteriors and calibration costs.

    No labels are needed. The caller must persist calibration row IDs and freeze
    this result before validation. This function cannot infer a tensor's data
    provenance. For the five-model project all 120 orders are considered.
    """
    p = np.asarray(calibration_probabilities, dtype=np.float64)
    if p.ndim != 3 or len(p) == 0 or p.shape[2] < 2:
        raise ValueError('Order selection requires a nonempty calibration tensor')
    weights, _ = _weights_order(weights, range(p.shape[1]))
    if len(weights) > 8:
        raise ValueError('Exhaustive fixed-order search is limited to eight experts')
    ranking = []
    reference = full_reference(p, weights).argmax(axis=1)
    for order in permutations(range(len(weights))):
        result = replay_scheduler(p, weights, order, tolerance=tolerance)
        if not np.array_equal(result.labels, reference):
            raise RuntimeError('Certificate violated full ensemble label identity')
        cost = estimated_row_cost(result, expert_costs,
            shared_preprocessing_cost=shared_preprocessing_cost,
            preprocessing_groups=preprocessing_groups, preprocessing_costs=preprocessing_costs)
        ranking.append(dict(order=list(order), estimated_mean_cost=float(cost.mean()),
                            mean_evaluations=float(result.counts.mean()),
                            certified_fraction=float(result.certified.mean())))
    ranking.sort(key=lambda row: (row['estimated_mean_cost'], tuple(row['order'])))
    return dict(**ranking[0], ranking=ranking, orders_evaluated=len(ranking),
                calibration_rows=len(p), selection_uses_labels=False,
                cost_scope='additive per-decision proxy; excludes scheduler overhead',
                tolerance=float(tolerance))
