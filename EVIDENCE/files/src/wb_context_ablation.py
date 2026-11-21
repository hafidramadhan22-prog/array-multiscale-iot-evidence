"""Matched context controls derived from verified prior-only endpoint vectors.

No labels or endpoint IDs enter reconstruction. Existing feature generation,
estimators and completed runs remain unchanged.
"""
import numpy as np
import pandas as pd
from e4 import DomainFeatures, columns
from wb_behavior import HOST_NAMES, WINDOWS

CONTEXTS = ("source20", "destination20", "destination_duplicate40", "ordered40", "symmetric40")


def endpoint_vectors(source, symmetric):
    """Invert the unordered two-vector representation using the saved source.

    Both vectors matching source is an allowed tie: the destination then equals
    source too. Anything else is rejected, never approximately matched.
    """
    source = np.asarray(source, dtype=float)
    symmetric = np.asarray(symmetric, dtype=float)
    width = len(WINDOWS) * len(HOST_NAMES)
    if (source.ndim != 2 or source.shape[1] != width
            or symmetric.shape != (len(source), 2 * width)
            or not np.isfinite(source).all() or not np.isfinite(symmetric).all()):
        raise ValueError("Invalid verified context shapes/values")
    higher, lower = symmetric[:, :width], symmetric[:, width:]
    is_higher = np.all(source == higher, axis=1)
    is_lower = np.all(source == lower, axis=1)
    if not np.all(is_higher | is_lower):
        raise ValueError("Source is not one of the two saved endpoint vectors")
    destination = np.where(is_higher[:, None], lower, higher)
    # Numeric key matches PriorEndpointRing._activity_key; never use an ID.
    h = len(HOST_NAMES)
    keys = np.c_[source[:, -h:], source]
    other = np.c_[destination[:, -h:], destination]
    greater = np.zeros(len(source), dtype=bool)
    tied = np.ones(len(source), dtype=bool)
    for j in range(keys.shape[1]):
        greater |= tied & (keys[:, j] > other[:, j])
        tied &= keys[:, j] == other[:, j]
    recovered = np.c_[np.where((greater | tied)[:, None], source, destination),
                      np.where((greater | tied)[:, None], destination, source)]
    if not np.array_equal(recovered, symmetric):
        raise ValueError("Saved endpoint order does not reproduce the canonical rule")
    return source.copy(), destination.copy()


def context_array(source, destination, context):
    if context == "source20": return source.copy()
    if context == "destination20": return destination.copy()
    if context == "destination_duplicate40": return np.c_[destination, destination]
    if context == "ordered40": return np.c_[source, destination]
    raise ValueError("Only new controls are constructed here")


def context_names(context):
    host = [f"{int(w * 1000)}ms_{n}" for w in WINDOWS for n in HOST_NAMES]
    slots = {"source20": ["source"], "destination20": ["destination"],
             "destination_duplicate40": ["destination_copy1", "destination_copy2"],
             "ordered40": ["source", "destination"]}
    if context not in slots: raise ValueError("Unknown matched context")
    return [f"matched_{slot}_{name}" for slot in slots[context] for name in host]


def matched_frame(raw, data, base, context, *, swap_context_slots=False):
    if base not in ("legacy25", "legacy29"): raise ValueError("Invalid base view")
    source, destination = endpoint_vectors(data["prior_source"], data["prior_symmetric"])
    if swap_context_slots:
        if context != "ordered40": raise ValueError("Slot exchange is defined only for ordered40")
        source, destination = destination, source
    extra = context_array(source, destination, context)
    mode = "noport25" if base == "legacy25" else "dstport29"
    numeric, categorical = columns(mode)
    names = context_names(context)
    if len(raw) != len(extra): raise ValueError("Context/raw row count mismatch")
    if not np.array_equal(raw.index.to_numpy(), data["row_ids"]):
        raise ValueError("Context/raw row identity mismatch")
    X = pd.concat([DomainFeatures(mode).transform(raw),
                   pd.DataFrame(extra, index=raw.index, columns=names)], axis=1)
    return X, list(numeric) + names, list(categorical)


def confusion_by_blocks(y, predictions, capture, timestamps, seconds):
    """Non-overlapping time blocks within capture, conditional on this trace.

    predictions: (seeds, rows). Time starts at each capture's first evaluated
    target. Capture IDs organize resampling only, never model inputs.
    """
    y, predictions = np.asarray(y), np.asarray(predictions)
    capture, timestamps = np.asarray(capture), np.asarray(timestamps)
    if (seconds <= 0 or predictions.ndim != 2 or predictions.shape[1] != len(y)
            or capture.shape != y.shape or timestamps.shape != y.shape
            or not np.isin(y, np.arange(5)).all() or not np.isin(predictions, np.arange(5)).all()):
        raise ValueError("Invalid paired block inputs")
    rows, strata, support = [], [], []
    for cap in np.unique(capture):
        pos = np.flatnonzero(capture == cap)
        block = ((timestamps[pos] - timestamps[pos].min()) // int(seconds * 1e9)).astype(np.int64)
        for value in np.unique(block):
            idx = pos[block == value]
            rows.append(np.stack([np.bincount(5*y[idx]+p[idx], minlength=25).reshape(5,5)
                                  for p in predictions]))
            strata.append(int(cap)); support.append(dict(capture=int(cap), block=int(value), rows=len(idx)))
    return np.stack(rows), np.asarray(strata), support


def macro_from_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(-1) + cm.sum(-2)
    return np.divide(2*diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0).mean(-1)


def paired_block_bootstrap(left, right, strata, repeats, seed):
    """Paired within-capture block sensitivity, not a cross-capture CI claim."""
    if left.shape != right.shape or left.ndim != 4 or len(strata) != len(left):
        raise ValueError("Paired confusion tensors must have identical blocks/seeds")
    rng = np.random.default_rng(seed)
    draws = np.zeros(repeats)
    captures = np.unique(strata)
    for cap in captures:
        idx = np.flatnonzero(strata == cap)
        counts = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)), size=repeats)
        # Hold samples shared across both methods and all three fitted seeds.
        if cap == captures[0]:
            lc = np.tensordot(counts, left[idx], axes=(1,0))
            rc = np.tensordot(counts, right[idx], axes=(1,0))
        else:
            lc += np.tensordot(counts, left[idx], axes=(1,0))
            rc += np.tensordot(counts, right[idx], axes=(1,0))
    draws[:] = (macro_from_confusion(lc) - macro_from_confusion(rc)).mean(1)
    point = float((macro_from_confusion(left.sum(0))-macro_from_confusion(right.sum(0))).mean())
    return dict(point_delta=point, lower=float(np.quantile(draws,.025)), upper=float(np.quantile(draws,.975)),
                repeats=repeats, seed=seed), draws
