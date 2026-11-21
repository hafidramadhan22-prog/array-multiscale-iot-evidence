"""Replay the three frozen policies from prepared context85 features, without fit.

Only the explicitly owned model with the hard-coded digest is deserialized.
Run this script with the archived package versions; no external downloads occur.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/INDEPENDENT_CAPTURE_20261109_01'
MODEL_SHA = '77aa1beb6980f4eb9db9838af6505949899c5aa574f3be49d55adfdc1efcf31d'
CATALOG_SHA = '9b25f63c75777d7c1aad4d04c75405832062d994c2410985e58fc53f6513d537'
SCHEMA_SHA = '8d17054cbe64a71070cf1c71a35ca1ff42239e325f1fa2d8fe3bc811dd946b9d'
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_child(parent, relative):
    target = (parent / relative).resolve()
    if not target.is_relative_to(parent.resolve()):
        raise ValueError('Path outside archive root')
    return target


def verify():
    catalog = RUN / 'artifacts_sha256.json'
    if sha(catalog) != CATALOG_SHA:
        raise ValueError('Frozen catalog SHA-256 mismatch; refusing deserialization')
    for row in json.loads(catalog.read_text('utf8')):
        p = checked_child(RUN, row['path'])
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Frozen artifact mismatch: ' + row['path'])
    schema = ROOT / 'configs/fpr_normal_voting_schema_20261109.json'
    if sha(schema) != SCHEMA_SHA:
        raise ValueError('Feature schema changed')
    model = RUN / 'models/reference_five.joblib'
    manifest = json.loads((RUN / 'model_manifest.json').read_text('utf8'))
    if manifest['sha256'] != MODEL_SHA or sha(model) != MODEL_SHA:
        raise ValueError('Unknown model; refusing deserialization')
    # Check original inference source before importing its policy implementation.
    for row in json.loads((RUN / 'source_manifest.json').read_text('utf8')):
        if sha(checked_child(ROOT, row['path'])) != row['sha256']:
            raise ValueError('Frozen source changed: ' + row['path'])
    return model, json.loads(schema.read_text('utf8'))


def replay(features, output, audit_path):
    if output.exists() or audit_path.exists():
        raise FileExistsError('Choose new output paths; existing evidence is never overwritten')
    if output.resolve().is_relative_to(RUN.resolve()) or audit_path.resolve().is_relative_to(RUN.resolve()):
        raise ValueError('The sealed run is read-only')
    model, schema = verify()
    import numpy as np
    import pandas as pd
    import joblib
    from threadpoolctl import threadpool_limits
    from fpr_target8_20261109 import NAMES, CLASSES, alternatives
    from sql_guard_rf_rescue_20261109 import combine

    data = pd.read_csv(features, dtype={c: str for c in schema['categorical']},
                       keep_default_na=False, float_precision='round_trip', low_memory=False)
    expected = ['row_id'] + schema['numeric'] + schema['categorical']
    if list(data.columns) != expected or not len(data) or data.row_id.duplicated().any():
        raise ValueError('Expected unique row_id and exactly the frozen 85 features, in schema order')
    X = data.set_index('row_id')
    for c in schema['numeric']:
        X[c] = pd.to_numeric(X[c], errors='raise')
    if not np.isfinite(X[schema['numeric']].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite feature value')

    bundle = joblib.load(model)  # Reached only after all SHA/provenance gates above.
    for name in NAMES:
        np.testing.assert_array_equal(bundle['models'][name].classes_, np.arange(5))
    with threadpool_limits(limits=1):
        cube = np.stack([np.asarray(bundle['models'][n].predict_proba(X), dtype=float)
                         for n in NAMES], axis=1)
    probs = alternatives(cube, bundle['weights'])
    probs['sql_guard_rf_rescue'], reason = combine(probs['weighted_reference'], probs['random_forest'])
    frames = []
    for policy in ['weighted_reference', 'random_forest', 'sql_guard_rf_rescue']:
        p = probs[policy]
        f = pd.DataFrame(p, columns=['p_' + c for c in CLASSES])
        f.insert(0, 'prediction', np.asarray(CLASSES)[p.argmax(1)])
        f.insert(0, 'policy', policy)
        f.insert(0, 'row_id', X.index.to_numpy())
        f['reason_candidate'] = reason
        frames.append(f)
    result = pd.concat(frames, ignore_index=True)
    expected_saved = pd.read_csv(ROOT / 'reports/INDEPENDENT_CAPTURE_20261109/all_predictions_verified.csv',
                                 float_precision='round_trip', low_memory=False).set_index(['row_id', 'policy'])
    actual = result.set_index(['row_id', 'policy'])
    compared = expected_saved.loc[actual.index] if set(actual.index).issubset(set(expected_saved.index)) else None
    max_diff = None
    if compared is not None:
        columns = ['p_' + c for c in CLASSES]
        np.testing.assert_array_equal(actual.prediction, compared.prediction)
        max_diff = float(np.max(np.abs(actual[columns].to_numpy() - compared[columns].to_numpy())))
        np.testing.assert_allclose(actual[columns], compared[columns], rtol=0, atol=1e-12)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, mode='x')
    record = dict(status='PASS_FROZEN_MODEL_REPLAY', inputs=len(X), predictions=len(result),
                  no_fit=True, model_sha256=MODEL_SHA, frozen_catalog_sha256=CATALOG_SHA,
                  features_sha256=sha(features), prediction_sha256=sha(output),
                  saved_labels_identical=(compared is not None), max_probability_difference=max_diff)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open('x', encoding='utf8') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', type=Path, default=ROOT/'reports/INDEPENDENT_CAPTURE_20261109/all_external_features.csv')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    args = ap.parse_args()
    replay(args.features.resolve(), args.output.resolve(), args.audit.resolve())
