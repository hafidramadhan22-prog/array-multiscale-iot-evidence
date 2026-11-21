"""LIFT-ID extension. Historical engine stays unchanged; no implicit training on import."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import ta_hafid_engine as e
LEGACY_PREDICT = e.predict_bundle
import hashlib
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def probability(p, rows):
    p = np.asarray(p, dtype=float)
    if p.shape != (rows, 5) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid five-class probabilities')
    # Match the tolerance of the frozen engine (XGBoost stores float32 probabilities).
    if not np.allclose(p.sum(1), 1, rtol=0, atol=1e-5):
        raise ValueError('Probability rows do not sum to one')
    return p


class FrozenRun:
    """Only deserialize run-owned artifacts after recorded checksum verification."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.catalog = {r['path']: r for r in json.loads((self.root/'artifacts_sha256.json').read_text())}
        complete = json.loads(self.checked('run_complete.json').read_text())
        if complete['status'] != 'COMPLETE_CANONICAL_DATASET':
            raise ValueError('Need completed canonical FULL run')
        self.mode = json.loads(self.checked('feature_mode_freeze.json').read_text())['selected_feature_mode']
        self.selected = self.root/self.mode
        self.protocol = json.loads(self.checked(f'{self.mode}/manifests/protocol.json').read_text())
        self.loads = []

    def checked(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root) or relative not in self.catalog:
            raise ValueError('Uncatalogued or external artifact')
        record = self.catalog[relative]
        if not path.is_file() or path.stat().st_size != record['bytes'] or e.sha256(path) != record['sha256']:
            raise ValueError('Artifact checksum mismatch: '+relative)
        return path

    def load(self, relative):
        path = self.checked(relative)
        value = joblib.load(path)
        self.loads.append(dict(path=relative, sha256=self.catalog[relative]['sha256']))
        return value


def recover_oof(frozen):
    mode = frozen.mode
    assignment = pd.read_csv(frozen.checked(f'{mode}/tables/training_fold_assignment.csv'))
    if assignment.original_row_id.duplicated().any() or len(assignment) != 52684:
        raise ValueError('Invalid development assignment')
    ids = assignment.original_row_id.to_numpy()
    y = assignment.label.to_numpy()
    with np.load(frozen.checked('canonical_split_indices.npz'), allow_pickle=False) as split:
        if not np.array_equal(ids, split['training_row_ids']) or set(ids) & set(split['holdout_row_ids']):
            raise ValueError('Development/holdout identity mismatch')
    result = {}
    for case in e.CASES:
        seen = np.zeros(len(ids), int)
        for fold in range(1, 6):
            rel = f'{mode}/cv_cache/fold{fold}_{case}.joblib'
            value = frozen.load(rel)
            if value['signature'] != frozen.protocol['signature']:
                raise ValueError('Cache protocol signature mismatch')
            with np.load(frozen.checked(f'{mode}/tables/indices_outer{fold}_{case}.npz'), allow_pickle=False) as split:
                validation = split['outer_validation_row_ids']
                train = split['fit_train_row_ids']
                if set(train) & set(validation) or set(train) | set(validation) != set(ids):
                    raise ValueError('Outer split overlap or coverage mismatch')
            positions = pd.Index(ids).get_indexer(validation)
            if (positions < 0).any() or len(set(positions)) != len(positions):
                raise ValueError('Invalid OOF row mapping')
            if not np.array_equal(validation, ids[assignment.outer_validation_fold == fold]):
                raise ValueError('OOF cache order not consistent with saved split')
            entries = {(r['variant'], r['case'], r['model']): r for r in value['base']}
            expected = {(v, case, n) for v in e.VARIANTS for n in e.NAMES}
            if set(entries) != expected or len(entries) != len(value['base']):
                raise ValueError('Base model/variant identities invalid')
            for (variant, _, name), row in entries.items():
                key = f'{case}__{variant}__{name}'
                result.setdefault(key, np.empty((len(ids), 5)))[positions] = probability(row['probabilities'], len(positions))
            seen[positions] += 1
        if not np.all(seen == 1):
            raise ValueError('Every development row must be OOF exactly once')
    return assignment, result


def complementarity(y, dt, lr):
    y = np.asarray(y)
    dt, lr = probability(dt, len(y)), probability(lr, len(y))
    d, l = dt.argmax(1) == y, lr.argmax(1) == y
    n11, n10, n01, n00 = [int(np.sum(mask)) for mask in (d & l, d & ~l, ~d & l, ~d & ~l)]
    n = len(y)
    denominator = n11*n00+n01*n10
    return dict(rows=n, n11=n11, n10=n10, n01=n01, n00=n00,
                correctness_disagreement=(n01+n10)/n if n else None,
                label_disagreement=float(np.mean(dt.argmax(1) != lr.argmax(1))) if n else None,
                double_fault=n00/n if n else None,
                q_statistic=(n11*n00-n01*n10)/denominator if denominator else None,
                dt_error_rescue_fraction=n01/(n01+n00) if n01+n00 else None,
                hard_selection_oracle_accuracy=1-n00/n if n else None,
                oracle_scope='upper bound for selecting DT/LR labels, not general probability fusion')


def e1_tables(assignment, predictions):
    rows, scores = [], []
    y = assignment.label.to_numpy()
    for case in e.CASES:
        for variant in e.VARIANTS:
            dt = predictions[f'{case}__{variant}__decision_tree']
            lr = predictions[f'{case}__{variant}__logistic_regression']
            for fold in [0, 1, 2, 3, 4, 5]:
                mask = np.ones(len(y), bool) if fold == 0 else assignment.outer_validation_fold.to_numpy() == fold
                for cl in [-1, 0, 1, 2, 3, 4]:
                    subset = mask & (np.ones(len(y), bool) if cl == -1 else y == cl)
                    rows.append(dict(case=case, variant=variant, fold=fold, class_id=cl,
                                     **complementarity(y[subset], dt[subset], lr[subset])))
                for name in e.NAMES:
                    p = predictions[f'{case}__{variant}__{name}'][mask]
                    scores.append(dict(case=case, variant=variant, fold=fold, model=name, **e.metrics(y[mask], p)))
    return pd.DataFrame(rows), pd.DataFrame(scores)


def guard_roles(fit_ids, calibration_ids, evaluation_ids, development_ids, holdout_ids):
    groups = [np.asarray(x) for x in (fit_ids, calibration_ids, evaluation_ids)]
    sets = [set(x) for x in groups]
    if any(len(x) != len(s) or not len(x) for x, s in zip(groups, sets)):
        raise ValueError('Empty or duplicate partition identities')
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Fit/calibration/evaluation overlap')
    if any(s & set(holdout_ids) or not s <= set(development_ids) for s in sets):
        raise ValueError('Holdout or outside-development identity in learning procedure')


def tree_identity(dt):
    # Structural identity survives joblib round trips; pickle bytes need not.
    model = dt.named_steps['estimator']
    h = hashlib.sha256()
    for array in [model.classes_, model.tree_.children_left, model.tree_.children_right,
                  model.tree_.feature, model.tree_.threshold, model.tree_.value]:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode());h.update(a.tobytes())
    prep = dt.named_steps['preprocessor']
    if hasattr(prep, 'get_feature_names_out'):
        h.update(json.dumps(prep.get_feature_names_out().tolist()).encode())
    if hasattr(prep, 'named_transformers_'):
        for name, transform in prep.named_transformers_.items():
            if not hasattr(transform, 'steps'):continue
            for step, fitted in transform.steps:
                for attribute in ['statistics_','scale_','mean_','var_','categories_']:
                    if hasattr(fitted, attribute):
                        value=getattr(fitted,attribute)
                        value=[np.asarray(x).tolist() for x in value] if attribute=='categories_' else np.asarray(value).tolist()
                        h.update(json.dumps([name,step,attribute,value],sort_keys=True).encode())
    return h.hexdigest()


def learn_policy(leaves, y, dt, lr, regularization, tree_id, node_count, global_weight=False):
    leaves, y = np.asarray(leaves), np.asarray(y)
    dt, lr = probability(dt, len(y)), probability(lr, len(y))
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError('Invalid regularization')
    if leaves.shape != y.shape or not np.issubdtype(leaves.dtype, np.integer) or (leaves < 0).any() or (leaves >= node_count).any():
        raise ValueError('Invalid leaf identities')
    if not np.isin(y, np.arange(5)).all():
        raise ValueError('Invalid labels')
    d, residual = lr-dt, np.eye(5)[y]-dt
    a = np.einsum('ij,ij->i', d, residual)
    b = np.einsum('ij,ij->i', d, d)
    weights = np.zeros(node_count)
    support = np.bincount(leaves, minlength=node_count)
    groups = [np.arange(len(y))] if global_weight else [np.flatnonzero(leaves == leaf) for leaf in np.unique(leaves)]
    for group in groups:
        denominator = float(b[group].sum()+regularization)
        weight = float(np.clip(a[group].sum()/denominator, 0, 1)) if denominator > 0 else 0.
        if global_weight:
            weights[:] = weight
        else:
            weights[leaves[group[0]]] = weight
    return dict(weights=weights, support=support, tree_id=tree_id, regularization=float(regularization), global_weight=global_weight)


def policy_weights(policy, leaves, tree_id):
    if policy['tree_id'] != tree_id:
        raise ValueError('Leaf policy belongs to a different tree')
    leaves = np.asarray(leaves)
    if (leaves < 0).any() or (leaves >= len(policy['weights'])).any():
        raise ValueError('Leaf index out of bounds')
    return np.asarray(policy['weights'])[leaves]


def tree_predictions(dt, X):
    z = dt.named_steps['preprocessor'].transform(X)
    model = dt.named_steps['estimator']
    return model.predict_proba(z), model.apply(z)


def validate_model(bundle):
    if bundle.get('format') == 'lift_fusion_v1':
        if tree_identity(bundle['dt']) != bundle['tree_id'] or bundle['policy']['tree_id'] != bundle['tree_id']:
            raise ValueError('Tree changed after policy fitting')
        if not np.array_equal(bundle['dt'].classes_, np.arange(5)) or not np.array_equal(bundle['lr'].classes_, np.arange(5)):
            raise ValueError('Unexpected class order')


def predict_model(bundle, X, conditional=True):
    kind = bundle.get('format')
    if kind == 'lift_single_v1':
        return bundle['pipeline'].predict_proba(X)
    if kind == 'lift_fusion_v1':
        dt, leaves = tree_predictions(bundle['dt'], X)
        weights = policy_weights(bundle['policy'], leaves, bundle['tree_id'])
        positions = np.flatnonzero(weights > 0) if conditional else np.arange(len(X))
        result = np.array(dt, copy=True)
        if len(positions):
            lr = bundle['lr'].predict_proba(X.iloc[positions])
            w = weights[positions, None]
            result[positions] = (1-w)*dt[positions]+w*lr
        return result
    return LEGACY_PREDICT(bundle, X)


def tune_pair(X, y, case, cfg, excluded_ids, output, tag):
    """Only DT/LR are fitted. Inner validation and all calibration/evaluation rows remain original."""
    e.guard_training(X, cfg, excluded_ids, 'training')
    splits = list(StratifiedKFold(cfg['inner_splits'], shuffle=True, random_state=cfg['seed']+17).split(X, y))
    rows, plans, models, selected = [], [], {}, {}
    indices = {'fit_ids':X.index.to_numpy(), 'excluded_ids':np.asarray(excluded_ids)}
    for f, (tr, va) in enumerate(splits, 1):
        indices[f'inner{f}_fit_ids'] = X.index.to_numpy()[tr]
        indices[f'inner{f}_validation_ids'] = X.index.to_numpy()[va]
    np.savez_compressed(Path(output)/f'{tag}_{case}_inner_ids.npz', **indices)
    for name, mode in [('decision_tree','tree'), ('logistic_regression','linear')]:
        cache = []
        for fold, (tr, va) in enumerate(splits, 1):
            prep, z, yy, plan = e.prepared(X.iloc[tr], y[tr], case, mode, cfg, partition='training',
                                         validation_ids=np.concatenate([X.index.to_numpy()[va], excluded_ids]))
            plans.append(dict(tag=tag, model=name, inner_fold=fold, **plan))
            cache.append((va, z, yy, prep.transform(X.iloc[va])))
        candidates = e.candidates(name, cfg['smoke_test'])
        scores = []
        for ci, params in enumerate(candidates):
            results = []
            for va, z, yy, zv in cache:
                est = e.fit_est(name, params, z, yy, cfg)
                results.append(e.metrics(y[va], est.predict_proba(zv)))
            average = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
            average['tuning_score'] = e.tuning_score(average, cfg)
            scores.append(average)
            rows.append(dict(tag=tag, case=case, model=name, candidate=ci, params=json.dumps(params, sort_keys=True), **average))
        baseline_count = 1 if cfg['smoke_test'] else 2
        baseline = max(range(baseline_count), key=lambda i:scores[i]['macro_f1'])
        eligible = [i for i, s in enumerate(scores) if s['macro_f1'] >= scores[baseline]['macro_f1']-cfg['max_macro_f1_drop_from_legacy']]
        choice = max(eligible, key=lambda i:(scores[i]['tuning_score'], scores[i]['macro_f1'], -i))
        prep, z, yy, plan = e.prepared(X, y, case, mode, cfg, partition='training', validation_ids=excluded_ids)
        plans.append(dict(tag=tag, model=name, inner_fold=None, **plan))
        models[name] = Pipeline([('preprocessor',prep),('estimator',e.fit_est(name,candidates[choice],z,yy,cfg))])
        selected[name] = dict(candidate=choice, params=candidates[choice], validation_metrics=scores[choice], baseline_candidate=baseline)
        print('PAIR FIT', tag, case, name, choice, flush=True)
    pd.DataFrame(rows).to_csv(Path(output)/f'{tag}_{case}_tuning.csv', index=False)
    e.write_json(Path(output)/f'{tag}_{case}_training.json', dict(selected=selected, plans=plans))
    return models


def fit_policies(dt, lr, Xc, yc, cfg, output, tag, case, fit_ids):
    ids=set(Xc.index)
    if len(ids)!=len(Xc) or ids & set(fit_ids) or ids & set(cfg['_holdout_ids']) or not ids <= set(cfg['_training_ids']):
        raise ValueError('Calibration identity overlaps fitting/holdout or outside development')
    pc, leaves = tree_predictions(dt, Xc)
    lc = lr.predict_proba(Xc)
    identity = tree_identity(dt)
    node_count = dt.named_steps['estimator'].tree_.node_count
    rows, choices = [], {}
    splits = list(StratifiedKFold(cfg['inner_splits'], shuffle=True, random_state=cfg['seed']+211).split(Xc, yc))
    ids = {}
    for f, (tr, va) in enumerate(splits, 1):
        ids[f'policy{f}_fit_ids'] = Xc.index.to_numpy()[tr]
        ids[f'policy{f}_validation_ids'] = Xc.index.to_numpy()[va]
    np.savez_compressed(Path(output)/f'{tag}_{case}_policy_ids.npz', **ids)
    for global_weight in [False, True]:
        candidate_scores = []
        for penalty in cfg['lambda_grid']:
            results = []
            for tr, va in splits:
                policy = learn_policy(leaves[tr], yc[tr], pc[tr], lc[tr], penalty, identity, node_count, global_weight)
                weights = policy_weights(policy, leaves[va], identity)[:,None]
                results.append(e.metrics(yc[va], (1-weights)*pc[va]+weights*lc[va]))
            m = {k:float(np.mean([r[k] for r in results])) for k in results[0]}
            score = e.tuning_score(m, cfg)
            candidate_scores.append((score, float(penalty)))
            rows.append(dict(tag=tag, case=case, global_weight=global_weight, regularization=penalty, tuning_score=score, **m))
        # Deterministic tie: choose stronger regularization only at equal validation score.
        best_score = max(s for s, _ in candidate_scores)
        choice = max(p for s, p in candidate_scores if s >= best_score-1e-12)
        choices[global_weight] = learn_policy(leaves, yc, pc, lc, choice, identity, node_count, global_weight)
    fixed = dict(weights=np.full(node_count, .5), support=np.bincount(leaves,minlength=node_count), tree_id=identity, regularization=None, global_weight=True)
    policies = dict(fixed_fusion=fixed, global_learned=choices[True], leaf_unregularized=learn_policy(leaves,yc,pc,lc,0,identity,node_count), lift_id=choices[False])
    bundles = {'dt':dict(format='lift_single_v1',pipeline=dt), 'lr':dict(format='lift_single_v1',pipeline=lr)}
    for method, policy in policies.items():
        bundles[method] = dict(format='lift_fusion_v1',dt=dt,lr=lr,tree_id=identity,policy=policy)
    pd.DataFrame(rows).to_csv(Path(output)/f'{tag}_{case}_policy_selection.csv', index=False)
    e.write_json(Path(output)/f'{tag}_{case}_policy_freeze.json', dict(tree_id=identity, policies=policies,
                 calibration_ids=Xc.index.to_numpy(), use_holdout=False, rule='sum multiclass Brier + lambda*a^2; clip scalar to [0,1]'))
    np.savez_compressed(Path(output)/f'{tag}_{case}_calibration_evidence.npz',row_ids=Xc.index.to_numpy(),labels=yc,leaf_ids=leaves,dt=pc,lr=lc)
    return bundles, (pc, lc, leaves)


def leaf_diagnostics(y, dt, lr, leaves, policy, tree_id, predicted):
    dt_ok, lr_ok, lift_ok = dt.argmax(1)==y, lr.argmax(1)==y, predicted.argmax(1)==y
    rows=[]
    for leaf in np.unique(leaves):
        mask=leaves==leaf
        rows.append(dict(leaf=int(leaf), rows=int(mask.sum()), weight=float(policy_weights(policy,np.array([leaf]),tree_id)[0]),
                         calibration_support=int(policy['support'][leaf]),
                         lr_can_rescue=int(np.sum(mask & ~dt_ok & lr_ok)), lr_can_damage=int(np.sum(mask & dt_ok & ~lr_ok)),
                         actual_corrected=int(np.sum(mask & ~dt_ok & lift_ok)), actual_damaged=int(np.sum(mask & dt_ok & ~lift_ok))))
    return pd.DataFrame(rows)
