"""Linear student KD/margin pilot. No training or deserialization on import."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import ta_hafid_engine as e
from e4 import DomainFeatures


FAMILIES = ['lr_original', 'student_ce', 'student_cost_ce', 'kd', 'margin_kd', 'mped']
CONFIGURATIONS = ['teacher_full', 'teacher_shared', 'lr_balanced'] + FAMILIES


def probabilities(p, n):
    p = np.asarray(p, dtype=float)
    if p.shape != (n, 5) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError('Invalid five-class teacher/student probabilities')
    return p / p.sum(1, keepdims=True)


def loss_gradient(theta, X, y, q, weight, C, alpha, temperature, beta, fraction, cap, risk):
    """Weighted CE + T² KL + teacher-correct pairwise squared hinge + L2.

    Margin weights sum over true class against its four rivals. Risk rescales
    only this term; student_cost_ce controls a simple increase of CE class cost.
    """
    prepared=_prepare_targets(q,y,temperature,beta,fraction,cap)
    return _prepared_loss_gradient(theta,X,y,prepared,weight,C,alpha,temperature,beta,risk)


def _prepare_targets(q,y,temperature,beta,fraction,cap):
    qt_log=np.log(np.clip(q,1e-12,1))/temperature
    qt_log-=logsumexp(qt_log,axis=1,keepdims=True)
    target=None
    correct=None
    if beta:
        logq=np.log(np.clip(q,1e-12,1))
        target=fraction*np.clip(logq[np.arange(len(y)),y,None]-logq,0,cap)
        correct=q.argmax(1)==y
    return qt_log,np.exp(qt_log),target,correct


def _prepared_loss_gradient(theta, X, y, prepared, weight, C, alpha, temperature, beta, risk):
    n, d = X.shape
    qt_log,qt,target,correct=prepared
    W, b = theta[:d*5].reshape(d, 5), theta[d*5:]
    z = np.asarray(X @ W) + b
    lp = z - logsumexp(z, axis=1, keepdims=True)
    p = np.exp(lp)
    norm = weight.sum()
    w = weight / norm
    loss = -(1-alpha) * np.dot(w, lp[np.arange(n), y])
    g = (1-alpha) * p
    g[np.arange(n), y] -= (1-alpha)
    g *= w[:, None]
    if alpha:
        ls = z / temperature
        ls -= logsumexp(ls, axis=1, keepdims=True)
        loss += alpha * temperature**2 * np.dot(w, np.sum(qt*(qt_log-ls), axis=1))
        g += alpha * temperature * w[:, None] * (np.exp(ls)-qt)
    if beta:
        hinge = np.maximum(0, target - (z[np.arange(n), y, None] - z))
        hinge[~correct] = 0
        hinge[np.arange(n), y] = 0
        mw = w * risk
        loss += beta / 4 * np.dot(mw, np.square(hinge).sum(1))
        mg = beta / 2 * mw[:, None] * hinge
        mg[np.arange(n), y] -= mg.sum(1)
        g += mg
    reg = 1 / (C * norm)
    loss += .5 * reg * np.square(W).sum()
    gw = np.asarray(X.T @ g) + reg * W
    grad = np.r_[gw.ravel(), g.sum(0)]
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise ValueError('Nonfinite objective/gradient')
    return float(loss), grad


class LinearStudent(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1., alpha=.5, temperature=2., beta=0., fraction=.5, cap=4.,
                 ddos_weight=1.5, risk_ddos=1., maxiter=2000, ftol=1e-10, gtol=1e-6,
                 max_parameters=1024, maxcor=50):
        self.C=C; self.alpha=alpha; self.temperature=temperature; self.beta=beta
        self.fraction=fraction; self.cap=cap; self.ddos_weight=ddos_weight
        self.risk_ddos=risk_ddos; self.maxiter=maxiter; self.ftol=ftol
        self.gtol=gtol; self.max_parameters=max_parameters
        self.maxcor=maxcor

    def fit(self, X, y, teacher=None, initial=None):
        X = sparse.csr_matrix(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if set(y) != set(range(5)) or X.shape[0] != len(y):
            raise ValueError('Student fit requires aligned five-class data')
        if not (self.C > 0 and 0 <= self.alpha <= 1 and self.temperature > 0 and self.beta >= 0
                and 0 <= self.fraction <= 1 and self.cap > 0 and self.ddos_weight > 0 and self.risk_ddos > 0):
            raise ValueError('Invalid loss hyperparameters')
        nparams = (X.shape[1]+1)*5
        if nparams > self.max_parameters:
            raise ValueError('Student structural parameter budget exceeded')
        if teacher is None:
            if self.alpha or self.beta:
                raise ValueError('KD/margin requires explicit training teacher probabilities')
            q = np.eye(5)[y]
        else:
            q = probabilities(teacher, len(y))
        weight = np.where(y == 1, self.ddos_weight, 1.)
        risk = np.where(y == 1, self.risk_ddos, 1.)
        prepared=_prepare_targets(q,y,self.temperature,self.beta,self.fraction,self.cap)
        args = (X, y, prepared, weight, self.C, self.alpha, self.temperature, self.beta,risk)
        start = np.zeros(nparams) if initial is None else np.asarray(initial, dtype=float)
        if start.shape != (nparams,) or not np.isfinite(start).all():
            raise ValueError('Invalid training-only warm start')
        with threadpool_limits(limits=1):
            result = minimize(_prepared_loss_gradient, start, args=args, jac=True, method='L-BFGS-B',
                options=dict(maxiter=self.maxiter, ftol=self.ftol, gtol=self.gtol, maxls=40,maxcor=self.maxcor))
        if not result.success:
            raise ValueError(f'Student optimizer did not converge: {result.message}; iterations={result.nit}; objective={result.fun}; max_abs_gradient={np.max(np.abs(result.jac))}')
        d = X.shape[1]
        self.coef_ = result.x[:d*5].reshape(d,5).T
        self.intercept_ = result.x[d*5:]
        self.classes_ = np.arange(5)
        self.n_features_in_ = d
        self.optimization_ = dict(success=bool(result.success), iterations=int(result.nit),
            evaluations=int(result.nfev), objective=float(result.fun),
            max_abs_gradient=float(np.max(np.abs(result.jac))), message=str(result.message),
            parameters=nparams, teacher_training_rows=len(y) if teacher is not None else 0)
        return self

    def decision_function(self, X):
        return np.asarray(X @ self.coef_.T) + self.intercept_

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class TeacherPredictor(ClassifierMixin, BaseEstimator):
    """Inference wrapper for this run's newly fitted five-expert teacher."""
    def __init__(self, bundle, shared=False):
        self.bundle=bundle; self.shared=shared; self.classes_=np.arange(5)

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        raise RuntimeError('TeacherPredictor wraps an already fitted run-owned teacher; fit experts explicitly')

    def predict_proba(self, X):
        if not self.shared:
            return e.predict_bundle(self.bundle, X)
        models = self.bundle['models']
        treeprep = models['decision_tree']['preprocessor']
        if any(models[n]['preprocessor'] is not treeprep for n in e.NAMES if n != 'logistic_regression'):
            raise ValueError('Shared teacher requires identical reused tree preprocessing')
        Z = treeprep.transform(X)
        L = models['logistic_regression']['preprocessor'].transform(X)
        probs = [models[n]['estimator'].predict_proba(L if n == 'logistic_regression' else Z) for n in e.NAMES]
        return sum(self.bundle['weights'][n]*p for n,p in zip(e.NAMES, probs))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def raw_pipeline(preprocessor, estimator):
    return Pipeline([('domain', DomainFeatures('dstport29')), ('preprocessor', preprocessor), ('estimator', estimator)])


def teacher_pipeline(bundle, shared=False):
    return Pipeline([('domain', DomainFeatures('dstport29')), ('teacher', TeacherPredictor(bundle, shared))])


def recover_weights(frozen, fold):
    """Recover historical metadata exactly, without labels or score selection.

    Old caches contain predictions but no weights/model objects. Full column rank
    makes the original mixture weights identifiable from the cached mixture.
    """
    cache = frozen.load(f'dstport29/cv_cache/fold{fold}_balanced.joblib')
    if cache['signature'] != frozen.protocol['signature']:
        raise ValueError('Historical cache signature mismatch')
    base = {r['model']: r['probabilities'] for r in cache['base'] if r['variant'] == 'ddos_tuned'}
    matrix = np.stack([base[n] for n in e.NAMES], axis=-1).reshape(-1, 5)
    target = cache['preds']['ddos_tuned__balanced__weighted_voting'].reshape(-1)
    w, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    error = float(np.max(np.abs(matrix @ w-target)))
    condition = float(singular[0]/singular[-1])
    if rank != 5 or condition > 1e6 or error > 1e-10 or (w < 0).any() or not np.isclose(w.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError('Historical weights cannot be uniquely/exactly recovered')
    return dict(zip(e.NAMES, w)), dict(fold=fold, rank=int(rank), condition=condition,
        max_reconstruction_error=error, used_labels=False, new_performance_optimization=False)


def guard_roles(ids, fit, calibration, evaluation, holdout, groups=None):
    ids=np.asarray(ids)
    parts=[np.asarray(v) for v in [fit,calibration,evaluation]]
    if len(set(ids)) != len(ids) or set(ids) & set(holdout):
        raise ValueError('Development/holdout leakage or duplicate IDs')
    if any(len(p) != len(set(p)) for p in parts):
        raise ValueError('Duplicate role positions')
    if any(set(a)&set(b) for a,b in [(parts[0],parts[1]),(parts[0],parts[2]),(parts[1],parts[2])]):
        raise ValueError('Teacher/student fit, calibration and evaluation overlap')
    if set(np.concatenate(parts)) != set(range(len(ids))):
        raise ValueError('Role partition coverage mismatch')
    if groups is not None:
        if set(groups[np.r_[fit,calibration]]) & set(groups[evaluation]):
            raise ValueError('Outer grouped signature leakage')


def diagnostics(y, p, teacher):
    from e4 import metrics
    from sklearn.metrics import accuracy_score, roc_auc_score
    p=probabilities(p,len(y));teacher=probabilities(teacher,len(y))
    pred,ref=p.argmax(1),teacher.argmax(1)
    out=metrics(y,pred)
    out.update(accuracy=float(accuracy_score(y,pred)),auc_ovr_macro=float(roc_auc_score(y,p,multi_class='ovr',labels=np.arange(5))),
        teacher_agreement=float((pred==ref).mean()),teacher_correct_student_wrong=int(((ref==y)&(pred!=y)).sum()),
        teacher_wrong_student_correct=int(((ref!=y)&(pred==y)).sum()),
        ddos_negative_flips=int(((y==1)&(ref==y)&(pred!=y)).sum()),
        ddos_positive_flips=int(((y==1)&(ref!=y)&(pred==y)).sum()))
    return out
