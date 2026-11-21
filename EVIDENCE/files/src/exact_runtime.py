"""Execute this project's fitted preprocessing contract without refitting it.

CSR zeros remain absent (especially important for XGBoost). This is an
implementation optimization, not a new classifier or a general sklearn compiler.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
import ta_hafid_engine as e
from wb_scheduler import predict_lazy


@dataclass
class FrozenEncoding:
    columns: tuple
    numeric: tuple
    categorical: tuple
    medians: np.ndarray
    scale: np.ndarray
    categories: tuple
    categorical_fill: str
    sparse_output: bool

    @classmethod
    def from_models(cls, models):
        tree=models['decision_tree']['preprocessor']
        linear=models['logistic_regression']['preprocessor']
        if tree.remainder!='drop' or linear.remainder!='drop':
            raise ValueError('Unsupported remainder')
        for name in e.NAMES[:-1]:
            if models[name]['preprocessor'] is not tree:
                raise ValueError('Expected shared tree preprocessing')
        tn=tree.named_transformers_['numeric'];ln=linear.named_transformers_['numeric']
        num=tuple(tree.transformers_[0][2]);cat=tuple(tree.transformers_[1][2])
        if tuple(linear.transformers_[0][2])!=num or tuple(linear.transformers_[1][2])!=cat:
            raise ValueError('Column mismatch')
        if (tuple(tn.named_steps)!=('imputer',) or tuple(ln.named_steps)!=('imputer','scaler')
                or not tn['imputer'].keep_empty_features or tn['imputer'].strategy!='median'
                or ln['imputer'].strategy!='median' or not ln['imputer'].keep_empty_features
                or ln['scaler'].with_mean or not ln['scaler'].with_std):
            raise ValueError('Unsupported numeric preprocessing')
        np.testing.assert_array_equal(tn['imputer'].statistics_,ln['imputer'].statistics_)
        if bool(tree.sparse_output_)!=bool(linear.sparse_output_):raise ValueError('Sparse mismatch')
        categories=();fill='__MISSING__'
        if cat:
            tc=tree.named_transformers_['categorical'];lc=linear.named_transformers_['categorical']
            for block in (tc,lc):
                enc=block['onehot'];imp=block['imputer']
                if (tuple(block.named_steps)!=('imputer','onehot') or imp.strategy!='constant'
                        or imp.add_indicator or imp.fill_value!=fill or enc.handle_unknown!='ignore'
                        or enc.drop is not None or enc.min_frequency is not None
                        or enc.max_categories is not None or not enc.sparse_output
                        or enc.dtype!=np.float64):raise ValueError('Unsupported categorical preprocessing')
            categories=tuple(np.array(v,copy=True) for v in tc['onehot'].categories_)
            for a,b in zip(categories,lc['onehot'].categories_):np.testing.assert_array_equal(a,b)
            if any(any(not isinstance(v,str) for v in a) for a in categories):
                raise ValueError('This runtime supports string categories only')
        return cls(tuple(tree.feature_names_in_),num,cat,
                   np.array(tn['imputer'].statistics_,copy=True),np.array(ln['scaler'].scale_,copy=True),
                   categories,fill,bool(tree.sparse_output_))

    def tree(self, X):
        if tuple(X.columns)!=self.columns:raise ValueError('Explicit input column order required')
        numeric=X.loc[:,list(self.numeric)].to_numpy(dtype=np.float64,copy=True)
        if np.isinf(numeric).any():raise ValueError('Infinite input')
        mask=np.isnan(numeric)
        if mask.any():numeric[mask]=np.broadcast_to(self.medians,numeric.shape)[mask]
        if not np.isfinite(numeric).all():raise ValueError('Nonfinite imputation')
        if not self.categorical:return sparse.csr_matrix(numeric) if self.sparse_output else numeric
        values=X.loc[:,list(self.categorical)].to_numpy(dtype=object)
        rows=[];cols=[];offset=0
        for j,categories in enumerate(self.categories):
            col=values[:,j].copy()
            # Match SimpleImputer(missing_values=np.nan), not arbitrary invalid values.
            missing=np.array([isinstance(v,(float,np.floating)) and np.isnan(v) for v in col])
            col[missing]=self.categorical_fill
            if any(not isinstance(v,str) for v in col):raise ValueError('Expected strings or NaN')
            table={value:i for i,value in enumerate(categories)}
            encoded=np.fromiter((table.get(v,-1) for v in col),dtype=np.int64,count=len(col))
            valid=encoded>=0;rows.append(np.flatnonzero(valid));cols.append(encoded[valid]+offset)
            offset+=len(categories)
        rr=np.concatenate(rows);cc=np.concatenate(cols)
        cats=sparse.csr_matrix((np.ones(len(rr),dtype=np.float64),(rr,cc)),shape=(len(X),offset))
        if self.sparse_output:return sparse.hstack((sparse.csr_matrix(numeric),cats),format='csr')
        return np.hstack((numeric,cats.toarray()))

    def linear(self, Z):
        out=Z.copy();n=len(self.numeric)
        if sparse.issparse(out):
            is_numeric=out.indices<n
            out.data[is_numeric]/=self.scale[out.indices[is_numeric]]
        else:out[:,:n]/=self.scale
        return out


class ExactRuntime:
    def __init__(self, encoding, estimators, weights, order, policy='certified'):
        if policy not in ('shared','certified'):raise ValueError('Unsupported policy')
        if set(estimators)!=set(e.NAMES):raise ValueError('Five experts required')
        for est in estimators.values():
            np.testing.assert_array_equal(est.classes_,np.arange(5))
        self.encoding=encoding;self.estimators=estimators
        self.weights=np.asarray(weights,dtype=np.float64);self.order=list(order);self.policy=policy

    @classmethod
    def from_bundle(cls,bundle,policy='certified'):
        return cls(FrozenEncoding.from_models(bundle['models']),
                   {n:bundle['models'][n]['estimator'] for n in e.NAMES},bundle['weights'],bundle['order'],policy)

    def expert_cube(self,X):
        Z=self.encoding.tree(X);L=self.encoding.linear(Z)
        return np.stack([np.asarray(self.estimators[n].predict_proba(L if n=='logistic_regression' else Z),
                                  dtype=np.float64) for n in e.NAMES],axis=1)

    def predict_proba(self,X):
        p=self.expert_cube(X)
        return sum(float(self.weights[j])*p[:,j] for j in range(5))

    def predict(self,X):
        if self.policy=='shared':return self.predict_proba(X).argmax(1)
        matrices={}
        def callback(j):
            name=e.NAMES[j];mode='linear' if name=='logistic_regression' else 'tree'
            def call(active):
                if 'tree' not in matrices:matrices['tree']=self.encoding.tree(X)
                if mode=='linear' and 'linear' not in matrices:
                    matrices['linear']=self.encoding.linear(matrices['tree'])
                return np.asarray(self.estimators[name].predict_proba(matrices[mode][active]),dtype=np.float64)
            return call
        return predict_lazy(len(X),5,self.weights,self.order,[callback(j) for j in range(5)]).labels
