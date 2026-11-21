import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
import ta_hafid_engine as e
from exact_runtime import FrozenEncoding,ExactRuntime
from wb_research import cube,mixture,WBPredictor


def fitted(categorical=True):
    X=pd.DataFrame({'a':[0.,1.,np.nan,3.,4.,0.,6.,7.,8.,9.],
                    'empty':[np.nan]*10,'constant':[2.]*10})
    cats=[]
    if categorical:
        X['protocol']=['TCP','UDP',np.nan,'TCP','UDP']*2;cats=['protocol']
    num=['a','empty','constant'];preps={}
    for linear in [False,True]:
        steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
        if linear:steps.append(('scaler',StandardScaler(with_mean=False)))
        prep=ColumnTransformer([('numeric',Pipeline(steps),num),('categorical',Pipeline([
            ('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__')),
            ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cats)],
            sparse_threshold=1.,remainder='drop',verbose_feature_names_out=True)
        prep.fit(X);preps[linear]=prep
    models={}
    for name in e.NAMES:
        prep=preps[name=='logistic_regression']
        est=DecisionTreeClassifier(random_state=42,max_depth=4).fit(prep.transform(X),np.tile(np.arange(5),2))
        models[name]=Pipeline([('preprocessor',prep),('estimator',est)])
    return X,models


@pytest.mark.parametrize('categorical',[True,False])
def test_exact_encoding_zero_nan_unknown_empty_and_prediction(categorical):
    X,models=fitted(categorical);encoding=FrozenEncoding.from_models(models)
    X=X.iloc[[0,2,4,9]].copy()
    if categorical:X.loc[X.index[0],'protocol']='unseen'
    actual=encoding.tree(X);wanted=models['decision_tree']['preprocessor'].transform(X)
    assert sparse.issparse(actual)==sparse.issparse(wanted)
    for a,b in [(actual,wanted),(encoding.linear(actual),models['logistic_regression']['preprocessor'].transform(X))]:
        if sparse.issparse(a):
            np.testing.assert_array_equal(a.indptr,b.indptr)
            np.testing.assert_array_equal(a.indices,b.indices)
            np.testing.assert_array_equal(a.data,b.data)
            assert a.nnz < np.prod(a.shape) # zeros are not silently made explicit
        else:np.testing.assert_array_equal(a,b)
    bundle={'models':models,'weights':np.array([.11,.12,.13,.29,.35]),'order':[4,3,2,0,1]}
    for policy in ['shared','certified']:
        fast=ExactRuntime.from_bundle(bundle,policy)
        np.testing.assert_array_equal(fast.expert_cube(X),cube(models,X))
        np.testing.assert_array_equal(fast.predict_proba(X),mixture(cube(models,X),bundle['weights']))
        np.testing.assert_array_equal(fast.predict(X),WBPredictor(models,bundle['weights'],bundle['order'],policy).predict(X))
        for i in range(len(X)):
            np.testing.assert_array_equal(fast.predict(X.iloc[[i]]),fast.predict(X)[i:i+1])


def test_runtime_rejects_changed_input_contract():
    X,models=fitted();enc=FrozenEncoding.from_models(models)
    with pytest.raises(ValueError,match='column'):enc.tree(X[list(reversed(X.columns))])
    bad=X.copy();bad.loc[0,'a']=np.inf
    with pytest.raises(ValueError,match='Infinite'):enc.tree(bad)
    bad=X.copy();bad.loc[0,'protocol']=17
    with pytest.raises(ValueError,match='strings'):enc.tree(bad)
    models['logistic_regression']['preprocessor'].named_transformers_['numeric']['scaler'].with_mean=True
    with pytest.raises(ValueError,match='numeric'):FrozenEncoding.from_models(models)
