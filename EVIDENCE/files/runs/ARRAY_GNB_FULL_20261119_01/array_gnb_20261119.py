"""Dense adapter for saved GaussianNB pipelines."""
import numpy as np
from scipy import sparse

def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)
