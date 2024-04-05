import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression


def corr_dim(X: np.array):
    # X \in M(n, d)
    X = np.unique(X, axis=-2)
    d = pdist(X, metric='euclidean')
    d.sort()
    c = np.zeros_like(d)
    for i, t in enumerate(d):
        c[i] = ((d <= t).sum()) / X.shape[0] / (X.shape[0] - 1)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(d).reshape(-1, 1), np.log(c).reshape(-1, 1))
    return lr.coef_[0, 0]
