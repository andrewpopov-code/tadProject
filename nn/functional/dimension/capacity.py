import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression


def dist(i, j, d, m):
    # dist(u=X[i], v=X[j]) = d[m * i + j - ((i + 2) * (i + 1)) // 2]
    return d[m * i + j - ((i + 2) * (i + 1)) // 2]


def alg(d: np.array, R: float, m):
    c = []
    ok = True
    for i in range(m):
        for j in c:
            ok = ok and (dist(j, i, d, m) >= R)
        if ok:
            c.append(i)
    return len(c)


def capacity(X: np.array):
    X = np.unique(X, axis=-2)
    d = pdist(X, metric='euclidean')
    d.sort()
    m = np.zeros_like(d)
    for i, t in enumerate(d):
        m[i] = alg(d, t, X.shape[0])
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(d).reshape(-1, 1), np.log(m).reshape(-1, 1))
    return lr.coef_[0, 0]
