import numpy as np

from scipy.spatial.distance import pdist

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .homology import diagrams, drop_inf
from .magnitude import magnitude
from utils.math import beta1, beta1_intercept


def _dist(i, j, d, m):
    # dist(u=X[i], v=X[j]) = d[m * i + j - ((i + 2) * (i + 1)) // 2]
    return d[m * i + j - ((i + 2) * (i + 1)) // 2]


def _capacity_alg(d: np.ndarray, R: float, m):
    c = []
    ok = True
    for i in range(m):
        for j in c:
            ok = ok and (_dist(j, i, d, m) >= R)
        if ok:
            c.append(i)
    return len(c)


def capacity(X: np.ndarray):  # FIXME
    d = pdist(X, metric='euclidean')
    d.sort()
    m = np.zeros_like(d)
    for i, t in enumerate(d):
        m[i] = _capacity_alg(d, t, X.shape[0])
    return beta1(np.log(d), np.log(m))


def corr(X: np.ndarray, n_steps: int = 30):
    d = pdist(X, metric='euclidean')
    d.sort()
    u = np.linspace(d.min(), d.max(), n_steps)
    c = ((d.reshape(-1, 1) <= u).sum(axis=1)) / X.shape[0] / (X.shape[0] - 1)
    return beta1(np.log(d), np.log(c))


def mle(X: np.ndarray, k: int = 5):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors()

    return (k - 1) / np.log(np.expand_dims(distances[:, -1], 1) / distances).sum(axis=-1)


def mm(X: np.ndarray, k: int = 5):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors()

    Tk = distances[:, -1]
    T = distances.mean(axis=1)
    return T / (Tk - T)


def ols(X: np.ndarray, k: int = 5, slope_estimator=LinearRegression):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors()

    y = np.zeros_like(distances)
    for i in range(k):
        y[:, i] = distances[:, i] * (distances == distances[:, i].reshape(-1, 1)).sum(axis=-1) / (i + 1)

    d = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        lr = slope_estimator(fit_intercept=False).fit(distances[i].reshape(-1, 1), y[i].reshape(-1, 1))
        d[i] = lr.coef_[0, 0]

    return d


def pca(X: np.ndarray, explained_variance: float = 0.95):
    return PCA(n_components=explained_variance).fit(X).n_components_


def cluster_pca(X: np.ndarray, k: int = 5, explained_variance: float = 0.95):
    labels = KMeans(n_clusters=int(X.shape[0] / k)).fit(X)
    return np.array([
        PCA(n_components=explained_variance).fit(X[labels == l]).n_components_ for l in range(int(X.shape[0] / k))
    ])


def local_pca(X: np.ndarray, k: int = 5, explained_variance: float = 0.95):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, ix = nn.kneighbors()
    return np.array([pca(X[ix[i]], explained_variance) for i in range(X.shape[0])])


def two_nn(X: np.ndarray):
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    distances, _ = nn.kneighbors()
    return beta1(np.log(np.sort(distances[:, 1] / distances[:, 0])), -np.log(1 - np.linspace(0, 1 - 1/n, n)))


def persistence(X: np.ndarray, p: float = 1):
    n = np.arange(1, X.shape[0] + 1, X.shape[0] // 10)
    e = np.zeros(n.size)
    for i, ni in enumerate(n):
        dgms = drop_inf(diagrams(X[:ni])[0])
        e[i] = np.power(dgms[1] - dgms[0], p).sum()
    m = beta1(np.log(n), np.log(e))
    return p / (1 - m)


def magnitude_reg(X: np.ndarray, t: np.ndarray, i: int = None, j: int = None):
    m = np.zeros_like(t)
    for i in range(t.shape[0]):
        m[i] = magnitude(t[i] * X)
    i, j = i or 0, j or t.shape[0]
    return beta1_intercept(np.log(m[i:j]), np.log(t[i:j]))
