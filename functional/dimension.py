import numpy as np

from scipy.spatial.distance import pdist

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .homology import diagrams, drop_inf
from .information import entropy
from utils.math import beta1, beta1_intercept, magnitude


def information(X: np.ndarray):
    X = (X - X.mean(axis=-2)) / (X.max(axis=-2) - X.min(axis=-2))
    d = pdist(X)
    d.sort()
    s = np.zeros_like(d)
    for i in range(d.shape[0]):
        flat = np.histogramdd(X, bins=int(np.ceil(2 / d[i])))[0].reshape(-1)
        s[i] = -entropy(flat[flat > 0] / flat.sum(), np.log2)
    return np.abs(beta1(np.log2(d), s))


def information_renyi(X: np.ndarray, q: float):
    X = (X - X.mean(axis=-2)) / (X.max(axis=-2) - X.min(axis=-2))
    d = pdist(X)
    d.sort()
    s = np.zeros_like(d)
    for i in range(d.shape[0]):
        flat = np.histogramdd(X, bins=int(np.ceil(2 / d[i])))[0].reshape(-1)
        s[i] = np.log(np.power(flat[flat > 0] / flat.sum(), q).sum())
    return np.abs(beta1(np.log(d), s) / (q - 1))


def corrq(X: np.ndarray, q: float):
    X = (X - X.mean(axis=-2)) / (X.max(axis=-2) - X.min(axis=-2))
    d = pdist(X)
    d.sort()
    g = np.power(np.power(2 * np.arange(1, d.shape[0] + 1) / (X.shape[0] - 1), q - 1) / X.shape[0], 1 / (q - 1))
    return np.abs(beta1(np.log(d), np.log(g)))


def corr(X: np.ndarray):
    return corrq(X, 1)


def mle(X: np.ndarray, k: int = 5, distances: bool = False):
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()

    return (k - 1) / np.log(np.expand_dims(dist[:, -1], 1) / dist).sum(axis=-1)


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


def pca_sklearn(X: np.ndarray, explained_variance: float = 0.95):
    return PCA(n_components=explained_variance).fit(X).n_components_


def pca(X: np.ndarray, explained_variance: float = 0.95):
    X -= X.mean(axis=-2)
    S = np.linalg.svd(X, compute_uv=False)
    S /= np.square(X).sum()
    return S.size - np.sum(S.cumsum() >= explained_variance) + 1


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
