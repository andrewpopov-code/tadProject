import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .magnitude import magnitude
from .homology import vr_diagrams, drop_inf
from .information import entropy
from src.intrinsic.utils.math import beta1, beta1_intercept


def information(X: np.ndarray):
    d = pdist(X)
    d.sort()

    s = np.zeros_like(d)
    for i in range(d.shape[0]):
        flat = np.histogramdd(X, bins=int(np.ceil(d[-1] / d[i])))[0].reshape(-1)
        s[i] = -entropy(flat[flat > 0] / flat.sum(), np.log2)
    return np.abs(beta1_intercept(np.log2(d), s))


def information_renyi(X: np.ndarray, q: float):
    d = pdist(X)
    d.sort()
    s = np.zeros_like(d)
    for i in range(d.shape[0]):
        flat = np.histogramdd(X, bins=int(np.ceil(d[-1] / d[i])))[0].reshape(-1)
        s[i] = np.log(np.power(flat[flat > 0] / flat.sum(), q).sum())
    return np.abs(beta1_intercept(np.log(d), s) / (q - 1))


def corrq(X: np.ndarray, q: float):
    d = pdist(X)
    d.sort()
    g = np.power(np.power(2 * np.arange(1, d.shape[0] + 1) / (X.shape[0] - 1), q - 1) / X.shape[0], 1 / (q - 1))

    return np.abs(beta1_intercept(np.log(d), np.log(g)))


def corr(X: np.ndarray):
    d = pdist(X)
    d.sort()
    c = 2 * np.arange(1, d.shape[0] + 1) / (X.shape[0] - 1) / X.shape[0]
    return np.abs(beta1_intercept(np.log(d), np.log(c)))


def mle(X: np.ndarray, k: int = 5, distances: bool = False):
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()

    return (k - 1) / np.log(np.expand_dims(dist[:, -1], 1) / dist).sum(axis=-1)


def mm(X: np.ndarray, k: int = 5, distances: bool = False):
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()

    Tk = dist[:, -1]
    T = dist.mean(axis=1)
    return T / (Tk - T)


def ols(X: np.ndarray, k: int = 5, distances: bool = False):
    # Levina (in section 2)
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()

    x = np.log(np.arange(1, k + 1))
    y = np.log(dist.mean(axis=0))

    return beta1_intercept(x, y)


def pca_sklearn(X: np.ndarray, explained_variance: float = 0.95):
    return PCA(n_components=explained_variance).fit(X).n_components_


def pca(X: np.ndarray, explained_variance: float = 0.95):
    X -= X.mean(axis=-2)
    S = np.square(np.linalg.svd(X, compute_uv=False))
    S /= S.sum()
    return np.searchsorted(np.cumsum(S), explained_variance, side="right") + 1


def cluster_pca(X: np.ndarray, k: int = 5, explained_variance: float = 0.95):
    labels = KMeans(n_clusters=int(X.shape[0] / k)).fit(X)
    return np.array([
        PCA(n_components=explained_variance).fit(X[labels == l]).n_components_ for l in range(int(X.shape[0] / k))
    ])


def local_pca(X: np.ndarray, k: int = 5, explained_variance: float = 0.95):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, ix = nn.kneighbors()
    return np.array([pca(X[ix[i]], explained_variance) for i in range(X.shape[0])])


def two_nn(X: np.ndarray, distances: bool = False):
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=2, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()
    return beta1(np.log(np.sort(dist[:, 1] / dist[:, 0])), -np.log(1 - np.linspace(0, 1 - 1 / n, n)))


def persistence(X: np.ndarray, h_dim: int = 0, p: float = 1, distances: bool = False):
    n = np.arange(2, X.shape[0] + 1, X.shape[0] // 10)
    e = np.zeros(n.size)
    for i, ni in enumerate(n):
        dgms = drop_inf(vr_diagrams(X[:ni, :ni], distances=True))[h_dim] if distances else drop_inf(vr_diagrams(X[:ni]))[h_dim]
        e[i] = np.power(dgms[:, 1] - dgms[:, 0], p).sum()
    m = beta1_intercept(np.log(n), np.log(e))
    return p / (1 - m)


def magnitude_reg(X: np.ndarray, t: np.ndarray, i: int = None, j: int = None, distances: bool = False):
    m = np.zeros_like(t)
    if not distances:
        X = distance_matrix(X, X)
    for i in range(t.shape[0]):
        m[i] = magnitude(t[i] * X)
    i, j = i or 0, j or t.shape[0]
    return beta1_intercept(np.log(m[i:j]), np.log(t[i:j]))


def made(X: np.ndarray, k: int = 5, distances: bool = False, dim: int = None):
    "A. Farahmand, C. Szepesvári, J.-Y. Audibert: Manifold-adaptive dimension estimation"
    assert dim is not None or not distances
    dim = dim or X.shape[1]
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()
    ptw = np.log(2) / np.log(dist[:, -1] / dist[:, (k + 1) // 2])
    return np.ceil(np.min(ptw, dim).mean())


def landmarks(Y: np.ndarray, Z: np.ndarray = None, distances: bool = False, k1: int = 1, k2: int = 5):
    """
    K. Sricharan, R. Raich, A.O. Hero: Optimized intrinsic dimension estimation using nearest neighbor graphs
    :param Y: may be a matrix of distances between points in Y and Z
    """
    assert Z is not None or distances
    if not distances: Y = distance_matrix(Y, Z)
    nn = NearestNeighbors(n_neighbors=k2, metric='precomputed').fit(np.eye(Y.shape[1]))
    dist, _ = nn.kneighbors(Y)
    Tk1, Tk2 = np.mean(np.log(dist[:, k1])), np.mean(np.log(dist[:, k2]))
    return (np.log(k2 - 1) - np.log(k1 - 1)) / (Tk2 - Tk1)


def id_corr(D1: np.ndarray, D2: np.ndarray, dim_est, S: int):
    d1, d2, dc = dim_est(D1), dim_est(D2), dim_est(np.hstack([D1, D2]))
    rho = (d1 + d2 - dc) / max(d1, d2)
    cnt = 0
    ds = np.zeros(S)
    for i in range(S):
        np.random.shuffle(D2)
        ds[i] = dim_est(np.hstack([D1, D2]))
        cnt += ds[i] <= dc

    return rho, (cnt + 1) / (S + 1)


def reach(X: np.ndarray, k: int, j: int, distances: bool = False):
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, adj = nn.kneighbors()
    n = X.shape[0]

    for _ in range(j):
        adj = adj[adj].reshape(n, -1)
    np.sort(adj, -1)
    return np.mean(np.sum(adj[:, 1:] > adj[:, :-1], axis=-1) + 1)


def brito(X: np.ndarray, k: int, j: int, dim: int = None, distances: bool = False):
    assert dim is not None or not distances
    dim = dim or X.shape[1]
    r = reach(X, j, k, distances)
    ans, ans_d = np.inf, -1

    for d in range(dim):
        f = np.random.uniform(size=(X.shape[0], d))
        rf = reach(f, k, j)
        if np.abs(r - rf) < ans:
            ans, ans_d = np.abs(r - rf), d
    return ans_d


def steele(X: np.ndarray, distances: bool = False):
    Tcsr = minimum_spanning_tree(X if distances else distance_matrix(X, X)).toarray()
    deg = np.sum(Tcsr != 0, axis=-1)
    return np.square(deg).mean()
