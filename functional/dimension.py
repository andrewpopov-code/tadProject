import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .homology import persistence_norm, diagrams


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
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(d).reshape(-1, 1), np.log(m).reshape(-1, 1))
    return lr.coef_[0, 0]


def corr(X: np.ndarray):
    # X \in M(n, d)
    d = pdist(X, metric='euclidean')
    d.sort()
    c = np.zeros_like(d)
    for i, t in enumerate(d):
        c[i] = ((d <= t).sum()) / X.shape[0] / (X.shape[0] - 1)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(d).reshape(-1, 1), np.log(c).reshape(-1, 1))
    return lr.coef_[0, 0]


def _information_alg(d: np.ndarray, R: float, m):
    # B_i(r) = # {j | d(i, j) < r}
    a = np.zeros(m)
    b = np.zeros(m)
    for i in range(m):
        for j in range(i + 1, m):
            if _dist(i, j, d, m) < R:
                a[i] += 1
                a[j] += 1
            elif _dist(i, j, d, m) == R:
                b[i] += 1
                b[j] += 1
    return a + np.ones_like(a), b


def information(X: np.ndarray):  # FIXME
    d = pdist(X, metric='euclidean')
    R = shortest_path(csr_matrix(distance_matrix(X, X))).max(axis=-1)
    S = np.ceil(np.repeat(R.reshape(1, -1), d.size, axis=0).T / np.repeat(d.reshape(1, -1), X.shape[0], axis=0)).astype(int)

    D = np.zeros(X.shape[0], d.size)  # local dimensions
    for i, t in enumerate(d):
        n, b = _information_alg(d, t, X.shape[0])
        D[:, i] = t * (n / b)

    # Calculate the information entropy of different topological distance scale r for each node i.
    for i in range(X.shape[0]):
        p = np.zeros_like(d)
        for j in range(d.size):
            # d_sum = d[:, r]
            pass


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

    x = np.log(np.sort(distances[:, 1] / distances[:, 0]))
    y = -np.log(1 - np.linspace(0, 1 - 1/n, n))
    return np.sum(x*y) / np.square(x).sum()


def persistence(X: np.ndarray):
    n = X.shape[0]
    l = persistence_norm(diagrams(X))
    return np.log(n) / (np.log(n) - np.log(l))


def persistence_reg(X: np.ndarray):  # TODO: reimplement this
    n = np.arange(0, X.shape[0], X.shape[0] // 10)
    ...
