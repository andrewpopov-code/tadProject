import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from utils import unique_points


def _dist(i, j, d, m):
    # dist(u=X[i], v=X[j]) = d[m * i + j - ((i + 2) * (i + 1)) // 2]
    return d[m * i + j - ((i + 2) * (i + 1)) // 2]


def _capacity_alg(d: np.array, R: float, m):
    c = []
    ok = True
    for i in range(m):
        for j in c:
            ok = ok and (_dist(j, i, d, m) >= R)
        if ok:
            c.append(i)
    return len(c)


def capacity(X: np.array):
    X = np.unique(X, axis=-2)
    d = pdist(X, metric='euclidean')
    d.sort()
    m = np.zeros_like(d)
    for i, t in enumerate(d):
        m[i] = _capacity_alg(d, t, X.shape[0])
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(d).reshape(-1, 1), np.log(m).reshape(-1, 1))
    return lr.coef_[0, 0]


def corr(X: np.array):
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


def _information_alg(d: np.array, R: float, m):
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


def information(X: np.array):
    X = np.unique(X, axis=-2)
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


def mle(X: np.array, k: int = 5):
    X = unique_points(X)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors()

    return 1 / np.log(np.expand_dims(distances[:, k - 1], 1) / distances[:, :k - 1]).mean()


def mm(X: np.array, k: int):
    # TODO: figure out if this is correct

    X = unique_points(X)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors()

    w = distances[:, -1]
    m1 = distances.mean(axis=1)
    return -m1 / (m1 - w)


def pca(X: np.array, explained_variance: float = 0.95):
    X = unique_points(X)
    pca = PCA(n_components=explained_variance)
    pca.fit(X)

    return pca.n_components_


def two_nn(X: np.array):
    X = unique_points(X)
    n = X.shape[0]

    nn = NearestNeighbors(n_neighbors=2).fit(X)
    distances, _ = nn.kneighbors()

    mu = np.sort(distances[:, 1] / distances[:, 0])
    cdf = np.linspace(0, 1 - 1/n, n)
    lr = LinearRegression(fit_intercept=False)
    return lr.fit(np.log(mu), -np.log(1 - cdf)).coef_[0, 0]
