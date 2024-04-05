import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


def dist(i, j, d, m):
    # dist(u=X[i], v=X[j]) = d[m * i + j - ((i + 2) * (i + 1)) // 2]
    return d[m * i + j - ((i + 2) * (i + 1)) // 2]


def alg(d: np.array, R: float, m):
    # B_i(r) = # {j | d(i, j) < r}
    a = np.zeros(m)
    b = np.zeros(m)
    for i in range(m):
        for j in range(i + 1, m):
            if dist(i, j, d, m) < R:
                a[i] += 1
                a[j] += 1
            elif dist(i, j, d, m) == R:
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
        n, b = alg(d, t, X.shape[0])
        D[:, i] = t * (n / b)

    # Calculate the information entropy of different topological distance scale r for each node i.
    for i in range(X.shape[0]):
        p = np.zeros_like(d)
        for j in range(d.size):
            d_sum = d[:, r]
