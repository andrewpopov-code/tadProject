import numpy as np
from gph import ripser_parallel
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from scipy.spatial import distance_matrix
from .entropy import entropy
from utils.math import unique_points, inf_mask


def diagrams(X: np.array, maxdim: int = 1, distances: bool = False, gens: bool = False):
    if not gens:
        return ripser_parallel(X, maxdim=maxdim, metric='precomputed' if distances else 'euclidean')['dgms']
    ret = ripser_parallel(X, maxdim=maxdim, metric='precomputed' if distances else 'euclidean', return_generators=True)
    return ret['dgms'], ret['gens']


def drop_inf(diag: list[np.ndarray]) -> list[np.ndarray]:
    for dim in range(len(diag)):
        mask = inf_mask(diag[dim])
        if mask.shape:
            diag[dim] = diag[dim][~mask.any(axis=1)]
    return diag


def diagrams_barycenter(diag: list[list[np.ndarray]]) -> list[np.ndarray]:
    diag = list(map(drop_inf, diag))
    bary = []
    for dim in range(len(diag)):
        bary.append(lagrangian_barycenter([d[dim] for d in diag]))
    return bary


def betti(diag, n_bins: int = 100):
    diag = drop_inf(diag)
    global_min = min(diag[dim].min() if diag[dim].size else 0 for dim in range(len(diag)))
    global_max = max(diag[dim].max() if diag[dim].size else 0 for dim in range(len(diag)))
    steps = np.linspace(global_min, global_max, num=n_bins, endpoint=True)
    bc = [
        ((diag[dim][:, 0] <= steps.reshape(-1, 1)) & (diag[dim][:, 1] > steps.reshape(-1, 1))).sum(axis=1) if diag[dim].size else np.zeros_like(steps) for dim in range(len(diag))
    ]

    return np.array(bc)


def persistence_entropy(diag):
    diag = drop_inf(diag)
    L = persistence_norm(diag)
    prob = [(diag[dim][:, 1] - diag[dim][:, 0]) / L[dim] if diag[dim].size else None for dim in range(len(diag))]
    return np.array([entropy(prob[dim], np.log) if prob[dim] is not None else 0 for dim in range(len(diag))])


def persistence_norm(diag: list[np.ndarray]) -> np.ndarray:
    diag = drop_inf(diag)
    z = np.zeros(len(diag))
    for dim in range(len(diag)):
        for start, end in diag[dim]:
            z[dim] += end - start
    return z


def total_persistence(diag: list[np.ndarray], q: float) -> float:
    diag = np.vstack(drop_inf(diag))
    return np.power(diag[:, 1] - diag[:, 0], q).sum()


def amplitude(diag: list[np.ndarray], p: float) -> float:
    diag = np.vstack(drop_inf(diag))
    if p == np.inf:
        return np.max(diag[:, 1] - diag[:, 0]) / np.sqrt(2)
    return np.power(total_persistence(diag, p), 1 / p) / np.sqrt(2)


def landscapes(diag: list[np.ndarray], n_points: int = 100) -> np.ndarray:
    diag = np.vstack(drop_inf(diag))
    global_min, global_max = diag.min(), diag.max()
    steps = np.linspace(global_min, global_max, num=n_points, endpoint=True)
    f = ((diag[:, 0] <= steps.reshape(-1, 1)) & (diag.mean(axis=-1) > steps.reshape(-1, 1))) * (steps.reshape(-1, 1) - diag[:, 0])
    s = ((diag[:, 1] >= steps.reshape(-1, 1)) & (diag.mean(axis=-1) <= steps.reshape(-1, 1))) * (-steps.reshape(-1, 1) + diag[:, 1])
    return np.sort(f + s, axis=-1)[:, ::-1]


def ls_moment(diag: list[np.ndarray]):
    z = persistence_norm(diag)
    return np.sum(z[::2] - z[1::2])


def pairwise_dist(bc: np.array):
    return [
        distance_matrix(bc[:, dim], bc[:, dim]) for dim in range(bc.shape[1])
    ]


def _distance_alg(dist: np.ndarray) -> np.ndarray:
    u, v, p, way = np.zeros(dist.shape[0] + 1, dtype=int), np.zeros(dist.shape[0] + 1, dtype=int), np.zeros(dist.shape[0] + 1, dtype=int), np.zeros(dist.shape[0] + 1, dtype=int)
    for i in range(1, dist.shape[0] + 1):
        p[0] = i
        j0 = 0
        minv, used = np.full(dist.shape[1] + 1, np.inf), np.full(dist.shape[1] + 1, False)
        first = True
        while p[j0] != 0 or first:
            first = False
            used[j0] = True
            i0, d, j1 = p[j0], np.inf, None
            for j in range(1, dist.shape[0] + 1):
                if not used[j]:
                    cur = dist[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < d:
                        d = minv[j]
                        j1 = j
            for j in range(1, dist.shape[1] + 1):
                if used[j]:
                    u[p[j]] += d
                    v[j] -= d
                else:
                    minv[j] -= d
            j0 = j1

        first = True
        while j0 or first:
            first = False
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return p[1:] - 1


def wasserstein_distance(diagX: list[np.ndarray], diagY: list[np.ndarray], q: float = np.inf) -> float:
    diagX, diagY = np.vstack(drop_inf(diagX)), np.vstack(drop_inf(diagY))
    diagXp, diagYp = diagX.mean(axis=1) / 2, diagY.mean(axis=1) / 2
    dist = np.power(np.block(
        [
            [np.max(np.abs(diagX.reshape(-1, 1, 2) - diagY), axis=-1), np.max(np.abs(diagX.reshape(-1, 1, 2) - diagXp), axis=-1)],
            [np.max(np.abs(diagYp.reshape(-1, 1, 2) - diagY), axis=-1), np.zeros((diagYp.shape[0], diagXp.shape[0]))]
        ]
    ), 1 if q == np.inf else q)  # (X1, ..., Xn, Y1', ..., Ym') x (Y1, ..., Ym, X1', ..., Xn')
    return np.linalg.norm(
        np.max(
            np.abs(np.vstack([diagX, diagXp])[_distance_alg(dist)] - np.vstack([diagY, diagYp])), axis=1
        ), q
    )


def cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    X = unique_points(X)
    Y = unique_points(Y)
    return diagrams(np.vstack([X, Y]), maxdim=maxdim)


def r_cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    X = unique_points(X)
    Y = unique_points(Y)
    XX = distance_matrix(X, X)
    YY = distance_matrix(Y, Y)
    inf_block = np.triu(np.full_like(XX, np.inf), 1) + XX

    M = np.block([
        [XX, inf_block.T, np.zeros((XX.shape[0], 1))],
        [inf_block, np.minimum(XX, YY), np.full((XX.shape[0], 1), np.inf)],
        [np.zeros((1, XX.shape[0])), np.full((1, XX.shape[0]), np.inf), 0]
    ])
    return diagrams(M, maxdim=maxdim, distances=True)


def mtd(X: np.array, Y: np.array, maxdim: int = 1):
    return persistence_norm(cross_barcode(X, Y, maxdim))


def rtd(X: np.array, Y: np.array, maxdim: int = 1):
    return persistence_norm(r_cross_barcode(X, Y, maxdim))
