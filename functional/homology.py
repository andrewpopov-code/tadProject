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
    diag = drop_inf(diag)
    z = np.zeros(len(diag))
    for dim in range(len(diag)):
        for start, end in diag[dim]:
            z[dim] += np.power(end - start, q)
    return z.sum()


def ls_moment(diag: list[np.ndarray]):
    z = persistence_norm(diag)
    return np.sum(z[::2] - z[1::2])


def pairwise_dist(bc: np.array):
    return [
        distance_matrix(bc[:, dim], bc[:, dim]) for dim in range(bc.shape[1])
    ]


def cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    X = unique_points(X)
    Y = unique_points(Y)
    XX = distance_matrix(X, X)
    XY = distance_matrix(X, Y)
    YY = distance_matrix(Y, Y)
    M = np.block([
        [XX, XY],
        [XY.T, YY]
    ])
    return diagrams(M, maxdim=maxdim, distances=True)


def r_cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    # X and Y are the same point cloud, but with different embedding values
    X = unique_points(X)
    Y = unique_points(Y)
    XX = distance_matrix(X, X)
    YY = distance_matrix(Y, Y)
    inf_block = np.triu(np.full_like(XX, np.inf), 1) + XX

    M = np.block([
        [np.zeros_like(XX), inf_block.T, np.zeros((XX.shape[0], 1))],
        [inf_block, np.minimum(XX, YY), np.full((XX.shape[0], 1), np.inf)],
        [np.zeros((1, XX.shape[0])), np.full((1, XX.shape[0]), np.inf), 0]
    ])
    return diagrams(M, maxdim=maxdim, distances=True)


def mtd(X: np.array, Y: np.array, maxdim: int = 1):
    return persistence_norm(cross_barcode(X, Y, maxdim))


def rtd(X: np.array, Y: np.array, maxdim: int = 1):
    return persistence_norm(r_cross_barcode(X, Y, maxdim))
