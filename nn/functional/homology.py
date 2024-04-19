import numpy as np
from ripser import ripser
from scipy.spatial import distance_matrix
from .entropy import entropy
from utils.math import unique_points


def diagrams(X: np.array, maxdim: int = 1):
    return ripser(X, maxdim=maxdim, distance_matrix=True)['dgms']


def drop_inf(diag):
    for dim in range(len(diag)):
        mask = np.ma.fix_invalid(diag[dim]).mask
        if mask.shape:
            diag[dim] = diag[dim][~mask.any(axis=1)]
    return diag


def betti(diag, n_bins: int = 100):
    diag = drop_inf(diag)
    global_min = min(diag[dim].min() if diag[dim].size else 0 for dim in range(len(diag)))
    global_max = max(diag[dim].max() if diag[dim].size else 0 for dim in range(len(diag)))
    steps = np.linspace(global_min, global_max, num=n_bins, endpoint=True)
    bc = [
        ((diag[0][:, 0] <= steps.reshape(-1, 1)) & (diag[0][:, 1] > steps.reshape(-1, 1))).sum(axis=1) if diag[dim].size else np.zeros_like(steps) for dim in range(len(diag))
    ]

    return np.array(bc)


def persistence_entropy(diag):
    diag = drop_inf(diag)
    L = persistence_norm(diag)
    prob = [(diag[dim][:, 1] - diag[dim][:, 0]) / L[dim] if diag[dim].size else None for dim in range(len(diag))]
    return np.array([entropy(prob[dim], np.log) if prob[dim] is not None else 0 for dim in range(len(diag))])


def persistence_norm(diag):
    diag = drop_inf(diag)
    z = np.zeros(len(diag))
    for dim in range(len(diag)):
        for start, end in diag[dim]:
            z[dim] += end - start
    return z


def pairwise_dist(bc: np.array):
    return [
        distance_matrix(bc[:, dim], bc[:, dim]) for dim in range(len(bc[0]))
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
    return diagrams(M, maxdim)


def divergence(X: np.array, Y: np.array, maxdim: int = 1):
    return persistence_norm(cross_barcode(X, Y, maxdim))
