import numpy as np
from ripser import ripser
from scipy.spatial import distance_matrix
from .entropy import entropy


def diagrams(X: np.array, maxdim: int = 1):
    return ripser(X, maxdim=maxdim, distance_matrix=True)['dgms']


def betti(diagrams, n_bins: int = 100):
    maxdim = len(diagrams)
    diagrams = [
        diagrams[dim][diagrams[dim] != np.inf] for dim in range(maxdim)
    ]
    global_min = min(diagrams[dim].min() for dim in range(maxdim))
    global_max = max(diagrams[dim].max() for dim in range(maxdim))
    steps = np.linspace(global_min, global_max, num=n_bins, endpoint=True)
    bc = [
        (diagrams[dim] < steps.reshape(-1, 1)).sum(axis=1) for dim in range(maxdim)
    ]

    return bc


def persistence_entropy(diagrams):
    L = persistence_metric(diagrams)
    prob = [(diagrams[dim][:, 1] - diagrams[dim][:, 0]) / L[dim] for dim in range(len(diagrams))]
    return np.array([entropy(prob[dim], np.log) for dim in range(len(diagrams))])


def persistence_metric(diagrams):
    z = np.zeros(len(diagrams))
    for dim in range(len(diagrams)):
        for end, start in diagrams[dim]:
            z[dim] += end - start
    return z


def pairwise_dist(bc: np.array):
    return distance_matrix(bc, bc)
