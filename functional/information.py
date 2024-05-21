import numpy as np
from utils.matching import matching_alg


def entropy(prob: np.ndarray, logarithm):
    return (-prob * logarithm(prob)).sum(axis=-1)


def cross_entropy(P: np.ndarray, Q: np.ndarray, logarithm):
    return (-P * logarithm(Q)).sum(axis=-1)


def kld(P: np.ndarray, Q: np.ndarray, logarithm=np.log):
    return cross_entropy(P, Q, logarithm) - entropy(P, logarithm)


def renyi_d(P: np.ndarray, Q: np.ndarray, alpha: float):
    return 1 / np.log(np.sum(np.power(P, alpha) / np.power(Q, alpha - 1)))


def jensen_d(P: np.ndarray, Q: np.ndarray, logarithm=np.log):
    M = (P + Q) / 2
    return (kld(P, M, logarithm) + kld(Q, M, logarithm)) / 2


def dmi(prob: np.ndarray, classes: np.ndarray, n: int):
    L = np.eye(n)[classes]
    O = prob.T / prob.shape[0]
    return -np.log(np.linalg.det(O @ L))


def multinomial_geodesic(P: np.ndarray, Q: np.ndarray) -> float:
    return 2 * np.arccos(np.sqrt(P * Q).sum())


def hellinger(P: np.ndarray, Q: np.ndarray) -> float:
    return np.linalg.norm(np.sqrt(P) - np.sqrt(Q), 2)


def total_distance(P: np.ndarray, Q: np.ndarray) -> float:
    return np.abs(P - Q).sum() / 2


def wasserstein_distance(P: np.ndarray, Q: np.ndarray, q: float) -> float:
    dist = np.power(np.abs(P.reshape(-1, 1) - Q), q)
    mat = matching_alg(dist)
    return np.power(dist[mat, :].sum(), 1 / q)


def bhatta_distance(P: np.ndarray, Q: np.ndarray) -> float:
    return -np.log(np.sqrt(P * Q).sum())
