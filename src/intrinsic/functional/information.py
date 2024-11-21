import numpy as np
from scipy.spatial import distance_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from src.intrinsic.functional.kernel import rbf_kernel
from src.intrinsic.utils.math import top_k_dist


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
    return 2 * np.arccos(max(min(np.sqrt(P * Q).sum(), 1), -1))


def hellinger(P: np.ndarray, Q: np.ndarray) -> float:
    return np.linalg.norm(np.sqrt(P) - np.sqrt(Q), 2)


def total_distance(P: np.ndarray, Q: np.ndarray) -> float:
    return np.abs(P - Q).sum() / 2


def wasserstein_distance(P: np.ndarray, Q: np.ndarray, q: float) -> float:
    return np.power(np.power(np.abs(np.cumsum(P).reshape(-1, 1) - np.cumsum(Q)), q).sum(), 1 / q)


def bhatta_distance(P: np.ndarray, Q: np.ndarray) -> float:
    return -np.log(np.sqrt(P * Q).sum())


def sigma_dist(x: np.ndarray, k: int = 10):
    return np.mean(top_k_dist(x, k))


def matrix_entropy(A: np.ndarray, alpha: float):
    return 1 / (1 - alpha) * np.log2(np.power(np.linalg.eigvals(A), alpha).sum(axis=-1))


def renyi_matrix_entropy(x: np.ndarray, sigma: float, alpha: float):
    K = rbf_kernel(x, sigma)
    A = K / K.trace(axis1=-2, axis2=-1)
    return matrix_entropy(A, alpha)


def renyi_matrix_cross_entropy(x: np.ndarray, y: np.ndarray, sigma1: float, sigma2: float, alpha: float):
    K1, K2 = rbf_kernel(x, sigma1), rbf_kernel(y, sigma2)
    T = K1 * K2
    return matrix_entropy(T / T.trace(axis1=-2, axis2=-1), alpha)


def mutual_information(x: np.ndarray, y: np.ndarray, sigma1: float, sigma2: float, alpha: float):
    return renyi_matrix_entropy(x, sigma1, alpha) + renyi_matrix_entropy(y, sigma2, alpha) - renyi_matrix_cross_entropy(x, y, sigma1, sigma2, alpha)


def label_sharpness(x: np.ndarray, y: np.ndarray, M: int = 1000):
    # binary classification
    ind, _ = StratifiedShuffleSplit(1, train_size=M).split(x, y)
    A, B = x[ind][y[ind] == 0], x[ind][y[ind] == 1]
    return np.max(1 / distance_matrix(A, B))
