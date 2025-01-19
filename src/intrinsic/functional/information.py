import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import distance_matrix
from scipy.special import gammaln, xlogy
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import permutations
from src.intrinsic.functional.kernel import rbf_kernel
from src.intrinsic.utils.math import top_k_dist
from src.intrinsic.utils.compression import lzw, lzw_conditional  # , rle
from math import factorial


def entropy(prob: np.ndarray, base: float = np.e):
    return xlogy(-prob, prob).sum(axis=-1) / np.log(base)


def cross_entropy(P: np.ndarray, Q: np.ndarray, base: float = np.e):
    return xlogy(-P, Q).sum(axis=-1) / np.log(base)


def kld(P: np.ndarray, Q: np.ndarray, base: float = np.e):
    return cross_entropy(P, Q, base) - entropy(P, base)


def renyi_d(P: np.ndarray, Q: np.ndarray, alpha: float):
    return 1 / np.log(np.sum(np.power(P, alpha) / np.power(Q, alpha - 1)))


def jensen_d(P: np.ndarray, Q: np.ndarray, base=np.e):
    M = (P + Q) / 2
    return (kld(P, M, base) + kld(Q, M, base)) / 2


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


def permutation_entropy(density: np.ndarray, d: int):
    return entropy(density, np.log2) / gammaln(d)


def permutation_density(x: np.ndarray, d: int = 5):
    S = sliding_window_view(x, d)
    sh = []
    for p in permutations(range(d)):
        valid = np.logical_and.reduce(S[:, p][:, 1:] <= S[:, p][:, :-1], axis=1) / (x.size - d + 1)
        if valid > 0:
            sh.append(valid)
    return np.array(sh)


def disequilibrium(p: np.ndarray, q: np.ndarray, d: int):
    ...


def complexity(x: np.ndarray, d: int = 5):
    "https://www.sciencedirect.com/science/article/pii/S037843711100906X"
    p = permutation_density(x, d)
    pe = np.ones_like(p) / factorial(d)
    q_max = -((factorial(d) + 1) / factorial(d) * np.log(factorial(d) + 1) - 2 * gammaln(2 * d) + gammaln(d)) / 2
    q = entropy((p + pe) / 2, 2) + (factorial(d) - p.size) * (pe[0] / 2) * np.log2(pe[0] / 2) - entropy(p, 2) / 2 - np.log2(pe[0]) / 2
    return (q  / q_max) * permutation_entropy(p, d)


def compression_complexity(s: str, abc: list):
    abc = abc + list('0123456789')
    return len(lzw(s, abc))


def conditional_compression_complexity(s: str, t: str, abc: list):
    """C(s | t)"""
    abc = abc + list('0123456789')
    return len(lzw_conditional(s, t, abc))


def complexity_distance(s: str, t: str, abc: list):
    return min(conditional_compression_complexity(s, t, abc), conditional_compression_complexity(t, s, abc))


def diameter(l: list[str], abc: list):
    # TODO: which logarithm?
    ret = 0
    for i in range(len(l)):
        ret = max(ret, conditional_compression_complexity(l[i], ''.join(l[:i] + l[i + 1:]), abc))
    return ret + np.log(len(l)) / np.log(len(abc) + 10)


def complexity_bound(s: list[int], l: int):
    prob = np.eye(l)[s].sum(axis=0) / len(s)
    return 2 * (l + 1) * np.log(len(s)) / np.log(l) + len(s) * entropy(prob, l)
