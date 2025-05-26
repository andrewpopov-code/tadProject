import numpy as np
from src.intrinsic.functional.information import entropy
from src.intrinsic.utils.compression import lzw, lzw_conditional  # , rle
from src.intrinsic.functional.curvature import curvature


def compression_complexity(s: str, abc: list):
    return len(lzw(s, abc)) / np.ceil(np.log2(len(abc)))


def conditional_compression_complexity(s: str, t: str, abc: list):
    """C(s | t)"""
    return len(lzw_conditional(s, t, abc)) / np.ceil(np.log2(len(abc)))


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
    return 2 * l * np.log(len(s)) / np.log(l) + len(s) * entropy(prob, l)


def nid(s: str, t: str, abc: list):
    return complexity_distance(s, t, abc) / max(compression_complexity(s, abc), compression_complexity(t, abc))


def ncd(s: str, t: str, abc: list):
    return (
            compression_complexity(s + t, abc) - min(compression_complexity(s, abc), compression_complexity(t, abc))
    ) / max(compression_complexity(s, abc), compression_complexity(t, abc))


def batch_compression_curvature(X: list[str], abc: list, k: int = 5):
    ind = np.array([compression_complexity(s, abc) for s in X], dtype=float)
    cross = np.array([[conditional_compression_complexity(s, t, abc) for t in X] for s in X], dtype=float)
    X = np.minimum(cross, cross.T) / np.maximum(ind.reshape(-1, 1), ind.reshape(1, -1))
    X -= np.diag(np.diag(X))
    return curvature(X, k, distances=True)


def batch_complexity(X: list[str], abc: list):
    return np.array([
        compression_complexity(s, abc) for s in X
    ])


def plain_information(s: str, t: str, abc: list):
    return compression_complexity(t, abc) - conditional_compression_complexity(t, s, abc)


def plain_divergence(s: str, t: str, abc: list):
    return plain_information(s, t, abc) / np.sqrt(compression_complexity(s, abc) * compression_complexity(t, abc))
