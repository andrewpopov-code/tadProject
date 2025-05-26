import numpy as np
from curvature import curvature


def lzw(s: str, abc: list, d: dict = None, ret_dict: bool = False):
    d = d or {x: i for i, x in enumerate(abc)}
    k = len(d)
    x = s[0]
    ret = ""
    code_len = int(np.ceil(np.log2(k)))
    for i in range(1, len(s)):
        y = s[i]
        if x + y in d:
            x = x + y
            continue
        ret += bin(d[x])[2:].rjust(code_len, '0')
        d[x + y] = k
        k += 1
        code_len = int(np.ceil(np.log2(k)))
        x = y
    ret += bin(d[x])[2:].rjust(code_len, '0')

    if ret_dict:
        return ret, d
    return ret


def lzw_conditional(s: str, t: str, abc: list, d: dict = None, ret_dict: bool = False):
    """Compression of s after t"""
    _, d = lzw(t, abc, d, ret_dict=True)
    ret, d = lzw(s, abc, d, ret_dict=True)
    if ret_dict:
        return ret, d
    return ret


def compression_complexity(s: str, abc: list):
    abc = abc + list('0123456789')
    return len(lzw(s, abc)) * np.log(len(abc)) / np.log(len(abc) - 10)


def conditional_compression_complexity(s: str, t: str, abc: list):
    """C(s | t)"""
    abc = abc + list('0123456789')
    return len(lzw_conditional(s, t, abc)) * np.log(len(abc)) / np.log(len(abc) - 10)


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
