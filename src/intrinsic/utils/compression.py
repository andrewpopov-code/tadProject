import numpy as np


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


def rle(s: str):
    x = s[0]
    ret = ""
    k = 1
    for i in range(1, len(s)):
        if s[i] == x:
            k += 1
        else:
            ret += str(k) + x if k > 1 else x
            x = s[i]
            k = 1
    return ret
