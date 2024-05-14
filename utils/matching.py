import numpy as np


def _min_func(m: np.ndarray, rm: np.ndarray):
    m_j = -1
    m_val = np.inf
    for j in range(m.size):
        if rm[j] and m[j] < m_val:
            m_j, m_val = j, m[j]
    return m_j, m_val


def matching_alg(C: np.ndarray):
    n = C.shape[0]
    p = np.full(n, -1, dtype=int)
    a, b = np.zeros(n), np.zeros(n)
    for i in range(n):
        lp = np.full(n, False, dtype=bool)
        rm = np.full(n, True, dtype=bool)
        lp[i] = True
        m = C[i]  # min values
        mix = np.full_like(m, i)  # arg min

        while True:
            m_j, delta = _min_func(m, rm)
            v, u = mix[m_j], m_j
            a[lp] -= delta
            b[~rm] += delta
            m -= delta

            rm[v] = False
            if p[v] == -1:
                # invert mix
                inv = np.full_like(p, -1, dtype=int)
                for j in range(n):
                    inv[p[j]] = j

                # while prev != i -> continue
                t = u
                while mix[t] != i:
                    tmp = inv[mix[t]]
                    p[t] = mix[t]
                    t = tmp
                else:
                    p[t] = i
            else:
                lp[p[u]] = True
                for j in range(n):
                    if C[p[u], j] + a[p[u]] + b[j] < m[j]:
                        m[j] = C[p[u], j] + a[p[u]] + b[j]
                        mix[j] = p[u]
