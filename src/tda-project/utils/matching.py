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


def _matching_alg(dist: np.ndarray) -> np.ndarray:
    u, v, p, way = np.zeros(dist.shape[0] + 1, dtype=int), np.zeros(dist.shape[0] + 1, dtype=int), np.zeros(dist.shape[0] + 1, dtype=int), np.zeros(dist.shape[0] + 1, dtype=int)
    for i in range(1, dist.shape[0] + 1):
        p[0] = i
        j0 = 0
        minv, used = np.full(dist.shape[1] + 1, np.inf), np.full(dist.shape[1] + 1, False)
        first = True
        while p[j0] != 0 or first:
            first = False
            used[j0] = True
            i0, d, j1 = p[j0], np.inf, None
            for j in range(1, dist.shape[0] + 1):
                if not used[j]:
                    cur = dist[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < d:
                        d = minv[j]
                        j1 = j
            for j in range(1, dist.shape[1] + 1):
                if used[j]:
                    u[p[j]] += d
                    v[j] -= d
                else:
                    minv[j] -= d
            j0 = j1

        first = True
        while j0 or first:
            first = False
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return p[1:] - 1
