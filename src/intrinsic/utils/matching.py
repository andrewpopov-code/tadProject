import numpy as np
import torch


def matching_alg(dist: np.ndarray) -> np.ndarray:
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


def matching_alg_torch(dist: torch.Tensor) -> torch.Tensor:
    b, n = dist.shape[:2]
    u, v, p, way = torch.zeros((b, n + 1), dtype=int), torch.zeros((b, n + 1), dtype=int), torch.zeros((b, n + 1), dtype=int), torch.zeros((b, n + 1), dtype=int)
    for i in range(1, n + 1):
        p[:, 0] = i
        for batch in range(b):
            j0 = 0
            minv, used = torch.full(n + 1, torch.inf), torch.full(n + 1, False, dtype=bool)
            first = True
            while p[batch, j0] != 0 or first:
                first = False
                used[j0] = True
                i0, d, j1 = p[batch, j0], np.inf, None
                for j in range(1, dist.shape[0] + 1):
                    if not used[j]:
                        cur = dist[i0 - 1, j - 1] - u[batch, i0] - v[batch, j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[batch, j] = j0
                        if minv[j] < d:
                            d = minv[j]
                            j1 = j
                for j in range(1, dist.shape[1] + 1):
                    if used[j]:
                        u[batch, p[batch, j]] += d
                        v[batch, j] -= d
                    else:
                        minv[j] -= d
                j0 = j1

            first = True
            while j0 or first:
                first = False
                j1 = way[batch, j0]
                p[j0] = p[j1]
                j0 = j1

    return p[:, 1:] - 1
