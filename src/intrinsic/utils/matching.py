import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def matching_alg(dist: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(dist)
    return col_ind


def matching_alg_torch(dist: torch.Tensor) -> torch.Tensor:
    ret = np.zeros((dist.shape[0], dist.shape[1]))
    for b in range(dist.shape[0]):
        row_ind, col_ind = linear_sum_assignment(dist[b].numpy(force=True))
        ret[b] = col_ind

    return torch.tensor(ret).to(dist.device)
