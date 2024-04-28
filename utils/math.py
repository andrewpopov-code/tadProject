import numpy as np
import torch
from scipy.spatial import distance_matrix


def image_to_cloud(X: np.ndarray, channel_first: bool = False):
    if channel_first:
        X = X.T
    return X.reshape(X.shape[0] * X.shape[1], X.shape[2])


def unique_points(x: np.ndarray):
    return np.unique(x, axis=-2)


def compute_unique_distances(X: np.ndarray):
    X = unique_points(X)
    return distance_matrix(X, X)


def mle_aggregate(dim: np.ndarray):
    return 1 / np.mean(1 / dim)


def mle_aggregate_torch(dim: torch.Tensor):
    return 1 / torch.mean(1 / dim)


def inf_mask(arr: np.ndarray):
    return np.ma.fix_invalid(arr).mask


def diagrams_to_tensor(dgms: [list[np.ndarray], list[list[np.ndarray]]], fill_value=np.nan, requires_grad: bool = False) -> torch.Tensor:
    if isinstance(dgms[0], list):
        m_dgm = max((d for b in dgms for d in b), key=lambda x: x.shape[0]).shape[0]
        m_dgm1 = max((d for b in dgms for d in b), key=lambda x: x.shape[1]).shape[1]
        return torch.tensor(np.stack(
            [
                np.stack(
                    [
                        np.pad(dgms[b][dim], ((0, m_dgm - dgms[b][dim].shape[0]), (0, m_dgm1 - dgms[b][dim].shape[1])), constant_values=fill_value) for dim in range(len(dgms[b]))
                    ]
                ) for b in range(len(dgms))
            ]
        ), requires_grad=requires_grad)

    m_dgm = max(dgms, key=lambda x: x.shape[0]).shape[0]
    m_dgm1 = max(dgms, key=lambda x: x.shape[1]).shape[1]
    return torch.tensor(
        np.stack(
            [
                np.pad(dgms[dim], ((0, m_dgm - dgms[dim].shape[0]), (0, m_dgm1 - dgms[dim].shape[1])), constant_values=fill_value) for dim in range(len(dgms))
            ]
        ), requires_grad=requires_grad
    )
