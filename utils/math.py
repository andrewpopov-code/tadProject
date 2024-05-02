import numpy as np
import torch
from scipy.spatial import distance_matrix


def image_to_cloud(X: np.ndarray, channel_first: bool = False):
    if channel_first:
        X = X.T
    return X.reshape(X.shape[0] * X.shape[1], X.shape[2])


def unique_points(x: np.ndarray):
    return np.unique(x, axis=-2)


def compute_unique_distances(X: np.ndarray) -> np.ndarray:
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
        return torch.tensor(np.stack(
            [
                np.stack(
                    [
                        np.pad(dgms[b][dim], ((0, m_dgm - dgms[b][dim].shape[0]), (0, 0)), constant_values=fill_value) for dim in range(len(dgms[b]))
                    ]
                ) for b in range(len(dgms))
            ]
        ), requires_grad=requires_grad)

    m_dgm = max(dgms, key=lambda x: x.shape[0]).shape[0]
    return torch.tensor(
        np.stack(
            [
                np.pad(dgms[dim], ((0, m_dgm - dgms[dim].shape[0]), (0, 0)), constant_values=fill_value) for dim in range(len(dgms))
            ]
        ), requires_grad=requires_grad
    )


def gens_to_tensor(gens: [list[np.ndarray], list[list[np.ndarray]]], fill_value: int = -1):
    if isinstance(gens[0], list):
        for b in range(len(gens)):
            gens[b][0] = np.repeat(gens[b][0], [2, 1, 1], axis=-1)
        return diagrams_to_tensor(gens, fill_value=fill_value)
    gens[0] = np.repeat(gens[0], [2, 1, 1], axis=-1)
    return diagrams_to_tensor(gens, fill_value=fill_value)


def boundary_matrix(gens: np.ndarray, x: np.ndarray):
    return np.logical_or.reduce(gens.reshape(*gens.shape, 1) == x.reshape(1, -1), axis=-2).astype(int)


def beta1(x: np.ndarray, y: np.ndarray):
    return np.sum(x*y) / np.square(x).sum()


def beta1_intercept(x: np.ndarray, y: np.ndarray):
    n = x.size
    return (np.sum(x*y) - n*x.mean()*y.mean()) / (np.sum(x*x) - n*x.mean()*x.mean())
