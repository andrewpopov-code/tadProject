import numpy as np
import torch
from scipy.spatial import distance_matrix
from scipy.special import gamma


def image_to_cloud(X: np.ndarray, channel_first: bool = False):
    if channel_first:
        X = X.T
    return X.reshape(X.shape[0] * X.shape[1], X.shape[2])


def unique_points(X: np.ndarray):
    return np.unique(X, axis=-2)


def compute_unique_distances(X: np.ndarray) -> np.ndarray:
    X = unique_points(X)
    return distance_matrix(X, X)


def mle_aggregate(dim: np.ndarray):
    return 1 / np.mean(1 / dim)


def mle_aggregate_torch(dim: torch.Tensor):
    return 1 / torch.mean(1 / dim, dim=-1)


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


def beta1(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(x*y) / np.square(x).sum()


def beta1_intercept(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    return (np.sum(x*y) - n*x.mean()*y.mean()) / (np.sum(x*x) - n*x.mean()*x.mean())


def batch_select(X: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
    dummy = ix.unsqueeze(-1).expand(*ix.shape, X.shape[-1]).squeeze(-2)
    return torch.gather(X.detach().unsqueeze(1).repeat(1, ix.shape[1], 1, 1), 2, dummy)


def neighbors(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.cdist(X, X).topk(k=k + 1, largest=False, sorted=True)
    return dist.values[:, :, 1:], dist.indices[:, :, 1:]


def set_ops(X: torch.Tensor, Y: torch.Tensor) -> tuple:
    combined = torch.cat((X, Y), dim=-1)
    counts = torch.nn.functional.one_hot(combined).sum(dim=-2)

    union = torch.sum(counts > 0, dim=-1)
    intersection = torch.sum(counts > 1, dim=-1)
    delta = union - intersection
    diffX = X.shape[-1] - intersection  # only in X
    diffY = Y.shape[-1] - intersection  # only in Y

    return union, intersection, delta, diffX, diffY


def extended_distance(diagX: np.ndarray, diagY: np.ndarray, q: float) -> np.ndarray:
    diagXp, diagYp = diagX.mean(axis=1) / 2, diagY.mean(axis=1) / 2
    return np.power(np.block(
        [
            [np.max(np.abs(diagX.reshape(-1, 1, 2) - diagY), axis=-1),
             np.max(np.abs(diagX.reshape(-1, 1, 2) - diagXp), axis=-1)],
            [np.max(np.abs(diagYp.reshape(-1, 1, 2) - diagY), axis=-1), np.zeros((diagYp.shape[0], diagXp.shape[0]))]
        ]
    ), 1 if q == np.inf else q)  # (X1, ..., Xn, Y1', ..., Ym') x (Y1, ..., Ym, X1', ..., Xn')


def unit_ball_volume(n: float) -> float:
    return np.sqrt(np.power(np.pi, n)) / gamma(n / 2 + 1)
