import torch
import torch.nn as nn
from src.intrinsic.utils.math import neighbors


def mle(distances: torch.Tensor):
    _, n, k = distances.shape
    dim_ptw = (k - 1) / torch.log(distances[:, :, -1].unsqueeze(2) / distances).sum(dim=-1)
    return 1 / torch.mean(1 / dim_ptw, dim=-1)


def mom(distances: torch.Tensor):
    Tk = distances[:, :, -1]
    T = distances.mean(dim=-1)
    return (T / (Tk - T)).mean(dim=-1)


def two_nn(distances: torch.Tensor):
    n = distances.shape[1]
    x = torch.log(torch.sort(distances[:, :, 1] / distances[:, :, 0]).values)
    y = -torch.log(1 - torch.linspace(0, 1 - 1 / n, n))
    return torch.sum(x*y, dim=-1) / torch.square(x).sum(dim=-1)


class Dimension(nn.Module):
    def __init__(self, k: int = None, method: str = 'two_nn'):
        super().__init__()
        self.k = k or 2
        self.est = two_nn if method == 'two_nn' else mle if method == 'mle' else mom

    def forward(self, X: torch.Tensor):
        dist, _ = neighbors(X, self.k)
        return self.est(dist)
