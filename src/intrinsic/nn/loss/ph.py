import torch
import torch.nn as nn
import torch.nn.functional as F

from src.intrinsic.utils.matching import matching_alg_torch
from .functional import pq_loss, ph_dimension_loss, signature_loss
from .vr import VietorisRips


class PHRegressionLoss(nn.Module):
    def __init__(self, maxdim: int = 1, lt: float = 0.5, ld: float = 0.5):
        super().__init__()
        self.maxdim = maxdim
        self.ld, self.lt = ld, lt

    def forward(self, Z: torch.Tensor, Y: torch.Tensor):
        return self.lt * signature_loss(Z, Y) + self.ld * ph_dimension_loss(Z)


class PQLoss(nn.Module):
    def __init__(self, p: int, q: int, maxdim: int = 1):
        super().__init__()
        self.maxdim = maxdim
        self.filtration = VietorisRips.apply
        self.left, self.right = p, q

    def forward(self, X: torch.Tensor):
        dgms_tensorXb, dgms_tensorXd = self.filtration(X)[:2]
        dgms_tensorXb, dgms_tensorXd = torch.nan_to_num(dgms_tensorXb), torch.nan_to_num(dgms_tensorXd)
        return pq_loss(dgms_tensorXb, dgms_tensorXd, self.left, self.right)


class WassersteinLoss(nn.Module):
    def __init__(self, q: float, maxdim: int = 1):
        super().__init__()
        self.q = q
        self.filtration = VietorisRips.apply
        self.maxdim = maxdim

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        dgms_tensorXb, dgms_tensorXd = self.filtration(X)[:2]
        diagX = torch.stack(
            [dgms_tensorXb.unsqueeze(-1), dgms_tensorXd.unsqueeze(-1)], dim=-1
        ).reshape(X.shape[0], -1, 2)
        dgms_tensorYb, dgms_tensorYd = self.filtration(Y)[:2]
        diagY = torch.stack(
            [dgms_tensorYb.unsqueeze(-1), dgms_tensorYd.unsqueeze(-1)], dim=-1
        ).reshape(X.shape[0], -1, 2)
        diagX, diagY = torch.nan_to_num(diagX, 0, 0), torch.nan_to_num(diagY, 0, 0)
        lX, lY = diagX.shape[1], diagY.shape[1]
        if lX < lY:
            diagX = F.pad(diagX.transpose(-1, -2), (0, lY - lX)).transpose(-1, 2)
        elif lY < lX:
            diagY = F.pad(diagY.transpose(-1, -2), (0, lX - lY)).transpose(-1, 2)

        dist = torch.pow(
            torch.cdist(diagX, diagY, p=torch.inf), 1 if self.q == torch.inf else self.q
        )
        p = matching_alg_torch(dist)
        p.requires_grad = False

        norm = torch.norm(
            torch.cdist(diagX, torch.gather(diagY, 1, p), p=torch.inf), 1 if self.q == torch.inf else self.q, dim=1
        )
        return norm.sum()
