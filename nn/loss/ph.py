import torch
import torch.nn as nn
from .vr import VietorisRips
from .functional import pq_loss, ph_dimension_loss, signature_loss


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
