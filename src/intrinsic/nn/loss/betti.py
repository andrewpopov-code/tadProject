import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .vr import VietorisRips
from src.intrinsic.functional.homology import drop_inf


class BettiLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dgms_tensorXb: torch.Tensor, dgms_tensorXd: torch.Tensor, dgms_tensorYb: torch.Tensor, dgms_tensorYd: torch.Tensor):
        bY = np.array([np.array([x.shape[0] for x in drop_inf(list(dgms_tensorYb[b].unsqueeze(2).numpy(force=True)))]) for b in range(dgms_tensorYb.shape[0])])
        dgms_tensorXb, dgms_tensorXd = torch.nan_to_num(dgms_tensorXb), torch.nan_to_num(dgms_tensorXd)

        ctx.save_for_backward(dgms_tensorXb, dgms_tensorXd, dgms_tensorYb, dgms_tensorYd, torch.tensor(bY))
        l1 = torch.stack(
            [
                torch.stack([
                    torch.sum(1 - torch.square(dgms_tensorXd[b, dim, :bY[b, dim]] - dgms_tensorXb[b, dim, :bY[b, dim]])) for dim in range(dgms_tensorXb.shape[1])
                ]).sum() for b in range(bY.shape[0])
            ]
        )
        l2 = torch.stack(
            [
                torch.stack([
                    torch.sum(torch.square(dgms_tensorXd[b, dim, bY[b, dim]:] - dgms_tensorXb[b, dim, bY[b, dim]:]))
                    for dim in range(dgms_tensorXb.shape[1])
                ]).sum() for b in range(bY.shape[0])
            ]
        )

        return (l1 + l2).sum()

    @staticmethod
    def backward(ctx, grad_output):
        dgms_tensorXb, dgms_tensorXd, dgms_tensorYb, dgms_tensorYd, bY = ctx.saved_tensors
        return (
            torch.stack([
                torch.stack([
                    torch.cat([
                        2 * (dgms_tensorXd[b, dim, :bY[b, dim]] - dgms_tensorXb[b, dim, :bY[b, dim]]),
                        -2 * (dgms_tensorXd[b, dim, bY[b, dim]:] - dgms_tensorXb[b, dim, bY[b, dim]:])
                    ]) for dim in range(bY.shape[1])
                ]) for b in range(bY.shape[0])
            ]) * grad_output,
            torch.stack([
                torch.stack([
                    torch.cat([
                        -2 * (dgms_tensorXd[b, dim, :bY[b, dim]] - dgms_tensorXb[b, dim, :bY[b, dim]]),
                        2 * (dgms_tensorXd[b, dim, bY[b, dim]:] - dgms_tensorXb[b, dim, bY[b, dim]:])
                    ]) for dim in range(bY.shape[1])
                ]) for b in range(bY.shape[0])
            ]) * grad_output,
            None, None
        )


class BettiLoss(nn.Module):
    def __init__(self, maxdim: int = 1):
        super().__init__()
        self.maxdim = maxdim
        self.filtration = VietorisRips.apply
        self.loss_fn = BettiLossFunction.apply

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        dgms_tensorXb, dgms_tensorXd, dgms_tensorYb, dgms_tensorYd = *self.filtration(F.normalize(X))[:2], *self.filtration(F.normalize(Y))[:2]
        return self.loss_fn(dgms_tensorXb, dgms_tensorXd, dgms_tensorYb, dgms_tensorYd)


class BettiCurveLoss(nn.Module):
    def __init__(self, maxdim: int = 1, n_bins: int = 100):
        super().__init__()
        self.maxdim = maxdim
        self.n_bins = n_bins
        self.filtration = VietorisRips.apply

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        bcX, bcY = self.curves(X), self.curves(Y)
        return torch.square(bcX - bcY).sum()

    def curves(self, X: torch.Tensor):
        dgms_tensorXb, dgms_tensorXd = self.filtration(X)[:2]
        dgms_tensorXb, dgms_tensorXd = torch.nan_to_num(dgms_tensorXb), torch.nan_to_num(dgms_tensorXd, posinf=0)
        global_min, global_max = dgms_tensorXb.min(), dgms_tensorXd.max()
        steps = torch.linspace(global_min, global_max, self.n_bins)
        return ((dgms_tensorXb.unsqueeze(2) <= steps.reshape(1, 1, -1, 1)) & (dgms_tensorXd.unsqueeze(2) > steps.reshape(1, 1, -1, 1))).sum(dim=-1)
