import torch
import torch.nn.functional as F
import numpy as np
from .vr import VietorisRips
from functional.homology import drop_inf


class BettiLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, Y: torch.Tensor):
        dgms_tensorXb, dgms_tensorXd = VietorisRips(F.normalize(X))
        dgms_tensorYb, dgms_tensorYd = VietorisRips(F.normalize(Y))

        bY = np.array([np.array([x.shape[0] for x in drop_inf(list(dgms_tensorYb[b].unsqueeze(2).numpy(force=True)))]) for b in range(dgms_tensorYb.shape[0])])
        dgms_tensorXb, dgms_tensorXd = torch.nan_to_num(dgms_tensorXb), torch.nan_to_num(dgms_tensorXd)

        ctx.save_for_backward(dgms_tensorXb, dgms_tensorXd, dgms_tensorYb, dgms_tensorYd, bY)
        l1 = torch.tensor(
            [
                torch.tensor([
                    torch.sum(1 - torch.square(dgms_tensorXd[b, dim, :bY[b, dim]] - dgms_tensorXb[b, dim, :bY[b, dim]])) for dim in range(dgms_tensorXb.shape[1])
                ]).sum() for b in range(X.shape[0])
            ]
        )
        l2 = torch.tensor(
            [
                torch.tensor([
                    torch.sum(torch.square(dgms_tensorXd[b, dim, bY[b, dim]:] - dgms_tensorXb[b, dim, bY[b, dim]:]))
                    for dim in range(dgms_tensorXb.shape[1])
                ]).sum() for b in range(X.shape[0])
            ]
        )

        return l1 + l2

    @staticmethod
    def backward(ctx, grad_output):
        dX, dY, bY = ctx.saved_tensors
        return torch.stack([
            torch.stack(
                [2 - 2 * (dX[i][:bY[i], 1] - dX[i][:bY[i], 0]) for i in range(len(dX))]
            ),
            torch.stack(
                [2 * (dX[i][bY[i]:, 1] - dX[i][bY[i]:, 0]) for i in range(len(dX))]
            )
        ]) * grad_output
