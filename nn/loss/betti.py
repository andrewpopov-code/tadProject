import torch
import numpy as np
from .vr import VietorisRips


class BettiLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, Y: torch.Tensor):
        dgms_tensorXb, dgms_tensorXd = VietorisRips(X)
        dgms_tensorYb, dgms_tensorYd = VietorisRips(Y)

        bY = [a.shape[0] for a in dY]
        ctx.save_for_backward(dX, dY, bY)
        l1 = np.sum([np.sum(1 - (dX[i][:bY[i], 1] - dX[i][:bY[i], 0]) * (dX[i][:bY[i], 1] - dX[i][:bY[i], 0])) for i in range(len(dX))])
        l2 = np.sum([np.sum((dX[i][bY[i]:, 1] - dX[i][bY[i]:, 0]) * (dX[i][bY[i]:, 1] - dX[i][bY[i]:, 0])) for i in range(len(dX))])
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
