import torch
import torch.nn as nn
from utils.math import neighbors


class MLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distances: torch.Tensor):
        _, n, k = distances.shape
        dim_ptw = (k - 1) / torch.log(distances[:, :, -1].unsqueeze(2) / distances).sum(dim=-1)
        dim = 1 / torch.mean(1 / dim_ptw, dim=-1)
        ctx.save_for_backward(distances, torch.tensor(distances.shape), dim_ptw, dim)
        return dim

    @staticmethod
    def backward(ctx, grad_output):
        distances, shape, dim_ptw, dim = ctx.saved_tensors
        _, n, k = shape
        left = -1 / (n * (k - 1)* distances[:, :, :-1] * (dim_ptw * dim_ptw * dim * dim).unsqueeze(dim=2))
        right = 1 / (n * distances[:, :, -1] * dim_ptw * dim_ptw * dim * dim).unsqueeze(dim=2)
        return torch.cat((left, right), dim=2) * grad_output


class MOM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distances: torch.Tensor):
        ctx.save_for_backward(distances, torch.tensor(distances.shape))
        Tk = distances[:, :, -1]
        T = distances.mean(dim=-1)
        return (T / (Tk - T)).mean(dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        distances, shape = ctx.saved_tensors
        _, n, k = shape
        left = distances[:, :, :-1] / (n * k * (distances[:, :, :-1] - distances.mean(dim=-1).unsqueeze(dim=2)) * (distances[:, :, :-1] - distances.mean(dim=-1).unsqueeze(dim=2)))
        right = (distances[:, :, -1] - distances.sum(dim=-1)) / (n * k * (distances[:, :, -1] - distances.mean(dim=-1)) * (distances[:, :, -1] - distances.mean(dim=-1)))
        left.requires_grad = right.requires_grad = False
        return torch.cat((left, right.unsqueeze(-1)), dim=2) * grad_output


class TwoNN(nn.Module):
    def forward(self, distances: torch.Tensor):
        n = distances.shape[1]
        x = torch.log(torch.sort(distances[:, :, 1] / distances[:, :, 0]).values)
        y = -torch.log(1 - torch.linspace(0, 1 - 1 / n, n))
        return torch.sum(x*y, dim=-1) / torch.square(x).sum(dim=-1)


class Dimension(nn.Module):
    def __init__(self, k: int = None, method: str = 'two_nn'):
        super().__init__()
        self.k = k or 2
        self.est = TwoNN() if method == 'two_nn' else MLE.apply if method == 'mle' else MOM.apply

    def forward(self, X: torch.Tensor):
        dist, _ = neighbors(X, self.k)
        return self.est(dist)
