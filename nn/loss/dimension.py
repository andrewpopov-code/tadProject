import torch
import torch.nn as nn


class MLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distances: torch.Tensor):
        _, n, k = distances.shape[-1]
        dim_ptw = (k - 1) / torch.log(distances[:, -1].unsqueeze(1) / distances).sum(-1)
        dim = 1 / torch.mean(1 / dim_ptw, dim=-1)
        ctx.save_for_backward(distances, n, k, dim_ptw, dim)
        return dim

    @staticmethod
    def backward(ctx, grad_output):
        distances, n, k, dim_ptw, dim = ctx.saved_tensors
        left = -1 / (distances[:, :, :-1] * n * (k - 1) * dim_ptw * dim_ptw * dim * dim)
        right = 1 / (distances[:, :, -1] * n * dim_ptw * dim_ptw * dim * dim).unsqueeze(-1)
        return torch.hstack((left, right)) * grad_output


class MOM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distances: torch.Tensor):
        _, n, k = distances.shape[-1]
        ctx.save_for_backward(distances, n, k)
        Tk = distances[:, :, -1]
        T = distances.mean(dim=-1)
        return (T / (Tk - T)).mean(dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        distances, n, k = ctx.saved_tensors
        left = distances[:, :, -1] / (n * k * (distances[:, :, -1] - distances.mean(dim=-1)) * (distances[:, :, -1] - distances.mean(dim=-1)))
        right = (distances[:, :, -1] - distances.sum(dim=-1)) / (n * k * (distances[:, :, -1] - distances.mean(dim=-1)) * (distances[:, :, -1] - distances.mean(dim=-1)))
        return torch.hstack((left, right.unsqueeze(-1))) * grad_output


class Dimension(nn.Module):
    def __init__(self, k: int, method: str = 'mle'):
        super().__init__()
        self.k = k
        self.est = MLE.apply if method == 'mle' else MOM.apply

    def forward(self, X: torch.Tensor):
        dist = torch.cdist(X, X).topk(k=self.k, largest=False, sorted=True).values[:, :, 1:]
        return self.est(dist)
