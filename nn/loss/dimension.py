import torch
import torch.nn as nn


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


class Dimension(nn.Module):
    def __init__(self, k: int, method: str = 'mle'):
        super().__init__()
        self.k = k
        self.est = MLE.apply if method == 'mle' else MOM.apply

    def forward(self, X: torch.Tensor):
        dist = torch.cdist(X, X).topk(k=self.k + 1, largest=False, sorted=True).values[:, :, 1:]
        return self.est(dist)
