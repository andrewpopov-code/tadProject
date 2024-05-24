import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import normalize
from torch.utils.tensorboard import SummaryWriter

import math

from .base import IntrinsicBase
from .module import IntrinsicModule
from src.intrinsic.utils.math import image_to_cloud, compute_unique_distances
from src.intrinsic.functional.delta import delta_hyperbolicity, mobius_addition_torch, conformal_torch, exponential_map_torch


class DeltaHyperbolicity(IntrinsicModule):
    DISTANCES = False

    def __init__(self, tag: str = None, parents: list[IntrinsicBase] = (), writer: SummaryWriter = None):
        super().__init__(tag=tag or f'Delta Hyperbolicity {id(self)}', parents=parents, writer=writer)

    def _forward(self, x: torch.Tensor, *, label: str = IntrinsicModule.LABEL, logging: bool = IntrinsicModule.LOGGING, channel_first: bool = IntrinsicModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        delta = torch.zeros(x.shape[0])
        for b in range(x.shape[0]):
            d = x[b].numpy(force=True) if distances else compute_unique_distances(image_to_cloud(x[b].numpy(force=True)))
            delta[b] = delta_hyperbolicity(d)

        return delta

    def log(self, args: tuple, kwargs: dict, delta, tag: str, writer: SummaryWriter):
        writer.add_scalar('/'.join((kwargs['label'], tag)), delta.mean(), self.step)

        return delta

    def get_tag(self):
        return self.tag


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, c: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self._c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mx = self.linear(x)
        return torch.tanh(
            (mx.norm(dim=-1) / x.norm(dim=-1) * torch.atanh(torch.sqrt(self._c * x.norm(dim=-1))))
        ).unsqueeze(-1) * normalize(mx) / torch.sqrt(self._c)


class Concat(nn.Module):
    def __init__(self, in_features: int, out_features: int, c: float):
        super().__init__()
        self._c = c
        self.linear1 = Linear(in_features, out_features, c)
        self.linear2 = Linear(in_features, out_features, c)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mobius_addition_torch(self.linear1(x), self.linear2(y), self._c)


class HypSoftmax(nn.Module):
    def __init__(self, in_features: int, out_features: int, c: float):
        super().__init__()
        self.P = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=True)
        self.A = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-2)
        self._c = nn.Parameter(torch.tensor(c), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num = 2 * self._c * torch.sum(mobius_addition_torch(-self.P, x, self._c) * self.A.unsqueeze(-2), dim=-1)  # classes x batch
        den = (1 - self._c * torch.square(mobius_addition_torch(-self.P, x, self._c)).sum(dim=-1)) * torch.sum(self.A * self.A, dim=-1, keepdim=True)
        return self.softmax(
            (conformal_torch(self.P, self._c) / torch.norm(self.A, dim=-1) / torch.sqrt(self._c)).unsqueeze(-1) * torch.asinh(num / den)
        )


class ToPoincare(nn.Module):
    def __init__(self, c: float, d: int):
        super().__init__()
        self._c = c
        self.x = nn.Parameter(torch.randn(d), requires_grad=True)

    def forward(self, v: torch.Tensor):
        return exponential_map_torch(self.x, self._c, v)
