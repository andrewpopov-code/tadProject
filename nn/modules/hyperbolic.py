import torch
from torch.utils.tensorboard import SummaryWriter

from .base import IntrinsicBase
from .module import IntrinsicModule
from utils.math import image_to_cloud, compute_unique_distances
from functional.delta import delta_hyperbolicity


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
            d = x[b].detach().numpy() if distances else compute_unique_distances(image_to_cloud(x[b].detach().numpy()))
            delta[b] = delta_hyperbolicity(d)

        return torch.tensor(delta)

    def log(self, args: tuple, kwargs: dict, delta, tag: str, writer: SummaryWriter):
        writer.add_scalar('/'.join((kwargs['label'], tag)), delta.mean(), self.step)

        return delta

    def get_tag(self):
        return self.tag
