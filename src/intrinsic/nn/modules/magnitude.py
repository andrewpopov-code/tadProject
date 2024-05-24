import torch
from torch.utils.tensorboard import SummaryWriter

from .module import IntrinsicModule
from .base import IntrinsicBase
from src.intrinsic.utils.math import compute_unique_distances, magnitude


class Magnitude(IntrinsicModule):
    DISTANCES = False

    def __init__(self, tag: str = None, parents: list[IntrinsicBase] = (), writer: SummaryWriter = None):
        super().__init__(tag=tag or f'Magnitude Module {id(self)}', parents=parents, writer=writer)

    def _forward(self, x: torch.Tensor, *, label: str = IntrinsicModule.LABEL, logging: bool = IntrinsicModule.LOGGING, channel_first: bool = IntrinsicModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        if distances:
            return torch.tensor([
                magnitude(x[b].numpy(force=True)) for b in range(x.shape[0])
            ])
        return torch.tensor([
            magnitude(compute_unique_distances(x[b].numpy(force=True))) for b in range(x.shape[0])
        ])

    def get_tag(self):
        return self.tag

    def log(self, args: tuple, kwargs: dict, result: torch.Tensor, tag: str, writer: SummaryWriter):
        writer.add_scalar(
            '/'.join((kwargs['label'], tag)),
            result.mean(),
            self.step
        )

        return result
