import torch
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass

from ..base import IntrinsicBase
from ..module import IntrinsicModule
from ..functional.dimension import *
from utils.math import compute_unique_distances


@dataclass
class DimensionInformation:
    capacity: torch.Tensor
    # information: torch.Tensor
    corr: torch.Tensor
    pca: torch.Tensor
    local_pca: torch.Tensor
    two_nn: torch.Tensor
    mm: torch.Tensor
    ols: torch.Tensor
    mle: torch.Tensor


class Dimension(IntrinsicModule):
    DISTANCES = False

    def __init__(self, tag: str = None, parents: [IntrinsicBase] = (), writer: SummaryWriter = None):
        super().__init__(tag=tag or f'ID Profile {id(self)}', parents=parents, writer=writer)

        self.estimators = [
            capacity,
            # information,
            corr,
            pca,
            local_pca,
            two_nn,
            mm,
            ols,
            mle
        ]

    def _forward(self, x: torch.Tensor, *, label: str = IntrinsicModule.LABEL, logging: bool = IntrinsicModule.LOGGING, channel_first: bool = IntrinsicModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        dim = DimensionInformation(
            **{
                est.__name__: torch.zeros(x.shape[0]) for est in self.estimators
            }
        )

        for b in range(x.shape[0]):
            d = compute_unique_distances(x[b]) if not distances else x[b]
            for est in self.estimators:
                getattr(dim, est.__name__)[b] = est(d.detach().numpy())
        return dim

    def get_tag(self):
        return self.tag

    def log(self, args: tuple, kwargs: dict, result: DimensionInformation, tag: str, writer: SummaryWriter):
        writer.add_scalars(
            '/'.join((kwargs['label'] + ' (ID Estimate)', tag)),
            {
                est.__name__: getattr(result, est.__name__)[0] for est in self.estimators
            },
            self.step
        )

        return result
