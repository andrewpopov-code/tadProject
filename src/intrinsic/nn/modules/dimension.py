import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .base import IntrinsicBase
from .module import IntrinsicModule
from src.intrinsic.estimators.dimension import DimensionEstimator
from src.intrinsic.utils.math import image_to_cloud, unique_points


class Dimension(IntrinsicModule):
    DISTANCES = False

    def __init__(
            self, tag: str = None, parents: [IntrinsicBase] = (), writer: SummaryWriter = None,
            name: str = 'two_nn', aggregate=np.mean, slope_estimator_penalty: str = None, **kwargs
    ):
        super().__init__(tag=tag or f'ID Profile {id(self)}', parents=parents, writer=writer)
        self.estimator = DimensionEstimator(name=name, aggregate=aggregate, slope_estimator_penalty=slope_estimator_penalty, **kwargs)

    def _forward(self, x: torch.Tensor, *, label: str = IntrinsicModule.LABEL, logging: bool = IntrinsicModule.LOGGING, channel_first: bool = IntrinsicModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        dim = torch.zeros(x.shape[0])
        if x.ndim == 3:
            for b in range(x.shape[0]):
                dim[b] = self.estimator.fit_transform(unique_points(x[b].numpy(force=True)))
        else:
            for b in range(x.shape[0]):
                dim[b] = self.estimator.fit_transform(unique_points(image_to_cloud(x[b].numpy(force=True))))
        return dim

    def get_tag(self):
        return self.tag

    def log(self, args: tuple, kwargs: dict, dim: torch.Tensor, tag: str, writer: SummaryWriter):
        writer.add_scalar(
            '/'.join((kwargs['label'] + ' (ID Estimate)', tag)),
            dim.mean(),
            self.step
        )

        return dim
