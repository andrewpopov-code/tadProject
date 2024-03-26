import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dadapy import data

from utils import image_distance, euclidean_dist
from .topologybase import TopologyBase


class IntrinsicDimension(TopologyBase):
    def __init__(self, parent: [TopologyBase, None] = None):
        super().__init__(tag=f'ID Profile {id(self)}', parent=parent)

    def forward(self, x: torch.Tensor, *, label: str = '', distances: bool = True, logging: bool = True):
        x = x.numpy()
        if not distances:
            x = np.unique(x, axis=-2)
            if x.ndim == 3:  # image
                x = image_distance(x)
            else:  # T x D
                x = euclidean_dist(x, x)
        dim, err, _ = data.Data(distances=x).compute_id_2NN()
        return dim, err

    def get_tags(self):
        if self.parent is not None:
            return self.parent.get_tags() + [self.tag]
        return [self.tag]

    @staticmethod
    def log(self: 'IntrinsicDimension', args: tuple, kwargs: dict, result):
        if kwargs['logging']:
            kwargs['writer'].add_scalar(kwargs['tag'] + '/ID Estimate', result[0])
            kwargs['writer'].add_scalar(kwargs['tag'] + '/ID Estimate Error', result[1])

        return result
