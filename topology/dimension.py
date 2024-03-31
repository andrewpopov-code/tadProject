import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dadapy import data

from .base import TopologyBase
from .module import TopologyModule
from utils import compute_unique_distances


class IntrinsicDimension(TopologyModule):
    def __init__(self, tag: str = None, parent: TopologyBase = None, writer: SummaryWriter = None):
        super().__init__(tag=tag or f'ID Profile {id(self)}', parent=parent, writer=writer)

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = False, channel_first: bool = True, distances: bool = True):
        if channel_first:
            if x.ndim == 2 + batches:
                x = x.transpose(0 + batches, 1 + batches)
            else:
                x = x.transpose(0 + batches, 1 + batches).transpose(1 + batches, 2 + batches)
        if batches:
            dim, err = torch.zeros(x.shape[0]), torch.zeros(x.shape[0])
            for b in range(x.shape[0]):
                d = compute_unique_distances(x[b]) if not distances else x[b]
                dim[b], err[b], _ = data.Data(distances=d.detach().numpy()).compute_id_2NN()
        else:
            d = compute_unique_distances(x) if not distances else x
            dim, err, _ = data.Data(distances=d.detach().numpy()).compute_id_2NN()
        return dim, err

    def get_tags(self):
        if self.parent() is not None:
            return self.parent().get_tags() + [self.tag]
        return [self.tag]

    @staticmethod
    def log(self: 'IntrinsicDimension', args: tuple, kwargs: dict, result):
        dim, err = result

        if kwargs.get('batches', False):
            dim, err = dim[0], err[0]

        if kwargs.get('logging', True):
            kwargs['writer'].add_scalar('/'.join((kwargs['label'] + ' (ID Estimate)', kwargs['tag'])), dim, self.step)
            kwargs['writer'].add_scalar('/'.join((kwargs['label'] + ' (ID Estimate Error)', kwargs['tag'])), err, self.step)

        return result
