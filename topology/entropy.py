import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .base import TopologyBase
from .module import TopologyModule


class Entropy(TopologyModule):
    def __init__(self, tag: str = None, parent: TopologyBase = None, writer: SummaryWriter = None, base: str = 'nat'):
        super().__init__(tag=tag or f'Entropy Profile {id(self)}', parent=parent, writer=writer)
        self.logarithm = torch.log if base == 'nat' else torch.log2 if base == 'bits' else torch.log10

    def forward(self, prob: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = False):
        return self.entropy(prob)

    def get_tags(self):
        if self.parent() is not None:
            return self.parent().get_tags() + [self.tag]
        return [self.tag]

    @staticmethod
    def log(self: 'Entropy', args: tuple, kwargs: dict, entropy):
        if kwargs.get('batches', False):
            entropy = entropy[0]

        if kwargs.get('logging', True):
            kwargs['writer'].add_scalars(
                kwargs['tag'], {
                    f'Element {i}': entropy[0, i] for i in range(entropy.shape[1])
                }, self.step
            )
        return entropy

    def entropy(self, prob: torch.Tensor):
        return (-prob * self.logarithm(prob)).sum(dim=-2)
