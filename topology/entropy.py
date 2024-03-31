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
    def log(self: 'Entropy', args: tuple, kwargs: dict, entropy: torch.Tensor):
        _entropy = entropy[0] if kwargs.get('batches', False) else entropy

        if kwargs.get('logging', True):
            if _entropy.ndim == 3:  # Multiple 'heads'
                _entropy = _entropy.mean(dim=-1)
                kwargs['writer'].add_scalars(
                    '/'.join((kwargs['label'], kwargs['tag'])), {
                        f'Head {h}': _entropy[h] for h in range(entropy.shape[0])
                    }, self.step
                )
            _entropy = _entropy.mean(dim=-1)
            kwargs['writer'].add_scalars(
                '/'.join((kwargs['label'], kwargs['tag'])), {f'Average Entropy': _entropy[0]}, self.step
            )

        return entropy

    def entropy(self, prob: torch.Tensor):
        return (-prob * self.logarithm(prob)).sum(dim=-1)
