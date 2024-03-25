import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .topologybase import TopologyBase


class Entropy(TopologyBase):
    def __init__(self, parent: [TopologyBase, None] = None, base: str = 'nat'):
        super().__init__(tag='Entropy', parent=parent)
        self.logarithm = torch.log if base == 'nat' else torch.log2 if base == 'bits' else torch.log10

    def forward(self, prob: torch.Tensor, *args, label: str = '', **kwargs):
        return label, self.entropy(prob)

    def get_tags(self):
        if self.parent is not None:
            return self.parent.get_tags() + [self.tag]
        return [self.tag]

    def log(self, tag: str, entropy: torch.Tensor = None, *args, writer: SummaryWriter):
        if self.logging and entropy is not None:
            writer.add_scalars(
                tag, {
                    f'Element {i}': entropy[0, i] for i in range(entropy.shape[1])
                }
            )

        return tag, entropy

    def entropy(self, prob: torch.Tensor):
        return (-prob * self.logarithm(prob)).sum(dim=0)
