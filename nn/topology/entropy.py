import torch
from torch.utils.tensorboard import SummaryWriter

from ..base import TopologyBase
from ..module import TopologyModule


class Entropy(TopologyModule):
    def __init__(self, tag: str = None, parents: [TopologyBase] = (), writer: SummaryWriter = None, base: str = 'nat'):
        super().__init__(tag=tag or f'Entropy Profile {id(self)}', parents=parents, writer=writer)
        self.logarithm = torch.log if base == 'nat' else torch.log2 if base == 'bits' else torch.log10

    def forward(self, prob: torch.Tensor, *, label: str = TopologyModule.LABEL, logging: bool = TopologyModule.LOGGING, channel_first: bool = TopologyModule.CF, vectors: bool = True):
        if not channel_first:  # TODO: fix vectors or one channel squeezed tensors
            if prob.ndim == 3:
                prob = prob.transpose(1, 2)
            else:
                prob = prob.transpose(1, 2).transpose(2, 3)
        return self.entropy(prob)

    def get_tag(self):
        return self.tag

    def log(self, args: tuple, kwargs: dict, entropy: torch.Tensor, tag: str, writer: SummaryWriter):
        if kwargs.get('vectors', True):
            return self.log_cloud(kwargs, entropy, tag, writer)
        else:
            return self.log_heads(kwargs, entropy, tag, writer)

    def log_cloud(self, kwargs: dict, entropy: torch.Tensor, tag: str, writer: SummaryWriter):
        # TODO: figure out if this is correct
        _entropy = entropy[0]

        if _entropy.ndim == 2:  # Multiple 'heads'
            _entropy = _entropy.mean(dim=-1)
            writer.add_scalars(
                '/'.join((kwargs['label'], tag)), {
                    f'Head {h}': _entropy[h] for h in range(entropy.shape[0])
                }, self.step
            )

        _entropy = _entropy.mean(dim=-1)
        writer.add_scalars(
            '/'.join((kwargs['label'], tag)), {f'Average Entropy': _entropy[0]}, self.step
        )

        return entropy

    def log_heads(self, kwargs: dict, entropy: torch.Tensor, tag: str, writer: SummaryWriter):
        _entropy = entropy[0]

        if _entropy.ndim == 3:  # Multiple 'heads'
            _entropy = _entropy.mean(dim=-1)
            writer.add_scalars(
                '/'.join((kwargs['label'], tag)), {
                    f'Head {h}': _entropy[h] for h in range(entropy.shape[0])
                }, self.step
            )
        _entropy = _entropy.mean(dim=-1)
        writer.add_scalars(
            '/'.join((kwargs['label'], tag)), {f'Average Entropy': _entropy[0]}, self.step
        )

        return entropy

    def entropy(self, prob: torch.Tensor):
        return (-prob * self.logarithm(prob)).sum(dim=-1)
