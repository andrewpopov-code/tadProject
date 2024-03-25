import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial

from .module import TopologyModule
from topology import Persistence, IntrinsicDimension, Entropy


class TopologyMixin(TopologyModule):
    def __init__(self, writer: SummaryWriter = None):
        super().__init__(writer)
        self.Filtration = Persistence(parent=self)
        self.Dimension = IntrinsicDimension(parent=self)
        self.Entropy = Entropy(parent=self)


class AttentionMixin(TopologyMixin):
    def __init__(self, writer: SummaryWriter = None, display_heads: bool = True):
        super().__init__(writer)
        self.display_heads = display_heads

    def register(self, m: nn.Module, writer: SummaryWriter = None):
        if isinstance(m, nn.MultiheadAttention):
            m.register_forward_pre_hook(self.attn_pre_hook, with_kwargs=True)
            m.register_forward_hook(partial(self.attn_hook, module=id(m)))
        else:
            super().register(m)

    def attn_pre_hook(self, *args, **kwargs):
        return args, kwargs | dict(need_weights=True, average_attn_weights=not self.display_heads)

    def attn_hook(self, attn_output, attn_output_weights, module: int):
        self.Entropy(attn_output_weights, label='Entropy Analysis of Attention')

        m = self.metric(attn_output_weights)
        self.Filtration(m, label=f'Persistence Analysis of Attention at MultiheadAttn({module})')
        self.Dimension(m, label=f'Dimensional Analysis of Attention')

        return attn_output, attn_output_weights

    @staticmethod
    def metric(attn: torch.Tensor) -> torch.Tensor:
        ...
