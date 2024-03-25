import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial

from .network import TopologicalNetwork
from topology import Persistence, IntrinsicDimension


class TopologyMixin(TopologicalNetwork):
    def __init__(self, writer: SummaryWriter = None):
        super().__init__(writer)
        self.Filtration = Persistence(parent=self)
        self.Dimension = IntrinsicDimension(parent=self)


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

    def attn_hook(self, attn_output, attn_output_weights, *args, module: int):
        m = self.metric(attn_output_weights)
        self.Filtration(m, tag=f'Persistence Analysis of Attention at MultiheadAttn({module})')
        self.Dimension(m, tag=f'Dimensional Analysis of Attention at MultiheadAttn({module})')

        return attn_output, attn_output_weights, args

    @staticmethod
    def metric(attn: torch.Tensor) -> torch.Tensor:
        ...
