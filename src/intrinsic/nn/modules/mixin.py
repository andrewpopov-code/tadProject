import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial

from .base import IntrinsicBase
from .module import IntrinsicModule
from .homology import Persistence
from .dimension import Dimension
from .entropy import Entropy
from .hyperbolic import DeltaHyperbolicity


class IntrinsicMixin(IntrinsicBase):
    def __init__(self, tag: str = None, writer: SummaryWriter = None):
        super().__init__(tag=tag or f'Intrinsic Module: {id(self)}', writer=writer)
        self.Filtration = Persistence()
        self.Dimension = Dimension()
        self.Entropy = Entropy()
        self.DeltaHyperbolicity = DeltaHyperbolicity()

        assert isinstance(self, nn.Module), "Should be inherited by a nn.Module class"

        self.apply(self.register)
        self.register_forward_hook(self.increment)

    def register(self, m: nn.Module):
        if isinstance(m, IntrinsicModule) and m is not self:
            self.topology_children[id(m)] = m
            m.add_parent(self)

    @staticmethod
    def increment(self: 'IntrinsicMixin', args: tuple, result):
        self.step += 1
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result


class AttentionMixin(IntrinsicMixin):
    def __init__(self, tag: str = None, writer: SummaryWriter = None, display_heads: bool = True):
        super().__init__(tag=tag, writer=writer)
        self.display_heads = display_heads

    def register(self, m: nn.Module):
        if isinstance(m, nn.MultiheadAttention):  # TODO: figure out other attention mechanisms
            m.register_forward_pre_hook(partial(self.attn_pre_hook, self=self), with_kwargs=True)
            m.register_forward_hook(partial(self.attn_hook, self=self))
        else:
            super().register(m)

    @staticmethod
    def attn_pre_hook(self_attn: nn.MultiheadAttention, args: tuple, kwargs: dict, *, self: 'AttentionMixin'):
        return args, kwargs | dict(need_weights=True, average_attn_weights=not self.display_heads)

    @staticmethod
    def attn_hook(self_attn: nn.MultiheadAttention, args: tuple, result, *, self: 'AttentionMixin'):
        attn_output, attn_output_weights = result

        if self.display_heads:
            for h in range(self_attn.num_heads):
                head_weights = attn_output_weights[:, h]

                self.Entropy(head_weights, label=f'Entropy Analysis of Attention at head {h}')

                m = self.metric(head_weights)
                self.Filtration(m, label=f'Persistence Analysis of Attention at MultiheadAttn({id(self_attn)}) at head {h}')
                self.Dimension(m, label=f'Dimensional Analysis of Attention at head {h}')
                self.DeltaHyperbolicity(m, label=f'Hyperbolicity Analysis of Attention at head {h}')

            attn_output_weights = attn_output_weights.mean(dim=-3)

        self.Entropy(attn_output_weights, label='Entropy Analysis of Attention')

        m = self.metric(attn_output_weights)
        self.Filtration(m, label=f'Persistence Analysis of Attention at MultiheadAttn({id(self_attn)})')
        self.Dimension(m, label=f'Dimensional Analysis of Attention')
        self.DeltaHyperbolicity(m, label=f'Hyperbolicity Analysis of Attention')

        return result

    @staticmethod
    def metric(attn: torch.Tensor) -> torch.Tensor:
        ...
