import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial

from .module import TopologyModule
from topology import Persistence, IntrinsicDimension, Entropy, DeltaHyperbolicity


class TopologyMixin(TopologyModule):
    def __init__(self, writer: SummaryWriter = None):
        super().__init__(writer)
        self.Filtration = Persistence()
        self.Dimension = IntrinsicDimension()
        self.Entropy = Entropy()
        self.DeltaHyperbolicity = DeltaHyperbolicity()


class AttentionMixin(TopologyMixin):
    def __init__(self, writer: SummaryWriter = None, display_heads: bool = True):
        super().__init__(writer)
        self.display_heads = display_heads

    def register(self, m: nn.Module):
        if isinstance(m, nn.MultiheadAttention):
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
        batches = attn_output.ndim == 3

        if self.display_heads:
            for h in range(self_attn.num_heads):
                head_weights = attn_output_weights[:, h] if batches else attn_output_weights[h]

                self.Entropy(head_weights, label=f'Entropy Analysis of Attention at head {h}', batches=batches)

                m = self.metric(head_weights)
                self.Filtration(m, label=f'Persistence Analysis of Attention at MultiheadAttn({id(self_attn)}) at head {h}', batches=batches)
                self.Dimension(m, label=f'Dimensional Analysis of Attention at head {h}', batches=batches)
                self.DeltaHyperbolicity(m, label=f'Hyperbolicity Analysis of Attention at head {h}', batches=batches)

            attn_output_weights = attn_output_weights.mean(dim=-3)

        self.Entropy(attn_output_weights, label='Entropy Analysis of Attention', batches=batches)

        m = self.metric(attn_output_weights)
        self.Filtration(m, label=f'Persistence Analysis of Attention at MultiheadAttn({id(self_attn)})', batches=batches)
        self.Dimension(m, label=f'Dimensional Analysis of Attention', batches=batches)
        self.DeltaHyperbolicity(m, label=f'Hyperbolicity Analysis of Attention', batches=batches)

        return result

    @staticmethod
    def metric(attn: torch.Tensor) -> torch.Tensor:
        ...
