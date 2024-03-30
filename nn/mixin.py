import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial

from topology import TopologyBase, TopologyModule, Persistence, IntrinsicDimension, Entropy, DeltaHyperbolicity


class TopologyMixin(TopologyBase):
    def __init__(self, tag: str = None, writer: SummaryWriter = None):
        super().__init__(tag=tag or f'Topology Module: {id(self)}', parent=None, writer=writer)
        self.Filtration = Persistence()
        self.Dimension = IntrinsicDimension()
        self.Entropy = Entropy()
        self.DeltaHyperbolicity = DeltaHyperbolicity()

        self.apply(self.register)
        self.register_forward_hook(self.increment)

    @staticmethod
    def post_init(self: nn.Module, args: tuple):
        self.apply(self.register)
        self.post_init_handle.remove()

        return args

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and m is not self:
            self.topology_modules[id(m)] = m
            m.add_log_hook()  # Added once
            m.set_parent(self)  # Set by the immediate parent: last forward call is performed closest to the module

    @staticmethod
    def increment(self: 'TopologyMixin', args: tuple, result):
        self.step += 1
        for k in self.topology_modules:
            self.topology_modules[k].flush()
        return result


class AttentionMixin(TopologyMixin):
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
