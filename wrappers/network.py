import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial

from topology import TopologyBase


class TopologicalNetwork(TopologyBase):  # Inherit to enable
    def __init__(self, net: nn.Module, writer: SummaryWriter = None):
        super().__init__(tag=f'Topological Network {id(self)}', logging=writer is not None)
        self.writer = writer
        self.post_init_handle = net.register_forward_pre_hook(self.post_init, with_kwargs=True)

    def post_init(self, *args, **kwargs):
        self.register_forward_pre_hook(self.tag_hook, with_kwargs=True)
        self.register_forward_hook(self.increment)

        self.apply(self.register)
        self.post_init_handle.remove()

        return args, kwargs

    def register(self, m: nn.Module):
        if isinstance(m, TopologyBase):
            self.layers[id(m)] = m
            m.parent = self

            m.register_forward_pre_hook(m.tag_hook, with_kwargs=True)
            m.register_forward_hook(partial(m.log, writer=self.writer))
            m.register_forward_hook(m.outer_hook)
            m.register_forward_hook(m.increment)


class TopologyObserver(TopologyBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(self, net: nn.Module = None, topology_modules: list[TopologyBase] = None, writer: SummaryWriter = None):
        super().__init__()
        self.writer = writer
        self.layers: list[TopologyBase] = topology_modules or []
        self.net = net

        if self.net is not None:
            self.net.register_forward_pre_hook(self.tag_hook, with_kwargs=True)
            self.net.register_forward_hook(self.increment)

            self.net.apply(self.register)

    def tag_hook(self, *args, tag: str = None, **kwargs):
        self.tag = f'Step {self.step}'
        return args, dict(tag=self.tag) | kwargs


class TopologyWrapper(TopologyBase):
    # Wraps around a module, adding topology hooks to every child
    def __init__(self, net: nn.Module, writer: SummaryWriter = None):
        super().__init__()
        self.writer = writer

        net.apply(self.register)

    def register(self, m: nn.Module):
        if self.writer is not None:
            m.register_forward_pre_hook(m.tag_hook, with_kwargs=True)
            m.register_forward_hook(partial(m.log, writer=self.writer))

        m.register_forward_hook(m.outer_hook)
        m.register_forward_hook(m.increment)
