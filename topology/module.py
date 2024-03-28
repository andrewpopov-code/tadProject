import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .base import TopologyBase


class TopologyModule(nn.Module, TopologyBase):
    def __init__(self, tag: str = None, parent: TopologyBase = None, writer: SummaryWriter = None):
        nn.Module.__init__(self)
        TopologyBase.__init__(self, tag=tag or f'Topology Module {id(self)}', parent=parent, writer=writer)
        self.post_init_handle = self.register_forward_pre_hook(self.post_init)
        self.register_forward_hook(self.increment)

    @staticmethod
    def post_init(self: 'TopologyModule', args: tuple):
        self.apply(self.register)
        self.post_init_handle.remove()

        return args

    def add_log_hook(self):
        self.register_forward_hook(self.log, prepend=True, with_kwargs=True)
        self.register_forward_hook(self.writer_hook, prepend=True, with_kwargs=True)
        self.register_forward_hook(self.tag_hook, prepend=True, with_kwargs=True)

    @staticmethod
    def tag_hook(self: 'TopologyModule', args: tuple, kwargs: dict, result):
        kwargs['tag'] = '/'.join(self.get_tags()) + ': ' + kwargs.get('label')
        return result

    @staticmethod
    def writer_hook(self: 'TopologyModule', args: tuple, kwargs: dict, result):
        kwargs['writer'] = self.get_writer()
        return result

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and m is not self:
            self.topology_modules[id(m)] = m
            if m.parent() is None:
                m.add_log_hook()  # Added once
            m.set_parent(self)  # Set by the immediate parent: last forward call is performed closest to the module

    @staticmethod
    def log(self: 'TopologyModule', args: tuple, kwargs: dict, result):
        ...

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = False, **kwargs):
        ...

    @staticmethod
    def increment(self: 'TopologyModule', args: tuple, result):
        self.step += 1
        for k in self.topology_modules:
            self.topology_modules[k].flush()
        return result
