import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .base import TopologyBase


class TopologyModule(nn.Module, TopologyBase):
    def __init__(self, tag: str = None, parent: TopologyBase = None, writer: SummaryWriter = None):
        nn.Module.__init__(self)
        TopologyBase.__init__(self, tag=tag or f'Topology Module {id(self)}', parent=parent, writer=writer)
        self.register_forward_hook(self.increment)
        self.apply(self.register)
        self.added_log_hook = False

    @staticmethod
    def post_init(self: 'TopologyModule', args: tuple):
        self.apply(self.register)
        self.post_init_handle.remove()

        return args

    def add_or_skip_log_hook(self):
        if not self.added_log_hook:
            self.register_forward_hook(self._log, with_kwargs=True)
            self.added_log_hook = True

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and m is not self:
            self.topology_children[id(m)] = m
            m.add_or_skip_log_hook()  # Added once
            m.set_parent(self)  # Set by the immediate parent: last forward call is performed closest to the module

    @staticmethod
    def _log(self: 'TopologyModule', args: tuple, kwargs: dict, result):
        if kwargs.get('logging', True):
            return self.log(args, kwargs, result, '/'.join(self.get_tags()), self.get_writer())
        return result

    def log(self, args: tuple, kwargs: dict, result, tag: str, writer: SummaryWriter):
        ...

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = True, channel_first: bool = True, **kwargs):
        ...

    @staticmethod
    def increment(self: 'TopologyModule', args: tuple, result):
        self.step += 1
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result
