import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .base import TopologyBase


class TopologyModule(nn.Module, TopologyBase):
    LABEL = ''
    LOGGING = True
    CF = True

    def __init__(self, tag: str = None, parents: [TopologyBase] = (), writer: SummaryWriter = None):
        nn.Module.__init__(self)
        TopologyBase.__init__(self, tag=tag or f'Topology Module {id(self)}', parents=parents, writer=writer)
        self.register_forward_hook(self.increment)
        self.apply(self.register)
        self.added_log_hook = False

    def add_or_skip_log_hook(self):
        if not self.added_log_hook:
            self.register_forward_hook(self._log, with_kwargs=True, prepend=True)
            self.added_log_hook = True

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and m is not self:
            self.topology_children[id(m)] = m
            m.add_parent(self)

    @staticmethod
    def _log(self: 'TopologyModule', args: tuple, kwargs: dict, result):
        if not kwargs.get('logging', True):
            return result

        tags = self.get_tags()
        for ws, ts in tags:
            for w in ws:
                self.log(args, kwargs, result, '/'.join(ts), w)

        return result

    def log(self, args: tuple, kwargs: dict, result, tag: str, writer: SummaryWriter):
        ...

    def forward(self, x: torch.Tensor, *, label: str = LABEL, logging: bool = LOGGING, channel_first: bool = CF, **kwargs):
        ...

    @staticmethod
    def increment(self: 'TopologyModule', args: tuple, result):
        self.step += 1
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result
