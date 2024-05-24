import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .base import IntrinsicBase


class IntrinsicModule(nn.Module, IntrinsicBase):
    LABEL = ''
    LOGGING = True
    CF = True

    def __init__(self, tag: str = None, parents: [IntrinsicBase] = (), writer: SummaryWriter = None):
        nn.Module.__init__(self)
        IntrinsicBase.__init__(self, tag=tag or f'Intrinsic Module {id(self)}', parents=parents, writer=writer)
        self.apply(self.register)

    def register(self, m: nn.Module):
        if isinstance(m, IntrinsicModule) and m is not self:
            self.topology_children[id(m)] = m
            m.add_parent(self)

    def forward(self, *args, **kwargs):
        tags = list(filter(lambda x: x[2], self.get_tags()))
        if not tags:
            return
        result = self._forward(*args, **kwargs)
        if kwargs.get('logging', self.LOGGING):
            for ws, ts, _ in tags:
                for w in ws:
                    self.log(args, kwargs, result, '/'.join(ts), w)
        self.step += 1
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result

    def log(self, args: tuple, kwargs: dict, result, tag: str, writer: SummaryWriter):
        ...

    def log_epoch(self, epoch: int, label: str, results: list, writer: SummaryWriter):
        ...

    def _forward(self, x: torch.Tensor, *, label: str = LABEL, logging: bool = LOGGING, channel_first: bool = CF, **kwargs):
        ...
