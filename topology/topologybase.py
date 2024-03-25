import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial


class TopologyBase(nn.Module):
    def __init__(self, tag: str = '', *layers: 'TopologyBase', parent: ['TopologyBase', None] = None, logging: bool = False):
        super().__init__()
        self.step = 0
        self.tag = tag
        self.parent = parent
        self.layers: dict[int, 'TopologyBase'] = {id(x): x for x in layers}
        self.logging = logging

        self.register_forward_hook(self.outer_hook)
        self.register_forward_hook(self.increment)

    def add_log_hook(self, writer: SummaryWriter = None):
        self.register_forward_hook(partial(self.log, writer=writer), prepend=True)
        self.register_forward_hook(self.tag_hook, prepend=True)
        self.logging = writer is not None

    def log(self, tag: str, *args, writer: SummaryWriter):
        ...

    def tag_hook(self, label: str, *args):
        return '/'.join(self.get_tags()) + ': ' + label, args

    def increment(self, *args):
        self.step += 1
        for k in self.layers:
            self.layers[k].flush()

        return args

    def get_tags(self):
        if self.parent is not None:
            return self.parent.get_tags() + [self.tag + f' (Call {self.step})']
        return [self.tag + f' (Call {self.step})']

    def flush(self):
        self.step = 0
        for k in self.layers:
            self.layers[k].flush()

    def forward(self, x: torch.Tensor, *args, label: str = '', **kwargs):
        ...

    def suppress_logs(self):
        for k in self.layers:
            self.layers[k].suppress_logs()
        self.logging = False

    def enable_logs(self):
        # TODO: figure out the writer
        for k in self.layers:
            self.layers[k].enable_logs()
        self.logging = True

    def outer_hook(self, tag: str, *args):
        return args
