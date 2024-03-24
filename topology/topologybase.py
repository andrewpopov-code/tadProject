import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TopologyBase(nn.Module):
    def __init__(self, tag: str = '', parent: ['TopologyBase', None] = None, logging: bool = False, *layers: 'TopologyBase'):
        super().__init__()
        self.step = 0
        self.tag = tag
        self.parent = parent
        self.layers: dict[int, 'TopologyBase'] = {id(x): x for x in layers}
        self.logging = logging

    def log(self, tag: str, *args, writer: SummaryWriter):
        ...

    def tag_hook(self, *args, tag: str, **kwargs):
        return args, kwargs | dict(tag='/'.join(self.get_tags() + [f'(Call {self.step}) ' + tag]))

    def increment(self, *args):
        self.step += 1
        for k in self.layers:
            self.layers[k].flush()

        return args

    def get_tags(self):
        if self.parent is not None:
            return self.parent.get_tags() + [self.tag]
        return [self.tag]

    def flush(self):
        self.step = 0
        for k in self.layers:
            self.layers[k].flush()

    def forward(self, x: torch.Tensor, *args, tag: str = '', **kwargs):
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
