import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial


class TopologyBase(nn.Module):
    def __init__(self, tag: str = '', *layers: 'TopologyBase', parent: ['TopologyBase', None] = None):
        super().__init__()
        self.step = 0
        self.tag = tag
        self.parent = parent
        self.layers: dict[int, 'TopologyBase'] = {id(x): x for x in layers}

        self.register_forward_hook(self.increment)

    def add_log_hook(self):
        self.register_forward_hook(self.log, prepend=True)
        self.register_forward_hook(self.writer_hook, prepend=True)
        self.register_forward_hook(self.tag_hook, prepend=True, with_kwargs=True)

    @staticmethod
    def tag_hook(self: 'TopologyBase', args: tuple, kwargs: dict, result):
        kwargs['tag'] = '/'.join(self.get_tags()) + ': ' + kwargs.get('label')
        return result

    @staticmethod
    def writer_hook(self: 'TopologyBase', args: tuple, kwargs: dict, result):
        kwargs['writer'] = self.get_writer()
        return result

    def get_writer(self):
        if self.parent is not None:
            return getattr(self, 'writer') or self.parent.get_writer()
        return getattr(self, 'writer')

    @staticmethod
    def increment(self: 'TopologyBase', args: tuple, result):
        self.step += 1
        for k in self.layers:
            self.layers[k].flush()
        return result

    def get_tags(self):
        if self.parent is not None:
            return self.parent.get_tags() + [self.tag + f' (Call {self.step})']
        return [self.tag + f' (Call {self.step})']

    def flush(self):
        self.step = 0
        for k in self.layers:
            self.layers[k].flush()

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, **kwargs):
        raise NotImplementedError
