import torch
import torch.nn as nn


class ParentWrapper:
    def __init__(self, parent: ['TopologyBase', None]):
        self.obj = parent


class TopologyBase(nn.Module):
    def __init__(self, tag: str = '', *layers: 'TopologyBase', parent: ['TopologyBase', None] = None):
        super().__init__()
        self.step = 0
        self.tag = tag
        self._parent = ParentWrapper(parent)
        self.layers: dict[int, 'TopologyBase'] = {id(x): x for x in layers}

        self.register_forward_hook(self.increment)

    def parent(self):
        return self._parent.obj

    def set_parent(self, parent: ['TopologyBase', None]):
        self._parent.obj = parent

    def add_log_hook(self):
        self.register_forward_hook(self.log, prepend=True, with_kwargs=True)
        self.register_forward_hook(self.writer_hook, prepend=True, with_kwargs=True)
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
        if self.parent() is not None:
            return getattr(self, 'writer', None) or self.parent().get_writer()
        return getattr(self, 'writer', None)

    @staticmethod
    def increment(self: 'TopologyBase', args: tuple, result):
        self.step += 1
        for k in self.layers:
            self.layers[k].flush()
        return result

    def get_tags(self):
        if self.parent() is not None:
            return self.parent().get_tags() + [self.tag + f' (Call {self.step})']
        return [self.tag + f' (Call {self.step})']

    def flush(self):
        self.step = 0
        for k in self.layers:
            self.layers[k].flush()

    @staticmethod
    def log(self: nn.Module, args: tuple, kwargs: dict, result):
        ...

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = False, **kwargs):
        ...
