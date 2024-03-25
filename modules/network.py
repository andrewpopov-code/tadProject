import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from functools import partial

from topology import TopologyBase


class TopologicalNetwork(TopologyBase):  # Inherit to enable
    def __init__(self,  writer: SummaryWriter = None):
        super().__init__(tag=f'Topological Network {id(self)}', logging=writer is not None)
        self.writer = writer
        self.post_init_handle = self.register_forward_pre_hook(self.post_init, with_kwargs=True)

    def post_init(self, *args, **kwargs):
        self.apply(self.register)
        self.post_init_handle.remove()

        return args, kwargs

    def register(self, m: nn.Module):
        if isinstance(m, TopologyBase):
            self.layers[id(m)] = m
            m.parent = self
            m.add_log_hook(self.writer)
