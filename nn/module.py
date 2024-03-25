import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from topology import TopologyBase


class TopologyModule(TopologyBase):  # Inherit to enable
    def __init__(self,  writer: SummaryWriter = None):
        super().__init__(tag=f'Topology Module {id(self)}', logging=writer is not None)
        self.writer = writer
        self.post_init_handle = self.register_forward_pre_hook(self.post_init, with_kwargs=True)

    def post_init(self, *args, **kwargs):
        self.apply(self.register)
        self.post_init_handle.remove()

        return args, kwargs

    def register(self, m: nn.Module):
        if isinstance(m, TopologyBase):
            self.layers[id(m)] = m
            if m.parent is None:
                m.add_log_hook(getattr(m, 'writer') or self.writer)  # Added once
            m.parent = self  # Set by the immediate parent: last forward call is performed closest to the module
