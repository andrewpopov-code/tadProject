import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from topology import TopologyBase


class TopologyModule(TopologyBase):  # Inherit to enable
    def __init__(self,  writer: SummaryWriter = None):
        super().__init__(tag=f'Topology Module {id(self)}')
        self.writer = writer
        self.post_init_handle = self.register_forward_pre_hook(self.post_init)

    @staticmethod
    def post_init(self: 'TopologyModule', args: tuple):
        self.apply(self.register)
        self.post_init_handle.remove()

        return args

    def register(self, m: nn.Module):
        if isinstance(m, TopologyBase) and m is not self:
            self.layers[id(m)] = m
            if m.parent() is None and hasattr(m, 'log'):
                m.add_log_hook()  # Added once
            m.set_parent(self)  # Set by the immediate parent: last forward call is performed closest to the module
