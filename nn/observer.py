import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from topology import TopologyBase, TopologyModule


class TopologyObserver(TopologyBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(self, net: nn.Module, topology_modules: list[TopologyModule] = (), writer: SummaryWriter = None, reset: bool = True):
        super().__init__(
            tag=f'Topology Observer {id(net or self)}',
            writer=writer,
            layers=topology_modules
        )
        self.reset = reset
        self.net = net

        net.register_forward_hook(self.increment)
        for m in topology_modules:
            self.register(m)
        net.apply(self.register)

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule):
            self.topology_modules[id(m)] = m
            m.add_log_hook()
            if self.reset:
                m.set_parent(self)  # Set to be the observer

    def increment(self, m: nn.Module, args: tuple, result):
        self.step += 1
        for k in self.topology_modules:
            self.topology_modules[k].flush()
        return result


class TopologyTrainingObserver(TopologyObserver):
    def __init__(self, net: nn.Module, topology_modules: list[TopologyModule] = (), writer: SummaryWriter = None, reset: bool = True, log_every_train: int = 1, log_every_val: int = 1):
        super().__init__(net, topology_modules=topology_modules, writer=writer, reset=reset)
        self.log_every_train = log_every_train
        self.log_every_val = log_every_val
        self.val_step = 0
        self.training = False

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and id(m) not in self.topology_modules:
            m.register_forward_pre_hook(self.set_logging, with_kwargs=True)
        super().register(m)

    def increment(self, m: nn.Module, args: tuple, result):
        if m.training:
            self.step += 1
            self.val_step = 0
        else:
            self.val_step += 1
        for k in self.topology_modules:
            self.topology_modules[k].flush()
        return result

    def set_logging(self, m: TopologyModule, args: tuple, kwargs: dict):
        kwargs['logging'] = (self.step % self.log_every_train if m.training else self.val_step % self.log_every_val) == 0
        return args, kwargs

    def flush(self):
        self.val_step = 0
        super().flush()

    def get_tags(self):
        if self.net.training:
            tag = f' (Training Call {self.step})'
        else:
            tag = f' (Validation Call {self.step} + {self.val_step}'
        return [self.tag + tag]
