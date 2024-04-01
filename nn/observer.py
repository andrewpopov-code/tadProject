import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Callable, Union

from topology import TopologyBase, TopologyModule


class _Hook:
    def __init__(self, f: tuple[Callable], tm: TopologyModule, kwargs: dict):
        self.f = f
        self.tm = tm
        self.kwargs = kwargs

    def __call__(self, s, args, result):
        if self.f:
            self.tm(self.f[0](result), **self.kwargs)
        else:
            self.tm(result, **self.kwargs)
        return result


class _PreHook:
    def __init__(self, f: tuple[Callable], tm: TopologyModule, kwargs: dict):
        self.f = f
        self.tm = tm
        self.kwargs = kwargs

    def __call__(self, s, args, kwargs):
        if self.f:
            self.tm(self.f[0](*args, **kwargs), **self.kwargs)
        else:
            self.tm(args[0], **self.kwargs)


class TopologyObserver(TopologyBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(
            self, net: nn.Module, *, topology_modules: list[TopologyModule] = (),
            writer: SummaryWriter = None, reset: bool = False,
            pre_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            post_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = ()
    ):
        super().__init__(
            tag=f'Topology Observer {id(net or self)}',
            writer=writer,
            layers=topology_modules
        )
        self.reset = reset
        self.net = net

        for m, tms in post_topology:
            for tm, kwargs, *f in tms:
                m.register_forward_hook(_Hook(f, tm, kwargs))
                self.register(tm)
        for m, tms in pre_topology:
            for tm, kwargs, *f in tms:
                m.register_forward_pre_hook(_PreHook(f, tm, kwargs), with_kwargs=True)
                self.register(tm)

        net.register_forward_hook(self.increment)
        for m in topology_modules:
            self.register(m)
        net.apply(self.register)

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and m is not self.net:
            self.topology_children[id(m)] = m
            m.add_or_skip_log_hook()
            if m.parent() is None or self.reset:
                m.set_parent(self)  # Set to be the observer

    def increment(self, m: nn.Module, args: tuple, result):
        self.step += 1
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result


class TopologyTrainingObserver(TopologyObserver):
    def __init__(
            self, net: nn.Module, *, topology_modules: list[TopologyModule] = (),
            writer: SummaryWriter = None, reset: bool = False,
            pre_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            post_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            log_every_train: int = 1, log_every_val: int = 1, topology_every: bool = False
    ):
        super().__init__(net, topology_modules=topology_modules, writer=writer, reset=reset, pre_topology=pre_topology, post_topology=post_topology)

        self.log_every_train = log_every_train
        self.topology_every = topology_every
        self.log_every_val = log_every_val
        self.val_step = 0

        for k in self.topology_children:
            if isinstance(self.topology_children[k], TopologyModule):
                self.topology_children[k].forward = self.get_forward(self.topology_children[k], self.topology_children[k].forward)

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule):
            m.register_forward_pre_hook(self.set_logging, with_kwargs=True)
        super().register(m)

    def increment(self, m: nn.Module, args: tuple, result):
        if m.training:
            self.step += 1
            self.val_step = 0
        else:
            self.val_step += 1
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result

    def set_logging(self, m: TopologyModule, args: tuple, kwargs: dict):
        if kwargs.get('logging', True) and m.parent() is self:  # May be set to False in the forward(...) call
            kwargs['logging'] = (self.step % self.log_every_train if m.training else self.val_step % self.log_every_val) == 0
        return args, kwargs

    def get_forward(self, m: TopologyModule, f: Callable):
        def forward(x: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = False, channel_first: bool = True, **kwargs):
            if self.topology_every or logging:
                return f(x, label=label, logging=logging, batches=batches, channel_first=channel_first, **kwargs)
            else:
                return None

        return forward

    def flush(self):
        self.val_step = 0
        super().flush()

    def get_tags(self):
        if self.net.training:
            tag = f' (Training Call {self.step})'
        else:
            tag = f' (Validation Call {self.step - 1} + {self.val_step})'
        return [self.tag + tag]
