import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Callable, Union

from topology import TopologyBase, TopologyModule, Persistence, IntrinsicDimension, DeltaHyperbolicity


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
            writer: SummaryWriter = None, reset: bool = False, connect: bool = False, batches: bool = True,
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
        self.connect = connect
        self.net = net
        self.batches = batches

        # TODO: figure out if results should be saved into .pt files
        self.information: list[dict[tuple[int, str], list]] = []  # FIXME: [None] is appended for some reason

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
        # net.apply(self.register)  TODO: figure out if we need this

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and m is not self.net and id(m) not in self.topology_children:
            m.register_forward_hook(self.accumulate, with_kwargs=True)

            self.topology_children[id(m)] = m
            m.add_or_skip_log_hook()
            if m.parent() is None or (isinstance(m.parent(), TopologyObserver) and self.reset) or self.connect:
                m.set_parent(self)  # Set to be the observer

    def increment(self, m: nn.Module, args: tuple, result):
        self.step += 1
        self.information.append({})
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result

    def accumulate(self, m: TopologyModule, args: tuple, kwargs: dict, result):
        if result is None:
            return result

        if not self.information:
            self.information = [{}]

        if id(m) in self.information[-1]:
            self.information[-1][id(m), kwargs['label']].append(result)
        else:
            self.information[-1][id(m), kwargs['label']] = [result]
        return result


class TopologyTrainingObserver(TopologyObserver):
    def __init__(
            self, net: nn.Module, *, topology_modules: list[TopologyModule] = (),
            writer: SummaryWriter = None, reset: bool = False, connect: bool = False,
            pre_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            post_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            log_every_train: int = 1, log_every_val: int = 1, topology_every: bool = False
    ):
        super().__init__(net, topology_modules=topology_modules, writer=writer, reset=reset, connect=connect,
                         pre_topology=pre_topology, post_topology=post_topology)

        self.log_every_train = log_every_train
        self.topology_every = topology_every
        self.log_every_val = log_every_val
        self.val_step = 0
        self.train_epoch_information: list[dict[tuple[int, str], list]] = []  # information yielded at each step
        self.val_epoch_information: list[dict[tuple[int, str], list]] = []  # information yielded at each step

        for k in self.topology_children:
            if isinstance(self.topology_children[k], TopologyModule):
                self.topology_children[k].forward = self.get_forward(self.topology_children[k],
                                                                     self.topology_children[k].forward)
        self.net.register_forward_hook(self.accumulate_epoch)

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule):
            m.register_forward_pre_hook(self.set_logging, with_kwargs=True)
        super().register(m)

    def accumulate_epoch(self, m: nn.Module, args: tuple, result):
        if not self.information:
            return result

        if m.training:
            self.train_epoch_information.append(self.information[-1])
        else:
            self.val_epoch_information.append(self.information[-1])
        self.information = []
        return result

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
        if kwargs.get('logging', True) and m.maximal_parent() is self:  # May be set to False in the forward(...) call
            kwargs['logging'] = (self.step % self.log_every_train if m.training else self.val_step % self.log_every_val) == 0
        return args, kwargs

    def get_forward(self, m: TopologyModule, f: Callable):
        def forward(x: torch.Tensor, *, label: str = '', logging: bool = True, channel_first: bool = True, **kwargs):
            if self.topology_every or logging:
                return f(x, label=label, logging=logging, channel_first=channel_first, **kwargs)
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


class AttentionTopologyObserver(TopologyObserver):
    ...


class AttentionTopologyTrainingObserver(TopologyTrainingObserver):
    ...


class EmbeddingTopologyTrainingObserver(TopologyTrainingObserver):
    def __init__(self, net: nn.Module, *, embedding_modules: list[nn.Embedding] = (),
                 writer: SummaryWriter = None, reset: bool = False, connect: bool = False,
                 log_every_train: int = 1, log_every_val: int = 1, topology_every: bool = False):
        self.Filtration = Persistence()
        self.Dimension = IntrinsicDimension()
        self.DeltaHyperbolicity = DeltaHyperbolicity()

        TopologyTrainingObserver.__init__(
            self, net, topology_modules=[self.Filtration, self.Dimension, self.DeltaHyperbolicity],
            writer=writer, reset=reset, connect=connect,
            log_every_train=log_every_train, log_every_val=log_every_val, topology_every=topology_every
        )

        self.embedding_modules = embedding_modules
        net.register_forward_hook(self.embedding_topology)

    def embedding_topology(self, s: nn.Module, args: tuple, result):
        for em in self.embedding_modules:
            w = em.weight.unsqueeze(0)
            self.Filtration(w, label=f'Embedding Module {id(em)}')
            self.Dimension(w, label=f'Embedding Module {id(em)}')
            self.DeltaHyperbolicity(w, label=f'Embedding Module {id(em)}')

        return result
