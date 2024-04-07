from torch.utils.tensorboard import SummaryWriter


class ParentContainer:
    def __init__(self, parent: ['TopologyBase', None]):
        self.obj = parent


class TopologyBase:
    def __init__(self, tag: str = '', parents: list['TopologyBase'] = (), writer: SummaryWriter = None, topology_children: list['TopologyBase'] = ()):
        self.step = 0
        self.tag = tag
        self.writer = writer
        self._parents = [ParentContainer(p) for p in parents]
        self.topology_children: dict[int, 'TopologyBase'] = {id(x): x for x in topology_children}

    def parents(self) -> list['TopologyBase']:
        return [p.obj for p in self._parents]

    def parents_iter(self):
        for p in self._parents:
            yield p.obj

    def add_parent(self, parent: 'TopologyBase'):
        self._parents.append(ParentContainer(parent))

    def get_tags(self):
        # Idea: get writers and tags from parents and extend them with ours; returns variants x writers x tags
        tags = [
            [ws | {self.writer} if self.writer is not None else ws, ts + [self.get_tag()]] for p in self.parents_iter() for ws, ts in p.get_tags()
        ]
        return tags or [[{self.writer} if self.writer is not None else set(), [self.get_tag()]]]

    def get_tag(self):
        return self.tag + f' (Call {self.step})'

    def flush(self):
        self.step = 0
        for k in self.topology_children:
            self.topology_children[k].flush()
